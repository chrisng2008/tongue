"""
多任务模型集合 - 支持多种 Backbone
包括 ConvNeXt Base/Large, Swin Transformer, ViT 等
"""
import torch
import torch.nn as nn
import torchvision.models as models


class MultiTaskBackbone(nn.Module):
    """
    支持多种 Backbone 的多任务模型

    支持的 backbone:
        - convnext_base (89M params)
        - convnext_large (198M params)
        - swin_base (88M params)
        - swin_large (197M params)
        - vit_base (86M params)
    """
    def __init__(self, backbone_name='convnext_base', pretrained=True, freeze_backbone=False):
        super().__init__()

        self.backbone_name = backbone_name
        self.feature_dim = self._get_feature_dim(backbone_name)

        # 加载 backbone
        self.features = self._load_backbone(backbone_name, pretrained)

        # 全局平均池化
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # 冻结 backbone
        if freeze_backbone:
            for param in self.features.parameters():
                param.requires_grad = False

        # === Shape Head: 多标签分类 ===
        self.shape_head = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(self.feature_dim, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 3)  # spots, cracks, teethmarks
        )

        # === Coat Head: 多类分类 ===
        self.coat_head = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(self.feature_dim, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 3)  # greasy, rotten, nospecialgreasy
        )

        print(f"✓ 使用 Backbone: {backbone_name}")
        print(f"  特征维度: {self.feature_dim}")
        print(f"  预训练: {pretrained}")

    def _get_feature_dim(self, backbone_name):
        """获取不同 backbone 的特征维度"""
        dims = {
            'convnext_tiny': 768,
            'convnext_small': 768,
            'convnext_base': 1024,
            'convnext_large': 1536,
            'swin_tiny': 768,
            'swin_small': 768,
            'swin_base': 1024,
            'swin_large': 1536,
            'vit_base_patch16_224': 768,
            'vit_large_patch16_224': 1024,
        }
        return dims.get(backbone_name, 1024)

    def _load_backbone(self, backbone_name, pretrained):
        """加载指定的 backbone"""
        # ConvNeXt 系列
        if backbone_name.startswith('convnext'):
            weights = self._get_weights(backbone_name, pretrained)
            model = models.convnext_base(weights=weights) if 'base' in backbone_name else \
                     models.convnext_large(weights=weights)
            return model.features

        # Swin Transformer 系列
        elif backbone_name.startswith('swin'):
            weights = self._get_weights(backbone_name, pretrained)
            if 'base' in backbone_name:
                model = models.swin_b(weights=weights)
            elif 'large' in backbone_name:
                model = models.swin_l(weights=weights)
            else:
                model = models.swin_s(weights=weights)
            # Swin 返回整个模型，提取特征
            return model.features

        # ViT 系列
        elif backbone_name.startswith('vit'):
            weights = self._get_weights(backbone_name, pretrained)
            if 'base' in backbone_name:
                model = models.vit_b_16(weights=weights)
            elif 'large' in backbone_name:
                model = models.vit_l_16(weights=weights)

            # ViT 需要特殊处理
            return ViTFeaturesExtractor(model)

        else:
            raise ValueError(f"不支持的 backbone: {backbone_name}")

    def _get_weights(self, backbone_name, pretrained):
        """获取预训练权重"""
        if not pretrained:
            return None

        weights_map = {
            'convnext_base': models.ConvNeXt_Base_Weights.IMAGENET1K_V1,
            'convnext_large': models.ConvNeXt_Large_Weights.IMAGENET1K_V1,
            'swin_base': models.Swin_B_Weights.IMAGENET1K_V1,
            'swin_small': models.Swin_S_Weights.IMAGENET1K_V1,
            # 'swin_large': models.Swin_L_Weights.IMAGENET1K_V1,  # 可能不可用
            'vit_base_patch16_224': models.ViT_B_16_Weights.IMAGENET1K_V1,
        }

        return weights_map.get(backbone_name)

    def forward(self, x):
        """前向传播"""
        # 提取特征
        if 'vit' in self.backbone_name:
            # ViT 特殊处理
            features = self.features(x)
        elif 'swin' in self.backbone_name:
            # Swin Transformer: 返回 (batch, H, W, channels)
            features = self.features(x)
            # 转换为 (batch, channels, H, W) 格式
            features = features.permute(0, 3, 1, 2)  # (batch, 1024, 7, 7)
        else:
            # ConvNeXt: 返回 (batch, channels, H, W)
            features = self.features(x)

        # 获取实际的特征维度
        batch_size = features.size(0)
        if len(features.shape) == 4:  # (batch, channels, H, W)
            # 全局平均池化
            pooled = self.avgpool(features)  # (batch, channels, 1, 1)
            flattened = pooled.view(batch_size, -1)  # (batch, channels)
        elif len(features.shape) == 3:  # (batch, seq_len, channels)
            # 对序列特征进行平均
            flattened = features.mean(dim=1)  # (batch, channels)
        else:
            raise ValueError(f"不支持的 feature shape: {features.shape}")

        # 双头输出
        logits_shape = self.shape_head(flattened)
        logits_coat = self.coat_head(flattened)

        return logits_shape, logits_coat

    def unfreeze_backbone(self):
        """解冻 backbone"""
        for param in self.features.parameters():
            param.requires_grad = True
        print("✓ Backbone已解冻")

    def get_parameters(self, stage=1):
        """获取不同阶段的参数组"""
        if stage == 1:
            # Stage 1: 只训练heads
            return [
                {'params': self.shape_head.parameters(), 'lr': 1e-3},
                {'params': self.coat_head.parameters(), 'lr': 1e-3},
            ]
        else:
            # Stage 2: 全量训练，backbone用较小学习率
            return [
                {'params': self.features.parameters(), 'lr': 3e-5},
                {'params': self.shape_head.parameters(), 'lr': 1e-4},
                {'params': self.coat_head.parameters(), 'lr': 1e-4},
            ]

    def get_num_parameters(self):
        """返回参数数量"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        return total, trainable, frozen


class ViTFeaturesExtractor(nn.Module):
    """ViT 特征提取器（特殊处理）"""
    def __init__(self, vit_model):
        super().__init__()
        self.vit = vit_model

    def forward(self, x):
        # ViT 的卷积层
        x = self.vit.conv_proj(x)
        x = self.vit.encoder._process_input(x)

        n, c, h, w = x.shape
        x = x.view(n, c, h * w).permute(0, 2, 1)  # N, H*W, C

        # 添加位置编码
        x = self.vit.encoder.pos_embedding(x)
        x = self.vit.encoder.pos_dropout(x)

        # 重塑为特征图格式
        batch_size, seq_length, dim = x.shape
        h = w = int(seq_length ** 0.5)
        x = x.permute(0, 2, 1).reshape(batch_size, dim, h, w)

        return x


class EnsembleMultiTaskModel(nn.Module):
    """
    集成模型：融合多个模型的预测
    """
    def __init__(self, backbone_list, pretrained=True):
        super().__init__()

        self.models = nn.ModuleList([
            MultiTaskBackbone(backbone, pretrained=pretrained)
            for backbone in backbone_list
        ])

        self.num_models = len(backbone_list)
        print(f"✓ 创建集成模型: {backbone_list}")
        print(f"  模型数量: {self.num_models}")

    def forward(self, x):
        """前向传播：平均所有模型的预测"""
        all_logits_shape = []
        all_logits_coat = []

        for model in self.models:
            logits_shape, logits_coat = model(x)
            all_logits_shape.append(logits_shape)
            all_logits_coat.append(logits_coat)

        # 平均
        avg_logits_shape = torch.stack(all_logits_shape).mean(dim=0)
        avg_logits_coat = torch.stack(all_logits_coat).mean(dim=0)

        return avg_logits_shape, avg_logits_coat

    def get_num_parameters(self):
        """返回参数数量"""
        total = sum(p.numel() for model in self.models for p in model.parameters())
        trainable = sum(p.numel() for model in self.models
                       for p in model.parameters() if p.requires_grad)
        frozen = total - trainable
        return total, trainable, frozen


if __name__ == "__main__":
    print("="*60)
    print("测试多模型支持")
    print("="*60)

    # 测试 ConvNeXt Large
    print("\n1. 测试 ConvNeXt Large...")
    model_large = MultiTaskBackbone('convnext_large', pretrained=False)
    total, trainable, frozen = model_large.get_num_parameters()
    print(f"总参数: {total:,} ({total/1e6:.1f}M)")
    print(f"可训练: {trainable:,}")

    # 测试前向传播
    x = torch.randn(2, 3, 224, 224)
    logits_shape, logits_coat = model_large(x)
    print(f"✓ 前向传播成功: shape={logits_shape.shape}, coat={logits_coat.shape}")

    # 测试 Ensemble
    print("\n2. 测试集成模型...")
    ensemble = EnsembleMultiTaskModel(['convnext_base', 'swin_base'], pretrained=False)
    total, trainable, frozen = ensemble.get_num_parameters()
    print(f"总参数: {total:,} ({total/1e6:.1f}M)")
    print(f"可训练: {trainable:,}")

    logits_shape, logits_coat = ensemble(x)
    print(f"✓ 前向传播成功: shape={logits_shape.shape}, coat={logits_coat.shape}")

    print("\n" + "="*60)
    print("所有测试通过！")
    print("="*60)
