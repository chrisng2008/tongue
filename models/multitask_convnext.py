"""
多任务 ConvNeXt 模型
共享 Backbone + 双 Head（舌形多标签 + 腻腐苔多类）
"""
import torch
import torch.nn as nn
import torchvision.models as models


class MultiTaskConvNeXt(nn.Module):
    """
    基于 ConvNeXt Base 的多任务舌象分类模型

    结构:
        - Backbone: ConvNeXt Base (IMAGENET1K_V1)
        - Shape Head: 多标签分类 (spots, cracks, teethmarks)
        - Coat Head: 多类分类 (greasy, rotten, nospecialgreasy)
    """
    def __init__(self, pretrained=True, freeze_backbone=False):
        super().__init__()

        # 加载预训练 ConvNeXt Base
        weights = 'IMAGENET1K_V1' if pretrained else None
        self.backbone = models.convnext_base(weights=weights)

        # 获取特征维度
        self.feature_dim = 1024  # ConvNeXt Base 的特征维度

        # 提取特征层（去掉原分类头）
        self.features = self.backbone.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # 冻结 backbone（用于两阶段训练的Stage 1）
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
            nn.Linear(512, 3)  # 3个标签: spots, cracks, teethmarks
        )

        # === Coat Head: 多类分类 ===
        self.coat_head = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(self.feature_dim, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 3)  # 3个类别: greasy, rotten, nospecialgreasy
        )

    def forward(self, x):
        """
        前向传播

        参数:
            x: [B, 3, H, W] 输入图像

        返回:
            logits_shape: [B, 3] 舌形多标签logits
            logits_coat: [B, 3] 腻腐苔多类logits
        """
        # 提取特征 [B, 1024, H/32, W/32]
        features = self.features(x)

        # 全局平均池化 [B, 1024, 1, 1]
        pooled = self.avgpool(features)

        # 展平 [B, 1024]
        flattened = pooled.view(pooled.size(0), -1)

        # 双头输出
        logits_shape = self.shape_head(flattened)  # [B, 3]
        logits_coat = self.coat_head(flattened)    # [B, 3]

        return logits_shape, logits_coat

    def unfreeze_backbone(self):
        """解冻 backbone（用于Stage 2）"""
        for param in self.features.parameters():
            param.requires_grad = True
        print("Backbone已解冻")

    def get_parameters(self, stage=1):
        """
        获取不同阶段的参数组

        参数:
            stage: 1=只训heads, 2=全量训练

        返回:
            param_groups: list of dict
        """
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
        """返回可训练参数数量"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        return total, trainable, frozen


class MultiTaskLoss(nn.Module):
    """
    多任务损失函数

    组合:
        - Shape: BCEWithLogitsLoss (多标签)
        - Coat: CrossEntropyLoss (多类) 或 FocalLoss
    """
    def __init__(self, w_shape=1.0, w_coat=1.0, class_weights=None, use_focal=False, focal_gamma=2.0, focal_alpha=None):
        super().__init__()

        self.w_shape = w_shape
        self.w_coat = w_coat
        self.use_focal = use_focal

        # Shape损失（多标签BCE）
        self.shape_loss_fn = nn.BCEWithLogitsLoss(reduction='none')

        # Coat损失配置
        if use_focal:
            # 使用 Focal Loss
            self.focal_gamma = focal_gamma
            self.focal_alpha = focal_alpha  # [alpha_greasy, alpha_rotten, alpha_nospecial]
            if self.focal_alpha is not None:
                self.focal_alpha = torch.tensor(self.focal_alpha, dtype=torch.float32)
            print(f"使用 Focal Loss (gamma={focal_gamma}, alpha={self.focal_alpha})")
        else:
            # 使用加权 CE
            if class_weights is not None:
                class_weights = torch.tensor(class_weights, dtype=torch.float32)
            self.coat_loss_fn = nn.CrossEntropyLoss(
                weight=class_weights,
                reduction='none'
            )
            print(f"使用加权 CE (class_weights={class_weights})")

    def forward(self, logits_shape, logits_coat, y_shape, y_coat, sample_weights=None):
        """
        计算多任务损失

        参数:
            logits_shape: [B, 3]
            logits_coat: [B, 3]
            y_shape: [B, 3] float
            y_coat: [B] int (class indices)
            sample_weights: [B] float (可选，质量权重)

        返回:
            loss: scalar
            loss_dict: 各分量损失
        """
        batch_size = logits_shape.size(0)

        # Shape损失 [B, 3] -> 平均到每个样本 [B]
        loss_shape_per_label = self.shape_loss_fn(logits_shape, y_shape)
        loss_shape_per_sample = loss_shape_per_label.mean(dim=1)  # [B]

        # Coat损失 [B]
        if self.use_focal:
            loss_coat_per_sample = self._focal_loss(logits_coat, y_coat)
        else:
            loss_coat_per_sample = self.coat_loss_fn(logits_coat, y_coat)

        # 组合损失（每个样本）
        loss_per_sample = (
            self.w_shape * loss_shape_per_sample +
            self.w_coat * loss_coat_per_sample
        )

        # 应用样本质量权重
        if sample_weights is not None:
            loss_per_sample = loss_per_sample * sample_weights

        # Batch平均
        loss = loss_per_sample.mean()

        # 记录各分量
        loss_dict = {
            'total': loss.item(),
            'shape': loss_shape_per_sample.mean().item(),
            'coat': loss_coat_per_sample.mean().item(),
        }

        return loss, loss_dict

    def _focal_loss(self, logits, targets):
        """
        Focal Loss 实现

        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

        参数:
            logits: [B, 3]
            targets: [B] int (0, 1, 2)

        返回:
            loss: [B]
        """
        ce_loss = nn.functional.cross_entropy(logits, targets, reduction='none')

        # 获取概率
        probs = nn.functional.softmax(logits, dim=1)

        # 获取对应类别的概率 p_t
        pt = probs[range(len(targets)), targets]

        # Focal Loss
        focal_weight = (1 - pt) ** self.focal_gamma

        # 应用 alpha
        if self.focal_alpha is not None:
            if self.focal_alpha.device != logits.device:
                self.focal_alpha = self.focal_alpha.to(logits.device)
            alpha_t = self.focal_alpha[targets]
            focal_loss = alpha_t * focal_weight * ce_loss
        else:
            focal_loss = focal_weight * ce_loss

        return focal_loss


if __name__ == "__main__":
    # 测试模型
    print("测试多任务 ConvNeXt 模型...")

    # 创建模型
    model = MultiTaskConvNeXt(pretrained=False, freeze_backbone=False)

    # 打印参数量
    total, trainable, frozen = model.get_num_parameters()
    print(f"\n总参数: {total:,}")
    print(f"可训练参数: {trainable:,}")
    print(f"冻结参数: {frozen:,}")

    # 测试前向传播
    batch_size = 4
    x = torch.randn(batch_size, 3, 224, 224)
    y_shape = torch.randint(0, 2, (batch_size, 3)).float()
    y_coat = torch.randint(0, 3, (batch_size,))

    logits_shape, logits_coat = model(x)
    print(f"\n输入shape: {x.shape}")
    print(f"舌形logits shape: {logits_shape.shape}")
    print(f"腻腐苔logits shape: {logits_coat.shape}")

    # 测试损失
    loss_fn = MultiTaskLoss(w_shape=1.0, w_coat=1.0)
    loss, loss_dict = loss_fn(logits_shape, logits_coat, y_shape, y_coat)
    print(f"\n总损失: {loss.item():.4f}")
    print(f"  Shape损失: {loss_dict['shape']:.4f}")
    print(f"  Coat损失: {loss_dict['coat']:.4f}")

    # 测试冻结backbone
    print("\n测试冻结backbone...")
    model_frozen = MultiTaskConvNeXt(pretrained=False, freeze_backbone=True)
    total, trainable, frozen = model_frozen.get_num_parameters()
    print(f"冻结后可训练参数: {trainable:,}")
    print(f"冻结参数: {frozen:,}")

    print("\n模型测试通过！")
