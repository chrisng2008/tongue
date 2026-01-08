import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pandas as pd
from PIL import Image, ImageFilter
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
import math
from torch.cuda.amp import GradScaler, autocast
import random
import warnings

warnings.filterwarnings('ignore')

# 设置随机种子保证可重复性
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# ==================== 高级数据增强 ====================
class AdvancedAugmentation:
    """增强的数据预处理和增强类"""
    def __init__(self):
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
    def get_train_transform(self, size=224):
        return transforms.Compose([
            transforms.Resize((int(size*1.15), int(size*1.15))),
            transforms.RandomCrop(size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            transforms.ToTensor(),
            self.normalize,
        ])
    
    def get_val_transform(self, size=224):
        return transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            self.normalize,
        ])

# ==================== 舌象数据集 ====================
class TongueDataset(Dataset):
    """舌象图像数据集"""
    def __init__(self, df, image_folder, transform=None, is_train=True, augment_times=1):
        self.df = df
        self.image_folder = image_folder
        self.transform = transform
        self.is_train = is_train
        self.augment_times = augment_times
        
        # 如果是训练集且需要增强，复制数据条目
        if self.is_train and self.augment_times > 1:
            self.df = pd.concat([self.df] * self.augment_times, ignore_index=True)
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_folder, self.df.iloc[idx]['file_name'])
        
        # 增加文件存在检查
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"无法加载图像: {img_path}, 错误: {e}")
            # 返回默认图像或抛出异常，这里选择返回一个空白图像
            image = Image.new('RGB', (224, 224), color='white')
        
        label = self.df.iloc[idx]['Tongueshape_spots']  # 修改为点刺标签
        
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label, dtype=torch.long)


# ==================== 基于ConvNeXt的舌象分类器 ====================
class ConvNeXtTongueClassifier(nn.Module):
    """基于ConvNeXt的舌象分类模型"""
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        
        # 使用ConvNeXt作为骨干网络
        self.backbone = models.convnext_base(pretrained=pretrained)
        
        # 修改分类头
        self.backbone.classifier = nn.Sequential(
            nn.LayerNorm((1024,), eps=1e-06, elementwise_affine=True),
            nn.Flatten(1),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
        # 注意力机制
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(1024, 1024//16, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(1024//16, 1024, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 全局平均池化层，用于处理特征图
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
    def forward(self, x):
        # 提取特征
        x = self.backbone.features(x)
        
        # 应用通道注意力
        attention = self.channel_attention(x)
        x = x * attention
        
        # 添加全局平均池化
        x = self.global_pool(x)
        
        # 显式调整张量形状为 [batch_size, features]
        x = x.view(x.size(0), -1)
        
        # 分类
        x = self.backbone.classifier(x)
        
        return x

# ==================== 训练和评估函数 ====================
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=50, patience=10):
    """训练模型函数"""
    best_val_loss = float('inf')
    best_model_wts = model.state_dict().copy()
    epochs_no_improve = 0
    
    scaler = GradScaler()  # 混合精度训练
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_auc': []
    }
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # 训练阶段
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in tqdm(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            # 混合精度训练
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            # 反向传播和优化
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # 统计
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc.item())
        
        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        
        # 验证阶段
        val_loss, val_acc, val_auc = evaluate_model(model, val_loader, criterion, device)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_auc'].append(val_auc)
        
        print(f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} AUC: {val_auc:.4f}')
        
        # 学习率调整
        scheduler.step(val_loss)
        
        # 早停机制
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = model.state_dict().copy()
            epochs_no_improve = 0
            print('Best model saved!')
        else:
            epochs_no_improve += 1
            print(f'Epochs with no improvement: {epochs_no_improve}')
            if epochs_no_improve >= patience:
                print('Early stopping triggered!')
                break
        
        print()
    
    # 加载最佳模型权重
    model.load_state_dict(best_model_wts)
    return model, history

def evaluate_model(model, dataloader, criterion, device):
    """评估模型函数"""
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    all_labels = []
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            _, preds = torch.max(outputs, 1)
            probs = F.softmax(outputs, dim=1)[:, 1]  # 正类概率
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # 计算评估指标
    avg_loss = running_loss / len(dataloader.dataset)
    accuracy = running_corrects.double() / len(dataloader.dataset)
    
    # 计算AUC
    if len(np.unique(all_labels)) > 1:
        auc = roc_auc_score(all_labels, all_probs)
    else:
        auc = 0.5  # 如果只有一个类别，则AUC设为0.5
    
    # 计算详细分类指标
    if len(np.unique(all_labels)) > 1:
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
        print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}')
    else:
        precision, recall, f1 = 0, 0, 0
        print(f'只有一个类别的样本，无法计算Precision, Recall和F1-score')
    
    return avg_loss, accuracy.item(), auc

# 测试时增强
def predict_with_tta(model, image, transform, tta_steps=5):
    """测试时增强(TTA)函数"""
    model.eval()
    all_preds = []
    
    with torch.no_grad():
        # 原始图像
        input_tensor = transform(image).unsqueeze(0).to(device)
        output = model(input_tensor)
        all_preds.append(F.softmax(output, dim=1))
        
        # 水平翻转
        for _ in range(tta_steps):
            augmented_img = transforms.RandomHorizontalFlip()(image)
            input_tensor = transform(augmented_img).unsqueeze(0).to(device)
            output = model(input_tensor)
            all_preds.append(F.softmax(output, dim=1))
            
            # 轻微旋转
            augmented_img = transforms.RandomRotation(10)(image)
            input_tensor = transform(augmented_img).unsqueeze(0).to(device)
            output = model(input_tensor)
            all_preds.append(F.softmax(output, dim=1))
    
    # 集成所有预测
    avg_preds = torch.mean(torch.cat(all_preds), dim=0)
    return avg_preds.cpu().numpy()

# ==================== 标签处理工具 ====================
class LabelProcessor:
    """标签处理工具类"""
    @staticmethod
    def process_labels(df, label_column='Tongueshape_spots'):  # 修改为点刺标签
        """
        处理标签列，将各种格式的标签转换为0和1
        """
        print(f"开始处理标签列: {label_column}")
        
        # 检查标签列是否存在
        if label_column not in df.columns:
            raise ValueError(f"CSV文件中缺少'{label_column}'标签列")
        
        # 获取标签列的唯一值
        unique_labels = df[label_column].unique()
        print(f"发现的唯一标签值: {unique_labels}")
        
        # 尝试多种标签转换方法
        conversion_methods = [
            LabelProcessor._convert_boolean,
            LabelProcessor._convert_numeric,
            LabelProcessor._convert_string,
            LabelProcessor._convert_custom
        ]
        
        converted_df = df.copy()
        success = False
        
        for method in conversion_methods:
            try:
                converted_df, success = method(converted_df, label_column)
                if success:
                    break
            except Exception as e:
                print(f"使用方法 {method.__name__} 转换失败: {e}")
        
        if not success:
            print("所有标签转换方法均失败，请检查CSV文件中的标签格式")
            print("将尝试使用自定义映射，请提供标签映射字典")
            custom_mapping = {}
            for label in unique_labels:
                try:
                    value = int(input(f"请输入标签 '{label}' 对应的数值 (0或1): "))
                    if value not in [0, 1]:
                        print("输入无效，默认为0")
                        value = 0
                    custom_mapping[label] = value
                except:
                    print("输入无效，默认为0")
                    custom_mapping[label] = 0
            
            converted_df[label_column] = converted_df[label_column].map(custom_mapping)
            success = True
        
        # 验证转换结果
        positive_df = converted_df[converted_df[label_column] == 1]
        negative_df = converted_df[converted_df[label_column] == 0]
        print(f"转换后正类样本数: {len(positive_df)}")
        print(f"转换后反类样本数: {len(negative_df)}")
        print(f"未转换的样本数: {len(converted_df) - len(positive_df) - len(negative_df)}")
        
        if len(positive_df) + len(negative_df) < len(converted_df):
            # 移除未成功转换的样本
            converted_df = pd.concat([positive_df, negative_df])
            print(f"移除未成功转换的样本后，剩余样本数: {len(converted_df)}")
        
        return converted_df
    
    @staticmethod
    def _convert_boolean(df, label_column):
        """尝试将标签转换为布尔值"""
        try:
            # 尝试直接转换为布尔值
            df[label_column] = df[label_column].astype(bool).astype(int)
            print("成功将标签转换为布尔值")
            return df, True
        except:
            print("无法将标签转换为布尔值")
            return df, False
    
    @staticmethod
    def _convert_numeric(df, label_column):
        """尝试将标签转换为数值"""
        try:
            # 尝试直接转换为数值
            df[label_column] = pd.to_numeric(df[label_column])
            # 假设大于0的值为1，其余为0
            df[label_column] = (df[label_column] > 0).astype(int)
            print("成功将标签转换为数值")
            return df, True
        except:
            print("无法将标签转换为数值")
            return df, False
    
    @staticmethod
    def _convert_string(df, label_column):
        """尝试将字符串标签转换为数值"""
        try:
            # 点刺分类的正类表示
            positive_labels = ['true', 'yes', 'positive', '是', '有', '点刺', '芒刺', '1']  # 添加点刺相关标签
            # 点刺分类的反类表示
            negative_labels = ['false', 'no', 'negative', '否', '无', '无点刺', '光滑', '0']  # 添加点刺相关标签
            
            # 将标签转换为小写字符串
            str_labels = df[label_column].astype(str).str.lower().str.strip()
            
            # 创建映射字典
            mapping = {}
            for label in str_labels.unique():
                if label in positive_labels:
                    mapping[label] = 1
                elif label in negative_labels:
                    mapping[label] = 0
                else:
                    mapping[label] = None  # 未知标签设为None
            
            # 应用映射
            df[label_column] = str_labels.map(mapping)
            
            # 检查是否有未映射的标签
            if df[label_column].isna().sum() > 0:
                print(f"警告: 有 {df[label_column].isna().sum()} 个标签未能映射")
            
            # 移除未映射的标签
            df = df.dropna(subset=[label_column])
            
            print("成功将字符串标签转换为数值")
            return df, True
        except:
            print("无法将字符串标签转换为数值")
            return df, False
    
    @staticmethod
    def _convert_custom(df, label_column):
        """使用自定义映射转换标签"""
        try:
            # 定义点刺分类的中文字符标签映射
            custom_mapping = {
                '点刺': 1, '芒刺': 1, '有刺': 1,
                '无点刺': 0, '光滑': 0, '无刺': 0,
                'True': 1, 'False': 0,
                'true': 1, 'false': 0,
                '1': 1, '0': 0,
                '是': 1, '否': 0,
                '有': 1, '无': 0,
            }
            
            # 应用映射
            df[label_column] = df[label_column].map(custom_mapping)
            
            # 检查是否有未映射的标签
            if df[label_column].isna().sum() > 0:
                print(f"警告: 有 {df[label_column].isna().sum()} 个标签未能映射")
            
            # 移除未映射的标签
            df = df.dropna(subset=[label_column])
            
            print("成功使用自定义映射转换标签")
            return df, True
        except:
            print("无法使用自定义映射转换标签")
            return df, False

# ==================== 主函数 ====================
if __name__ == "__main__":
    # 设置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 数据路径 - 修改为点刺分类的数据路径
    csv_path = '/home/liujian/分类模型/质控/高质量数据/filtered_data_Tongueshape_spots.csv'
    image_folder = '/home/liujian/分类模型/label_segmented_sam_yolov8'
    
    # 读取CSV文件
    try:
        df = pd.read_csv(csv_path)
        print(f"成功读取CSV文件，共有 {len(df)} 条记录")
    except Exception as e:
        print(f"读取CSV文件失败: {e}")
        exit(1)
    
    # 处理标签 - 修改为点刺标签
    df = LabelProcessor.process_labels(df, 'Tongueshape_spots')
    
    # 分离正类和反类
    positive_df = df[df['Tongueshape_spots'] == 1]
    negative_df = df[df['Tongueshape_spots'] == 0]
    
    print(f"过滤后总样本数: {len(df)}")
    print(f"正类样本数: {len(positive_df)}")
    print(f"反类样本数: {len(negative_df)}")
    
    # 检查是否有足够的样本
    if len(positive_df) == 0 and len(negative_df) == 0:
        raise ValueError("数据集中没有有效样本，请检查数据路径和标签")
    elif len(positive_df) == 0:
        print("警告: 数据集中没有正类样本，将只使用反类样本进行训练")
    elif len(negative_df) == 0:
        print("警告: 数据集中没有反类样本，将只使用正类样本进行训练")
    
    # 验证集固定各取50个样本，不足则增强
    val_positive_size = min(50, len(positive_df))
    val_negative_size = min(50, len(negative_df))
    
    val_positive_df = positive_df.sample(val_positive_size, random_state=42)
    val_negative_df = negative_df.sample(val_negative_size, random_state=42)
    
    # 计算需要增强的倍数
    pos_augment_times = int(np.ceil(50 / val_positive_size)) if val_positive_size < 50 else 1
    neg_augment_times = int(np.ceil(50 / val_negative_size)) if val_negative_size < 50 else 1
    
    # 创建验证集DataFrame
    val_df = pd.concat([
        val_positive_df.sample(50, replace=True, random_state=42),
        val_negative_df.sample(50, replace=True, random_state=42)
    ]).reset_index(drop=True)
    
    # 训练集为剩余样本
    train_df = df[~df['id'].isin(val_df['id'])]
    
    print(f"训练集大小: {len(train_df)}")
    print(f"验证集大小: {len(val_df)}")
    
    # 数据增强和预处理
    augmentor = AdvancedAugmentation()
    train_transform = augmentor.get_train_transform(size=224)
    val_transform = augmentor.get_val_transform(size=224)
    
    # 创建数据集
    train_dataset = TongueDataset(
        df=train_df,
        image_folder=image_folder,
        transform=train_transform,
        is_train=True,
        augment_times=2  # 训练集增强2倍
    )
    
    val_dataset = TongueDataset(
        df=val_df,
        image_folder=image_folder,
        transform=val_transform,
        is_train=False
    )
    
    # 处理数据不平衡
    class_counts = train_df['Tongueshape_spots'].value_counts().to_dict()
    class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
    sample_weights = [class_weights[label] for label in train_df['Tongueshape_spots']]
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_dataset),
        replacement=True
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        sampler=sampler,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # 初始化模型
    model = ConvNeXtTongueClassifier(num_classes=2, pretrained=True).to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    
    # 为不同层设置不同的学习率
    params = [
        {'params': model.backbone.features.parameters(), 'lr': 1e-4},
        {'params': model.backbone.classifier.parameters(), 'lr': 1e-3},
        {'params': model.channel_attention.parameters(), 'lr': 1e-3}
    ]
    
    optimizer = optim.AdamW(params, weight_decay=1e-5)
    
    # 学习率调度器
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    # 训练模型
    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=50,
        patience=10
    )
    
    # 保存模型 - 修改为点刺分类的模型名称
    torch.save(model.state_dict(), 'convnext_tongue_spots_classifier.pth')
    
    # 绘制训练历史 - 修改为点刺分类的图表标题
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title('Loss History')
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.plot(history['val_auc'], label='Validation AUC')
    plt.legend()
    plt.title('spots_Accuracy and AUC History')
    
    plt.tight_layout()
    plt.savefig('/home/liujian/分类模型/高质量数据模型/models/点刺/convnext_tongue_spots_training_history.png')