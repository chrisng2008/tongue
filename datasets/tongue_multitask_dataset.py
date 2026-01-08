"""
多任务舌象数据集
支持舌形多标签分类和腻腐苔多类分类
"""
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import warnings
warnings.filterwarnings('ignore')


class MultiTaskAugmentation:
    """多任务数据增强（复用点刺代码，去掉vertical flip）"""
    def __init__(self):
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def get_train_transform(self, size=224):
        return transforms.Compose([
            transforms.Resize((int(size*1.15), int(size*1.15))),
            transforms.RandomCrop(size),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomVerticalFlip(),  # 去掉垂直翻转
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


class TongueMultiTaskDataset(Dataset):
    """
    舌象多任务数据集

    返回:
        image: [3, H, W] tensor
        y_shape: [3] float tensor (spots, cracks, teethmarks)
        y_coat: int (0=greasy, 1=rotten, 2=nospecialgreasy)
        sample_weight: float (质量权重)
        file_name: str
    """
    def __init__(self, df, image_folder, transform=None):
        self.df = df.reset_index(drop=True)
        self.image_folder = image_folder
        self.transform = transform

        # 舌形标签列
        self.shape_labels = ['Tongueshape_spots', 'Tongueshape_cracks', 'Tongueshape_teethmarks']

        # 腻腐苔标签列
        self.coat_labels = ['Tonguecoat_greasy', 'Tonguecoat_rotten', 'Tonguecoat_nospecialgreasy']

        # 质量权重映射
        self.quality_weights = {
            'HQ-HQ': 1.0,
            'HQ-MQ': 0.85,
            'MQ-HQ': 0.85,
            'MQ-MQ': 0.7,
            'LQ-LQ': 0.5,
            'LQ-MQ': 0.6,
            'LQ-HQ': 0.7,
            None: 1.0,  # 默认权重
        }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_name = row['file_name']

        # 加载图像
        img_path = os.path.join(self.image_folder, file_name)

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            # 图像加载失败，返回空白张图（训练前应已过滤）
            print(f"Warning: 无法加载图像 {img_path}, 错误: {e}")
            image = Image.new('RGB', (224, 224), color='gray')

        # 应用变换
        if self.transform:
            image = self.transform(image)

        # 获取舌形标签（多标签，float）
        y_shape = torch.tensor([
            float(row.get('Tongueshape_spots', 0)),
            float(row.get('Tongueshape_cracks', 0)),
            float(row.get('Tongueshape_teethmarks', 0))
        ], dtype=torch.float32)

        # 获取腻腐苔标签（互斥3类，转换为class index）
        # 0=greasy, 1=rotten, 2=nospecialgreasy
        greasy = int(row.get('Tonguecoat_greasy', 0))
        rotten = int(row.get('Tonguecoat_rotten', 0))
        nospecial = int(row.get('Tonguecoat_nospecialgreasy', 0))

        if greasy:
            y_coat = 0
        elif rotten:
            y_coat = 1
        else:
            y_coat = 2

        # 获取质量权重
        quality_pair = row.get('quality_pair', None)
        if quality_pair is None:
            # 尝试从shape_quality和coat_quality组合
            shape_q = row.get('shape_quality', 'HQ')
            coat_q = row.get('coat_quality', 'HQ')
            quality_pair = f'{shape_q}-{coat_q}'

        sample_weight = torch.tensor(
            self.quality_weights.get(quality_pair, 1.0),
            dtype=torch.float32
        )

        return image, y_shape, y_coat, sample_weight, file_name


def load_and_clean_data(excel_path, image_folder, drop_conflicts=True):
    """
    从Excel加载数据并进行清洗

    参数:
        excel_path: Excel文件路径
        image_folder: 图像文件夹路径
        drop_conflicts: 是否丢弃逻辑冲突的样本

    返回:
        df: 清洗后的DataFrame
        missing_df: 缺失图像的样本
        conflict_df: 逻辑冲突的样本
    """
    print(f"正在读取Excel文件: {excel_path}")

    # 尝试读取不同的sheet
    try:
        df = pd.read_excel(excel_path, sheet_name=0)
        print(f"成功读取，共 {len(df)} 条记录")
    except Exception as e:
        print(f"读取Excel失败: {e}")
        return None, None, None

    # 检查必需列
    required_cols = ['file_name',
                     'Tongueshape_spots', 'Tongueshape_cracks', 'Tongueshape_teethmarks', 'Tongueshape_nospecialshape',
                     'Tonguecoat_greasy', 'Tonguecoat_rotten', 'Tonguecoat_nospecialgreasy']

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"警告: Excel中缺少列: {missing_cols}")

    # 标准化标签值为0/1
    label_cols = [col for col in df.columns if 'Tongueshape_' in col or 'Tonguecoat_' in col]
    for col in label_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

    # 检查缺失图像
    print("\n检查图像文件...")
    missing_images = []
    for idx, row in df.iterrows():
        img_path = os.path.join(image_folder, row['file_name'])
        if not os.path.exists(img_path):
            missing_images.append({
                'file_name': row['file_name'],
                'reason': 'Image not found'
            })

    missing_df = pd.DataFrame(missing_images)
    print(f"发现 {len(missing_df)} 个缺失图像")

    # 过滤掉缺失图像的样本
    if len(missing_df) > 0:
        df = df[~df['file_name'].isin(missing_df['file_name'])]
        print(f"过滤后剩余 {len(df)} 个样本")

    # 检查逻辑冲突
    print("\n检查逻辑一致性...")
    conflicts = []

    for idx, row in df.iterrows():
        conflict_reasons = []

        # 舌形逻辑：nospecialshape == NOT(spots OR cracks OR teethmarks)
        has_spots = bool(row['Tongueshape_spots'])
        has_cracks = bool(row['Tongueshape_cracks'])
        has_teethmarks = bool(row['Tongueshape_teethmarks'])
        has_nospecial = bool(row['Tongueshape_nospecialshape'])

        has_any_special = has_spots or has_cracks or has_teethmarks

        if has_nospecial and has_any_special:
            conflict_reasons.append('nospecialshape与特殊舌形共现')

        if not has_nospecial and not has_any_special:
            conflict_reasons.append('既无特殊舌形也无nospecialshape')

        # 腻腐苔互斥检查
        has_greasy = bool(row['Tonguecoat_greasy'])
        has_rotten = bool(row['Tonguecoat_rotten'])
        has_nospecial_greasy = bool(row['Tonguecoat_nospecialgreasy'])

        # nospecial不能与greasy或rotten共现
        if has_nospecial_greasy and (has_greasy or has_rotten):
            conflict_reasons.append('nospecialgreasy与greasy/rotten共现')

        # greasy与rotten不应共现（可选）
        if has_greasy and has_rotten:
            conflict_reasons.append('greasy与rotten共现')

        if conflict_reasons:
            conflicts.append({
                'file_name': row['file_name'],
                'conflicts': '; '.join(conflict_reasons),
                'spots': has_spots,
                'cracks': has_cracks,
                'teethmarks': has_teethmarks,
                'nospecialshape': has_nospecial,
                'greasy': has_greasy,
                'rotten': has_rotten,
                'nospecialgreasy': has_nospecial_greasy
            })

    conflict_df = pd.DataFrame(conflicts)
    print(f"发现 {len(conflict_df)} 个逻辑冲突样本")

    # 统计冲突类型
    if len(conflict_df) > 0:
        print("\n冲突类型分布:")
        conflict_types = conflict_df['conflicts'].str.split('; ').explode().value_counts()
        for conflict_type, count in conflict_types.items():
            print(f"  {conflict_type}: {count}")

    # 丢弃冲突样本
    if drop_conflicts and len(conflict_df) > 0:
        df = df[~df['file_name'].isin(conflict_df['file_name'])]
        print(f"丢弃冲突样本后剩余 {len(df)} 个样本")

    return df, missing_df, conflict_df


def split_train_val(df, val_ratio=0.2, random_state=42, stratify_by_coat=True):
    """
    划分训练集和验证集（支持按腻腐苔类别分层抽样）

    参数:
        df: 清洗后的DataFrame
        val_ratio: 验证集比例
        random_state: 随机种子
        stratify_by_coat: 是否按腻腐苔类别分层抽样（默认True）

    返回:
        train_df, val_df, split_stats (dict)
    """
    from sklearn.model_selection import train_test_split

    # 创建腻腐苔类别标签（用于分层）
    df['_coat_class'] = 0  # 默认 greasy
    df.loc[df['Tonguecoat_rotten'] == 1, '_coat_class'] = 1  # rotten
    df.loc[df['Tonguecoat_nospecialgreasy'] == 1, '_coat_class'] = 2  # nospecial

    # 检查各类数量
    coat_counts = df['_coat_class'].value_counts().to_dict()
    print(f"\n腻腐苔类别分布:")
    for class_id, name in [(0, 'greasy'), (1, 'rotten'), (2, 'nospecialgreasy')]:
        count = coat_counts.get(class_id, 0)
        print(f"  {name}: {count} 样本")

    # 检查 rotten 数量是否足够
    rotten_count = coat_counts.get(1, 0)
    min_val_rotten = 5

    if stratify_by_coat and rotten_count >= min_val_rotten:
        # 使用分层抽样
        print(f"\n使用 stratified split（按腻腐苔类别）")

        try:
            train_df, val_df = train_test_split(
                df,
                test_size=val_ratio,
                random_state=random_state,
                stratify=df['_coat_class']
            )
        except ValueError as e:
            print(f"Stratified split 失败: {e}")
            print("回退到普通随机 split")
            train_df, val_df = train_test_split(
                df,
                test_size=val_ratio,
                random_state=random_state
            )
    else:
        if rotten_count < min_val_rotten:
            print(f"\n警告: rotten 类别样本过少 ({rotten_count} < {min_val_rotten})")
            print("建议使用 k-fold 交叉验证或收集更多数据")
            print("使用普通随机 split（可能不稳定）")

        train_df, val_df = train_test_split(
            df,
            test_size=val_ratio,
            random_state=random_state
        )

    # 删除临时列
    train_df = train_df.drop('_coat_class', axis=1).reset_index(drop=True)
    val_df = val_df.drop('_coat_class', axis=1).reset_index(drop=True)

    # 统计划分信息
    split_stats = {
        'total_samples': len(df),
        'train_samples': len(train_df),
        'val_samples': len(val_df),
        'val_ratio': val_ratio,
        'stratified': stratify_by_coat and rotten_count >= min_val_rotten,
    }

    # 验证集各类数量
    for col in ['Tonguecoat_greasy', 'Tonguecoat_rotten', 'Tonguecoat_nospecialgreasy']:
        split_stats[f'val_{col}'] = int(val_df[col].sum())
    for col in ['Tongueshape_spots', 'Tongueshape_cracks', 'Tongueshape_teethmarks']:
        split_stats[f'val_{col}'] = int(val_df[col].sum())

    print(f"\n训练集: {len(train_df)} 样本")
    print(f"验证集: {len(val_df)} 样本")

    # 打印验证集类别分布
    print(f"\n验证集腻腐苔分布:")
    print(f"  greasy: {split_stats['val_Tonguecoat_greasy']}")
    print(f"  rotten: {split_stats['val_Tonguecoat_rotten']}")
    print(f"  nospecialgreasy: {split_stats['val_Tonguecoat_nospecialgreasy']}")

    return train_df, val_df, split_stats


def compute_sample_weights(train_df):
    """
    计算样本权重用于WeightedRandomSampler
    按腻腐苔三类计算权重，让rotten类更易被采样

    返回:
        sample_weights: list of float
    """
    # 统计三类数量
    coat_labels = ['Tonguecoat_greasy', 'Tonguecoat_rotten', 'Tonguecoat_nospecialgreasy']
    class_counts = []

    for col in coat_labels:
        count = train_df[col].sum()
        class_counts.append(max(count, 1))  # 避免除0

    # 计算class weight（inverse frequency）
    total = sum(class_counts)
    class_weights = [total / (len(class_counts) * c) for c in class_counts]

    # 给rotten更高的权重（索引1）
    class_weights[1] *= 2.0

    print(f"\n腻腐苔类别统计:")
    print(f"  greasy: {class_counts[0]}, weight: {class_weights[0]:.3f}")
    print(f"  rotten: {class_counts[1]}, weight: {class_weights[1]:.3f}")
    print(f"  nospecialgreasy: {class_counts[2]}, weight: {class_weights[2]:.3f}")

    # 为每个样本分配权重
    sample_weights = []
    for _, row in train_df.iterrows():
        if row['Tonguecoat_greasy']:
            sample_weights.append(class_weights[0])
        elif row['Tonguecoat_rotten']:
            sample_weights.append(class_weights[1])
        else:
            sample_weights.append(class_weights[2])

    return sample_weights


def print_dataset_statistics(df, name="Dataset"):
    """打印数据集统计信息"""
    print(f"\n{'='*60}")
    print(f"{name} 统计信息")
    print(f"{'='*60}")

    print(f"\n总样本数: {len(df)}")

    # 舌形分布
    print("\n舌形标签分布:")
    for label in ['Tongueshape_spots', 'Tongueshape_cracks', 'Tongueshape_teethmarks']:
        count = df[label].sum()
        ratio = count / len(df) * 100
        print(f"  {label}: {count} ({ratio:.1f}%)")

    # 腻腐苔分布
    print("\n腻腐苔标签分布:")
    coat_labels = ['Tonguecoat_greasy', 'Tonguecoat_rotten', 'Tonguecoat_nospecialgreasy']
    for label in coat_labels:
        count = df[label].sum()
        ratio = count / len(df) * 100
        print(f"  {label}: {count} ({ratio:.1f}%)")

    # 质量分布
    if 'quality_pair' in df.columns:
        print("\n质量组合分布:")
        quality_counts = df['quality_pair'].value_counts()
        for quality, count in quality_counts.items():
            ratio = count / len(df) * 100
            print(f"  {quality}: {count} ({ratio:.1f}%)")

    # 共现模式
    print("\n舌形共现模式:")
    df['special_count'] = (
        df['Tongueshape_spots'] +
        df['Tongueshape_cracks'] +
        df['Tongueshape_teethmarks']
    )
    for count in range(4):
        num = (df['special_count'] == count).sum()
        ratio = num / len(df) * 100
        print(f"  {count}个特征共现: {num} ({ratio:.1f}%)")

    print(f"{'='*60}\n")


if __name__ == "__main__":
    # 测试代码
    excel_path = "/home/wuyongxi/tongue/planner/merge_label/merged_dataset.xlsx"
    image_folder = "/home/wuyongxi/tongue/planner/image"

    # 加载和清洗数据
    df, missing_df, conflict_df = load_and_clean_data(excel_path, image_folder)

    if df is not None:
        # 打印统计
        print_dataset_statistics(df, "完整数据集")

        # 划分训练验证集
        train_df, val_df = split_train_val(df)
        print_dataset_statistics(train_df, "训练集")
        print_dataset_statistics(val_df, "验证集")

        # 保存报告
        if missing_df is not None and len(missing_df) > 0:
            missing_df.to_csv("missing_images_report.csv", index=False)
            print(f"\n已保存缺失图像报告: missing_images_report.csv")

        if conflict_df is not None and len(conflict_df) > 0:
            conflict_df.to_csv("conflicts_report.csv", index=False)
            print(f"已保存冲突报告: conflicts_report.csv")
