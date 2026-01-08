"""
舌象多任务联合训练脚本
共享 ConvNeXt Base Backbone + 双 Head（舌形多标签 + 腻腐苔多类）

运行方式:
    conda activate tongue
    python train_multitask.py

注意: 配置参数在文件顶部 CONFIG 区域修改，不使用命令行参数
"""
import os
import sys
import json
import shutil
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datasets.tongue_multitask_dataset import (
    load_and_clean_data, split_train_val, compute_sample_weights,
    print_dataset_statistics, TongueMultiTaskDataset, MultiTaskAugmentation
)
from models.multitask_models import MultiTaskBackbone
from models.multitask_convnext import MultiTaskLoss
from utils.metrics import (
    MultiTaskMetrics, save_confusion_matrix, save_predictions_csv,
    compute_class_weights, search_best_thresholds, compute_metrics_with_thresholds
)


# ==================== 配置区域（CONFIG） ====================
class CONFIG:
    # ========== 数据路径 ==========
    EXCEL_PATH = "/home/wuyongxi/tongue/planner/merge_label/merged_dataset.xlsx"
    IMAGE_FOLDER = "/home/wuyongxi/tongue/planner/image"

    # ========== 输出目录（自动添加时间戳） ==========
    OUTPUT_DIR = f"/home/wuyongxi/tongue/planner/outputs/{datetime.now().strftime('%Y%m%d_%H%M%S')}_multitask"

    # ========== GPU配置（必须使用 cuda:3） ==========
    DEVICE_ID = 1

    # ========== 训练参数 ==========
    BATCH_SIZE = 8  # ConvNeXt Large 需要更小的 batch size（原16）
    NUM_WORKERS = 4

    # ========== 两阶段训练配置 ==========
    STAGE1_EPOCHS = 5   # Stage 1: 冻结backbone
    STAGE2_EPOCHS = 30  # Stage 2: 全量训练

    # ========== 学习率 ==========
    STAGE1_LR_HEAD = 1e-3
    STAGE2_LR_BACKBONE = 3e-5
    STAGE2_LR_HEAD = 1e-4

    # ========== 优化器 ==========
    WEIGHT_DECAY = 1e-5
    MOMENTUM = 0.9

    # ========== 学习率调度 ==========
    SCHEDULER_TYPE = 'cosine'  # 'cosine' or 'plateau'
    T_MAX = 10  # for cosine
    PATIENCE = 5  # for plateau

    # ========== Early stopping ==========
    EARLY_STOPPING_PATIENCE = 10

    # ==================== 改进选项 ====================
    # ========== 数据划分 ==========
    STRATIFIED_SPLIT = True  # 使用 stratified split（按腻腐苔类别）

    # ========== 阈值搜索 ==========
    SEARCH_BEST_THRESHOLDS = True  # 是否搜索最优阈值
    THRESHOLD_SEARCH_STEP = 0.01   # 搜索步长（0.01更精细，0.05更快）
    THRESHOLD_SEARCH_RANGE = (0.05, 0.95)  # 搜索范围

    # ========== Rotten 类别强化 ==========
    USE_FOCAL_LOSS = True  # 使用 Focal Loss（推荐）或加权 CE
    FOCAL_GAMMA = 2.0      # Focal Loss gamma
    FOCAL_ALPHA = None     # Focal Loss alpha (None=自动, 或手动设置如 [0.5, 2.0, 0.5])

    # 如果不使用 Focal Loss，使用加权 CE 的最大权重限制
    MAX_CLASS_WEIGHT = 30  # 避免权重爆炸

    # ========== 实验配置 ==========
    EXPERIMENT_NAME = "convnext_large"  # 使用 ConvNeXt Large
    LOG_SAMPLING_STATS = True  # 记录每个 epoch 的采样统计（rotten 出现次数）

    # 损失权重
    W_SHAPE = 1.0
    W_COAT = 1.0

    # 数据增强
    IMG_SIZE = 224
    TRAIN_AUGMENT_TIMES = 1  # 训练集增强倍数

    # 数据清洗
    DROP_CONFLICTS = True  # 是否丢弃逻辑冲突样本

    # 验证集划分
    VAL_RATIO = 0.2
    RANDOM_STATE = 42

    # 质量权重（在Dataset中定义，这里仅作说明）
    # HQ-HQ: 1.0, HQ-MQ: 0.85, MQ-HQ: 0.85, MQ-MQ: 0.7, LQ-LQ: 0.5

    # 保存频率
    SAVE_FREQ = 5  # 每N个epoch保存一次checkpoint


# ==================== 工具函数 ====================
def set_seed(seed):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)

    # 确保cudnn的确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_output_dir(base_dir):
    """创建输出目录结构"""
    dirs = [
        base_dir,
        os.path.join(base_dir, 'checkpoints'),
        os.path.join(base_dir, 'logs'),
        os.path.join(base_dir, 'reports'),
        os.path.join(base_dir, 'figures'),
    ]

    for d in dirs:
        os.makedirs(d, exist_ok=True)

    return base_dir


def save_checkpoint(model, optimizer, scheduler, epoch, metrics, is_best, save_dir, filename):
    """保存checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'metrics': metrics,
    }

    path = os.path.join(save_dir, 'checkpoints', filename)
    torch.save(checkpoint, path)

    if is_best:
        best_path = os.path.join(save_dir, 'checkpoints', 'best.pt')
        shutil.copy(path, best_path)


def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    """加载checkpoint"""
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return checkpoint['epoch'], checkpoint['metrics']


def plot_training_history(history, save_dir):
    """绘制训练历史"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train Loss')
    axes[0, 0].plot(history['val_loss'], label='Val Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Shape Loss
    axes[0, 1].plot(history['train_shape_loss'], label='Train Shape Loss')
    axes[0, 1].plot(history['val_shape_loss'], label='Val Shape Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Shape Loss (Multi-label)')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Coat Loss
    axes[1, 0].plot(history['train_coat_loss'], label='Train Coat Loss')
    axes[1, 0].plot(history['val_coat_loss'], label='Val Coat Loss')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('Coat Loss (Multi-class)')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Combined Score
    axes[1, 1].plot(history['val_score'], label='Val Combined Score')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_title('Combined Score (0.5*shape_f1 + 0.5*coat_f1)')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'figures/training_history.png'), dpi=150)
    plt.close()


# ==================== 训练和评估函数 ====================
def train_one_epoch(model, dataloader, criterion, optimizer, scaler, device, log_sampling=False):
    """训练一个epoch（可选记录采样统计）"""
    model.train()

    total_loss = 0
    total_shape_loss = 0
    total_coat_loss = 0
    num_batches = 0

    # 采样统计
    sampling_stats = {
        'greasy_count': 0,
        'rotten_count': 0,
        'nospecial_count': 0,
        'total_samples': 0
    }

    pbar = tqdm(dataloader, desc='Training')
    for images, y_shape, y_coat, sample_weights, _ in pbar:
        images = images.to(device)
        y_shape = y_shape.to(device)
        y_coat = y_coat.to(device)
        sample_weights = sample_weights.to(device)

        # 记录采样统计
        if log_sampling:
            for coat_class in y_coat.cpu().numpy():
                sampling_stats['total_samples'] += 1
                if coat_class == 0:  # greasy
                    sampling_stats['greasy_count'] += 1
                elif coat_class == 1:  # rotten
                    sampling_stats['rotten_count'] += 1
                else:  # nospecial
                    sampling_stats['nospecial_count'] += 1

        optimizer.zero_grad()

        # 混合精度训练
        with autocast():
            logits_shape, logits_coat = model(images)
            loss, loss_dict = criterion(logits_shape, logits_coat, y_shape, y_coat, sample_weights)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # 累积
        total_loss += loss_dict['total']
        total_shape_loss += loss_dict['shape']
        total_coat_loss += loss_dict['coat']
        num_batches += 1

        # 更新进度条
        pbar.set_postfix({
            'loss': f"{loss_dict['total']:.4f}",
            'shape': f"{loss_dict['shape']:.4f}",
            'coat': f"{loss_dict['coat']:.4f}"
        })

    return {
        'total': total_loss / num_batches,
        'shape': total_shape_loss / num_batches,
        'coat': total_coat_loss / num_batches
    }, sampling_stats if log_sampling else None


def evaluate(model, dataloader, criterion, device):
    """评估模型"""
    model.eval()

    total_loss = 0
    total_shape_loss = 0
    total_coat_loss = 0
    num_batches = 0

    metrics_calculator = MultiTaskMetrics()
    all_file_names = []

    with torch.no_grad():
        for images, y_shape, y_coat, sample_weights, file_names in tqdm(dataloader, desc='Evaluating'):
            images = images.to(device)
            y_shape = y_shape.to(device)
            y_coat = y_coat.to(device)
            sample_weights = sample_weights.to(device)

            # 前向传播
            with autocast():
                logits_shape, logits_coat = model(images)
                loss, loss_dict = criterion(logits_shape, logits_coat, y_shape, y_coat, sample_weights)

            # 累积损失
            total_loss += loss_dict['total']
            total_shape_loss += loss_dict['shape']
            total_coat_loss += loss_dict['coat']
            num_batches += 1

            # 累积预测
            metrics_calculator.update(logits_shape, logits_coat, y_shape, y_coat)

            # 记录文件名
            all_file_names.extend(file_names)

    # 计算指标
    metrics, (y_shape, y_coat, pred_shape, pred_coat, prob_shape, prob_coat) = metrics_calculator.compute()

    return {
        'loss': {
            'total': total_loss / num_batches,
            'shape': total_shape_loss / num_batches,
            'coat': total_coat_loss / num_batches
        },
        'metrics': metrics,
        'predictions': {
            'file_names': all_file_names,
            'y_shape': y_shape,
            'y_coat': y_coat,
            'pred_shape': pred_shape,
            'pred_coat': pred_coat,
            'prob_shape': prob_shape,
            'prob_coat': prob_coat
        }
    }


# ==================== 主训练函数 ====================
def main():
    print("="*60)
    print("舌象多任务联合训练")
    print("="*60)

    # 设置随机种子
    set_seed(CONFIG.RANDOM_STATE)

    # 设备配置
    device = torch.device(f"cuda:{CONFIG.DEVICE_ID}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(CONFIG.DEVICE_ID)
        gpu_name = torch.cuda.get_device_name(CONFIG.DEVICE_ID)
        print(f"\n使用GPU: {gpu_name} (cuda:{CONFIG.DEVICE_ID})")
    else:
        print("\n使用CPU")

    # 创建输出目录
    output_dir = setup_output_dir(CONFIG.OUTPUT_DIR)
    print(f"\n输出目录: {output_dir}")

    # 保存配置
    config_dict = {k: v for k, v in CONFIG.__dict__.items() if not k.startswith('_')}
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config_dict, f, indent=2)

    # ==================== 加载和清洗数据 ====================
    print("\n" + "="*60)
    print("数据加载与清洗")
    print("="*60)

    df, missing_df, conflict_df = load_and_clean_data(
        CONFIG.EXCEL_PATH,
        CONFIG.IMAGE_FOLDER,
        drop_conflicts=CONFIG.DROP_CONFLICTS
    )

    if df is None:
        print("数据加载失败，退出！")
        return

    # 打印统计信息
    print_dataset_statistics(df, "清洗后数据集")

    # 保存报告
    if missing_df is not None and len(missing_df) > 0:
        missing_path = os.path.join(output_dir, 'reports', 'missing_images.csv')
        missing_df.to_csv(missing_path, index=False)
        print(f"\n已保存缺失图像报告: {missing_path}")

    if conflict_df is not None and len(conflict_df) > 0:
        conflict_path = os.path.join(output_dir, 'reports', 'conflicts.csv')
        conflict_df.to_csv(conflict_path, index=False)
        print(f"已保存冲突报告: {conflict_path}")

    # ==================== 划分训练验证集 ====================
    print("\n" + "="*60)
    print("数据集划分")
    print("="*60)

    # 使用 stratified split
    train_df, val_df, split_stats = split_train_val(
        df,
        CONFIG.VAL_RATIO,
        CONFIG.RANDOM_STATE,
        stratify_by_coat=CONFIG.STRATIFIED_SPLIT
    )

    # 保存划分和统计
    train_df.to_csv(os.path.join(output_dir, 'reports', 'train_split.csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, 'reports', 'val_split.csv'), index=False)

    # 保存划分统计
    with open(os.path.join(output_dir, 'reports', 'split_stats.json'), 'w') as f:
        json.dump(split_stats, f, indent=2)

    print_dataset_statistics(train_df, "训练集")
    print_dataset_statistics(val_df, "验证集")

    # ==================== 创建数据集和加载器 ====================
    print("\n" + "="*60)
    print("创建数据加载器")
    print("="*60)

    # 数据增强
    augmentor = MultiTaskAugmentation()
    train_transform = augmentor.get_train_transform(CONFIG.IMG_SIZE)
    val_transform = augmentor.get_val_transform(CONFIG.IMG_SIZE)

    # 数据集
    train_dataset = TongueMultiTaskDataset(train_df, CONFIG.IMAGE_FOLDER, train_transform)
    val_dataset = TongueMultiTaskDataset(val_df, CONFIG.IMAGE_FOLDER, val_transform)

    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")

    # 采样器（按腻腐苔三类加权）
    sample_weights = compute_sample_weights(train_df)
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_dataset),
        replacement=True
    )

    # 数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG.BATCH_SIZE,
        sampler=sampler,
        num_workers=CONFIG.NUM_WORKERS,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG.BATCH_SIZE,
        shuffle=False,
        num_workers=CONFIG.NUM_WORKERS,
        pin_memory=True
    )

    # ==================== 创建模型 ====================
    print("\n" + "="*60)
    print("创建模型")
    print("="*60)

    # 计算腻腐苔class weights
    class_weights = compute_class_weights(train_df)
    print(f"\n腻腐苔class weights: {class_weights}")

    # 模型 - 使用 ConvNeXt Large
    model = MultiTaskBackbone(
        backbone_name='convnext_large',
        pretrained=True,
        freeze_backbone=True
    ).to(device)

    total_params, trainable_params, frozen_params = model.get_num_parameters()
    print(f"\n总参数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    print(f"冻结参数: {frozen_params:,}")

    # 损失函数（支持 Focal Loss 或加权 CE）
    if CONFIG.USE_FOCAL_LOSS:
        # 使用 Focal Loss（推荐）
        focal_alpha = CONFIG.FOCAL_ALPHA
        if focal_alpha is None:
            # 自动设置 alpha：rotten 权重更高
            total = sum([max(c, 1) for c in [train_df['Tonguecoat_greasy'].sum(),
                                              train_df['Tonguecoat_rotten'].sum(),
                                              train_df['Tonguecoat_nospecialgreasy'].sum()]])
            n_classes = 3
            greasy_w = total / (n_classes * max(train_df['Tonguecoat_greasy'].sum(), 1))
            rotten_w = total / (n_classes * max(train_df['Tonguecoat_rotten'].sum(), 1)) * 2.0  # rotten加倍
            nospecial_w = total / (n_classes * max(train_df['Tonguecoat_nospecialgreasy'].sum(), 1))
            focal_alpha = [greasy_w, rotten_w, nospecial_w]

        criterion = MultiTaskLoss(
            w_shape=CONFIG.W_SHAPE,
            w_coat=CONFIG.W_COAT,
            use_focal=True,
            focal_gamma=CONFIG.FOCAL_GAMMA,
            focal_alpha=focal_alpha
        ).to(device)

        # 保存损失配置
        loss_config = {
            'type': 'focal_loss',
            'gamma': CONFIG.FOCAL_GAMMA,
            'alpha': focal_alpha
        }
    else:
        # 使用加权 CE（限制最大权重避免爆炸）
        clamped_weights = [min(w, CONFIG.MAX_CLASS_WEIGHT) for w in class_weights]
        criterion = MultiTaskLoss(
            w_shape=CONFIG.W_SHAPE,
            w_coat=CONFIG.W_COAT,
            class_weights=clamped_weights,
            use_focal=False
        ).to(device)

        # 保存损失配置
        loss_config = {
            'type': 'weighted_ce',
            'class_weights': clamped_weights,
            'max_weight_limit': CONFIG.MAX_CLASS_WEIGHT
        }

    # 保存损失配置到文件
    with open(os.path.join(output_dir, 'reports', 'loss_config.json'), 'w') as f:
        json.dump(loss_config, f, indent=2)

    # ==================== 两阶段训练 ====================
    history = {
        'train_loss': [],
        'train_shape_loss': [],
        'train_coat_loss': [],
        'val_loss': [],
        'val_shape_loss': [],
        'val_coat_loss': [],
        'val_score': [],
        'val_score_thresholded': [],  # 使用最优阈值的得分
        'sampling_log': []  # 采样统计日志
    }

    best_score = 0
    epochs_no_improve = 0
    global_epoch = 0

    for stage in [1, 2]:
        print("\n" + "="*60)
        if stage == 1:
            print(f"Stage {stage}: 冻结Backbone，只训练Heads ({CONFIG.STAGE1_EPOCHS} epochs)")
            num_epochs = CONFIG.STAGE1_EPOCHS
            lr = CONFIG.STAGE1_LR_HEAD
        else:
            print(f"Stage {stage}: 解冻Backbone，全量训练 ({CONFIG.STAGE2_EPOCHS} epochs)")
            num_epochs = CONFIG.STAGE2_EPOCHS
            model.unfreeze_backbone()
            total_params, trainable_params, frozen_params = model.get_num_parameters()
            print(f"可训练参数: {trainable_params:,}")

        # 优化器
        param_groups = model.get_parameters(stage=stage)
        optimizer = optim.AdamW(param_groups, weight_decay=CONFIG.WEIGHT_DECAY)

        # 学习率调度器
        if CONFIG.SCHEDULER_TYPE == 'cosine':
            scheduler = CosineAnnealingLR(optimizer, T_max=CONFIG.T_MAX)
        else:
            scheduler = ReduceLROnPlateau(
                optimizer, mode='max', factor=0.5, patience=CONFIG.PATIENCE
            )

        # GradScaler
        scaler = GradScaler()

        # 训练循环
        for epoch in range(num_epochs):
            global_epoch += 1
            print(f"\nEpoch {global_epoch}/{CONFIG.STAGE1_EPOCHS + CONFIG.STAGE2_EPOCHS} (Stage {stage})")
            print("-" * 60)

            # 训练（带采样日志）
            train_losses, sampling_stats = train_one_epoch(
                model, train_loader, criterion, optimizer, scaler, device,
                log_sampling=CONFIG.LOG_SAMPLING_STATS
            )

            # 记录采样日志
            if CONFIG.LOG_SAMPLING_STATS and sampling_stats:
                sampling_stats['epoch'] = global_epoch
                sampling_stats['stage'] = stage
                history['sampling_log'].append(sampling_stats)

                # 打印采样统计
                rotten_pct = 100.0 * sampling_stats['rotten_count'] / sampling_stats['total_samples']
                print(f"\n采样统计: Rotten={sampling_stats['rotten_count']} ({rotten_pct:.1f}%)")

            # 验证
            val_results = evaluate(model, val_loader, criterion, device)

            # 记录历史
            history['train_loss'].append(train_losses['total'])
            history['train_shape_loss'].append(train_losses['shape'])
            history['train_coat_loss'].append(train_losses['coat'])

            history['val_loss'].append(val_results['loss']['total'])
            history['val_shape_loss'].append(val_results['loss']['shape'])
            history['val_coat_loss'].append(val_results['loss']['coat'])

            current_score = val_results['metrics']['combined']
            history['val_score'].append(current_score)

            # 阈值搜索（如果启用）
            if CONFIG.SEARCH_BEST_THRESHOLDS:
                pred = val_results['predictions']
                best_thresholds, best_f1s, _ = search_best_thresholds(
                    pred['y_shape'],
                    pred['prob_shape'],
                    metric='f1',
                    search_range=CONFIG.THRESHOLD_SEARCH_RANGE,
                    step=CONFIG.THRESHOLD_SEARCH_STEP
                )

                # 计算使用最优阈值的指标
                shape_metrics_thresholded = compute_metrics_with_thresholds(
                    pred['y_shape'], pred['prob_shape'], best_thresholds
                )

                # 计算thresholded combined score
                combined_thresholded = 0.5 * shape_metrics_thresholded['macro_f1'] + \
                                       0.5 * val_results['metrics']['coat']['macro_f1']
                history['val_score_thresholded'].append(combined_thresholded)

                print(f"\n【最优阈值】")
                for i, label in enumerate(['spots', 'cracks', 'teethmarks']):
                    print(f"  {label}: {best_thresholds[i]:.2f} (F1={best_f1s[i]:.3f})")
                print(f"Thresholded Combined Score: {combined_thresholded:.3f}")

                # 保存最佳阈值（最后一轮）
                if global_epoch == CONFIG.STAGE1_EPOCHS + CONFIG.STAGE2_EPOCHS or \
                   global_epoch % CONFIG.SAVE_FREQ == 0:
                    thresholds_path = os.path.join(output_dir, 'reports', 'thresholds.json')
                    with open(thresholds_path, 'w') as f:
                        json.dump({
                            'thresholds': best_thresholds,
                            'f1_scores': best_f1s,
                            'epoch': global_epoch
                        }, f, indent=2)

            # 打印结果
            print(f"\nTrain Loss: {train_losses['total']:.4f} "
                  f"(Shape: {train_losses['shape']:.4f}, Coat: {train_losses['coat']:.4f})")
            print(f"Val Loss: {val_results['loss']['total']:.4f} "
                  f"(Shape: {val_results['loss']['shape']:.4f}, Coat: {val_results['loss']['coat']:.4f})")

            # 打印详细指标
            metrics_calculator = MultiTaskMetrics()
            metrics_calculator.print_metrics(val_results['metrics'])

            # 学习率调整
            if CONFIG.SCHEDULER_TYPE == 'cosine':
                scheduler.step()
            else:
                scheduler.step(current_score)

            # 保存checkpoint
            is_best = current_score > best_score
            if is_best:
                best_score = current_score
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if global_epoch % CONFIG.SAVE_FREQ == 0 or is_best:
                save_checkpoint(
                    model, optimizer, scheduler,
                    global_epoch, val_results['metrics'], is_best,
                    output_dir, f'checkpoint_epoch_{global_epoch}.pt'
                )

            # Early stopping
            if epochs_no_improve >= CONFIG.EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping triggered! ({epochs_no_improve} epochs without improvement)")
                break

    # ==================== 最终评估和保存 ====================
    print("\n" + "="*60)
    print("训练完成，最终评估")
    print("="*60)

    # 加载最佳模型
    best_path = os.path.join(output_dir, 'checkpoints', 'best.pt')
    if os.path.exists(best_path):
        checkpoint = torch.load(best_path, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("已加载最佳模型")

    # 最终验证
    final_results = evaluate(model, val_loader, criterion, device)
    metrics_calculator = MultiTaskMetrics()
    metrics_calculator.print_metrics(final_results['metrics'])

    # ==================== 阈值搜索和保存 ====================
    print("\n" + "="*60)
    print("阈值搜索与最终指标")
    print("="*60)

    if CONFIG.SEARCH_BEST_THRESHOLDS:
        pred = final_results['predictions']

        # 搜索最优阈值
        best_thresholds, best_f1s, all_scores = search_best_thresholds(
            pred['y_shape'],
            pred['prob_shape'],
            metric='f1',
            search_range=CONFIG.THRESHOLD_SEARCH_RANGE,
            step=CONFIG.THRESHOLD_SEARCH_STEP
        )

        # 使用最优阈值计算指标
        shape_metrics_thresholded = compute_metrics_with_thresholds(
            pred['y_shape'], pred['prob_shape'], best_thresholds
        )

        # 计算thresholded combined score
        combined_thresholded = 0.5 * shape_metrics_thresholded['macro_f1'] + \
                               0.5 * final_results['metrics']['coat']['macro_f1']

        print("\n【舌形阈值优化结果】")
        for i, label in enumerate(['spots', 'cracks', 'teethmarks']):
            orig_f1 = final_results['metrics']['shape'][label]['f1']
            thresh_f1 = shape_metrics_thresholded[label]['f1']
            improvement = thresh_f1 - orig_f1
            print(f"  {label:12s}: 阈值={best_thresholds[i]:.2f}, "
                  f"F1 {orig_f1:.3f}→{thresh_f1:.3f} (Δ{improvement:+.3f})")

        print(f"\nShape Macro F1: {final_results['metrics']['shape']['macro_f1']:.3f} → "
              f"{shape_metrics_thresholded['macro_f1']:.3f}")
        print(f"Combined Score: {final_results['metrics']['combined']:.3f} → "
              f"{combined_thresholded:.3f}")

        # 保存阈值和详细指标
        thresholds_report = {
            'best_thresholds': best_thresholds,
            'best_f1_scores': best_f1s,
            'shape_metrics_fixed': final_results['metrics']['shape'],
            'shape_metrics_thresholded': shape_metrics_thresholded,
            'combined_fixed': float(final_results['metrics']['combined']),
            'combined_thresholded': float(combined_thresholded)
        }

        with open(os.path.join(output_dir, 'reports', 'val_metrics_thresholded.json'), 'w') as f:
            json.dump(thresholds_report, f, indent=2)

        print(f"\n已保存阈值报告: reports/val_metrics_thresholded.json")
    else:
        combined_thresholded = final_results['metrics']['combined']
        best_thresholds = [0.5, 0.5, 0.5]  # 默认阈值

    # ==================== 采样日志保存 ====================
    if CONFIG.LOG_SAMPLING_STATS and len(history['sampling_log']) > 0:
        sampling_log_df = pd.DataFrame(history['sampling_log'])
        sampling_log_path = os.path.join(output_dir, 'reports', 'epoch_sampling_log.csv')
        sampling_log_df.to_csv(sampling_log_path, index=False)

        # 统计平均采样率
        avg_rotten_pct = 100.0 * sampling_log_df['rotten_count'].sum() / \
                         sampling_log_df['total_samples'].sum()
        print(f"\n【采样统计】")
        print(f"  平均 Rotten 采样率: {avg_rotten_pct:.1f}%")
        print(f"  已保存采样日志: reports/epoch_sampling_log.csv")

    # ==================== 保存报告 ====================
    # 保存训练历史图
    plot_training_history(history, output_dir)

    # 保存混淆矩阵
    save_confusion_matrix(
        final_results['metrics']['coat']['confusion_matrix'],
        MultiTaskMetrics.COAT_CLASSES,
        os.path.join(output_dir, 'figures/confusion_matrix.png'),
        'Coat Confusion Matrix'
    )

    # 保存预测结果（增强版）
    pred = final_results['predictions']

    # 计算使用最优阈值的预测
    pred_shape_thresholded = (pred['prob_shape'] >= np.array(best_thresholds)).astype(int)

    # 保存增强预测CSV
    enhanced_predictions = []
    for i in range(len(pred['file_names'])):
        row = {
            'file_name': pred['file_names'][i],
            # GT
            'GT_spots': int(pred['y_shape'][i, 0]),
            'GT_cracks': int(pred['y_shape'][i, 1]),
            'GT_teethmarks': int(pred['y_shape'][i, 2]),
            'GT_coat_class': int(pred['y_coat'][i]),
            # Shape概率
            'prob_spots': float(pred['prob_shape'][i, 0]),
            'prob_cracks': float(pred['prob_shape'][i, 1]),
            'prob_teethmarks': float(pred['prob_shape'][i, 2]),
            # Shape预测（固定0.5阈值）
            'pred_spots_fixed': int(pred['pred_shape'][i, 0]),
            'pred_cracks_fixed': int(pred['pred_shape'][i, 1]),
            'pred_teethmarks_fixed': int(pred['pred_shape'][i, 2]),
            # Shape预测（最优阈值）
            'pred_spots_thr': int(pred_shape_thresholded[i, 0]),
            'pred_cracks_thr': int(pred_shape_thresholded[i, 1]),
            'pred_teethmarks_thr': int(pred_shape_thresholded[i, 2]),
            # Coat概率和预测
            'prob_greasy': float(pred['prob_coat'][i, 0]),
            'prob_rotten': float(pred['prob_coat'][i, 1]),
            'prob_nospecialgreasy': float(pred['prob_coat'][i, 2]),
            'pred_coat_class': int(pred['pred_coat'][i]),
        }
        enhanced_predictions.append(row)

    enhanced_pred_df = pd.DataFrame(enhanced_predictions)
    enhanced_pred_df.to_csv(os.path.join(output_dir, 'reports/val_predictions_enhanced.csv'), index=False)
    print(f"已保存增强预测结果: reports/val_predictions_enhanced.csv")

    # ==================== 保存summary ====================
    summary = {
        'experiment_name': CONFIG.EXPERIMENT_NAME,
        'timestamp': datetime.now().isoformat(),
        'config': {
            'stratified_split': CONFIG.STRATIFIED_SPLIT,
            'use_focal_loss': CONFIG.USE_FOCAL_LOSS,
            'search_thresholds': CONFIG.SEARCH_BEST_THRESHOLDS,
            'log_sampling': CONFIG.LOG_SAMPLING_STATS,
        },
        'data': {
            'total_samples': len(df),
            'train_samples': len(train_df),
            'val_samples': len(val_df),
            'missing_images': len(missing_df) if missing_df is not None else 0,
            'conflict_samples': len(conflict_df) if conflict_df is not None else 0,
        },
        'training': {
            'total_epochs': global_epoch,
            'best_val_score': float(best_score),
            'best_epoch': int(global_epoch - epochs_no_improve),
        },
        'final_metrics_fixed': {
            'shape_spots_f1': float(final_results['metrics']['shape']['spots']['f1']),
            'shape_cracks_f1': float(final_results['metrics']['shape']['cracks']['f1']),
            'shape_teethmarks_f1': float(final_results['metrics']['shape']['teethmarks']['f1']),
            'shape_macro_f1': float(final_results['metrics']['shape']['macro_f1']),
            'coat_macro_f1': float(final_results['metrics']['coat']['macro_f1']),
            'combined_score': float(final_results['metrics']['combined']),
        },
        'final_metrics_thresholded' if CONFIG.SEARCH_BEST_THRESHOLDS else None: {
            'shape_spots_f1': float(shape_metrics_thresholded['spots']['f1']),
            'shape_cracks_f1': float(shape_metrics_thresholded['cracks']['f1']),
            'shape_teethmarks_f1': float(shape_metrics_thresholded['teethmarks']['f1']),
            'shape_macro_f1': float(shape_metrics_thresholded['macro_f1']),
            'coat_macro_f1': float(final_results['metrics']['coat']['macro_f1']),
            'combined_score': float(combined_thresholded),
            'best_thresholds': [float(t) for t in best_thresholds],
        } if CONFIG.SEARCH_BEST_THRESHOLDS else None,
        'val_support': split_stats  # 验证集支持数
    }

    # 移除None值
    summary = {k: v for k, v in summary.items() if v is not None}

    with open(os.path.join(output_dir, 'reports/summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    # ==================== 最终输出 ====================
    print("\n" + "="*60)
    print("训练完成！")
    print("="*60)
    print(f"\n输出目录: {output_dir}")
    print(f"最佳验证集得分: {best_score:.4f}")
    print(f"\n模型保存在: {os.path.join(output_dir, 'checkpoints/')}")
    print(f"  - best.pt: 最佳模型")
    print(f"  - checkpoint_epoch_{global_epoch}.pt: 最终checkpoint")

    # 打印统计摘要
    print(f"\n数据统计:")
    print(f"  总样本数: {len(df)}")
    print(f"  训练样本: {len(train_df)}")
    print(f"  验证样本: {len(val_df)}")
    print(f"  缺失图像: {len(missing_df) if missing_df is not None else 0}")
    print(f"  逻辑冲突: {len(conflict_df) if conflict_df is not None else 0}")

    print("\n验证集支持数:")
    print(f"  greasy: {split_stats['val_Tonguecoat_greasy']}")
    print(f"  rotten: {split_stats['val_Tonguecoat_rotten']}")
    print(f"  nospecialgreasy: {split_stats['val_Tonguecoat_nospecialgreasy']}")
    print(f"  spots: {split_stats['val_Tongueshape_spots']}")
    print(f"  cracks: {split_stats['val_Tongueshape_cracks']}")
    print(f"  teethmarks: {split_stats['val_Tongueshape_teethmarks']}")

    print("\n最终指标 (固定阈值 0.5):")
    for label in ['spots', 'cracks', 'teethmarks']:
        f1 = final_results['metrics']['shape'][label]['f1']
        print(f"  shape_{label}: F1={f1:.3f}")
    print(f"  coat_macro_f1: {final_results['metrics']['coat']['macro_f1']:.3f}")
    print(f"  combined_score: {final_results['metrics']['combined']:.3f}")

    if CONFIG.SEARCH_BEST_THRESHOLDS:
        print("\n最终指标 (最优阈值):")
        for i, label in enumerate(['spots', 'cracks', 'teethmarks']):
            f1 = shape_metrics_thresholded[label]['f1']
            print(f"  shape_{label}: F1={f1:.3f} (阈值={best_thresholds[i]:.2f})")
        print(f"  coat_macro_f1: {final_results['metrics']['coat']['macro_f1']:.3f}")
        print(f"  combined_score: {combined_thresholded:.3f}")

    print("\n" + "="*60)


if __name__ == "__main__":
    main()
