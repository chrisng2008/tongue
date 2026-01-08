"""
复杂模型对比测试脚本
支持 ConvNeXt Large, Swin Transformer, Ensemble 等
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
from models.multitask_models import MultiTaskBackbone, EnsembleMultiTaskModel
from models.multitask_convnext import MultiTaskLoss
from datasets.tongue_multitask_dataset import (
    load_and_clean_data, split_train_val,
    MultiTaskAugmentation, TongueMultiTaskDataset
)
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.cuda.amp import GradScaler, autocast
import torch.optim as optim


def test_complex_models():
    """测试不同复杂度的模型"""
    print("="*60)
    print("复杂模型对比测试")
    print("="*60)

    # 设备配置
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(3)
        gpu_name = torch.cuda.get_device_name(3)
        print(f"\n使用GPU: {gpu_name}")
    else:
        print("\n使用CPU")

    # 加载数据
    print("\n" + "="*60)
    print("加载数据")
    print("="*60)

    excel_path = "/home/wuyongxi/tongue/planner/merge_label/merged_dataset.xlsx"
    image_folder = "/home/wuyongxi/tongue/planner/image"

    df, missing_df, conflict_df = load_and_clean_data(excel_path, image_folder, drop_conflicts=True)
    train_df, val_df, split_stats = split_train_val(df, stratify_by_coat=True)

    # 数据增强
    augmentor = MultiTaskAugmentation()
    train_transform = augmentor.get_train_transform()
    val_transform = augmentor.get_val_transform()

    # 数据集
    train_dataset = TongueMultiTaskDataset(train_df, image_folder, train_transform)
    val_dataset = TongueMultiTaskDataset(val_df, image_folder, val_transform)

    # 采样器
    sample_weights = [1.0] * len(train_dataset)  # 简化，不使用加权
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(train_dataset), replacement=True)

    # 数据加载器
    train_loader = DataLoader(train_dataset, batch_size=8, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)

    print(f"训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}")

    # ==================== 测试不同的模型 ====================
    models_to_test = [
        ('convnext_base', 'ConvNeXt Base (89M)'),
        ('convnext_large', 'ConvNeXt Large (198M)'),
        ('swin_base', 'Swin Transformer Base (88M)'),
    ]

    results = []

    for backbone_name, display_name in models_to_test:
        print("\n" + "="*60)
        print(f"测试: {display_name}")
        print("="*60)

        # 创建模型
        model = MultiTaskBackbone(backbone_name, pretrained=True, freeze_backbone=False).to(device)
        total, trainable, frozen = model.get_num_parameters()
        print(f"参数量: {total/1e6:.1f}M (可训练: {trainable/1e6:.1f}M)")

        # 损失函数
        criterion = MultiTaskLoss(w_shape=1.0, w_coat=1.0, use_focal=True, focal_gamma=2.0).to(device)

        # 测试前向传播
        print("\n测试前向传播...")
        model.eval()
        with torch.no_grad():
            for images, y_shape, y_coat, sample_weights, _ in val_loader:
                images = images.to(device)
                y_shape = y_shape.to(device)
                y_coat = y_coat.to(device)

                logits_shape, logits_coat = model(images)
                loss, loss_dict = criterion(logits_shape, logits_coat, y_shape, y_coat)

                print(f"  Batch Loss: {loss_dict['total']:.4f} "
                      f"(Shape: {loss_dict['shape']:.4f}, Coat: {loss_dict['coat']:.4f})")
                print(f"  Logits shape: {logits_shape.shape}, {logits_coat.shape}")
                break

        # 简单测试训练（1个epoch）
        print("\n测试训练速度（1个batch）...")
        model.train()
        optimizer = optim.AdamW(model.parameters(), lr=1e-4)
        scaler = GradScaler()

        import time
        for images, y_shape, y_coat, sample_weights, _ in train_loader:
            images = images.to(device)
            y_shape = y_shape.to(device)
            y_coat = y_coat.to(device)
            sample_weights = sample_weights.to(device)

            optimizer.zero_grad()
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start = time.time()

            with autocast():
                logits_shape, logits_coat = model(images)
                loss, loss_dict = criterion(logits_shape, logits_coat, y_shape, y_coat, sample_weights)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            torch.cuda.synchronize() if torch.cuda.is_available() else None
            elapsed = time.time() - start

            print(f"  训练时间: {elapsed:.3f}s/batch (batch_size=8)")
            print(f"  预估速度: {8/elapsed:.1f} samples/sec")
            break

        # 评估验证集（仅计算loss）
        print("\n快速验证集评估...")
        model.eval()
        val_losses = []
        with torch.no_grad():
            for images, y_shape, y_coat, sample_weights, _ in val_loader:
                images = images.to(device)
                y_shape = y_shape.to(device)
                y_coat = y_coat.to(device)
                sample_weights = sample_weights.to(device)

                with autocast():
                    logits_shape, logits_coat = model(images)
                    loss, loss_dict = criterion(logits_shape, logits_coat, y_shape, y_coat, sample_weights)
                val_losses.append(loss_dict['total'])

                if len(val_losses) >= 10:  # 只测试前10个batch
                    break

        avg_val_loss = sum(val_losses) / len(val_losses)
        print(f"  验证Loss (前10批): {avg_val_loss:.4f}")

        results.append({
            'model': display_name,
            'params_M': total / 1e6,
            'val_loss': avg_val_loss,
            'speed_samples_per_sec': 8/elapsed
        })

        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ==================== 打印对比结果 ====================
    print("\n" + "="*60)
    print("模型对比结果")
    print("="*60)
    print(f"{'模型':<25} {'参数(M)':<10} {'验证Loss':<12} {'速度(samples/s)':<15}")
    print("-" * 60)
    for r in results:
        print(f"{r['model']:<25} {r['params_M']:>8.1f}  {r['val_loss']:>10.4f}  {r['speed_samples_per_sec']:>13.1f}")

    print("\n推荐:")
    best_loss = min(results, key=lambda x: x['val_loss'])
    print(f"  验证Loss最低: {best_loss['model']}")

    fastest = max(results, key=lambda x: x['speed_samples_per_sec'])
    print(f"  训练最快: {fastest['model']}")

    largest = max(results, key=lambda x: x['params_M'])
    print(f"  参数最多: {largest['model']} (可能效果最好)")

    print("\n" + "="*60)


if __name__ == "__main__":
    test_complex_models()
