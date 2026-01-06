# 舌象多任务联合训练系统

## 项目概述

实现了基于 **ConvNeXt Base** 的舌象多任务联合训练系统，共享 Backbone + 双 Head 架构：
- **Task A**: 舌形多标签分类（spots, cracks, teethmarks）
- **Task B**: 腻腐苔多类分类（greasy, rotten, nospecialgreasy）

## 文件结构

```
/home/wuyongxi/tongue/planner/
├── train_multitask.py              # 主训练脚本
├── datasets/
│   ├── __init__.py
│   └── tongue_multitask_dataset.py # 多任务Dataset & 数据增强
├── models/
│   ├── __init__.py
│   └── multitask_convnext.py       # 多任务ConvNeXt模型
├── utils/
│   ├── __init__.py
│   └── metrics.py                  # 评估指标工具
└── outputs/
    └── <timestamp>_multitask/       # 训练输出目录（自动创建）
        ├── checkpoints/            # best.pt, last.pt
        ├── logs/
        ├── reports/                # summary.json, conflicts.csv等
        └── figures/                # 训练历史图、混淆矩阵
```

## 核心特性

### 1. 数据处理
- ✓ 自动读取 `merged_dataset.xlsx`
- ✓ 缺失图像检测与过滤（发现1个）
- ✓ 逻辑冲突检测（发现2个 greasy-rotten 共现）
- ✓ 训练/验证集划分（0.8/0.2，固定seed=42）
- ✓ 质量权重支持（HQ-HQ=1.0, MQ-MQ=0.7等）
- ✓ WeightedRandomSampler（腻腐苔三类采样）

### 2. 模型架构
- **Backbone**: ConvNeXt Base (ImageNet-1K 预训练)
- **Shape Head**: 3个输出（sigmoid激活，多标签）
- **Coat Head**: 3个输出（softmax激活，多类）
- **参数量**: 89.6M 总参数

### 3. 训练策略
- **两阶段训练**:
  - Stage 1 (5 epochs): 冻结backbone，只训heads
  - Stage 2 (30 epochs): 全量fine-tune
- **混合精度训练**: AMP (GradScaler)
- **学习率调度**: CosineAnnealingLR 或 ReduceLROnPlateau
- **Early Stopping**: 10 epochs无提升自动停止
- **多任务损失**: `loss = w_shape * BCE + w_coat * CE`

### 4. 评估指标
- **舌形（多标签）**:
  - Precision, Recall, F1, AUROC（每个标签分别计算）
  - Macro-average F1/Precision/Recall
- **腻腐苔（多类）**:
  - Macro-F1, Weighted-F1
  - Balanced Accuracy
  - Confusion Matrix
- **综合得分**: `0.5 * shape_macro_f1 + 0.5 * coat_macro_f1`

### 5. 输出报告
训练完成后自动生成：
- `checkpoints/best.pt` - 最佳模型
- `checkpoints/last.pt` - 最后一个checkpoint
- `reports/summary.json` - 训练摘要
- `reports/conflicts.csv` - 逻辑冲突样本
- `reports/missing_images.csv` - 缺失图像
- `reports/val_predictions.csv` - 验证集预测详情
- `figures/training_history.png` - 训练曲线
- `figures/confusion_matrix.png` - 腻腐苔混淆矩阵

## 数据统计

### 当前数据集（清洗后）
- **总样本**: 1263 张
- **训练集**: 1010 张
- **验证集**: 253 张
- **丢弃样本**:
  - 缺失图像: 1 张
  - 逻辑冲突: 2 张（greasy-rotten共现）

### 舌形分布（训练集）
- spots: ~30%
- cracks: ~15%
- teethmarks: ~40%

### 腻腐苔分布（训练集）
- greasy: 315 样本
- rotten: 20 样本（极不均衡，已加权）
- nospecialgreasy: 675 样本

## 使用方法

### 1. 激活环境
```bash
conda activate tongue
```

### 2. 修改配置（可选）
编辑 [train_multitask.py](train_multitask.py) 顶部的 `CONFIG` 类：
```python
class CONFIG:
    # 数据路径
    EXCEL_PATH = "/home/wuyongxi/tongue/planner/merge_label/merged_dataset.xlsx"
    IMAGE_FOLDER = "/home/wuyongxi/tongue/planner/image"

    # GPU（已固定为 cuda:3）
    DEVICE_ID = 3

    # 训练参数
    BATCH_SIZE = 16
    STAGE1_EPOCHS = 5
    STAGE2_EPOCHS = 30

    # 损失权重
    W_SHAPE = 1.0
    W_COAT = 1.0

    # 数据清洗
    DROP_CONFLICTS = True  # 是否丢弃冲突样本
```

### 3. 开始训练
```bash
cd /home/wuyongxi/tongue/planner
python train_multitask.py
```

训练过程中会：
1. 加载并清洗数据
2. 打印详细统计信息
3. 创建输出目录（自动添加时间戳）
4. 保存配置到 `config.json`
5. 开始两阶段训练
6. 实时显示训练/验证损失和指标
7. 自动保存最佳模型

### 4. 监控训练
终端会实时显示：
```
Epoch 10/35 (Stage 2)
------------------------------------------------------------
Training: 100%|████████| 63/63 [00:45<00:00,  1.39it/s]
Train Loss: 0.8234 (Shape: 0.3124, Coat: 0.5110)
Val Loss: 0.7456 (Shape: 0.2891, Coat: 0.4565)

【舌形多标签】
spots        : P=0.723, R=0.681, F1=0.701, AUROC=0.812
cracks       : P=0.545, R=0.512, F1=0.528, AUROC=0.721
teethmarks   : P=0.812, R=0.834, F1=0.823, AUROC=0.901

【腻腐苔多类】
macro_f1: 0.654
balanced_accuracy: 0.678
```

### 5. 查看结果
训练完成后检查：
```bash
cd /home/wuyongxi/tongue/planner/outputs/<timestamp>_multitask

# 查看摘要
cat reports/summary.json

# 查看验证集预测
head reports/val_predictions.csv

# 查看训练曲线
open figures/training_history.png
```

## 代码复用说明

本实现充分复用了 [model/点刺.py](model/点刺.py) 中的组件：

### 复用的部分
1. **数据增强**: `AdvancedAugmentation` → `MultiTaskAugmentation`
   - 保留所有增强操作（去除vertical flip）
   - ImageNet归一化参数

2. **训练循环**: `train_model` → `train_one_epoch` + `evaluate`
   - AMP混合精度训练（GradScaler）
   - 学习率调度（CosineAnnealingWarmRestarts → CosineAnnealingLR）
   - Early stopping机制
   - 进度条显示（tqdm）

3. **采样策略**: `WeightedRandomSampler`
   - 从点刺二类扩展到腻腐苔三类
   - 稀疏类（rotten）权重加倍

### 改造的部分
1. **Dataset**: 单标签 → 多任务输出
   - 原始: `return image, label`
   - 现在: `return image, y_shape, y_coat, sample_weight, file_name`

2. **模型**: 单头 → 双头
   - 原始: 单classifier输出2类
   - 现在: shape_head(3) + coat_head(3)

3. **损失函数**: CE → BCE + CE
   - 原始: `CrossEntropyLoss`
   - 现在: `MultiTaskLoss = w_shape * BCE + w_coat * CE`

4. **评估**: 单类指标 → 多任务综合指标
   - 原始: accuracy, AUC, F1
   - 现在: 舌形三标签 + 腻腐苔三类的完整指标矩阵

## 测试验证

已完成的测试：
- ✓ 数据加载与清洗（1263样本）
- ✓ Dataset创建与迭代
- ✓ 模型构建（ConvNeXt Base）
- ✓ 前向传播（logits_shape, logits_coat）
- ✓ 多任务损失计算
- ✓ 评估指标计算
- ✓ 完整pipeline端到端测试

## GPU配置

- **设备**: cuda:3 (NVIDIA GeForce RTX 3080 Ti)
- **混合精度**: 支持FP16加速
- **内存占用**: 预计 ~8GB（batch_size=16）

## 注意事项

1. **数据质量**: 建议定期检查 `reports/conflicts.csv` 和 `missing_images.csv`
2. **类别不均衡**: rotten类仅20样本，训练可能不稳定，建议：
   - 增加 `W_COAT` 权重
   - 使用更多数据增强
   - 考虑focal loss
3. **超参数调整**:
   - 若过拟合：增加 `WEIGHT_DECAY` 或 `Dropout`
   - 若欠拟合：增加 `STAGE2_EPOCHS` 或降低学习率
4. **Early Stopping**: 当前patience=10，可根据数据集大小调整

## 下一步优化建议

1. **测试时增强（TTA）**: 复用点刺模型的TTA逻辑
2. **阈值搜索**: 为舌形三标签搜索最佳F1阈值（当前固定0.5）
3. **模型集成**: 训练多个seed取平均
4. **特征可视化**: t-SNE可视化共享特征空间
5. **注意力图**: CAM/Grad-CAM可视化模型关注区域

## 问题排查

如果遇到问题：
1. **ImportError**: 确保所有 `__init__.py` 文件已创建
2. **CUDA OOM**: 降低 `BATCH_SIZE` 或 `IMG_SIZE`
3. **数据加载慢**: 减少 `NUM_WORKERS` 或设置 `num_workers=0`
4. **NaN loss**: 检查数据标签是否正确（0/1）

---

创建时间: 2026-01-06
基于文件: [model/点刺.py](model/点刺.py)
