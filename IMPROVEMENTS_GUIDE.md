# 多任务训练改进版 - 使用指南

## 已完成的改进

### 1. ✅ Stratified Split（按腻腐苔类别分层）
- **文件**: [datasets/tongue_multitask_dataset.py](datasets/tongue_multitask_dataset.py)
- **函数**: `split_train_val()` 现在支持 `stratify_by_coat=True`
- **效果**: 确保验证集 rotten 类别有足够样本（至少5个）

### 2. ✅ 阈值搜索（Threshold Search）
- **文件**: [utils/metrics.py](utils/metrics.py)
- **新增函数**:
  - `search_best_thresholds()`: 网格搜索最优F1阈值
  - `compute_metrics_with_thresholds()`: 使用指定阈值计算指标
- **参数**:
  - `search_range=(0.05, 0.95)`: 搜索范围
  - `step=0.01`: 步长（可调为0.05更快）

### 3. ✅ Focal Loss（Rotten 强化）
- **文件**: [models/multitask_convnext.py](models/multitask_convnext.py)
- **参数**:
  - `use_focal=True`: 启用 Focal Loss
  - `focal_gamma=2.0`: Focal 系数
  - `focal_alpha`: 可手动设置或自动计算（rotten权重×2）
- **回退选项**: 仍支持传统的加权 CE（带权重上限保护）

### 4. ✅ 采样统计日志
- **文件**: [train_multitask.py](train_multitask.py)
- **函数**: `train_one_epoch()` 现在记录每个epoch的采样统计
- **输出**: `greasy_count`, `rotten_count`, `nospecial_count`

### 5. ✅ 增强配置选项
- **文件**: [train_multitask.py](train_multitask.py) 的 `CONFIG` 类
- **新增配置**:
  ```python
  # 数据划分
  STRATIFIED_SPLIT = True  # 启用分层抽样

  # 阈值搜索
  SEARCH_BEST_THRESHOLDS = True
  THRESHOLD_SEARCH_STEP = 0.01
  THRESHOLD_SEARCH_RANGE = (0.05, 0.95)

  # Rotten强化
  USE_FOCAL_LOSS = True
  FOCAL_GAMMA = 2.0
  FOCAL_ALPHA = None  # None=自动计算

  # 实验配置
  EXPERIMENT_NAME = "improved"
  LOG_SAMPLING_STATS = True
  ```

## 运行方式

### 默认配置（所有改进启用）
```bash
conda activate tongue
cd /home/wuyongxi/tongue/planner
python train_multitask.py
```

### 自定义配置
编辑 [train_multitask.py](train_multitask.py) 顶部的 `CONFIG` 类：

```python
class CONFIG:
    # ... 基本配置 ...

    # ========== 改进选项 ==========
    STRATIFIED_SPLIT = True       # 分层抽样
    SEARCH_BEST_THRESHOLDS = True # 阈值搜索
    USE_FOCAL_LOSS = True         # Focal Loss

    # 调整 Focal Loss 参数
    FOCAL_GAMMA = 2.0
    FOCAL_ALPHA = [0.5, 3.0, 0.5]  # 手动设置 [greasy, rotten, nospecial]

    # 或使用传统加权 CE
    USE_FOCAL_LOSS = False
    MAX_CLASS_WEIGHT = 30  # 权重上限
```

## 输出文件

训练完成后，在 `outputs/<timestamp>_multitask/` 下会生成：

### 报告文件 (reports/)
1. **split_stats.json** - 数据划分统计
   - 训练/验证样本数
   - 验证集各类别数量（greasy/rotten/nospecial）
   - 是否使用了 stratified split

2. **loss_config.json** - 损失函数配置
   - 类型（focal_loss 或 weighted_ce）
   - 参数（gamma, alpha, class_weights）

3. **thresholds.json** - 最优阈值（如果启用搜索）
   - 每个标签的最佳阈值
   - 对应的F1得分

4. **val_metrics_thresholded.json** - 使用最优阈值的指标

5. **epoch_sampling_log.csv** - 每个epoch的采样统计
   - epoch, greasy_count, rotten_count, nospecial_count, total

6. **val_predictions.csv** - 验证集预测详情
   - 包含概率、预测、GT

7. **conflicts.csv** - 逻辑冲突样本
8. **missing_images.csv** - 缺失图像
9. **summary.json** - 训练摘要

### 图表 (figures/)
1. **training_history.png** - 训练曲线
2. **confusion_matrix.png** - 腻腐苔混淆矩阵

### 模型 (checkpoints/)
1. **best.pt** - 最佳模型（基于 combined_score）
2. **last.pt** - 最后一轮的checkpoint
3. **checkpoint_epoch_*.pt** - 定期保存的checkpoint

## 实验矩阵

建议按以下顺序运行实验对比：

### E0: Baseline（基线）
```python
STRATIFIED_SPLIT = False
SEARCH_BEST_THRESHOLDS = False
USE_FOCAL_LOSS = False
EXPERIMENT_NAME = "E0_baseline"
```

### E1: + 阈值调优
```python
STRATIFIED_SPLIT = False
SEARCH_BEST_THRESHOLDS = True
USE_FOCAL_LOSS = False
EXPERIMENT_NAME = "E1_threshold"
```

### E2: + Rotten强化
```python
STRATIFIED_SPLIT = True
SEARCH_BEST_THRESHOLDS = True
USE_FOCAL_LOSS = True
EXPERIMENT_NAME = "E2_full_improved"
```

### E3: + 双阶段训练
```python
# 保持 E2 配置，调整训练参数
STAGE1_EPOCHS = 10  # 增加Stage 1
STAGE2_EPOCHS = 50  # 增加Stage 2
EXPERIMENT_NAME = "E3_longer_training"
```

## 指标解读

### 输出指标说明
训练过程会输出：

```
【舌形多标签】
spots        : P=0.723, R=0.681, F1=0.701, AUROC=0.812
cracks       : P=0.545, R=0.512, F1=0.528, AUROC=0.721
teethmarks   : P=0.812, R=0.834, F1=0.823, AUROC=0.901

Macro Avg:   F1=0.684, P=0.693, R=0.676

【腻腐苔多类】
greasy       : P=0.712, R=0.734, F1=0.723, Support=63
rotten       : P=0.200, R=0.100, F1=0.133, Support=10  ← 关注这个！
nospecialgreasy: P=0.823, R=0.856, F1=0.839, Support=180

Macro F1:       0.565
Balanced Acc:   0.563
Accuracy:        0.783

【综合得分】
Combined Score (fixed):    0.624 (使用0.5阈值)
Combined Score (threshold): 0.651 (使用最优阈值)  ← 主要看这个！
```

### 关键指标
- **spots F1**: 目标提升到 0.5+
- **rotten Recall**: 目标 > 0（不再是0）
- **coat_macro_f1**: 目标不下降
- **combined_score**: 主要优化目标

## 故障排查

### 问题1: Rotten recall 仍为 0
**原因**: 验证集 rotten 样本太少（<5）
**解决**:
1. 检查 `reports/split_stats.json` 确认验证集 rotten 数量
2. 如果确实太少，考虑：
   - 使用 k-fold 交叉验证
   - 减少 val_ratio（如 0.15）
   - 收集更多 rotten 样本

### 问题2: 阈值搜索很慢
**解决**:
- 增大 `THRESHOLD_SEARCH_STEP` 到 0.05
- 或缩小 `THRESHOLD_SEARCH_RANGE` 到 (0.2, 0.8)

### 问题3: Focal Loss 导致过拟合
**解决**:
- 降低 `FOCAL_GAMMA` 到 1.5 或 1.0
- 减小 `FOCAL_ALPHA` 中的 rotten 权重
- 或回退到加权 CE

### 问题4: 采样统计显示 rotten 很少
**检查**:
1. 查看 `reports/epoch_sampling_log.csv`
2. 如果每个epoch rotten < 50 次，说明采样权重不够
3. 增加 rotten 权重（在 FOCAL_ALPHA 或 class_weights 中）

## 下一步优化

1. **K-Fold 交叉验证**: 当 rotten 样本极少时（<30）
2. **测试时增强(TTA)**: 提升验证集指标稳定性
3. **阈值搜索方法**: 当前基于F1，可尝试基于 Matthews Correlation Coefficient
4. **后处理规则**: 基于逻辑一致性校准预测结果

## 对比分析

训练完成后，可以使用以下命令对比不同实验：

```bash
# 提取关键指标
cd /home/wuyongxi/tongue/planner/outputs

# 查看各实验的 summary
grep -h "combined_score" */reports/summary.json

# 对比 rotten recall
grep -h "rotten" */reports/val_metrics_thresholded.json
```

或创建对比脚本（见 `compare_experiments.py`，待实现）。

---

**创建时间**: 2026-01-06
**改进版本**: v2.0
**基于**: [train_multitask.py](train_multitask.py)
