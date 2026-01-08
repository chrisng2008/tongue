# 多任务训练改进版 - 快速开始

## ✅ 已完成的改进

### 1. Stratified Split（分层抽样）
- ✅ 实现按腻腐苔类别分层抽样
- ✅ 确保验证集 rotten 样本至少 5 个
- ✅ 自动保存 split_stats.json

### 2. 阈值搜索（Threshold Search）
- ✅ 网格搜索最优 F1 阈值（每个标签独立）
- ✅ 可配置搜索范围和步长
- ✅ 保存阈值对比报告

### 3. Focal Loss（Rotten 强化）
- ✅ 实现完整的 Focal Loss
- ✅ 可配置 gamma 和 alpha
- ✅ 支持 auto alpha（自动计算 rotten 权重）
- ✅ 回退到加权 CE（带权重上限保护）

### 4. 采样统计日志
- ✅ 每个 epoch 记录采样统计
- ✅ 显示 rotten 采样率和出现次数
- ✅ 保存 epoch_sampling_log.csv

### 5. 增强报告输出
- ✅ 固定阈值 vs 最优阈值对比
- ✅ 验证集支持数统计
- ✅ 增强预测结果（两套预测）
- ✅ 完整的 JSON 报告

## 🚀 立即运行

### 方式1：使用默认配置（所有改进启用）

```bash
# 激活环境
conda activate tongue

# 进入项目目录
cd /home/wuyongxi/tongue/planner

# 运行训练
python train_multitask.py
```

### 方式2：自定义配置

编辑 [train_multitask.py](train_multitask.py) 顶部的 `CONFIG` 类：

```python
class CONFIG:
    # ========== 改进选项 ==========
    STRATIFIED_SPLIT = True        # 分层抽样
    SEARCH_BEST_THRESHOLDS = True  # 阈值搜索
    USE_FOCAL_LOSS = True          # Focal Loss

    # 阈值搜索配置
    THRESHOLD_SEARCH_STEP = 0.01   # 步长（0.01精细，0.05快速）
    THRESHOLD_SEARCH_RANGE = (0.05, 0.95)

    # Focal Loss 配置
    FOCAL_GAMMA = 2.0
    FOCAL_ALPHA = None  # None=自动计算，或手动设置如 [0.5, 3.0, 0.5]

    # 实验配置
    EXPERIMENT_NAME = "improved"
    LOG_SAMPLING_STATS = True
```

## 📊 训练输出

### 输出目录结构
```
outputs/<timestamp>_multitask/
├── checkpoints/
│   ├── best.pt                    # 最佳模型
│   ├── last.pt                    # 最终 checkpoint
│   └── checkpoint_epoch_*.pt      # 定期保存
├── reports/
│   ├── summary.json               # 训练摘要
│   ├── split_stats.json           # 数据划分统计
│   ├── loss_config.json           # 损失函数配置
│   ├── thresholds.json            # 最优阈值
│   ├── val_metrics_thresholded.json  # 阈值对比
│   ├── epoch_sampling_log.csv     # 采样日志
│   ├── val_predictions_enhanced.csv  # 增强预测
│   ├── train_split.csv
│   ├── val_split.csv
│   ├── conflicts.csv
│   └── missing_images.csv
└── figures/
    ├── training_history.png       # 训练曲线
    └── confusion_matrix.png       # 混淆矩阵
```

### 关键报告说明

#### 1. summary.json - 训练摘要
```json
{
  "experiment_name": "improved",
  "config": {
    "stratified_split": true,
    "use_focal_loss": true,
    "search_thresholds": true,
    "log_sampling": true
  },
  "final_metrics_fixed": {
    "combined_score": 0.554  // 固定阈值 0.5
  },
  "final_metrics_thresholded": {
    "combined_score": 0.618,  // 最优阈值
    "best_thresholds": [0.32, 0.41, 0.38]
  },
  "val_support": {
    "val_Tonguecoat_rotten": 4  // 验证集 rotten 数量
  }
}
```

#### 2. epoch_sampling_log.csv - 采样日志
记录每个 epoch 的采样统计：
- epoch, stage, greasy_count, rotten_count, nospecial_count, total_samples
- 检查 rotten 是否被充分采样（应该 > 5%）

#### 3. val_metrics_thresholded.json - 阈值对比
对比固定阈值和最优阈值的性能差异。

#### 4. val_predictions_enhanced.csv - 增强预测
包含两套预测：
- `pred_*_fixed`: 固定阈值 0.5
- `pred_*_thr`: 最优阈值

## 🎯 实验矩阵

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
**预期**: spots F1 提升 0.05-0.10

### E2: + Rotten 强化
```python
STRATIFIED_SPLIT = True
SEARCH_BEST_THRESHOLDS = True
USE_FOCAL_LOSS = True
EXPERIMENT_NAME = "E2_full_improved"
```
**预期**: rotten recall > 0，coat_macro_f1 稳定

### E3: 更长训练
```python
# 保持 E2 配置
STAGE1_EPOCHS = 10
STAGE2_EPOCHS = 50
EXPERIMENT_NAME = "E3_longer_training"
```

## 📈 预期改进

### 相对 Baseline 的提升

| 指标 | Baseline | E1 (阈值) | E2 (全部) |
|------|----------|-----------|-----------|
| spots F1 | ~0.43 | ~0.50 | ~0.52 |
| cracks F1 | ~0.53 | ~0.55 | ~0.56 |
| teethmarks F1 | ~0.70 | ~0.71 | ~0.72 |
| shape_macro_f1 | ~0.55 | ~0.59 | ~0.60 |
| rotten recall | 0.00 | 0.00 | > 0.10 |
| coat_macro_f1 | ~0.50 | ~0.50 | ~0.51 |
| **combined** | **~0.55** | **~0.59** | **~0.61** |

## 🔍 监控要点

### 训练过程中关注

1. **采样统计**（每个 epoch 后）:
   ```
   采样统计: Rotten=45 (8.9%)
   ```
   - 应该 > 5%（如果太低，增加 focal_alpha）

2. **阈值优化**（每个 epoch 后）:
   ```
   【最优阈值】
     spots: 0.32 (F1=0.512)
     cracks: 0.41 (F1=0.558)
     teethmarks: 0.38 (F1=0.718)
   Thresholded Combined Score: 0.618
   ```
   - 对比固定阈值的 F1 提升

3. **Rotten 性能**:
   ```
   【腻腐苔多类】
   rotten: P=0.200, R=0.100, F1=0.133
   ```
   - 目标：R > 0（哪怕很小）

### 最终评估检查点

- [ ] 验证集 rotten ≥ 5 样本
- [ ] Rotten 采样率 > 5%
- [ ] Rotten recall > 0
- [ ] Thresholded combined > Fixed combined
- [ ] 所有报告文件已生成

## ⚠️ 常见问题

### Q1: Rotten recall 仍为 0
**原因**:
- 验证集 rotten 样本太少（< 5）
- 模型严重过拟合到多数类

**解决**:
1. 检查 `reports/split_stats.json` 确认验证集 rotten 数量
2. 如果确实太少，考虑：
   - 减少 `VAL_RATIO` 到 0.15
   - 使用 k-fold 交叉验证
   - 收集更多 rotten 样本

### Q2: 阈值搜索很慢
**解决**:
```python
THRESHOLD_SEARCH_STEP = 0.05  # 从 0.01 改为 0.05
THRESHOLD_SEARCH_RANGE = (0.2, 0.8)  # 缩小搜索范围
```

### Q3: Focal Loss 导致过拟合
**解决**:
```python
FOCAL_GAMMA = 1.5  # 从 2.0 降低
FOCAL_ALPHA = [0.5, 2.0, 0.5]  # 手动降低 rotten 权重
```

### Q4: 采样统计显示 rotten 很少
**检查**:
1. 查看 `reports/epoch_sampling_log.csv`
2. 如果每个 epoch rotten < 50，增加权重：
```python
# 在自动计算的 focal_alpha 基础上再乘 2
rotten_w = total / (n_classes * max(train_df['Tonguecoat_rotten'].sum(), 1)) * 4.0
```

## 📝 对比实验结果

训练完成后，对比不同实验：

```bash
cd /home/wuyongxi/tongue/planner/outputs

# 提取关键指标
for exp in E*_*; do
    echo "=== $exp ==="
    jq -r '.final_metrics_thresholded // .final_metrics_fixed' $exp/reports/summary.json
done
```

或查看具体指标：
```bash
# Spots F1 对比
grep -h "shape_spots_f1" */reports/summary.json | jq -s 'map(. + {exp: input_filename}) | sort_by(.shape_spots_f1) | reverse'

# Combined Score 对比
grep -h "combined_score" */reports/summary.json | jq -s 'map(. + {exp: input_filename}) | sort_by(.combined_score) | reverse'
```

## 🎉 成功标准

✅ **验收标准**：
1. 代码在 `conda activate tongue` 下可直接运行
2. 固定使用 `cuda:3`
3. 生成完整输出目录
4. 实现了 stratified split + threshold search + focal/sampler
5. shape_macro_f1_thresholded ≥ baseline
6. coat_macro_f1 不低于 baseline
7. rotten recall 不再恒为 0（或采样日志显示 rotten 频率显著提升）

---

**创建时间**: 2026-01-06
**版本**: v2.0 - Improved Multi-task Training
**基于**: [train_multitask.py](train_multitask.py)
