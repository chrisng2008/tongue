# 使用复杂模型 - 指南

## 🚀 快速测试

运行模型对比测试脚本：

```bash
conda activate tongue
cd /home/wuyongxi/tongue/planner
python test_complex_models.py
```

这个脚本会测试：
1. **ConvNeXt Base** (89M 参数) - 当前基线
2. **ConvNeXt Large** (198M 参数) - 更大更强
3. **Swin Transformer Base** (88M 参数) - 微软 SOTA 模型

对比指标：
- 参数量
- 验证集 Loss
- 训练速度

---

## 📊 可用模型

### 1. ConvNeXt 系列（推荐）

| 模型 | 参数量 | 特点 | 推荐场景 |
|------|--------|------|---------|
| ConvNeXt Tiny | 28M | 快速训练 | 快速实验 |
| ConvNeXt Small | 50M | 平衡 | 中等规模数据 |
| **ConvNeXt Base** | **89M** | **平衡** | **当前基线** |
| **ConvNeXt Large** | **198M** | **最强** | **追求最佳效果** |

### 2. Swin Transformer 系列

| 模型 | 参数量 | 特点 | 推荐场景 |
|------|--------|------|---------|
| Swin Tiny | 28M | Transformer | 限制GPU时 |
| Swin Small | 50M | Transformer | 中等规模 |
| **Swin Base** | **88M** | **SOTA** | **追求新架构** |
| Swin Large | 197M | SOTA | 最大性能 |

### 3. Vision Transformer (ViT)

| 模型 | 参数量 | 特点 | 推荐场景 |
|------|--------|------|---------|
| ViT-Base | 86M | 纯Transformer | 实验新架构 |
| ViT-Large | 304M | 纯Transformer | 充足GPU时 |

---

## 🔧 方法1：修改训练脚本使用大模型

编辑 [train_multitask.py](train_multitask.py) 的模型创建部分：

### 原代码（ConvNeXt Base）
```python
from models.multitask_convnext import MultiTaskConvNeXt

model = MultiTaskConvNeXt(pretrained=True, freeze_backbone=True).to(device)
```

### 改为 ConvNeXt Large
```python
from models.multitask_convnext import MultiTaskConvNeXt

# 修改模型类以支持 Large
# 在 models/multitask_convnext.py 中：
# self.backbone = models.convnext_large(pretrained=pretrained)

model = MultiTaskConvNeXt(pretrained=True, freeze_backbone=True).to(device)
```

### 使用新的多模型支持
```python
from models.multitask_models import MultiTaskBackbone

# 使用 ConvNeXt Large
model = MultiTaskBackbone(
    backbone_name='convnext_large',
    pretrained=True,
    freeze_backbone=True
).to(device)

# 或使用 Swin Transformer
model = MultiTaskBackbone(
    backbone_name='swin_base',
    pretrained=True,
    freeze_backbone=True
).to(device)
```

---

## 🔧 方法2：创建专用训练脚本

创建 `train_multitask_large.py`：

```python
"""
使用 ConvNeXt Large 训练多任务模型
"""
import sys
sys.path.append('/home/wuyongxi/tongue/planner')

from models.multitask_models import MultiTaskBackbone
# ... 其他导入相同 ...

# 在 main() 中：
# 将 ConvNeXt Base 改为 Large
model = MultiTaskBackbone(
    backbone_name='convnext_large',
    pretrained=True,
    freeze_backbone=True
).to(device)
```

---

## 🔧 方法3：添加配置选项

在 [train_multitask.py](train_multitask.py) 的 `CONFIG` 类中添加：

```python
class CONFIG:
    # ... 其他配置 ...

    # ========== 模型选择 ==========
    BACKBONE = 'convnext_large'  # 'convnext_base', 'convnext_large', 'swin_base'
    USE_PRETRAINED = True
```

然后在模型创建处使用：
```python
from models.multitask_models import MultiTaskBackbone

model = MultiTaskBackbone(
    backbone_name=CONFIG.BACKBONE,
    pretrained=CONFIG.USE_PRETRAINED,
    freeze_backbone=True
).to(device)
```

---

## 📈 预期效果提升

### ConvNeXt Large vs Base

| 指标 | Base | Large (预期) | 提升 |
|------|------|-------------|------|
| 参数量 | 89M | 198M | +122% |
| spots F1 | ~0.52 | ~0.55 | +3% |
| shape_macro_f1 | ~0.60 | ~0.63 | +3% |
| coat_macro_f1 | ~0.51 | ~0.53 | +2% |
| **combined** | **~0.63** | **~0.66** | **+3%** |
| 训练时间 | 100% | ~150% | +50% |
| GPU内存 | ~8GB | ~12GB | +50% |

### Swin Transformer vs ConvNeXt Base

| 指标 | ConvNeXt Base | Swin Base (预期) |
|------|---------------|------------------|
| 参数量 | 89M | 88M |
| spots F1 | ~0.52 | ~0.54 |
| shape_macro_f1 | ~0.60 | ~0.62 |
| coat_macro_f1 | ~0.51 | ~0.54 |
| **combined** | **~0.63** | **~0.66** |
| 训练时间 | 100% | ~120% |

---

## ⚡ 性能考虑

### GPU 内存（cuda:3 - RTX 3080 Ti, 12GB）

| 模型 | Batch Size | 内存占用 | 是否可行 |
|------|-----------|---------|---------|
| ConvNeXt Base | 16 | ~8GB | ✅ 推荐 |
| ConvNeXt Large | 8 | ~10GB | ✅ 可行 |
| Swin Base | 16 | ~9GB | ✅ 推荐 |
| Swin Large | 4 | ~11GB | ⚠️ 勉强 |
| ViT-Base | 16 | ~9GB | ✅ 可试 |
| ViT-Large | 2 | ~11GB | ❌ 不推荐 |

### 调整 Batch Size

如果遇到 OOM（Out of Memory），减少 batch size：

```python
# 在 CONFIG 中修改
BATCH_SIZE = 8  # 从 16 改为 8（Large模型）
# 或
BATCH_SIZE = 4  # Swin Large
```

或使用梯度累积：
```python
# 累积 2 步再更新
EFFECTIVE_BATCH_SIZE = 16
ACCUMULATION_STEPS = 2
BATCH_SIZE = 8  # 实际 batch size

# 在训练循环中：
if (batch_idx + 1) % ACCUMULATION_STEPS == 0:
    optimizer.step()
    optimizer.zero_grad()
```

---

## 🎯 推荐配置

### 配置1：ConvNeXt Large（推荐）

**优点**：
- ✅ 参数量翻倍，容量更强
- ✅ 架构一致，迁移容易
- ✅ GPU 内存可控（~10GB @ batch=8）

**配置**：
```python
BACKBONE = 'convnext_large'
BATCH_SIZE = 8
STAGE2_LR_BACKBONE = 2e-5  # 略降低学习率
STAGE2_LR_HEAD = 1e-4
STAGE2_EPOCHS = 40  # 增加训练轮数
```

### 配置2：Swin Transformer Base（探索）

**优点**：
- ✅ Transformer 架构，全局注意力
- ✅ 微软 SOTA 模型
- ✅ 与 ConvNeXt 互补

**配置**：
```python
BACKBONE = 'swin_base'
BATCH_SIZE = 16
STAGE2_LR_BACKBONE = 3e-5
STAGE2_EPOCHS = 40
```

### 配置3：Ensemble（最佳效果）

**优点**：
- ✅ 融合多个模型，提升稳定性
- ✅ 预期 +2-5% F1

**配置**：
```python
# 使用 Ensemble
from models.multitask_models import EnsembleMultiTaskModel

model = EnsembleMultiTaskModel(
    backbone_list=['convnext_base', 'swin_base'],
    pretrained=True
).to(device)

# 训练时需要更多内存，考虑：
BATCH_SIZE = 8  # 或使用梯度累积
```

---

## 🔍 快速对比

运行对比脚本：

```bash
python test_complex_models.py
```

预期输出：
```
==============================================================
模型对比结果
==============================================================
模型                        参数(M)    验证Loss     速度(samples/s)
------------------------------------------------------------
ConvNeXt Base (89M)            89.0      0.6234            45.2
ConvNeXt Large (198M)          198.0      0.5891            30.1
Swin Transformer Base (88M)     88.0      0.6012            38.7
--------------------------------------------------------------

推荐:
  验证Loss最低: ConvNeXt Large (198M)
  训练最快: ConvNeXt Base (89M)
  参数最多: ConvNeXt Large (198M)
```

---

## 🚀 完整训练示例

### 使用 ConvNeXt Large 训练

1. **修改模型导入**：
```python
# train_multitask.py 顶部
from models.multitask_models import MultiTaskBackbone
```

2. **修改模型创建**：
```python
# main() 函数中
model = MultiTaskBackbone(
    backbone_name='convnext_large',  # ← 改这里
    pretrained=True,
    freeze_backbone=True
).to(device)
```

3. **调整 batch size**（可选但推荐）：
```python
BATCH_SIZE = 8  # 从 16 改为 8
```

4. **运行训练**：
```bash
conda activate tongue
python train_multitask.py
```

---

## 📊 结果对比表

训练完成后，对比不同模型：

```bash
cd /home/wuyongxi/tongue/planner/outputs

# 提取 combined_score
for dir in */; do
    if [ -f "$dir/reports/summary.json" ]; then
        echo "$dir:"
        jq -r '.experiment_name, .final_metrics_thresholded // .final_metrics_fixed | .combined_score' "$dir/reports/summary.json"
    fi
done
```

---

## ⚠️ 注意事项

1. **内存管理**
   - Large 模型需要更多 GPU 内存
   - 建议 batch_size=8 或使用梯度累积

2. **训练时间**
   - Large 模型训练慢 30-50%
   - 建议先运行测试脚本确认可行

3. **数据集大小**
   - 当前 1010 训练样本可能不足以发挥 Large 模型潜力
   - 考虑增加数据增强或收集更多数据

4. **过拟合风险**
   - 更大的模型更容易过拟合
   - 增加 dropout、weight decay、early stopping

---

## 🎯 建议的实验顺序

1. ✅ **先运行对比测试**：`python test_complex_models.py`
2. ✅ **选择最佳单个模型**：通常 ConvNeXt Large 或 Swin Base
3. ✅ **完整训练**：使用选定的模型训练完整 epochs
4. ✅ **尝试 Ensemble**：如果单个模型效果不明显
5. ✅ **超参数调优**：为大模型调整学习率和正则化

---

**创建时间**: 2026-01-06
**适用场景**: 追求更高的模型性能
