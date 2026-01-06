"""
多任务评估指标工具
"""
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix, classification_report,
    precision_recall_fscore_support, roc_auc_score,
    accuracy_score
)
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
import os


class MultiTaskMetrics:
    """多任务评估指标计算"""

    # 舌形标签名称
    SHAPE_LABELS = ['spots', 'cracks', 'teethmarks']

    # 腻腐苔类别名称
    COAT_CLASSES = ['greasy', 'rotten', 'nospecialgreasy']

    def __init__(self):
        self.reset()

    def reset(self):
        """重置所有累积"""
        self.all_y_shape = []
        self.all_y_coat = []
        self.all_pred_shape = []
        self.all_pred_coat = []
        self.all_prob_shape = []
        self.all_prob_coat = []

    def update(self, logits_shape, logits_coat, y_shape, y_coat):
        """
        更新累积的预测和标签

        参数:
            logits_shape: [B, 3]
            logits_coat: [B, 3]
            y_shape: [B, 3]
            y_coat: [B]
        """
        # 舌形概率（sigmoid）
        prob_shape = torch.sigmoid(logits_shape)

        # 舌形预测（阈值0.5）
        pred_shape = (prob_shape >= 0.5).int()

        # 腻腐苔概率（softmax）
        prob_coat = torch.softmax(logits_coat, dim=1)

        # 腋腐苔预测（argmax）
        pred_coat = torch.argmax(prob_coat, dim=1)

        # 累积到CPU
        self.all_y_shape.append(y_shape.cpu())
        self.all_y_coat.append(y_coat.cpu())
        self.all_pred_shape.append(pred_shape.cpu())
        self.all_pred_coat.append(pred_coat.cpu())
        self.all_prob_shape.append(prob_shape.cpu())
        self.all_prob_coat.append(prob_coat.cpu())

    def compute(self):
        """
        计算所有指标

        返回:
            metrics_dict: 包含所有指标的字典
        """
        # 合并所有batch
        y_shape = torch.cat(self.all_y_shape).numpy()
        y_coat = torch.cat(self.all_y_coat).numpy()
        pred_shape = torch.cat(self.all_pred_shape).numpy()
        pred_coat = torch.cat(self.all_pred_coat).numpy()
        prob_shape = torch.cat(self.all_prob_shape).numpy()
        prob_coat = torch.cat(self.all_prob_coat).numpy()

        metrics = {}

        # === 舌形指标（多标签） ===
        metrics['shape'] = self._compute_shape_metrics(y_shape, pred_shape, prob_shape)

        # === 腻腐苔指标（多类） ===
        metrics['coat'] = self._compute_coat_metrics(y_coat, pred_coat, prob_coat)

        # === 综合指标 ===
        metrics['combined'] = self._compute_combined_score(metrics)

        return metrics, (y_shape, y_coat, pred_shape, pred_coat, prob_shape, prob_coat)

    def _compute_shape_metrics(self, y_true, y_pred, y_prob):
        """计算舌形多标签指标"""
        metrics = {}

        for i, label in enumerate(self.SHAPE_LABELS):
            # 每个标签的指标
            true_i = y_true[:, i]
            pred_i = y_pred[:, i]
            prob_i = y_prob[:, i]

            precision, recall, f1, _ = precision_recall_fscore_support(
                true_i, pred_i, average='binary', zero_division=0
            )

            # AUROC（如果正负样本都存在）
            try:
                auroc = roc_auc_score(true_i, prob_i) if len(np.unique(true_i)) > 1 else 0.5
            except:
                auroc = 0.5

            metrics[label] = {
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                'auroc': float(auroc),
                'support': int(true_i.sum())
            }

        # 平均指标
        metrics['macro_f1'] = np.mean([metrics[label]['f1'] for label in self.SHAPE_LABELS])
        metrics['macro_precision'] = np.mean([metrics[label]['precision'] for label in self.SHAPE_LABELS])
        metrics['macro_recall'] = np.mean([metrics[label]['recall'] for label in self.SHAPE_LABELS])

        return metrics

    def _compute_coat_metrics(self, y_true, y_pred, y_prob):
        """计算腻腐苔多类指标"""
        metrics = {}

        # 混淆矩阵
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
        metrics['confusion_matrix'] = cm

        # 各类指标
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=[0, 1, 2], zero_division=0
        )

        for i, class_name in enumerate(self.COAT_CLASSES):
            metrics[class_name] = {
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1': float(f1[i]),
                'support': int(support[i])
            }

        # 宏平均
        metrics['macro_f1'] = float(f1.mean())
        metrics['macro_precision'] = float(precision.mean())
        metrics['macro_recall'] = float(recall.mean())

        # 加权平均
        weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )
        metrics['weighted_f1'] = float(weighted_f1)
        metrics['weighted_precision'] = float(weighted_precision)
        metrics['weighted_recall'] = float(weighted_recall)

        # Balanced accuracy
        metrics['balanced_accuracy'] = float(recall.mean())

        # Overall accuracy
        metrics['accuracy'] = float(accuracy_score(y_true, y_pred))

        return metrics

    def _compute_combined_score(self, metrics):
        """计算综合得分"""
        shape_f1_mean = metrics['shape']['macro_f1']
        coat_f1 = metrics['coat']['macro_f1']

        combined = 0.5 * shape_f1_mean + 0.5 * coat_f1
        return combined

    def print_metrics(self, metrics):
        """打印评估指标"""
        print("\n" + "="*60)
        print("评估指标")
        print("="*60)

        # 舌形
        print("\n【舌形多标签】")
        print("-" * 40)
        for label in self.SHAPE_LABELS:
            m = metrics['shape'][label]
            print(f"{label:12s}: P={m['precision']:.3f}, R={m['recall']:.3f}, "
                  f"F1={m['f1']:.3f}, AUROC={m['auroc']:.3f}, Support={m['support']}")

        print(f"\nMacro Avg:   F1={metrics['shape']['macro_f1']:.3f}, "
              f"P={metrics['shape']['macro_precision']:.3f}, R={metrics['shape']['macro_recall']:.3f}")

        # 腻腐苔
        print("\n【腻腐苔多类】")
        print("-" * 40)
        for class_name in self.COAT_CLASSES:
            m = metrics['coat'][class_name]
            print(f"{class_name:15s}: P={m['precision']:.3f}, R={m['recall']:.3f}, "
                  f"F1={m['f1']:.3f}, Support={m['support']}")

        print(f"\nMacro F1:       {metrics['coat']['macro_f1']:.3f}")
        print(f"Macro Precision: {metrics['coat']['macro_precision']:.3f}")
        print(f"Macro Recall:    {metrics['coat']['macro_recall']:.3f}")
        print(f"Balanced Acc:    {metrics['coat']['balanced_accuracy']:.3f}")
        print(f"Accuracy:        {metrics['coat']['accuracy']:.3f}")

        # 综合
        print("\n【综合得分】")
        print("-" * 40)
        print(f"Combined Score: {metrics['combined']:.3f} (0.5*shape_f1 + 0.5*coat_f1)")

        print("="*60 + "\n")


def save_confusion_matrix(cm, class_names, save_path, title="Confusion Matrix"):
    """保存混淆矩阵图"""
    fig, ax = plt.subplots(figsize=(8, 6))

    if HAS_SEABORN:
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names,
                    ax=ax)
    else:
        # 使用matplotlib绘制混淆矩阵
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)

        # 设置刻度
        tick_marks = np.arange(len(class_names))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names)

        # 添加数值标注
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")

    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    ax.set_title(title)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_predictions_csv(file_names, y_shape, y_coat, pred_shape, pred_coat,
                         prob_shape, prob_coat, save_path):
    """
    保存预测结果到CSV

    参数:
        file_names: list of str
        y_shape: [N, 3] numpy array
        y_coat: [N] numpy array
        pred_shape: [N, 3] numpy array
        pred_coat: [N] numpy array
        prob_shape: [N, 3] numpy array
        prob_coat: [N, 3] numpy array
        save_path: str
    """
    data = []

    for i in range(len(file_names)):
        row = {
            'file_name': file_names[i],
            # GT
            'GT_spots': int(y_shape[i, 0]),
            'GT_cracks': int(y_shape[i, 1]),
            'GT_teethmarks': int(y_shape[i, 2]),
            'GT_coat_class': int(y_coat[i]),
            # Shape预测
            'pred_spots': int(pred_shape[i, 0]),
            'pred_cracks': int(pred_shape[i, 1]),
            'pred_teethmarks': int(pred_shape[i, 2]),
            'prob_spots': float(prob_shape[i, 0]),
            'prob_cracks': float(prob_shape[i, 1]),
            'prob_teethmarks': float(prob_shape[i, 2]),
            # Coat预测
            'pred_coat_class': int(pred_coat[i]),
            'prob_greasy': float(prob_coat[i, 0]),
            'prob_rotten': float(prob_coat[i, 1]),
            'prob_nospecialgreasy': float(prob_coat[i, 2]),
        }
        data.append(row)

    df = pd.DataFrame(data)
    df.to_csv(save_path, index=False)
    print(f"已保存预测结果: {save_path}")


def compute_class_weights(df):
    """
    从DataFrame计算腻腐苔三类权重

    返回:
        class_weights: list of float (用于CrossEntropyLoss)
    """
    coat_labels = ['Tonguecoat_greasy', 'Tonguecoat_rotten', 'Tonguecoat_nospecialgreasy']
    counts = []

    for label in coat_labels:
        count = df[label].sum()
        counts.append(max(count, 1))  # 避免除0

    # Inverse frequency
    total = sum(counts)
    n_classes = len(counts)
    weights = [total / (n_classes * c) for c in counts]

    # 给rotten更高权重
    weights[1] *= 2.0

    # 归一化
    weights = [w / sum(weights) * n_classes for w in weights]

    return weights


def search_best_thresholds(y_true, y_prob, metric='f1'):
    """
    为多标签搜索最佳阈值

    参数:
        y_true: [N, 3]
        y_prob: [N, 3]
        metric: 'f1' or 'balanced_accuracy'

    返回:
        best_thresholds: list of 3 floats
    """
    from sklearn.metrics import f1_score, balanced_accuracy_score

    best_thresholds = []
    n_labels = y_true.shape[1]

    for i in range(n_labels):
        best_score = 0
        best_thresh = 0.5

        for thresh in np.arange(0.1, 0.9, 0.05):
            pred = (y_prob[:, i] >= thresh).astype(int)

            if metric == 'f1':
                score = f1_score(y_true[:, i], pred, zero_division=0)
            else:
                score = balanced_accuracy_score(y_true[:, i], pred)

            if score > best_score:
                best_score = score
                best_thresh = thresh

        best_thresholds.append(best_thresh)

    return best_thresholds


if __name__ == "__main__":
    # 测试代码
    print("测试评估指标...")

    # 模拟数据
    n_samples = 100
    n_batches = 5

    metrics_calculator = MultiTaskMetrics()

    for _ in range(n_batches):
        batch_size = n_samples // n_batches

        # 模拟logits
        logits_shape = torch.randn(batch_size, 3)
        logits_coat = torch.randn(batch_size, 3)

        # 模拟标签
        y_shape = torch.randint(0, 2, (batch_size, 3)).float()
        y_coat = torch.randint(0, 3, (batch_size,))

        metrics_calculator.update(logits_shape, logits_coat, y_shape, y_coat)

    # 计算指标
    metrics, _ = metrics_calculator.compute()
    metrics_calculator.print_metrics(metrics)

    # 测试混淆矩阵保存
    cm = metrics['coat']['confusion_matrix']
    os.makedirs('test_outputs', exist_ok=True)
    save_confusion_matrix(cm, MultiTaskMetrics.COAT_CLASSES,
                          'test_outputs/confusion_matrix.png')
    print("已保存测试混淆矩阵")

    print("\n测试通过！")
