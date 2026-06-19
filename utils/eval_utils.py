# ============================================================
# utils/eval_utils.py
# Metrics, confusion matrix, CSV saving, training curves.
# ============================================================

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score,
                             classification_report, confusion_matrix)


def compute_metrics(y_true, y_pred, class_names):
    labels = list(range(len(class_names)))
    acc         = accuracy_score(y_true, y_pred)
    prec_macro  = precision_score(y_true, y_pred, labels=labels, average='macro', zero_division=0)
    rec_macro   = recall_score(y_true, y_pred, labels=labels, average='macro', zero_division=0)
    f1_macro    = f1_score(y_true, y_pred, labels=labels, average='macro', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, labels=labels, average='weighted', zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    per_class_acc = {}
    for i, name in enumerate(class_names):
        row = cm[i].sum()
        per_class_acc[name] = float(cm[i, i] / row) if row > 0 else 0.0

    report = classification_report(
        y_true, y_pred, labels=labels, target_names=class_names,
        zero_division=0, output_dict=True)

    return {
        'accuracy':           round(acc, 4),
        'precision_macro':    round(prec_macro, 4),
        'recall_macro':       round(rec_macro, 4),
        'f1_macro':           round(f1_macro, 4),
        'f1_weighted':        round(f1_weighted, 4),
        'per_class_accuracy': per_class_acc,
        'classification_report': report,
    }


def save_metrics(metrics, save_dir, tag='eval'):
    os.makedirs(save_dir, exist_ok=True)
    json_path = os.path.join(save_dir, f'{tag}_metrics.json')
    with open(json_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    flat = {k: v for k, v in metrics.items() if not isinstance(v, dict)}
    pd.DataFrame([flat]).to_csv(
        os.path.join(save_dir, f'{tag}_summary.csv'), index=False)
    print(f"  Metrics → {json_path}")


def save_predictions_csv(records, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    pd.DataFrame(records).to_csv(save_path, index=False)
    print(f"  Predictions CSV → {save_path}")


def save_confusion_matrix(y_true, y_pred, class_names,
                           save_path, title='Confusion Matrix'):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    labels  = list(range(len(class_names)))
    cm      = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = np.divide(
        cm.astype(float),
        cm.sum(axis=1, keepdims=True),
        out=np.zeros_like(cm, dtype=float),
        where=cm.sum(axis=1, keepdims=True) != 0,
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[0])
    axes[0].set_title(f'{title} (counts)')
    axes[0].set_ylabel('True'); axes[0].set_xlabel('Predicted')

    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[1])
    axes[1].set_title(f'{title} (normalised)')
    axes[1].set_ylabel('True'); axes[1].set_xlabel('Predicted')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Confusion matrix → {save_path}")


def save_training_curves(history, save_dir, tag='train'):
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, history['train_loss'], label='Train')
    axes[0].plot(epochs, history['val_loss'],   label='Val')
    axes[0].set_title('Loss'); axes[0].set_xlabel('Epoch')
    axes[0].legend()
    axes[1].plot(epochs, history['train_acc'], label='Train')
    axes[1].plot(epochs, history['val_acc'],   label='Val')
    axes[1].set_title('Accuracy'); axes[1].set_xlabel('Epoch')
    axes[1].legend()
    plt.tight_layout()
    path = os.path.join(save_dir, f'{tag}_curves.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Training curves → {path}")


def save_comparison_plot(before_metrics, after_metrics,
                          class_names, save_path):
    """
    Bar chart comparing per-class accuracy before vs after
    robot fine-tuning. Used in the report notebook.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    before = [before_metrics['per_class_accuracy'].get(c, 0)
              for c in class_names]
    after  = [after_metrics['per_class_accuracy'].get(c, 0)
              for c in class_names]

    x     = np.arange(len(class_names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(10, len(class_names) * 0.8), 5))
    ax.bar(x - width/2, before, width, label='Before robot fine-tune',
           color='steelblue', alpha=0.85)
    ax.bar(x + width/2, after,  width, label='After robot fine-tune',
           color='darkorange', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Per-class accuracy')
    ax.set_title('Before vs After robot fine-tuning — per-class accuracy')
    ax.legend()
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Comparison plot → {save_path}")
