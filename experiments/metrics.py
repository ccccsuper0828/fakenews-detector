"""
Unified metrics module for all experiment groups.

Provides consistent evaluation across RNN, LLM, multi-agent,
and KG+RL experiments.
"""

import time
import json
import os
import csv
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Any

import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

matplotlib.rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False


@dataclass
class EvalResult:
    """Standardized evaluation result across all experiment groups."""
    experiment_name: str
    group: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    auc_roc: float
    confusion_matrix: List[List[int]]
    param_count: int = 0
    inference_time_ms: float = 0.0
    extra: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if d["extra"] is None:
            d["extra"] = {}
        return d

    def summary(self) -> str:
        return (
            f"[{self.experiment_name}] "
            f"Acc={self.accuracy:.4f} P={self.precision:.4f} "
            f"R={self.recall:.4f} F1={self.f1:.4f} AUC={self.auc_roc:.4f} "
            f"Params={self.param_count:,} Infer={self.inference_time_ms:.1f}ms"
        )


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    experiment_name: str = "",
    group: str = "",
    param_count: int = 0,
    inference_time_ms: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
) -> EvalResult:
    """Compute all standard metrics from true labels and predictions."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    if y_prob is not None:
        try:
            auc = roc_auc_score(y_true, y_prob)
        except ValueError:
            auc = 0.0
    else:
        auc = 0.0

    cm = confusion_matrix(y_true, y_pred).tolist()

    return EvalResult(
        experiment_name=experiment_name,
        group=group,
        accuracy=acc,
        precision=prec,
        recall=rec,
        f1=f1,
        auc_roc=auc,
        confusion_matrix=cm,
        param_count=param_count,
        inference_time_ms=inference_time_ms,
        extra=extra or {},
    )


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def measure_inference_time(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str,
    num_batches: int = 10,
) -> float:
    """Measure average per-sample inference time in milliseconds."""
    model.eval()
    total_time = 0.0
    total_samples = 0

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
            if isinstance(batch, (list, tuple)):
                inputs = batch[0].to(device)
                bs = inputs.size(0)
            else:
                inputs = batch.to(device)
                bs = inputs.size(0)

            start = time.perf_counter()
            model(inputs)
            if device == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()

            total_time += (end - start) * 1000
            total_samples += bs

    return total_time / max(total_samples, 1)


class ResultLogger:
    """Logs experiment results to a CSV file for cross-group comparison."""

    COLUMNS = [
        "experiment_name", "group", "accuracy", "precision", "recall",
        "f1", "auc_roc", "param_count", "inference_time_ms",
    ]

    def __init__(self, log_path: str = "outputs/results_log.csv"):
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        if not os.path.exists(log_path):
            with open(log_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.COLUMNS)
                writer.writeheader()

    def log(self, result: EvalResult):
        row = {k: getattr(result, k) for k in self.COLUMNS}
        with open(self.log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.COLUMNS)
            writer.writerow(row)
        print(f"  [Logger] {result.summary()}")

    def log_many(self, results: List[EvalResult]):
        for r in results:
            self.log(r)


def plot_comparison_table(
    results: List[EvalResult],
    save_path: Optional[str] = None,
    title: str = "Experiment Comparison",
):
    """Bar chart comparing F1, Accuracy, AUC across experiments."""
    names = [r.experiment_name for r in results]
    f1s = [r.f1 for r in results]
    accs = [r.accuracy for r in results]
    aucs = [r.auc_roc for r in results]

    x = np.arange(len(names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(10, len(names) * 1.5), 6))
    ax.bar(x - width, accs, width, label="Accuracy", color="#4c72b0")
    ax.bar(x, f1s, width, label="F1", color="#55a868")
    ax.bar(x + width, aucs, width, label="AUC-ROC", color="#c44e52")

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Score")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Chart saved: {save_path}")
    plt.close()


def plot_confusion_heatmap(
    result: EvalResult,
    save_path: Optional[str] = None,
):
    cm = np.array(result.confusion_matrix)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Fake (0)", "Real (1)"],
        yticklabels=["Fake (0)", "Real (1)"],
        ax=ax, annot_kws={"size": 16},
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix - {result.experiment_name}", fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
