"""
Evaluation Metrics and Statistical Reporting Utilities (Phase 5).

Provides functions for:
  1. Anomaly detection performance (ROC-AUC, Precision, Recall/TPR, FPR, F1, Accuracy).
  2. Perturbation vs. Evasion trade-off aggregation.
  3. Structured export to JSON and Markdown/CSV tables for thesis documentation.
"""

import json
import csv
import os
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix


def compute_detection_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> Dict[str, Any]:
    """Computes binary classification metrics for anomaly detection given a threshold.

    An anomaly is predicted if score > threshold (positive class = 1 / Malicious).

    Args:
        y_true: Binary ground truth labels (0 for benign, 1 for malicious).
        scores: Anomaly reconstruction scores (e.g. RMSE).
        threshold: Alert cutoff threshold theta.

    Returns:
        Dictionary containing TP, FP, TN, FN, TPR/Recall, FPR, Precision, F1, Accuracy.
    """
    assert len(y_true) == len(scores), "Label and score length mismatch"
    y_pred = (scores > threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    tpr = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    f1 = (
        float(2 * (precision * tpr) / (precision + tpr))
        if (precision + tpr) > 0
        else 0.0
    )
    accuracy = float((tp + tn) / len(y_true)) if len(y_true) > 0 else 0.0
    balanced_acc = float((tpr + (1.0 - fpr)) / 2.0)

    return {
        "threshold": float(threshold),
        "total_samples": int(len(y_true)),
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
        "rates": {
            "tpr_recall": tpr,
            "fpr": fpr,
            "precision": precision,
            "f1_score": f1,
            "accuracy": accuracy,
            "balanced_accuracy": balanced_acc,
        },
    }


def compute_roc_auc(
    y_true: np.ndarray,
    scores: np.ndarray,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Computes Receiver Operating Characteristic (ROC) curve and AUC.

    Args:
        y_true: Binary ground truth labels (0=benign, 1=malicious).
        scores: Continuous anomaly scores (RMSE).

    Returns:
        Tuple of (roc_auc, fpr_array, tpr_array, thresholds_array).
    """
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    roc_auc_val = float(auc(fpr, tpr))
    return roc_auc_val, fpr, tpr, thresholds


def aggregate_tradeoff_curve(
    sweep_results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Aggregates evaluation results into a sorted distortion vs evasion trade-off list.

    Args:
        sweep_results: List of dictionaries from PipelineEvaluator.evaluate_evasion,
                       each augmented with experimental parameters
                       (e.g. 'param_name', 'param_value').

    Returns:
        Sorted list summarizing parameter, Linf, L2, ASR_cond, and ER_total.
    """
    tradeoff = []
    for item in sweep_results:
        summary = {
            "param_name": item.get("param_name", "epsilon"),
            "param_value": item.get("param_value", 0.0),
            "attack_type": item.get("attack_type", "Unknown"),
            "conditional_asr_pct": item.get("conditional_asr_pct", 0.0),
            "overall_evasion_rate_pct": item.get("overall_evasion_rate_pct", 0.0),
            "adversarial_mean_rmse": item.get("rmse_metrics", {}).get(
                "adversarial_mean", 0.0
            ),
            "mean_linf": item.get("perturbation_distances", {}).get("mean_linf", 0.0),
            "mean_l2": item.get("perturbation_distances", {}).get("mean_l2", 0.0),
            "mean_l0": item.get("perturbation_distances", {}).get(
                "mean_l0_modified_features", 0.0
            ),
        }
        tradeoff.append(summary)

    tradeoff.sort(key=lambda x: (x["param_value"]))
    return tradeoff


def format_markdown_table(
    headers: List[str],
    rows: List[List[Any]],
    alignments: Optional[List[str]] = None,
) -> str:
    """Renders a clean GitHub-flavored markdown table.

    Args:
        headers: List of column header names.
        rows: List of row values.
        alignments: Optional list of alignments ('left', 'center', 'right').

    Returns:
        Formatted markdown table string.
    """
    if not headers:
        return ""

    if alignments is None:
        alignments = ["left"] + ["center"] * (len(headers) - 1)

    align_map = {
        "left": ":---",
        "center": ":---:",
        "right": "---:",
    }

    header_line = "| " + " | ".join(str(h) for h in headers) + " |"
    separator_line = (
        "| " + " | ".join(align_map.get(a, ":---") for a in alignments) + " |"
    )

    row_lines = []
    for row in rows:
        formatted_cells = []
        for cell in row:
            if isinstance(cell, float):
                formatted_cells.append(f"{cell:.4f}")
            else:
                formatted_cells.append(str(cell))
        row_lines.append("| " + " | ".join(formatted_cells) + " |")

    return "\n".join([header_line, separator_line] + row_lines)


def export_metrics_json(data: Dict[str, Any], filepath: str) -> None:
    """Exports structured metrics to a JSON file, ensuring parent directory exists."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def export_metrics_csv(
    headers: List[str],
    rows: List[List[Any]],
    filepath: str,
) -> None:
    """Exports tabular metrics to a CSV file for plotting."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)
