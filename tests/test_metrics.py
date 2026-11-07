"""
Unit Tests for Evaluation Metrics and Reporting Utilities (Phase 5).
"""

import os
import json
import numpy as np
import pytest

from src.utils.metrics import (
    compute_detection_metrics,
    compute_roc_auc,
    aggregate_tradeoff_curve,
    format_markdown_table,
    export_metrics_json,
    export_metrics_csv,
)


def test_compute_detection_metrics():
    # 3 benign (0), 3 malicious (1)
    y_true = np.array([0, 0, 0, 1, 1, 1])
    # Threshold = 0.5
    # Predictions: [0, 0, 1, 0, 1, 1] -> TN=2, FP=1, FN=1, TP=2
    scores = np.array([0.1, 0.4, 0.8, 0.3, 0.6, 0.9])

    metrics = compute_detection_metrics(y_true, scores, threshold=0.5)

    assert metrics["threshold"] == 0.5
    assert metrics["total_samples"] == 6
    cm = metrics["confusion_matrix"]
    assert cm["true_negatives"] == 2
    assert cm["false_positives"] == 1
    assert cm["false_negatives"] == 1
    assert cm["true_positives"] == 2

    rates = metrics["rates"]
    assert pytest.approx(rates["tpr_recall"], 1e-4) == 2.0 / 3.0
    assert pytest.approx(rates["fpr"], 1e-4) == 1.0 / 3.0
    assert pytest.approx(rates["precision"], 1e-4) == 2.0 / 3.0
    assert pytest.approx(rates["accuracy"], 1e-4) == 4.0 / 6.0


def test_compute_roc_auc():
    y_true = np.array([0, 0, 1, 1])
    # Perfectly separated
    scores = np.array([0.1, 0.2, 0.8, 0.9])

    auc_val, fpr, tpr, thresholds = compute_roc_auc(y_true, scores)
    assert pytest.approx(auc_val, 1e-4) == 1.0
    assert len(fpr) == len(tpr) == len(thresholds)


def test_aggregate_tradeoff_curve():
    dummy_results = [
        {
            "param_name": "epsilon",
            "param_value": 0.5,
            "attack_type": "FGSM",
            "conditional_asr_pct": 10.0,
            "overall_evasion_rate_pct": 20.0,
            "rmse_metrics": {"adversarial_mean": 1.2},
            "perturbation_distances": {
                "mean_linf": 0.5,
                "mean_l2": 2.1,
                "mean_l0_modified_features": 80.0,
            },
        },
        {
            "param_name": "epsilon",
            "param_value": 0.1,
            "attack_type": "FGSM",
            "conditional_asr_pct": 2.0,
            "overall_evasion_rate_pct": 13.0,
            "rmse_metrics": {"adversarial_mean": 2.0},
            "perturbation_distances": {
                "mean_linf": 0.1,
                "mean_l2": 0.5,
                "mean_l0_modified_features": 75.0,
            },
        },
    ]

    tradeoff = aggregate_tradeoff_curve(dummy_results)
    assert len(tradeoff) == 2
    # Should be sorted by param_value ascending
    assert tradeoff[0]["param_value"] == 0.1
    assert tradeoff[1]["param_value"] == 0.5
    assert tradeoff[0]["conditional_asr_pct"] == 2.0


def test_format_markdown_table():
    headers = ["Method", "ASR (%)", "Linf"]
    rows = [
        ["Baseline", 0.0, 0.0],
        ["FGSM", 15.5, 0.25],
    ]
    table = format_markdown_table(headers, rows)
    assert "| Method | ASR (%) | Linf |" in table
    assert "| :--- | :---: | :---: |" in table
    assert "| Baseline | 0.0000 | 0.0000 |" in table
    assert "| FGSM | 15.5000 | 0.2500 |" in table


def test_export_metrics(tmp_path):
    json_path = os.path.join(tmp_path, "sub", "test_metrics.json")
    csv_path = os.path.join(tmp_path, "sub", "test_metrics.csv")

    data = {"test_metric": 42.0, "status": "success"}
    export_metrics_json(data, json_path)
    assert os.path.exists(json_path)
    with open(json_path) as f:
        loaded = json.load(f)
    assert loaded["test_metric"] == 42.0

    headers = ["ColA", "ColB"]
    rows = [[1, 2], [3, 4]]
    export_metrics_csv(headers, rows, csv_path)
    assert os.path.exists(csv_path)
    with open(csv_path) as f:
        content = f.read()
    assert "ColA,ColB" in content
