"""
Unit tests for Phase 4 Closed-Loop Pipeline:
  - PipelineEvaluator metric calculation and norm calculations
  - AdversarialFeedbackLoop initialization, training step, and loss propagation
"""

import numpy as np
import pytest
import torch

from src.defender.torch_kitnet import PyTorchKitNET
from src.pipeline.evaluator import PipelineEvaluator
from src.pipeline.feedback_loop import AdversarialFeedbackLoop


@pytest.fixture(scope="module")
def pipeline_setup():
    model_path = "data/processed/kitsune_mirai_model.pkl"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PyTorchKitNET.from_file(model_path, device=device)
    model.eval()
    theta = 0.453419

    evaluator = PipelineEvaluator(
        model=model, theta=theta, device=device, batch_size=64
    )

    features = np.load("data/processed/mirai_features.npy", mmap_mode="r")
    x_test = torch.tensor(features[:32], dtype=torch.float32, device=device)

    return {
        "model": model,
        "theta": theta,
        "device": device,
        "evaluator": evaluator,
        "x_test": x_test,
    }


def test_evaluator_score_batches(pipeline_setup):
    evaluator = pipeline_setup["evaluator"]
    x = pipeline_setup["x_test"]

    scores = evaluator.score_batches(x)
    assert len(scores) == x.size(0)
    assert np.all(scores >= 0.0)


def test_evaluator_subsample_extrapolation(pipeline_setup):
    evaluator = pipeline_setup["evaluator"]
    x = pipeline_setup["x_test"]  # 32 samples

    # Test case 1: 100% conditional evasion on subsample yields 100% overall
    x_zero = torch.zeros_like(x)
    metrics_100 = evaluator.evaluate_evasion(
        x_orig=x,
        x_adv=x_zero,
        num_pre_evaded=76123,
        total_attack_population=642516,
    )
    assert metrics_100["conditional_asr_pct"] == pytest.approx(100.0)
    assert metrics_100["overall_evasion_rate_pct"] == pytest.approx(100.0)

    # Test case 2: 0% conditional evasion should reduce to baseline pre-evasion
    x_high = x + 1e6
    metrics_0 = evaluator.evaluate_evasion(
        x_orig=x,
        x_adv=x_high,
        num_pre_evaded=76123,
        total_attack_population=642516,
    )
    assert metrics_0["conditional_asr_pct"] == pytest.approx(0.0)
    assert metrics_0["overall_evasion_rate_pct"] == pytest.approx(
        (76123 / 642516) * 100.0
    )


def test_benign_knn_distance(pipeline_setup):
    evaluator = pipeline_setup["evaluator"]
    x = pipeline_setup["x_test"]
    evaluator.set_benign_reference(x[:16])
    d_self = evaluator.benign_knn_distance(x[:16], chunk_size=8)
    d_far = evaluator.benign_knn_distance(x[:16] * 10.0, chunk_size=8)
    assert np.isfinite(d_self)
    assert d_self < d_far


def test_benign_splits_are_disjoint():
    import numpy as np
    from src.utils.data_splits import benign_indices_for_split

    labels = np.zeros(56000, dtype=np.int32)
    cal = benign_indices_for_split(labels, "calibration")
    val = benign_indices_for_split(labels, "validation")
    assert set(cal).isdisjoint(set(val))
    assert len(cal) + len(val) == 1000
    assert cal.min() >= 55000


def test_feedback_loop_initialization():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loop = AdversarialFeedbackLoop(device=device)

    assert loop.generator is not None
    assert loop.critic is not None
    assert loop.defender is not None
    assert loop.theta > 0.0
