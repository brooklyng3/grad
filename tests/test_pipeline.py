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


def test_evaluator_evasion_metrics(pipeline_setup):
    evaluator = pipeline_setup["evaluator"]
    x = pipeline_setup["x_test"]

    metrics = evaluator.evaluate_evasion(
        x, x, num_pre_evaded=10, total_attack_population=100
    )
    assert "conditional_asr_pct" in metrics
    assert "overall_evasion_rate_pct" in metrics
    assert "perturbation_distances" in metrics
    assert metrics["perturbation_distances"]["mean_linf"] == 0.0


def test_feedback_loop_initialization():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loop = AdversarialFeedbackLoop(device=device)

    assert loop.generator is not None
    assert loop.critic is not None
    assert loop.defender is not None
    assert loop.theta > 0.0
