"""End-to-end checks for the Phase 6 evaluation CLI."""

import json
import os
import subprocess
import sys

import numpy as np
import torch

import run_pipeline
from src.attacker.generator import PolymorphicGenerator
from src.defender.torch_kitnet import PyTorchKitNET
from src.pipeline.evaluator import PipelineEvaluator
from src.utils.data_splits import attack_indices_for_split, benign_indices_for_split


def _components():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PyTorchKitNET.from_file(
        "data/processed/kitsune_mirai_model.pkl", device=device
    )
    model.eval()
    with open("data/processed/threshold_config.yaml", "r") as f:
        import yaml

        theta = float(yaml.safe_load(f)["threshold"])
    evaluator = PipelineEvaluator(
        model=model, theta=theta, device=device, batch_size=64
    )
    evaluator.num_pre_evaded = 76123
    evaluator.total_attack_population = 642516
    return model, evaluator, device


def test_cli_help_exposes_phase6_flags():
    out = subprocess.run(
        [sys.executable, "run_pipeline.py", "--help"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    for flag in (
        "--seed",
        "--eps",
        "--retries",
        "--train_epochs",
        "--train_batch_size",
        "--deepfool_max_eps",
        "--refine_steps",
    ):
        assert flag in out


def test_sampler_uses_only_eval_split():
    _, evaluator, device = _components()
    features = np.load("data/processed/mirai_features.npy", mmap_mode="r")
    labels = run_pipeline.load_labels()
    train = set(attack_indices_for_split(np.where(labels == 1)[0], "train").tolist())
    evals = set(attack_indices_for_split(np.where(labels == 1)[0], "eval").tolist())

    selected, indices = run_pipeline.get_detected_attack_sample(
        features, evaluator, 20, device, phase="scanning", return_indices=True
    )
    assert selected.size(0) == len(indices)
    assert all(int(i) in evals for i in indices)
    assert all(int(i) not in train for i in indices)


def test_metadata_round_trip(tmp_path):
    path = tmp_path / "gen.pt"
    meta = tmp_path / "gen.meta.json"
    generator = PolymorphicGenerator(
        100, 32, [128, 256, 128], max_eps=1.5, mode="additive"
    )
    torch.save(generator.state_dict(), path)
    meta.write_text(json.dumps({"eps": 1.5, "mode": "additive"}))

    loaded, eps = run_pipeline._load_generator(
        str(path), "additive", torch.device("cpu"), default_eps=2.0
    )
    assert eps == 1.5
    assert loaded.max_eps == 1.5


def test_trained_manifold_checkpoint_evades():
    model, evaluator, device = _components()
    generator, _ = run_pipeline._load_generator(
        "data/attack_samples/wgan_manifold_generator.pt",
        "manifold",
        device,
        default_eps=2.0,
    )
    features = np.load("data/processed/mirai_features.npy", mmap_mode="r")
    x, _ = run_pipeline.get_detected_attack_sample(
        features, evaluator, 64, device, phase="flood", return_indices=True
    )
    with torch.no_grad():
        adv = generator.mutate_raw(
            x, evaluator.norm_min, evaluator.scales, eps=2.0, mode="manifold"
        )
        rmse = model(adv)
    assert float((rmse <= evaluator.theta).float().mean()) >= 0.9


def test_clean_eval_uses_validation_split_only(tmp_path):
    labels = run_pipeline.load_labels()
    validation = benign_indices_for_split(labels, "validation")
    calibration = set(benign_indices_for_split(labels, "calibration").tolist())
    assert validation.size == 33310
    assert not calibration.intersection(set(validation.tolist()))


def test_evaluate_all_emits_full_row_set(tmp_path):
    output = tmp_path / "reports"
    subprocess.run(
        [
            sys.executable,
            "run_pipeline.py",
            "--mode",
            "evaluate_all",
            "--samples",
            "20",
            "--attack_phase",
            "scanning",
            "--output_dir",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": "."},
    )
    table = (output / "evaluation_table.md").read_text()
    for row in (
        "Envelope Clamp Only",
        "DeepFool",
        "best-of-k",
        "feedback refine",
        "Manifold, trained",
        "Manifold, untrained",
        "kNN->benign",
    ):
        assert row in table
