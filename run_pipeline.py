#!/usr/bin/env python3
"""
End-to-End Adversarial ML Evaluation Pipeline CLI (Phase 6).

Unified command-line orchestrator for evaluating Kitsune NIDS vs. Polymorphic
Adversarial Attacks (FGSM, PGD, WGAN-GP).

Usage:
  python run_pipeline.py --mode evaluate_clean
  python run_pipeline.py --mode fgsm_sweep --samples 1000
  python run_pipeline.py --mode pgd_attack --samples 1000
  python run_pipeline.py --mode train_wgan --samples 5000
  python run_pipeline.py --mode evaluate_all --samples 1000
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, Any, List, Tuple
import numpy as np
import torch
import yaml

# Ensure project root and Kitsune-py are importable
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("Kitsune-py"))

from src.defender.torch_kitnet import PyTorchKitNET  # noqa: E402
from src.attacker.perturbations import (  # noqa: E402
    deepfool_attack,
    envelope_projection,
    fgsm_attack,
    pgd_attack,
    wgan_best_of_k,
    wgan_refine,
)
from src.attacker.generator import PolymorphicGenerator  # noqa: E402
from src.pipeline.evaluator import PipelineEvaluator  # noqa: E402
from src.pipeline.feedback_loop import AdversarialFeedbackLoop  # noqa: E402
from src.utils.data_splits import (  # noqa: E402
    ATTACK_START,
    attack_indices_for_split,
    benign_indices_for_split,
)
from src.utils.metrics import (  # noqa: E402
    compute_detection_metrics,
    compute_roc_auc,
    format_markdown_table,
    export_metrics_json,
    export_metrics_csv,
)
from src.utils.seed import seed_everything  # noqa: E402


def parse_args() -> argparse.Namespace:
    # Pull defaults from the WGAN config so every documented key is live.
    try:
        with open("configs/wgan_config.yaml", "r") as f:
            _wcfg = yaml.safe_load(f) or {}
    except FileNotFoundError:
        _wcfg = {}
    _pert = _wcfg.get("perturbation", {})
    _train = _wcfg.get("training", {})

    parser = argparse.ArgumentParser(
        description="Kitsune NIDS vs Polymorphic Adversarial ML Pipeline"
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=[
            "evaluate_clean",
            "fgsm_sweep",
            "pgd_attack",
            "train_wgan",
            "evaluate_all",
        ],
        help="Pipeline execution mode.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/kitsune_config.yaml",
        help="Path to Kitsune YAML configuration file.",
    )
    parser.add_argument(
        "--wgan_config",
        type=str,
        default="configs/wgan_config.yaml",
        help="Path to WGAN YAML configuration file.",
    )
    parser.add_argument(
        "--threshold_config",
        type=str,
        default="data/processed/threshold_config.yaml",
        help="Path to precomputed threshold YAML file.",
    )
    parser.add_argument(
        "--features_path",
        type=str,
        default="data/processed/mirai_features.npy",
        help="Path to memory-mapped features .npy file.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="data/processed/kitsune_mirai_model.pkl",
        help="Path to serialized KitNET model weights.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="reports",
        help="Directory to save metric reports, JSON dumps, and tables.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="data/attack_samples",
        help="Directory for WGAN checkpoints and metadata.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=1000,
        help="Number of detected attack samples to evaluate during attack sweeps.",
    )
    parser.add_argument(
        "--attack_phase",
        type=str,
        default="both",
        choices=["scanning", "flood", "both"],
        help="Attack regime: early scanning, volumetric flood, or both.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for Python, NumPy, and Torch.",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=float(_pert.get("eps_linf", 2.0)),
        help="Perturbation budget used for WGAN training and evaluation.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=int(_pert.get("best_of_k_retries", 10)),
        help="Maximum generator draws per packet for best-of-k evasion.",
    )
    parser.add_argument(
        "--refine_steps",
        type=int,
        default=10,
        help="Generator feedback-refinement steps for the box WGAN row.",
    )
    parser.add_argument(
        "--deepfool_max_eps",
        type=float,
        default=float(_pert.get("eps_linf", 2.0)),
        help="L-inf budget cap for the DeepFool baseline.",
    )
    parser.add_argument(
        "--deepfool_steps",
        type=int,
        default=int(_pert.get("max_iterations", 50)),
        help="Iterations for the DeepFool baseline.",
    )
    parser.add_argument(
        "--deepfool_step",
        type=float,
        default=float(_pert.get("step_size", 0.1)),
        help="Step scale for the DeepFool baseline.",
    )
    parser.add_argument(
        "--train_epochs",
        type=int,
        default=int(_train.get("epochs", 10)),
        help="WGAN feedback-loop training epochs.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=int(_train.get("batch_size", 128)),
        help="WGAN feedback-loop training batch size.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Mini-batch size for GPU inference.",
    )
    parser.add_argument(
        "--manifold_shaping",
        action="store_true",
        default=False,
        help="Train WGAN-GP generator with manifold envelope shaping.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Execution device ('cuda' or 'cpu'). Defaults to CUDA if available.",
    )
    return parser.parse_args()


def get_device(requested: str | None = None) -> torch.device:
    if requested:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_labels(path: str = "data/raw/mirai_labels.csv") -> np.ndarray:
    """Loads the binary ground-truth label vector."""
    with open(path, "r") as f:
        return np.array(
            [int(line.strip()) for line in f if line.strip()], dtype=np.int32
        )


def load_pipeline_components(
    args: argparse.Namespace, device: torch.device
) -> Tuple[PyTorchKitNET, float, np.ndarray, PipelineEvaluator]:
    """Loads PyTorchKitNET, threshold theta, feature matrix, and evaluator."""
    print(f"[*] Initializing pipeline on device: {device}")

    # 1. Load KitNET
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {args.model_path}")
    torch_model = PyTorchKitNET.from_file(args.model_path, device=device)

    # 2. Load Threshold
    if os.path.exists(args.threshold_config):
        with open(args.threshold_config, "r") as f:
            tcfg = yaml.safe_load(f)
            theta = float(tcfg.get("threshold", 0.455986))
            attack_meta = tcfg.get("attack_metrics", {})
            num_pre_evaded = int(attack_meta.get("evaded_count", 76123))
            total_attack_population = int(attack_meta.get("num_evaluated", 642516))
    else:
        theta = 0.455986
        num_pre_evaded = 76123
        total_attack_population = 642516
        print(f"[!] Warning: {args.threshold_config} missing, using default {theta}")

    print(
        f"[*] KitNET loaded ({len(torch_model.ensemble)} micro-AEs), theta={theta:.6f}"
    )

    # 3. Load Features (memory-mapped)
    if not os.path.exists(args.features_path):
        raise FileNotFoundError(f"Features file not found: {args.features_path}")
    feats_mmap = np.load(args.features_path, mmap_mode="r")
    print(f"[*] Memory-mapped features loaded: {feats_mmap.shape} ({feats_mmap.dtype})")

    # 4. Evaluator Engine
    evaluator = PipelineEvaluator(
        model=torch_model,
        theta=theta,
        device=device,
        batch_size=args.batch_size,
    )
    evaluator.num_pre_evaded = num_pre_evaded
    evaluator.total_attack_population = total_attack_population
    labels = load_labels()
    validation = benign_indices_for_split(labels, "validation")
    evaluator.set_benign_reference(
        torch.tensor(np.array(feats_mmap[validation[:2000]], dtype=np.float32))
    )
    return torch_model, theta, feats_mmap, evaluator


def get_detected_attack_sample(
    feats_mmap: np.ndarray,
    evaluator: PipelineEvaluator,
    num_samples: int,
    device: torch.device,
    phase: str = "flood",
    return_indices: bool = False,
) -> Any:
    """Extracts confirmed detected attack packets (RMSE > theta).

    Args:
        feats_mmap: Memory-mapped features array.
        evaluator: PipelineEvaluator engine.
        num_samples: Target number of detected packets.
        device: Torch device.
        phase: 'scanning' (early reconnaissance phase) or 'flood' (volumetric flood).
        return_indices: If True, also return the selected absolute indices.

    Returns:
        Selected feature tensor, or (tensor, indices) when ``return_indices``.
    """
    labels = load_labels()
    eval_attacks = attack_indices_for_split(np.where(labels == 1)[0], "eval")

    if phase == "scanning":
        # The first few hundred attack packets are still ramping up and often
        # score below theta, so never use a candidate window smaller than 10k.
        scan_window = max(10000, num_samples * 10)
        candidate_indices = eval_attacks[(eval_attacks - ATTACK_START) < scan_window]
    else:
        later = eval_attacks[(eval_attacks - ATTACK_START) >= 10000]
        n_candidates = max(1000, num_samples * 2)
        positions = np.linspace(
            0, len(later) - 1, min(n_candidates, len(later)), dtype=int
        )
        candidate_indices = later[positions]

    candidate_feats = torch.from_numpy(
        np.array(feats_mmap[candidate_indices], dtype=np.float32)
    )

    scores = evaluator.score_batches(candidate_feats)
    detected_mask = scores > evaluator.theta
    detected_feats = candidate_feats[detected_mask]

    count = min(len(detected_feats), num_samples)
    selected = detected_feats[:count]
    selected_indices = candidate_indices[detected_mask][:count]
    mean_rmse = (
        float(np.mean(evaluator.score_batches(selected))) if count else float("nan")
    )
    print(
        f"[*] [{phase.upper()}] Selected {len(selected)} detected packets. "
        f"Initial mean RMSE: {mean_rmse:.4f}"
    )
    if len(selected) < num_samples:
        print(
            f"[!] Only {len(selected)} detected packets available; "
            f"requested {num_samples}."
        )
    if return_indices:
        return selected.to(device), selected_indices
    return selected.to(device)


# -------------------------------------------------------------------------
# Pipeline Modes
# -------------------------------------------------------------------------


def run_evaluate_clean(
    args: argparse.Namespace,
    torch_model: PyTorchKitNET,
    theta: float,
    feats_mmap: np.ndarray,
    evaluator: PipelineEvaluator,
) -> Dict[str, Any]:
    """Evaluates clean baseline detection performance on benign and malicious sets."""
    print("\n" + "=" * 70)
    print("MODE: Clean Baseline Detection Evaluation")
    print("=" * 70)

    # Disjoint benign validation split (calibration used even parity)
    labels = load_labels()
    if len(labels) != len(feats_mmap):
        raise ValueError("Label and feature lengths do not match")
    benign_indices = benign_indices_for_split(labels, "validation")
    benign_feats = torch.from_numpy(
        np.array(feats_mmap[benign_indices], dtype=np.float32)
    )
    print(f"[*] Benign validation subset size: {len(benign_feats):,} packets")

    # Sample attack slice (up to 50,000 for high-confidence metrics)
    attack_stride = max(1, evaluator.total_attack_population // 50000)
    attack_indices = np.arange(ATTACK_START, len(feats_mmap), attack_stride)
    attack_feats = torch.from_numpy(
        np.array(feats_mmap[attack_indices], dtype=np.float32)
    )
    print(f"[*] Malicious evaluation subset size: {len(attack_feats):,} packets")

    t0 = time.time()
    benign_scores = evaluator.score_batches(benign_feats)
    attack_scores = evaluator.score_batches(attack_feats)
    elapsed = time.time() - t0

    y_true = np.concatenate([np.zeros(len(benign_scores)), np.ones(len(attack_scores))])
    all_scores = np.concatenate([benign_scores, attack_scores])

    # Compute Detection Metrics
    metrics = compute_detection_metrics(y_true, all_scores, threshold=theta)
    roc_auc_val, _, _, _ = compute_roc_auc(y_true, all_scores)
    metrics["roc_auc"] = roc_auc_val
    metrics["eval_time_seconds"] = elapsed
    benign_queries = torch.from_numpy(
        np.array(feats_mmap[benign_indices[2000:2200]], dtype=np.float32)
    )
    metrics["benign_knn_baseline"] = evaluator.benign_knn_distance(benign_queries)
    benign_queries = torch.from_numpy(
        np.array(feats_mmap[benign_indices[2000:2200]], dtype=np.float32)
    )
    metrics["benign_knn_baseline"] = evaluator.benign_knn_distance(benign_queries)

    # Display Markdown Table
    table_headers = ["Metric", "Value"]
    rates = metrics["rates"]
    table_rows = [
        ["Alert Threshold (theta)", f"{theta:.6f}"],
        ["ROC-AUC", f"{roc_auc_val:.6f}"],
        ["True Positive Rate (TPR / Recall)", f"{rates['tpr_recall'] * 100:.2f}%"],
        ["False Positive Rate (FPR)", f"{rates['fpr'] * 100:.4f}%"],
        ["Precision", f"{rates['precision'] * 100:.2f}%"],
        ["F1-Score", f"{rates['f1_score']:.4f}"],
        ["Overall Accuracy", f"{rates['accuracy'] * 100:.2f}%"],
        ["Benign kNN baseline (norm. L2)", f"{metrics['benign_knn_baseline']:.4f}"],
        ["Inference Throughput", f"{len(all_scores) / elapsed:,.1f} pkts/sec"],
    ]
    md_table = format_markdown_table(table_headers, table_rows)
    print("\n" + md_table + "\n")

    out_file = os.path.join(args.output_dir, "clean_detection_metrics.json")
    export_metrics_json(metrics, out_file)
    print(f"[+] Saved clean detection metrics to: {out_file}")
    return metrics


def run_fgsm_sweep(
    args: argparse.Namespace,
    torch_model: PyTorchKitNET,
    evaluator: PipelineEvaluator,
    mal_detected: torch.Tensor,
) -> List[Dict[str, Any]]:
    """Sweeps single-step FGSM over multiple epsilon budgets."""
    print("\n" + "=" * 70)
    print("MODE: Single-Step FGSM Epsilon Perturbation Sweep")
    print("=" * 70)

    epsilons = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0]
    sweep_results = []
    headers = [
        "Epsilon",
        "ASR_cond (%)",
        "ER_total (%)",
        "Mean RMSE",
        "Mean Linf",
        "Mean L2",
    ]
    rows = []

    for eps in epsilons:
        x_adv, _ = fgsm_attack(
            model=torch_model,
            x=mal_detected,
            eps=eps,
            scales=evaluator.scales,
        )
        res = evaluator.evaluate_evasion(
            x_orig=mal_detected,
            x_adv=x_adv,
            num_pre_evaded=evaluator.num_pre_evaded,
            total_attack_population=evaluator.total_attack_population,
        )
        res["param_name"] = "epsilon"
        res["param_value"] = eps
        res["attack_type"] = "FGSM"
        sweep_results.append(res)

        rows.append(
            [
                f"{eps:.2f}",
                f"{res['conditional_asr_pct']:.2f}%",
                f"{res['overall_evasion_rate_pct']:.2f}%",
                f"{res['rmse_metrics']['adversarial_mean']:.4f}",
                f"{res['perturbation_distances']['mean_linf']:.4f}",
                f"{res['perturbation_distances']['mean_l2']:.4f}",
            ]
        )
        torch.cuda.empty_cache()

    md_table = format_markdown_table(headers, rows)
    print("\n" + md_table + "\n")

    csv_rows = [
        [
            r["param_value"],
            r["conditional_asr_pct"],
            r["overall_evasion_rate_pct"],
            r["rmse_metrics"]["adversarial_mean"],
            r["perturbation_distances"]["mean_linf"],
            r["perturbation_distances"]["mean_l2"],
        ]
        for r in sweep_results
    ]
    out_csv = os.path.join(args.output_dir, "fgsm_tradeoff.csv")
    export_metrics_csv(
        ["epsilon", "asr_cond", "er_total", "mean_rmse", "mean_linf", "mean_l2"],
        csv_rows,
        out_csv,
    )
    print(f"[+] Saved FGSM sweep trade-off to: {out_csv}")
    return sweep_results


def run_pgd_attack(
    args: argparse.Namespace,
    torch_model: PyTorchKitNET,
    evaluator: PipelineEvaluator,
    mal_detected: torch.Tensor,
) -> List[Dict[str, Any]]:
    """Evaluates multi-step PGD and Manifold Traffic-Shaping."""
    print("\n" + "=" * 70)
    print("MODE: Multi-Step PGD & Manifold Traffic-Shaping Benchmarking")
    print("=" * 70)

    configs = [
        {
            "name": "PGD (Box, 20 steps, eps=1.0)",
            "steps": 20,
            "eps": 1.0,
            "alpha": 0.05,
            "manifold": False,
        },
        {
            "name": "PGD (Box, 50 steps, eps=2.0)",
            "steps": 50,
            "eps": 2.0,
            "alpha": 0.05,
            "manifold": False,
        },
        {
            "name": "Manifold Shaping (20 steps)",
            "steps": 20,
            "eps": 10.0,
            "alpha": 0.10,
            "manifold": True,
        },
        {
            "name": "Manifold Shaping (50 steps)",
            "steps": 50,
            "eps": 10.0,
            "alpha": 0.10,
            "manifold": True,
        },
    ]

    results = []
    headers = [
        "Attack Configuration",
        "Steps",
        "ASR_cond (%)",
        "ER_total (%)",
        "Mean RMSE",
        "Mean Linf",
    ]
    rows = []

    for cfg in configs:
        x_adv, _ = pgd_attack(
            model=torch_model,
            x=mal_detected,
            theta=evaluator.theta,
            scales=evaluator.scales,
            eps=cfg["eps"],
            alpha=cfg["alpha"],
            num_steps=cfg["steps"],
            manifold_projection=cfg["manifold"],
            norm_min=evaluator.norm_min,
            norm_max=evaluator.norm_max,
        )
        res = evaluator.evaluate_evasion(
            x_orig=mal_detected,
            x_adv=x_adv,
            num_pre_evaded=evaluator.num_pre_evaded,
            total_attack_population=evaluator.total_attack_population,
        )
        res["config_name"] = cfg["name"]
        results.append(res)

        rows.append(
            [
                cfg["name"],
                cfg["steps"],
                f"{res['conditional_asr_pct']:.2f}%",
                f"{res['overall_evasion_rate_pct']:.2f}%",
                f"{res['rmse_metrics']['adversarial_mean']:.4f}",
                f"{res['perturbation_distances']['mean_linf']:.4f}",
            ]
        )
        torch.cuda.empty_cache()

    md_table = format_markdown_table(headers, rows)
    print("\n" + md_table + "\n")

    out_file = os.path.join(args.output_dir, "pgd_benchmark.json")
    export_metrics_json({"experiments": results}, out_file)
    print(f"[+] Saved PGD benchmark results to: {out_file}")
    return results


def run_train_wgan(
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, Any]:
    """Executes closed-loop WGAN-GP feedback training."""
    mode_str = "Manifold Envelope" if args.manifold_shaping else "Additive Box"
    print("\n" + "=" * 70)
    print(f"MODE: Closed-Loop WGAN-GP Training ({mode_str})")
    print("=" * 70)

    feedback_loop = AdversarialFeedbackLoop(
        config_path=args.wgan_config,
        kitsune_config_path=args.config,
        device=device,
        manifold_shaping=args.manifold_shaping,
        eps=args.eps,
        threshold_config_path=args.threshold_config,
        model_path=args.model_path,
        seed=args.seed,
    )

    ckpt_dir = args.checkpoint_dir
    os.makedirs(ckpt_dir, exist_ok=True)
    g_name = (
        "wgan_manifold_generator.pt" if args.manifold_shaping else "wgan_generator.pt"
    )

    res = feedback_loop.train_loop(
        num_epochs=args.train_epochs,
        batch_size=args.train_batch_size,
        lambda_evade=50.0 if args.manifold_shaping else 30.0,
        lambda_dist=0.01 if args.manifold_shaping else 0.05,
        save_dir=ckpt_dir,
        checkpoint_name=g_name,
    )

    # Persist the training budget/mode alongside the checkpoint so evaluation
    # can never silently use a different epsilon.
    g_path = os.path.join(ckpt_dir, g_name)
    meta_path = os.path.join(ckpt_dir, g_name.replace(".pt", ".meta.json"))
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "eps": res["eps"],
                "seed": res["seed"],
                "mode": "manifold" if args.manifold_shaping else "additive",
                "best_epoch": res["best_epoch"],
                "best_selection_asr": res["best_selection_asr"],
                "test_asr": res["test_asr"],
            },
            f,
            indent=2,
        )
    print(f"[+] Generator saved to: {g_path}")
    print(f"[+] Training metadata saved to: {meta_path}")

    out_file = os.path.join(
        args.output_dir,
        (
            "wgan_manifold_summary.json"
            if args.manifold_shaping
            else "wgan_training_summary.json"
        ),
    )
    export_metrics_json(res, out_file)
    return res


def _load_generator(
    path: str, mode: str, device: torch.device, default_eps: float
) -> Tuple[PolymorphicGenerator, float]:
    """Loads a generator checkpoint plus its persisted training budget."""
    meta_path = path.replace(".pt", ".meta.json")
    meta: Dict[str, Any] = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    eps = float(meta.get("eps", default_eps))
    generator = PolymorphicGenerator(
        feature_dim=100,
        latent_dim=32,
        hidden_dims=[128, 256, 128],
        max_eps=eps,
        mode=mode,
    ).to(device)
    generator.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    generator.eval()
    return generator, eps


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def evaluate_attack_suite(
    suite_title: str,
    mal_detected: torch.Tensor,
    torch_model: PyTorchKitNET,
    evaluator: PipelineEvaluator,
    device: torch.device,
    retries: int = 10,
    refine_steps: int = 10,
    default_eps: float = 2.0,
    deepfool_max_eps: float = 2.0,
    deepfool_steps: int = 50,
    deepfool_step: float = 0.1,
    checkpoint_dir: str = "data/attack_samples",
    seed: int = 42,
) -> Tuple[str, List[List[Any]]]:
    """Evaluates all adversarial paradigms on a given detected malicious subset."""
    summary_rows = []
    headers = [
        "Paradigm / Attack",
        "Perturbation Constraint",
        "Draws/pkt",
        "Mean Attack RMSE",
        "Conditional ASR (%)",
        "Overall Evasion Rate (%)",
        "Mean Linf",
        "kNN->benign (norm. L2)",
        "Latency (ms / 1k pkts)",
    ]

    def add_row(
        name: str,
        constraint: str,
        x_adv: torch.Tensor,
        latency_ms: float,
        attempts: float = 1.0,
    ) -> None:
        res = evaluator.evaluate_evasion(
            x_orig=mal_detected,
            x_adv=x_adv,
            num_pre_evaded=evaluator.num_pre_evaded,
            total_attack_population=evaluator.total_attack_population,
        )
        summary_rows.append(
            [
                name,
                constraint,
                f"{attempts:.2f}",
                res["rmse_metrics"]["adversarial_mean"],
                f"{res['conditional_asr_pct']:.2f}%",
                f"{res['overall_evasion_rate_pct']:.2f}%",
                res["perturbation_distances"]["mean_linf"],
                f"{evaluator.benign_knn_distance(x_adv):.4f}",
                f"{latency_ms * 1000.0 / max(1, len(mal_detected)):.1f} ms",
            ]
        )

    # Warmup GPU (both forward inference and autograd backward kernels)
    with torch.no_grad():
        _ = torch_model(mal_detected[:10])
    _warm_x = mal_detected[:2].clone().detach().requires_grad_(True)
    _warm_loss = torch_model(_warm_x).sum()
    _warm_loss.backward()
    del _warm_x, _warm_loss
    _sync(device)

    # 1. Clean unperturbed
    add_row("Clean Baseline", "None (Unperturbed)", mal_detected, 0.0, 0.0)

    # 2. Envelope projection only (the mechanism behind the manifold attack)
    _sync(device)
    t0 = time.perf_counter()
    x_clamp = envelope_projection(mal_detected, evaluator.norm_min, evaluator.norm_max)
    _sync(device)
    add_row(
        "Envelope Clamp Only",
        "Unbudgeted [min, max]",
        x_clamp,
        (time.perf_counter() - t0) * 1000.0,
    )

    # 3. Single-step FGSM baselines
    for eps in (0.05, 0.25):
        _sync(device)
        t0 = time.perf_counter()
        x_fgsm, _ = fgsm_attack(
            model=torch_model, x=mal_detected, eps=eps, scales=evaluator.scales
        )
        _sync(device)
        add_row(
            "Single-Step FGSM",
            f"eps = {eps:g} * s",
            x_fgsm,
            (time.perf_counter() - t0) * 1000.0,
        )

    # 4. Multi-Step PGD (box, eps=2.0)
    _sync(device)
    t0 = time.perf_counter()
    x_pgd, _ = pgd_attack(
        model=torch_model,
        x=mal_detected,
        theta=evaluator.theta,
        scales=evaluator.scales,
        eps=2.0,
        alpha=0.05,
        num_steps=50,
        manifold_projection=False,
        norm_min=evaluator.norm_min,
        norm_max=evaluator.norm_max,
    )
    _sync(device)
    add_row(
        "Multi-Step PGD (Box)",
        "50 steps, eps = 2.0 * s",
        x_pgd,
        (time.perf_counter() - t0) * 1000.0,
    )

    # 5. DeepFool (bounded at eps=2.0)
    _sync(device)
    t0 = time.perf_counter()
    x_deepfool, _, _ = deepfool_attack(
        model=torch_model,
        x=mal_detected,
        theta=evaluator.theta,
        scales=evaluator.scales,
        max_iter=deepfool_steps,
        step_size=deepfool_step,
        max_eps=deepfool_max_eps,
    )
    _sync(device)
    add_row(
        "DeepFool (bounded)",
        "50 iters, eps <= 2.0 * s",
        x_deepfool,
        (time.perf_counter() - t0) * 1000.0,
    )

    # 6. Multi-Step PGD (unbudgeted manifold projection)
    _sync(device)
    t0 = time.perf_counter()
    x_manifold, _ = pgd_attack(
        model=torch_model,
        x=mal_detected,
        theta=evaluator.theta,
        scales=evaluator.scales,
        eps=10.0,
        alpha=0.10,
        num_steps=50,
        manifold_projection=True,
        norm_min=evaluator.norm_min,
        norm_max=evaluator.norm_max,
    )
    _sync(device)
    add_row(
        "Multi-Step PGD (Manifold)",
        "Unbudgeted [min, max], 50 steps",
        x_manifold,
        (time.perf_counter() - t0) * 1000.0,
    )

    # 7. WGAN additive generator: single pass and best-of-k retry loop
    ckpt_additive = os.path.join(checkpoint_dir, "wgan_generator.pt")
    if os.path.exists(ckpt_additive):
        gen_add, add_eps = _load_generator(
            ckpt_additive, "additive", device, default_eps
        )
        _sync(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            x_wgan_add = gen_add.mutate_raw(
                mal_detected,
                evaluator.norm_min,
                evaluator.scales,
                eps=add_eps,
                mode="additive",
            )
        _sync(device)
        add_row(
            "Polymorphic WGAN-GP (Box, 1 draw)",
            f"eps = {add_eps:g} * s",
            x_wgan_add,
            (time.perf_counter() - t0) * 1000.0,
        )

        _sync(device)
        t0 = time.perf_counter()
        x_wgan_k, _, attempts = wgan_best_of_k(
            gen_add,
            torch_model,
            mal_detected,
            evaluator.theta,
            evaluator.norm_min,
            evaluator.scales,
            eps=add_eps,
            mode="additive",
            max_retries=retries,
        )
        _sync(device)
        add_row(
            "Polymorphic WGAN-GP (Box, best-of-k)",
            f"eps = {add_eps:g} * s, retry until theta",
            x_wgan_k,
            (time.perf_counter() - t0) * 1000.0,
            attempts=float(attempts.float().mean().item()),
        )

        _sync(device)
        t0 = time.perf_counter()
        x_wgan_r, _, refine_attempts = wgan_refine(
            gen_add,
            torch_model,
            mal_detected,
            evaluator.theta,
            evaluator.norm_min,
            evaluator.scales,
            eps=add_eps,
            steps=refine_steps,
            mode="additive",
        )
        _sync(device)
        add_row(
            "Polymorphic WGAN-GP (Box, feedback refine)",
            f"eps = {add_eps:g} * s x {refine_steps} steps (aggregate)",
            x_wgan_r,
            (time.perf_counter() - t0) * 1000.0,
            attempts=float(refine_attempts.float().mean().item()),
        )

    # 8. Trained manifold generator (unbudgeted envelope + learned jitter)
    ckpt_manifold = os.path.join(checkpoint_dir, "wgan_manifold_generator.pt")
    if os.path.exists(ckpt_manifold):
        gen_man, man_eps = _load_generator(
            ckpt_manifold, "manifold", device, default_eps
        )
        _sync(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            x_wgan_man = gen_man.mutate_raw(
                mal_detected,
                evaluator.norm_min,
                evaluator.scales,
                eps=man_eps,
                mode="manifold",
            )
        _sync(device)
        add_row(
            "WGAN-GP (Manifold, trained)",
            "Unbudgeted learned target, 1 draw",
            x_wgan_man,
            (time.perf_counter() - t0) * 1000.0,
        )

    # 9. Untrained manifold generator ablation
    seed_everything(seed + 999)
    gen_untrained = PolymorphicGenerator(
        feature_dim=100,
        latent_dim=32,
        hidden_dims=[128, 256, 128],
        max_eps=default_eps,
        mode="manifold",
    ).to(device)
    gen_untrained.eval()
    _sync(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        x_wgan_untrained = gen_untrained.mutate_raw(
            mal_detected,
            evaluator.norm_min,
            evaluator.scales,
            eps=default_eps,
            mode="manifold",
        )
    _sync(device)
    add_row(
        "WGAN-GP (Manifold, untrained)",
        "Same forward, random weights",
        x_wgan_untrained,
        (time.perf_counter() - t0) * 1000.0,
    )
    seed_everything(seed)

    md_table = format_markdown_table(headers, summary_rows)
    print(f"\n### {suite_title}\n")
    print(md_table + "\n")
    return md_table, summary_rows


def run_evaluate_all(
    args: argparse.Namespace,
    torch_model: PyTorchKitNET,
    theta: float,
    feats_mmap: np.ndarray,
    evaluator: PipelineEvaluator,
    device: torch.device,
) -> None:
    """Executes a full comparative evaluation across all paradigms and traffic
    regimes."""
    print("\n" + "=" * 70)
    print("MODE: Comprehensive Comparative Evaluation (All Paradigms)")
    print("=" * 70)

    # 1. Clean Baseline Detection
    _ = run_evaluate_clean(args, torch_model, theta, feats_mmap, evaluator)

    combined_markdown = ["# Final Comparative Benchmark Summary\n"]

    phases = []
    if args.attack_phase in ["scanning", "both"]:
        phases.append(
            ("Phase A: Reconnaissance / Scanning Traffic (Early Attack)", "scanning")
        )
    if args.attack_phase in ["flood", "both"]:
        phases.append(("Phase B: Volumetric Flooding Traffic (Heavy Attack)", "flood"))

    for title, phase_key in phases:
        mal_subset = get_detected_attack_sample(
            feats_mmap, evaluator, args.samples, device, phase=phase_key
        )
        md_table, _ = evaluate_attack_suite(
            title,
            mal_subset,
            torch_model,
            evaluator,
            device,
            retries=args.retries,
            refine_steps=args.refine_steps,
            default_eps=args.eps,
            deepfool_max_eps=args.deepfool_max_eps,
            deepfool_steps=args.deepfool_steps,
            deepfool_step=args.deepfool_step,
            checkpoint_dir=args.checkpoint_dir,
            seed=args.seed,
        )
        combined_markdown.append(f"## {title}\n\n{md_table}\n")

    # Save to disk
    os.makedirs(args.output_dir, exist_ok=True)
    table_file = os.path.join(args.output_dir, "evaluation_table.md")
    with open(table_file, "w") as f:
        f.write("\n".join(combined_markdown))
    print(f"[+] Saved comparative benchmark tables to: {table_file}")


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = get_device(args.device)

    if args.mode == "train_wgan":
        run_train_wgan(args, device)
        return

    # Load components for evaluation modes
    torch_model, theta, feats_mmap, evaluator = load_pipeline_components(args, device)

    if args.mode == "evaluate_clean":
        run_evaluate_clean(args, torch_model, theta, feats_mmap, evaluator)

    elif args.mode == "fgsm_sweep":
        phase = args.attack_phase if args.attack_phase != "both" else "flood"
        mal_detected = get_detected_attack_sample(
            feats_mmap, evaluator, args.samples, device, phase=phase
        )
        run_fgsm_sweep(args, torch_model, evaluator, mal_detected)

    elif args.mode == "pgd_attack":
        phase = args.attack_phase if args.attack_phase != "both" else "flood"
        mal_detected = get_detected_attack_sample(
            feats_mmap, evaluator, args.samples, device, phase=phase
        )
        run_pgd_attack(args, torch_model, evaluator, mal_detected)

    elif args.mode == "evaluate_all":
        run_evaluate_all(args, torch_model, theta, feats_mmap, evaluator, device)


if __name__ == "__main__":
    main()
