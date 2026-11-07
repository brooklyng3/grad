#!/usr/bin/env python3
"""
End-to-End Adversarial ML Evaluation Pipeline CLI (Phase 5).

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
from src.attacker.perturbations import fgsm_attack, pgd_attack  # noqa: E402
from src.attacker.generator import PolymorphicGenerator  # noqa: E402
from src.pipeline.evaluator import PipelineEvaluator  # noqa: E402
from src.pipeline.feedback_loop import AdversarialFeedbackLoop  # noqa: E402
from src.utils.metrics import (  # noqa: E402
    compute_detection_metrics,
    compute_roc_auc,
    format_markdown_table,
    export_metrics_json,
    export_metrics_csv,
)

BENIGN_EVAL_START = 55000
BENIGN_EVAL_END = 121621
ATTACK_START = 121621
TOTAL_ATTACKS = 642516
NUM_PRE_EVADED = 76123  # Naturally evading baseline attack packets


def parse_args() -> argparse.Namespace:
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
            theta = float(tcfg.get("threshold", 0.453419))
    else:
        theta = 0.453419
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
    return torch_model, theta, feats_mmap, evaluator


def get_detected_attack_sample(
    feats_mmap: np.ndarray,
    evaluator: PipelineEvaluator,
    num_samples: int,
    device: torch.device,
    phase: str = "flood",
) -> torch.Tensor:
    """Extracts confirmed detected attack packets (RMSE > theta).

    Args:
        feats_mmap: Memory-mapped features array.
        evaluator: PipelineEvaluator engine.
        num_samples: Target number of detected packets.
        device: Torch device.
        phase: 'scanning' (early reconnaissance phase) or 'flood' (volumetric flood).
    """
    if phase == "scanning":
        candidate_indices = np.arange(
            ATTACK_START, min(ATTACK_START + num_samples * 10, len(feats_mmap))
        )
    else:
        candidate_indices = np.linspace(
            ATTACK_START + 10000, len(feats_mmap) - 1, num_samples * 2, dtype=int
        )

    candidate_feats = torch.from_numpy(
        np.array(feats_mmap[candidate_indices], dtype=np.float32)
    )

    scores = evaluator.score_batches(candidate_feats)
    detected_mask = scores > evaluator.theta
    detected_feats = candidate_feats[detected_mask]

    selected = (
        detected_feats[:num_samples]
        if len(detected_feats) >= num_samples
        else detected_feats
    )
    mean_rmse = float(np.mean(evaluator.score_batches(selected)))
    print(
        f"[*] [{phase.upper()}] Selected {len(selected)} detected packets. "
        f"Initial mean RMSE: {mean_rmse:.4f}"
    )
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

    # Benign validation slice (post-grace: 55,000 to 121,621)
    benign_feats = torch.from_numpy(
        np.array(feats_mmap[BENIGN_EVAL_START:BENIGN_EVAL_END], dtype=np.float32)
    )
    print(f"[*] Benign evaluation subset size: {len(benign_feats):,} packets")

    # Sample attack slice (up to 50,000 for high-confidence metrics)
    attack_stride = max(1, TOTAL_ATTACKS // 50000)
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
            num_pre_evaded=NUM_PRE_EVADED,
            total_attack_population=TOTAL_ATTACKS,
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
            num_pre_evaded=NUM_PRE_EVADED,
            total_attack_population=TOTAL_ATTACKS,
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
    )

    res = feedback_loop.train_loop(
        num_epochs=10,
        batch_size=128,
        lambda_evade=50.0 if args.manifold_shaping else 30.0,
        lambda_dist=0.01 if args.manifold_shaping else 0.05,
    )

    # Save Checkpoint
    ckpt_dir = "data/attack_samples"
    os.makedirs(ckpt_dir, exist_ok=True)
    if args.manifold_shaping:
        g_path = os.path.join(ckpt_dir, "wgan_manifold_generator.pt")
    else:
        g_path = os.path.join(ckpt_dir, "wgan_generator.pt")
    torch.save(feedback_loop.generator.state_dict(), g_path)
    print(f"[+] Generator saved to: {g_path}")

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


def evaluate_attack_suite(
    suite_title: str,
    mal_detected: torch.Tensor,
    torch_model: PyTorchKitNET,
    evaluator: PipelineEvaluator,
    device: torch.device,
) -> Tuple[str, List[List[Any]]]:
    """Evaluates all adversarial paradigms on a given detected malicious subset."""
    summary_rows = []
    headers = [
        "Paradigm / Attack",
        "Perturbation Constraint",
        "Mean Attack RMSE",
        "Conditional ASR (%)",
        "Overall Evasion Rate (%)",
        "Mean Linf",
        "Latency (ms / 1k pkts)",
    ]

    # Warmup GPU (both forward inference and autograd backward kernels)
    with torch.no_grad():
        _ = torch_model(mal_detected[:10])
    _warm_x = mal_detected[:2].clone().detach().requires_grad_(True)
    _warm_loss = torch_model(_warm_x).sum()
    _warm_loss.backward()
    del _warm_x, _warm_loss
    if device.type == "cuda":
        torch.cuda.synchronize()

    # 1. Clean unperturbed
    clean_eval = evaluator.evaluate_evasion(
        x_orig=mal_detected,
        x_adv=mal_detected,
        num_pre_evaded=NUM_PRE_EVADED,
        total_attack_population=TOTAL_ATTACKS,
    )
    summary_rows.append(
        [
            "Clean Baseline",
            "None (Unperturbed)",
            clean_eval["rmse_metrics"]["adversarial_mean"],
            f"{clean_eval['conditional_asr_pct']:.2f}%",
            f"{clean_eval['overall_evasion_rate_pct']:.2f}%",
            0.0000,
            "0.0 ms",
        ]
    )

    # 2. FGSM eps=0.05
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    x_fgsm_05, _ = fgsm_attack(
        model=torch_model,
        x=mal_detected,
        eps=0.05,
        scales=evaluator.scales,
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_fgsm_05 = (time.perf_counter() - t0) * 1000.0

    fgsm_05_eval = evaluator.evaluate_evasion(
        x_orig=mal_detected,
        x_adv=x_fgsm_05,
        num_pre_evaded=NUM_PRE_EVADED,
        total_attack_population=TOTAL_ATTACKS,
    )
    summary_rows.append(
        [
            "Single-Step FGSM",
            "eps = 0.05 * s",
            fgsm_05_eval["rmse_metrics"]["adversarial_mean"],
            f"{fgsm_05_eval['conditional_asr_pct']:.2f}%",
            f"{fgsm_05_eval['overall_evasion_rate_pct']:.2f}%",
            fgsm_05_eval["perturbation_distances"]["mean_linf"],
            f"{t_fgsm_05:.1f} ms",
        ]
    )

    # 3. FGSM eps=0.25
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    x_fgsm_25, _ = fgsm_attack(
        model=torch_model,
        x=mal_detected,
        eps=0.25,
        scales=evaluator.scales,
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_fgsm_25 = (time.perf_counter() - t0) * 1000.0

    fgsm_25_eval = evaluator.evaluate_evasion(
        x_orig=mal_detected,
        x_adv=x_fgsm_25,
        num_pre_evaded=NUM_PRE_EVADED,
        total_attack_population=TOTAL_ATTACKS,
    )
    summary_rows.append(
        [
            "Single-Step FGSM",
            "eps = 0.25 * s",
            fgsm_25_eval["rmse_metrics"]["adversarial_mean"],
            f"{fgsm_25_eval['conditional_asr_pct']:.2f}%",
            f"{fgsm_25_eval['overall_evasion_rate_pct']:.2f}%",
            fgsm_25_eval["perturbation_distances"]["mean_linf"],
            f"{t_fgsm_25:.1f} ms",
        ]
    )

    # 4. Multi-Step PGD (50 steps, Box eps=2.0)
    if device.type == "cuda":
        torch.cuda.synchronize()
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
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_pgd = (time.perf_counter() - t0) * 1000.0

    pgd_eval = evaluator.evaluate_evasion(
        x_orig=mal_detected,
        x_adv=x_pgd,
        num_pre_evaded=NUM_PRE_EVADED,
        total_attack_population=TOTAL_ATTACKS,
    )
    summary_rows.append(
        [
            "Multi-Step PGD (Box)",
            "50 steps, eps = 2.0 * s",
            pgd_eval["rmse_metrics"]["adversarial_mean"],
            f"{pgd_eval['conditional_asr_pct']:.2f}%",
            f"{pgd_eval['overall_evasion_rate_pct']:.2f}%",
            pgd_eval["perturbation_distances"]["mean_linf"],
            f"{t_pgd:.1f} ms",
        ]
    )

    # 5. Multi-Step PGD (50 steps, Manifold Shaping)
    if device.type == "cuda":
        torch.cuda.synchronize()
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
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_manifold = (time.perf_counter() - t0) * 1000.0

    manifold_eval = evaluator.evaluate_evasion(
        x_orig=mal_detected,
        x_adv=x_manifold,
        num_pre_evaded=NUM_PRE_EVADED,
        total_attack_population=TOTAL_ATTACKS,
    )
    summary_rows.append(
        [
            "Multi-Step PGD (Manifold)",
            "50 steps, [min, max]",
            manifold_eval["rmse_metrics"]["adversarial_mean"],
            f"{manifold_eval['conditional_asr_pct']:.2f}%",
            f"{manifold_eval['overall_evasion_rate_pct']:.2f}%",
            manifold_eval["perturbation_distances"]["mean_linf"],
            f"{t_manifold:.1f} ms",
        ]
    )

    # 6. Polymorphic WGAN-GP (Additive Box Generator)
    ckpt_additive = "data/attack_samples/wgan_generator.pt"
    if os.path.exists(ckpt_additive):
        gen_add = PolymorphicGenerator(
            feature_dim=100,
            latent_dim=32,
            hidden_dims=[128, 256, 128],
            max_eps=2.0,
            mode="additive",
        ).to(device)
        gen_add.load_state_dict(
            torch.load(ckpt_additive, map_location=device, weights_only=True)
        )
        gen_add.eval()

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            x_wgan_add = gen_add.mutate_raw(
                mal_detected, evaluator.norm_min, evaluator.scales, mode="additive"
            )
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_wgan_add = (time.perf_counter() - t0) * 1000.0

        wgan_add_eval = evaluator.evaluate_evasion(
            x_orig=mal_detected,
            x_adv=x_wgan_add,
            num_pre_evaded=NUM_PRE_EVADED,
            total_attack_population=TOTAL_ATTACKS,
        )
        summary_rows.append(
            [
                "Polymorphic WGAN-GP (Box)",
                "eps = 2.0 * s, closed loop",
                wgan_add_eval["rmse_metrics"]["adversarial_mean"],
                f"{wgan_add_eval['conditional_asr_pct']:.2f}%",
                f"{wgan_add_eval['overall_evasion_rate_pct']:.2f}%",
                wgan_add_eval["perturbation_distances"]["mean_linf"],
                f"{t_wgan_add:.1f} ms",
            ]
        )

    # 7. Polymorphic WGAN-GP (Manifold Shaping Generator)
    ckpt_manifold = "data/attack_samples/wgan_manifold_generator.pt"
    if os.path.exists(ckpt_manifold):
        gen_man = PolymorphicGenerator(
            feature_dim=100,
            latent_dim=32,
            hidden_dims=[128, 256, 128],
            max_eps=2.0,
            mode="manifold",
        ).to(device)
        gen_man.load_state_dict(
            torch.load(ckpt_manifold, map_location=device, weights_only=True)
        )
        gen_man.eval()

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            x_wgan_man = gen_man.mutate_raw(
                mal_detected, evaluator.norm_min, evaluator.scales, mode="manifold"
            )
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_wgan_man = (time.perf_counter() - t0) * 1000.0

        wgan_man_eval = evaluator.evaluate_evasion(
            x_orig=mal_detected,
            x_adv=x_wgan_man,
            num_pre_evaded=NUM_PRE_EVADED,
            total_attack_population=TOTAL_ATTACKS,
        )
        summary_rows.append(
            [
                "Polymorphic WGAN-GP (Manifold)",
                "Envelope Shaping, single pass",
                wgan_man_eval["rmse_metrics"]["adversarial_mean"],
                f"{wgan_man_eval['conditional_asr_pct']:.2f}%",
                f"{wgan_man_eval['overall_evasion_rate_pct']:.2f}%",
                wgan_man_eval["perturbation_distances"]["mean_linf"],
                f"{t_wgan_man:.1f} ms",
            ]
        )

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
            title, mal_subset, torch_model, evaluator, device
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
