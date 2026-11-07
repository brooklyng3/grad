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
        "--batch_size",
        type=int,
        default=256,
        help="Mini-batch size for GPU inference.",
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
) -> torch.Tensor:
    """Extracts a slice of confirmed detected attack packets (RMSE > theta)."""
    print(f"[*] Extracting {num_samples} confirmed detected attack samples...")
    candidate_indices = np.linspace(
        ATTACK_START, len(feats_mmap) - 1, num_samples * 2, dtype=int
    )
    candidate_feats = torch.from_numpy(
        np.array(feats_mmap[candidate_indices], dtype=np.float32)
    )

    scores = evaluator.score_batches(candidate_feats)
    detected_mask = scores > evaluator.theta
    detected_feats = candidate_feats[detected_mask]

    if len(detected_feats) < num_samples:
        print(
            f"[!] Warning: Found {len(detected_feats)} detected packets "
            f"(requested {num_samples})"
        )
        selected = detected_feats
    else:
        selected = detected_feats[:num_samples]

    print(
        f"[*] Selected {len(selected)} detected packets. Initial mean RMSE: "
        f"{float(np.mean(evaluator.score_batches(selected))):.4f}"
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
    torch_model: PyTorchKitNET,
    evaluator: PipelineEvaluator,
    feats_mmap: np.ndarray,
    device: torch.device,
) -> Dict[str, Any]:
    """Executes closed-loop WGAN-GP feedback training."""
    print("\n" + "=" * 70)
    print("MODE: Closed-Loop WGAN-GP Adversarial Feedback Training")
    print("=" * 70)

    # Benign pool (10,000 post-grace)
    benign_pool = torch.from_numpy(
        np.array(feats_mmap[BENIGN_EVAL_START : BENIGN_EVAL_START + 10000], np.float32)
    )
    # Malicious detected pool
    stride = max(1, TOTAL_ATTACKS // 20000)
    candidate_indices = np.arange(ATTACK_START, len(feats_mmap), stride)
    candidates = torch.from_numpy(
        np.array(feats_mmap[candidate_indices], dtype=np.float32)
    )
    scores = evaluator.score_batches(candidates)
    mal_pool = candidates[scores > evaluator.theta][:10000]

    print(f"[*] Training pools: Benign={len(benign_pool)}, Malicious={len(mal_pool)}")

    feedback_loop = AdversarialFeedbackLoop(
        defender_model=torch_model,
        theta=evaluator.theta,
        benign_data=benign_pool,
        malicious_data=mal_pool,
        latent_dim=32,
        batch_size=128,
        lr_g=1e-4,
        lr_c=1e-4,
        lambda_gp=10.0,
        lambda_evade=5.0,
        lambda_dist=0.5,
        eps_linf=2.0,
        device=device,
    )

    history = feedback_loop.train(num_steps=100, n_critic=5, log_interval=25)

    # Save Checkpoints
    ckpt_dir = "data/attack_samples"
    os.makedirs(ckpt_dir, exist_ok=True)
    g_path = os.path.join(ckpt_dir, "wgan_generator.pt")
    c_path = os.path.join(ckpt_dir, "wgan_critic.pt")
    torch.save(feedback_loop.generator.state_dict(), g_path)
    torch.save(feedback_loop.critic.state_dict(), c_path)
    print(f"[+] Generator saved to: {g_path}")
    print(f"[+] Critic saved to: {c_path}")

    # Evaluate trained generator
    test_mal = mal_pool[: args.samples].to(device)
    with torch.no_grad():
        feedback_loop.generator.eval()
        z = torch.randn(len(test_mal), 32, device=device)
        norm_min, norm_max, scales = torch_model.get_global_feature_bounds(
            as_torch=True
        )
        norm_mal = (test_mal - norm_min) / torch.clamp(scales, min=1e-8)
        norm_adv = feedback_loop.generator(norm_mal, z)
        x_adv = torch.clamp(norm_adv * scales + norm_min, min=0.0)

    res = evaluator.evaluate_evasion(
        x_orig=test_mal,
        x_adv=x_adv,
        num_pre_evaded=NUM_PRE_EVADED,
        total_attack_population=TOTAL_ATTACKS,
    )

    print(
        f"[*] Final WGAN Evasion: Conditional ASR = {res['conditional_asr_pct']:.2f}%, "
        f"Overall ER = {res['overall_evasion_rate_pct']:.2f}%"
    )

    out_file = os.path.join(args.output_dir, "wgan_training_summary.json")
    export_metrics_json({"history": history, "eval_result": res}, out_file)
    return res


def run_evaluate_all(
    args: argparse.Namespace,
    torch_model: PyTorchKitNET,
    theta: float,
    feats_mmap: np.ndarray,
    evaluator: PipelineEvaluator,
    mal_detected: torch.Tensor,
    device: torch.device,
) -> None:
    """Executes a full comparative evaluation across all paradigms."""
    print("\n" + "=" * 70)
    print("MODE: Comprehensive Comparative Evaluation (All Paradigms)")
    print("=" * 70)

    # 1. Clean Baseline Detection
    _ = run_evaluate_clean(args, torch_model, theta, feats_mmap, evaluator)

    # 2. Benchmarking Table
    summary_rows = []
    headers = [
        "Paradigm / Attack",
        "Perturbation Constraint",
        "Mean Attack RMSE",
        "Conditional ASR (%)",
        "Overall Evasion Rate (%)",
        "Mean Linf",
    ]

    # Clean unperturbed
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
        ]
    )

    # FGSM eps=0.05
    x_fgsm_05, _ = fgsm_attack(
        model=torch_model,
        x=mal_detected,
        eps=0.05,
        scales=evaluator.scales,
    )
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
        ]
    )

    # FGSM eps=0.25
    x_fgsm_25, _ = fgsm_attack(
        model=torch_model,
        x=mal_detected,
        eps=0.25,
        scales=evaluator.scales,
    )
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
        ]
    )

    # Multi-Step PGD (50 steps, eps=2.0)
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
    pgd_eval = evaluator.evaluate_evasion(
        x_orig=mal_detected,
        x_adv=x_pgd,
        num_pre_evaded=NUM_PRE_EVADED,
        total_attack_population=TOTAL_ATTACKS,
    )
    summary_rows.append(
        [
            "Multi-Step PGD",
            "50 steps, eps = 2.0 * s",
            pgd_eval["rmse_metrics"]["adversarial_mean"],
            f"{pgd_eval['conditional_asr_pct']:.2f}%",
            f"{pgd_eval['overall_evasion_rate_pct']:.2f}%",
            pgd_eval["perturbation_distances"]["mean_linf"],
        ]
    )

    # Manifold Traffic-Shaping (50 steps)
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
    manifold_eval = evaluator.evaluate_evasion(
        x_orig=mal_detected,
        x_adv=x_manifold,
        num_pre_evaded=NUM_PRE_EVADED,
        total_attack_population=TOTAL_ATTACKS,
    )
    summary_rows.append(
        [
            "Manifold Traffic-Shaping",
            "50 steps, [min, max]",
            manifold_eval["rmse_metrics"]["adversarial_mean"],
            f"{manifold_eval['conditional_asr_pct']:.2f}%",
            f"{manifold_eval['overall_evasion_rate_pct']:.2f}%",
            manifold_eval["perturbation_distances"]["mean_linf"],
        ]
    )

    # WGAN-GP Generator (if checkpoint exists)
    ckpt_path = "data/attack_samples/wgan_generator.pt"
    if os.path.exists(ckpt_path):
        gen = PolymorphicGenerator(
            feature_dim=100,
            latent_dim=32,
            hidden_dims=[128, 256, 128],
            max_eps=2.0,
        ).to(device)
        gen.load_state_dict(
            torch.load(ckpt_path, map_location=device, weights_only=True)
        )
        gen.eval()
        with torch.no_grad():
            z = torch.randn(len(mal_detected), 32, device=device)
            norm_min, norm_max, scales = torch_model.get_global_feature_bounds(
                as_torch=True
            )
            norm_mal = (mal_detected - norm_min) / torch.clamp(scales, min=1e-8)
            norm_adv = gen(norm_mal, z)
            x_wgan = torch.clamp(norm_adv * scales + norm_min, min=0.0)

        wgan_eval = evaluator.evaluate_evasion(
            x_orig=mal_detected,
            x_adv=x_wgan,
            num_pre_evaded=NUM_PRE_EVADED,
            total_attack_population=TOTAL_ATTACKS,
        )
        summary_rows.append(
            [
                "Polymorphic WGAN-GP",
                "eps = 2.0 * s, closed loop",
                wgan_eval["rmse_metrics"]["adversarial_mean"],
                f"{wgan_eval['conditional_asr_pct']:.2f}%",
                f"{wgan_eval['overall_evasion_rate_pct']:.2f}%",
                wgan_eval["perturbation_distances"]["mean_linf"],
            ]
        )

    md_table = format_markdown_table(headers, summary_rows)
    print("\n" + "=" * 70)
    print("FINAL COMPARATIVE BENCHMARK SUMMARY")
    print("=" * 70)
    print(md_table + "\n")

    # Save to disk
    os.makedirs(args.output_dir, exist_ok=True)
    table_file = os.path.join(args.output_dir, "evaluation_table.md")
    with open(table_file, "w") as f:
        f.write("# Final Comparative Benchmark Summary\n\n" + md_table + "\n")
    print(f"[+] Saved comparative markdown table to: {table_file}")


def main() -> None:
    args = parse_args()
    device = get_device(args.device)

    # Load components
    torch_model, theta, feats_mmap, evaluator = load_pipeline_components(args, device)

    if args.mode == "evaluate_clean":
        run_evaluate_clean(args, torch_model, theta, feats_mmap, evaluator)

    elif args.mode == "fgsm_sweep":
        mal_detected = get_detected_attack_sample(
            feats_mmap, evaluator, args.samples, device
        )
        run_fgsm_sweep(args, torch_model, evaluator, mal_detected)

    elif args.mode == "pgd_attack":
        mal_detected = get_detected_attack_sample(
            feats_mmap, evaluator, args.samples, device
        )
        run_pgd_attack(args, torch_model, evaluator, mal_detected)

    elif args.mode == "train_wgan":
        run_train_wgan(args, torch_model, evaluator, feats_mmap, device)

    elif args.mode == "evaluate_all":
        mal_detected = get_detected_attack_sample(
            feats_mmap, evaluator, args.samples, device
        )
        run_evaluate_all(
            args,
            torch_model,
            theta,
            feats_mmap,
            evaluator,
            mal_detected,
            device,
        )


if __name__ == "__main__":
    main()
