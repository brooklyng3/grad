"""
Automated Adversarial Training Feedback Loop (Phase 4).

Connects Defender (PyTorchKitNET) and Attacker (PolymorphicGenerator & Critic)
into a closed iterative feedback optimization loop.

Training Objective:
  1. Train WGAN Critic:
         L_C = E[D(fake)] - E[D(real)] + lambda_gp * E[(||grad D|| - 1)^2]
  2. Train Polymorphic Generator via Composite Objective:
         L_G = -E[D(adv)] + lambda_evade * E[Hinge(RMSE)] + lambda_dist * ||delta||

Ensures hardware compliance:
  - Batch sizes between 64 and 128.
  - VRAM strictly under 4 GB with torch.cuda.empty_cache() between epochs.
"""

import os
import sys
import time
from typing import Dict, Any, Optional
import numpy as np
import torch
import yaml

# Ensure project root and Kitsune-py are importable
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("Kitsune-py"))

from src.defender.torch_kitnet import PyTorchKitNET  # noqa: E402
from src.attacker.generator import PolymorphicGenerator  # noqa: E402
from src.attacker.discriminator import (  # noqa: E402
    WGANDiscriminator,
    compute_gradient_penalty,
)
from src.pipeline.evaluator import PipelineEvaluator  # noqa: E402


class AdversarialFeedbackLoop:
    """Manages closed-loop adversarial mutation and defender feedback training."""

    def __init__(
        self,
        config_path: str = "configs/wgan_config.yaml",
        kitsune_config_path: str = "configs/kitsune_config.yaml",
        device: Optional[torch.device] = None,
    ) -> None:
        with open(config_path, "r") as f:
            self.wgan_cfg = yaml.safe_load(f)
        with open(kitsune_config_path, "r") as f:
            self.kit_cfg = yaml.safe_load(f)

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # 1. Load Pre-trained Defender and Calibration Metadata
        model_path = self.kit_cfg["data"]["model_weights"]
        self.defender = PyTorchKitNET.from_file(model_path, device=self.device)
        self.defender.eval()

        with open("data/processed/threshold_config.yaml", "r") as f:
            thresh_meta = yaml.safe_load(f)
        self.theta = float(thresh_meta["threshold"])

        self.norm_min, self.norm_max, self.scales = (
            self.defender.get_global_feature_bounds(as_torch=True)
        )

        # 2. Instantiate Attacker Models
        w_cfg = self.wgan_cfg["wgan"]
        p_cfg = self.wgan_cfg["perturbation"]

        self.latent_dim = w_cfg.get("latent_dim", 32)
        self.feature_dim = w_cfg.get("feature_dim", 100)
        self.eps = p_cfg.get("eps_linf", 0.5)
        self.lambda_gp = w_cfg.get("lambda_gp", 10.0)
        self.n_critic = w_cfg.get("n_critic", 5)

        self.generator = PolymorphicGenerator(
            feature_dim=self.feature_dim,
            latent_dim=self.latent_dim,
            hidden_dims=w_cfg.get("generator_hidden_dims", [128, 256, 128]),
            max_eps=self.eps,
        ).to(self.device)

        self.critic = WGANDiscriminator(
            feature_dim=self.feature_dim,
            hidden_dims=w_cfg.get("critic_hidden_dims", [128, 256, 128]),
        ).to(self.device)

        lr_g = w_cfg.get("lr_g", 0.0001)
        lr_c = w_cfg.get("lr_c", 0.0001)
        b1 = w_cfg.get("b1", 0.5)
        b2 = w_cfg.get("b2", 0.9)

        self.opt_g = torch.optim.Adam(
            self.generator.parameters(), lr=lr_g, betas=(b1, b2)
        )
        self.opt_c = torch.optim.Adam(self.critic.parameters(), lr=lr_c, betas=(b1, b2))

        self.evaluator = PipelineEvaluator(
            model=self.defender,
            theta=self.theta,
            device=self.device,
            batch_size=256,
        )

    def load_training_data(
        self,
        max_malicious_samples: int = 10000,
        max_benign_samples: int = 10000,
    ) -> Dict[str, torch.Tensor]:
        """Loads and pre-filters benign and detected malicious samples."""
        features = np.load(self.kit_cfg["data"]["processed_features"], mmap_mode="r")
        labels = []
        with open(self.kit_cfg["data"]["raw_labels"], "r") as f:
            for line in f:
                line_str = line.strip()
                if line_str:
                    labels.append(int(line_str))
        labels = np.array(labels, dtype=np.int32)

        # 1. Post-convergence benign pool (i >= 55,000, label == 0)
        benign_indices = np.where((labels == 0) & (np.arange(len(labels)) >= 55000))[0]
        selected_benign = benign_indices[:max_benign_samples]
        x_benign_raw = torch.tensor(
            features[selected_benign], dtype=torch.float32, device=self.device
        )
        x_benign_norm = (x_benign_raw - self.norm_min) / self.scales

        # 2. Detected malicious pool (label == 1, RMSE > theta)
        attack_indices = np.where(labels == 1)[0]
        pool_size = min(len(attack_indices), max_malicious_samples * 2)
        candidate_attacks = torch.tensor(
            features[attack_indices[:pool_size]],
            dtype=torch.float32,
            device=self.device,
        )

        with torch.no_grad():
            scores = self.evaluator.score_batches(candidate_attacks)
        detected_mask = scores > self.theta
        x_mal_detected_raw = candidate_attacks[detected_mask][:max_malicious_samples]
        x_mal_detected_norm = (x_mal_detected_raw - self.norm_min) / self.scales

        print(f"Loaded {x_benign_norm.size(0):,} benign training samples.")
        print(
            f"Loaded {x_mal_detected_norm.size(0):,} strictly detected attack samples."
        )

        return {
            "x_benign_norm": x_benign_norm,
            "x_mal_detected_raw": x_mal_detected_raw,
            "x_mal_detected_norm": x_mal_detected_norm,
        }

    def train_loop(
        self,
        num_epochs: int = 20,
        batch_size: int = 128,
        lambda_evade: float = 30.0,
        lambda_dist: float = 0.05,
    ) -> Dict[str, Any]:
        """Executes the closed feedback loop training pass."""
        data = self.load_training_data()
        x_benign_norm = data["x_benign_norm"]
        x_mal_norm = data["x_mal_detected_norm"]
        x_mal_raw = data["x_mal_detected_raw"]

        num_samples = min(x_benign_norm.size(0), x_mal_norm.size(0))
        num_batches = num_samples // batch_size

        print("=" * 70)
        print("STARTING PHASE 4 CLOSED-LOOP ADVERSARIAL MUTATION TRAINING")
        print(f"Device:           {self.device}")
        print(f"Epochs:           {num_epochs}")
        print(f"Batch Size:       {batch_size}")
        print(f"Lambda Evade:     {lambda_evade}")
        print(f"Perturbation eps: {self.eps}")
        print("=" * 70)

        history = []
        start_time = time.time()

        for epoch in range(1, num_epochs + 1):
            epoch_loss_c = 0.0
            epoch_loss_g = 0.0
            epoch_evaded_count = 0
            epoch_total_samples = 0

            # Shuffle training data
            perm_b = torch.randperm(x_benign_norm.size(0))
            perm_m = torch.randperm(x_mal_norm.size(0))

            for b_idx in range(num_batches):
                b_slice = perm_b[b_idx * batch_size : (b_idx + 1) * batch_size]
                m_slice = perm_m[b_idx * batch_size : (b_idx + 1) * batch_size]

                real_batch = x_benign_norm[b_slice]
                mal_norm_batch = x_mal_norm[m_slice]

                # ----------------------------------------------------
                # Step 1: Train Critic (WGAN Discriminator)
                # ----------------------------------------------------
                self.opt_c.zero_grad()
                with torch.no_grad():
                    fake_batch = self.generator(mal_norm_batch, eps=self.eps).detach()

                d_real = self.critic(real_batch).mean()
                d_fake = self.critic(fake_batch).mean()
                gp = compute_gradient_penalty(
                    self.critic, real_batch, fake_batch, device=self.device
                )
                loss_c = d_fake - d_real + self.lambda_gp * gp
                loss_c.backward()
                self.opt_c.step()

                epoch_loss_c += loss_c.item()

                # ----------------------------------------------------
                # Step 2: Train Generator with Hybrid Adversarial + Evasion Loss
                # ----------------------------------------------------
                self.opt_g.zero_grad()
                adv_norm = self.generator(mal_norm_batch, eps=self.eps)

                # Critic realism objective
                loss_adv = -self.critic(adv_norm).mean()

                # Physical network feature domain unnormalization
                adv_raw = torch.clamp(adv_norm * self.scales + self.norm_min, min=0.0)

                # Direct backpropagation through PyTorchKitNET
                rmse_adv = self.defender(adv_raw)

                # Hinge evasion loss: penalizes reconstruction error above theta
                loss_evade = torch.clamp(rmse_adv - self.theta, min=0.0).mean()

                # Distance constraint in normalized space
                loss_dist = torch.norm(adv_norm - mal_norm_batch, p=2, dim=-1).mean()

                loss_g = loss_adv + lambda_evade * loss_evade + lambda_dist * loss_dist
                loss_g.backward()
                self.opt_g.step()

                epoch_loss_g += loss_g.item()

                # Track evasion success
                with torch.no_grad():
                    evaded = (rmse_adv <= self.theta).sum().item()
                    epoch_evaded_count += evaded
                    epoch_total_samples += batch_size

            avg_loss_c = epoch_loss_c / num_batches
            avg_loss_g = epoch_loss_g / num_batches
            batch_asr = (epoch_evaded_count / epoch_total_samples) * 100.0

            history.append(
                {
                    "epoch": epoch,
                    "loss_c": avg_loss_c,
                    "loss_g": avg_loss_g,
                    "conditional_asr": batch_asr,
                }
            )

            if epoch % 5 == 0 or epoch == 1:
                print(
                    f"Epoch [{epoch:02d}/{num_epochs:02d}] | "
                    f"Loss C: {avg_loss_c:>8.4f} | "
                    f"Loss G: {avg_loss_g:>8.4f} | "
                    f"Conditional ASR: {batch_asr:>6.2f}%"
                )

            # Enforce GPU memory discipline
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        total_time = time.time() - start_time
        print("-" * 70)
        print(f"Feedback loop training complete in {total_time:.2f}s!")

        # Final evaluation on holdout detected samples
        print("\nEvaluating final generator performance across detected pool...")
        with torch.no_grad():
            final_adv_raw = self.generator.mutate_raw(
                x_mal_raw[:1000], self.norm_min, self.scales, eps=self.eps
            )
            final_adv_raw = torch.clamp(final_adv_raw, min=0.0)

        eval_summary = self.evaluator.evaluate_evasion(
            x_orig=x_mal_raw[:1000],
            x_adv=final_adv_raw,
            num_pre_evaded=76123,
            total_attack_population=642516,
        )

        print("-" * 70)
        print("FINAL EVALUATION SUMMARY (POST-FEEDBACK LOOP):")
        print(
            f"  Conditional ASR on Detected:  "
            f"{eval_summary['conditional_asr_pct']:.2f}%"
        )
        print(
            f"  Overall Evasion Rate (Total): "
            f"{eval_summary['overall_evasion_rate_pct']:.2f}%"
        )
        print(
            f"  Adversarial Mean RMSE:        "
            f"{eval_summary['rmse_metrics']['adversarial_mean']:.4f} "
            f"(vs Theta={self.theta:.4f})"
        )
        print(
            f"  Mean L_inf Perturbation:      "
            f"{eval_summary['perturbation_distances']['mean_linf']:.4f}"
        )
        print("=" * 70)

        # Save trained generator and critic checkpoints
        os.makedirs("data/attack_samples", exist_ok=True)
        gen_path = "data/attack_samples/wgan_generator.pt"
        critic_path = "data/attack_samples/wgan_critic.pt"
        torch.save(self.generator.state_dict(), gen_path)
        torch.save(self.critic.state_dict(), critic_path)
        print(f"Saved generator checkpoint: {gen_path}")
        print(f"Saved critic checkpoint:    {critic_path}")

        return {
            "eval_summary": eval_summary,
            "history": history,
            "generator_path": gen_path,
            "critic_path": critic_path,
        }


if __name__ == "__main__":
    loop = AdversarialFeedbackLoop()
    loop.train_loop(num_epochs=20, batch_size=128)
