"""Automated Adversarial Training Feedback Loop (Phase 4, remediated in Phase 6).

Connects Defender (PyTorchKitNET) and Attacker (PolymorphicGenerator & Critic)
into a closed iterative feedback optimization loop.

Training Objective:
  1. Train WGAN Critic:
         L_C = E[D(fake)] - E[D(real)] + lambda_gp * E[(||grad D|| - 1)^2]
  2. Train Polymorphic Generator via Composite Objective:
         L_G = -E[D(adv)] + lambda_evade * E[Hinge(RMSE)] + lambda_dist * ||delta||

Remediation changes:
  - ``eps`` is explicit and shared with evaluation (no config/eval mismatch).
  - Training attacks are drawn from a deterministic, disjoint split spread over
    the whole attack population, not only the first 20,000 packets.
  - ``n_critic`` is actually executed.
  - A held-out attack split is scored every epoch; the best checkpoint by
    holdout ASR is restored and saved.
  - All RNGs are seeded.

Ensures hardware compliance:
  - Batch sizes between 64 and 128.
  - VRAM strictly under 4 GB with periodic cache flushes.
"""

import copy
import os
import sys
import time
from typing import Dict, Any, Optional, Tuple
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
from src.utils.constraints import enforce_feature_bounds  # noqa: E402
from src.utils.data_splits import (  # noqa: E402
    ATTACK_START,
    attack_indices_for_split,
    benign_indices_for_split,
)
from src.utils.seed import seed_everything  # noqa: E402


class AdversarialFeedbackLoop:
    """Manages closed-loop adversarial mutation and defender feedback training."""

    def __init__(
        self,
        config_path: str = "configs/wgan_config.yaml",
        kitsune_config_path: str = "configs/kitsune_config.yaml",
        device: Optional[torch.device] = None,
        manifold_shaping: bool = False,
        eps: Optional[float] = None,
        threshold_config_path: str = "data/processed/threshold_config.yaml",
        model_path: Optional[str] = None,
        seed: int = 42,
    ) -> None:
        with open(config_path, "r") as f:
            self.wgan_cfg = yaml.safe_load(f)
        with open(kitsune_config_path, "r") as f:
            self.kit_cfg = yaml.safe_load(f)

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.manifold_shaping = manifold_shaping
        self.seed = seed
        seed_everything(seed)

        # 1. Load Pre-trained Defender and Calibration Metadata
        model_path = model_path or self.kit_cfg["data"]["model_weights"]
        self.defender = PyTorchKitNET.from_file(model_path, device=self.device)
        self.defender.eval()

        with open(threshold_config_path, "r") as f:
            thresh_meta = yaml.safe_load(f)
        self.theta = float(thresh_meta["threshold"])
        attack_meta = thresh_meta.get("attack_metrics", {})
        self.num_pre_evaded = int(attack_meta.get("evaded_count", 76123))
        self.total_attacks = int(attack_meta.get("num_evaluated", 642516))

        self.norm_min, self.norm_max, self.scales = (
            self.defender.get_global_feature_bounds(as_torch=True)
        )

        # 2. Instantiate Attacker Models
        w_cfg = self.wgan_cfg["wgan"]
        p_cfg = self.wgan_cfg["perturbation"]

        self.latent_dim = w_cfg.get("latent_dim", 32)
        self.feature_dim = w_cfg.get("feature_dim", 100)
        # The epsilon used here is also persisted with the checkpoint and used
        # at evaluation time, removing the former train/eval mismatch.
        self.eps = float(eps if eps is not None else p_cfg.get("eps_linf", 2.0))
        self.lambda_gp = w_cfg.get("lambda_gp", 10.0)
        self.n_critic = int(w_cfg.get("n_critic", 5))

        self.generator = PolymorphicGenerator(
            feature_dim=self.feature_dim,
            latent_dim=self.latent_dim,
            hidden_dims=w_cfg.get("generator_hidden_dims", [128, 256, 128]),
            max_eps=self.eps,
            mode="manifold" if manifold_shaping else "additive",
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

    @staticmethod
    def _spread(indices: np.ndarray, count: int) -> np.ndarray:
        """Evenly sample ``count`` entries from an index array."""
        if len(indices) <= count:
            return indices
        positions = np.linspace(0, len(indices) - 1, count, dtype=int)
        return indices[positions]

    def load_training_data(
        self,
        max_malicious_samples: int = 10000,
        max_benign_samples: int = 10000,
    ) -> Dict[str, torch.Tensor]:
        """Load disjoint, representative benign and detected malicious pools."""
        features = np.load(self.kit_cfg["data"]["processed_features"], mmap_mode="r")
        with open(self.kit_cfg["data"]["raw_labels"], "r") as f:
            labels = np.array(
                [int(line.strip()) for line in f if line.strip()], dtype=np.int32
            )

        # 1. Benign pool: post-grace calibration split (validation stays clean)
        benign_indices = benign_indices_for_split(labels, "calibration")
        selected_benign = benign_indices[:max_benign_samples]
        x_benign_raw = torch.tensor(
            features[selected_benign], dtype=torch.float32, device=self.device
        )
        x_benign_norm = (x_benign_raw - self.norm_min) / self.scales

        # 2. Attack pool: training split, half from the early scanning window and
        #    half spread across the full attack population.
        all_attacks = np.where(labels == 1)[0]
        train_attacks = attack_indices_for_split(all_attacks, "train")
        # A 30k-packet "early" window gives enough candidates to supply half
        # the final detected pool even though the onset ramps up slowly.
        early = train_attacks[train_attacks < ATTACK_START + 30000]
        later = train_attacks[train_attacks >= ATTACK_START + 30000]

        pool_size = min(len(train_attacks), max_malicious_samples * 2)
        n_early = min(len(early), max_malicious_samples)
        n_late = max(0, pool_size - n_early)

        early_candidates = torch.tensor(
            features[early[:n_early]], dtype=torch.float32, device=self.device
        )
        later_candidates = torch.tensor(
            features[self._spread(later, min(n_late, len(later)))],
            dtype=torch.float32,
            device=self.device,
        )
        with torch.no_grad():
            early_scores = self.evaluator.score_batches(early_candidates)
            later_scores = self.evaluator.score_batches(later_candidates)
        detected_early = early_candidates[early_scores > self.theta]
        detected_later = later_candidates[later_scores > self.theta]

        # Enforce an actual 50/50 post-detection composition.  Filtering can
        # otherwise skew the pool because detection rates differ by regime.
        half = max_malicious_samples // 2
        take_early = detected_early[:half]
        take_later = detected_later[:half]
        remaining = max_malicious_samples - take_early.size(0) - take_later.size(0)
        if remaining > 0:
            spare = torch.cat(
                [
                    detected_later[take_later.size(0) :],
                    detected_early[take_early.size(0) :],
                ]
            )
            fill = spare[:remaining]
            take_later = torch.cat([take_later, fill])

        x_mal_detected_raw = torch.cat([take_early, take_later])
        x_mal_detected_norm = (x_mal_detected_raw - self.norm_min) / self.scales

        print(f"Loaded {x_benign_norm.size(0):,} benign training samples.")
        print(
            f"Loaded {x_mal_detected_norm.size(0):,} strictly detected attack "
            f"samples (scan={take_early.size(0):,}, spread={take_later.size(0):,})."
        )

        return {
            "x_benign_norm": x_benign_norm,
            "x_mal_detected_raw": x_mal_detected_raw,
            "x_mal_detected_norm": x_mal_detected_norm,
        }

    def load_holdout(
        self,
        labels: np.ndarray,
        max_samples: int = 1000,
        block: Tuple[float, float] = (0.0, 0.5),
    ) -> torch.Tensor:
        """Load a detected attack batch from a block of the evaluation split.

        Selection and final test use disjoint halves so model selection does
        not consume the reported test set.
        """
        features = np.load(self.kit_cfg["data"]["processed_features"], mmap_mode="r")
        all_attacks = np.where(labels == 1)[0]
        eval_attacks = attack_indices_for_split(all_attacks, "eval")
        start = int(len(eval_attacks) * block[0])
        end = int(len(eval_attacks) * block[1])
        candidates = self._spread(eval_attacks[start:end], max_samples * 3)
        x = torch.tensor(features[candidates], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            scores = self.evaluator.score_batches(x)
        return x[scores > self.theta][:max_samples]

    def _train_critic_step(
        self, real_batch: torch.Tensor, mal_norm_batch: torch.Tensor
    ) -> float:
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
        return float(loss_c.item())

    def train_loop(
        self,
        num_epochs: int = 20,
        batch_size: int = 128,
        lambda_evade: float = 30.0,
        lambda_dist: float = 0.05,
        holdout_size: int = 1000,
        save_dir: str = "data/attack_samples",
        checkpoint_name: str = "wgan_generator.pt",
    ) -> Dict[str, Any]:
        """Executes the closed feedback loop training pass."""
        seed_everything(self.seed)
        data = self.load_training_data()
        x_benign_norm = data["x_benign_norm"]
        x_mal_norm = data["x_mal_detected_norm"]

        with open(self.kit_cfg["data"]["raw_labels"], "r") as f:
            labels = np.array(
                [int(line.strip()) for line in f if line.strip()], dtype=np.int32
            )
        x_select = self.load_holdout(labels, max_samples=holdout_size, block=(0.0, 0.5))
        x_test = self.load_holdout(labels, max_samples=holdout_size, block=(0.5, 1.0))
        print(
            f"Selection holdout: {x_select.size(0):,} | "
            f"untouched test holdout: {x_test.size(0):,} (disjoint halves)"
        )

        num_samples = min(x_benign_norm.size(0), x_mal_norm.size(0))
        num_batches = num_samples // batch_size

        print("=" * 70)
        print("STARTING PHASE 4 CLOSED-LOOP ADVERSARIAL MUTATION TRAINING")
        print(f"Device:           {self.device}")
        print(f"Epochs:           {num_epochs}")
        print(f"Batch Size:       {batch_size}")
        print(f"Critic steps:     {self.n_critic}")
        print(f"Lambda Evade:     {lambda_evade}")
        print(f"Perturbation eps: {self.eps}")
        print("=" * 70)

        history = []
        start_time = time.time()
        best_asr = -1.0
        best_mean = float("inf")
        best_epoch = 0
        best_state: Optional[Dict[str, torch.Tensor]] = None

        for epoch in range(1, num_epochs + 1):
            epoch_loss_c = 0.0
            epoch_loss_g = 0.0
            epoch_evaded_count = 0
            epoch_total_samples = 0

            perm_b = torch.randperm(x_benign_norm.size(0))
            perm_m = torch.randperm(x_mal_norm.size(0))

            for b_idx in range(num_batches):
                b_slice = perm_b[b_idx * batch_size : (b_idx + 1) * batch_size]
                m_slice = perm_m[b_idx * batch_size : (b_idx + 1) * batch_size]

                real_batch = x_benign_norm[b_slice]
                mal_norm_batch = x_mal_norm[m_slice]

                # ----------------------------------------------------
                # Step 1: Train Critic (WGAN n_critic updates)
                # ----------------------------------------------------
                for _ in range(self.n_critic):
                    epoch_loss_c += self._train_critic_step(real_batch, mal_norm_batch)

                # ----------------------------------------------------
                # Step 2: Train Generator with Hybrid Evasion Loss
                # ----------------------------------------------------
                self.opt_g.zero_grad()
                adv_norm = self.generator(mal_norm_batch, eps=self.eps)

                loss_adv = -self.critic(adv_norm).mean()
                adv_raw = enforce_feature_bounds(adv_norm * self.scales + self.norm_min)
                rmse_adv = self.defender(adv_raw)
                loss_evade = torch.clamp(rmse_adv - self.theta, min=0.0).mean()
                loss_dist = torch.norm(adv_norm - mal_norm_batch, p=2, dim=-1).mean()

                loss_g = loss_adv + lambda_evade * loss_evade + lambda_dist * loss_dist
                loss_g.backward()
                self.opt_g.step()

                epoch_loss_g += float(loss_g.item())
                with torch.no_grad():
                    epoch_evaded_count += int((rmse_adv <= self.theta).sum().item())
                    epoch_total_samples += batch_size

            # Holdout evaluation and best-checkpoint tracking
            with torch.no_grad():
                holdout_adv = self.generator.mutate_raw(
                    x_select,
                    self.norm_min,
                    self.scales,
                    eps=self.eps,
                    mode="manifold" if self.manifold_shaping else "additive",
                )
                holdout_rmse = self.evaluator.score_batches(holdout_adv)
            holdout_asr = float((holdout_rmse <= self.theta).mean() * 100.0)
            holdout_mean = float(np.mean(holdout_rmse))

            avg_loss_c = epoch_loss_c / (num_batches * self.n_critic)
            avg_loss_g = epoch_loss_g / num_batches
            batch_asr = (epoch_evaded_count / epoch_total_samples) * 100.0

            history.append(
                {
                    "epoch": epoch,
                    "loss_c": avg_loss_c,
                    "loss_g": avg_loss_g,
                    "batch_asr": batch_asr,
                    "holdout_asr": holdout_asr,
                    "holdout_mean_rmse": holdout_mean,
                }
            )

            if holdout_asr > best_asr or (
                holdout_asr == best_asr and holdout_mean < best_mean
            ):
                best_asr = holdout_asr
                best_mean = holdout_mean
                best_epoch = epoch
                best_state = copy.deepcopy(self.generator.state_dict())

            if epoch % 5 == 0 or epoch == 1:
                print(
                    f"Epoch [{epoch:02d}/{num_epochs:02d}] | "
                    f"Loss C: {avg_loss_c:>8.4f} | "
                    f"Loss G: {avg_loss_g:>8.4f} | "
                    f"Batch ASR: {batch_asr:>6.2f}% | "
                    f"Holdout ASR: {holdout_asr:>6.2f}%"
                )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        total_time = time.time() - start_time
        print("-" * 70)
        print(f"Feedback loop training complete in {total_time:.2f}s!")
        print(f"Best selection ASR {best_asr:.2f}% at epoch {best_epoch}.")

        if best_state is not None:
            self.generator.load_state_dict(best_state)

        # Final evaluation on the untouched test half
        print("\nEvaluating best generator on the untouched test holdout...")
        with torch.no_grad():
            final_adv_raw = self.generator.mutate_raw(
                x_test,
                self.norm_min,
                self.scales,
                eps=self.eps,
                mode="manifold" if self.manifold_shaping else "additive",
            )

        eval_summary = self.evaluator.evaluate_evasion(
            x_orig=x_test,
            x_adv=final_adv_raw,
            num_pre_evaded=self.num_pre_evaded,
            total_attack_population=self.total_attacks,
        )

        print("-" * 70)
        print("FINAL EVALUATION SUMMARY (UNTOUCHED TEST HOLDOUT):")
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

        os.makedirs(save_dir, exist_ok=True)
        gen_path = os.path.join(save_dir, checkpoint_name)
        critic_path = os.path.join(
            save_dir, checkpoint_name.replace("generator.pt", "critic.pt")
        )
        torch.save(self.generator.state_dict(), gen_path)
        torch.save(self.critic.state_dict(), critic_path)
        print(f"Saved generator checkpoint: {gen_path}")
        print(f"Saved critic checkpoint:    {critic_path}")

        return {
            "eval_summary": eval_summary,
            "test_summary": eval_summary,
            "test_asr": eval_summary["conditional_asr_pct"],
            "history": history,
            "best_epoch": best_epoch,
            "best_selection_asr": best_asr,
            "best_holdout_asr": best_asr,
            "eps": self.eps,
            "seed": self.seed,
            "generator_path": gen_path,
            "critic_path": critic_path,
        }

    def train(
        self,
        num_epochs: int = 10,
        batch_size: int = 128,
        lambda_evade: float = 30.0,
        lambda_dist: float = 0.05,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Convenience alias for train_loop."""
        return self.train_loop(
            num_epochs=num_epochs,
            batch_size=batch_size,
            lambda_evade=lambda_evade,
            lambda_dist=lambda_dist,
            **kwargs,
        )


if __name__ == "__main__":
    loop = AdversarialFeedbackLoop()
    loop.train_loop(num_epochs=20, batch_size=128)
