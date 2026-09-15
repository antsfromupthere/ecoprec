from __future__ import annotations

import math
from dataclasses import dataclass

from src.common.quantizers import (
    fsq_levels_for_budget,
    ifsq_levels_for_budget,
    target_fsq_feature_dims,
)
from src.common.feedback_budget import direct_w_feedback_bits_for


@dataclass(frozen=True)
class DirectWConfig:
    N1: int = 12
    O1: int = 4
    K: int = 4
    L: int = 3
    N_psk: int = 4
    amp_bits: int = 3
    phase_bits: int = 2
    tx_power: float = 1.0

    n_pilots: int = 6
    feedback_bits: int = 0
    fsq_levels: tuple[int, ...] = ()
    fsq_bound: str = "isigmoid"
    fsq_target_bits_per_feature: float | None = None
    ue_hidden: tuple[int, ...] = (1024, 512, 256)
    ue_norm: str = "batch"
    learn_pilots: bool = True

    # ALBERT-style BS map.
    d_model: int = 128
    attn_heads: int = 4
    attn_layers: int = 2
    ffn_dim: int = 256
    dropout: float = 0.0

    row_temperature: float = 0.7
    support_temperature: float = 0.35
    power_floor_fraction: float = 0.0
    n_amp_levels: int = 7

    def __post_init__(self):

        feedback_bits = direct_w_feedback_bits_for(
            self.N1,
            self.O1,
            self.L,
            self.amp_bits,
            self.phase_bits,
        )
        target_dims = (
            None if self.fsq_target_bits_per_feature is None else
            target_fsq_feature_dims(
                feedback_bits,
                self.fsq_target_bits_per_feature,
                3 if self.fsq_bound == "isigmoid" else 4,
            )
        )
        levels = (
            ifsq_levels_for_budget(
                feedback_bits,
                min_dims=(7 if target_dims is None else target_dims),
            )
            if self.fsq_bound == "isigmoid"
            else fsq_levels_for_budget(
                feedback_bits,
                min_dims=(7 if target_dims is None else target_dims),
            )
        )
        object.__setattr__(self, "feedback_bits", feedback_bits)
        object.__setattr__(self, "fsq_levels", tuple(levels))
        capacity = sum(math.log2(level) for level in self.fsq_levels)

    @property
    def Nt(self) -> int:
        return self.N1

    @property
    def n_coefficients(self) -> int:
        return self.O1 * self.N1

    @property
    def amplitude_values(self) -> tuple[float, ...]:
        return tuple(2.0 ** (-0.5 * c) for c in range(self.n_amp_levels))


@dataclass(frozen=True)
class TrainConfig:

    name: str = "type_ii"
    epochs: int = 400
    batches_per_epoch: int = 20
    batch_size: int = 1024
    optimizer: str = "adam"
    learning_rate: float = 5e-4
    weight_decay: float = 1e-4
    grad_clip: float = 10.0
    lr_warmup_epochs: int = 100
    feedback_warmup_epochs: int = 0
    soft_rate_weight: float = 1.0
    soft_rate_anneal_epochs: int = 300
    projection_weight: float = 0.05
    eval_every: int = 10
    eval_size: int = 10000
    compute_diagnostics: bool = False
    seed: int = 43
    val_seed: int = 2024

    front_end_checkpoint: str = ""
    warm_start_front_end: bool = False
    freeze_front_end: bool = False
    ema_decay: float = 0.0
    early_patience: int = 0
    early_min_epochs: int = 0
    save_best: bool = False

    Lp: int = 3
    snr_db: float = 10.0
    mainlobe_ue_deg: float = 0.0
    halfbw_ue_deg: float = 45.0
    lsf_ue: float = 0.0

    def __post_init__(self):
        if self.optimizer != "adam":
            raise ValueError("optimizer must be 'adam'")
        positive = {
            "epochs": self.epochs,
            "batches_per_epoch": self.batches_per_epoch,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "grad_clip": self.grad_clip,
            "eval_every": self.eval_every,
            "eval_size": self.eval_size,
            "Lp": self.Lp,
        }