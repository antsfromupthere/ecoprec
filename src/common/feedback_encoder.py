"""Shared per-UE encoder for finite-rate feedback."""

import torch
import torch.nn as nn

from .numeric import EPS_DIV
from .quantizers import FSQ


class UEFeedbackEncoder(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        hidden: tuple,
        *,
        fsq: FSQ,
        latent_rms_norm: bool = False,
        norm: str = "batch",
    ):
        super().__init__()
        self.norm = norm
        self.fsq = fsq
        self.feat_dim = fsq.d
        self.latent_rms_norm = latent_rms_norm

        layers = []
        previous_dim = obs_dim
        for hidden_dim in hidden:
            layers.append(nn.Linear(previous_dim, hidden_dim))
            if self.norm == "batch":
                layers.append(nn.BatchNorm1d(hidden_dim))
            elif self.norm == "layer":
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            previous_dim = hidden_dim
        layers.append(nn.Linear(previous_dim, self.feat_dim))
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        y_noisy: torch.Tensor,
        quantize: bool = True,
    ) -> torch.Tensor:
        batch_size, users, _ = y_noisy.shape
        latent = self.net(y_noisy.reshape(batch_size * users, -1))
        latent = latent.reshape(batch_size, users, self.feat_dim)
        if self.latent_rms_norm:
            rms = latent.pow(2).mean(dim=-1, keepdim=True)
            latent = latent / rms.clamp_min(EPS_DIV).sqrt()
        return self.fsq(latent, quantize=quantize)


__all__ = ["UEFeedbackEncoder"]
