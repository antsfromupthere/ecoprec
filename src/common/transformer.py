"""ALBERT-style Transformer """

from __future__ import annotations

import torch
import torch.nn as nn


class SharedTransformerEncoder(nn.Module):

    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_steps: int,
        dim_feedforward: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.num_steps = int(num_steps)
        self.layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, token: torch.Tensor) -> torch.Tensor:
        for _ in range(self.num_steps):
            token = self.layer(token)
        return self.norm(token)


__all__ = ["SharedTransformerEncoder"]
