"""Sum-rate objective for strict Type-I direct-w learning."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from src.common.rate import compute_sum_rate

from .model import DirectWOutput


@dataclass
class ObjectiveValue:
    loss: torch.Tensor
    strict_rate: torch.Tensor
    soft_rate: torch.Tensor | None
    projection_leakage: torch.Tensor | None

    def scalars(self) -> dict[str, float]:
        result = {
            "loss": float(self.loss.detach()),
            "strict_rate": float(self.strict_rate.detach()),
        }
        if self.soft_rate is not None:
            result["soft_rate"] = float(self.soft_rate.detach())
        if self.projection_leakage is not None:
            result["projection_leakage"] = float(
                self.projection_leakage.detach())
        return result


def direct_w_objective(
    output: DirectWOutput,
    channels: torch.Tensor,
    noise_power: float,
    *,
    soft_rate_weight: float = 0.0,
    projection_weight: float = 0.0,
    include_diagnostics: bool = False,
) -> ObjectiveValue:

    strict_rate = compute_sum_rate(
        channels, output.v_strict, noise_power).mean()
    soft_rate = (
        compute_sum_rate(channels, output.v_soft, noise_power).mean()
        if soft_rate_weight or include_diagnostics else None
    )
    leakage = (
        (1.0 - output.captured_energy).mean()
        if projection_weight or include_diagnostics else None
    )
    loss = -strict_rate
    if soft_rate_weight:
        loss = loss - float(soft_rate_weight) * soft_rate
    if projection_weight:
        loss = loss + float(projection_weight) * leakage
    return ObjectiveValue(
        loss=loss,
        strict_rate=strict_rate,
        soft_rate=soft_rate,
        projection_leakage=leakage,
    )
