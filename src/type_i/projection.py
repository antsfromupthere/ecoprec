"""Hard one-beam Type-I projection"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from .config import DirectWConfig
    from .model import DirectWOutput


def straight_through(hard: torch.Tensor, soft: torch.Tensor) -> torch.Tensor:
    return soft + (hard - soft).detach()


def project_one_beam(
    energy: torch.Tensor,
    *,
    temperature: float,
    use_ste: bool,
    eps: float = 1e-12,
) -> dict[str, torch.Tensor]:

    beam_index = energy.argmax(dim=-1)
    hard = F.one_hot(beam_index, energy.shape[-1]).to(energy.dtype)
    probability = F.softmax(
        torch.log(energy.clamp_min(eps)) / temperature,
        dim=-1,
    )
    mask = straight_through(hard, probability) if use_ste else hard

    best_two = torch.topk(energy, 2, dim=-1).values
    beam_margin = (best_two[..., 0] - best_two[..., 1]) \
        / best_two[..., 0].clamp_min(eps)
    return {
        "beam_index": beam_index,
        "hard_one_hot": hard,
        "surrogate_probability": probability,
        "mask": mask,
        "beam_margin": beam_margin,
    }


@torch.no_grad()
def audit_strict_output(
    output: "DirectWOutput",
    dictionary: torch.Tensor,
    cfg: "DirectWConfig",
    atol: float = 2e-6,
) -> dict[str, int | float]:

    v = output.v_strict
    B, K, Nt = v.shape
    index = output.beam_index
    power = output.power

    report: dict[str, int | float] = {}
    report["bad_beam_range"] = int(
        ((index < 0) | (index >= cfg.n_beams)).sum())
    report["wrong_cardinality"] = int(
        (output.hard_one_hot.sum(dim=-1) != 1).sum())
    report["nonbinary_mask"] = int(
        ((output.hard_one_hot != 0) & (output.hard_one_hot != 1)).sum())
    report["bad_power"] = int(
        ((power < 0) | ~torch.isfinite(power)).sum()
        + (power.sum(dim=-1) - cfg.tx_power).abs().gt(2e-5).sum())

    selected = dictionary[index.reshape(-1)].reshape(B, K, Nt)
    reconstructed = selected \
        * power.sqrt().unsqueeze(-1).to(selected.dtype)
    error = (reconstructed - v).abs().amax()
    report["reconstruction_max_error"] = float(error)
    report["reconstruction_mismatch"] = int(error > atol)

    scalar = (selected.conj() * v).sum(dim=-1)
    phase_error = scalar.imag.abs().amax()
    negative_scale = (scalar.real < -atol).sum()
    report["phase_shift_max_error"] = float(phase_error)
    report["phase_shift_violation"] = int(
        phase_error > atol) + int(negative_scale)

    violation_keys = (
        "bad_beam_range",
        "wrong_cardinality",
        "nonbinary_mask",
        "bad_power",
        "reconstruction_mismatch",
        "phase_shift_violation",
    )
    report["total_violations"] = sum(
        int(report[key]) for key in violation_keys)
    return report
