"""Complete unit-norm oversampled-DFT Type-I dictionary."""

from __future__ import annotations

import math

import torch


def dft_beam(m: int, N1: int, O1: int, device=None) -> torch.Tensor:
    """Return global Type-I beam ``m`` as a unit-norm complex64 vector."""

    period = int(N1) * int(O1)
    if not 0 <= int(m) < period:
        raise ValueError(f"beam index m must lie in [0,{period}); got {m}")
    n = torch.arange(int(N1), dtype=torch.long, device=device)
    residue = (int(m) * n) % period
    angle = residue.to(torch.float64) * (2.0 * math.pi / period)
    unit = torch.complex(torch.cos(angle), torch.sin(angle)).to(torch.complex64)
    return unit / math.sqrt(N1)


def build_dictionary(N1: int, O1: int, device=None) -> torch.Tensor:
    """Return all ``O1*N1`` Type-I beams, shape ``[C,N1]``."""

    return torch.stack([
        dft_beam(m, N1, O1, device=device)
        for m in range(int(N1) * int(O1))
    ])


@torch.no_grad()
def dictionary_error(dictionary: torch.Tensor) -> dict[str, float]:
    """Report unit-norm and equal-index-period consistency errors."""

    norms = dictionary.abs().square().sum(dim=-1)
    return {
        "unit_norm_max_error": float((norms - 1.0).abs().amax()),
        "max_magnitude_error": float(
            (dictionary.abs() - 1.0 / math.sqrt(dictionary.shape[-1]))
            .abs().amax()),
    }
