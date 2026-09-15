"""Projection of a complex DFT coefficient field onto legal Type-II structure. """

from __future__ import annotations

import math
import os
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from .config import DirectWConfig
    from .model import DirectWOutput


def straight_through(hard: torch.Tensor, soft: torch.Tensor) -> torch.Tensor:
    """Forward value is ``hard``; backward derivative is that of ``soft``."""
    return soft + (hard - soft).detach()


def phase_alphabet(N_psk: int, device=None) -> torch.Tensor:
    """Exact roots for QPSK; high-precision construction for other PSK orders."""
    if N_psk == 4:
        return torch.tensor(
            (1 + 0j, 0 + 1j, -1 + 0j, 0 - 1j),
            dtype=torch.complex64,
            device=device,
        )
    q = torch.arange(N_psk, dtype=torch.float64, device=device)
    angle = 2.0 * math.pi * q / N_psk
    return torch.complex(torch.cos(angle), torch.sin(angle)).to(torch.complex64)


def amplitude_alphabet(n_levels: int, device=None) -> torch.Tensor:
    """The nonzero Type-II ladder [1, 2^-1/2, ..., 2^{-(C-1)/2}]."""
    c = torch.arange(n_levels, dtype=torch.float32, device=device)
    return torch.pow(2.0, -0.5 * c)


def quantized_phase(
    weights: torch.Tensor,
    roots: torch.Tensor,
    *,
    use_ste: bool,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return phase units on the finite grid and their integer indices"""

    n_psk = roots.numel()
    angle = torch.atan2(weights.imag, weights.real)
    index = torch.round(angle * (n_psk / (2.0 * math.pi))).long().remainder(n_psk)
    hard = roots[index]
    magnitude = weights.abs()
    soft = weights / magnitude.clamp_min(eps).to(weights.dtype)
    soft = torch.where(
        (magnitude > eps),
        soft,
        torch.ones((), dtype=weights.dtype, device=weights.device),
    )
    return (straight_through(hard, soft) if use_ste else hard), index


def _bisection_threshold(
    detached: torch.Tensor,
    L: int,
    temperature: float,
    iterations: int,
) -> torch.Tensor:

    lo = detached.amin(dim=-1, keepdim=True) - 20.0 * temperature
    hi = detached.amax(dim=-1, keepdim=True) + 20.0 * temperature
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        mass = torch.sigmoid((detached - mid) / temperature).sum(
            dim=-1, keepdim=True)
        # A low threshold admits too much mass, so move the lower bound up.
        lo = torch.where(mass > L, mid, lo)
        hi = torch.where(mass > L, hi, mid)
    return 0.5 * (lo + hi)


_BISECTION_GRAPHS: dict[tuple, tuple] = {}
_GRAPHS_BROKEN = False


def _graphed_threshold(
    detached: torch.Tensor,
    L: int,
    temperature: float,
    iterations: int,
) -> torch.Tensor:
    global _GRAPHS_BROKEN
    key = (detached.device.index, tuple(detached.shape), L,
           float(temperature), iterations)
    entry = _BISECTION_GRAPHS.get(key)
    if entry is None:
        try:
            static_in = detached.clone()
            side = torch.cuda.Stream(device=detached.device)
            side.wait_stream(torch.cuda.current_stream(detached.device))
            with torch.cuda.stream(side):
                for _ in range(3):  # warmup, per the capture recipe
                    _bisection_threshold(static_in, L, temperature, iterations)
            torch.cuda.current_stream(detached.device).wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                static_out = _bisection_threshold(
                    static_in, L, temperature, iterations)
        except Exception as error:  # driver/platform dependent
            _GRAPHS_BROKEN = True
            print(f"[projection] CUDA-graph capture failed ({error!r}); "
                  f"using the eager bisection from now on")
            return _bisection_threshold(detached, L, temperature, iterations)
        entry = (graph, static_in, static_out)
        _BISECTION_GRAPHS[key] = entry
    graph, static_in, static_out = entry
    static_in.copy_(detached)
    graph.replay()
    return static_out.clone()


def relaxed_top_l(
    log_energy: torch.Tensor,
    L: int,
    temperature: float,
    iterations: int = 32,
) -> torch.Tensor:

    N = log_energy.shape[-1]
    if L == N:
        return torch.ones_like(log_energy)
    detached = log_energy.detach()
    use_graph = (
        detached.is_cuda
        and not _GRAPHS_BROKEN
        and os.environ.get("NLW_CUDA_DEBUG") != "1"
    )
    threshold = (
        _graphed_threshold(detached, L, temperature, iterations)
        if use_graph
        else _bisection_threshold(detached, L, temperature, iterations)
    )
    return torch.sigmoid((log_energy - threshold) / temperature)


def project_one_row_top_l(
    energy: torch.Tensor,
    L: int,
    *,
    row_temperature: float,
    support_temperature: float,
    eps: float = 1e-12,
) -> dict[str, torch.Tensor]:

    O1, N1 = energy.shape[-2:]
    top_energy, top_index = torch.topk(energy, L, dim=-1)
    row_score = top_energy.sum(dim=-1)                       # [..., O1]
    row_index = row_score.argmax(dim=-1)                     # [...]
    chosen = torch.gather(
        top_index,
        -2,
        row_index.unsqueeze(-1).unsqueeze(-1).expand(
            *row_index.shape, 1, L),
    ).squeeze(-2)
    support_index = chosen.sort(dim=-1).values

    flat_index = row_index.unsqueeze(-1) * N1 + support_index
    hard_flat = torch.zeros(
        *energy.shape[:-2], O1 * N1,
        dtype=energy.dtype,
        device=energy.device,
    )
    hard_flat.scatter_(-1, flat_index, 1.0)
    hard_mask = hard_flat.view_as(energy)

    log_energy = torch.log(energy.clamp_min(eps))
    within = relaxed_top_l(log_energy, L, support_temperature)
    row_probability = F.softmax(
        torch.log(row_score.clamp_min(eps)) / row_temperature, dim=-1)
    surrogate_mask = within * row_probability.unsqueeze(-1)

    kept = (energy * hard_mask).sum(dim=(-1, -2))
    total = energy.sum(dim=(-1, -2))
    captured = torch.where(total > eps, kept / total.clamp_min(eps),
                           torch.ones_like(total))

    if O1 > 1:
        best_two = torch.topk(row_score, 2, dim=-1).values
        row_margin = (best_two[..., 0] - best_two[..., 1]) \
            / best_two[..., 0].clamp_min(eps)
    else:
        row_margin = torch.ones_like(row_score[..., 0])

    return {
        "hard_mask": hard_mask,
        "surrogate_mask": surrogate_mask,
        "row_index": row_index,
        "support_index": support_index,
        "row_score": row_score,
        "row_margin": row_margin,
        "captured_energy": captured,
    }


def gather_selected(
    field: torch.Tensor,
    row_index: torch.Tensor,
    support_index: torch.Tensor,
) -> torch.Tensor:

    N1 = field.shape[-1]
    flat = field.reshape(*field.shape[:-2], -1)
    flat_index = row_index.unsqueeze(-1) * N1 + support_index
    return torch.gather(flat, -1, flat_index)


def quantize_relative_amplitude(
    relative_amplitude: torch.Tensor,
    alphabet: torch.Tensor,
    *,
    use_ste: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Nearest legal relative amplitude and integer grid index."""
    bounded = relative_amplitude.clamp(min=alphabet[-1]).clamp(max=1.0)
    index = (bounded.unsqueeze(-1) - alphabet).abs().argmin(dim=-1)
    hard = alphabet[index]
    return (straight_through(hard, bounded) if use_ste else hard), index


@torch.no_grad()
def audit_strict_output(
    output: "DirectWOutput",
    dictionary: torch.Tensor,
    cfg: "DirectWConfig",
    atol: float = 8e-5,
) -> dict[str, int | float]:
    """Independently reconstruct and validate the strict Type-II precoder."""

    v = output.v_strict
    B, K, Nt = v.shape
    row = output.row_index
    support = output.support_index
    amp_index = output.amplitude_index
    phase_index = output.selected_phase_index
    power = output.power

    report: dict[str, int | float] = {}
    report["bad_row_range"] = int(((row < 0) | (row >= cfg.O1)).sum())
    report["bad_support_range"] = int(
        ((support < 0) | (support >= cfg.N1)).sum())
    report["bad_amplitude_range"] = int(
        ((amp_index < 0) | (amp_index >= cfg.n_amp_levels)).sum())
    report["bad_phase_range"] = int(
        ((phase_index < 0) | (phase_index >= cfg.N_psk)).sum())
    report["wrong_cardinality"] = 0 if support.shape[-1] == cfg.L else B * K
    sorted_support = support.sort(dim=-1).values
    report["duplicate_support"] = int(
        (sorted_support[..., 1:] == sorted_support[..., :-1]).sum())
    report["missing_reference_amplitude"] = int(
        (amp_index != 0).all(dim=-1).sum())
    report["bad_power"] = int(
        ((power < 0) | ~torch.isfinite(power)).sum()
        + (power.sum(dim=-1) - cfg.tx_power).abs().gt(2e-5).sum())

    beams = dictionary[row.reshape(-1)]                         # [BK,N1,Nt]
    beams = torch.gather(
        beams,
        1,
        support.reshape(-1, cfg.L, 1).expand(-1, -1, Nt),
    )
    amps = amplitude_alphabet(cfg.n_amp_levels, v.device)[amp_index]
    phases = phase_alphabet(cfg.N_psk, v.device)[phase_index]
    coefficients = (amps * phases).reshape(-1, cfg.L)
    direction = (coefficients.unsqueeze(-1) * beams).sum(dim=1)
    direction = direction / direction.norm(
        dim=-1, keepdim=True).clamp_min(1e-12)
    reconstructed = direction.reshape(B, K, Nt) \
        * power.sqrt().unsqueeze(-1).to(direction.dtype)
    error = (reconstructed - v).abs().amax()
    report["reconstruction_max_error"] = float(error)
    report["reconstruction_mismatch"] = int(error > atol)

    violation_keys = (
        "bad_row_range",
        "bad_support_range",
        "bad_amplitude_range",
        "bad_phase_range",
        "wrong_cardinality",
        "duplicate_support",
        "missing_reference_amplitude",
        "bad_power",
        "reconstruction_mismatch",
    )
    report["total_violations"] = sum(int(report[key]) for key in violation_keys)
    return report

