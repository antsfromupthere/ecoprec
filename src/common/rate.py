import torch

from .numeric import EPS_DIV, complex_safe_einsum


def snr_db_to_noise_power(snr_db: float, tx_power: float) -> float:
    return tx_power / (10.0 ** (snr_db / 10.0))


def compute_sum_rate(
    h: torch.Tensor,
    v: torch.Tensor,
    noise_power: float,
) -> torch.Tensor:
    """Return per-sample MU-MISO downlink sum rate in bit/s/Hz."""

    inner = complex_safe_einsum("bkt,bjt->bkj", h, v)
    received_power = inner.real**2 + inner.imag**2
    users = h.shape[1]
    index = torch.arange(users, device=h.device)
    signal = received_power[:, index, index]
    interference = received_power.sum(dim=-1) - signal
    sinr = signal / (interference + noise_power).clamp_min(EPS_DIV)
    rate = torch.log2(1.0 + sinr)
    return rate.sum(dim=-1)


__all__ = ["snr_db_to_noise_power", "compute_sum_rate"]
