import math

import torch
import torch.nn as nn

from .numeric import EPS_DIV, complex_safe_einsum


_BCAST_UE_CACHE: dict = {}


def generate_batch_data_torch(
    batch_size: int,
    Nt: int,
    K: int,
    Lp: int,
    lsf_ue,
    mainlobe_ue_deg,
    halfbw_ue_deg,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Generate the geometric multi-path channel model directly on ``device``."""

    complex_dtype = (
        torch.complex128 if dtype == torch.float64 else torch.complex64
    )

    def broadcast_ue(value, target_device: torch.device) -> torch.Tensor:
        if isinstance(value, (int, float)):
            signature = float(value)
        else:
            signature = tuple(
                float(item)
                for item in torch.as_tensor(value).reshape(-1).tolist()
            )
        key = (signature, K, dtype, str(target_device))
        cached = _BCAST_UE_CACHE.get(key)
        if cached is not None:
            return cached
        tensor = torch.as_tensor(
            value,
            dtype=dtype,
            device=target_device,
        ).reshape(-1)
        if tensor.numel() == 1:
            tensor = tensor.expand(K)
        result = tensor.reshape(1, 1, K)
        _BCAST_UE_CACHE[key] = result
        return result

    lsf = broadcast_ue(lsf_ue, device)
    mainlobe = broadcast_ue(mainlobe_ue_deg, device)
    half_bandwidth = broadcast_ue(halfbw_ue_deg, device)

    shape = (batch_size, Lp, K)
    alpha_real = (
        torch.randn(shape, dtype=dtype, device=device)
        * (1.0 / math.sqrt(2))
        + lsf
    )
    alpha_imag = (
        torch.randn(shape, dtype=dtype, device=device)
        * (1.0 / math.sqrt(2))
        + lsf
    )
    uniform = torch.rand(shape, dtype=dtype, device=device)
    theta_deg = mainlobe + (2.0 * uniform - 1.0) * half_bandwidth
    theta = theta_deg * (math.pi / 180.0)

    alpha = torch.complex(alpha_real, alpha_imag)
    antenna = torch.arange(Nt, dtype=dtype, device=device)
    phase = math.pi * torch.sin(theta).unsqueeze(-1) * antenna
    steering = torch.complex(torch.cos(phase), torch.sin(phase))
    channels = (
        (alpha.unsqueeze(-1) * steering).sum(dim=1)
        * (1.0 / math.sqrt(Lp))
    )
    return channels.to(complex_dtype)


def draw_channel_batch(
    batch_size: int,
    N1: int,
    K: int,
    Lp: int,
    lsf_ue,
    mainlobe_ue_deg,
    halfbw_ue_deg,
    device: torch.device,
) -> torch.Tensor:

    return generate_batch_data_torch(
        batch_size,
        N1,
        K,
        Lp,
        lsf_ue,
        mainlobe_ue_deg,
        halfbw_ue_deg,
        device,
    )


class LearnablePilots(nn.Module):
    """Learnable, row-normalized downlink pilot matrix."""

    def __init__(
        self,
        n_pilots: int,
        Nt: int,
        power: float = 1.0,
        learn: bool = True,
    ):
        super().__init__()
        self.n_pilots = n_pilots
        self.Nt = Nt
        self.power = power
        if n_pilots <= Nt:
            rows = (torch.arange(n_pilots) * Nt) // n_pilots
            antenna = torch.arange(Nt, dtype=torch.float32)
            angle = (
                -2
                * math.pi
                * rows.unsqueeze(-1).float()
                * antenna
                / Nt
            )
            scale = math.sqrt(power / Nt)
            pilot_real = scale * torch.cos(angle)
            pilot_imag = scale * torch.sin(angle)
        else:
            scale = math.sqrt(power / (2 * Nt))
            pilot_real = torch.randn(n_pilots, Nt) * scale
            pilot_imag = torch.randn(n_pilots, Nt) * scale
        self.Xp_r = nn.Parameter(pilot_real, requires_grad=learn)
        self.Xp_i = nn.Parameter(pilot_imag, requires_grad=learn)

    def pilot_matrix(self) -> torch.Tensor:
        squared_norm = (self.Xp_r**2 + self.Xp_i**2).sum(
            dim=1,
            keepdim=True,
        )
        scale = math.sqrt(self.power) / squared_norm.sqrt().clamp_min(EPS_DIV)
        return torch.complex(self.Xp_r * scale, self.Xp_i * scale)

    def forward(self, channels: torch.Tensor) -> torch.Tensor:
        pilot = self.pilot_matrix()
        observation = complex_safe_einsum(
            "lt,bkt->bkl",
            pilot,
            channels,
        )
        return torch.cat([observation.real, observation.imag], dim=-1)


__all__ = [
    "generate_batch_data_torch",
    "draw_channel_batch",
    "LearnablePilots",
]
