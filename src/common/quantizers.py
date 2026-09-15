import math

import torch
import torch.nn as nn


def fsq_bits(levels) -> float:
    return float(sum(math.log2(int(level)) for level in levels))


def target_fsq_feature_dims(
    bits: int,
    bits_per_feature: float,
    min_level: int,
) -> int:
    """Choose a moderate automatic FSQ width under a bit budget."""

    bits = int(bits)
    bits_per_feature = float(bits_per_feature)
    min_level = int(min_level)

    maximum = int(bits // math.log2(min_level))
    target = max(1, int(bits // bits_per_feature))
    return min(target, maximum)


def fsq_levels_for_budget(
    bits: int,
    preferred=(8, 6, 5, 4, 3),
    min_dims: int = 7,
) -> list:

    def greedy(pref):
        levels, remaining = [], float(bits)
        while True:
            for level in sorted(pref, reverse=True):
                if math.log2(level) <= remaining + 1e-9:
                    levels.append(int(level))
                    remaining -= math.log2(level)
                    break
            else:
                return levels

    levels = greedy(preferred)
    if len(levels) >= min_dims:
        return levels

    dimensions = max(1, min(min_dims, int(bits // math.log2(4))))
    levels = [4] * dimensions

    def total(values):
        return sum(math.log2(level) for level in values)

    for level in [value for value in sorted(preferred) if value > 4]:
        for index in range(dimensions):
            candidate = [
                *levels[:index],
                level,
                *levels[index + 1 :],
            ]
            if levels[index] < level and total(candidate) <= bits + 1e-9:
                levels[index] = level
    return levels


def ifsq_levels_for_budget(
    bits: int,
    preferred=(9, 5, 3),
    min_dims: int = 7,
) -> list:

    def greedy(pref):
        levels, remaining = [], float(bits)
        while True:
            for level in sorted(pref, reverse=True):
                if math.log2(level) <= remaining + 1e-9:
                    levels.append(int(level))
                    remaining -= math.log2(level)
                    break
            else:
                return levels

    levels = greedy(preferred)
    if len(levels) >= min_dims:
        return levels

    dimensions = max(1, min(min_dims, int(bits // math.log2(3))))
    levels = [3] * dimensions

    def total(values):
        return sum(math.log2(level) for level in values)

    for level in [value for value in sorted(preferred) if value > 3]:
        for index in range(dimensions):
            candidate = [
                *levels[:index],
                level,
                *levels[index + 1 :],
            ]
            if levels[index] < level and total(candidate) <= bits + 1e-9:
                levels[index] = level
    return levels


class FSQ(nn.Module):

    EPS = 1e-3

    def __init__(
        self,
        levels,
        feedback_bits: int = None,
        bound: str = "tanh",
    ):
        super().__init__()
        levels = [int(level) for level in levels]

        self.level_list = levels
        self.d = len(levels)
        self.bits = fsq_bits(levels)
        self.bound_mode = bound
        if feedback_bits is not None:
            self.assert_budget(feedback_bits)

        levels_t = torch.tensor(levels, dtype=torch.float32)
        half_l = (levels_t - 1) * (1 + self.EPS) / 2
        even = levels_t.to(torch.long) % 2 == 0
        offset = torch.where(
            even,
            torch.full_like(levels_t, 0.5),
            torch.zeros_like(levels_t),
        )
        unit_offset = offset / half_l
        shift = (
            torch.atanh(unit_offset)
            if bound == "tanh"
            else 1.25 * torch.atanh(unit_offset)
        )
        half_width = torch.floor(levels_t / 2)
        self.register_buffer("levels", levels_t)
        self.register_buffer("half_l", half_l)
        self.register_buffer("offset", offset)
        self.register_buffer("shift", shift)
        self.register_buffer("half_width", half_width)

    def assert_budget(self, feedback_bits: int):

        assert self.bits <= feedback_bits + 1e-9, (
            f"FSQ capacity {self.bits:.4f} bits "
            f"(levels {self.level_list}) exceeds the B-bit budget "
            f"{feedback_bits}"
        )

    def _squash(self, values: torch.Tensor) -> torch.Tensor:
        if self.bound_mode == "tanh":
            return torch.tanh(values)
        return 2.0 * torch.sigmoid(1.6 * values) - 1.0

    def bound(self, z: torch.Tensor) -> torch.Tensor:
        """Return the bounded continuous latent used during warm-up."""

        bounded = self._squash(z + self.shift) * self.half_l - self.offset
        return bounded / self.half_width

    def quantize(self, z: torch.Tensor) -> torch.Tensor:
        bounded = self._squash(z + self.shift) * self.half_l - self.offset
        quantized_integer = bounded + (
            torch.round(bounded) - bounded
        ).detach()
        return quantized_integer / self.half_width

    def forward(
        self,
        z: torch.Tensor,
        quantize: bool = True,
    ) -> torch.Tensor:
        return self.quantize(z) if quantize else self.bound(z)

    def values_to_indices(self, quantized: torch.Tensor) -> torch.Tensor:
        """Convert normalized grid values to per-dimension level indices."""

        quantized_integer = torch.round(quantized * self.half_width)
        return (quantized_integer + self.half_width).to(torch.long)

    def indices_to_values(self, indices: torch.Tensor) -> torch.Tensor:
        """Reconstruct normalized grid values from level indices."""

        return (
            indices.to(torch.float32) - self.half_width
        ) / self.half_width


__all__ = [
    "FSQ",
    "fsq_bits",
    "target_fsq_feature_dims",
    "fsq_levels_for_budget",
    "ifsq_levels_for_budget",
]
