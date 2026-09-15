"""Numeric precision and complex contraction helpers."""

from __future__ import annotations

import torch


RDTYPE: torch.dtype = torch.float32
CDTYPE: torch.dtype = torch.complex64
EPS_DIV: float = 1e-12


def complex_safe_einsum(
    equation: str,
    *operands: torch.Tensor,
    allow_double: bool = False,
) -> torch.Tensor:

    dtype = operands[0].dtype
    for operand in operands[1:]:
        dtype = torch.promote_types(dtype, operand.dtype)
    if not allow_double:
        assert dtype in (CDTYPE, RDTYPE), (
            f"complex_safe_einsum({equation!r}) promoted to {dtype}, but the "
            f"live precision is {RDTYPE}/{CDTYPE}. Operand dtypes were "
            f"{[operand.dtype for operand in operands]}; cast at the boundary "
            "or pass allow_double=True for an intentional double-precision "
            "contraction."
        )
    return torch.einsum(
        equation,
        *(operand.to(dtype) for operand in operands),
    )


# Compatibility with the historical private name used by the source modules.
_complex_safe_einsum = complex_safe_einsum


__all__ = [
    "RDTYPE",
    "CDTYPE",
    "EPS_DIV",
    "complex_safe_einsum",
    "_complex_safe_einsum",
]
