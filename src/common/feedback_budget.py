"""Feedback-bit accounting used by the direct-weight Type-II method."""

import math
from math import comb


def direct_w_feedback_bits_for(
    N1: int,
    O1: int,
    L: int,
    amp_bits: int = 3,
    phase_bits: int = 2,
) -> int:

    N1, O1, L = int(N1), int(O1), int(L)
    amp_bits, phase_bits = int(amp_bits), int(phase_bits)

    group_bits = math.ceil(math.log2(O1))
    support_bits = math.ceil(math.log2(comb(N1, L)))
    return group_bits + support_bits + (amp_bits + phase_bits) * L


__all__ = ["direct_w_feedback_bits_for"]
