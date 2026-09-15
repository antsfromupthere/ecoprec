"""Shared primitives for the standalone FSQ--Transformer methods."""

from .data import LearnablePilots, draw_channel_batch, generate_batch_data_torch
from .checkpoint import initialize_albert_from_checkpoint
from .feedback_budget import direct_w_feedback_bits_for
from .feedback_encoder import UEFeedbackEncoder
from .numeric import CDTYPE, EPS_DIV, RDTYPE, complex_safe_einsum
from .quantizers import (
    FSQ,
    fsq_bits,
    fsq_levels_for_budget,
    ifsq_levels_for_budget,
    target_fsq_feature_dims,
)
from .rate import compute_sum_rate, snr_db_to_noise_power
from .runtime import get_device, set_matmul_precision, set_seed
from .sweep_io import (
    load_existing_run,
    method_component,
    prepare_layout,
    run_directory,
)
from .transformer import SharedTransformerEncoder

__all__ = [
    "CDTYPE",
    "EPS_DIV",
    "FSQ",
    "LearnablePilots",
    "RDTYPE",
    "SharedTransformerEncoder",
    "UEFeedbackEncoder",
    "complex_safe_einsum",
    "compute_sum_rate",
    "direct_w_feedback_bits_for",
    "draw_channel_batch",
    "fsq_bits",
    "fsq_levels_for_budget",
    "generate_batch_data_torch",
    "get_device",
    "ifsq_levels_for_budget",
    "initialize_albert_from_checkpoint",
    "load_existing_run",
    "method_component",
    "prepare_layout",
    "run_directory",
    "set_matmul_precision",
    "set_seed",
    "snr_db_to_noise_power",
    "target_fsq_feature_dims",
]
