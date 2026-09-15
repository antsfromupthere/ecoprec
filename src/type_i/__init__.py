"""Standalone Type-I conventional-FSQ method with ALBERT-style sharing.

The deployed direction for every UE is exactly one beam from the complete
``O1*N1`` oversampled-DFT dictionary. One ordinary Transformer block is shared
across encoder depth.
"""

from src.common.transformer import SharedTransformerEncoder

from .config import DirectWConfig, TrainConfig
from .model import DirectWOutput, SoftDFTWNet
from .projection import audit_strict_output

__all__ = [
    "DirectWConfig",
    "TrainConfig",
    "DirectWOutput",
    "SoftDFTWNet",
    "audit_strict_output",
    "SharedTransformerEncoder",
]
