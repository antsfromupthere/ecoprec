"""Type-II FSQ-Transformer learning over a fixed oversampled DFT dictionary.

The package exposes the proposed Type-II method through a deliberately small
public API.  Its BS mixer applies one ordinary pre-LN Transformer block at each
encoder step with ALBERT-style parameter sharing.

    DirectWConfig       geometry and model configuration
    SoftDFTWNet         pilots -> feedback -> complex w[o, n] -> precoders
    DirectWOutput       soft, projected-continuous, and strict Type-II outputs
    audit_strict_output independent legality/reconstruction check
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
