import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, CPU Torch, and CUDA Torch RNG streams."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def set_matmul_precision(tf32: bool) -> None:
    """Enable or disable TF32 consistently for matmul and cuDNN."""

    torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
    torch.backends.cudnn.allow_tf32 = bool(tf32)


def get_device(device_str: str = "auto") -> torch.device:
    """Resolve the training device and configure the CUDA precision policy."""

    if device_str != "auto":
        device = torch.device(device_str)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if device.type == "cuda":
        set_matmul_precision(os.environ.get("NEWLOGI_TF32", "1") != "0")
    return device


__all__ = ["set_seed", "set_matmul_precision", "get_device"]
