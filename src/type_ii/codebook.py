import math
import torch


def dft_beam(m: int, N1: int, O1: int, device=None) -> torch.Tensor:
    period = int(N1) * int(O1)
    n = torch.arange(int(N1), dtype=torch.long, device=device)
    residue = (int(m) * n) % period
    angle = residue.to(torch.float64) * (2.0 * math.pi / period)
    unit = torch.complex(torch.cos(angle), torch.sin(angle)).to(torch.complex64)
    return unit / math.sqrt(N1)


def build_dictionary(N1: int, O1: int, device=None) -> torch.Tensor:
     return torch.stack([
        torch.stack([
            dft_beam(O1 * n + o, N1, O1, device=device)
            for n in range(N1)
        ])
        for o in range(O1)
    ])


@torch.no_grad()
def dictionary_error(dictionary: torch.Tensor) -> float:
    O1, N1, _ = dictionary.shape
    eye = torch.eye(N1, device=dictionary.device, dtype=dictionary.dtype)
    gram = dictionary @ dictionary.conj().transpose(-1, -2)
    return float((gram - eye.expand(O1, -1, -1)).abs().amax())

