"""Type-I FSQ direct-w model with a shared Transformer encoder block."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from src.common.data import LearnablePilots
from src.common.feedback_encoder import UEFeedbackEncoder
from src.common.quantizers import FSQ
from src.common.transformer import SharedTransformerEncoder

from .codebook import build_dictionary
from .config import DirectWConfig
from .projection import project_one_beam


@dataclass
class DirectWOutput:
    """Soft learned field and its strict Type-I interpretation."""

    weights: torch.Tensor
    basis_coefficients: torch.Tensor
    v_soft: torch.Tensor
    v_strict: torch.Tensor
    hard_one_hot: torch.Tensor
    surrogate_probability: torch.Tensor
    beam_index: torch.Tensor
    power: torch.Tensor
    captured_energy: torch.Tensor
    beam_margin: torch.Tensor
    feedback: torch.Tensor | None = None

    def deployed(self) -> torch.Tensor:
        return self.v_strict


class DirectWDecoder(nn.Module):
    """Finite feedback -> complex field -> one Type-I beam plus power."""
    def __init__(self, feature_dim: int, cfg: DirectWConfig, device=None):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Linear(feature_dim, cfg.d_model)
        self.user_mixer = SharedTransformerEncoder(
            d_model=cfg.d_model,
            nhead=cfg.attn_heads,
            num_steps=cfg.attn_layers,
            dim_feedforward=cfg.ffn_dim,
            dropout=cfg.dropout,
        )
        self.weight_head = nn.Linear(cfg.d_model, 2 * cfg.n_beams)
        nn.init.xavier_uniform_(self.weight_head.weight, gain=0.1)
        nn.init.zeros_(self.weight_head.bias)

        self.register_buffer(
            "dictionary", build_dictionary(cfg.N1, cfg.O1, device=device))

    def weights_from_feedback(self, feedback: torch.Tensor) -> torch.Tensor:
        """Return ``w[B,K,C]`` over all ``C=O1*N1`` beams."""

        token = self.user_mixer(self.embedding(feedback))
        pair = self.weight_head(token).reshape(
            *token.shape[:2], self.cfg.n_beams, 2)
        return torch.complex(pair[..., 0], pair[..., 1])

    def _power_shares(self, score: torch.Tensor) -> torch.Tensor:
        total = score.sum(dim=-1, keepdim=True)
        equal = torch.full_like(score, 1.0 / score.shape[-1])
        learned = score / total.clamp_min(1e-12)
        learned = torch.where(total > 1e-12, learned, equal)
        floor = self.cfg.power_floor_fraction
        return (1.0 - floor) * learned + floor * equal

    def _unit_direction(self, vector: torch.Tensor) -> torch.Tensor:
        norm = vector.norm(dim=-1, keepdim=True)
        unit = vector / norm.clamp_min(1e-12)
        fallback = self.dictionary[0].view(
            *((1,) * (vector.ndim - 1)), self.cfg.Nt)
        return torch.where(norm > 1e-12, unit, fallback)

    def precoders_from_weights(
        self,
        weights: torch.Tensor,
        *,
        use_ste: bool | None = None,
    ) -> DirectWOutput:

        if weights.shape[-1] != self.cfg.n_beams:
            raise ValueError(
                f"expected [...,{self.cfg.n_beams}] weights, "
                f"got {tuple(weights.shape)}")
        use_ste = self.training if use_ste is None else bool(use_ste)

        soft_unscaled = torch.einsum("bkc,ct->bkt", weights, self.dictionary)
        soft_score = soft_unscaled.abs().square().sum(dim=-1)
        power = self.cfg.tx_power * self._power_shares(soft_score)
        v_soft = self._unit_direction(soft_unscaled) \
            * power.sqrt().unsqueeze(-1).to(soft_unscaled.dtype)

        basis_coefficients = torch.einsum(
            "ct,bkt->bkc", self.dictionary.conj(), soft_unscaled)
        energy = basis_coefficients.abs().square()
        projection = project_one_beam(
            energy,
            temperature=self.cfg.beam_temperature,
            use_ste=use_ste,
        )

        # Forward: the mask is one-hot, so this is precisely dictionary[m].
        strict_direction = torch.einsum(
            "bkc,ct->bkt",
            projection["mask"].to(self.dictionary.dtype),
            self.dictionary,
        )
        v_strict = strict_direction \
            * power.sqrt().unsqueeze(-1).to(strict_direction.dtype)

        selected_energy = torch.gather(
            energy, -1, projection["beam_index"].unsqueeze(-1)).squeeze(-1)
        captured_energy = torch.where(
            soft_score > 1e-12,
            selected_energy / soft_score.clamp_min(1e-12),
            torch.ones_like(soft_score),
        ).clamp(max=1.0)

        return DirectWOutput(
            weights=weights,
            basis_coefficients=basis_coefficients,
            v_soft=v_soft,
            v_strict=v_strict,
            hard_one_hot=projection["hard_one_hot"],
            surrogate_probability=projection["surrogate_probability"],
            beam_index=projection["beam_index"],
            power=power,
            captured_energy=captured_energy,
            beam_margin=projection["beam_margin"],
        )

    def forward(self, feedback: torch.Tensor) -> DirectWOutput:
        return self.precoders_from_weights(
            self.weights_from_feedback(feedback))


class SoftDFTWNet(nn.Module):
    """Pilots -> Type-I-sized feedback -> one beam plus learned power."""

    def __init__(self, cfg: DirectWConfig | None = None, device=None):
        super().__init__()
        self.cfg = cfg or DirectWConfig()
        self.pilots = LearnablePilots(
            self.cfg.n_pilots,
            self.cfg.Nt,
            power=1.0,
            learn=self.cfg.learn_pilots,
        )
        self.fsq = FSQ(
            list(self.cfg.fsq_levels),
            feedback_bits=self.cfg.feedback_bits,
            bound=self.cfg.fsq_bound,
        )
        self.ue_encoder = UEFeedbackEncoder(
            2 * self.cfg.n_pilots,
            self.cfg.ue_hidden,
            fsq=self.fsq,
            latent_rms_norm=False,
            norm=self.cfg.ue_norm,
        )
        self.decoder = DirectWDecoder(
            self.ue_encoder.feat_dim, self.cfg, device=device)
        self.front_end_frozen = False

    def parameter_breakdown(self) -> dict[str, int]:
        """Return trainable counts for the blocks shown by the sweep banner."""

        def count(module: nn.Module) -> int:
            return sum(
                parameter.numel() for parameter in module.parameters()
                if parameter.requires_grad)

        return {
            "pilots": count(self.pilots),
            "ue_encoder": count(self.ue_encoder),
            "decoder_mixer": (
                count(self.decoder.embedding)
                + count(self.decoder.user_mixer)
            ),
            "decoder_head": count(self.decoder.weight_head),
            "decoder_total": count(self.decoder),
            "total": count(self),
        }

    def load_front_end(
        self,
        checkpoint_path: str,
        device,
        *,
        freeze: bool = False,
    ) -> None:
        """Load a shape-compatible Type-I front end from a checkpoint."""

        stored = torch.load(
            checkpoint_path, map_location=device, weights_only=False)
        state = stored.get("model_state_dict", stored)
        prefixes = ("pilots.", "fsq.", "ue_encoder.")
        front_end = {
            key: value for key, value in state.items()
            if key.startswith(prefixes)
        }

        try:
            missing, unexpected = self.load_state_dict(front_end, strict=False)
        except RuntimeError as error:
            raise ValueError(
                "front-end checkpoint is incompatible with this Type-I "
                "feedback geometry") from error

        if freeze:
            for module in (self.pilots, self.ue_encoder):
                for parameter in module.parameters():
                    parameter.requires_grad_(False)
            self.front_end_frozen = True

    def train(self, mode: bool = True):
        super().train(mode)
        if self.front_end_frozen:
            self.pilots.eval()
            self.ue_encoder.eval()
        return self

    def encode(
        self,
        channels: torch.Tensor,
        noise_std: float,
        *,
        noise: torch.Tensor | None = None,
        quantize_feedback: bool = True,
    ) -> torch.Tensor:
        observation = self.pilots(channels)
        if noise is None:
            noise = torch.randn_like(observation)
        return self.ue_encoder(
            observation + float(noise_std) * noise,
            quantize=quantize_feedback,
        )

    def forward(
        self,
        channels: torch.Tensor,
        noise_std: float,
        *,
        noise: torch.Tensor | None = None,
        quantize_feedback: bool = True,
    ) -> DirectWOutput:
        feedback = self.encode(
            channels,
            noise_std,
            noise=noise,
            quantize_feedback=quantize_feedback,
        )
        output = self.decoder(feedback)
        output.feedback = feedback
        return output

    @property
    def dictionary(self) -> torch.Tensor:
        return self.decoder.dictionary
