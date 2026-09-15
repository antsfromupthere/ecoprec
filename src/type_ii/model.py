from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from src.common.quantizers import FSQ
from src.common.feedback_encoder import UEFeedbackEncoder
from src.common.data import LearnablePilots

from .codebook import build_dictionary
from .config import DirectWConfig
from .projection import (
    amplitude_alphabet,
    gather_selected,
    phase_alphabet,
    project_one_row_top_l,
    quantize_relative_amplitude,
    quantized_phase,
    straight_through,
)
from src.common.transformer import SharedTransformerEncoder


@dataclass
class DirectWOutput:
    weights: torch.Tensor
    basis_coefficients: torch.Tensor
    v_soft: torch.Tensor
    v_continuous: torch.Tensor
    v_strict: torch.Tensor
    hard_mask: torch.Tensor
    surrogate_mask: torch.Tensor
    row_index: torch.Tensor
    support_index: torch.Tensor
    phase_index: torch.Tensor
    selected_phase_index: torch.Tensor
    amplitude_index: torch.Tensor
    power: torch.Tensor
    captured_energy: torch.Tensor
    row_margin: torch.Tensor
    feedback: torch.Tensor | None = None

    def deployed(self, strict_amplitude: bool = True) -> torch.Tensor:
        return self.v_strict if strict_amplitude else self.v_continuous


class DirectWDecoder(nn.Module):
    """Finite feedback -> complex w field -> constrained precoders."""

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
        # The sole BS output head.  Re/Im pairs directly represent w[o,n].
        self.weight_head = nn.Linear(cfg.d_model, 2 * cfg.O1 * cfg.N1)
        nn.init.xavier_uniform_(self.weight_head.weight, gain=0.1)
        nn.init.zeros_(self.weight_head.bias)

        self.register_buffer(
            "dictionary", build_dictionary(cfg.N1, cfg.O1, device=device))
        self.register_buffer(
            "phase_roots", phase_alphabet(cfg.N_psk, device=device))
        self.register_buffer(
            "amp_values", amplitude_alphabet(cfg.n_amp_levels, device=device))

    def weights_from_feedback(self, feedback: torch.Tensor) -> torch.Tensor:
        """Return the only learned action, w[B,K,O1,N1] complex64."""

        token = self.user_mixer(self.embedding(feedback))
        pair = self.weight_head(token).reshape(
            *token.shape[:2], self.cfg.O1, self.cfg.N1, 2)
        return torch.complex(pair[..., 0], pair[..., 1])

    def _power_shares(self, score: torch.Tensor) -> torch.Tensor:
        """Relative |w| scale across users is the power allocator."""

        total = score.sum(dim=-1, keepdim=True)
        equal = torch.full_like(score, 1.0 / score.shape[-1])
        learned = score / total.clamp_min(1e-12)
        learned = torch.where(total > 1e-12, learned, equal)
        floor = self.cfg.power_floor_fraction
        return (1.0 - floor) * learned + floor * equal

    def _unit_direction(self, unnormalized: torch.Tensor) -> torch.Tensor:
        norm = unnormalized.norm(dim=-1, keepdim=True)
        unit = unnormalized / norm.clamp_min(1e-12)
        fallback = self.dictionary[0, 0].view(
            *((1,) * (unnormalized.ndim - 1)), self.cfg.Nt)
        return torch.where((norm > 1e-12), unit, fallback)

    def _apply_power(
        self,
        unnormalized: torch.Tensor,
        score: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        power = self.cfg.tx_power * self._power_shares(score)
        direction = self._unit_direction(unnormalized)
        return direction * power.sqrt().unsqueeze(-1).to(direction.dtype), power

    def precoders_from_weights(
        self,
        weights: torch.Tensor,
        *,
        use_ste: bool | None = None,
    ) -> DirectWOutput:

        if weights.shape[-2:] != (self.cfg.O1, self.cfg.N1):
            raise ValueError(
                f"expected [...,{self.cfg.O1},{self.cfg.N1}] weights, "
                f"got {tuple(weights.shape)}")
        use_ste = self.training if use_ste is None else bool(use_ste)

        soft_unscaled = torch.einsum(
            "bkon,ont->bkt", weights, self.dictionary)
        soft_score = soft_unscaled.abs().square().sum(dim=-1)
        v_soft, _ = self._apply_power(soft_unscaled, soft_score)

        basis_coefficients = torch.einsum(
            "ont,bkt->bkon", self.dictionary.conj(), soft_unscaled)
        amplitude = basis_coefficients.abs()
        energy = amplitude.square()
        phase_unit, phase_index = quantized_phase(
            basis_coefficients, self.phase_roots, use_ste=use_ste)

        projection = project_one_row_top_l(
            energy,
            self.cfg.L,
            row_temperature=self.cfg.row_temperature,
            support_temperature=self.cfg.support_temperature,
        )
        hard_mask = projection["hard_mask"]
        surrogate_mask = projection["surrogate_mask"]
        mask = (straight_through(hard_mask, surrogate_mask)
                if use_ste else hard_mask)

        continuous_coefficients = (
            phase_unit * amplitude.to(phase_unit.dtype)
            * mask.to(phase_unit.dtype)
        )
        continuous_unscaled = torch.einsum(
            "bkon,ont->bkt", continuous_coefficients, self.dictionary)
        v_continuous, power = self._apply_power(
            continuous_unscaled, soft_score)

        selected_max = (amplitude * hard_mask).amax(
            dim=(-1, -2), keepdim=True)
        relative = amplitude / selected_max.clamp_min(1e-12)
        relative = torch.where(
            selected_max > 1e-12, relative, torch.ones_like(relative))
        quantized_amplitude, amplitude_index_field = \
            quantize_relative_amplitude(
                relative, self.amp_values, use_ste=use_ste)
        strict_coefficients = (
            phase_unit * quantized_amplitude.to(phase_unit.dtype)
            * mask.to(phase_unit.dtype)
        )
        strict_unscaled = torch.einsum(
            "bkon,ont->bkt", strict_coefficients, self.dictionary)
        strict_direction = self._unit_direction(strict_unscaled)
        v_strict = strict_direction \
            * power.sqrt().unsqueeze(-1).to(strict_direction.dtype)

        row_index = projection["row_index"]
        support_index = projection["support_index"]
        selected_phase_index = gather_selected(
            phase_index, row_index, support_index)
        amplitude_index = gather_selected(
            amplitude_index_field, row_index, support_index)
        retained_energy = (energy * hard_mask).sum(dim=(-1, -2))
        captured_energy = torch.where(
            soft_score > 1e-12,
            retained_energy / soft_score.clamp_min(1e-12),
            torch.ones_like(soft_score),
        ).clamp(max=1.0)

        return DirectWOutput(
            weights=weights,
            basis_coefficients=basis_coefficients,
            v_soft=v_soft,
            v_continuous=v_continuous,
            v_strict=v_strict,
            hard_mask=hard_mask,
            surrogate_mask=surrogate_mask,
            row_index=row_index,
            support_index=support_index,
            phase_index=phase_index,
            selected_phase_index=selected_phase_index,
            amplitude_index=amplitude_index,
            power=power,
            captured_energy=captured_energy,
            row_margin=projection["row_margin"],
        )

    def forward(self, feedback: torch.Tensor) -> DirectWOutput:
        return self.precoders_from_weights(
            self.weights_from_feedback(feedback))


class SoftDFTWNet(nn.Module):
    """Pilots -> B-bit feedback -> direct complex w field -> precoders."""
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

        stored = torch.load(
            checkpoint_path, map_location=device, weights_only=False)
        state = stored.get("model_state_dict", stored)
        prefixes = ("pilots.", "fsq.", "ue_encoder.")
        front_end = {
            key: value for key, value in state.items()
            if key.startswith(prefixes)
        }

        missing, unexpected = self.load_state_dict(front_end, strict=False)
        missing_front_end = [
            key for key in missing if key.startswith(prefixes)]
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

