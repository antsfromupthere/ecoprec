from __future__ import annotations

from dataclasses import asdict
import os

import torch


_FULL_MODEL_SWEEPS = {"snr", "snr_db", "Lp", "K"}


def _pilot_slot_mapping(
    source_count: int,
    target_count: int,
    antennas: int,
) -> list[int]:

    if target_count <= antennas:
        source_rows = [
            index * antennas // source_count
            for index in range(source_count)
        ]
        target_rows = [
            index * antennas // target_count
            for index in range(target_count)
        ]
        target_by_row = {
            row: index for index, row in enumerate(target_rows)
        }
        if all(row in target_by_row for row in source_rows):
            return [target_by_row[row] for row in source_rows]
    return list(range(source_count))


def initialize_albert_from_checkpoint(
    model,
    target_cfg,
    checkpoint_path: str,
    *,
    sweep_variable: str,
) -> dict:

    device = next(model.parameters()).device
    payload = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    source_cfg = payload.get("model_config")
    source_state = payload.get("model_state_dict")
    target_values = asdict(target_cfg)
    ignored = (
        {"K"}
        if sweep_variable == "K"
        else {"n_pilots"}
        if sweep_variable == "n_pilots"
        else set()
    )
    mismatched = {
        key: (source_cfg.get(key), target_values.get(key))
        for key in set(source_cfg) | set(target_values)
        if key not in ignored
        and source_cfg.get(key) != target_values.get(key)
    }
    if mismatched:
        raise ValueError(
            "initialization checkpoint has incompatible model settings: "
            f"{mismatched}"
        )

    if sweep_variable in _FULL_MODEL_SWEEPS:
        model.load_state_dict(source_state, strict=True)
        return {
            "transfer_kind": "exact_full_model",
            "source_checkpoint": os.path.abspath(checkpoint_path),
        }

    source_count = int(source_cfg["n_pilots"])
    target_count = int(target_cfg.n_pilots)
    if source_count == target_count:
        model.load_state_dict(source_state, strict=True)
        return {
            "transfer_kind": "exact_full_model",
            "source_checkpoint": os.path.abspath(checkpoint_path),
        }

    mapping = _pilot_slot_mapping(
        source_count,
        target_count,
        target_cfg.Nt,
    )
    target_state = model.state_dict()
    pilot_keys = ("pilots.Xp_r", "pilots.Xp_i")
    first_weight_key = "ue_encoder.net.0.weight"
    special = {*pilot_keys, first_weight_key}

    for key, target in target_state.items():
        if key in special:
            continue
        source = source_state.get(key)
        if source is None or source.shape != target.shape:
            raise ValueError(
                f"cannot transfer parameter {key!r}: source "
                f"{None if source is None else tuple(source.shape)} versus "
                f"target {tuple(target.shape)}"
            )
        target_state[key] = source.to(
            device=target.device,
            dtype=target.dtype,
        )

    for key in pilot_keys:
        source = source_state[key]
        target = target_state[key].clone()
        for source_index, target_index in enumerate(mapping):
            target[target_index] = source[source_index].to(
                device=target.device,
                dtype=target.dtype,
            )
        target_state[key] = target

    source_weight = source_state[first_weight_key]
    target_weight = torch.zeros_like(target_state[first_weight_key])
    for source_index, target_index in enumerate(mapping):
        target_weight[:, target_index] = source_weight[:, source_index]
        target_weight[:, target_count + target_index] = source_weight[
            :, source_count + source_index
        ]
    target_state[first_weight_key] = target_weight
    model.load_state_dict(target_state, strict=True)
    return {
        "transfer_kind": "nested_pilot_embedding",
        "source_checkpoint": os.path.abspath(checkpoint_path),
        "source_n_pilots": source_count,
        "target_n_pilots": target_count,
        "source_to_target_pilot_slots": mapping,
    }


__all__ = ["initialize_albert_from_checkpoint"]
