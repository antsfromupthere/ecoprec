
from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
import os


_INVALID_WINDOWS_CHARS = set('<>:"/\\|?*')


def _plain_config(config) -> dict:
    if is_dataclass(config) and not isinstance(config, type):
        value = asdict(config)
    elif isinstance(config, dict):
        value = config
    return json.loads(json.dumps(value))


def _changed_fields(expected: dict, stored: dict | None) -> list[str]:
    stored = stored or {}
    return sorted(
        key
        for key in set(expected) | set(stored)
        if expected.get(key) != stored.get(key)
    )


def load_existing_run(
    run_path: str,
    model_cfg,
    train_cfg,
    *,
    overwrite_mismatched: bool = True,
    overwrite_existing: bool = False,
    initialization_identity: dict | None = None,
) -> dict | None:

    if not os.path.isfile(run_path):
        return None
    if overwrite_existing:
        print(f"[overwrite] rerunning existing run: {run_path}")
        return None

    try:
        with open(run_path, encoding="utf-8") as stream:
            result = json.load(stream)
    except (OSError, json.JSONDecodeError) as error:
        if overwrite_mismatched:
            print(f"[overwrite] unreadable run: {run_path}")
            return None
        raise RuntimeError(f"cannot reuse {run_path!r}") from error

    train_changes = _changed_fields(
        _plain_config(train_cfg),
        result.get("train_config"),
    )
    model_changes = _changed_fields(
        _plain_config(model_cfg),
        result.get("model_config"),
    )
    initialization_changed = (
        result.get("initialization_identity") != initialization_identity
    )
    if not train_changes and not model_changes and not initialization_changed:
        print(f"[skip] exact completed run: {run_path}")
        return result

    differences = []
    if train_changes:
        differences.append(f"TrainConfig {train_changes}")
    if model_changes:
        differences.append(f"model config {model_changes}")
    if initialization_changed:
        differences.append("initialization source")
    detail = "; ".join(differences)
    if overwrite_mismatched:
        print(f"[overwrite] mismatched run: {run_path}")
        print(f"[overwrite] changed fields: {detail}")
        return None


def method_component(name: str) -> str:

    value = str(name).strip()
    if (
        not value
        or value in {".", ".."}
        or any(character in _INVALID_WINDOWS_CHARS for character in value)
    ):
        raise ValueError(
            f"training.name must be a safe folder name, got {name!r}"
        )
    return value


def prepare_layout(
    output_root: str,
    sweep_variable: str,
    codebook: str,
    method_name: str,
) -> tuple[str, str, str]:
    """Return ``(sweep_root, method_folder, summary_dir)``."""

    method = method_component(method_name)
    sweep_root = os.path.abspath(
        os.path.join(output_root, f"sweep_{sweep_variable}")
    )
    summary_dir = os.path.join(
        sweep_root,
        "results",
        codebook,
        method,
    )
    return sweep_root, method, summary_dir


def run_directory(
    sweep_root: str,
    sweep_variable: str,
    value_label: str,
    codebook: str,
    method: str,
    seed: int,
) -> str:

    return os.path.join(
        sweep_root,
        f"{sweep_variable}_{value_label}",
        codebook,
        method,
        f"seed_{seed}",
    )


__all__ = [
    "load_existing_run",
    "method_component",
    "prepare_layout",
    "run_directory",
]
