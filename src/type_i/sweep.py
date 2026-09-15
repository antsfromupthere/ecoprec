"""One-variable sweep loop for the standalone Type-I method."""

from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict, replace

from src.common.quantizers import fsq_bits
from src.common.sweep_io import load_existing_run, prepare_layout, run_directory

from .model import SoftDFTWNet


_MODEL_VARIABLES = {"N1", "O1", "K", "n_pilots"}
_TRAIN_VARIABLES = {"Lp", "snr", "snr_db"}


def _label(value) -> str:
    return format(value, ".12g") if isinstance(value, float) else str(value)


def _parameter_counts(model_cfg) -> dict[str, int]:
    return SoftDFTWNet(model_cfg).parameter_breakdown()


def _parameter_line(parameters: dict[str, int]) -> str:
    return (
        f"# parameters | mixer {parameters['decoder_mixer']:,} + "
        f"head {parameters['decoder_head']:,} = "
        f"decoder {parameters['decoder_total']:,} | "
        f"total {parameters['total']:,}"
    )


def _set_sweep_value(base, training, variable: str, value):
    if variable in _MODEL_VARIABLES:
        return replace(base, **{variable: value}), training
    if variable in _TRAIN_VARIABLES:
        field = "snr_db" if variable == "snr" else variable
        return base, replace(training, **{field: value})
    allowed = sorted(_MODEL_VARIABLES | _TRAIN_VARIABLES)
    raise ValueError(f"unknown sweep variable {variable!r}; choose from {allowed}")


def _summary_row(result, sweep_variable, value, output_dir) -> dict:
    final = result["final"]
    audit = final.get("audit", {})
    model = result["model_config"]
    train = result["train_config"]
    return {
        "method": result["train_config"]["name"],
        "sweep_variable": sweep_variable,
        "sweep_value": value,
        "codebook": "type_i",
        "seed": train["seed"],
        "N1": model["N1"],
        "O1": model["O1"],
        "n_beams": model["N1"] * model["O1"],
        "K": model["K"],
        "n_pilots": model["n_pilots"],
        "Lp": train["Lp"],
        "snr_db": train["snr_db"],
        "feedback_bits": model["feedback_bits"],
        "fsq_feature_dim": len(model["fsq_levels"]),
        "fsq_target_bits_per_feature": model.get(
            "fsq_target_bits_per_feature"),
        "fsq_levels": json.dumps(model["fsq_levels"]),
        "fsq_bits_used": fsq_bits(model["fsq_levels"]),
        "initialization_mode": (
            result.get("initialization", {}).get("transfer_kind")
            if result.get("initialization") else "from_scratch"),
        "initialization_source_value": (
            result.get("initialization_identity", {}).get("source_sweep_value")
            if result.get("initialization_identity") else None),
        "initial_strict_rate": (
            result.get("initial", {}).get("strict_rate")
            if result.get("initial") else None),
        "best_epoch": result["best_epoch"],
        "best_strict_rate": result["best_strict_rate"],
        "final_strict_rate": final["strict_rate"],
        "final_soft_rate": final.get("soft_rate"),
        "captured_energy": final.get("captured_energy"),
        "violations": audit.get("total_violations"),
        "epochs_run": result["epochs_run"],
        "wall_seconds": result["wall_seconds"],
        "output_dir": os.path.abspath(output_dir),
    }


def _anchor_reuse_result(
    checkpoint_path: str,
    model_cfg,
    train_cfg,
    initialization_identity: dict,
    run_path: str,
):

    anchor_run = os.path.join(os.path.dirname(checkpoint_path), "run.json")
    if not os.path.isfile(anchor_run):
        return None
    with open(anchor_run, "r", encoding="utf-8") as stream:
        anchor = json.load(stream)

    def canonical(value) -> str:
        return json.dumps(value, sort_keys=True, default=list)

    if canonical(anchor.get("model_config")) != canonical(asdict(model_cfg)) \
            or canonical(anchor.get("train_config")) \
            != canonical(asdict(train_cfg)):
        return None
    result = {
        **anchor,
        "initialization_identity": initialization_identity,
        "initialization": {
            **initialization_identity,
            "transfer_kind": "anchor_reuse_no_training",
        },
    }
    os.makedirs(os.path.dirname(run_path), exist_ok=True)
    with open(run_path, "w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    print(f"[reuse] anchor result recorded without retraining: {run_path}")
    return result


def _write_summary(rows: list[dict], results_dir: str) -> None:
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, "summary.csv")
    json_path = os.path.join(results_dir, "summary.json")
    with open(csv_path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with open(json_path, "w", encoding="utf-8") as stream:
        json.dump(rows, stream, indent=2)
    print(f"summary: {csv_path}")


def run_sweep(
    base,
    training,
    *,
    train_fn,
    sweep_variable: str,
    sweep_values,
    output_root: str,
    device: str = "auto",
    overwrite_mismatched: bool = True,
    overwrite_existing: bool = False,
    checkpoint_source_by_sweep_value: dict | None = None,
    continuation_learning_rate: float | None = None,
):

    values = list(sweep_values)

    checkpoint_sources = dict(checkpoint_source_by_sweep_value or {})
    positions = {value: index for index, value in enumerate(values)}
    uses_checkpoint = (
        training.warm_start_front_end or training.freeze_front_end)

    init_label = "checkpoint" if uses_checkpoint else "scratch"
    sweep_root, method, summary_dir = prepare_layout(
        output_root, sweep_variable, "type_i", training.name)
    rows = []
    completed = {}
    for value in values:
        model_cfg, train_cfg = _set_sweep_value(
            base, training, sweep_variable, value)
        source_spec = checkpoint_sources.get(value)
        source_value = source_spec
        initialization_identity = None
        initialization_checkpoint = None
        if source_spec is not None:
            if isinstance(source_spec, (str, os.PathLike)):
                initialization_checkpoint = os.fspath(source_spec)
                source_value = "anchor"
                source = None
            else:
                source = completed[source_spec]
                initialization_checkpoint = os.path.join(
                    source["output_dir"], "best.pt")
            initialization_identity = {
                "mode": "previous_sweep_best",
                "sweep_variable": sweep_variable,
                "source_sweep_value": source_value,
                "source_checkpoint": os.path.abspath(initialization_checkpoint),
            }
            if source is not None:
                initialization_identity.update(
                    source_best_epoch=source["result"]["best_epoch"],
                    source_best_strict_rate=(
                        source["result"]["best_strict_rate"]),
                )
            if continuation_learning_rate is not None:
                train_cfg = replace(
                    train_cfg, learning_rate=continuation_learning_rate)

        output_dir = run_directory(
            sweep_root,
            sweep_variable,
            _label(value),
            "type_i",
            method,
            train_cfg.seed,
        )
        run_path = os.path.join(output_dir, "run.json")
        result = load_existing_run(
            run_path,
            model_cfg,
            train_cfg,
            overwrite_mismatched=overwrite_mismatched,
            overwrite_existing=overwrite_existing,
            initialization_identity=initialization_identity,
        )
        if result is None and initialization_checkpoint is not None \
                and source_value == "anchor":
            result = _anchor_reuse_result(
                initialization_checkpoint,
                model_cfg,
                train_cfg,
                initialization_identity,
                run_path,
            )
        if result is None:
            parameters = _parameter_counts(model_cfg)
            run_init_label = (
                f"best@{source_value}" if source_spec is not None
                else init_label)
            print("#" * 78)
            print(
                f"# {training.name} | Type-I | {sweep_variable}={value} | "
                f"seed={train_cfg.seed} | init={run_init_label}")
            print(
                f"# beams={model_cfg.n_beams} | B={model_cfg.feedback_bits} | "
                f"fsq_levels={model_cfg.fsq_levels} | "
                f"used={fsq_bits(model_cfg.fsq_levels):.4f} bits")
            print(_parameter_line(parameters))
            print(f"# outputs: {output_dir}")
            print("#" * 78)
            result = train_fn(
                model_cfg,
                train_cfg,
                device=device,
                output_dir=output_dir,
                initialization_checkpoint=initialization_checkpoint,
                initialization_sweep_variable=(
                    sweep_variable if initialization_checkpoint else None),
                initialization_metadata=initialization_identity,
            )
        rows.append(_summary_row(
            result, sweep_variable, value, output_dir))
        _write_summary(rows, summary_dir)
        completed_dir = output_dir
        reuse_info = (result or {}).get("initialization") or {}
        if reuse_info.get("transfer_kind") == "anchor_reuse_no_training":
            completed_dir = os.path.dirname(
                reuse_info.get("source_checkpoint", ""))
        completed[value] = {"result": result, "output_dir": completed_dir}
    return rows
