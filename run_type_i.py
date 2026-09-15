"""Run the proposed method for Type-I feedback."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace
from functools import partial
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from src.common.quantizers import fsq_bits
from src.type_i import DirectWConfig, TrainConfig
from src.type_i.sweep import run_sweep
from src.type_i.train import train


FSQ_LEVELS_BY_N1 = {
    4: (4, 4),
    8: (6, 5),
}


class N1TunedDirectWConfig(DirectWConfig):
    def __post_init__(self) -> None:
        super().__post_init__()
        levels = FSQ_LEVELS_BY_N1.get(self.N1)
        if levels is None:
            return
        capacity = fsq_bits(levels)
        if capacity > self.feedback_bits + 1e-9:
            raise ValueError(
                f"manual FSQ capacity {capacity:.4f} exceeds the "
                f"{self.feedback_bits}-bit budget for N1={self.N1}"
            )
        object.__setattr__(self, "fsq_levels", levels)
        object.__setattr__(self, "fsq_target_bits_per_feature", None)


BASE_CONFIG = N1TunedDirectWConfig(
    N1=16,
    O1=4,
    K=4,
    tx_power=1.0,
    n_pilots=6,
    fsq_bound="tanh",
    fsq_target_bits_per_feature=2.0,
    ue_hidden=(256, 128, 64),
    ue_norm="layer",
    learn_pilots=True,
    d_model=172,
    attn_heads=4,
    attn_layers=2,
    ffn_dim=264,
    dropout=0.0,
    beam_temperature=0.5,
    power_floor_fraction=0.0,
)

TRAIN_CONFIG = TrainConfig(
    name="fsq_transformer_type_i",
    epochs=1500,
    batches_per_epoch=20,
    batch_size=1024,
    learning_rate=2e-4,
    weight_decay=0.0,
    grad_clip=1e2,
    lr_warmup_epochs=0,
    feedback_warmup_epochs=50,
    soft_rate_weight=0.5,
    soft_rate_anneal_epochs=300,
    projection_weight=0.0,
    eval_every=5,
    eval_size=10000,
    compute_diagnostics=False,
    seed=43,
    val_seed=2024,
    Lp=2,
    snr_db=10.0,
    mainlobe_ue_deg=0.0,
    halfbw_ue_deg=40.0,
    lsf_ue=0.0,
    warm_start_front_end=False,
    freeze_front_end=False,
    ema_decay=0.0,
    early_patience=500,
    early_min_epochs=1000,
    save_best=True,
)

SWEEPS = {
    "n_pilots": ([2, 4, 8, 10, 12], {}),
    "Lp": ([1, 2, 3, 4, 5, 6], {}),
    "N1": ([8, 12, 16, 4, 6], {}),
    "K": ([1, 2, 3, 4, 5, 6], {}),
    "snr": ([0, 5, 15, 20, 25], {}),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sweep", choices=("all", *SWEEPS), default="Lp",
        help="one sweep to run (default: Lp)",
    )
    parser.add_argument(
        "--values", nargs="+",
        help="optional values for one selected sweep; custom values run from scratch",
    )
    parser.add_argument(
        "--output-root", type=Path, default=PROJECT_ROOT / "outputs",
        help="result directory (default: <project>/outputs)",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument(
        "--overwrite-existing", action="store_true",
        help="retrain runs whose saved configuration matches",
    )
    parser.add_argument(
        "--overwrite-mismatched", action="store_true",
        help="replace runs whose saved configuration differs",
    )
    parser.add_argument(
        "--epochs", type=int,
        help="override the proposed 1500-epoch schedule",
    )
    parser.add_argument(
        "--batches-per-epoch", type=int,
        help="override the proposed 20 batches per epoch",
    )
    parser.add_argument("--batch-size", type=int, help="override batch size 1024")
    parser.add_argument("--eval-size", type=int, help="override evaluation set size 10000")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.sweep == "all" and args.values:
        parser.error("--values can only be used with one selected sweep")
    training = replace(
        TRAIN_CONFIG,
        **{
            key: value
            for key, value in {
                "epochs": args.epochs,
                "batches_per_epoch": args.batches_per_epoch,
                "batch_size": args.batch_size,
                "eval_size": args.eval_size,
            }.items()
            if value is not None
        },
    )
    selected = SWEEPS.items() if args.sweep == "all" else [(args.sweep, SWEEPS[args.sweep])]
    run = partial(run_sweep, train_fn=train)
    for sweep_variable, (sweep_values, sources) in selected:
        if args.values:
            value_type = float if sweep_variable == "snr" else int
            sweep_values = [value_type(value) for value in args.values]
            sources = {}
        run(
            BASE_CONFIG,
            training,
            sweep_variable=sweep_variable,
            sweep_values=sweep_values,
            checkpoint_source_by_sweep_value=sources,
            output_root=os.fspath(args.output_root),
            device=args.device,
            overwrite_mismatched=args.overwrite_mismatched,
            overwrite_existing=args.overwrite_existing,
        )


if __name__ == "__main__":
    main()
