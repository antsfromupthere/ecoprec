"""Run the proposed method for Type-II feedback."""

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

from src.type_ii import DirectWConfig, TrainConfig
from src.type_ii.sweep import run_sweep
from src.type_ii.train import train


BASE_CONFIG = DirectWConfig(
    N1=16,
    O1=4,
    K=4,
    L=2,
    N_psk=4,
    amp_bits=3,
    phase_bits=2,
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
    row_temperature=0.5,
    support_temperature=1.0,
    power_floor_fraction=0.0,
    n_amp_levels=7,
)

TRAIN_CONFIG = TrainConfig(
    name="fsq_transformer_type_ii",
    epochs=1500,
    batches_per_epoch=20,
    batch_size=1024,
    learning_rate=2e-4,
    weight_decay=0.0,
    grad_clip=1e2,
    lr_warmup_epochs=0,
    feedback_warmup_epochs=0,
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
    early_min_epochs=500,
    save_best=True,
)

SWEEPS = {
    "n_pilots": ([2, 4, 6, 8, 10, 12], {8: 6, 10: 8, 12: 10}),
    "Lp": ([2, 1, 3, 4, 5, 6], {1: 2, 3: 2, 4: 3, 5: 4, 6: 5}),
    "K": ([4, 3, 2, 1, 5, 6], {3: 4, 2: 3, 1: 2, 5: 4, 6: 5}),
    "N1": ([12, 16, 8, 6, 4], {}),
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
            codebooks=("type_ii",),
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
