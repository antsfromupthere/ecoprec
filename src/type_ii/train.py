""" trainer for the proposed method.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict

import torch

from src.common.checkpoint import initialize_albert_from_checkpoint
from src.common.runtime import get_device, set_seed
from src.common.data import draw_channel_batch
from src.common.rate import snr_db_to_noise_power

from .config import DirectWConfig, TrainConfig
from .model import SoftDFTWNet
from .objective import direct_w_objective
from .projection import audit_strict_output


class _EMA:
    """Optimizer-step EMA used only for evaluation/checkpoint readout."""

    def __init__(self, parameters, decay: float):
        self.parameters = [
            parameter for parameter in parameters if parameter.requires_grad]
        self.decay = float(decay)
        self.shadow = [
            parameter.detach().clone() for parameter in self.parameters]
        self.backup = None

    @torch.no_grad()
    def update(self) -> None:
        for shadow, parameter in zip(self.shadow, self.parameters):
            shadow.mul_(self.decay).add_(
                parameter, alpha=1.0 - self.decay)

    @torch.no_grad()
    def swap_in(self) -> None:
        self.backup = [
            parameter.detach().clone() for parameter in self.parameters]
        for parameter, shadow in zip(self.parameters, self.shadow):
            parameter.copy_(shadow)

    @torch.no_grad()
    def swap_out(self) -> None:
        for parameter, backup in zip(self.parameters, self.backup):
            parameter.copy_(backup)
        self.backup = None


def _soft_weight(cfg: TrainConfig, epoch: int) -> float:
    if cfg.soft_rate_weight == 0:
        return 0.0
    if cfg.soft_rate_anneal_epochs == 0:
        return cfg.soft_rate_weight
    remaining = max(0.0, 1.0 - epoch / cfg.soft_rate_anneal_epochs)
    return cfg.soft_rate_weight * remaining


def _draw(cfg: DirectWConfig, train_cfg: TrainConfig,
          count: int, device: torch.device) -> torch.Tensor:
    return draw_channel_batch(
        count,
        cfg.N1,
        cfg.K,
        train_cfg.Lp,
        train_cfg.lsf_ue,
        train_cfg.mainlobe_ue_deg,
        train_cfg.halfbw_ue_deg,
        device,
    )


@torch.no_grad()
def evaluate(
    model: SoftDFTWNet,
    channels: torch.Tensor,
    pilot_noise: torch.Tensor,
    noise_std: float,
    noise_power: float,
    *,
    compute_diagnostics: bool = False,
) -> dict:
    model.eval()
    output = model(
        channels,
        noise_std,
        noise=pilot_noise,
        quantize_feedback=True,
    )
    objective = direct_w_objective(
        output,
        channels,
        noise_power,
        include_diagnostics=compute_diagnostics,
    )
    result = objective.scalars()
    if compute_diagnostics:
        result.update(
            captured_energy=float(output.captured_energy.mean()),
            row_margin=float(output.row_margin.mean()),
            power_min=float(output.power.min()),
            power_max=float(output.power.max()),
            audit=audit_strict_output(
                output, model.dictionary, model.cfg),
        )
    return result


def train(
    model_cfg: DirectWConfig | None = None,
    train_cfg: TrainConfig | None = None,
    *,
    device: str = "auto",
    output_dir: str | None = None,
    verbose: bool = True,
    initialization_checkpoint: str | None = None,
    initialization_sweep_variable: str | None = None,
    initialization_metadata: dict | None = None,
) -> dict:
    """Train and return a JSON-serializable run record."""

    model_cfg = model_cfg or DirectWConfig()
    train_cfg = train_cfg or TrainConfig()
    run_device = get_device(device)
    noise_power = snr_db_to_noise_power(
        train_cfg.snr_db, model_cfg.tx_power)
    noise_std = math.sqrt(noise_power / 2.0)

    set_seed(train_cfg.seed)
    model = SoftDFTWNet(model_cfg, device=run_device).to(run_device)
    initialization = None
    if initialization_checkpoint is not None:
        initialization = initialize_albert_from_checkpoint(
        model,
        model_cfg,
        initialization_checkpoint,
        sweep_variable=initialization_sweep_variable,
    )
        if initialization_metadata is not None:
            initialization["identity"] = initialization_metadata
    if train_cfg.warm_start_front_end or train_cfg.freeze_front_end:
        model.load_front_end(
            train_cfg.front_end_checkpoint,
            run_device,
            freeze=train_cfg.freeze_front_end,
        )
    parameters = [parameter for parameter in model.parameters()
                  if parameter.requires_grad]
    optimizer = torch.optim.Adam(
        parameters,
        lr=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
    )
    ema = _EMA(parameters, train_cfg.ema_decay) \
        if train_cfg.ema_decay > 0 else None
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    set_seed(train_cfg.val_seed)
    validation = _draw(
        model_cfg, train_cfg, train_cfg.eval_size, run_device)
    validation_noise = torch.randn(
        train_cfg.eval_size,
        model_cfg.K,
        2 * model_cfg.n_pilots,
        device=run_device,
    )
    set_seed(train_cfg.seed + 1)

    history: list[dict] = []
    best_strict_rate = -float("inf")
    best_epoch = 0
    initial_metrics = None
    if initialization is not None:
        initial_metrics = evaluate(
            model,
            validation,
            validation_noise,
            noise_std,
            noise_power,
            compute_diagnostics=train_cfg.compute_diagnostics,
        )
        best_strict_rate = initial_metrics["strict_rate"]
        if train_cfg.save_best:
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": asdict(model_cfg),
                    "train_config": asdict(train_cfg),
                    "epoch": 0,
                    "best_strict_rate": best_strict_rate,
                    "initialization": initialization,
                },
                os.path.join(output_dir, "best.pt"),
            )
    started = time.time()
    for epoch in range(train_cfg.epochs):
        model.train()
        warmup = min(
            1.0,
            (epoch + 1) / max(train_cfg.lr_warmup_epochs, 1),
        ) if train_cfg.lr_warmup_epochs else 1.0
        for group in optimizer.param_groups:
            group["lr"] = train_cfg.learning_rate * warmup

        running = None
        for _ in range(train_cfg.batches_per_epoch):
            channels = _draw(
                model_cfg, train_cfg, train_cfg.batch_size, run_device)
            output = model(
                channels,
                noise_std,
                quantize_feedback=(
                    train_cfg.freeze_front_end
                    or epoch >= train_cfg.feedback_warmup_epochs),
            )
            value = direct_w_objective(
                output,
                channels,
                noise_power,
                soft_rate_weight=_soft_weight(train_cfg, epoch),
                projection_weight=train_cfg.projection_weight,
            )
            optimizer.zero_grad(set_to_none=True)
            value.loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                parameters, train_cfg.grad_clip)
            if bool(torch.isfinite(value.loss) & torch.isfinite(grad_norm)):
                optimizer.step()
                if ema is not None:
                    ema.update()
            running = value

        if (epoch + 1) % train_cfg.eval_every == 0 \
                or epoch + 1 == train_cfg.epochs:
            live_strict_rate = None
            if ema is not None:
                if train_cfg.compute_diagnostics:
                    live_metrics = evaluate(
                        model,
                        validation,
                        validation_noise,
                        noise_std,
                        noise_power,
                    )
                    live_strict_rate = live_metrics["strict_rate"]
                ema.swap_in()
            metrics = evaluate(
                model,
                validation,
                validation_noise,
                noise_std,
                noise_power,
                compute_diagnostics=train_cfg.compute_diagnostics,
            )
            metrics.update(
                epoch=epoch + 1,
                soft_weight=_soft_weight(train_cfg, epoch),
                train_loss=float(running.loss.detach()) if running else None,
                train_soft_rate=(
                    float(running.soft_rate.detach())
                    if running is not None and running.soft_rate is not None
                    else None
                ),
                live_strict_rate=live_strict_rate,
                seconds=round(time.time() - started, 2),
            )
            quantization_active = (
                train_cfg.freeze_front_end
                or epoch >= train_cfg.feedback_warmup_epochs)
            improved = (
                quantization_active
                and metrics["strict_rate"] > best_strict_rate)
            if improved:
                best_strict_rate = metrics["strict_rate"]
                best_epoch = epoch + 1
                if train_cfg.save_best:
                    torch.save(
                        {
                            "model_state_dict": model.state_dict(),
                            "model_config": asdict(model_cfg),
                            "train_config": asdict(train_cfg),
                            "epoch": epoch + 1,
                            "best_strict_rate": best_strict_rate,
                            "initialization": initialization,
                        },
                        os.path.join(output_dir, "best.pt"),
                    )
            if ema is not None:
                ema.swap_out()
            history.append(metrics)
            if verbose:
                message = (
                    f"ep {epoch + 1:>4} | strict={metrics['strict_rate']:.4f} "
                    f"| best={best_strict_rate:.4f}")
                if metrics["train_soft_rate"] is not None:
                    message += (
                        f" | train_soft={metrics['train_soft_rate']:.4f}")
                if train_cfg.compute_diagnostics:
                    audit = metrics["audit"]
                    message += (
                        f" live={live_strict_rate:.4f}"
                        if live_strict_rate is not None else "")
                    message += (
                        f" | cont={metrics['continuous_rate']:.4f} "
                        f"soft={metrics['soft_rate']:.4f} "
                        f"captured={metrics['captured_energy']:.3f} "
                        f"viol={audit['total_violations']}")
                message += f" | {metrics['seconds']:.1f}s"
                message += (
                    "  (warmup)" if not quantization_active else
                    "  (saved)" if improved and train_cfg.save_best else "")
                print(message)
            if train_cfg.early_patience \
                    and math.isfinite(best_strict_rate) \
                    and (epoch + 1) >= train_cfg.early_min_epochs \
                    and (epoch + 1) - best_epoch >= train_cfg.early_patience:
                if verbose:
                    print(
                        f"early stop @ {epoch + 1}: best strict rate "
                        f"was at epoch {best_epoch}")
                break

    if ema is not None:
        ema.swap_in()
    final_metrics = evaluate(
        model,
        validation,
        validation_noise,
        noise_std,
        noise_power,
        compute_diagnostics=train_cfg.compute_diagnostics,
    )

    record = {
        "method": "type_ii",
        "model_config": asdict(model_cfg),
        "train_config": asdict(train_cfg),
        "trainable_parameters": sum(p.numel() for p in parameters),
        "initialization_identity": initialization_metadata,
        "initialization": initialization,
        "initial": initial_metrics,
        "history": history,
        "final": final_metrics,
        "best_strict_rate": best_strict_rate,
        "best_epoch": best_epoch,
        "epochs_run": history[-1]["epoch"] if history else train_cfg.epochs,
        "wall_seconds": round(time.time() - started, 2),
    }
    if output_dir:
        with open(os.path.join(output_dir, "run.json"), "w",
                  encoding="utf-8") as stream:
            json.dump(record, stream, indent=2)
        torch.save(
            {"model_state_dict": model.state_dict(),
             "model_config": asdict(model_cfg),
             "initialization": initialization},
            os.path.join(output_dir, "model.pt"),
        )
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batches-per-epoch", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-size", type=int, default=128)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--diagnostics", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    train_cfg = TrainConfig(
        epochs=args.epochs,
        batches_per_epoch=args.batches_per_epoch,
        batch_size=args.batch_size,
        eval_size=args.eval_size,
        eval_every=args.eval_every,
        compute_diagnostics=args.diagnostics,
    )
    train(
        DirectWConfig(),
        train_cfg,
        device=args.device,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()

