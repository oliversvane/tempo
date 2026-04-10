from __future__ import annotations

import argparse
from pathlib import Path

from .models import StreamingModelConfig
from .train import DataConfig, TrainingConfig, run_optuna_study, train_triplet_model


def _common_config_from_args(args: argparse.Namespace) -> TrainingConfig:
    data_config = DataConfig(
        manifest_path=args.manifest,
        chunk_seconds=args.chunk_seconds,
        labels_per_batch=args.labels_per_batch,
        samples_per_label=args.samples_per_label,
        train_batches_per_epoch=args.train_batches_per_epoch,
        val_batches_per_epoch=args.val_batches_per_epoch,
        train_num_workers=args.train_num_workers,
        val_num_workers=args.val_num_workers,
        pin_memory=args.pin_memory,
        prefetch_factor=args.prefetch_factor,
        basic_emotions_only=not args.all_emotions,
        val_ratio=args.val_ratio,
    )
    model_config = StreamingModelConfig(
        model_type=args.model_type,
        sample_rate=data_config.sample_rate,
        output_dim=args.output_dim,
        activation=args.activation,
    )
    return TrainingConfig(
        output_dir=args.output_dir,
        max_epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        margin=args.margin,
        loss_name=args.loss_name,
        loss_temperature=args.loss_temperature,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        device=args.device,
        log_every_n_steps=args.log_every_n_steps,
        progress_bar_refresh_rate=args.progress_bar_refresh_rate,
        embedding_log_limit=args.embedding_log_limit,
        model=model_config,
        data=data_config,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Train streaming emotion embedding models.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_shared_arguments(command_parser: argparse.ArgumentParser) -> None:
        command_parser.add_argument(
            "--manifest",
            default="data/processed/manifest.csv",
            type=Path,
        )
        command_parser.add_argument("--output-dir", default="runs/default", type=Path)
        command_parser.add_argument("--epochs", default=12, type=int)
        command_parser.add_argument("--model-type", choices=["conformer", "rnnt"], default="conformer")
        command_parser.add_argument("--activation", choices=["relu", "gelu", "silu", "mish"], default="silu")
        command_parser.add_argument("--device", default="auto")
        command_parser.add_argument("--chunk-seconds", default=3.2, type=float)
        command_parser.add_argument("--labels-per-batch", default=4, type=int)
        command_parser.add_argument("--samples-per-label", default=4, type=int)
        command_parser.add_argument("--train-batches-per-epoch", default=None, type=int)
        command_parser.add_argument("--val-batches-per-epoch", default=128, type=int)
        command_parser.add_argument("--train-num-workers", default=-1, type=int)
        command_parser.add_argument("--val-num-workers", default=-1, type=int)
        command_parser.add_argument("--prefetch-factor", default=4, type=int)
        command_parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=None)
        command_parser.add_argument("--val-ratio", default=0.1, type=float)
        command_parser.add_argument("--learning-rate", default=3e-4, type=float)
        command_parser.add_argument("--weight-decay", default=1e-4, type=float)
        command_parser.add_argument("--margin", default=0.2, type=float)
        command_parser.add_argument(
            "--loss-name",
            choices=["batch_hard_triplet", "batch_all_triplet", "supervised_contrastive"],
            default="batch_hard_triplet",
        )
        command_parser.add_argument("--loss-temperature", default=0.1, type=float)
        command_parser.add_argument("--early-stopping-patience", default=5, type=int)
        command_parser.add_argument("--early-stopping-min-delta", default=0.0, type=float)
        command_parser.add_argument("--output-dim", default=128, type=int)
        command_parser.add_argument("--log-every-n-steps", default=1, type=int)
        command_parser.add_argument("--progress-bar-refresh-rate", default=1, type=int)
        command_parser.add_argument("--embedding-log-limit", default=512, type=int)
        command_parser.add_argument("--all-emotions", action="store_true")

    train_parser = subparsers.add_parser("train")
    add_shared_arguments(train_parser)

    tune_parser = subparsers.add_parser("tune")
    add_shared_arguments(tune_parser)
    tune_parser.add_argument("--trials", default=20, type=int)
    tune_parser.add_argument("--timeout", default=None, type=float)
    tune_parser.add_argument("--study-name", default="emotion_embeddings")
    tune_parser.add_argument("--storage", default=None)
    tune_parser.add_argument(
        "--tune-monitor",
        choices=["val/loss", "val/triplet_accuracy", "val/separation_gap"],
        default="val/triplet_accuracy",
    )
    tune_parser.add_argument("--tune-monitor-mode", choices=["min", "max"], default="max")

    args = parser.parse_args()
    config = _common_config_from_args(args)

    if args.command == "train":
        result = train_triplet_model(config)
        print(f"Best val loss: {result.best_val_loss:.6f}")
        print(
            f"Selected metric: {result.best_metric_name} "
            f"({result.best_metric_mode}) = {result.best_metric_value:.6f}"
        )
        print(f"Best epoch: {result.best_epoch}")
        print(f"Checkpoint: {result.best_checkpoint_path}")
        print(f"TensorBoard logs: {result.tensorboard_dir}")
        return 0

    study = run_optuna_study(
        config,
        n_trials=args.trials,
        study_name=args.study_name,
        storage=args.storage,
        timeout=args.timeout,
        monitor=args.tune_monitor,
        monitor_mode=args.tune_monitor_mode,
    )
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best {args.tune_monitor}: {study.best_value:.6f}")
    print(f"Best params: {study.best_params}")
    return 0
