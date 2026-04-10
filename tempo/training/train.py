from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import lightning.pytorch as pl
import matplotlib
import optuna
import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import Callback, EarlyStopping, LearningRateMonitor, ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.loggers import TensorBoardLogger

from tempo.datasets.utils import PROJECT_ROOT

from .data import (
    AdditiveNoiseAugment,
    EmotionStreamDataset,
    EmotionSubsetDataset,
    NoiseAugmentationConfig,
    build_triplet_dataloader_from_dataset,
    speaker_group_key,
    stratified_speaker_split_indices,
)
from .models import StreamingModelConfig, build_streaming_emotion_model
from .triplet import sequence_metric_loss

matplotlib.use("Agg")
from matplotlib import pyplot as plt


@dataclass(frozen=True)
class DataConfig:
    manifest_path: str | Path = PROJECT_ROOT / "data/processed/manifest.csv"
    sample_rate: int = 16_000
    chunk_seconds: float = 3.2
    labels_per_batch: int = 4
    samples_per_label: int = 4
    train_batches_per_epoch: int | None = None
    val_batches_per_epoch: int | None = 128
    train_num_workers: int = -1
    val_num_workers: int = -1
    pin_memory: bool | None = None
    prefetch_factor: int = 4
    basic_emotions_only: bool = True
    allowed_emotions: tuple[str, ...] | None = None
    excluded_emotions: tuple[str, ...] = ("other",)
    min_examples_per_emotion: int = 2
    min_duration_seconds: float = 0.25
    val_ratio: float = 0.1
    split_seed: int = 7
    noise_probability: float = 0.8
    snr_db_min: float = 8.0
    snr_db_max: float = 28.0
    gain_db_min: float = -4.0
    gain_db_max: float = 4.0
    random_crop: bool = True
    peak_normalize: bool = False


@dataclass(frozen=True)
class TrainingConfig:
    output_dir: str | Path = PROJECT_ROOT / "runs/default"
    max_epochs: int = 12
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    margin: float = 0.2
    loss_name: str = "batch_hard_triplet"
    loss_temperature: float = 0.1
    grad_clip_norm: float = 5.0
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 0.0
    seed: int = 7
    device: str = "auto"
    log_every_n_steps: int = 1
    progress_bar_refresh_rate: int = 1
    log_embeddings_every_n_epochs: int = 1
    embedding_log_limit: int = 512
    model: StreamingModelConfig = field(default_factory=StreamingModelConfig)
    data: DataConfig = field(default_factory=DataConfig)


@dataclass(frozen=True)
class TrainingResult:
    best_val_loss: float
    best_epoch: int
    best_metric_name: str
    best_metric_mode: str
    best_metric_value: float
    output_dir: Path
    best_checkpoint_path: Path
    tensorboard_dir: Path
    history_path: Path


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True), encoding="utf-8")


def _resolve_lightning_device(device_name: str) -> tuple[str, int | list[int]]:
    if device_name == "auto":
        if torch.cuda.is_available():
            return "gpu", 1
        if torch.backends.mps.is_available():
            return "mps", 1
        return "cpu", 1

    if device_name == "cpu":
        return "cpu", 1
    if device_name == "mps":
        return "mps", 1
    if device_name in {"gpu", "cuda"}:
        return "gpu", 1
    if device_name.startswith("cuda:"):
        return "gpu", [int(device_name.split(":", maxsplit=1)[1])]
    raise ValueError(f"Unsupported device specifier: {device_name}")


def _extract_scalar_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    extracted: dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, torch.Tensor):
            if value.ndim != 0:
                continue
            extracted[key] = float(value.detach().cpu().item())
        elif isinstance(value, (float, int)):
            extracted[key] = float(value)
    return extracted


def _validate_monitor_mode(mode: str) -> str:
    normalized = mode.strip().lower()
    if normalized not in {"min", "max"}:
        raise ValueError(f"Unsupported monitor mode: {mode}")
    return normalized


def _is_better_metric(candidate: float, best: float, *, mode: str) -> bool:
    if mode == "min":
        return candidate < best
    return candidate > best


def _resolve_num_workers(requested_workers: int, *, default_cap: int) -> int:
    if requested_workers >= 0:
        return requested_workers
    cpu_count = os.cpu_count() or 1
    if cpu_count <= 2:
        return 0
    return min(default_cap, max(1, cpu_count - 1))


def _resolve_pin_memory(pin_memory: bool | None, device_name: str) -> bool:
    if pin_memory is not None:
        return pin_memory
    if not torch.cuda.is_available():
        return False
    return device_name in {"auto", "gpu", "cuda"} or device_name.startswith("cuda:")


def _flatten_hparams(value: Any, *, prefix: str = "") -> dict[str, bool | int | float | str]:
    if isinstance(value, dict):
        flattened: dict[str, bool | int | float | str] = {}
        for key, item in value.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(_flatten_hparams(item, prefix=next_prefix))
        return flattened
    if isinstance(value, (list, tuple)):
        joined = ",".join(str(item) for item in value)
        return {prefix: joined}
    if isinstance(value, Path):
        return {prefix: str(value)}
    if isinstance(value, (bool, int, float, str)):
        return {prefix: value}
    if value is None:
        return {prefix: "none"}
    return {prefix: str(value)}


def _config_hparams(config: TrainingConfig, datamodule: "EmotionTripletDataModule") -> dict[str, bool | int | float | str]:
    hparams_dict = {
        "model": asdict(config.model),
        "training": {
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "margin": config.margin,
            "loss_name": config.loss_name,
            "loss_temperature": config.loss_temperature,
            "grad_clip_norm": config.grad_clip_norm,
            "early_stopping_patience": config.early_stopping_patience,
            "early_stopping_min_delta": config.early_stopping_min_delta,
            "max_epochs": config.max_epochs,
        },
        "data": {
            "chunk_seconds": config.data.chunk_seconds,
            "labels_per_batch": config.data.labels_per_batch,
            "samples_per_label": config.data.samples_per_label,
            "basic_emotions_only": config.data.basic_emotions_only,
            "noise_probability": config.data.noise_probability,
            "snr_db_min": config.data.snr_db_min,
            "snr_db_max": config.data.snr_db_max,
            "gain_db_min": config.data.gain_db_min,
            "gain_db_max": config.data.gain_db_max,
            "train_batches_per_epoch": config.data.train_batches_per_epoch or "auto",
            "val_batches_per_epoch": config.data.val_batches_per_epoch or "auto",
            "train_num_workers": datamodule.train_num_workers,
            "val_num_workers": datamodule.val_num_workers,
            "pin_memory": datamodule.pin_memory,
            "prefetch_factor": datamodule.prefetch_factor,
            "val_ratio": config.data.val_ratio,
        },
    }
    return _flatten_hparams(hparams_dict)


def _color_for_label(label: str) -> torch.Tensor:
    palette = {
        "anger": (0.85, 0.24, 0.20),
        "boredom": (0.55, 0.52, 0.44),
        "calm": (0.20, 0.63, 0.72),
        "disgust": (0.36, 0.61, 0.25),
        "excitement": (0.99, 0.66, 0.18),
        "fear": (0.47, 0.38, 0.78),
        "frustration": (0.79, 0.31, 0.53),
        "happiness": (0.97, 0.78, 0.22),
        "neutral": (0.50, 0.54, 0.58),
        "sadness": (0.24, 0.45, 0.76),
        "surprise": (0.93, 0.49, 0.16),
    }
    if label in palette:
        return torch.tensor(palette[label], dtype=torch.float32)

    digest = hashlib.blake2s(label.encode("utf-8"), digest_size=3).digest()
    return torch.tensor([byte / 255.0 for byte in digest], dtype=torch.float32)


def _project_embeddings_to_3d(embeddings: torch.Tensor) -> torch.Tensor:
    if embeddings.numel() == 0:
        return embeddings.new_zeros((0, 3))

    centered = embeddings.float() - embeddings.float().mean(dim=0, keepdim=True)
    if centered.size(0) == 1:
        return F.pad(centered, (0, max(0, 3 - centered.size(1))))[:, :3]

    target_dim = min(3, centered.size(0), centered.size(1))
    if target_dim == 0:
        return centered.new_zeros((centered.size(0), 3))

    try:
        _, _, principal_components = torch.pca_lowrank(centered, q=target_dim)
        reduced = centered @ principal_components[:, :target_dim]
    except RuntimeError:
        reduced = centered[:, :target_dim]

    if reduced.size(1) < 3:
        reduced = F.pad(reduced, (0, 3 - reduced.size(1)))

    scale = reduced.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
    return reduced[:, :3] / scale


def _pairwise_distance_slices(
    embeddings: torch.Tensor,
    label_names: list[str],
) -> tuple[torch.Tensor, torch.Tensor]:
    if embeddings.size(0) < 2:
        empty = embeddings.new_empty((0,))
        return empty, empty

    distances = torch.cdist(embeddings.float(), embeddings.float(), p=2)
    label_ids = {label: index for index, label in enumerate(sorted(set(label_names)))}
    labels = torch.tensor([label_ids[label] for label in label_names], dtype=torch.long)
    same_label = labels.unsqueeze(0) == labels.unsqueeze(1)
    upper_mask = torch.triu(torch.ones_like(distances, dtype=torch.bool), diagonal=1)
    same_distances = distances[upper_mask & same_label]
    different_distances = distances[upper_mask & ~same_label]
    return same_distances, different_distances


def _build_tetrahedron_mesh(
    points: torch.Tensor,
    label_names: list[str],
    *,
    radius: float = 0.035,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if points.ndim != 2 or points.size(1) != 3:
        raise ValueError("Expected points to have shape [N, 3].")

    base_offsets = torch.tensor(
        [
            [1.0, 1.0, 1.0],
            [-1.0, -1.0, 1.0],
            [-1.0, 1.0, -1.0],
            [1.0, -1.0, -1.0],
        ],
        dtype=points.dtype,
        device=points.device,
    )
    base_offsets = F.normalize(base_offsets, dim=1) * radius
    face_pattern = torch.tensor(
        [
            [0, 1, 2],
            [0, 3, 1],
            [0, 2, 3],
            [1, 3, 2],
        ],
        dtype=torch.int64,
        device=points.device,
    )

    vertices: list[torch.Tensor] = []
    colors: list[torch.Tensor] = []
    faces: list[torch.Tensor] = []

    for index, (point, label_name) in enumerate(zip(points, label_names, strict=False)):
        start = index * base_offsets.size(0)
        point_vertices = point.unsqueeze(0) + base_offsets
        point_color = (_color_for_label(label_name).to(points.device) * 255.0).round().to(torch.uint8)
        vertices.append(point_vertices)
        colors.append(point_color.unsqueeze(0).expand(base_offsets.size(0), -1))
        faces.append(face_pattern + start)

    return (
        torch.cat(vertices, dim=0).unsqueeze(0),
        torch.cat(colors, dim=0).unsqueeze(0),
        torch.cat(faces, dim=0).unsqueeze(0),
    )


def _make_centroid_distance_figure(embeddings: torch.Tensor, label_names: list[str]):
    unique_labels = sorted(set(label_names))
    label_to_indices = {
        label: [index for index, current_label in enumerate(label_names) if current_label == label]
        for label in unique_labels
    }
    centroids = torch.stack(
        [embeddings[indices].mean(dim=0) for indices in label_to_indices.values()],
        dim=0,
    )
    distances = torch.cdist(centroids.float(), centroids.float(), p=2).cpu().numpy()

    figure, axis = plt.subplots(figsize=(max(5, 0.75 * len(unique_labels)), max(4, 0.65 * len(unique_labels))))
    image = axis.imshow(distances, cmap="viridis")
    axis.set_title("Validation Centroid Distances")
    axis.set_xticks(range(len(unique_labels)), unique_labels, rotation=45, ha="right")
    axis.set_yticks(range(len(unique_labels)), unique_labels)
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    return figure


def _make_pairwise_distance_figure(
    same_distances: torch.Tensor,
    different_distances: torch.Tensor,
):
    figure, axis = plt.subplots(figsize=(7, 4))
    if same_distances.numel() > 0:
        axis.hist(
            same_distances.cpu().numpy(),
            bins=30,
            alpha=0.65,
            label="same emotion",
            color="#2b6cb0",
        )
    if different_distances.numel() > 0:
        axis.hist(
            different_distances.cpu().numpy(),
            bins=30,
            alpha=0.55,
            label="different emotion",
            color="#d97706",
        )
    axis.set_title("Pairwise Distance Distribution")
    axis.set_xlabel("L2 distance")
    axis.set_ylabel("count")
    axis.legend()
    figure.tight_layout()
    return figure


def _make_label_count_figure(label_names: list[str]):
    counts = Counter(label_names)
    labels = list(sorted(counts))
    values = [counts[label] for label in labels]

    figure, axis = plt.subplots(figsize=(max(6, 0.7 * len(labels)), 4))
    axis.bar(labels, values, color="#2563eb")
    axis.set_title("Validation Embedding Counts")
    axis.set_xlabel("emotion")
    axis.set_ylabel("samples logged")
    axis.tick_params(axis="x", rotation=45)
    figure.tight_layout()
    return figure


def _make_embedding_scatter_figure(embeddings: torch.Tensor, label_names: list[str]):
    projection = _project_embeddings_to_3d(embeddings)[:, :2].cpu()
    figure, axis = plt.subplots(figsize=(6, 5))
    for label in sorted(set(label_names)):
        indices = [index for index, current in enumerate(label_names) if current == label]
        axis.scatter(
            projection[indices, 0],
            projection[indices, 1],
            label=label,
            s=18,
            alpha=0.8,
        )
    axis.set_title("Validation Embedding Scatter (2D PCA)")
    axis.set_xlabel("component 1")
    axis.set_ylabel("component 2")
    axis.legend(fontsize=7, loc="best", ncol=2)
    figure.tight_layout()
    return figure


def _find_tensorboard_logger(trainer: pl.Trainer) -> TensorBoardLogger | None:
    if isinstance(trainer.logger, TensorBoardLogger):
        return trainer.logger
    for logger in trainer.loggers:
        if isinstance(logger, TensorBoardLogger):
            return logger
    return None


class EmotionTripletDataModule(pl.LightningDataModule):
    def __init__(self, config: TrainingConfig) -> None:
        super().__init__()
        self.config = config
        self.train_num_workers = _resolve_num_workers(config.data.train_num_workers, default_cap=8)
        self.val_num_workers = _resolve_num_workers(config.data.val_num_workers, default_cap=4)
        self.pin_memory = _resolve_pin_memory(config.data.pin_memory, config.device)
        self.prefetch_factor = max(2, config.data.prefetch_factor)
        self.train_dataset: EmotionSubsetDataset | None = None
        self.val_dataset: EmotionSubsetDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        if self.train_dataset is not None and self.val_dataset is not None:
            return

        train_dataset_full = EmotionStreamDataset(
            self.config.data.manifest_path,
            sample_rate=self.config.data.sample_rate,
            chunk_seconds=self.config.data.chunk_seconds,
            training=True,
            basic_emotions_only=self.config.data.basic_emotions_only,
            allowed_emotions=self.config.data.allowed_emotions,
            excluded_emotions=self.config.data.excluded_emotions,
            min_examples_per_emotion=self.config.data.min_examples_per_emotion,
            min_duration_seconds=self.config.data.min_duration_seconds,
            random_crop=self.config.data.random_crop,
            peak_normalize=self.config.data.peak_normalize,
            noise_augment=None,
        )
        eval_dataset_full = EmotionStreamDataset(
            self.config.data.manifest_path,
            sample_rate=self.config.data.sample_rate,
            chunk_seconds=self.config.data.chunk_seconds,
            training=False,
            basic_emotions_only=self.config.data.basic_emotions_only,
            allowed_emotions=self.config.data.allowed_emotions,
            excluded_emotions=self.config.data.excluded_emotions,
            min_examples_per_emotion=self.config.data.min_examples_per_emotion,
            min_duration_seconds=self.config.data.min_duration_seconds,
            random_crop=False,
            peak_normalize=self.config.data.peak_normalize,
            noise_augment=None,
        )

        train_indices, val_indices = stratified_speaker_split_indices(
            train_dataset_full.examples,
            val_ratio=self.config.data.val_ratio,
            seed=self.config.data.split_seed,
        )
        self.train_dataset = EmotionSubsetDataset(train_dataset_full, train_indices)
        self.val_dataset = EmotionSubsetDataset(eval_dataset_full, val_indices)

        if self.config.data.noise_probability > 0.0:
            self.train_dataset.dataset.noise_augment = AdditiveNoiseAugment(
                NoiseAugmentationConfig(
                    probability=self.config.data.noise_probability,
                    snr_db_min=self.config.data.snr_db_min,
                    snr_db_max=self.config.data.snr_db_max,
                    gain_db_min=self.config.data.gain_db_min,
                    gain_db_max=self.config.data.gain_db_max,
                )
            )

    def train_dataloader(self):
        if self.train_dataset is None:
            raise RuntimeError("DataModule.setup() must run before requesting train_dataloader().")
        return build_triplet_dataloader_from_dataset(
            self.train_dataset,
            labels_per_batch=self.config.data.labels_per_batch,
            samples_per_label=self.config.data.samples_per_label,
            batches_per_epoch=self.config.data.train_batches_per_epoch,
            seed=self.config.seed,
            num_workers=self.train_num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
        )

    def val_dataloader(self):
        if self.val_dataset is None:
            raise RuntimeError("DataModule.setup() must run before requesting val_dataloader().")
        return build_triplet_dataloader_from_dataset(
            self.val_dataset,
            labels_per_batch=self.config.data.labels_per_batch,
            samples_per_label=self.config.data.samples_per_label,
            batches_per_epoch=self.config.data.val_batches_per_epoch,
            seed=self.config.seed + 10_000,
            num_workers=self.val_num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
        )

    def summary(self) -> dict[str, Any]:
        if self.train_dataset is None or self.val_dataset is None:
            self.setup("fit")
        assert self.train_dataset is not None
        assert self.val_dataset is not None

        train_speakers = {speaker_group_key(example) for example in self.train_dataset.examples}
        val_speakers = {speaker_group_key(example) for example in self.val_dataset.examples}
        return {
            "train_examples": len(self.train_dataset),
            "val_examples": len(self.val_dataset),
            "train_labels": len(self.train_dataset.label_to_indices),
            "val_labels": len(self.val_dataset.label_to_indices),
            "train_speakers": len(train_speakers),
            "val_speakers": len(val_speakers),
            "speaker_overlap": len(train_speakers & val_speakers),
            "train_num_workers": self.train_num_workers,
            "val_num_workers": self.val_num_workers,
            "pin_memory": self.pin_memory,
            "prefetch_factor": self.prefetch_factor,
        }


class TripletLightningModule(pl.LightningModule):
    def __init__(self, config: TrainingConfig) -> None:
        super().__init__()
        self.config = config
        self.model = build_streaming_emotion_model(config.model)
        self._val_embeddings: list[torch.Tensor] = []
        self._val_label_names: list[str] = []
        self._val_dataset_names: list[str] = []
        self._val_speakers: list[str] = []
        self._val_sample_ids: list[str] = []

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, self.config.max_epochs),
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def on_train_epoch_start(self) -> None:
        self._set_sampler_epoch(self.trainer.train_dataloader, self.current_epoch)

    def on_validation_epoch_start(self) -> None:
        self._val_embeddings.clear()
        self._val_label_names.clear()
        self._val_dataset_names.clear()
        self._val_speakers.clear()
        self._val_sample_ids.clear()
        self._set_sampler_epoch(self.trainer.val_dataloaders, self.current_epoch)

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        loss, _, stats = self._shared_step(batch)
        batch_size = int(batch.labels.size(0))

        self.log("train/loss_step", loss, on_step=True, on_epoch=False, prog_bar=True, batch_size=batch_size)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log(
            "train/triplet_accuracy",
            stats["triplet_accuracy"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        self.log(
            "train/mean_hardest_positive",
            stats["mean_hardest_positive"],
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log(
            "train/mean_hardest_negative",
            stats["mean_hardest_negative"],
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log(
            "train/valid_anchors",
            stats["valid_anchors"],
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log(
            "train/separation_gap",
            stats["mean_hardest_negative"] - stats["mean_hardest_positive"],
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        return loss

    def validation_step(self, batch, batch_idx: int) -> torch.Tensor:
        loss, pooled_embeddings, stats = self._shared_step(batch)
        batch_size = int(batch.labels.size(0))

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log(
            "val/triplet_accuracy",
            stats["triplet_accuracy"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        self.log(
            "val/mean_hardest_positive",
            stats["mean_hardest_positive"],
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log(
            "val/mean_hardest_negative",
            stats["mean_hardest_negative"],
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log(
            "val/valid_anchors",
            stats["valid_anchors"],
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log(
            "val/separation_gap",
            stats["mean_hardest_negative"] - stats["mean_hardest_positive"],
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )

        if self._should_log_embeddings():
            remaining = self.config.embedding_log_limit - len(self._val_label_names)
            if remaining > 0:
                embeddings = pooled_embeddings[:remaining].detach().cpu()
                self._val_embeddings.append(embeddings)
                self._val_label_names.extend(batch.label_names[:remaining])
                self._val_dataset_names.extend(batch.datasets[:remaining])
                self._val_speakers.extend(batch.speaker_ids[:remaining])
                self._val_sample_ids.extend(batch.sample_ids[:remaining])

        return loss

    def get_val_embedding_payload(self) -> tuple[torch.Tensor, list[str], list[list[str]]] | None:
        if not self._val_embeddings:
            return None
        metadata_rows = [
            [label, dataset, speaker, sample_id]
            for label, dataset, speaker, sample_id in zip(
                self._val_label_names,
                self._val_dataset_names,
                self._val_speakers,
                self._val_sample_ids,
                strict=False,
            )
        ]
        return torch.cat(self._val_embeddings, dim=0), list(self._val_label_names), metadata_rows

    def _shared_step(self, batch) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        sequence_embeddings, sequence_lengths = self.model(batch.waveforms, batch.lengths)
        loss, pooled_embeddings, stats = sequence_metric_loss(
            sequence_embeddings,
            sequence_lengths,
            batch.labels,
            loss_name=self.config.loss_name,
            margin=self.config.margin,
            temperature=self.config.loss_temperature,
        )
        return loss, pooled_embeddings, stats

    def _should_log_embeddings(self) -> bool:
        frequency = max(1, self.config.log_embeddings_every_n_epochs)
        return (self.current_epoch + 1) % frequency == 0

    @staticmethod
    def _set_sampler_epoch(dataloaders: Any, epoch: int) -> None:
        if dataloaders is None:
            return
        if not isinstance(dataloaders, (list, tuple)):
            dataloaders = [dataloaders]
        for dataloader in dataloaders:
            batch_sampler = getattr(dataloader, "batch_sampler", None)
            if hasattr(batch_sampler, "set_epoch"):
                batch_sampler.set_epoch(epoch)


class MetricsHistoryCallback(Callback):
    def __init__(self, history_path: Path, *, monitor: str = "val/loss", mode: str = "min") -> None:
        super().__init__()
        self.history_path = history_path
        self.monitor = monitor
        self.mode = _validate_monitor_mode(mode)
        self.history: list[dict[str, Any]] = []
        self.best_epoch = -1
        self.best_val_loss = float("inf")
        self.best_metric_value = float("inf") if self.mode == "min" else float("-inf")
        self.best_metrics: dict[str, float] = {}

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.sanity_checking:
            return

        metrics = _extract_scalar_metrics(trainer.callback_metrics)
        epoch_record = {
            "epoch": trainer.current_epoch,
            "metrics": {key: value for key, value in metrics.items() if key.startswith(("train/", "val/"))},
        }
        self.history.append(epoch_record)
        _write_json(self.history_path, self.history)

        val_loss = epoch_record["metrics"].get("val/loss")
        if val_loss is not None and val_loss < self.best_val_loss:
            self.best_val_loss = val_loss

        monitor_value = epoch_record["metrics"].get(self.monitor)
        if monitor_value is not None and _is_better_metric(
            monitor_value,
            self.best_metric_value,
            mode=self.mode,
        ):
            self.best_metric_value = monitor_value
            self.best_epoch = trainer.current_epoch
            self.best_metrics = dict(epoch_record["metrics"])


class TensorBoardEmbeddingCallback(Callback):
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: TripletLightningModule) -> None:
        if trainer.sanity_checking:
            return

        payload = pl_module.get_val_embedding_payload()
        if payload is None:
            return

        tensorboard_logger = _find_tensorboard_logger(trainer)
        if tensorboard_logger is None:
            return

        embeddings, label_names, metadata_rows = payload
        writer = tensorboard_logger.experiment
        global_step = trainer.current_epoch

        writer.add_embedding(
            embeddings,
            metadata=metadata_rows,
            global_step=global_step,
            tag=f"val_embedding_projector_epoch_{global_step:04d}",
            metadata_header=["emotion", "dataset", "speaker", "sample_id"],
        )

        projection = _project_embeddings_to_3d(embeddings)
        mesh_vertices, mesh_colors, mesh_faces = _build_tetrahedron_mesh(projection, label_names)
        writer.add_mesh(
            "val_embedding_3d",
            mesh_vertices,
            colors=mesh_colors,
            faces=mesh_faces,
            global_step=global_step,
        )
        writer.add_histogram("val_embedding_norm", embeddings.norm(dim=1), global_step)

        same_distances, different_distances = _pairwise_distance_slices(embeddings, label_names)
        if same_distances.numel() > 0:
            writer.add_histogram("val_pairwise_distance_same", same_distances, global_step)
        if different_distances.numel() > 0:
            writer.add_histogram("val_pairwise_distance_different", different_distances, global_step)

        centroid_figure = _make_centroid_distance_figure(embeddings, label_names)
        writer.add_figure("val_centroid_distance_heatmap", centroid_figure, global_step)
        plt.close(centroid_figure)

        distance_figure = _make_pairwise_distance_figure(same_distances, different_distances)
        writer.add_figure("val_pairwise_distance_histogram", distance_figure, global_step)
        plt.close(distance_figure)

        count_figure = _make_label_count_figure(label_names)
        writer.add_figure("val_embedding_label_counts", count_figure, global_step)
        plt.close(count_figure)

        scatter_figure = _make_embedding_scatter_figure(embeddings, label_names)
        writer.add_figure("val_embedding_scatter_2d", scatter_figure, global_step)
        plt.close(scatter_figure)
        writer.flush()


class TensorBoardHParamsCallback(Callback):
    def __init__(self, config: TrainingConfig, datamodule: EmotionTripletDataModule, history_callback: MetricsHistoryCallback) -> None:
        super().__init__()
        self.config = config
        self.datamodule = datamodule
        self.history_callback = history_callback

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        tensorboard_logger = _find_tensorboard_logger(trainer)
        if tensorboard_logger is None:
            return

        hparam_dict = _config_hparams(self.config, self.datamodule)
        best_metrics = dict(self.history_callback.best_metrics)
        final_metrics = _extract_scalar_metrics(trainer.callback_metrics)
        metric_dict = {
            "hparam/best_val_loss": self.history_callback.best_val_loss,
            "hparam/best_epoch": float(self.history_callback.best_epoch),
            "hparam/final_val_loss": final_metrics.get("val/loss", self.history_callback.best_val_loss),
            "hparam/final_val_triplet_accuracy": final_metrics.get(
                "val/triplet_accuracy",
                best_metrics.get("val/triplet_accuracy", 0.0),
            ),
            "hparam/final_val_separation_gap": final_metrics.get(
                "val/separation_gap",
                best_metrics.get("val/separation_gap", 0.0),
            ),
        }
        tensorboard_logger.log_hyperparams(hparam_dict, metric_dict, step=trainer.global_step)


class OptunaPruningCallback(Callback):
    def __init__(self, trial: optuna.Trial, monitor: str = "val/loss") -> None:
        super().__init__()
        self.trial = trial
        self.monitor = monitor

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.sanity_checking:
            return

        metric = trainer.callback_metrics.get(self.monitor)
        if metric is None:
            return

        score = float(metric.detach().cpu().item()) if isinstance(metric, torch.Tensor) else float(metric)
        self.trial.report(score, step=trainer.current_epoch)
        if self.trial.should_prune():
            raise optuna.TrialPruned()


def train_triplet_model(
    config: TrainingConfig,
    *,
    trial: optuna.Trial | None = None,
    monitor: str = "val/loss",
    monitor_mode: str = "min",
) -> TrainingResult:
    if config.model.sample_rate != config.data.sample_rate:
        raise ValueError("Model and data sample rates must match.")
    monitor_mode = _validate_monitor_mode(monitor_mode)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pl.seed_everything(config.seed, workers=True)
    datamodule = EmotionTripletDataModule(config)
    datamodule.setup("fit")

    config_path = output_dir / "config.json"
    history_path = output_dir / "history.json"
    _write_json(config_path, asdict(config))
    _write_json(output_dir / "dataset_summary.json", datamodule.summary())

    logger = TensorBoardLogger(
        save_dir=str(output_dir),
        name="tensorboard",
        version="",
        default_hp_metric=False,
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        filename="best",
        monitor=monitor,
        mode=monitor_mode,
        save_top_k=1,
        save_last=True,
    )
    history_callback = MetricsHistoryCallback(history_path, monitor=monitor, mode=monitor_mode)
    callbacks: list[Callback] = [
        checkpoint_callback,
        EarlyStopping(
            monitor=monitor,
            mode=monitor_mode,
            patience=config.early_stopping_patience,
            min_delta=config.early_stopping_min_delta,
        ),
        LearningRateMonitor(logging_interval="epoch"),
        TQDMProgressBar(refresh_rate=max(1, config.progress_bar_refresh_rate)),
        history_callback,
        TensorBoardEmbeddingCallback(),
        TensorBoardHParamsCallback(config, datamodule, history_callback),
    ]
    if trial is not None:
        callbacks.append(OptunaPruningCallback(trial, monitor=monitor))

    accelerator, devices = _resolve_lightning_device(config.device)
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        callbacks=callbacks,
        default_root_dir=str(output_dir),
        enable_progress_bar=True,
        gradient_clip_val=config.grad_clip_norm,
        log_every_n_steps=max(1, config.log_every_n_steps),
        logger=logger,
        max_epochs=config.max_epochs,
        num_sanity_val_steps=0,
    )

    model = TripletLightningModule(config)
    trainer.fit(model, datamodule=datamodule)

    best_checkpoint_path = Path(checkpoint_callback.best_model_path) if checkpoint_callback.best_model_path else output_dir / "checkpoints" / "best.ckpt"
    best_model_score = checkpoint_callback.best_model_score
    best_metric_value = (
        float(best_model_score.detach().cpu().item())
        if isinstance(best_model_score, torch.Tensor)
        else float(best_model_score)
        if best_model_score is not None
        else history_callback.best_metric_value
    )

    return TrainingResult(
        best_val_loss=history_callback.best_val_loss,
        best_epoch=history_callback.best_epoch,
        best_metric_name=monitor,
        best_metric_mode=monitor_mode,
        best_metric_value=best_metric_value,
        output_dir=output_dir,
        best_checkpoint_path=best_checkpoint_path,
        tensorboard_dir=Path(logger.log_dir),
        history_path=history_path,
    )


def suggest_model_config(trial: optuna.Trial, base_model: StreamingModelConfig) -> StreamingModelConfig:
    model_choices = (
        [base_model.model_type]
        if base_model.model_type in {"conformer", "rnnt"}
        else ["conformer", "rnnt"]
    )
    model_type = (
        model_choices[0]
        if len(model_choices) == 1
        else trial.suggest_categorical("model_type", model_choices)
    )
    activation = trial.suggest_categorical("activation", ["relu", "gelu", "silu", "mish"])
    n_mels = trial.suggest_categorical("n_mels", [64, 80])
    output_dim = trial.suggest_categorical("output_dim", [96, 128, 160])
    dropout = trial.suggest_float("dropout", 0.05, 0.25)

    if model_type == "conformer":
        encoder_dim = trial.suggest_categorical("encoder_dim", [96, 128, 160, 192])
        num_heads = trial.suggest_categorical("conformer_heads", [4, 8])
        if encoder_dim % num_heads != 0:
            encoder_dim = 128
            num_heads = 4
        return StreamingModelConfig(
            **{
                **asdict(base_model),
                "model_type": model_type,
                "activation": activation,
                "n_mels": n_mels,
                "output_dim": output_dim,
                "dropout": dropout,
                "encoder_dim": encoder_dim,
                "conformer_layers": trial.suggest_int("conformer_layers", 2, 6),
                "conformer_heads": num_heads,
                "conformer_ffn_multiplier": trial.suggest_categorical("conformer_ffn_multiplier", [2, 4]),
                "conformer_conv_kernel_size": trial.suggest_categorical("conformer_conv_kernel_size", [15, 31]),
            }
        )

    return StreamingModelConfig(
        **{
            **asdict(base_model),
            "model_type": model_type,
            "activation": activation,
            "n_mels": n_mels,
            "output_dim": output_dim,
            "dropout": dropout,
            "encoder_dim": trial.suggest_categorical("encoder_dim", [128, 192, 256]),
            "rnnt_layers": trial.suggest_int("rnnt_layers", 2, 5),
            "rnnt_time_reduction_factor": trial.suggest_categorical("rnnt_time_reduction_factor", [1, 2, 4]),
        }
    )


def suggest_training_config(trial: optuna.Trial, base_config: TrainingConfig) -> TrainingConfig:
    output_root = Path(base_config.output_dir)
    tuned_model = suggest_model_config(trial, base_config.model)
    loss_name = trial.suggest_categorical(
        "loss_name",
        ["batch_hard_triplet", "batch_all_triplet", "supervised_contrastive"],
    )
    tuned_data = DataConfig(
        **{
            **asdict(base_config.data),
            "chunk_seconds": trial.suggest_categorical("chunk_seconds", [2.4, 3.2, 4.0]),
            "labels_per_batch": trial.suggest_int("labels_per_batch", 3, 5),
            "samples_per_label": trial.suggest_int("samples_per_label", 2, 4),
            "train_batches_per_epoch": base_config.data.train_batches_per_epoch or 200,
            "val_batches_per_epoch": base_config.data.val_batches_per_epoch or 64,
            "noise_probability": trial.suggest_float("noise_probability", 0.4, 1.0),
            "snr_db_min": trial.suggest_float("snr_db_min", 4.0, 12.0),
            "snr_db_max": trial.suggest_float("snr_db_max", 18.0, 32.0),
        }
    )
    return TrainingConfig(
        **{
            **asdict(base_config),
            "output_dir": output_root / "optuna" / f"trial_{trial.number:04d}",
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 3e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
            "margin": trial.suggest_float("margin", 0.1, 0.5),
            "loss_name": loss_name,
            "loss_temperature": (
                trial.suggest_float("loss_temperature", 0.05, 0.2)
                if loss_name == "supervised_contrastive"
                else base_config.loss_temperature
            ),
            "model": tuned_model,
            "data": tuned_data,
        }
    )


def run_optuna_study(
    base_config: TrainingConfig,
    *,
    n_trials: int = 20,
    study_name: str = "emotion_embeddings",
    storage: str | None = None,
    timeout: float | None = None,
    monitor: str = "val/triplet_accuracy",
    monitor_mode: str = "max",
) -> optuna.Study:
    monitor_mode = _validate_monitor_mode(monitor_mode)
    study_dir = Path(base_config.output_dir) / "optuna"
    study_dir.mkdir(parents=True, exist_ok=True)
    sampler = optuna.samplers.TPESampler(seed=base_config.seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=2)
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        sampler=sampler,
        pruner=pruner,
        direction="minimize" if monitor_mode == "min" else "maximize",
        load_if_exists=True,
    )

    def objective(trial: optuna.Trial) -> float:
        config = suggest_training_config(trial, base_config)
        result = train_triplet_model(config, trial=trial, monitor=monitor, monitor_mode=monitor_mode)
        trial.set_user_attr("best_epoch", result.best_epoch)
        trial.set_user_attr("best_checkpoint_path", str(result.best_checkpoint_path))
        trial.set_user_attr("best_metric_name", result.best_metric_name)
        trial.set_user_attr("best_metric_mode", result.best_metric_mode)
        trial.set_user_attr("best_metric_value", result.best_metric_value)
        trial.set_user_attr("best_val_loss", result.best_val_loss)
        return result.best_metric_value

    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    _write_json(
        study_dir / "best_trial.json",
        {
            "monitor": monitor,
            "monitor_mode": monitor_mode,
            "best_value": study.best_value,
            "best_trial_number": study.best_trial.number,
            "best_params": study.best_params,
            "best_user_attrs": study.best_trial.user_attrs,
        },
    )

    try:
        dataframe = study.trials_dataframe()
    except Exception:
        dataframe = None
    if dataframe is not None:
        dataframe.to_csv(study_dir / "trials.csv", index=False)

    return study
