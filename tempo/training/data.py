from __future__ import annotations

import csv
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import soundfile as sf
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, Sampler

from tempo.datasets.utils import PROJECT_ROOT, normalize_emotion

BASIC_EMOTION_LABELS = (
    "anger",
    "boredom",
    "calm",
    "disgust",
    "excitement",
    "fear",
    "frustration",
    "happiness",
    "neutral",
    "sadness",
    "surprise",
)


@dataclass(frozen=True)
class EmotionStreamExample:
    dataset: str
    sample_id: str
    speaker_id: str
    emotion: str
    emotion_id: int
    audio_path: Path
    duration_seconds: float
    sample_rate_hz: int | None
    metadata: dict[str, Any]


@dataclass(frozen=True)
class NoiseAugmentationConfig:
    probability: float = 0.8
    snr_db_min: float = 8.0
    snr_db_max: float = 28.0
    gain_db_min: float = -4.0
    gain_db_max: float = 4.0


@dataclass
class EmotionStreamingBatch:
    waveforms: torch.Tensor
    lengths: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    label_names: list[str]
    sample_ids: list[str]
    datasets: list[str]
    speaker_ids: list[str]
    sample_rate: int

    def to(self, device: torch.device | str) -> "EmotionStreamingBatch":
        return EmotionStreamingBatch(
            waveforms=self.waveforms.to(device),
            lengths=self.lengths.to(device),
            attention_mask=self.attention_mask.to(device),
            labels=self.labels.to(device),
            label_names=self.label_names,
            sample_ids=self.sample_ids,
            datasets=self.datasets,
            speaker_ids=self.speaker_ids,
            sample_rate=self.sample_rate,
        )


class AdditiveNoiseAugment:
    def __init__(self, config: NoiseAugmentationConfig | None = None) -> None:
        self.config = config or NoiseAugmentationConfig()

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.numel() == 0:
            return waveform
        if torch.rand(1).item() > self.config.probability:
            return waveform

        snr_db = torch.empty(1).uniform_(self.config.snr_db_min, self.config.snr_db_max).item()
        gain_db = torch.empty(1).uniform_(self.config.gain_db_min, self.config.gain_db_max).item()
        noise = self._make_noise(waveform)

        signal_rms = waveform.pow(2).mean().sqrt().clamp_min(1e-6)
        noise_rms = noise.pow(2).mean().sqrt().clamp_min(1e-6)
        desired_noise_rms = signal_rms / (10 ** (snr_db / 20.0))
        augmented = waveform + noise * (desired_noise_rms / noise_rms)
        augmented = augmented * (10 ** (gain_db / 20.0))
        return augmented.clamp_(-1.0, 1.0)

    def _make_noise(self, waveform: torch.Tensor) -> torch.Tensor:
        if torch.randint(0, 2, ()).item() == 0:
            return torch.randn_like(waveform)

        brown = torch.cumsum(torch.randn_like(waveform), dim=0)
        brown = brown - brown.mean()
        return brown / brown.std().clamp_min(1e-6)


class EmotionStreamDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        manifest_path: str | Path,
        *,
        sample_rate: int = 16_000,
        chunk_seconds: float = 3.2,
        training: bool = True,
        basic_emotions_only: bool = False,
        allowed_emotions: Sequence[str] | None = None,
        excluded_emotions: Sequence[str] = ("other",),
        min_examples_per_emotion: int = 2,
        min_duration_seconds: float = 0.25,
        pad_to_chunk: bool = True,
        random_crop: bool = True,
        peak_normalize: bool = False,
        noise_augment: AdditiveNoiseAugment | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.sample_rate = sample_rate
        self.chunk_seconds = chunk_seconds
        self.chunk_num_samples = int(round(sample_rate * chunk_seconds))
        self.training = training
        self.pad_to_chunk = pad_to_chunk
        self.random_crop = random_crop
        self.peak_normalize = peak_normalize
        self.noise_augment = noise_augment if training else None

        allowed = {normalize_emotion(label) for label in allowed_emotions or ()}
        excluded = {normalize_emotion(label) for label in excluded_emotions}
        if basic_emotions_only:
            allowed = set(BASIC_EMOTION_LABELS) if not allowed else allowed & set(BASIC_EMOTION_LABELS)

        rows = self._read_manifest(self.manifest_path)
        filtered_rows = []
        for row in rows:
            emotion = normalize_emotion(row.get("emotion"))
            if not emotion:
                continue
            if allowed and emotion not in allowed:
                continue
            if emotion in excluded:
                continue

            duration_seconds = self._float_or_zero(row.get("duration_seconds"))
            if duration_seconds < min_duration_seconds:
                continue

            audio_path = self._resolve_audio_path(row.get("audio_path", ""))
            if not audio_path.exists():
                continue

            filtered_rows.append((row, emotion, audio_path, duration_seconds))

        counts = Counter(emotion for _, emotion, _, _ in filtered_rows)
        valid_emotions = {
            emotion for emotion, count in counts.items() if count >= min_examples_per_emotion
        }
        self.emotion_to_id = {
            emotion: index for index, emotion in enumerate(sorted(valid_emotions))
        }

        self.examples: list[EmotionStreamExample] = []
        self.label_to_indices: dict[int, list[int]] = defaultdict(list)
        for row, emotion, audio_path, duration_seconds in filtered_rows:
            if emotion not in valid_emotions:
                continue

            metadata = {}
            if row.get("metadata_json"):
                try:
                    metadata = json.loads(row["metadata_json"])
                except json.JSONDecodeError:
                    metadata = {"raw_metadata_json": row["metadata_json"]}

            example = EmotionStreamExample(
                dataset=row["dataset"],
                sample_id=row["sample_id"],
                speaker_id=row.get("speaker_id", ""),
                emotion=emotion,
                emotion_id=self.emotion_to_id[emotion],
                audio_path=audio_path,
                duration_seconds=duration_seconds,
                sample_rate_hz=self._int_or_none(row.get("sample_rate_hz")),
                metadata=metadata,
            )
            self.label_to_indices[example.emotion_id].append(len(self.examples))
            self.examples.append(example)

        if not self.examples:
            raise ValueError("No training examples remain after manifest filtering.")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        example = self.examples[index]
        waveform, sample_rate = self._load_audio(example.audio_path)
        waveform = self._crop_waveform(waveform)
        if self.peak_normalize:
            peak = waveform.abs().max().clamp_min(1e-6)
            waveform = waveform / peak
        if self.noise_augment is not None:
            waveform = self.noise_augment(waveform)

        length = waveform.numel()
        if self.pad_to_chunk and length < self.chunk_num_samples:
            waveform = F.pad(waveform, (0, self.chunk_num_samples - length))

        return {
            "waveform": waveform,
            "length": min(length, self.chunk_num_samples),
            "label": example.emotion_id,
            "label_name": example.emotion,
            "sample_id": example.sample_id,
            "dataset": example.dataset,
            "speaker_id": example.speaker_id,
            "sample_rate": sample_rate,
        }

    @staticmethod
    def _float_or_zero(value: str | None) -> float:
        try:
            return float(value or 0.0)
        except ValueError:
            return 0.0

    @staticmethod
    def _int_or_none(value: str | None) -> int | None:
        try:
            return int(value) if value not in (None, "") else None
        except ValueError:
            return None

    @staticmethod
    def _read_manifest(manifest_path: Path) -> list[dict[str, str]]:
        with manifest_path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    @staticmethod
    def _resolve_audio_path(audio_path: str) -> Path:
        candidate = Path(audio_path)
        if candidate.is_absolute():
            return candidate
        project_candidate = PROJECT_ROOT / candidate
        if project_candidate.exists():
            return project_candidate
        return candidate.resolve()

    def _load_audio(self, audio_path: Path) -> tuple[torch.Tensor, int]:
        audio, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)
        waveform = torch.from_numpy(audio).mean(dim=1)
        if sample_rate != self.sample_rate:
            waveform = self._resample_linear(waveform, sample_rate, self.sample_rate)
            sample_rate = self.sample_rate
        return waveform.contiguous(), sample_rate

    def _crop_waveform(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.numel() <= self.chunk_num_samples:
            return waveform

        if self.training and self.random_crop:
            max_start = waveform.numel() - self.chunk_num_samples
            start = torch.randint(0, max_start + 1, ()).item()
        else:
            start = (waveform.numel() - self.chunk_num_samples) // 2
        return waveform[start : start + self.chunk_num_samples]

    @staticmethod
    def _resample_linear(
        waveform: torch.Tensor,
        source_sample_rate: int,
        target_sample_rate: int,
    ) -> torch.Tensor:
        if source_sample_rate == target_sample_rate:
            return waveform
        new_length = max(1, round(waveform.numel() * target_sample_rate / source_sample_rate))
        resampled = F.interpolate(
            waveform.view(1, 1, -1),
            size=new_length,
            mode="linear",
            align_corners=False,
        )
        return resampled.view(-1)


class EmotionBalancedBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        label_to_indices: dict[int, list[int]],
        *,
        labels_per_batch: int = 4,
        samples_per_label: int = 4,
        batches_per_epoch: int | None = None,
        seed: int = 0,
        num_replicas: int = 1,
        rank: int = 0,
    ) -> None:
        self.label_to_indices = {
            label: list(indices) for label, indices in label_to_indices.items() if indices
        }
        self.labels = sorted(self.label_to_indices)
        self.labels_per_batch = labels_per_batch
        self.samples_per_label = samples_per_label
        self.batch_size = labels_per_batch * samples_per_label
        total_examples = sum(len(indices) for indices in self.label_to_indices.values())
        self.batches_per_epoch = batches_per_epoch or max(1, math.ceil(total_examples / self.batch_size))
        self.seed = seed
        self.epoch = 0
        self.num_replicas = max(1, num_replicas)
        self.rank = rank

        if len(self.labels) < 2:
            raise ValueError("Triplet mining requires at least two emotion classes.")
        if not 0 <= self.rank < self.num_replicas:
            raise ValueError("rank must be in the range [0, num_replicas).")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.batches_per_epoch

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        total_batches = self.batches_per_epoch * self.num_replicas
        for batch_index in range(total_batches):
            chosen_labels = self._sample_labels(rng)
            batch: list[int] = []
            for label in chosen_labels:
                indices = self.label_to_indices[label]
                if len(indices) >= self.samples_per_label:
                    batch.extend(rng.sample(indices, k=self.samples_per_label))
                else:
                    batch.extend(rng.choices(indices, k=self.samples_per_label))
            if batch_index % self.num_replicas == self.rank:
                yield batch

    def _sample_labels(self, rng: random.Random) -> list[int]:
        if len(self.labels) >= self.labels_per_batch:
            return rng.sample(self.labels, k=self.labels_per_batch)
        return rng.choices(self.labels, k=self.labels_per_batch)


class EmotionSubsetDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        dataset: EmotionStreamDataset,
        indices: Sequence[int],
    ) -> None:
        self.dataset = dataset
        self.indices = list(indices)
        self.examples = [dataset.examples[index] for index in self.indices]
        self.emotion_to_id = dict(dataset.emotion_to_id)
        self.label_to_indices: dict[int, list[int]] = defaultdict(list)
        for subset_index, source_index in enumerate(self.indices):
            label = dataset.examples[source_index].emotion_id
            self.label_to_indices[label].append(subset_index)
        self.sample_rate = dataset.sample_rate
        self.chunk_seconds = dataset.chunk_seconds

        if not self.indices:
            raise ValueError("EmotionSubsetDataset requires at least one example.")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.dataset[self.indices[index]]


def stratified_split_indices(
    label_to_indices: dict[int, list[int]],
    *,
    val_ratio: float = 0.1,
    seed: int = 0,
) -> tuple[list[int], list[int]]:
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1.")

    rng = random.Random(seed)
    train_indices: list[int] = []
    val_indices: list[int] = []

    for label in sorted(label_to_indices):
        indices = list(label_to_indices[label])
        rng.shuffle(indices)
        if len(indices) == 1:
            train_indices.extend(indices)
            continue

        val_count = max(1, round(len(indices) * val_ratio))
        if len(indices) >= 4:
            val_count = max(2, val_count)
        val_count = min(val_count, len(indices) - 1)

        val_indices.extend(indices[:val_count])
        train_indices.extend(indices[val_count:])

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return train_indices, val_indices


def speaker_group_key(example: EmotionStreamExample) -> str:
    speaker_id = example.speaker_id.strip() or example.sample_id
    return f"{example.dataset}:{speaker_id}"


def stratified_speaker_split_indices(
    examples: Sequence[EmotionStreamExample],
    *,
    val_ratio: float = 0.1,
    seed: int = 0,
) -> tuple[list[int], list[int]]:
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1.")

    speaker_to_indices: dict[str, list[int]] = defaultdict(list)
    speaker_to_label_counts: dict[str, Counter[int]] = defaultdict(Counter)
    label_total_counts: Counter[int] = Counter()
    label_speaker_counts: Counter[int] = Counter()

    for index, example in enumerate(examples):
        group_key = speaker_group_key(example)
        speaker_to_indices[group_key].append(index)
        speaker_to_label_counts[group_key][example.emotion_id] += 1
        label_total_counts[example.emotion_id] += 1

    for label_counts in speaker_to_label_counts.values():
        for label in label_counts:
            label_speaker_counts[label] += 1

    target_val_counts: Counter[int] = Counter()
    for label, total_count in label_total_counts.items():
        if total_count <= 1 or label_speaker_counts[label] <= 1:
            target_val_counts[label] = 0
            continue
        target_val_counts[label] = min(max(1, round(total_count * val_ratio)), total_count - 1)

    rng = random.Random(seed)
    group_keys = list(speaker_to_indices)
    rng.shuffle(group_keys)
    group_keys.sort(
        key=lambda key: (
            sum(1.0 / label_speaker_counts[label] for label in speaker_to_label_counts[key]),
            len(speaker_to_indices[key]),
        ),
        reverse=True,
    )

    remaining_label_counts = Counter(label_total_counts)
    remaining_speaker_groups = Counter(label_speaker_counts)
    assigned_val_counts: Counter[int] = Counter()
    assigned_train_speakers: Counter[int] = Counter()
    assigned_val_speakers: Counter[int] = Counter()
    val_indices: list[int] = []
    train_indices: list[int] = []
    target_total_val = round(len(examples) * val_ratio)

    for group_key in group_keys:
        group_indices = speaker_to_indices[group_key]
        group_label_counts = speaker_to_label_counts[group_key]
        group_size = len(group_indices)

        for label, count in group_label_counts.items():
            remaining_label_counts[label] -= count
            remaining_speaker_groups[label] -= 1

        can_assign_val = True
        val_cost = abs((len(val_indices) + group_size) - target_total_val)
        train_cost = abs(len(val_indices) - target_total_val)

        for label, count in group_label_counts.items():
            target_count = target_val_counts[label]
            val_after = assigned_val_counts[label] + count
            current = assigned_val_counts[label]

            if label_total_counts[label] - val_after < 1:
                can_assign_val = False
                break
            if assigned_train_speakers[label] + remaining_speaker_groups[label] < 1:
                can_assign_val = False
                break

            val_cost += abs(val_after - target_count)
            train_cost += abs(current - target_count)

            if current + remaining_label_counts[label] < target_count:
                train_cost += (target_count - (current + remaining_label_counts[label])) * 1000
            if (
                target_count > 0
                and assigned_val_speakers[label] == 0
                and remaining_speaker_groups[label] == 0
                and label_speaker_counts[label] > 1
            ):
                train_cost += 1000

        should_assign_val = can_assign_val and (
            val_cost < train_cost
            or (
                val_cost == train_cost
                and len(val_indices) < target_total_val
            )
        )

        if should_assign_val:
            val_indices.extend(group_indices)
            for label, count in group_label_counts.items():
                assigned_val_counts[label] += count
                assigned_val_speakers[label] += 1
        else:
            train_indices.extend(group_indices)
            for label in group_label_counts:
                assigned_train_speakers[label] += 1

    if not train_indices or not val_indices:
        label_to_indices: dict[int, list[int]] = defaultdict(list)
        for index, example in enumerate(examples):
            label_to_indices[example.emotion_id].append(index)
        return stratified_split_indices(label_to_indices, val_ratio=val_ratio, seed=seed)

    return sorted(train_indices), sorted(val_indices)


def streaming_collate(examples: list[dict[str, Any]]) -> EmotionStreamingBatch:
    waveforms = [example["waveform"] for example in examples]
    lengths = torch.tensor([example["length"] for example in examples], dtype=torch.long)
    labels = torch.tensor([example["label"] for example in examples], dtype=torch.long)
    padded = pad_sequence(waveforms, batch_first=True)
    time_axis = torch.arange(padded.size(1), device=padded.device).unsqueeze(0)
    attention_mask = time_axis < lengths.unsqueeze(1)
    sample_rate = int(examples[0]["sample_rate"])

    return EmotionStreamingBatch(
        waveforms=padded,
        lengths=lengths,
        attention_mask=attention_mask,
        labels=labels,
        label_names=[example["label_name"] for example in examples],
        sample_ids=[example["sample_id"] for example in examples],
        datasets=[example["dataset"] for example in examples],
        speaker_ids=[example["speaker_id"] for example in examples],
        sample_rate=sample_rate,
    )


def build_triplet_dataloader_from_dataset(
    dataset: EmotionStreamDataset | EmotionSubsetDataset,
    *,
    labels_per_batch: int = 4,
    samples_per_label: int = 4,
    batches_per_epoch: int | None = None,
    seed: int = 0,
    num_workers: int = 0,
    pin_memory: bool = False,
    prefetch_factor: int = 4,
    num_replicas: int = 1,
    rank: int = 0,
) -> DataLoader[EmotionStreamingBatch]:
    sampler = EmotionBalancedBatchSampler(
        dataset.label_to_indices,
        labels_per_batch=labels_per_batch,
        samples_per_label=samples_per_label,
        batches_per_epoch=batches_per_epoch,
        seed=seed,
        num_replicas=num_replicas,
        rank=rank,
    )
    dataloader_kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_sampler": sampler,
        "collate_fn": streaming_collate,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        dataloader_kwargs["prefetch_factor"] = max(2, prefetch_factor)
    return DataLoader(**dataloader_kwargs)


def build_triplet_dataloader(
    manifest_path: str | Path,
    *,
    sample_rate: int = 16_000,
    chunk_seconds: float = 3.2,
    labels_per_batch: int = 4,
    samples_per_label: int = 4,
    batches_per_epoch: int | None = None,
    seed: int = 0,
    num_workers: int = 0,
    pin_memory: bool = False,
    prefetch_factor: int = 4,
    training: bool = True,
    basic_emotions_only: bool = False,
    allowed_emotions: Sequence[str] | None = None,
    excluded_emotions: Sequence[str] = ("other",),
    min_examples_per_emotion: int = 2,
    min_duration_seconds: float = 0.25,
    noise_probability: float = 0.8,
    snr_db_min: float = 8.0,
    snr_db_max: float = 28.0,
    gain_db_min: float = -4.0,
    gain_db_max: float = 4.0,
    random_crop: bool = True,
) -> tuple[EmotionStreamDataset, DataLoader[EmotionStreamingBatch]]:
    noise_augment = None
    if training and noise_probability > 0.0:
        noise_augment = AdditiveNoiseAugment(
            NoiseAugmentationConfig(
                probability=noise_probability,
                snr_db_min=snr_db_min,
                snr_db_max=snr_db_max,
                gain_db_min=gain_db_min,
                gain_db_max=gain_db_max,
            )
        )

    dataset = EmotionStreamDataset(
        manifest_path,
        sample_rate=sample_rate,
        chunk_seconds=chunk_seconds,
        training=training,
        basic_emotions_only=basic_emotions_only,
        allowed_emotions=allowed_emotions,
        excluded_emotions=excluded_emotions,
        min_examples_per_emotion=min_examples_per_emotion,
        min_duration_seconds=min_duration_seconds,
        random_crop=random_crop,
        noise_augment=noise_augment,
    )
    dataloader = build_triplet_dataloader_from_dataset(
        dataset,
        labels_per_batch=labels_per_batch,
        samples_per_label=samples_per_label,
        batches_per_epoch=batches_per_epoch,
        seed=seed,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
    )
    return dataset, dataloader
