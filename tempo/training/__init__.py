"""PyTorch training utilities for streaming emotion models."""

from .data import (
    BASIC_EMOTION_LABELS,
    AdditiveNoiseAugment,
    EmotionBalancedBatchSampler,
    EmotionStreamDataset,
    EmotionStreamExample,
    EmotionStreamingBatch,
    EmotionSubsetDataset,
    NoiseAugmentationConfig,
    build_triplet_dataloader_from_dataset,
    build_triplet_dataloader,
    speaker_group_key,
    stratified_speaker_split_indices,
    stratified_split_indices,
    streaming_collate,
)
from .models import ConformerEncoder, RNNTStyleEncoder
from .models import StreamingEmotionModel, StreamingModelConfig, build_streaming_emotion_model
from .models import build_activation, lengths_to_padding_mask
from .train import (
    DataConfig,
    EmotionTripletDataModule,
    TrainingConfig,
    TrainingResult,
    TripletLightningModule,
    run_optuna_study,
    train_triplet_model,
)
from .triplet import batch_all_triplet_loss, batch_hard_triplet_loss, embedding_metric_loss, masked_mean_pool
from .triplet import l2_normalize_embeddings, sequence_metric_loss, sequence_triplet_loss, supervised_contrastive_loss

__all__ = [
    "AdditiveNoiseAugment",
    "BASIC_EMOTION_LABELS",
    "batch_all_triplet_loss",
    "ConformerEncoder",
    "DataConfig",
    "EmotionBalancedBatchSampler",
    "EmotionStreamDataset",
    "EmotionStreamExample",
    "EmotionStreamingBatch",
    "EmotionSubsetDataset",
    "EmotionTripletDataModule",
    "NoiseAugmentationConfig",
    "RNNTStyleEncoder",
    "StreamingEmotionModel",
    "StreamingModelConfig",
    "TrainingConfig",
    "TrainingResult",
    "TripletLightningModule",
    "batch_hard_triplet_loss",
    "build_activation",
    "build_streaming_emotion_model",
    "build_triplet_dataloader_from_dataset",
    "build_triplet_dataloader",
    "embedding_metric_loss",
    "l2_normalize_embeddings",
    "lengths_to_padding_mask",
    "masked_mean_pool",
    "run_optuna_study",
    "sequence_metric_loss",
    "sequence_triplet_loss",
    "speaker_group_key",
    "stratified_speaker_split_indices",
    "stratified_split_indices",
    "streaming_collate",
    "supervised_contrastive_loss",
    "train_triplet_model",
]
