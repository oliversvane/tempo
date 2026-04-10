from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def masked_mean_pool(sequence_embeddings: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    if sequence_embeddings.ndim != 3:
        raise ValueError("Expected sequence_embeddings to have shape [batch, time, dim].")
    if lengths.ndim != 1:
        raise ValueError("Expected lengths to have shape [batch].")

    time_index = torch.arange(sequence_embeddings.size(1), device=sequence_embeddings.device)
    mask = time_index.unsqueeze(0) < lengths.unsqueeze(1)
    masked = sequence_embeddings * mask.unsqueeze(-1)
    pooled = masked.sum(dim=1) / mask.sum(dim=1).clamp_min(1).unsqueeze(-1)
    return pooled


def l2_normalize_embeddings(embeddings: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if embeddings.ndim != 2:
        raise ValueError("Expected embeddings to have shape [batch, dim].")
    return F.normalize(embeddings, p=2, dim=-1, eps=eps)


def _validate_embeddings_and_labels(embeddings: torch.Tensor, labels: torch.Tensor) -> None:
    if embeddings.ndim != 2:
        raise ValueError("Expected embeddings to have shape [batch, dim].")
    if labels.ndim != 1:
        raise ValueError("Expected labels to have shape [batch].")
    if embeddings.size(0) != labels.size(0):
        raise ValueError("Embeddings and labels must agree on batch size.")


def _pairwise_masks(labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    same_label = labels.unsqueeze(0) == labels.unsqueeze(1)
    same_label.fill_diagonal_(False)
    different_label = ~same_label
    different_label.fill_diagonal_(False)
    return same_label, different_label


def _distance_diagnostics(
    distances: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    same_label, different_label = _pairwise_masks(labels)
    hardest_positive = distances.masked_fill(~same_label, float("-inf")).max(dim=1).values
    hardest_negative = distances.masked_fill(~different_label, float("inf")).min(dim=1).values
    valid = same_label.any(dim=1) & different_label.any(dim=1)
    stats = {
        "valid_anchors": int(valid.sum().item()),
        "mean_hardest_positive": float(hardest_positive[valid].mean().item()) if valid.any() else 0.0,
        "mean_hardest_negative": float(hardest_negative[valid].mean().item()) if valid.any() else 0.0,
        "triplet_accuracy": float((hardest_negative[valid] > hardest_positive[valid]).float().mean().item())
        if valid.any()
        else 0.0,
    }
    return same_label, different_label, valid, stats


def batch_hard_triplet_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    *,
    margin: float = 0.2,
) -> tuple[torch.Tensor, dict[str, Any]]:
    _validate_embeddings_and_labels(embeddings, labels)
    distances = torch.cdist(embeddings, embeddings, p=2)
    same_label, different_label, valid, stats = _distance_diagnostics(distances, labels)

    hardest_positive = distances.masked_fill(~same_label, float("-inf")).max(dim=1).values
    hardest_negative = distances.masked_fill(~different_label, float("inf")).min(dim=1).values
    triplet_losses = F.relu(hardest_positive - hardest_negative + margin)
    loss = triplet_losses[valid].mean() if valid.any() else embeddings.new_tensor(0.0)
    return loss, stats


def batch_all_triplet_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    *,
    margin: float = 0.2,
) -> tuple[torch.Tensor, dict[str, Any]]:
    _validate_embeddings_and_labels(embeddings, labels)
    distances = torch.cdist(embeddings, embeddings, p=2)
    same_label, different_label, _, stats = _distance_diagnostics(distances, labels)

    triplet_mask = same_label.unsqueeze(2) & different_label.unsqueeze(1)
    triplet_losses = distances.unsqueeze(2) - distances.unsqueeze(1) + margin
    triplet_losses = F.relu(triplet_losses.masked_fill(~triplet_mask, 0.0))
    valid_triplets = triplet_mask.sum()
    active_triplets = (triplet_losses > 0).sum()
    loss = triplet_losses[triplet_mask].mean() if valid_triplets.item() > 0 else embeddings.new_tensor(0.0)

    stats["valid_triplets"] = int(valid_triplets.item())
    stats["active_triplets"] = int(active_triplets.item())
    return loss, stats


def supervised_contrastive_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    *,
    temperature: float = 0.1,
) -> tuple[torch.Tensor, dict[str, Any]]:
    _validate_embeddings_and_labels(embeddings, labels)
    similarities = embeddings @ embeddings.transpose(0, 1)
    logits = similarities / max(temperature, 1e-6)
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()

    same_label, _, valid, stats = _distance_diagnostics(torch.cdist(embeddings, embeddings, p=2), labels)
    logits_mask = ~torch.eye(labels.size(0), dtype=torch.bool, device=labels.device)
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))

    positive_mask = same_label & logits_mask
    positive_counts = positive_mask.sum(dim=1)
    mean_log_prob_positive = (log_prob * positive_mask).sum(dim=1) / positive_counts.clamp_min(1)
    loss = -mean_log_prob_positive[valid].mean() if valid.any() else embeddings.new_tensor(0.0)
    stats["contrastive_temperature"] = float(temperature)
    return loss, stats


def embedding_metric_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    *,
    loss_name: str = "batch_hard_triplet",
    margin: float = 0.2,
    temperature: float = 0.1,
) -> tuple[torch.Tensor, dict[str, Any]]:
    normalized_name = loss_name.strip().lower()
    if normalized_name == "batch_hard_triplet":
        return batch_hard_triplet_loss(embeddings, labels, margin=margin)
    if normalized_name == "batch_all_triplet":
        return batch_all_triplet_loss(embeddings, labels, margin=margin)
    if normalized_name == "supervised_contrastive":
        return supervised_contrastive_loss(embeddings, labels, temperature=temperature)
    raise ValueError(f"Unsupported loss_name: {loss_name}")


def sequence_metric_loss(
    sequence_embeddings: torch.Tensor,
    sequence_lengths: torch.Tensor,
    labels: torch.Tensor,
    *,
    loss_name: str = "batch_hard_triplet",
    margin: float = 0.2,
    temperature: float = 0.1,
    l2_normalize: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    pooled_embeddings = masked_mean_pool(sequence_embeddings, sequence_lengths)
    if l2_normalize:
        pooled_embeddings = l2_normalize_embeddings(pooled_embeddings)
    loss, stats = embedding_metric_loss(
        pooled_embeddings,
        labels,
        loss_name=loss_name,
        margin=margin,
        temperature=temperature,
    )
    return loss, pooled_embeddings, stats


def sequence_triplet_loss(
    sequence_embeddings: torch.Tensor,
    sequence_lengths: torch.Tensor,
    labels: torch.Tensor,
    *,
    margin: float = 0.2,
    l2_normalize: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    return sequence_metric_loss(
        sequence_embeddings,
        sequence_lengths,
        labels,
        loss_name="batch_hard_triplet",
        margin=margin,
        l2_normalize=l2_normalize,
    )
