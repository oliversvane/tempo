# tempo

Dataset ingestion utilities for audio emotion classification.

The repo now contains one source module per dataset with:

- `download_dataset()`: downloads raw source data into `data/raw/<source>/`
- `preprocess_dataset()`: exports a normalized audio-first manifest into `data/processed/<source>/manifest.csv`

The standardized processed layout is:

```text
data/
  raw/<source>/...
  processed/<source>/
    audio/...
    manifest.csv
  processed/manifest.csv
```

Each manifest row follows the same schema and points at audio placed under `data/processed/<source>/audio/`.

## Sources

- `emodb`
- `iemocap`
- `cremad`
- `savee`
- `tess`
- `ravdess`
- `emotale`
- `cameo`

## Usage

Install the environment with `uv`:

```bash
uv sync
```

List the available dataset sources:

```bash
uv run python -m tempo.datasets list
```

Download everything:

```bash
uv run python -m tempo.datasets download all
```

Preprocess everything into the common schema:

```bash
uv run python -m tempo.datasets preprocess all
```

Run download and preprocess together:

```bash
uv run python -m tempo.datasets run all
```

You can also target a single source, for example:

```bash
uv run python -m tempo.datasets run ravdess
```

Generate a dataset report from the processed manifest:

```bash
uv run python -m tempo.datasets report all
```

This writes:

- `data/processed/report.md`
- `data/processed/report.json`
- `data/processed/plots/*.png`

## PyTorch Streaming Dataloader

The repo now also includes a Torch data pipeline for triplet-mined emotion training in [tempo/training/data.py](/home/svane/projects/tempo/tempo/training/data.py) and [tempo/training/triplet.py](/home/svane/projects/tempo/tempo/training/triplet.py).

It is designed for a Conformer or RNN-T style encoder that consumes raw audio chunks and emits a sequence of embeddings. The usual pattern is:

1. Use the dataloader to produce class-balanced audio batches with additive noise.
2. Run the batch through the encoder to get `B x T x D` embeddings.
3. Convert the sequence into utterance-level embeddings with `sequence_triplet_loss(...)`.
4. Optimize those embeddings with built-in batch-hard triplet mining.

Example:

```python
from tempo.training import build_triplet_dataloader, sequence_triplet_loss

dataset, loader = build_triplet_dataloader(
    "data/processed/manifest.csv",
    sample_rate=16_000,
    chunk_seconds=3.2,
    labels_per_batch=4,
    samples_per_label=4,
    basic_emotions_only=True,
)

batch = next(iter(loader))
# encoder_outputs, frame_lengths = encoder(batch.waveforms, batch.lengths)
# loss, utterance_embeddings, stats = sequence_triplet_loss(
#     encoder_outputs,
#     frame_lengths,
#     batch.labels,
# )
```

## Training And Tuning

Two streaming encoder families are available in [tempo/training/models.py](/home/svane/projects/tempo/tempo/training/models.py):

- `conformer`: causal log-mel frontend plus Conformer blocks
- `rnnt`: causal log-mel frontend plus an RNN-T style recurrent encoder

The Lightning-based training stack in [tempo/training/train.py](/home/svane/projects/tempo/tempo/training/train.py) handles:

- speaker-disjoint, label-aware train/validation splits from the combined manifest
- tunable metric-learning losses including batch-hard triplet, batch-all triplet, and supervised contrastive loss
- tunable activation functions across both encoder families
- Lightning progress bars, learning-rate tracking, and per-step/per-epoch metric logging during training
- TensorBoard logging for losses, separation metrics, learning rate, an epoch-by-epoch projector view, a 3D validation embedding, centroid-distance heatmaps, pairwise-distance histograms, label-count plots, and an HParams tab for comparing runs
- checkpointing of the best model
- early stopping with configurable patience and minimum delta
- Optuna hyperparameter search across both encoder families, activation functions, and metric losses

Train a single model:

```bash
uv run python -m tempo.training train \
  --manifest data/processed/manifest.csv \
  --model-type conformer \
  --activation gelu \
  --loss-name batch_all_triplet \
  --output-dir runs/conformer_baseline \
  --log-every-n-steps 1
```

Run Optuna tuning:

```bash
uv run python -m tempo.training tune \
  --manifest data/processed/manifest.csv \
  --output-dir runs/search \
  --trials 20
```

Monitor training with TensorBoard:

```bash
uv run tensorboard --logdir runs
```

Useful training knobs:

- `--train-num-workers -1` and `--val-num-workers -1` enable auto worker selection
- `--prefetch-factor 4` keeps more batches queued in RAM
- `--pin-memory` enables CUDA-friendly host-to-device transfers
- `--early-stopping-patience` and `--early-stopping-min-delta` control stopping behavior

## Notes

- Kaggle sources require Kaggle credentials in the environment or a configured local Kaggle setup.
- `cameo` overlaps with some standalone corpora such as `cremad` and `ravdess`, so the combined manifest intentionally keeps duplicate source entries.
- The preprocessing step keeps audio in a common folder layout and normalizes emotion labels, but it does not transcode every source to a single audio codec.
