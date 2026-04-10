#!/bin/bash
#SBATCH -J tempo-train
#SBATCH -p gpu
#SBATCH --gres=gpu:3
#SBATCH -c 24
#SBATCH --mem=128G
#SBATCH -t 24:00:00
#SBATCH -o logs/%x-%j.out

set -euo pipefail

# ---- config ----
REPO_DIR="${SLURM_SUBMIT_DIR:-$HOME/projects/tempo}"
TEMPO_COMMAND="${TEMPO_COMMAND:-train}"
TEMPO_MANIFEST="${TEMPO_MANIFEST:-data/processed/manifest.csv}"
TEMPO_OUTPUT_DIR="${TEMPO_OUTPUT_DIR:-runs/slurm_${SLURM_JOB_ID}}"
TEMPO_STUDY_NAME="${TEMPO_STUDY_NAME:-emotion_embeddings}"
TEMPO_TRIALS="${TEMPO_TRIALS:-20}"
TEMPO_TUNE_MONITOR="${TEMPO_TUNE_MONITOR:-val/triplet_accuracy}"
TEMPO_TUNE_MONITOR_MODE="${TEMPO_TUNE_MONITOR_MODE:-max}"
TEMPO_DEVICE="${TEMPO_DEVICE:-auto}"
TEMPO_MATMUL_PRECISION="${TEMPO_MATMUL_PRECISION:-high}"

mkdir -p "$REPO_DIR/logs"
cd "$REPO_DIR"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID

echo "JOB: ${SLURM_JOB_ID:-none}"
echo "HOST: $(hostname)"
echo "GPUS: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "CPUS: ${SLURM_CPUS_ON_NODE:-unset}"
echo "COMMAND: $TEMPO_COMMAND"
echo "OUTPUT_DIR: $TEMPO_OUTPUT_DIR"
echo "DEVICE: $TEMPO_DEVICE"
echo "MATMUL_PRECISION: $TEMPO_MATMUL_PRECISION"

nvidia-smi

# ---- training ----
if [ "$#" -eq 0 ]; then
  if [ "$TEMPO_COMMAND" = "tune" ]; then
    set -- \
      tune \
      --manifest "$TEMPO_MANIFEST" \
      --output-dir "$TEMPO_OUTPUT_DIR" \
      --study-name "$TEMPO_STUDY_NAME" \
      --trials "$TEMPO_TRIALS" \
      --tune-monitor "$TEMPO_TUNE_MONITOR" \
      --tune-monitor-mode "$TEMPO_TUNE_MONITOR_MODE" \
      --device "$TEMPO_DEVICE" \
      --matmul-precision "$TEMPO_MATMUL_PRECISION"
  else
    set -- \
      train \
      --manifest "$TEMPO_MANIFEST" \
      --output-dir "$TEMPO_OUTPUT_DIR" \
      --device "$TEMPO_DEVICE" \
      --matmul-precision "$TEMPO_MATMUL_PRECISION"
  fi
fi

echo "Running: srun uv run python -m tempo.training $*"
srun uv run python -m tempo.training "$@"
