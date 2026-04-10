#!/bin/bash
#SBATCH --job-name=tempo-train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:3
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j-%N.err

set -euo pipefail

# Example cluster-specific settings:
##SBATCH --partition=gpu
##SBATCH --account=your-account

REPO_DIR="${SLURM_SUBMIT_DIR:-$HOME/projects/tempo}"
cd "$REPO_DIR"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID

: "${TEMPO_COMMAND:=tune}"
: "${TEMPO_OUTPUT_DIR:=runs/slurm_${SLURM_JOB_ID}}"
: "${TEMPO_MANIFEST:=data/processed/manifest.csv}"
: "${TEMPO_TRIALS:=20}"

if [ "$#" -eq 0 ]; then
  if [ "$TEMPO_COMMAND" = "tune" ]; then
    set -- \
      tune \
      --manifest "$TEMPO_MANIFEST" \
      --output-dir "$TEMPO_OUTPUT_DIR" \
      --trials "$TEMPO_TRIALS" \
      --device auto
  else
    set -- \
      train \
      --manifest "$TEMPO_MANIFEST" \
      --output-dir "$TEMPO_OUTPUT_DIR" \
      --device auto
  fi
fi

echo "Running: uv run python -m tempo.training $*"
srun uv run python -m tempo.training "$@"
