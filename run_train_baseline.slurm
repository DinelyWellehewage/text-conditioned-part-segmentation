#!/bin/bash

#SBATCH --job-name=partseg-baseline
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=logs/baseline_%j.out
#SBATCH --error=logs/baseline_%j.err

set -e

cd "$HOME/text-conditioned-part-segmentation"

mkdir -p logs

module load python/3.12-base
source .venv/bin/activate

export PASCAL_PART_ROOT=/home/vault/v123be/v123be51/text-conditioned-part-segmentation/data/PascalPart116
export HF_HOME=/home/vault/v123be/v123be51/huggingface
export HF_HUB_OFFLINE=1

echo "Node:"
hostname

echo
echo "GPU:"
nvidia-smi

echo
echo "Dataset:"
echo "$PASCAL_PART_ROOT"

echo
echo "Starting baseline training..."

python -m src.train_baseline

echo
echo "Training finished."