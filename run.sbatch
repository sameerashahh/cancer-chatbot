#!/bin/bash
#SBATCH --job-name=instruct_eval
#SBATCH --output=eval_%j.out
#SBATCH --error=eval_%j.err
#SBATCH --time=06:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euo pipefail

echo "=== Job info ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node(s): ${SLURM_JOB_NODELIST}"
echo "Start time: $(date)"
echo "Workdir: $(pwd)"
echo "================"

module purge
module load mamba/latest
module load cuda-12.5.0-gcc-12.1.0

source activate instruct-eval
PY=/home/tghadge1/.conda/envs/instruct-eval/bin/python

echo "=== Python env debug ==="
echo "Using PY=$PY"
$PY -V
$PY -c "import sys; print('sys.executable:', sys.executable)"
echo "========================"

echo "=== Import smoke test ==="
$PY -c "import torch; print('torch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA device count:', torch.cuda.device_count())"
$PY -c "import transformers; print('transformers:', transformers.__version__)"
$PY -c "import datasets; print('datasets:', datasets.__version__)"
echo "========================="

export HF_HOME="${HOME}/.cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"

echo "=== Running eval.py ==="
$PY -u eval.py

echo "=== Done ==="
echo "End time: $(date)"
