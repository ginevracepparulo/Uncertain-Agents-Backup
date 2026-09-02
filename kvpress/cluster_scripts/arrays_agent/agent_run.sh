#!/bin/bash
#SBATCH --job-name=agent_run
#SBATCH -A naiss2026-3-450-gpu
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out
#SBATCH -e logs/slurm-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_80
#SBATCH --mail-user=your.email@kth.se

set -euo pipefail

PROJ=/nobackup/proj/disk/large-pml-2025-storage/personal/gicepp

module purge
ml GPU/Python/3.13.5-bare-gcc-2025b-eb        # <-- must match how you built the venv
source "$PROJ/venvs/entropy/bin/activate"

unset HF_HUB_OFFLINE HF_DATASETS_OFFLINE
export HF_HOME="$PROJ/hf_cache"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MSWEA_SILENT_STARTUP=1
export MSWEA_CONFIGURED=true

cd "$PROJ/Caching/kvpress"

MODEL=unsloth/Llama-3.1-8B-Instruct
MODEL_TAG=llama31_8b
TASK=prime_task
RUN=./results/agent_runs/$MODEL_TAG/$TASK
WORK=$RUN/workdir
mkdir -p "$RUN" "$WORK"

mini -c default.yaml \
    -c mini_textbased.yaml \
    -c model.model_class=kvpress.mini_swe_agent_model.KVPressLocalModel \
    -c model.model_name="$MODEL" \
    -c model.compression_ratio=0.0 \
    -c model.max_new_tokens=512 \
    -c model.device=cuda \
    -c model.log_path="$RUN/agent_log.jsonl" \
    -c environment.cwd="$WORK" \
    -c agent.step_limit=50 \
    -t "write a python file prime.py with a function is_prime(n)" \
    -o "$RUN/agent.traj.json"
