#!/bin/bash
#SBATCH -A naiss2026-3-450-gpu
#SBATCH -J sampled_decode
#SBATCH -p gpu
#SBATCH --gpus 1
#SBATCH -t 06:00:00
#SBATCH -o logs/sampled_decode_%j.out
#SBATCH -e logs/sampled_decode_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_80
#SBATCH --mail-user=your.email@kth.se

PROJ=/nobackup/proj/disk/large-pml-2025-storage/personal/gicepp

ml GPU/Python/3.13.5-bare-gcc-2025b-eb        # <-- must match how you built the venv
source "$PROJ/venvs/entropy/bin/activate"

export HF_HOME="$PROJ/hf_cache"
unset HF_HUB_OFFLINE HF_DATASETS_OFFLINE

cd "$PROJ/Caching/kvpress"

# fail immediately rather than burning 5 hours on CPU
python -c "import torch; assert torch.cuda.is_available(), 'NO GPU VISIBLE'; print(torch.__version__, torch.cuda.get_device_name(0))"

OUT=./results/entropy_analysis/llma31_8b/longbench/trec/sampled_decode
mkdir -p "$OUT"

python evaluation/entropy_analysis.py \
    --model unsloth/Llama-3.1-8B-Instruct \
    --teacher_forcing False \
    --dataset longbench --data_dir trec --n_samples 150 --n_mc_samples 50 \
    --compression_ratios "[0.0, 0.25, 0.50, 0.75, 0.95]"  --decoding_compression_interval 1 \
    --mc_batch_size 8 \
    --device cuda \
    --output_dir "$OUT" \
    2>&1 | tee "$OUT/my_log.log"
