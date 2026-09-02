#!/bin/bash
#SBATCH -A naiss2026-3-450-gpu
#SBATCH -J agent_sampled_prefill
#SBATCH -p gpu
#SBATCH --gpus 1
#SBATCH -t 06:00:00
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_80
#SBATCH --mail-user=your.email@kth.se

MODELS=(unsloth/Llama-3.1-8B-Instruct Qwen/Qwen2.5-7B-Instruct)
MODELTAGS=(llama31_8b qwen2.5_7b)
PRESSES=(random snapkv)

# cartesian product: 4 tasks = 2 models x 2 presses
n_m=${#MODELS[@]}
n_p=${#PRESSES[@]}

m=$(( SLURM_ARRAY_TASK_ID % n_m ))
p=$(( SLURM_ARRAY_TASK_ID / n_m ))

MODEL=${MODELS[$m]}
MODELTAG=${MODELTAGS[$m]}
PRESS=${PRESSES[$p]}
TASK=prime_task

[[ -n "$MODEL" && -n "$PRESS"]] || { echo "bad index $SLURM_ARRAY_TASK_ID"; exit 1; }
echo "task $SLURM_ARRAY_TASK_ID: model=$MODEL press=$PRESS"

PROJ=/nobackup/proj/disk/large-pml-2025-storage/personal/gicepp

module purge
ml GPU/Python/3.13.5-bare-gcc-2025b-eb        # <-- must match how you built the venv
source "$PROJ/venvs/entropy/bin/activate"

unset HF_HUB_OFFLINE HF_DATASETS_OFFLINE
export HF_HOME="$PROJ/hf_cache"

cd "$PROJ/Caching/kvpress"

# fail immediately rather than burning 6 hours on CPU
python -c "import torch; assert torch.cuda.is_available(), 'NO GPU VISIBLE'; print(torch.__version__, torch.cuda.get_device_name(0))"

RUN=./results/agent_runs/$MODELTAG/$TASK

OUT=./results/entropy_analysis/$MODELTAG/agent/$TASK/$PRESS/sampled_prefill
mkdir -p "$OUT"

python evaluation/entropy_analysis.py \
    --model $MODEL \
    --press_name "$PRESS" \
    --teacher_forcing False \
    --trajectory_path "$RUN/agent_log.jsonl" --n_samples 150 --n_mc_samples 50 \
    --compression_ratios "[0.0, 0.25, 0.50, 0.75, 0.95]" \
    --mc_batch_size 8 \
    --ece_metric action_f1 \
    --device cuda \
    --output_dir "$OUT" \
    2>&1 | tee "$OUT/my_log.log"
