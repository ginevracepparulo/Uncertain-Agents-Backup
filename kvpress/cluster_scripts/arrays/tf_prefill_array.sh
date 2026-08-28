#!/bin/bash
#SBATCH -A naiss2026-3-450-gpu
#SBATCH -J tf_prefill
#SBATCH -p gpu
#SBATCH --gpus 1
#SBATCH -t 00:30:00
#SBATCH --array=0-15
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_80
#SBATCH --mail-user=your.email@kth.se

MODELS=(unsloth/Llama-3.1-8B-Instruct Qwen/Qwen2.5-7B-Instruct)
MODELTAGS=(llama31_8b qwen2.5_7b)
PRESSES=(random snapkv)
DATASETS=(trec hotpotqa narrativeqa repobench-p)

# cartesian product: 16 tasks = 2 models x 2 presses x 4 datasets
n_m=${#MODELS[@]}
n_p=${#PRESSES[@]}
n_d=${#DATASETS[@]}

d=$(( SLURM_ARRAY_TASK_ID % n_d ))
p=$(( SLURM_ARRAY_TASK_ID / n_d % n_p ))
m=$(( SLURM_ARRAY_TASK_ID / (n_d * n_p) ))

MODEL=${MODELS[$m]}
MODELTAG=${MODELTAGS[$m]}
PRESS=${PRESSES[$p]}
DATASET=${DATASETS[$d]}

[[ -n "$MODEL" && -n "$PRESS" && -n "$DATASET" ]] || { echo "bad index $SLURM_ARRAY_TASK_ID"; exit 1; }
echo "task $SLURM_ARRAY_TASK_ID: model=$MODEL press=$PRESS dataset=${DATASETS[$d]}"

PROJ=/nobackup/proj/disk/large-pml-2025-storage/personal/gicepp
ml GPU/Python/3.13.5-bare-gcc-2025b-eb
source "$PROJ/venvs/entropy/bin/activate"
export HF_HOME="$PROJ/hf_cache"
unset HF_HUB_OFFLINE HF_DATASETS_OFFLINE
cd "$PROJ/Caching/kvpress"

python -c "import torch; assert torch.cuda.is_available(), 'NO GPU VISIBLE'; print(torch.__version__, torch.cuda.get_device_name(0))"

OUT=./results/entropy_analysis/$MODELTAG/longbench/$DATASET/$PRESS/tf_prefill
mkdir -p "$OUT"

python evaluation/entropy_analysis.py \
    --model "$MODEL" \
    --press_name "$PRESS" \
    --teacher_forcing True \
    --dataset longbench --data_dir "$DATASET" --n_samples 150 \
    --compression_ratios "[0.0, 0.25, 0.50, 0.75, 0.95]" \
    --device cuda \
    --output_dir "$OUT" \
    2>&1 | tee "$OUT/my_log.log"