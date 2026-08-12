# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Measure how KV-cache eviction distorts a model's next-token / sequence
predictions, comparing FULL attention against COMPRESSED attention (each user
compression ratio) across a dataset of tasks.

Two axes:

- `regime` -- how uncertainty is measured: `teacher_forced` or `sampled`.
- `kv_caching` -- the cache under test: `full` (no eviction) or `compressed` (a
   ratio). Every run scores `full` plus each user ratio; ratio 0.0 is
   skipped since it is identical to `full`.

================================  regime = teacher_forced  ========================
Both kv_caching configs are teacher-forced on the *same* reference sequence
O = (o_1..o_T) (the gold answer if the dataset provides one, else greedily generated
under full attention). Because both see the same prefix at every position t, the two
next-token distributions p_full(.|o_<t, x) and p_comp(.|o_<t, x) are directly
comparable. 

Per task, sequence position t (in bits):
  * H_full(t)    = -sum_v p_full(v) log2 p_full(v)                 per-token entropy, full
  * H_comp(t)    = -sum_v p_comp(v) log2 p_comp(v)                 per-token entropy, compressed
  * IG(t)        = H_comp(t) - H_full(t)                           extra uncertainty from eviction
  * KL(t)        = sum_v p_full(v) log2( p_full(v)/p_comp(v) )     distribution shift
  * ce_full(t)   = -log2 p_full(o_t)                               surprisal of the gold token, full --what is this?
  * ce_comp(t)   = -log2 p_comp(o_t)                               surprisal of the gold token, compressed --what is this?
  * excess_ce(t) = ce_comp(t) - ce_full(t)                         extra bits to predict gold (quality drop) --what is this?

Per task aggregated sequence positions t (in bits):
  * H_full_sum    = mean_t H_full(t)                               mean per-token entropy, full
  * H_comp_sum    = mean_t H_comp(t)                               mean per-token entropy, compressed
  * IG_sum        = sum_t IG(t)                                    total extra uncertainty from eviction
  * KL_sum        = mean_t KL(t)                                   mean distribution shift
  * ce_full_sum   = mean_t ce_full(t)                              mean surprisal of the gold token, full
  * ce_comp_sum   = mean_t ce_comp(t)                              mean surprisal of the gold token, compressed
  * excess_ce_sum = mean_t excess_ce(t)                            mean extra bits to predict gold (quality drop)
  
Remarks:
- There is no benchmark accuracy here: teacher-forcing never generates an answer, so its
quality metric is `excess_ce` (how much harder the gold answer is to predict once the
cache is evicted). 

- Decode-time eviction here means that the press is active during the teacher-forcing pass, 
so it can evict tokens from the reference sequence.

- Prefill-time eviction here means that the press is only active during the prefill pass, 
so it can evict tokens from the prompt but not from the reference sequence.

==================================  regime = sampled  =============================
No shared reference: each kv_caching config free-generates. Per (task, config), draw
N = n_mc_samples continuations y^(i) ~ p(.|x) at temperature 1.0 (no top-k/top-p, so the
plug-in estimate is unbiased), and additionally greedy-decode ONE answer to score 
against the dataset's gold answer (if available) using the dataset's correctness metric.

Per task:
  * H_seq_full     = (1/N) sum_i [ -log2 p_full(y^(i)|x) ]                    sequence entropy, full
  * H_seq_comp     = (1/N) sum_i [ -log2 p_comp(y^(i)|x) ]                    sequence entropy, compressed
  * H_seq_full_se  = sqrt( varentropy_seq / N )                               varentropy_seq = Var_i(-log2 p_full(y^(i)))
  * H_seq_comp_se  = sqrt( varentropy_seq / N )                               varentropy_seq = Var_i(-log2 p_comp(y^(i)))
  * KL_seq         = (1/N) sum_i [log2 p_full(y^(i)|x)-log2 p_comp(y^(i)|x)]  y^(i) ~ p_full, MC estimate of KL(p_full || p_comp), bits/sequence.
                                                                              KL needs both models evaluated on one common set of sequences.
  * KL_seq_se      = sqrt( varkl_seq / N )                                    varkl_seq = Var_i(log2 p_full(y^(i)) - log2 p_comp(y^(i)) )
  * IG_seq         = H_seq_comp - H_seq_full                                  extra uncertainty from eviction, bits/sequence   

Per (task, kv_caching, ratio):
  * confidence     = exp( -H_seq_nats / mean_length )                         per-token geo-mean prob = 1/perplexity in (0,1] --what is this?
  * primary_score  = token-F1 for hotpotqa;                                   benchmark metric used to score greedy answer vs gold, 
                                                                              in [0,1] see evaluation/benchmarks/<dataset>/calculate_metrics.py
  * error          = 1 - primary_score                                        --this might not be correct depending on the primary_score?
  * mean_score     = mean_task(primary_score)                                 mean benchmark score over tasks
  * ECE            = sum_b (n_b/N)|mean_score_b - conf_b|                     confidence vs benchmark score, binned by confidence. 
                                                                              The score is continuous, so a bin reads "tasks this confident scored X 
                                                                              on average" rather than "X% were correct". No correct/incorrect threshold 
                                                                              enters either number, or the plots.

  ===================================  outputs (both regimes)  ======================
Each run writes two CSVs into the output dir:

  * comparison.csv  
      teacher_forced: one row per (task, compression ratio) --> per task aggregated sequence positions t (in bits)
        H_full_sum, H_comp_sum, IG_sum, KL_sum, ce_full_sum, ce_comp_sum, excess_ce_sum
      
      sampled: one row per (task, compression ratio): full vs compressed side by side             
        H_seq_full, H_seq_comp, IG_seq, KL_seq, confidence_full/confidence_comp, primary_score_full/primary_score_comp
  
  * per_config.csv  -- one row per (task, kv_caching=full/@compression ratio, compression ratio)  [ratio = NA for full]
      teacher_forced: H_mean, ce_mean
      sampled:        H_seq, H_seq_se, confidence, KL_seq, KL_seq_se, predicted_answer, gold, primary_score, correct
 

plus a dataset-level summary.csv and diagnostic plots.

Useful Dataset:
MultiFieldQA-en (avg context window: 4,559 tokens)
HotpotQA (avg context window: 9,151 tokens)
TREC (avg context window: 5,177 tokens)

Usage
-----
from ssh
# 1) teacher-forced, prefill-only eviction
OUT=./results/entropy_analysis/llma31_8b/longbench/trec/tf_prefill
mkdir -p "$OUT"
python evaluation/entropy_analysis.py \
    --model unsloth/Llama-3.1-8B-Instruct \
    --teacher_forcing True \
    --dataset longbench --data_dir trec --n_samples 150 \
    --compression_ratios "[0.0, 0.25, 0.50, 0.75, 0.95]" \
    --device cuda \
    --output_dir "$OUT" \
    2>&1 | tee "$OUT/my_log.log"

# 2) teacher-forced, decode-time eviction
OUT=./results/entropy_analysis/llma31_8b/longbench/trec/tf_decode
mkdir -p "$OUT"
python evaluation/entropy_analysis.py \
    --model unsloth/Llama-3.1-8B-Instruct \
    --teacher_forcing True \
    --dataset longbench --data_dir trec --n_samples 150 \
    --compression_ratios "[0.0, 0.25, 0.50, 0.75, 0.95]" --decoding_compression_interval 1 \
    --device cuda \
    --output_dir "$OUT" \
    2>&1 | tee "$OUT/my_log.log"

    
# 3) sampled, prefill-only eviction
OUT=./results/entropy_analysis/llma31_8b/longbench/trec/sampled_prefill
mkdir -p "$OUT"
python evaluation/entropy_analysis.py \
    --model unsloth/Llama-3.1-8B-Instruct \
    --teacher_forcing False \
    --dataset longbench --data_dir trec --n_samples 150 --n_mc_samples 50 \
    --compression_ratios "[0.0, 0.25, 0.50, 0.75, 0.95]"\
    --mc_batch_size 8 \
    --device cuda \
    --output_dir "$OUT" \
    2>&1 | tee "$OUT/my_log.log"

# 4) sampled, decode-time eviction
OUT=./results/entropy_analysis/llma31_8b/longbench/trec/sampled_decode
mkdir -p "$OUT"
python evaluation/entropy_analysis.py \
    --model unsloth/Llama-3.1-8B-Instruct \
    --teacher_forcing False \
    --dataset longbench --data_dir trec --n_samples 150 --n_mc_samples 50 \
    --compression_ratios "[0.0, 0.25, 0.50, 0.75, 0.95]" --decoding_compression_interval 4 \
    --mc_batch_size 8 \
    --device cuda \
    --output_dir "$OUT" \
    2>&1 | tee "$OUT/my_log.log"
------------------------------------------------------------------------    
from local
# 1) teacher-forced, prefill-only eviction
OUT=./results/entropy_analysis/llma31_8b/longbench/hotpotqa/tf_prefill
mkdir -p "$OUT"
nohup .venv/bin/python evaluation/entropy_analysis.py \
    --model unsloth/Llama-3.1-8B-Instruct \
    --teacher_forcing True \
    --dataset longbench --data_dir hotpotqa --n_samples 150 \
    --compression_ratios "[0.0, 0.25, 0.75]" \
    --device mps \
    --output_dir "$OUT" \
    2>&1 | tee "$OUT/my_log.log"

# 2) teacher-forced, decode-time eviction
OUT=./results/entropy_analysis/llma31_8b/longbench/trec/tf_decode
mkdir -p "$OUT"
nohup .venv/bin/python evaluation/entropy_analysis.py \
    --model unsloth/Llama-3.1-8B-Instruct \
    --teacher_forcing True \
    --dataset longbench --data_dir trec --n_samples 150 \
    --compression_ratios "[0.0, 0.25, 0.75]" --decoding_compression_interval 1 \
    --device mps \
    --output_dir "$OUT" \
    2>&1 | tee "$OUT/my_log.log"

    
# 3) sampled, prefill-only eviction
OUT=./results/entropy_analysis/llma31_8b/longbench/trec/sampled_prefill
mkdir -p "$OUT"
nohup .venv/bin/python evaluation/entropy_analysis.py \
    --model unsloth/Llama-3.1-8B-Instruct \
    --teacher_forcing False \
    --dataset longbench --data_dir trec --n_samples 150 --n_mc_samples 50 \
    --compression_ratios "[0.0, 0.25, 0.75]" \
    --device mps \
    --output_dir "$OUT" \
    2>&1 | tee "$OUT/my_log.log"

# 4) sampled, decode-time eviction
OUT=./results/entropy_analysis/llma31_8b/longbench/trec/sampled_decode
mkdir -p "$OUT"
nohup .venv/bin/python evaluation/entropy_analysis.py \
    --model unsloth/Llama-3.1-8B-Instruct \
    --teacher_forcing False \
    --dataset longbench --data_dir trec --n_samples 150 --n_mc_samples 50 \
    --compression_ratios "[0.0, 0.25, 0.75]" --decoding_compression_interval 4 \
    --device mps \
    --output_dir "$OUT" \
    2>&1 | tee "$OUT/my_log.log"

# ad-hoc single task, or a local JSONL with context/question/answer fields
nohup .venv/bin/python evaluation/entropy_analysis.py --context "..." --question "..." --answer "..."
python evaluation/entropy_analysis.py --dataset_path ./my_tasks.jsonl
"""

import contextlib
import itertools
import json
import logging
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from fire import Fire
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache, PreTrainedModel, PreTrainedTokenizer

from evaluate_registry import DATASET_REGISTRY, SCORER_REGISTRY

from entropy_metrics import (
    SampledOutput,
    SequenceKLEstimate,
    TeacherForcedOutput,
    compute_ece,
    compute_sequence_metrics,
    sequence_entropy_estimate,
    sequence_confidence,
    sequence_KL_estimate,
)
from entropy_plots import (
    plot_accuracy_vs_ratio,
    plot_cache_composition,
    plot_dataset_summary,
    plot_distortion_traces,
    plot_entropy_vs_error,
    plot_position_traces,
    plot_quality_traces,
    plot_reliability,
    plot_sampled_distortion,
    set_format,
)
from kvpress import DecodingPress, PrefillDecodingPress, StreamingLLMPress
from kvpress.presses.base_press import BasePress

logger = logging.getLogger(__name__)

METRIC_COLUMNS = ["h_full", "h_comp", "IG", "KL", "ce_full", "ce_comp"]

# Gold-answer column per DATASET_REGISTRY entry: unlike context/question (uniformly
# named across every benchmark's HF push), the gold-answer column isn't standardized.
# Each benchmark's own calculate_metrics.py reads it under its own name. Verified
# directly against each dataset's schema; needle_in_haystack has no gold-answer column
# at all, so its tasks always fall back to a generated reference.
ANSWER_FIELD_REGISTRY = {
    "loogle": "answer",
    "ruler": "answer",
    "zero_scrolls": "answer",
    "infinitebench": "answer",
    "longbench": "answers",
    "longbench-e": "answers",
    "longbench-v2": "answer",
    "needle_in_haystack": None,
    "aime25": "answer",
    "math500": "answer",
}


@dataclass
class Task:
    task_id: str
    context: str
    question: str
    answer: Optional[str] = None
    # Original dataset row minus context/question, kept so the benchmark scorers in
    # SCORER_REGISTRY can read whatever answer/answers/task/all_classes/... columns
    # they need at scoring time. None for ad-hoc --context/--question tasks.
    raw: Optional[dict] = None


# TeacherForcedOutput, SampledOutput and SequenceEntropyEstimate now live in entropy_metrics.


def load_model_and_tokenizer(model_name: str, device: Optional[str] = None):
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    dtype = {"cuda": torch.bfloat16, "mps": torch.float16}.get(device, torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(device)
    model.eval()
    return model, tokenizer


def load_tasks(
    dataset: Optional[str],
    data_dir: Optional[str],
    dataset_split: str,
    dataset_path: Optional[str],
    n_samples: int,
) -> list[Task]:
    """
    Load a list of tasks either from a local JSONL/CSV file (`dataset_path`) or
    from a registered benchmark dataset (`dataset`, a DATASET_REGISTRY key, e.g. 
    `--dataset longbench --data_dir trec`). Streamed so only `n_samples` examples are
    downloaded. Every registered benchmark's HF push uses "context"/"question"
    for these two fields; the gold-answer column varies and is resolved via
    `ANSWER_FIELD_REGISTRY`.
    """
    if dataset_path is not None:
        path = Path(dataset_path)
        if path.suffix == ".jsonl":
            with open(path) as f:
                rows = [json.loads(line) for line in itertools.islice(f, n_samples)]
        else:
            rows = pd.read_csv(path).head(n_samples).to_dict("records")
        answer_field = "answer"
    else:
        from datasets import load_dataset

        assert dataset in DATASET_REGISTRY, f"'{dataset}' not in DATASET_REGISTRY: {list(DATASET_REGISTRY)}"
        ds = load_dataset(DATASET_REGISTRY[dataset], data_dir=data_dir, split=dataset_split, streaming=True)
        rows = list(itertools.islice(ds, n_samples))
        answer_field = ANSWER_FIELD_REGISTRY.get(dataset)

    tasks = []
    for i, row in enumerate(rows):
        answer = row.get(answer_field) if answer_field is not None else None
        if isinstance(answer, list):
            answer = answer[0] if answer else None
        # Keep every original column except the two large text fields so the benchmark
        # scorers can later read the answer/task/all_classes/... columns they expect.
        raw = {k: v for k, v in row.items() if k not in ("context", "question")}
        tasks.append(Task(task_id=str(i), context=row["context"], question=row["question"], answer=answer, raw=raw))
    return tasks


def build_prompt_ids(
    tokenizer: PreTrainedTokenizer,
    context: str,
    question: str,
    device: str,
    max_context_length: Optional[int] = None,
) -> torch.Tensor:
    if max_context_length is not None:
        context_ids = tokenizer.encode(context, add_special_tokens=False)
        if len(context_ids) > max_context_length:
            context = tokenizer.decode(context_ids[:max_context_length])

    messages = [{"role": "user", "content": f"{context}\n\n{question}"}]
    text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    return tokenizer.encode(text, return_tensors="pt", add_special_tokens=False).to(device)


@torch.no_grad()
def generate_reference(model: PreTrainedModel, prompt_ids: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
    """Greedily generate a reference continuation under full attention (no eviction)."""
    cache = DynamicCache()
    outputs = model.generate(
        input_ids=prompt_ids,
        past_key_values=cache,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    return outputs[:, prompt_ids.shape[1] :]


def get_reference_ids(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt_ids: torch.Tensor,
    answer: Optional[str],
    max_new_tokens: int,
    max_reference_tokens: int,
) -> tuple[torch.Tensor, str]:
    """Use the dataset's gold answer as the reference if available, else generate one."""
    if answer:
        ref_ids = tokenizer.encode(answer, return_tensors="pt", add_special_tokens=False).to(model.device)
        if ref_ids.shape[1] > max_reference_tokens:
            ref_ids = ref_ids[:, :max_reference_tokens]
        return ref_ids, "gold"
    return generate_reference(model, prompt_ids, max_new_tokens), "generated"


@torch.no_grad()
def run_teacher_forced_task(
    model: PreTrainedModel,
    prompt_ids: torch.Tensor,
    reference_ids: torch.Tensor,
    press: Optional[BasePress],
    decode_per_token: bool = False,
) -> TeacherForcedOutput:
    """
    args:
        model: PreTrainedModel, the LM to run
        prompt_ids: (1, prompt_len) tensor of token ids to prefill the cache with
        reference_ids: (1, ref_len) tensor of token ids to teacher-force after prefill
        press: optional BasePress to apply during prefill (and optionally decoding)
        decode_per_token: 
                if True, the reference is scored one token at a time under `press` 
                (for decode-time eviction to be exercised); 
                if False, the reference is scored in one batched forward call after 
                prefill (faster but implies that decode-time eviction is not exercised)
    outputs:
        TeacherForcedOutput with per-position log-probabilities and the cache length 
        after prefill

    Prefill `prompt_ids` (optionally compressing the KV cache with `press`), then
    teacher-force `reference_ids` and return per-position log-probabilities.

    Position ids for the teacher-forced tokens continue from `prompt_ids.shape[1]`
    (the *original*, uncompressed prompt length), not from the compressed cache
    length: pruned tokens keep the RoPE position they were encoded with, so the
    query positions must be numbered as if no eviction had happened.

    If `decode_per_token` is False (default), the reference is scored in one
    batched forward call after `press` has already been removed. This is fine for
    a prefill-only press: it only ever prunes once, before this call happens, so
    batching vs. looping token-by-token makes no numerical difference, and batching
    is much faster.

    If `decode_per_token` is True, `press` is kept active (and the reference is fed
    one token at a time) through the whole teacher-forcing pass too. This matters
    for a press like `PrefillDecodingPress` that can also evict *during* decoding:
    a decode-time eviction event only affects tokens generated *after* it fires, so
    if we batched the whole reference into one call, any such event would happen
    only after every prediction we're measuring had already been computed --
    making it invisible to the metrics. Looping one token at a time makes each
    eviction decision causally affect the predictions that come after it.
    """
    prompt_length = prompt_ids.shape[1]
    cache = DynamicCache()

    def prefill():
        # Prefill the cache with the prompt
        # Meaning: it runs the model backbone on the prompt and stores the Key-Value pairs for each prompt token in the cache.
        # If press is not None, it will also prune the cache according to the press's eviction policy.
        model.model(input_ids=prompt_ids, past_key_values=cache)

    if press is not None and decode_per_token:
        # press(model) is a context manager that wraps the model's forward pass to apply the press's eviction policy.
        with press(model):
            prefill()
            log_probs_rows = []
            for i in range(reference_ids.shape[1]):
                position_ids = torch.tensor([[prompt_length + i]], device=model.device)
                outputs = model(
                    input_ids=reference_ids[:, i : i + 1], past_key_values=cache, position_ids=position_ids
                )
                log_probs_rows.append(torch.log_softmax(outputs.logits[0, -1].float(), dim=-1))
            log_probs = torch.stack(log_probs_rows, dim=0)
        cache_seq_length = cache.get_seq_length()
    else:
        if press is not None:
            with press(model):
                prefill()
        else:
            prefill()
        cache_seq_length = cache.get_seq_length()

        position_ids = torch.arange(
            prompt_length, prompt_length + reference_ids.shape[1], device=model.device
        ).unsqueeze(0)
        outputs = model(input_ids=reference_ids, past_key_values=cache, position_ids=position_ids)
        log_probs = torch.log_softmax(outputs.logits[0].float(), dim=-1)

    return TeacherForcedOutput(log_probs=log_probs, cache_seq_length=cache_seq_length)


def resolve_mc_chunks(n_samples: int, mc_batch_size: Optional[int]) -> list[int]:
    """
    Split `n_samples` Monte Carlo draws into per-pass batch sizes of at most `mc_batch_size`.

    `None` (the default) means one pass over all draws, i.e. the unchunked behaviour.

    Chunking exists purely to decouple `n_mc_samples` from GPU memory. Peak memory in the
    sampled regime is ~ batch x context (every row keeps its own KV cache alive for the whole
    decode loop, and prefill activations scale the same way), so a single pass over N draws at
    a long context does not fit. The draws are i.i.d. from p(.|x) and never interact, so
    splitting them across sequential passes is numerically free: each pass allocates its cache,
    finishes, and frees it before the next begins, and per-sample surprisal/length depend only
    on that row's own tokens. Total FLOPs are unchanged -- only GPU utilisation drops.

    The one visible difference is the RNG stream: drawing N samples in k passes consumes
    randomness differently from one pass of N, so a rerun at the same seed gives different
    (identically distributed) draws.
    """
    if mc_batch_size is None or mc_batch_size >= n_samples:
        return [n_samples]
    assert mc_batch_size > 0, "mc_batch_size must be positive"
    full, remainder = divmod(n_samples, mc_batch_size)
    return [mc_batch_size] * full + ([remainder] if remainder else [])


def pad_sequences_to(sequences: torch.Tensor, width: int) -> torch.Tensor:
    """
    Right-pad a (rows, T) token-id block out to `width` columns.

    Chunks stop as soon as *their own* rows have all emitted EOS, so different chunks come back
    with different T. Padding is safe because every consumer masks columns at or beyond a row's
    `lengths[i]`: the padded ids are fed to the model during re-scoring but never contribute to
    any log-probability, and rows are independent so they cannot disturb each other.
    """
    if sequences.shape[1] >= width:
        return sequences
    pad = torch.zeros(
        sequences.shape[0], width - sequences.shape[1], dtype=sequences.dtype, device=sequences.device
    )
    return torch.cat([sequences, pad], dim=1)


@torch.no_grad()
def run_sampled_task(
    model: PreTrainedModel,
    prompt_ids: torch.Tensor,
    press: Optional[BasePress],
    n_samples: int,
    max_new_tokens: int,
    eos_token_id: int,
    mc_batch_size: Optional[int] = None,
) -> SampledOutput:
    """
    args:
        model: PreTrainedModel, the LM to run
        prompt_ids: (1, prompt_len) tensor of token ids to prefill the cache with
        press: optional BasePress to apply during prefill (and optionally decoding)
        n_samples: number of continuations to sample from p(. | prompt_ids)
        max_new_tokens: maximum number of tokens to generate per sample
        eos_token_id: token id of the EOS token, used to stop sampling early
        mc_batch_size: draws per forward pass; None = all `n_samples` at once (see
            `resolve_mc_chunks` for why splitting them is numerically free)
    outputs:
        SampledOutput with per-sample surprisal, generated lengths, and the cache length
        after prefill

    Plug-in Monte Carlo estimator of sequence-level entropy:

        H(Y|x) = E_{y ~ p(.|x)} [ -log p(y | x) ]
        H_hat    = (1/N) * sum_i [ -log p(y^(i) | x) ],   y^(i) ~ p(.|x)

    Draws `n_samples` continuations from p(. | prompt_ids) by explicit ancestral
    sampling at temperature 1.0 -- no top-k/top-p/repetition penalty, since any of
    those would sample from a distorted distribution and bias the estimate -- under
    `press`'s cache-eviction policy.

    `press` is kept active through prefill *and* the whole sampling loop (unlike
    `run_teacher_forced_task`'s batched branch), mirroring `run_teacher_forced_task`'s
    decode_per_token branch:
    since sampling is inherently sequential (each token depends on the last), there
    is no batched fast path here, so a press that also evicts during decoding always
    gets the chance to affect later samples.

    Position ids continue from `prompt_ids.shape[1]` for the same reason as in
    `run_teacher_forced_task`: pruned tokens keep the RoPE position they were encoded with, so
    generated tokens must be numbered as if no eviction had happened.
    """
    chunks = resolve_mc_chunks(n_samples, mc_batch_size)
    if len(chunks) > 1:
        outs = [
            sample_chunk(model, prompt_ids, press, size, max_new_tokens, eos_token_id) for size in chunks
        ]
        width = max(o.sequences.shape[1] for o in outs)
        return SampledOutput(
            surprisal=torch.cat([o.surprisal for o in outs]),
            lengths=torch.cat([o.lengths for o in outs]),
            cache_seq_length=outs[0].cache_seq_length,  # same prompt and press in every chunk
            sequences=torch.cat([pad_sequences_to(o.sequences, width) for o in outs], dim=0),
        )
    return sample_chunk(model, prompt_ids, press, n_samples, max_new_tokens, eos_token_id)


@torch.no_grad()
def sample_chunk(
    model: PreTrainedModel,
    prompt_ids: torch.Tensor,
    press: Optional[BasePress],
    n_samples: int,
    max_new_tokens: int,
    eos_token_id: int,
) -> SampledOutput:
    """One ancestral-sampling pass over `n_samples` rows. See `run_sampled_task` for the method."""
    device = model.device
    prompt_length = prompt_ids.shape[1]
    prompt = prompt_ids.expand(n_samples, -1).contiguous()
    cache = DynamicCache()

    with press(model) if press is not None else contextlib.nullcontext():
        # logits_to_keep=1: the LM head would otherwise project every prompt
        # position to vocab-size logits for the whole n_samples batch, even
        # though only the last position's logits are ever used below.
        outputs = model(input_ids=prompt, past_key_values=cache, logits_to_keep=1)
        cache_seq_length = cache.get_seq_length()
        next_logits = outputs.logits[:, -1, :]

        seq_logp = torch.zeros(n_samples, device=device)
        lengths = torch.zeros(n_samples, dtype=torch.long, device=device)
        active = torch.ones(n_samples, dtype=torch.bool, device=device)
        # Kept so the drawn sequences can be re-scored under another press later
        # (`score_sequences_under_press`), which is what the sequence-KL estimator needs.
        sampled_tokens: list[torch.Tensor] = []

        for i in range(max_new_tokens):
            log_probs = torch.log_softmax(next_logits.float(), dim=-1)
            next_token = torch.multinomial(log_probs.exp(), num_samples=1).squeeze(-1)
            step_logp = log_probs.gather(1, next_token.unsqueeze(1)).squeeze(1)
            sampled_tokens.append(next_token)

            # Only accumulate for samples still active at the start of this step;
            # the step that emits EOS is counted (p(EOS | ...) is part of p(y)).
            seq_logp = seq_logp + step_logp * active.float()
            lengths = lengths + active.long()
            active = active & (next_token != eos_token_id)
            if not active.any():
                break

            position_ids = torch.full((n_samples, 1), prompt_length + i, device=device, dtype=torch.long)
            outputs = model(input_ids=next_token.unsqueeze(1), past_key_values=cache, position_ids=position_ids)
            next_logits = outputs.logits[:, -1, :]

    return SampledOutput(
        surprisal=-seq_logp,
        lengths=lengths,
        cache_seq_length=cache_seq_length,
        sequences=torch.stack(sampled_tokens, dim=1),
    )


@torch.no_grad()
def score_sequences_under_press(
    model: PreTrainedModel,
    prompt_ids: torch.Tensor,
    sequences: torch.Tensor,
    lengths: torch.Tensor,
    press: Optional[BasePress],
    mc_batch_size: Optional[int] = None,
) -> torch.Tensor:
    """
    args:
        model: PreTrainedModel, the LM to run
        prompt_ids: (1, prompt_len) tensor of token ids to prefill the cache with
        sequences: (n_samples, T_max) token ids to score -- drawn from ANOTHER config
        lengths: (n_samples,) true length of each row of `sequences` (later entries ignored)
        press: optional BasePress whose eviction policy the scoring runs under
        mc_batch_size: rows per forward pass; None = all at once. Row order is preserved, so
            `logp[i]` always corresponds to `sequences[i]` regardless of the chunking.
    outputs:
        (n_samples,) tensor of natural-log sequence log-probabilities log p(y^(i) | prompt)
        under `press`

    Score already-drawn sequences under a (possibly compressed) cache, without generating.

    This is the second half of the sequence-KL estimator: `run_sampled_task` draws
    y^(i) ~ p_full and reports log p_full(y^(i)|x); this reports log p_comp(y^(i)|x)
    for those *same* sequences, so the two can be differenced (see `sequence_KL_estimate`).

    Structurally a clone of `run_sampled_task`'s loop with the only difference being how the
    next token is chosen: there it is drawn with `torch.multinomial`, here it is read out of
    `sequences`. Everything else must match for the comparison to be valid -- `press` is held
    active through prefill *and* the decode loop so decode-time eviction fires on the same
    schedule, and `position_ids = prompt_length + t` numbers queries as if no eviction had
    happened, for the same RoPE reason as in `run_sampled_task`.

    Rows that finished early are masked out via `lengths` rather than truncated: the batch
    decodes in lockstep, exactly as it did while sampling.
    """
    chunks = resolve_mc_chunks(sequences.shape[0], mc_batch_size)
    if len(chunks) > 1:
        parts, start = [], 0
        for size in chunks:
            rows = slice(start, start + size)
            # Trim each chunk to its own longest row: columns at or past every row's length
            # contribute nothing, so scoring them would just be wasted decode steps.
            width = int(lengths[rows].max().item())
            parts.append(score_chunk(model, prompt_ids, sequences[rows, :width], lengths[rows], press))
            start += size
        return torch.cat(parts)
    return score_chunk(model, prompt_ids, sequences, lengths, press)


@torch.no_grad()
def score_chunk(
    model: PreTrainedModel,
    prompt_ids: torch.Tensor,
    sequences: torch.Tensor,
    lengths: torch.Tensor,
    press: Optional[BasePress],
) -> torch.Tensor:
    """One forced-token scoring pass over `sequences`. See `score_sequences_under_press`."""
    device = model.device
    n_samples, n_steps = sequences.shape
    prompt_length = prompt_ids.shape[1]
    prompt = prompt_ids.expand(n_samples, -1).contiguous()
    cache = DynamicCache()

    with press(model) if press is not None else contextlib.nullcontext():
        outputs = model(input_ids=prompt, past_key_values=cache, logits_to_keep=1)
        next_logits = outputs.logits[:, -1, :]

        seq_logp = torch.zeros(n_samples, device=device)
        for t in range(n_steps):
            log_probs = torch.log_softmax(next_logits.float(), dim=-1)
            token = sequences[:, t]
            step_logp = log_probs.gather(1, token.unsqueeze(1)).squeeze(1)

            # Mirrors run_sampled_task's `active` mask: only positions inside a row's own
            # generated length contribute to that row's sequence log-probability.
            seq_logp = seq_logp + step_logp * (t < lengths).float()
            if t == n_steps - 1:
                break

            position_ids = torch.full((n_samples, 1), prompt_length + t, device=device, dtype=torch.long)
            outputs = model(input_ids=token.unsqueeze(1), past_key_values=cache, position_ids=position_ids)
            next_logits = outputs.logits[:, -1, :]

    return seq_logp


@torch.no_grad()
def greedy_generate_output(
    model: PreTrainedModel,
    prompt_ids: torch.Tensor,
    press: Optional[BasePress],
    max_new_tokens: int,
    eos_token_id: int,
) -> torch.Tensor:
    """
    args:
        model: PreTrainedModel, the LM to run
        prompt_ids: (1, prompt_len) tensor of token ids to prefill the cache with
        press: optional BasePress to apply during prefill (and optionally decoding)
        max_new_tokens: maximum number of tokens to generate
        eos_token_id: token id of the EOS token, used to stop generation early
    outputs:
        (gen_len,) tensor of generated token ids, excluding the prompt

    Greedily decode one continuation under `press`, returning the generated token ids.

    This is the answer that gets scored against the gold answer. Greedy rather than sampled: 
    it is the model's single canonical answer, so the accuracy number isn't itself a random draw. 
    The *uncertainty* paired with it still comes from `run_sampled_task`'s temperature-1.0 
    Monte Carlo estimate of H(Y|x).

    Structure mirrors `run_sampled_task` exactly -- press active through prefill *and* the
    decode loop (so decode-time eviction can fire and affect later tokens), and explicit
    `position_ids = prompt_length + i` so pruned tokens keep the RoPE positions they were
    encoded with. Only the token choice differs: argmax instead of `torch.multinomial`.
    """
    device = model.device
    prompt_length = prompt_ids.shape[1]
    cache = DynamicCache()
    tokens: list[int] = []

    with press(model) if press is not None else contextlib.nullcontext():
        outputs = model(input_ids=prompt_ids, past_key_values=cache, logits_to_keep=1)
        next_logits = outputs.logits[:, -1, :]

        for i in range(max_new_tokens):
            next_token = next_logits.argmax(dim=-1)  # (1,)
            if next_token.item() == eos_token_id:
                break
            tokens.append(int(next_token.item()))

            position_ids = torch.full((1, 1), prompt_length + i, device=device, dtype=torch.long)
            outputs = model(input_ids=next_token.unsqueeze(1), past_key_values=cache, position_ids=position_ids)
            next_logits = outputs.logits[:, -1, :]

    return torch.tensor(tokens, dtype=torch.long)


def build_comp_press(
    ratio: float, n_sink: int, prompt_length: int, decoding_compression_interval: Optional[int],
    decoding_target_size: Optional[int],
) -> tuple[BasePress, bool]:
    """
    args: 
        ratio: compression ratio for StreamingLLMPress (0.0 = no eviction, 1.0 = max eviction)
        n_sink: number of sink tokens for StreamingLLMPress
        prompt_length: length of the prompt (used to compute decoding_target_size if not provided)
        decoding_compression_interval: if set, apply a DecodingPress every this many decode steps
        decoding_target_size: if set, target size for DecodingPress; if None, computed from prompt_length 
        and ratio
    outputs:
        BasePress to use for the "compressed" Task, and a boolean indicating whether decode-time 
        eviction is exercised 
        (i.e., whether `run_teacher_forced_task` should loop per token)

    Build the press used for the "compressed" Task, and whether it needs the
    per-token decode loop in `run_teacher_forced_task` to actually exercise decode-time eviction.

    If `decoding_compression_interval` is None, this is the prefill-only
    behaviour: a bare StreamingLLMPress, pruned once during prefill and never again.

    If `decoding_compression_interval` is set, StreamingLLMPress is also wrapped in
    a `DecodingPress` (via `PrefillDecodingPress`), which re-applies StreamingLLM's
    sink+recency scoring rule every `decoding_compression_interval` decode steps,
    pruning the cache back down to `decoding_target_size` tokens each time.

    ratio=0.0 always disables eviction entirely, at both prefill and decode, even
    if decode-time compression is requested: `target_size = prompt_length` would
    otherwise still let the cache get trimmed once it grows past the *original*
    prompt length during decoding.
    """
    prefill_press = StreamingLLMPress(compression_ratio=ratio, n_sink=n_sink)
    if decoding_compression_interval is None or ratio == 0.0:
        return prefill_press, False

    target_size = decoding_target_size or max(n_sink + 1, int(prompt_length * (1 - ratio)))
    decode_press = DecodingPress(
        base_press=StreamingLLMPress(compression_ratio=0.0, n_sink=n_sink),
        compression_interval=decoding_compression_interval,
        target_size=target_size,
        hidden_states_buffer_size=0,
    )
    return PrefillDecodingPress(prefilling_press=prefill_press, decoding_press=decode_press), True


def process_teacher_forced_task(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    task: Task,
    compression_ratios: list[float],
    n_sink: int,
    max_new_tokens: int,
    max_reference_tokens: int,
    max_context_length: Optional[int],
    decoding_compression_interval: Optional[int] = None,
    decoding_target_size: Optional[int] = None,
) -> Optional[pd.DataFrame]:
    """
    Teacher-forced regime. Scores the reference continuation (gold if available, else
    generated under full attention) under full attention and each compression ratio,
    both seeing the same prefix at every position, and returns per-position
    entropy / KL / cross-entropy. One row per (task, ratio, position); the `full` vs
    `compressed` columns (h_full/h_comp, ce_full/ce_comp, ...) come paired from
    `compute_sequence_metrics`. Ratio 0.0 is skipped (identical to full).
    """
    prompt_ids = build_prompt_ids(tokenizer, task.context, task.question, model.device, max_context_length)
    reference_ids, ref_source = get_reference_ids(
        model, tokenizer, prompt_ids, task.answer, max_new_tokens, max_reference_tokens
    )
    if reference_ids.shape[1] == 0:
        logger.warning(f"task {task.task_id}: empty reference sequence, skipping")
        return None

    logger.info(
        f"task {task.task_id}: prompt_len={prompt_ids.shape[1]} ref_len={reference_ids.shape[1]} "
        f"ref_source={ref_source}"
    )

    full = run_teacher_forced_task(model, prompt_ids, reference_ids, press=None)

    records = []
    for ratio in compression_ratios:
        if ratio == 0.0:  # identical to full attention; skip
            continue
        press, decode_per_token = build_comp_press(
            ratio, n_sink, prompt_ids.shape[1], decoding_compression_interval, decoding_target_size
        )
        compressed = run_teacher_forced_task(model, prompt_ids, reference_ids, press=press, decode_per_token=decode_per_token)
        df = compute_sequence_metrics(full, compressed, reference_ids)
        df.insert(0, "ratio", ratio)
        df.insert(0, "task_id", task.task_id)
        df["cache_seq_length_full"] = full.cache_seq_length
        df["cache_seq_length_comp"] = compressed.cache_seq_length
        records.append(df)

    return pd.concat(records, ignore_index=True) if records else None


def process_task_sampled(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    task: Task,
    compression_ratios: list[float],
    n_sink: int,
    n_mc_samples: int,
    max_new_tokens: int,
    max_context_length: Optional[int],
    decoding_compression_interval: Optional[int] = None,
    decoding_target_size: Optional[int] = None,
    compute_sequence_KL: bool = True,
    mc_batch_size: Optional[int] = None,
) -> pd.DataFrame:
    """
    Sampled regime. For each KV-caching config -- `full` attention plus every user
    compression ratio (0.0 skipped, == full) -- this does two passes over the prompt:
    `run_sampled_task` (n_mc_samples draws -> sequence entropy H(Y|x) and a confidence)
    and `greedy_generate_output` (one deterministic answer, scored against gold later,
    in `main`, by the dataset's benchmark scorer).

    With `compute_sequence_KL`, each *compressed* config additionally gets a third pass:
    `score_sequences_under_press` re-scores the sequences the `full` config drew, giving
    log p_comp(y^(i)|x) for y^(i) ~ p_full and hence the naive Monte Carlo estimate of
    KL(p_full || p_comp) (see `sequence_KL_estimate`). This is why `full` must be run
    first and its `SampledOutput` held across the ratio loop: KL needs both models evaluated
    on one common set of sequences, and the compressed configs' own draws are different
    strings that cannot be differenced against full's.

    Returns the long `per_config` layout: one row per (task, kv_caching, ratio). `full`
    is `kv_caching="full", ratio=NaN`; each compressed config is `kv_caching="compressed",
    ratio=r`. `config_label` ("full" / "compressed@r") is a convenience key for grouping
    and per-config plots. The KL columns are NaN on the `full` row (KL against itself is 0
    by construction and carries no information).
    """
    prompt_ids = build_prompt_ids(tokenizer, task.context, task.question, model.device, max_context_length)
    eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else model.config.eos_token_id
    gold = task.answer or ""

    logger.info(
        f"task {task.task_id}: prompt_len={prompt_ids.shape[1]} n_mc_samples={n_mc_samples} "
        f"gold_len={len(gold)} chars"
    )

    def config_row(
        kv_caching: str,
        config_label: str,
        ratio: float,
        press: Optional[BasePress],
        sampled: Optional[SampledOutput] = None,
        KL_estimate: Optional[SequenceKLEstimate] = None,
    ) -> dict:
        if sampled is None:
            sampled = run_sampled_task(
                model, prompt_ids, press, n_mc_samples, max_new_tokens, eos_token_id, mc_batch_size
            )
        estimate = sequence_entropy_estimate(sampled)
        generated_ids = greedy_generate_output(model, prompt_ids, press, max_new_tokens, eos_token_id)
        predicted_answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        return {
            "regime": "sampled",
            "task_id": task.task_id,
            "kv_caching": kv_caching,
            "ratio": ratio,
            "config_label": config_label,
            "n_mc_samples": n_mc_samples,
            "H_seq": estimate.H_hat,
            "H_seq_se": estimate.se,
            "varentropy_seq": estimate.varentropy,
            "mean_length": estimate.mean_length,
            "confidence": sequence_confidence(estimate),
            "greedy_length": int(generated_ids.numel()),
            "cache_seq_length": estimate.cache_seq_length,
            "KL_seq": KL_estimate.KL if KL_estimate is not None else float("nan"),
            "KL_seq_se": KL_estimate.se if KL_estimate is not None else float("nan"),
            "n_KL_samples": KL_estimate.n_samples if KL_estimate is not None else 0,
            "gold": gold,
            "predicted_answer": predicted_answer,
        }

    # `full` first, and its draws kept: they are the y^(i) ~ p_full the KL estimator needs.
    full_sampled = run_sampled_task(
        model, prompt_ids, None, n_mc_samples, max_new_tokens, eos_token_id, mc_batch_size
    )
    rows = [config_row("full", "full", float("nan"), None, sampled=full_sampled)]

    for ratio in compression_ratios:
        if ratio == 0.0:  # identical to full attention; skip
            continue
        press, _ = build_comp_press(
            ratio, n_sink, prompt_ids.shape[1], decoding_compression_interval, decoding_target_size
        )
        KL_estimate = None
        if compute_sequence_KL:
            logp_comp = score_sequences_under_press(
                model, prompt_ids, full_sampled.sequences, full_sampled.lengths, press, mc_batch_size
            )
            KL_estimate = sequence_KL_estimate(-full_sampled.surprisal, logp_comp)
            if KL_estimate.KL < 0:
                logger.warning(
                    f"task {task.task_id} ratio {ratio}: KL_seq={KL_estimate.KL:.3f} bits < 0 at "
                    f"N={KL_estimate.n_samples} (se={KL_estimate.se:.3f}). KL is non-negative by "
                    "definition, so this estimate is variance-dominated -- do not trust its magnitude."
                )
        rows.append(config_row("compressed", f"compressed@{ratio}", ratio, press, KL_estimate=KL_estimate))

    return pd.DataFrame.from_records(rows)


def build_sampled_comparison(per_config: pd.DataFrame) -> pd.DataFrame:
    """
    Pair `full` against each `compressed` config into one comparison row per (task, ratio):
    full vs compressed entropy, confidence, and benchmark score side by side.

    `context_length` is carried through from the *full* config's cache length, which is the
    prompt length (nothing is evicted there) -- the compressed rows' `cache_seq_length` is the
    surviving cache, a different quantity. It rides along so the per-task trace plots can color
    by context without rejoining `per_config`.
    """
    # drop_duplicates before set_index: a duplicated task would make `.loc` below return a
    # DataFrame rather than a Series, so every arithmetic column would silently become an
    # object-dtype Series-of-Series. `main` already dedupes; this keeps the function safe to
    # call on a raw CSV too.
    full = (
        per_config[per_config["kv_caching"] == "full"]
        .drop_duplicates(subset="task_id", keep="last")
        .set_index("task_id")
    )
    rows = []
    for _, c in per_config[per_config["kv_caching"] == "compressed"].iterrows():
        f = full.loc[c["task_id"]]
        rows.append(
            {
                "regime": "sampled",
                "task_id": c["task_id"],
                "ratio": c["ratio"],
                "context_length": f.get("cache_seq_length", float("nan")),
                "H_seq_full": f["H_seq"],
                "H_seq_comp": c["H_seq"],
                "IG_seq": c["H_seq"] - f["H_seq"],
                "KL_seq": c.get("KL_seq", float("nan")),
                "KL_seq_se": c.get("KL_seq_se", float("nan")),
                "confidence_full": f["confidence"],
                "confidence_comp": c["confidence"],
                "score_full": f.get("primary_score", float("nan")),
                "score_comp": c.get("primary_score", float("nan")),
                "delta_score": c.get("primary_score", float("nan")) - f.get("primary_score", float("nan")),
            }
        )
    return pd.DataFrame.from_records(rows)


def aggregate_teacher_forced_task_metrics(records: pd.DataFrame) -> pd.DataFrame:
    """
    Dataset-level H (Y | x) estimate per compression ratio.

    Each task is first averaged over its own positions (giving one bits/token
    number per task), then averaged across tasks, so long and short reference
    sequences count equally as one sample each.
    """
    per_task = records.groupby(["ratio", "task_id"])[METRIC_COLUMNS].mean().reset_index()
    summary = per_task.groupby("ratio")[METRIC_COLUMNS].agg(["mean", "std"])
    summary.columns = ["_".join(c) for c in summary.columns]
    summary = summary.reset_index()
    summary["excess_ce"] = summary["ce_comp_mean"] - summary["ce_full_mean"]
    summary["n_tasks"] = per_task.groupby("ratio").size().values
    return summary.rename(
        columns={
            "h_full_mean": "H_full_per_token",
            "h_comp_mean": "H_comp_per_token",
            "IG_mean": "IG",
            "KL_mean": "mean_KL",
        }
    )


def build_tf_comparison(per_position: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse the teacher-forced per-position records to one comparison row per (task, ratio):
    full vs compressed metrics averaged over positions, plus excess_ce. (The internal
    per-position columns still use the historical `*_comp` names -> `*_comp` here.)
    """
    agg = (
        per_position.groupby(["task_id", "ratio"])
        .agg(
            h_full=("h_full", "mean"),
            h_comp=("h_comp", "mean"),
            IG=("IG", "mean"),
            KL=("KL", "mean"),
            ce_full=("ce_full", "mean"),
            ce_comp=("ce_comp", "mean"),
        )
        .reset_index()
    )
    agg["excess_ce"] = agg["ce_comp"] - agg["ce_full"]
    agg.insert(0, "regime", "teacher_forced")
    return agg


def build_tf_per_config(per_position: pd.DataFrame) -> pd.DataFrame:
    """
    Teacher-forced `per_config` layout: one row per (task, kv_caching, ratio), with mean
    predictive entropy per token and mean surprisal of the gold reference. `full` (computed
    once per task) is one row with ratio=NaN; each compressed ratio adds a row.
    """
    compressed = (
        per_position.groupby(["task_id", "ratio"])
        .agg(h_mean=("h_comp", "mean"), ce_mean=("ce_comp", "mean"))
        .reset_index()
    )
    compressed["kv_caching"] = "compressed"
    full = (
        per_position.groupby("task_id")
        .agg(h_mean=("h_full", "mean"), ce_mean=("ce_full", "mean"))
        .reset_index()
    )
    full["kv_caching"] = "full"
    full["ratio"] = float("nan")
    out = pd.concat([full, compressed], ignore_index=True)
    out.insert(0, "regime", "teacher_forced")
    return out[["regime", "task_id", "kv_caching", "ratio", "h_mean", "ce_mean"]]


# --------------------------------------------------------------------------------------
# Benchmark-backed accuracy scoring
#
# All accuracy numbers come from each dataset's own scorer in evaluation/benchmarks/*
# (via SCORER_REGISTRY). Two shapes are needed:
#   - aggregate, per (Task, ratio) group -> the benchmark's dataset-level
#     metric(s), the same numbers evaluate.py reports;
#   - per-example, one scalar in [0, 1] -> drives the ECE / entropy-vs-error calibration.
# The scorers return wildly different shapes (scalar, {task: {...}}, bucketed dict,
# list-of-dicts, {}), so `flatten_metrics` normalises them and `PRIMARY_METRIC` names
# the single key used as "the" accuracy per dataset. loogle is special-cased because its
# scorer always runs BERTScore (too slow per example), so its per-example path reuses the
# loogle BLEU/ROUGE/METEOR helpers directly and skips BERT.
# --------------------------------------------------------------------------------------

# Metric key (post-flatten) used as the calibration accuracy per dataset. None => the
# benchmark exposes no usable per-example score, so calibration is skipped.
PRIMARY_METRIC = {
    "loogle": "rouge-1",
    "ruler": "string_match",
    "zero_scrolls": None,
    "infinitebench": "score",
    "longbench": "score",
    "longbench-e": "score",
    "longbench-v2": "average",
    "needle_in_haystack": "rouge-l",
    "aime25": "accuracy",
    "math500": "accuracy",
}

# Scorers that report on a 0-100 percentage scale; their per-example score is divided by
# 100 so the calibration threshold and the confidence axis both live in [0, 1].
PERCENT_SCALE = {"longbench", "longbench-e", "ruler", "infinitebench"}


def _clean(value) -> str:
    return value if isinstance(value, str) and value.strip() else "<NONE>"


def _mean(xs) -> float:
    xs = [x for x in xs if x == x]  # drop NaN
    return sum(xs) / len(xs) if xs else float("nan")


def _leaf(value) -> float:
    """Reduce one metric value to a float: rouge {r,p,f} -> f; any other dict -> mean."""
    if isinstance(value, dict):
        if "f" in value:
            return float(value["f"])
        nums = [_leaf(v) for v in value.values()]
        return _mean(nums)
    return float(value)


def flatten_metrics(obj) -> dict:
    """Normalise a benchmark scorer's return (scalar / {task:{...}} / dict / list) to a flat dict."""
    if isinstance(obj, (int, float)):
        return {"score": float(obj)}
    if isinstance(obj, list):  # e.g. needle: list of per-row rouge dicts
        if not obj:
            return {}
        keys = obj[0].keys()
        return {k: _mean([_leaf(d[k]) for d in obj if k in d]) for k in keys}
    if isinstance(obj, dict):
        if obj and all(isinstance(v, dict) for v in obj.values()):  # {task: metrics}
            metric_keys = set().union(*(set(v.keys()) for v in obj.values()))
            return {k: _mean([_leaf(v[k]) for v in obj.values() if k in v]) for k in metric_keys}
        return {k: _leaf(v) for k, v in obj.items()}
    return {}


def _loogle_per_row(df: pd.DataFrame) -> list[dict]:
    """Per-row BLEU/ROUGE/METEOR (no BERT) by reusing the loogle benchmark's own scorers."""
    import nltk
    from benchmarks.loogle.calculate_metrics import (
        get_bleu_score,
        get_meteor_score,
        get_rouge_score,
        try_except_metric,
    )

    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)
    metric_fns = [try_except_metric(fn) for fn in (get_bleu_score, get_rouge_score, get_meteor_score)]
    rows = []
    for gold, pred in zip(df["gold"], df["predicted_answer"]):
        row: dict = {}
        for fn in metric_fns:
            row.update(fn(_clean(gold), _clean(pred)))
        rows.append(row)
    return rows


def _loogle_aggregate(df: pd.DataFrame, compute_bertscore: bool) -> dict:
    """loogle group metrics reusing its scorers: BLEU/ROUGE/METEOR means (+ batched BERT-F1)."""
    per_row = _loogle_per_row(df)
    if not per_row:
        return {}
    out = {k: _mean([r[k] for r in per_row]) for k in per_row[0]}
    if compute_bertscore:
        from bert_score import score

        golds = [_clean(v) for v in df["gold"]]
        preds = [_clean(v) for v in df["predicted_answer"]]
        out["bert"] = float(score(preds, golds, lang="EN")[2].mean().item())
    return out


def benchmark_metrics_aggregate(group_df: pd.DataFrame, dataset_key: Optional[str], compute_bertscore: bool) -> dict:
    """Dataset-level metrics for one (Task, ratio) group, straight from the benchmark scorer."""
    if dataset_key is None:  # local/ad-hoc data: no registered scorer, reuse loogle metrics
        return _loogle_aggregate(group_df, compute_bertscore)
    if dataset_key == "loogle":
        if compute_bertscore:
            return flatten_metrics(SCORER_REGISTRY["loogle"](group_df.copy()))
        return _loogle_aggregate(group_df, compute_bertscore=False)
    return flatten_metrics(SCORER_REGISTRY[dataset_key](group_df.copy()))


def benchmark_score_per_example(df: pd.DataFrame, dataset_key: Optional[str]) -> list[float]:
    """
    Per-row correctness in [0, 1] for calibration, obtained by reusing the benchmark's own
    scorer: apply it to one-row DataFrames (each scorer already reduces its input) and pull
    the dataset's PRIMARY_METRIC. loogle / local data reuse the loogle ROUGE helper instead
    (its full scorer would run BERTScore per row). Returns NaN where no per-example score is
    available (e.g. zero_scrolls).
    """
    if dataset_key is None:
        return [r["rouge-1"] for r in _loogle_per_row(df)]
    primary = PRIMARY_METRIC.get(dataset_key)
    if primary is None:
        return [float("nan")] * len(df)
    if dataset_key == "loogle":
        return [r["rouge-1"] for r in _loogle_per_row(df)]
    scorer = SCORER_REGISTRY[dataset_key]
    scale = 100.0 if dataset_key in PERCENT_SCALE else 1.0
    scores = []
    for _, row in df.iterrows():
        flat = flatten_metrics(scorer(pd.DataFrame([row]).copy()))
        scores.append(flat.get(primary, float("nan")) / scale)
    return scores


def aggregate_sampled(
    per_config: pd.DataFrame, dataset_key: Optional[str], compute_bertscore: bool, n_ece_bins: int, ece_bin_strategy: str
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, float], list[str]]:
    """
    Dataset-level sampled summary: one row per kv_caching config (`full`, then each
    compressed ratio), carrying the benchmark's own metric(s), the mean primary score across
    tasks, the ECE, and mean/std of entropy/confidence across tasks.

    `accuracy` is the plain mean of `primary_score` over tasks, and the ECE bins compare
    confidence against that same continuous score -- no thresholding into correct/incorrect
    anywhere in the pipeline.

    Also returns each config's per-bin calibration table (for the reliability diagram),
    its ECE (keyed by `config_label`), and the benchmark-metric column names present (for
    the accuracy-vs-ratio plot; dataset-dependent).
    """
    summary_rows = []
    bin_tables: dict[str, pd.DataFrame] = {}
    eces: dict[str, float] = {}
    metric_columns: set[str] = set()

    for label, group in per_config.groupby("config_label", sort=False):
        ece, bin_table = compute_ece(group["confidence"], group["primary_score"], n_ece_bins, ece_bin_strategy)
        bin_tables[label] = bin_table
        eces[label] = ece

        # The benchmark scorer already aggregates over the group's tasks, so this is one
        # number per metric per config -- stored as `{metric}_mean` for the plot's sake.
        benchmark = benchmark_metrics_aggregate(group, dataset_key, compute_bertscore)
        metric_columns.update(benchmark.keys())

        row = {
            "regime": "sampled",
            "config_label": label,
            "kv_caching": group["kv_caching"].iloc[0],
            "ratio": group["ratio"].iloc[0],
            "n_tasks": len(group),
            "accuracy": group["primary_score"].mean(),
            "ece": ece,
        }
        for metric, value in benchmark.items():
            row[f"{metric}_mean"] = value
        # KL_seq is absent from records written before it existed, and is all-NaN for the
        # `full` config (KL against itself is 0 by construction), which pandas renders as NaN.
        for column in ["primary_score", "H_seq", "confidence", "greedy_length", "KL_seq"]:
            if column not in group:
                continue
            row[f"{column}_mean"] = group[column].mean()
            row[f"{column}_std"] = group[column].std()
        summary_rows.append(row)

    return pd.DataFrame(summary_rows), bin_tables, eces, sorted(metric_columns)


def write_run_config(out_dir: Path, run_args: dict) -> None:
    """
    Dump the exact args `main` was called with -- defaults included, not just the ones
    passed on the CLI -- plus provenance (full command, git commit, UTC timestamp) to
    `config.json` in the results folder. This is the record of *how* a run was produced;
    the directory name only carries the few fields worth browsing by. Reload these across
    runs to reconstruct or compare a whole sweep.
    """
    try:
        commit = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        commit = None

    config = {
        "args": run_args,
        "command": "python " + " ".join(sys.argv),
        "git_commit": commit,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2, default=str)


def add_folder_log_handler(out_dir: Path, piped_log_name: str = "my_log.log") -> None:
    """
    Fallback file log: if the run was NOT piped through `tee "$OUT/<piped_log_name>"`,
    attach a logging handler that writes <out_dir>/run.log, so a log still lands next to
    the CSVs. When you DO pipe, `tee` creates that file at launch (before main() runs),
    so its presence means "already being logged" and we skip the redundant backup. This
    is an independent logging sink -- it never touches stdout/stderr, so it can't fight
    `tee`; it also only captures logging-module output, not raw print()/tqdm/library
    stderr (pipe through `tee` for the full stream).

    Detection is heuristic: it keys off a hardcoded filename and mere existence, so a
    stale <piped_log_name> left by a PRIOR piped run into the same folder will suppress
    the backup on a later un-piped run. Delete that file (or the folder) to reset.
    """
    if (out_dir / piped_log_name).exists():
        return  # being piped to tee already; skip the redundant backup log
    file_handler = logging.FileHandler(out_dir / "run.log")
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logging.getLogger().addHandler(file_handler)


def main(
    model: str = "Qwen/Qwen2.5-0.5B-Instruct",
    context: Optional[str] = None,
    question: Optional[str] = None,
    answer: Optional[str] = None,
    dataset: Optional[str] = "loogle",
    data_dir: Optional[str] = "shortdep_qa",
    dataset_split: str = "test",
    dataset_path: Optional[str] = None,
    n_samples: int = 5,
    compression_ratios: str = "[0.0, 0.25, 0.5, 0.75]",
    n_sink: int = 4,
    max_new_tokens: int = 64,
    max_reference_tokens: int = 48,
    max_context_length: Optional[int] = None,
    decoding_compression_interval: Optional[int] = None,
    decoding_target_size: Optional[int] = None,
    teacher_forcing: bool = True,
    n_mc_samples: int = 200,
    mc_batch_size: Optional[int] = None,
    compute_sequence_KL: bool = True,
    compute_bertscore: bool = True,
    ece_metric: str = "rouge-1",
    n_ece_bins: int = 5,
    ece_bin_strategy: str = "quantile",
    device: Optional[str] = None,
    output_dir: str = "./results/entropy_analysis",
    fig_format: str = "png",
    seed: int = 42,
):
    """
    Compare full attention against compression across a sweep of ratios and
    a dataset of tasks. See the module docstring for the metrics and output CSVs.

    `teacher_forcing` picks the regime:
    - True  -> teacher-forced per-position entropy / KL / cross-entropy (quality metric =
      excess_ce); no answer is generated, so no benchmark accuracy.
    - False -> sampled sequence entropy H(Y|x) PLUS benchmark accuracy: each config also
      greedy-decodes one answer, scored by the dataset's own scorer (SCORER_REGISTRY),
      with a confidence-vs-accuracy ECE / reliability / entropy-vs-error analysis.

    Both regimes always score `full` attention plus every user compression ratio (0.0
    skipped, == full), and write `comparison.csv` (full vs compressed per task) and
    `per_config.csv` (one row per kv_caching config). `decoding_compression_interval`
    additionally re-applies every N decode steps (decode-time eviction).
    `ece_metric` is the fallback primary metric for local/ad-hoc data; `compute_bertscore=
    False` skips loogle's roberta-large download. `compute_sequence_KL` (sampled regime only)
    adds one extra scoring pass per compressed ratio to estimate KL(p_full || p_comp);
    setting it False restores the pre-KL runtime exactly.

    `mc_batch_size` caps how many of the `n_mc_samples` draws share a forward pass (default:
    all of them). Peak GPU memory in the sampled regime scales with batch x context, so this
    is the knob that lets a large `n_mc_samples` -- which is what shrinks the H_seq_se and
    KL_seq_se error bars -- coexist with a long `max_context_length`. Chunking costs GPU
    utilisation, not FLOPs, and does not change any per-sample quantity; it does change the
    RNG stream, so chunked runs are not bit-reproducible against unchunked ones.

    `fig_format` picks the file format for every figure (png / svg / pdf); the figure size and
    type size are the same either way.
    """
    run_args = dict(locals())  # exact keyword args main() got (incl. defaults); capture before any locals
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    torch.manual_seed(seed)
    # Validated up front: the plots are the last thing this writes, and an unsupported format
    # should not surface after the GPU work is already done.
    set_format(fig_format)

    ratios = eval(compression_ratios) if isinstance(compression_ratios, str) else list(compression_ratios)
    mode_suffix = "decode" if decoding_compression_interval is not None else "prefill"
    regime_suffix = "tf" if teacher_forcing else "sampled"
    out_dir = Path(output_dir)
    if output_dir.endswith(f"{regime_suffix}_{mode_suffix}"):
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = out_dir.with_name(f"{out_dir.name}_{regime_suffix}_{mode_suffix}")
        out_dir.mkdir(parents=True, exist_ok=True)

    write_run_config(out_dir, run_args)
    add_folder_log_handler(out_dir)  # fallback log in the results folder; coexists with `tee`
    logger.info(f"Writing results to {out_dir}")

    if context is not None and question is not None:
        tasks = [Task(task_id="0", context=context, question=question, answer=answer)]
    else:
        tasks = load_tasks(dataset, data_dir, dataset_split, dataset_path, n_samples)
    logger.info(f"Loaded {len(tasks)} task(s)")

    # dataset_key indexes SCORER_REGISTRY; None for ad-hoc --context/--question or local
    # --dataset_path data, which have no registered benchmark scorer.
    dataset_key = None if (context is not None or dataset_path is not None) else dataset
    # Raw dataset rows for every task, captured before the resume filter below drops the
    # already-completed ones, so the benchmark scorers can read their answer/task/... columns.
    raw_by_id = {task.task_id: (task.raw or {}) for task in tasks}

    # records.csv is the incremental, resumable per-task file (per-config rows for sampled,
    # per-position rows for teacher-forced); the two headline CSVs are derived from it.
    records_path = out_dir / "records.csv"
    if records_path.exists():
        completed = set(pd.read_csv(records_path, usecols=["task_id"], dtype={"task_id": str})["task_id"])
        tasks = [task for task in tasks if task.task_id not in completed]
        logger.info(f"Resuming: {len(completed)} task(s) already in {records_path.name}, {len(tasks)} remaining")

    model_, tokenizer = load_model_and_tokenizer(model, device)

    if not teacher_forcing:
        # ---- sampled regime: entropy H(Y|x) + benchmark accuracy + calibration ----
        for task in tasks:
            df = process_task_sampled(
                model_, tokenizer, task, ratios, n_sink, n_mc_samples, max_new_tokens, max_context_length,
                decoding_compression_interval, decoding_target_size, compute_sequence_KL, mc_batch_size,
            )
            df.to_csv(records_path, mode="a", header=not records_path.exists(), index=False)

        if not records_path.exists():
            logger.error("No task produced usable records, aborting.")
            return

        per_config = pd.read_csv(
            records_path,
            dtype={"task_id": str, "predicted_answer": str, "gold": str},
            keep_default_na=False,
            na_values=[""],          # only truly-empty fields become NaN; "N/A" stays the string
        )

        # records.csv is append-only, and the resume filter at startup cannot protect against
        # two jobs writing the same file concurrently -- which leaves the same (task, config)
        # recorded twice. Left alone that silently double-weights those tasks in every mean,
        # accuracy and ECE, and makes `build_sampled_comparison`'s per-task lookup return two
        # rows instead of one (turning its arithmetic into object-dtype Series and blowing up
        # the first numeric aggregation downstream). Last write wins, as for a resumed task.
        n_before = len(per_config)
        per_config = per_config.drop_duplicates(subset=["task_id", "config_label"], keep="last")
        if len(per_config) < n_before:
            logger.warning(
                f"{n_before - len(per_config)} duplicate (task_id, config_label) row(s) in "
                f"{records_path.name} -- keeping the last of each. This usually means two runs "
                "appended to the same results dir; consider re-running into a clean one."
            )


        # Re-attach the benchmark scorers' input columns (answer/answers/task/all_classes/...)
        # from the in-memory raw rows rather than the CSV, which would have turned any list
        # column (e.g. longbench "answers") into a string and broken the scorer.
        raw_df = pd.DataFrame([{"task_id": tid, **raw} for tid, raw in raw_by_id.items()])
        per_config = per_config.merge(raw_df, on="task_id", how="left")

        # Correctness for the calibration axis is the dataset's own primary metric (via its
        # benchmark scorer); ece_metric is only the fallback for local/ad-hoc data.
        primary_metric = PRIMARY_METRIC.get(dataset_key) if dataset_key is not None else ece_metric
        calibrated = primary_metric is not None
        per_config["primary_score"] = benchmark_score_per_example(per_config, dataset_key)
        per_config["error"] = 1.0 - per_config["primary_score"]
        per_config.to_csv(out_dir / "per_config.csv", index=False)

        comparison = build_sampled_comparison(per_config)
        comparison.to_csv(out_dir / "comparison.csv", index=False)

        summary, bin_tables, eces, metric_columns = aggregate_sampled(
            per_config, dataset_key, compute_bertscore, n_ece_bins, ece_bin_strategy
        )
        summary.to_csv(out_dir / "summary.csv", index=False)

        report_columns = ["config_label", "n_tasks", "accuracy", "ece", "H_seq_mean", "confidence_mean"] + [
            f"{c}_mean" for c in metric_columns
        ]
        logger.info(
            f"Sampled summary (benchmark metrics, primary={primary_metric}):\n"
            f"{summary[report_columns].to_string(index=False)}"
        )

        if compute_sequence_KL and not comparison.empty:
            distortion = (
                comparison.groupby("ratio")[["IG_seq", "KL_seq", "KL_seq_se"]]
                .agg(["mean", "std"])
                .round(3)
            )
            logger.info(f"Sampled distortion vs full attention (bits/sequence):\n{distortion.to_string()}")
            n_negative = int((comparison["KL_seq"] < 0).sum())
            if n_negative:
                logger.warning(
                    f"{n_negative}/{len(comparison)} per-task KL estimates are negative, which is "
                    f"impossible for a true KL. The naive Monte Carlo estimator is variance-dominated "
                    f"at n_mc_samples={n_mc_samples}; treat the magnitudes as unusable and raise "
                    "n_mc_samples before quoting them."
                )

        n_tasks = per_config["task_id"].nunique()
        if calibrated and n_tasks < 4 * n_ece_bins:
            logger.warning(
                f"ECE is being estimated from {n_tasks} tasks across {n_ece_bins} bins "
                f"(~{n_tasks / n_ece_bins:.1f} tasks/bin). Treat it as illustrative only -- "
                f"a trustworthy reliability diagram needs on the order of 100-200 tasks."
            )

        if calibrated:
            plot_entropy_vs_error(per_config, out_dir, primary_metric)
            plot_reliability(bin_tables, eces, out_dir, primary_metric)
            plot_quality_traces(per_config, out_dir, primary_metric)
        else:
            logger.warning(
                f"dataset={dataset!r} exposes no per-example score (PRIMARY_METRIC is None); "
                "skipping the entropy-vs-error scatter and reliability diagram, reporting aggregate metrics only."
            )
        plot_accuracy_vs_ratio(summary, out_dir, metric_columns, primary_metric)
        if compute_sequence_KL:
            plot_sampled_distortion(comparison, out_dir)
            # `comparison` is pairwise (full vs ratio r) and carries no cache sizes; the prompt
            # length lives on the full-attention per_config row, so attach it for the color axis.
            comparison = comparison.assign(
                context_length=comparison["task_id"].map(
                    per_config[per_config["kv_caching"] == "full"].set_index("task_id")["cache_seq_length"]
                )
            )
            plot_distortion_traces(
                comparison, out_dir,
                delta_column="IG_seq", KL_column="KL_seq",
                delta_ylabel="IG = Ĥ(Y|x)_comp - Ĥ(Y|x)_full (bits/sequence)",
                KL_ylabel="KL(p_full || p_comp) (bits/sequence)",
            )
        logger.info(f"Saved per_config.csv, comparison.csv, summary and plots to {out_dir}")
        return

    # ---- teacher-forced regime: per-position entropy / KL / cross-entropy ----
    elif teacher_forcing:
        for task in tasks:
            df = process_teacher_forced_task(
                model_, tokenizer, task, ratios, n_sink, max_new_tokens, max_reference_tokens, max_context_length,
                decoding_compression_interval, decoding_target_size,
            )
            if df is not None:
                df.to_csv(records_path, mode="a", header=not records_path.exists(), index=False)

        if not records_path.exists():
            logger.error("No task produced usable records, aborting.")
            return

        records = pd.read_csv(records_path, dtype={"task_id": str})

        build_tf_per_config(records).to_csv(out_dir / "per_config.csv", index=False)
        build_tf_comparison(records).to_csv(out_dir / "comparison.csv", index=False)

        summary = aggregate_teacher_forced_task_metrics(records)
        summary.to_csv(out_dir / "summary.csv", index=False)
        logger.info(f"Dataset-level summary:\n{summary.to_string(index=False)}")

        per_task = records.groupby(["ratio", "task_id"])[METRIC_COLUMNS].mean().reset_index()
        plot_dataset_summary(per_task, summary, out_dir)
        # cache_seq_length_full is the cache after an uncompressed prefill, i.e. the prompt length.
        per_task["context_length"] = per_task["task_id"].map(
            records.groupby("task_id")["cache_seq_length_full"].first()
        )
        plot_distortion_traces(
            per_task, out_dir,
            delta_column="IG", KL_column="KL",
            delta_ylabel="IG = H_comp - H_full (bits/token)",
            KL_ylabel="KL(p_full || p_comp) (bits/token)",
        )
        for ratio in ratios:
            if ratio == 0.0:
                continue
            plot_position_traces(records, ratio, out_dir)
            plot_cache_composition(records, ratio, n_sink, out_dir)
        logger.info(f"Saved per_config.csv, comparison.csv, summary and plots to {out_dir}")


if __name__ == "__main__":
    Fire(main)