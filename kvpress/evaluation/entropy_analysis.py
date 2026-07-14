# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Post-hoc predictive-entropy / KL-divergence analysis comparing full attention
against StreamingLLM KV cache eviction, across a dataset of tasks.

Both regimes are teacher-forced on the *same* reference continuation (a gold
answer if the dataset provides one, otherwise greedily generated under full
attention), so the per-position output distributions are directly comparable:

    p_full(.  | prefix_t)    vs    p_stream(.  | prefix_t)

Per (task, position) we compute predictive entropy H_full / H_stream (bits),
delta_H = H_stream - H_full, KL(p_full || p_stream), and the cross-entropy of
the reference token under each regime. Pooling these over many tasks gives a
dataset-level estimate of the (chain-rule/teacher-forced) conditional entropy
H(O | M) for each cache regime M, and the "information lost to eviction":

    delta_I = H(O | M_stream) - H(O | M_full) >= 0

This is *not* a full mutual information I(O; M) in the strict sense (that
would require a joint distribution over O and a random cache M, plus an
unconditioned H(O) baseline we never estimate). What is estimated here is the
standard, tractable proxy: how many extra bits, on average, the model needs to
predict the true continuation once the cache has been degraded from full to
StreamingLLM. Equivalently, excess_ce = mean(ce_stream) - mean(ce_full) is an
(approximate) estimator of the same quantity via CE(p_full, p_stream) =
H(p_full) + KL(p_full || p_stream).

Usage
-----
# Single ad hoc task
python entropy_analysis.py --context "..." --question "..." --answer "..."

# Loop over a real dataset (default: simonjegou/loogle, shortdep_qa split)
python entropy_analysis.py \
    --dataset simonjegou/loogle --dataset_config shortdep_qa \
    --n_samples 20 --compression_ratios "[0.0, 0.25, 0.5, 0.75]"

# Loop over a local JSONL file with "context"/"question"/"answer" fields
python entropy_analysis.py --dataset_path ./my_tasks.jsonl
"""

import itertools
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import torch
from fire import Fire
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache, PreTrainedModel, PreTrainedTokenizer

from kvpress import DecodingPress, PrefillDecodingPress, StreamingLLMPress
from kvpress.presses.base_press import BasePress

logger = logging.getLogger(__name__)

LOG2 = torch.log(torch.tensor(2.0)).item()

METRIC_COLUMNS = ["h_full", "h_stream", "delta_h", "kl_full_stream", "ce_full", "ce_stream"]


@dataclass
class Task:
    task_id: str
    context: str
    question: str
    answer: Optional[str] = None


@dataclass
class RegimeOutput:
    """Per-position teacher-forced quantities for one attention regime."""

    log_probs: torch.Tensor  # (ref_len, vocab), natural-log probabilities, float32
    cache_seq_length: int  # number of KV entries retained after prefill


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
    dataset_config: Optional[str],
    dataset_split: str,
    dataset_path: Optional[str],
    context_field: str,
    question_field: str,
    answer_field: str,
    n_samples: int,
) -> list[Task]:
    """
    Load a list of tasks either from a local JSONL/CSV file (`dataset_path`) or
    from a Hugging Face Hub dataset (`dataset`, streamed so only `n_samples`
    examples are downloaded).
    """
    if dataset_path is not None:
        path = Path(dataset_path)
        if path.suffix == ".jsonl":
            with open(path) as f:
                rows = [json.loads(line) for line in itertools.islice(f, n_samples)]
        else:
            rows = pd.read_csv(path).head(n_samples).to_dict("records")
    else:
        from datasets import load_dataset

        ds = load_dataset(dataset, dataset_config, split=dataset_split, streaming=True)
        rows = list(itertools.islice(ds, n_samples))

    tasks = []
    for i, row in enumerate(rows):
        answer = row.get(answer_field)
        if isinstance(answer, list):
            answer = answer[0] if answer else None
        tasks.append(
            Task(task_id=str(i), context=row[context_field], question=row[question_field], answer=answer)
        )
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
def run_regime(
    model: PreTrainedModel,
    prompt_ids: torch.Tensor,
    reference_ids: torch.Tensor,
    press: Optional[BasePress],
    decode_per_token: bool = False,
) -> RegimeOutput:
    """
    Prefill `prompt_ids` (optionally compressing the KV cache with `press`), then
    teacher-force `reference_ids` and return per-position log-probabilities.

    Position ids for the teacher-forced tokens continue from `prompt_ids.shape[1]`
    (the *original*, uncompressed prompt length), not from the compressed cache
    length: pruned tokens keep the RoPE position they were encoded with, so the
    query positions must be numbered as if no eviction had happened.

    If `decode_per_token` is False (the default), the reference is scored in one
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
        model.model(input_ids=prompt_ids, past_key_values=cache)

    if press is not None and decode_per_token:
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

    return RegimeOutput(log_probs=log_probs, cache_seq_length=cache_seq_length)


def entropy_bits(log_probs: torch.Tensor) -> torch.Tensor:
    """Shannon entropy per row, in bits. log_probs: (seq_len, vocab) natural log."""
    probs = log_probs.exp()
    return -(probs * log_probs).sum(dim=-1) / LOG2


def kl_bits(log_p: torch.Tensor, log_q: torch.Tensor) -> torch.Tensor:
    """KL(p || q) per row, in bits."""
    p = log_p.exp()
    return (p * (log_p - log_q)).sum(dim=-1) / LOG2


def cross_entropy_bits(log_probs: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    """-log p(token) per position, in bits."""
    return -log_probs.gather(1, token_ids.unsqueeze(1)).squeeze(1) / LOG2


def compute_metrics(full: RegimeOutput, stream: RegimeOutput, reference_ids: torch.Tensor) -> pd.DataFrame:
    ref = reference_ids[0].to(full.log_probs.device)
    h_full = entropy_bits(full.log_probs)
    h_stream = entropy_bits(stream.log_probs)
    return pd.DataFrame(
        {
            "position": range(len(ref)),
            "h_full": h_full.tolist(),
            "h_stream": h_stream.tolist(),
            "delta_h": (h_stream - h_full).tolist(),
            "kl_full_stream": kl_bits(full.log_probs, stream.log_probs).tolist(),
            "ce_full": cross_entropy_bits(full.log_probs, ref).tolist(),
            "ce_stream": cross_entropy_bits(stream.log_probs, ref).tolist(),
        }
    )


def build_stream_press(
    ratio: float, n_sink: int, prompt_length: int, decoding_compression_interval: Optional[int],
    decoding_target_size: Optional[int],
) -> tuple[BasePress, bool]:
    """
    Build the press used for the "stream" regime, and whether it needs the
    per-token decode loop in `run_regime` to actually exercise decode-time eviction.

    If `decoding_compression_interval` is None, this is the original prefill-only
    behaviour: a bare StreamingLLMPress, pruned once during prefill and never again.

    If `decoding_compression_interval` is set, StreamingLLMPress is also wrapped in
    a `DecodingPress` (via `PrefillDecodingPress`), which re-applies StreamingLLM's
    sink+recency scoring rule every `decoding_compression_interval` decode steps,
    pruning the cache back down to `decoding_target_size` tokens each time.

    ratio=0.0 always disables eviction entirely, at both prefill and decode, even
    if decode-time compression is requested: `target_size = prompt_length` would
    otherwise still let the cache get trimmed once it grows past the *original*
    prompt length during decoding -- a real, if small, eviction that would break
    the "ratio=0.0 must exactly reproduce full attention" invariant.
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


def process_task(
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

    full = run_regime(model, prompt_ids, reference_ids, press=None)

    records = []
    for ratio in compression_ratios:
        press, decode_per_token = build_stream_press(
            ratio, n_sink, prompt_ids.shape[1], decoding_compression_interval, decoding_target_size
        )
        stream = run_regime(model, prompt_ids, reference_ids, press=press, decode_per_token=decode_per_token)
        df = compute_metrics(full, stream, reference_ids)
        df.insert(0, "ratio", ratio)
        df.insert(0, "task_id", task.task_id)
        df["cache_seq_length_full"] = full.cache_seq_length
        df["cache_seq_length_stream"] = stream.cache_seq_length
        records.append(df)

    return pd.concat(records, ignore_index=True)


def aggregate(records: pd.DataFrame) -> pd.DataFrame:
    """
    Dataset-level H(O | M) estimate per compression ratio.

    Each task is first averaged over its own positions (giving one bits/token
    number per task), then averaged across tasks, so long and short reference
    sequences count equally as one sample each.
    """
    per_task = records.groupby(["ratio", "task_id"])[METRIC_COLUMNS].mean().reset_index()
    summary = per_task.groupby("ratio")[METRIC_COLUMNS].agg(["mean", "std"])
    summary.columns = ["_".join(c) for c in summary.columns]
    summary = summary.reset_index()
    summary["excess_ce_bits"] = summary["ce_stream_mean"] - summary["ce_full_mean"]
    summary["n_tasks"] = per_task.groupby("ratio").size().values
    return summary.rename(
        columns={
            "h_full_mean": "H_full_bits_per_token",
            "h_stream_mean": "H_stream_bits_per_token",
            "delta_h_mean": "delta_I_loss_bits",
            "kl_full_stream_mean": "mean_kl_bits",
        }
    )


def plot_dataset_summary(per_task: pd.DataFrame, summary: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].errorbar(
        summary["ratio"], summary["delta_I_loss_bits"], yerr=summary["delta_h_std"], marker="o", capsize=3
    )
    axes[0].axhline(0, color="black", linewidth=0.5)
    axes[0].set_xlabel("StreamingLLM compression ratio")
    axes[0].set_ylabel("ΔI_loss = H(O|M_stream) - H(O|M_full) (bits/token)")
    axes[0].set_title("Information lost to eviction (mean ± std over tasks)")

    axes[1].errorbar(
        summary["ratio"], summary["mean_kl_bits"], yerr=summary["kl_full_stream_std"], marker="o", capsize=3
    )
    axes[1].set_xlabel("StreamingLLM compression ratio")
    axes[1].set_ylabel("KL(p_full || p_stream) (bits/token)")
    axes[1].set_title("Distribution shift (mean ± std over tasks)")

    fig.tight_layout()
    fig.savefig(output_dir / "distortion_vs_compression_ratio.png", dpi=150)
    plt.close(fig)

    ratios = sorted(per_task["ratio"].unique())
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot(
        [per_task.loc[per_task["ratio"] == r, "kl_full_stream"] for r in ratios],
        tick_labels=[str(r) for r in ratios],
    )
    ax.set_xlabel("StreamingLLM compression ratio")
    ax.set_ylabel("per-task mean KL (bits/token)")
    ax.set_title("Spread of distortion across tasks")
    fig.tight_layout()
    fig.savefig(output_dir / "kl_spread_across_tasks.png", dpi=150)
    plt.close(fig)


def plot_position_traces(records: pd.DataFrame, ratio: float, output_dir: Path) -> None:
    """
    ΔH_t and KL_t against position t in the reference sequence, one line per task.

    Note: with KVPress, StreamingLLMPress only prunes the cache once during
    prefill (compression is skipped once generation starts, see BasePress).
    So the sink/eviction boundary is fixed *inside the context* before any of
    these positions are reached -- it does not move along this t axis. See
    `plot_cache_composition` for where the boundary actually falls.
    """
    subset = records[records["ratio"] == ratio]
    if subset.empty:
        return

    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    for _, group in subset.groupby("task_id"):
        group = group.sort_values("position")
        axes[0].plot(group["position"], group["delta_h"], alpha=0.5, linewidth=1)
        axes[1].plot(group["position"], group["kl_full_stream"], alpha=0.5, linewidth=1)

    axes[0].axhline(0, color="black", linewidth=0.5)
    axes[0].set_ylabel("ΔH = H_stream - H_full (bits)")
    axes[1].set_ylabel("KL(p_full || p_stream) (bits)")
    axes[1].set_xlabel("position in reference sequence (t)")
    n_tasks = subset["task_id"].nunique()
    fig.suptitle(
        f"compression_ratio={ratio}  (n={n_tasks} tasks, one line per task)\n"
        "cache is pruned once during prefill and fixed thereafter -- see cache_composition plot for the boundary",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(output_dir / f"per_position_traces_ratio_{ratio}.png", dpi=150)
    plt.close(fig)


def plot_cache_composition(records: pd.DataFrame, ratio: float, n_sink: int, output_dir: Path, max_tasks: int = 40) -> None:
    """
    Where the StreamingLLM sink/eviction boundary falls inside the *original context*
    for each task: sink tokens (always kept), the evicted middle span, and the
    recent window (kept). This is the boundary "position" -- it lives on the
    context axis, not on the reference-sequence axis used by `plot_position_traces`.
    """
    subset = records[records["ratio"] == ratio].drop_duplicates("task_id").sort_values("task_id").head(max_tasks)
    if subset.empty:
        return

    fig, ax = plt.subplots(figsize=(9, 0.3 * len(subset) + 1.5))
    for i, row in enumerate(subset.itertuples()):
        prompt_length = row.cache_seq_length_full
        n_kept = row.cache_seq_length_stream
        n_pruned = prompt_length - n_kept
        sink_end = min(n_sink, prompt_length)
        evicted_end = sink_end + n_pruned

        ax.barh(i, sink_end, color="tab:green", label="sink (kept)" if i == 0 else None)
        ax.barh(i, evicted_end - sink_end, left=sink_end, color="lightgray", label="evicted" if i == 0 else None)
        ax.barh(
            i, prompt_length - evicted_end, left=evicted_end, color="tab:blue",
            label="recent window (kept)" if i == 0 else None,
        )

    ax.set_yticks(range(len(subset)))
    ax.set_yticklabels(subset["task_id"])
    ax.set_xlabel("position in original context")
    ax.set_ylabel("task_id")
    ax.set_title(f"StreamingLLM cache composition at compression_ratio={ratio} (n_sink={n_sink})")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_dir / f"cache_composition_ratio_{ratio}.png", dpi=150)
    plt.close(fig)


def main(
    model: str = "Qwen/Qwen2.5-0.5B-Instruct",
    context: Optional[str] = None,
    question: Optional[str] = None,
    answer: Optional[str] = None,
    dataset: Optional[str] = "simonjegou/loogle",
    dataset_config: Optional[str] = "shortdep_qa",
    dataset_split: str = "test",
    dataset_path: Optional[str] = None,
    context_field: str = "context",
    question_field: str = "question",
    answer_field: str = "answer",
    n_samples: int = 5,
    compression_ratios: str = "[0.0, 0.25, 0.5, 0.75]",
    n_sink: int = 4,
    max_new_tokens: int = 64,
    max_reference_tokens: int = 48,
    max_context_length: Optional[int] = 1024,
    decoding_compression_interval: Optional[int] = None,
    decoding_target_size: Optional[int] = None,
    device: Optional[str] = None,
    output_dir: str = "./results/entropy_analysis",
    seed: int = 42,
):
    """
    Compare per-position predictive entropy and KL divergence between full
    attention and StreamingLLM eviction, teacher-forced on a shared reference
    continuation, across a sweep of compression ratios and a dataset of tasks.

    Pass `--context`/`--question` (and optionally `--answer`) to run a single
    ad hoc task instead of a dataset.

    By default StreamingLLM only prunes once, during prefill (KVPress's normal
    behaviour). Set `decoding_compression_interval` to also re-apply StreamingLLM's
    sink+recency rule every N decode steps (pruning back down to
    `decoding_target_size` tokens, or `(1 - ratio) * prompt_length` if left unset).
    This switches the reference to being scored one token at a time instead of in
    one batched call, so decode-time eviction can actually affect later positions.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    torch.manual_seed(seed)

    ratios = eval(compression_ratios) if isinstance(compression_ratios, str) else list(compression_ratios)
    mode_suffix = "decode" if decoding_compression_interval is not None else "prefill"
    out_dir = Path(output_dir)
    out_dir = out_dir.with_name(f"{out_dir.name}_{mode_suffix}")
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Writing results to {out_dir}")

    if context is not None and question is not None:
        tasks = [Task(task_id="0", context=context, question=question, answer=answer)]
    else:
        tasks = load_tasks(
            dataset, dataset_config, dataset_split, dataset_path, context_field, question_field, answer_field,
            n_samples,
        )
    logger.info(f"Loaded {len(tasks)} task(s)")

    records_path = out_dir / "records.csv"
    completed_task_ids: set[str] = set()
    if records_path.exists():
        completed_task_ids = set(pd.read_csv(records_path, usecols=["task_id"], dtype={"task_id": str})["task_id"])
        tasks = [task for task in tasks if task.task_id not in completed_task_ids]
        logger.info(
            f"Resuming: {len(completed_task_ids)} task(s) already in {records_path.name}, "
            f"{len(tasks)} remaining to run"
        )

    model_, tokenizer = load_model_and_tokenizer(model, device)

    for task in tasks:
        df = process_task(
            model_, tokenizer, task, ratios, n_sink, max_new_tokens, max_reference_tokens, max_context_length,
            decoding_compression_interval, decoding_target_size,
        )
        if df is not None:
            df.to_csv(records_path, mode="a", header=not records_path.exists(), index=False)

    if not records_path.exists():
        logger.error("No task produced usable records, aborting.")
        return

    records = pd.read_csv(records_path, dtype={"task_id": str})

    per_task = records.groupby(["ratio", "task_id"])[METRIC_COLUMNS].mean().reset_index()
    per_task.to_csv(out_dir / "per_task_summary.csv", index=False)

    summary = aggregate(records)
    summary.to_csv(out_dir / "summary.csv", index=False)
    logger.info(f"Dataset-level summary:\n{summary.to_string(index=False)}")

    zero_ratio_summary = summary[summary["ratio"] == 0.0]
    if not zero_ratio_summary.empty:
        max_kl = zero_ratio_summary["mean_kl_bits"].iloc[0]
        if max_kl > 1e-3:
            logger.warning(
                f"Sanity check failed: compression_ratio=0.0 should reproduce full attention "
                f"exactly, but mean KL = {max_kl:.6f} bits. Investigate before trusting other ratios."
            )
        else:
            logger.info(f"Sanity check passed: compression_ratio=0.0 matches full attention (mean KL={max_kl:.2e} bits).")

    plot_dataset_summary(per_task, summary, out_dir)
    for ratio in ratios:
        if ratio == 0.0:
            continue
        plot_position_traces(records, ratio, out_dir)
        plot_cache_composition(records, ratio, n_sink, out_dir)
    logger.info(f"Saved records, summary and plots to {out_dir}")


if __name__ == "__main__":
    Fire(main)
