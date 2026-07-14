# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Replay a mini-swe-agent trajectory log (produced by kvpress.mini_swe_agent_model.KVPressLocalModel)
through the same full-attention vs. StreamingLLM entropy/KL analysis entropy_analysis.py runs for
static QA tasks -- except now over a live agent's own turns.

For each logged turn, the exact conversation prefix (`messages`) and the text the model actually
generated that turn (`completion`) are taken as a fixed reference sequence, then teacher-forced
under full attention and under StreamingLLM at a sweep of compression ratios -- reusing
entropy_analysis.py's `run_regime`/`compute_metrics`/`aggregate`/plotting functions unchanged.
This gives per-turn and trajectory-level entropy/KL distortion numbers, directly comparable to
the numbers this project already reports for static QA datasets.

Usage
-----
python replay_agent_log.py --log_path ./agent_run_log.jsonl --compression_ratios "[0.0, 0.25, 0.5, 0.75]"
"""

import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from fire import Fire
from transformers import PreTrainedModel, PreTrainedTokenizer

from entropy_analysis import (
    METRIC_COLUMNS,
    aggregate,
    build_stream_press,
    compute_metrics,
    load_model_and_tokenizer,
    plot_cache_composition,
    plot_dataset_summary,
    plot_position_traces,
    run_regime,
)

logger = logging.getLogger(__name__)


def load_log_turns(log_path: str, n_turns: Optional[int]) -> list[dict]:
    records = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    records.sort(key=lambda r: r["turn"])
    if n_turns is not None:
        records = records[:n_turns]
    return records


def build_prompt_ids_from_messages(tokenizer: PreTrainedTokenizer, messages: list[dict], device: str) -> torch.Tensor:
    clean_messages = [{"role": m["role"], "content": m["content"]} for m in messages]
    text = tokenizer.apply_chat_template(clean_messages, add_generation_prompt=True, tokenize=False)
    return tokenizer.encode(text, return_tensors="pt", add_special_tokens=False).to(device)


def process_turn(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    record: dict,
    compression_ratios: list[float],
    n_sink: int,
    max_reference_tokens: int,
    max_context_length: Optional[int],
    decoding_compression_interval: Optional[int],
    decoding_target_size: Optional[int],
) -> Optional[pd.DataFrame]:
    turn_id = str(record["turn"])
    completion = record.get("completion", "")
    if not completion.strip():
        logger.warning(f"turn {turn_id}: empty completion, skipping")
        return None

    prompt_ids = build_prompt_ids_from_messages(tokenizer, record["messages"], model.device)
    if max_context_length is not None and prompt_ids.shape[1] > max_context_length:
        logger.warning(
            f"turn {turn_id}: prompt is {prompt_ids.shape[1]} tokens, exceeds max_context_length="
            f"{max_context_length}, skipping (mid-transcript truncation isn't safe to do automatically)"
        )
        return None

    reference_ids = tokenizer.encode(completion, return_tensors="pt", add_special_tokens=False).to(model.device)
    if reference_ids.shape[1] > max_reference_tokens:
        reference_ids = reference_ids[:, :max_reference_tokens]
    if reference_ids.shape[1] == 0:
        return None

    logger.info(f"turn {turn_id}: prompt_len={prompt_ids.shape[1]} completion_len={reference_ids.shape[1]}")

    full = run_regime(model, prompt_ids, reference_ids, press=None)

    records_out = []
    for ratio in compression_ratios:
        press, decode_per_token = build_stream_press(
            ratio, n_sink, prompt_ids.shape[1], decoding_compression_interval, decoding_target_size
        )
        stream = run_regime(model, prompt_ids, reference_ids, press=press, decode_per_token=decode_per_token)
        df = compute_metrics(full, stream, reference_ids)
        df.insert(0, "ratio", ratio)
        df.insert(0, "task_id", turn_id)
        df["cache_seq_length_full"] = full.cache_seq_length
        df["cache_seq_length_stream"] = stream.cache_seq_length
        records_out.append(df)

    return pd.concat(records_out, ignore_index=True)


def main(
    log_path: str,
    model: Optional[str] = None,
    compression_ratios: str = "[0.0, 0.25, 0.5, 0.75]",
    n_sink: int = 4,
    max_reference_tokens: int = 256,
    max_context_length: Optional[int] = None,
    decoding_compression_interval: Optional[int] = None,
    decoding_target_size: Optional[int] = None,
    n_turns: Optional[int] = None,
    device: Optional[str] = None,
    output_dir: Optional[str] = None,
    seed: int = 42,
):
    """
    Replay a mini-swe-agent trajectory log through the same entropy/KL distortion analysis
    entropy_analysis.py runs for static QA tasks, one row per logged turn.

    `compression_ratios`/`decoding_compression_interval` here are independent of whatever
    compression (if any) was used to originally generate the logged completions -- we're
    asking "how would this compression setting have affected predicting the text that was
    actually produced/acted on," which is well posed regardless of how that text came to be.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    torch.manual_seed(seed)

    ratios = eval(compression_ratios) if isinstance(compression_ratios, str) else list(compression_ratios)
    turns = load_log_turns(log_path, n_turns)
    if not turns:
        logger.error(f"No turns found in {log_path}")
        return
    logger.info(f"Loaded {len(turns)} turn(s) from {log_path}")

    model_name = model or turns[0]["model_name"]
    logger.info(f"Using model: {model_name}")

    mode_suffix = "decode" if decoding_compression_interval is not None else "prefill"
    if output_dir is None:
        output_dir = f"./results/replay_{Path(log_path).stem}"
    out_dir = Path(output_dir)
    out_dir = out_dir.with_name(f"{out_dir.name}_{mode_suffix}")
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Mode: {mode_suffix} -> writing results to {out_dir}")

    records_path = out_dir / "records.csv"
    completed_turn_ids: set[str] = set()
    if records_path.exists():
        completed_turn_ids = set(pd.read_csv(records_path, usecols=["task_id"], dtype={"task_id": str})["task_id"])
        turns = [t for t in turns if str(t["turn"]) not in completed_turn_ids]
        logger.info(
            f"Resuming: {len(completed_turn_ids)} turn(s) already in {records_path.name}, "
            f"{len(turns)} remaining to run"
        )

    model_, tokenizer = load_model_and_tokenizer(model_name, device)

    for record in turns:
        df = process_turn(
            model_, tokenizer, record, ratios, n_sink, max_reference_tokens, max_context_length,
            decoding_compression_interval, decoding_target_size,
        )
        if df is not None:
            df.to_csv(records_path, mode="a", header=not records_path.exists(), index=False)

    if not records_path.exists():
        logger.error("No turn produced usable records, aborting.")
        return

    records = pd.read_csv(records_path, dtype={"task_id": str})
    per_task = records.groupby(["ratio", "task_id"])[METRIC_COLUMNS].mean().reset_index()
    per_task.to_csv(out_dir / "per_task_summary.csv", index=False)

    summary = aggregate(records)
    summary.to_csv(out_dir / "summary.csv", index=False)
    logger.info(f"Trajectory-level summary:\n{summary.to_string(index=False)}")

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
