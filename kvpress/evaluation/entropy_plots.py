# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
All plotting for `entropy_analysis.py`. Every function takes already-computed
DataFrames / dicts plus an output directory and writes a PNG (dpi=150); none of
them call back into the metric math or the inference/orchestration code, so this
module is a pure sink in the import graph.
"""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_dataset_summary(per_task: pd.DataFrame, summary: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].errorbar(
        summary["ratio"], summary["delta_I_loss_bits"], yerr=summary["delta_h_std"], marker="o", capsize=3
    )
    axes[0].axhline(0, color="black", linewidth=0.5)
    axes[0].set_xlabel("StreamingLLM compression ratio")
    axes[0].set_ylabel("ΔI_loss = H(O|M_compressed) - H(O|M_full) (bits/token)")
    axes[0].set_title("Information lost to eviction (mean ± std over tasks)")

    axes[1].errorbar(
        summary["ratio"], summary["mean_kl_bits"], yerr=summary["kl_full_compressed_std"], marker="o", capsize=3
    )
    axes[1].set_xlabel("StreamingLLM compression ratio")
    axes[1].set_ylabel("KL(p_full || p_compressed) (bits/token)")
    axes[1].set_title("Distribution shift (mean ± std over tasks)")

    fig.tight_layout()
    fig.savefig(output_dir / "distortion_vs_compression_ratio.png", dpi=150)
    plt.close(fig)

    ratios = sorted(per_task["ratio"].unique())
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot(
        [per_task.loc[per_task["ratio"] == r, "kl_full_compressed"] for r in ratios],
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
        axes[1].plot(group["position"], group["kl_full_compressed"], alpha=0.5, linewidth=1)

    axes[0].axhline(0, color="black", linewidth=0.5)
    axes[0].set_ylabel("ΔH = H_compressed - H_full (bits)")
    axes[1].set_ylabel("KL(p_full || p_compressed) (bits)")
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
        n_kept = row.cache_seq_length_compressed
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


def plot_entropy_vs_error(
    records: pd.DataFrame, output_dir: Path, primary_metric: str, threshold: float
) -> None:
    """
    Sequence entropy vs error, one panel per kv_caching config, one point per task.

    The generation analog of the uncertainty-error scatter in Meronen et al. (WACV 2024)
    Fig. 2/4: points in the upper *left* are the bad ones -- the model got the answer
    wrong while reporting low entropy (confidently wrong). Upper right means it was wrong
    but knew it was uncertain, which is the well-behaved failure mode.
    """
    labels = list(dict.fromkeys(records["config_label"]))
    n_cols = min(3, len(labels))
    n_rows = math.ceil(len(labels) / n_cols)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4.3 * n_cols, 3.9 * n_rows), squeeze=False, sharex=True, sharey=True
    )

    for i, label in enumerate(labels):
        ax = axes[i // n_cols][i % n_cols]
        group = records[records["config_label"] == label]
        is_correct = group["correct"].astype(bool)
        ax.scatter(
            group.loc[is_correct, "H_bits"], group.loc[is_correct, "error"],
            s=20, alpha=0.6, color="tab:blue", label="correct",
        )
        ax.scatter(
            group.loc[~is_correct, "H_bits"], group.loc[~is_correct, "error"],
            s=20, alpha=0.6, color="tab:orange", label="incorrect",
        )
        ax.set_title(f"{label}  (accuracy={is_correct.mean():.2f})", fontsize=9)
        ax.set_xlabel("sequence entropy Ĥ(Y|x) (bits)")
        ax.set_ylabel(f"error = 1 - {primary_metric}")
    for i in range(len(labels), n_rows * n_cols):
        axes[i // n_cols][i % n_cols].axis("off")

    axes[0][0].legend(loc="lower right", fontsize=8)
    fig.suptitle(
        f"Sequence entropy vs error  (correct = {primary_metric} >= {threshold}, "
        f"n={records['task_id'].nunique()} tasks)\n"
        "upper-left = confidently wrong; upper-right = wrong but appropriately uncertain",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(output_dir / "entropy_vs_error.png", dpi=150)
    plt.close(fig)


def plot_reliability(
    bin_tables: dict[str, pd.DataFrame], eces: dict[str, float], output_dir: Path, primary_metric: str
) -> None:
    """Reliability diagram: one curve per kv_caching config, with its ECE in the legend."""
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="perfect calibration")

    for label, table in bin_tables.items():
        if table.empty:
            continue
        ax.plot(
            table["mean_confidence"], table["accuracy"], marker="o",
            label=f"{label} (ECE={eces[label]:.3f})",
        )

    ax.set_xlabel("confidence = exp(-Ĥ / mean_length)  (per-token geometric-mean probability)")
    ax.set_ylabel(f"accuracy (fraction with {primary_metric} >= threshold)")
    ax.set_title("Calibration of sequence entropy against answer quality")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "calibration.png", dpi=150)
    plt.close(fig)


def plot_accuracy_vs_ratio(summary: pd.DataFrame, output_dir: Path, metric_columns: list[str], primary_metric: str) -> None:
    """
    Benchmark accuracy metrics and entropy against compression ratio, with the
    full-attention config drawn as a horizontal baseline (it has no ratio of its own).

    `metric_columns` are the benchmark-metric names present in `summary` (each has a
    `{col}_mean` column); `primary_metric` is the one used for the full-attention
    baseline line. These are dataset-dependent (the benchmark's own metrics), so they
    are passed in rather than hardcoded.
    """
    stream = summary[summary["kv_caching"] == "compressed"].sort_values("ratio")
    full = summary[summary["kv_caching"] == "full"]
    if stream.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for column in metric_columns:
        mean_col = f"{column}_mean"
        if mean_col in stream and stream[mean_col].notna().any():
            axes[0].plot(stream["ratio"], stream[mean_col], marker="o", label=column)
    primary_mean = f"{primary_metric}_mean"
    if not full.empty and primary_mean in full and full[primary_mean].notna().any():
        axes[0].axhline(
            full[primary_mean].iloc[0], color="gray", linestyle="--", linewidth=1,
            label=f"{primary_metric} (full attention)",
        )
    axes[0].set_xlabel("StreamingLLM compression ratio")
    axes[0].set_ylabel("mean score over tasks")
    axes[0].set_title("Answer quality vs compression")
    axes[0].legend(fontsize=8)

    axes[1].errorbar(
        stream["ratio"], stream["H_bits_mean"], yerr=stream["H_bits_std"], marker="o", capsize=3,
        label="Ĥ(Y|x) compressed",
    )
    if not full.empty:
        axes[1].axhline(
            full["H_bits_mean"].iloc[0], color="gray", linestyle="--", linewidth=1, label="full attention",
        )
    axes[1].set_xlabel("StreamingLLM compression ratio")
    axes[1].set_ylabel("Ĥ(Y|x) (bits, sequence-level)")
    axes[1].set_title("Uncertainty vs compression (mean ± std over tasks)")
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_dir / "accuracy_vs_compression_ratio.png", dpi=150)
    plt.close(fig)
