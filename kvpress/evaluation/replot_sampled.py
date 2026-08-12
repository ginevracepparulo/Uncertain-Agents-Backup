# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Rebuild every sampled-regime plot (and the derived CSVs) for a finished results directory,
from `per_config.csv` alone -- no model, no GPU, no re-sampling.

Why this exists: the plots are the cheap tail of a run whose expensive part is already on
disk. A run that crashed *after* writing `per_config.csv` (or one that predates a plot you
have since added) does not need 30 GPU-minutes repeated to get its figures.

    python evaluation/replot_sampled.py <results_dir> [--fig_format svg] [--tikz]

`--tikz` writes PGFPlots source (`<stem>.tex`) beside each image, for figures that go into a
LaTeX document; the image file is still written either way.

What it reuses and what it recomputes
-------------------------------------
`primary_score` is read from the CSV, never recomputed. It cannot be recomputed here: the
benchmark scorers need the dataset's own raw columns, and a CSV round-trip has already turned
list-valued ones (LongBench `answers`) into their string repr -- re-scoring a trec run this way
silently drops it from 0.66 to 0.22. `entropy_analysis.main` avoids that by re-merging the raw
rows from memory; this script has no dataset loaded, so it treats the stored per-example scores
as ground truth and derives accuracy, ECE and every plot from them.

The dataset-level benchmark metrics (`{metric}_mean` from each scorer) are for the same reason
not reproduced; the accuracy panel is drawn from `primary_score` instead, which is the number
those metrics reduce to anyway for the calibration analysis.
"""

import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from entropy_analysis import build_sampled_comparison  # noqa: E402
from entropy_metrics import compute_ece  # noqa: E402
from entropy_plots import (  # noqa: E402
    plot_accuracy_vs_ratio,
    plot_distortion_traces,
    plot_entropy_vs_error,
    plot_quality_traces,
    plot_reliability,
    plot_sampled_distortion,
    set_format,
    set_tikz,
)

logger = logging.getLogger(__name__)

SUMMARY_COLUMNS = ["primary_score", "H_seq", "confidence", "greedy_length", "KL_seq"]

# Results directories written before the kl -> KL / delta_H -> IG rename still carry the old
# spellings on disk. Renaming on load (rather than migrating the CSVs) keeps those runs
# replottable without rewriting results that are already the record of a finished experiment.
LEGACY_COLUMNS = {
    "kl": "KL",
    "kl_seq": "KL_seq",
    "kl_seq_se": "KL_seq_se",
    "delta_H": "IG",
    "delta_H_seq": "IG_seq",
    "h_compressed": "h_comp",
    "ce_compressed": "ce_comp",
}


def load_per_config(results_dir: Path) -> tuple[pd.DataFrame, bool]:
    """
    Read per_config.csv, map any legacy column names, and drop the duplicate rows a
    concurrently-written run leaves behind.

    Returns the frame and whether any duplicates were dropped -- the caller writes the file
    back only in that case. The legacy rename is a read-time convenience, so persisting it
    would quietly migrate a finished experiment's stored results as a side effect of replotting.
    """
    per_config = pd.read_csv(
        results_dir / "per_config.csv",
        dtype={"task_id": str, "predicted_answer": str, "gold": str},
        keep_default_na=False,
        na_values=[""],
    )
    renamed = [c for c in per_config.columns if c in LEGACY_COLUMNS]
    if renamed:
        logger.info(f"mapped legacy column name(s) {renamed} to their current spellings")
        per_config = per_config.rename(columns=LEGACY_COLUMNS)

    n_before = len(per_config)
    per_config = per_config.drop_duplicates(subset=["task_id", "config_label"], keep="last")
    deduped = len(per_config) < n_before
    if deduped:
        logger.warning(
            f"dropped {n_before - len(per_config)} duplicate (task_id, config_label) row(s) "
            "-- two runs had appended to this directory, double-weighting those tasks"
        )
    return per_config, deduped


def summarize(per_config: pd.DataFrame, n_ece_bins: int, ece_bin_strategy: str):
    """
    One summary row per config, mirroring `entropy_analysis.aggregate_sampled` minus the
    benchmark-scorer call (see the module docstring for why that one cannot be replayed).
    """
    rows, bin_tables, eces = [], {}, {}
    for label, group in per_config.groupby("config_label", sort=False):
        ece, bin_table = compute_ece(group["confidence"], group["primary_score"], n_ece_bins, ece_bin_strategy)
        bin_tables[label], eces[label] = bin_table, ece

        row = {
            "regime": "sampled",
            "config_label": label,
            "kv_caching": group["kv_caching"].iloc[0],
            "ratio": group["ratio"].iloc[0],
            "n_tasks": len(group),
            "accuracy": group["primary_score"].mean(),
            "ece": ece,
        }
        for column in SUMMARY_COLUMNS:
            if column in group:
                row[f"{column}_mean"] = group[column].mean()
                row[f"{column}_std"] = group[column].std()
        rows.append(row)
    return pd.DataFrame(rows), bin_tables, eces


def main(
    results_dir: str,
    n_ece_bins: int = 5,
    ece_bin_strategy: str = "quantile",
    fig_format: str = "png",
    tikz: bool = False,
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    # Before anything is read, so an unsupported format (or a missing tikzplotlib) fails
    # immediately rather than after the CSVs have been rewritten.
    set_format(fig_format)
    set_tikz(tikz)
    out_dir = Path(results_dir)

    per_config, deduped = load_per_config(out_dir)
    if deduped:
        per_config.to_csv(out_dir / "per_config.csv", index=False)

    comparison = build_sampled_comparison(per_config)
    comparison.to_csv(out_dir / "comparison.csv", index=False)

    summary, bin_tables, eces = summarize(per_config, n_ece_bins, ece_bin_strategy)
    summary.to_csv(out_dir / "summary.csv", index=False)
    logger.info(
        "Sampled summary (accuracy from the stored per-example primary_score):\n"
        f"{summary[['config_label', 'n_tasks', 'accuracy', 'ece', 'H_seq_mean', 'confidence_mean']].to_string(index=False)}"
    )
    if "KL_seq" in comparison and comparison["KL_seq"].notna().any():
        logger.info(
            "Distortion vs full attention (bits/sequence):\n"
            f"{comparison.groupby('ratio')[['IG_seq', 'KL_seq', 'KL_seq_se']].agg(['mean', 'std']).round(3).to_string()}"
        )

    plot_entropy_vs_error(per_config, out_dir, "primary_score")
    plot_reliability(bin_tables, eces, out_dir, "primary_score")
    plot_quality_traces(per_config, out_dir, "primary_score")
    plot_accuracy_vs_ratio(summary, out_dir, ["primary_score"], "primary_score")
    plot_sampled_distortion(comparison, out_dir)
    plot_distortion_traces(
        comparison, out_dir,
        delta_column="IG_seq", KL_column="KL_seq",
        delta_ylabel="I(Y;X_res) = Ĥ(Y|X_comp) - Ĥ(Y|X_full) (bits/sequence)",
        KL_ylabel="KL(p_full || p_compressed) (bits/sequence)",
    )
    written = f".{fig_format}" + (" and .tex" if tikz else "")
    logger.info(f"Rewrote comparison.csv, summary.csv and all {written} plots in {out_dir}")


if __name__ == "__main__":
    from fire import Fire

    Fire(main)
