# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Rebuild every teacher-forced CSV and plot for a finished results directory, from
`records.csv` alone -- no model, no GPU, no re-scoring.

    python evaluation/replot_teacher_forced.py <results_dir> [--fig_format svg] [--tikz]

`--tikz` writes PGFPlots source (`<stem>.tex`) beside each image, for figures that go into a
LaTeX document; the image file is still written either way.

Why this exists: `records.csv` holds the per-position entropy / KL / cross-entropy that cost
the GPU time, and every other teacher-forced artifact is a pure pandas aggregation of it. A run
that predates a new aggregation or a new figure does not need its forward passes repeated --
which matters here because the derived files carry *two* position aggregations (mean over
positions, bits/token; and sum, bits/sequence) that older runs were never written with.

This is the counterpart to `replot_sampled.py`, and differs from it in one important respect:
that script cannot recompute `primary_score` (the benchmark scorers need dataset columns a CSV
round-trip has already mangled), so it treats stored per-example scores as ground truth. The
teacher-forced regime has no benchmark scoring at all -- its quality metric is `excess_ce`,
computed from `records.csv` like everything else -- so this script reproduces the full output
of a real run exactly, with nothing carried over on trust.
"""

import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from entropy_analysis import write_teacher_forced_outputs  # noqa: E402
from entropy_plots import set_format, set_tikz  # noqa: E402

logger = logging.getLogger(__name__)

# Results directories written before the kl -> KL / delta_H -> IG / *_compressed -> *_comp
# rename still carry the old spellings on disk. Renaming on load (rather than migrating the
# CSVs) keeps those runs replottable without rewriting results that are already the record of
# a finished experiment.
LEGACY_COLUMNS = {
    "h_compressed": "h_comp",
    "delta_H": "IG",
    "kl": "KL",
    "ce_compressed": "ce_comp",
    "cache_seq_length_compressed": "cache_seq_length_comp",
}


def load_records(results_dir: Path) -> pd.DataFrame:
    """
    Read the per-position `records.csv` and map any legacy column names.

    The rename is read-time only and is never written back: persisting it would quietly migrate
    a finished experiment's stored results as a side effect of replotting.
    """
    records = pd.read_csv(results_dir / "records.csv", dtype={"task_id": str})
    renamed = [c for c in records.columns if c in LEGACY_COLUMNS]
    if renamed:
        logger.info(f"mapped legacy column name(s) {renamed} to their current spellings")
        records = records.rename(columns=LEGACY_COLUMNS)

    # Unlike the sampled regime's append-per-task-per-config file, a duplicated (task, ratio,
    # position) row here would silently double-weight that task in every aggregate.
    keys = ["task_id", "ratio", "position"]
    n_before = len(records)
    records = records.drop_duplicates(subset=keys, keep="last")
    if len(records) < n_before:
        logger.warning(
            f"dropped {n_before - len(records)} duplicate {tuple(keys)} row(s) -- two runs had "
            "appended to this directory, double-weighting those tasks"
        )
    return records


def main(results_dir: str, n_sink: int = 4, fig_format: str = "png", tikz: bool = False) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    # Before anything is read, so an unsupported format (or a missing tikzplotlib) fails
    # immediately rather than after the CSVs have been rewritten.
    set_format(fig_format)
    set_tikz(tikz)
    out_dir = Path(results_dir)

    records = load_records(out_dir)
    # The ratios actually present, rather than a CLI default: a resumed or partial run may hold
    # fewer than the sweep asked for, and `write_teacher_forced_outputs` only uses these to
    # decide which per-ratio figures to draw.
    ratios = sorted(records["ratio"].unique())
    logger.info(f"{records['task_id'].nunique()} task(s), ratios {ratios}")

    write_teacher_forced_outputs(records, out_dir, ratios, n_sink)
    written = f".{fig_format}" + (" and .tex" if tikz else "")
    logger.info(f"Rewrote per_config.csv, comparison.csv, summary.csv and all {written} plots in {out_dir}")


if __name__ == "__main__":
    from fire import Fire

    Fire(main)
