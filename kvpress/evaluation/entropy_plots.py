# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
All plotting for `entropy_analysis.py`. Every function takes already-computed
DataFrames / dicts plus an output directory and writes one figure file; none of
them call back into the metric math or the inference/orchestration code, so this
module is a pure sink in the import graph.

Every figure is drawn at a single target size (`FIGSIZE`, 30.55 cm x 21.44 cm)
with 18 pt type throughout, so the files drop into a document at 1:1 scale
without per-figure rescaling -- which is what would otherwise make the nominal
18 pt render at a different size in each figure.

The file format is `DEFAULT_FORMAT` unless a caller selects another one through
`set_format`; the size and type above are identical either way.

`set_tikz` additionally writes each figure as PGFPlots source (`.tex`) next to
the image file. That output is deliberately *not* held to the sizing contract
above: a `.tex` figure takes its fonts from the surrounding document and its
width from `\\textwidth` at compile time, which is the reason to emit it at all.
"""

import math
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize


# Matplotlib works in inches; the target page box is specified in centimetres.
CM_PER_INCH = 2.54
FIG_WIDTH_IN = 30.55 / CM_PER_INCH
FIG_HEIGHT_IN = 21.44 / CM_PER_INCH
FIGSIZE = (FIG_WIDTH_IN, FIG_HEIGHT_IN)

FONT_SIZE = 18

# How many categorical y labels fit down FIG_HEIGHT_IN at FONT_SIZE without colliding.
MAX_LABELLED_ROWS = 25

SUPPORTED_FORMATS = ("png", "svg", "pdf")
DEFAULT_FORMAT = "png"
PNG_DPI = 200  # ~2400 px across a 30.55 cm figure; the vector formats ignore it

FIG_FORMAT = DEFAULT_FORMAT

EMIT_TIKZ = False

# Applied at import, so every figure this module draws inherits it. Set on all text
# elements rather than `font.size` alone: the tick/legend/title sizes default to
# multipliers of the base ("medium", "large"), which would leave labels at other sizes.
plt.rcParams.update(
    {
        "font.size": FONT_SIZE,
        "axes.titlesize": FONT_SIZE,
        "axes.labelsize": FONT_SIZE,
        "xtick.labelsize": FONT_SIZE,
        "ytick.labelsize": FONT_SIZE,
        "legend.fontsize": FONT_SIZE,
        "figure.titlesize": FONT_SIZE,
        "figure.figsize": FIGSIZE,
        "savefig.format": DEFAULT_FORMAT,
        "savefig.dpi": PNG_DPI,
    }
)


def set_format(fig_format: str) -> None:
    """
    Choose the file format every figure in this module is written in. Call once from a
    script's entry point, before any plotting.

    Module-level rather than an argument on each plotting function: it is one choice for a
    whole run, and threading it through the nine public functions would put an argument on
    every call site to repeat the same value. It also matches how this module already
    handles style -- the rcParams block above is global too.
    """
    global FIG_FORMAT

    fig_format = fig_format.lower().lstrip(".")
    if fig_format not in SUPPORTED_FORMATS:
        raise ValueError(f"fig_format must be one of {SUPPORTED_FORMATS}, got {fig_format!r}")
    FIG_FORMAT = fig_format
    plt.rcParams["savefig.format"] = fig_format


def set_tikz(enabled: bool) -> None:
    """
    Also write every figure as PGFPlots source (`<stem>.tex`) beside the image file.

    Module-level and set once from a script's entry point, for the same reason as `set_format`.
    It is a toggle rather than a fourth `SUPPORTED_FORMATS` value because the two outputs are
    complementary: the `.tex` is what goes into the document, and the image beside it stays
    viewable without a LaTeX run -- which matters here, because the conversion is lossy for
    some of the figures below (see `_save`).
    """
    global EMIT_TIKZ

    if enabled:
        # Import eagerly so a missing dependency fails here, at the entry point, rather than
        # after the first few figures have already been written.
        import tikzplotlib  # noqa: F401
    EMIT_TIKZ = enabled


def _figure_path(output_dir: Path, stem: str) -> Path:
    """
    Output path for a figure, with the currently selected format appended.

    The extension is appended here rather than left to `rcParams["savefig.format"]`, which
    matplotlib only falls back to when the filename has *no* extension: two of the stems
    below embed a compression ratio (`cache_composition_ratio_0.5`), and matplotlib would
    read that trailing `.5` as an extension named "5" and fail.
    """
    return output_dir / f"{stem}.{FIG_FORMAT}"


def _save(fig, output_dir: Path, stem: str) -> None:
    """
    The single point where a figure reaches disk: the image file always, plus PGFPlots
    source when `set_tikz` is on. Called before `plt.close(fig)` -- the tikz conversion
    walks the live artists, so it cannot run on a closed figure.

    Checked against tikzplotlib 0.10.1.post13 / matplotlib 3.11 on the sampled figures: the
    multi-panel figures come out as `groupplot`s, the shared colorbars of `_add_context_colorbar`
    and `plot_entropy_vs_error` survive as a hidden axis carrying `colormap/viridis` and the
    `point meta` range, and `fig.suptitle` lands as a node. The one lossy case is `ax.boxplot`
    (`KL_spread_across_tasks`): matplotlib hands over loose Line2D segments, so the box and
    whiskers are emitted as literal `\\addplot` paths rather than a pgfplots `boxplot`. It renders
    correctly but cannot be restyled from the LaTeX side.

    Compiling the output needs `\\usepackage{pgfplots}` and `\\usepgfplotslibrary{groupplots}`.
    """
    fig.savefig(_figure_path(output_dir, stem))

    if EMIT_TIKZ:
        import tikzplotlib

        # No axis_width/axis_height: sizing is the document's job (see the module docstring).
        # float_format trims the coordinate lists -- the trace figures carry 150 series, and
        # full double precision makes the .tex several times larger for digits pgfplots
        # cannot resolve on a page anyway.
        tikzplotlib.save(output_dir / f"{stem}.tex", figure=fig, float_format=".5g")


DISTORTION_COMBINED_STEM = "distortion_vs_compression_ratio"
DISTORTION_PANEL_STEMS = ("information_gain_vs_compression_ratio", "KL_vs_compression_ratio")
KL_SPREAD_STEM = "KL_spread_across_tasks"


def _draw_summary_panel(ax, x, mean, std, ylabel: str, title: str) -> None:
    """One mean ± std trend against compression ratio."""
    ax.errorbar(x, mean, yerr=std, marker="o", capsize=3)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Compression ratio")
    ax.set_ylabel(ylabel)
    ax.set_title(title)


def _render_summary_panels(panels: list[tuple], output_dir: Path, combined: str, separate: tuple) -> None:
    """
    Write the panels twice: side by side in one figure, and each on its own.

    The combined figure is for reading the two quantities against each other; the standalone
    ones are for dropping a single panel into a slide or paper at full width, where half of a
    two-panel figure would be scaled down or cropped. Same data and same axes either way, so
    the two forms can never disagree.

    `combined` and `separate` are filename stems; the extension comes from `set_format`.
    """
    fig, axes = plt.subplots(1, len(panels), figsize=FIGSIZE, squeeze=False)
    for ax, panel in zip(axes[0], panels):
        _draw_summary_panel(ax, *panel)
    fig.tight_layout()
    _save(fig, output_dir, combined)
    plt.close(fig)

    for panel, stem in zip(panels, separate):
        fig, ax = plt.subplots(figsize=FIGSIZE)
        _draw_summary_panel(ax, *panel)
        fig.tight_layout()
        _save(fig, output_dir, stem)
        plt.close(fig)


def plot_dataset_summary(
    per_task: pd.DataFrame,
    summary: pd.DataFrame,
    output_dir: Path,
    *,
    unit: str = "bits/token",
    spread_ylabel: str = "per-task mean KL (bits/token)",
    combined_stem: str = DISTORTION_COMBINED_STEM,
    panel_stems: tuple[str, str] = DISTORTION_PANEL_STEMS,
    spread_stem: str = KL_SPREAD_STEM,
) -> None:
    """
    Dataset-level IG and KL against compression ratio, plus the spread of KL across tasks.

    The teacher-forced regime calls this once per position-aggregation (mean over positions ->
    bits/token, sum -> bits/sequence), so `unit` and the stems are parameters: the defaults
    reproduce the mean variant's labels and filenames exactly, and the sum variant passes
    suffixed stems so it cannot overwrite them. `spread_ylabel` is separate from `unit` rather
    than derived, because its wording names the *position* aggregation ("per-task mean KL")
    which `unit` alone cannot convey.
    """
    _render_summary_panels(
        [
            (summary["ratio"], summary["IG_mean"], summary["IG_std"],
             f"IG = H_compressed - H_full ({unit})", "Information gain (mean ± std over tasks)"),
            (summary["ratio"], summary["KL_mean"], summary["KL_std"],
             f"KL(p_full || p_compressed) ({unit})", "Distribution shift (mean ± std over tasks)"),
        ],
        output_dir, combined_stem, panel_stems,
    )

    ratios = sorted(per_task["ratio"].unique())
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.boxplot(
        [per_task.loc[per_task["ratio"] == r, "KL"] for r in ratios],
        tick_labels=[str(r) for r in ratios],
    )
    ax.set_xlabel("Compression ratio")
    ax.set_ylabel(spread_ylabel)
    ax.set_title("Spread of KL divergence across tasks")
    fig.tight_layout()
    _save(fig, output_dir, spread_stem)
    plt.close(fig)


def plot_position_traces(records: pd.DataFrame, ratio: float, output_dir: Path) -> None:
    """
    IG_t and KL_t against position t in the reference sequence, one line per task.

    Note: with KVPress, StreamingLLMPress only prunes the cache once during
    prefill (compression is skipped once generation starts, see BasePress).
    So the sink/eviction boundary is fixed *inside the context* before any of
    these positions are reached -- it does not move along this t axis. See
    `plot_cache_composition` for where the boundary actually falls.
    """
    subset = records[records["ratio"] == ratio]
    if subset.empty:
        return

    fig, axes = plt.subplots(2, 1, figsize=FIGSIZE, sharex=True)
    for _, group in subset.groupby("task_id"):
        group = group.sort_values("position")
        axes[0].plot(group["position"], group["IG"], alpha=0.5, linewidth=1)
        axes[1].plot(group["position"], group["KL"], alpha=0.5, linewidth=1)

    axes[0].axhline(0, color="black", linewidth=0.5)
    # Wrapped labels: stacked panels are half the figure height each, which at 18 pt is less
    # than the single-line form of either label needs, so unwrapped they run into each other.
    axes[0].set_ylabel("I = H_compressed - H_full\n(bits)")
    axes[1].set_ylabel("KL(p_full || p_compressed)\n(bits)")
    axes[1].set_xlabel("position in reference sequence (t)")
    n_tasks = subset["task_id"].nunique()
    fig.suptitle(
        f"compression_ratio={ratio}  (n={n_tasks} tasks, one line per task)\n"
        "cache is pruned once during prefill and fixed thereafter\n"
        "(see cache_composition plot for the boundary)"
    )
    fig.tight_layout()
    _save(fig, output_dir, f"per_position_traces_ratio_{ratio}")
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

    fig, ax = plt.subplots(figsize=FIGSIZE)
    for i, row in enumerate(subset.itertuples()):
        prompt_length = row.cache_seq_length_full
        n_kept = row.cache_seq_length_comp
        n_pruned = prompt_length - n_kept
        sink_end = min(n_sink, prompt_length)
        evicted_end = sink_end + n_pruned

        ax.barh(i, sink_end, color="tab:green", label="sink (kept)" if i == 0 else None)
        ax.barh(i, evicted_end - sink_end, left=sink_end, color="lightgray", label="evicted" if i == 0 else None)
        ax.barh(
            i, prompt_length - evicted_end, left=evicted_end, color="tab:blue",
            label="recent window (kept)" if i == 0 else None,
        )

    # The bars stay one per task, but at 18 pt only ~25 labels fit in the fixed figure height,
    # so label every k-th row rather than letting task ids overplot into an unreadable band.
    step = max(1, math.ceil(len(subset) / MAX_LABELLED_ROWS))
    ax.set_yticks(range(0, len(subset), step))
    ax.set_yticklabels(subset["task_id"].iloc[::step])
    ax.set_xlabel("position in original context")
    ax.set_ylabel("task_id")
    ax.set_title(f"Cache composition at compression_ratio={ratio} (n_sink={n_sink})")
    ax.legend(loc="upper right")
    fig.tight_layout()
    _save(fig, output_dir, f"cache_composition_ratio_{ratio}")
    plt.close(fig)


def plot_sampled_distortion(comparison: pd.DataFrame, output_dir: Path) -> None:
    """
    Sampled-regime analog of `plot_dataset_summary`, from the sampled `comparison` frame
    (one row per task/ratio): sequence entropy change and sequence KL against ratio, plus
    the spread of per-task KL.

    Both quantities are per *sequence*, not per token as in the teacher-forced figure of the
    same filename -- the axis labels say so, because the two are not interchangeable. The zero
    line on the KL panel is a correctness reference rather than a neutral level: KL >= 0 by
    definition, so points dipping below it mark tasks where the Monte Carlo estimate is
    variance-dominated at this n_mc_samples.
    """
    if comparison.empty or comparison["KL_seq"].isna().all():
        return

    ratios = sorted(comparison["ratio"].unique())
    IG = comparison.groupby("ratio")["IG_seq"].agg(["mean", "std"]).reindex(ratios)
    KL = comparison.groupby("ratio")["KL_seq"].agg(["mean", "std"]).reindex(ratios)

    _render_summary_panels(
        [
            (ratios, IG["mean"], IG["std"],
             "IG = H_compressed - H_full (bits/sequence)", "Information gain (mean ± std over tasks)"),
            (ratios, KL["mean"], KL["std"],
             "KL(p_full || p_compressed) (bits/sequence)", "Distribution shift (mean ± std over tasks)"),
        ],
        output_dir, DISTORTION_COMBINED_STEM, DISTORTION_PANEL_STEMS,
    )

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.boxplot(
        [comparison.loc[comparison["ratio"] == r, "KL_seq"].dropna() for r in ratios],
        tick_labels=[str(r) for r in ratios],
    )
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Compression ratio")
    ax.set_ylabel("per-task sequence KL (bits/sequence)")
    ax.set_title("Spread of KL divergence across tasks")
    fig.tight_layout()
    _save(fig, output_dir, KL_SPREAD_STEM)
    plt.close(fig)


def plot_distortion_traces(
    per_task: pd.DataFrame,
    output_dir: Path,
    delta_column: str,
    KL_column: str,
    delta_ylabel: str,
    KL_ylabel: str,
    stem: str = "distortion_traces_per_task",
    context_column: Optional[str] = "context_length",
    *,
    panel_stems: tuple[str, str] = ("information_gain_traces_per_task", "KL_traces_per_task"),
) -> None:
    """
    Information gain and distribution shift against compression ratio, one line per task.

    The companion to the mean ± std figures: those collapse every task into one point per
    ratio, which hides whether a rising mean is the whole dataset drifting or a handful of
    tasks blowing up. These traces show the per-task curves that the mean is summarising.

    Traces are colored by `context_column` (per-task prompt length), not by task identity: at
    these task counts per-series hues would be a cycled rainbow encoding nothing. Color here
    answers the question the figure exists for -- whether the tasks that degrade worst under
    eviction are the long-context ones.

    Works for both regimes; the caller supplies the column names and units (teacher-forced is
    bits/token over reference positions, sampled is bits/sequence over Monte Carlo draws).

    `panel_stems` names the two standalone figures, alongside `stem` for the combined one. It
    has to be a parameter, not a literal: a caller that plots the same quantities twice (the
    teacher-forced regime does, once per position-aggregation) would otherwise overwrite these
    two files even after passing a distinct `stem`.
    """
    missing = [c for c in (delta_column, KL_column) if c not in per_task]
    if per_task.empty or missing or per_task[KL_column].isna().all():
        return

    _render_trace_figure(
        per_task, output_dir, stem,
        panels=[(delta_column, delta_ylabel, "Information gain", panel_stems[0]),
                (KL_column, KL_ylabel, "Distribution shift", panel_stems[1])],
        suptitle="Per-task distortion vs compression ratio",
        zero_line=True,
        context_column=context_column,
    )


BASE_TRACE_WIDTH = 1.0
MIN_TRACE_WIDTH = 0.3
MAX_TRACE_WIDTH = 2.6

# Context length is still a magnitude, so the ramp must stay *ordered* -- but viridis spans
# purple -> blue -> green -> yellow, so neighbouring traces are separable by hue and not only
# by shade, which a single-hue Blues ramp could not do once 150 semi-transparent lines overlap.
#
# This is not the rainbow anti-pattern: viridis is perceptually uniform and monotonic in
# lightness, so it reads as an ordered scale and survives grayscale, where jet/rainbow have
# non-monotonic lightness and invent category boundaries. It is also colorblind-safe. Untruncated
# because both ends are already visible against white (unlike Blues, whose first steps vanish).
CONTEXT_CMAP = plt.get_cmap("viridis")


def _trace_widths(lengths: pd.Series) -> pd.Series:
    """
    Map per-task context lengths to line widths, *proportionally* to the length itself.

        width = clip(BASE * length / median(length), MIN, MAX)

    Proportional rather than min-max normalised on purpose. Stretching the observed range
    across a fixed width band would make any spread look maximal: a HotpotQA run truncated to
    max_context_length=1024 has prompt lengths spanning only 1070-1110 tokens (under 4%), and
    rank-normalising that would draw hairlines next to thick lines and imply a variation in
    context that does not exist. Anchoring on the median instead means a narrow spread renders
    as near-uniform thickness (truthful) while a genuinely wide one -- e.g. an untruncated
    LongBench trec run at 1.9k-11.4k tokens -- separates clearly.
    """
    median = lengths.median()
    if not median > 0:
        return pd.Series(BASE_TRACE_WIDTH, index=lengths.index)
    return (BASE_TRACE_WIDTH * lengths / median).clip(MIN_TRACE_WIDTH, MAX_TRACE_WIDTH)


SUBSET_TASKS = 50


def _render_trace_figure(
    per_task: pd.DataFrame,
    output_dir: Path,
    combined_stem: str,
    panels: list[tuple[str, str, str]],
    suptitle: str,
    zero_line: bool,
    context_column: Optional[str] = None,
) -> None:
    """
    Shared body of the per-task trace figures: one line per task across compression ratios.
    `panels` is [(column, ylabel, title, standalone_stem), ...].

    Each panel is written twice: side by side in the combined figure, and on its own under
    `standalone_stem`, for the same reason as `_render_summary_panels` -- a single panel goes
    into a slide at full width without cropping half a two-panel figure. Both come from the
    same `_draw_trace_panel` call, so they cannot disagree.

    That happens once per task subset: the full set, and a `_<SUBSET_TASKS>tasks` variant
    holding the first N task ids. At 150 traces the figure reads as a density; the subset
    exists so individual curves can actually be followed. The subset is the first N by task id
    (deterministic, and dataset order is already arbitrary), matching `plot_cache_composition`.

    `zero_line` is for panels whose zero is a meaningful reference (a signed difference, or
    a KL that cannot legitimately go negative); it is off for quantities like a benchmark
    score or an entropy, where zero is just the bottom of the range.

    `context_column` names a per-task context length, encoded as trace color on a single-hue
    light->dark ramp (a magnitude, so never a rainbow) and *redundantly* as line thickness via
    `_trace_widths`. Double-encoding one variable is deliberate: overlapping thin lines lose
    their hue where they cross, and the width channel survives grayscale printing. The colorbar
    is the key for both, so the width legend is dropped. A run without that column draws
    uniform blue traces and loses only that dimension.
    """
    task_ids = sorted(per_task["task_id"].unique())
    variants = [(task_ids, "")]
    if len(task_ids) > SUBSET_TASKS:
        variants.append((task_ids[:SUBSET_TASKS], f"_{SUBSET_TASKS}tasks"))

    for subset_ids, suffix in variants:
        _render_trace_variant(
            per_task[per_task["task_id"].isin(subset_ids)],
            output_dir, combined_stem, suffix, panels, suptitle, zero_line, context_column,
        )


def _draw_trace_panel(ax, per_task, column, ylabel, title, zero_line, lengths, widths, norm) -> None:
    """One panel of per-task traces; `lengths`/`widths`/`norm` are None when uncolored."""
    for task_id, group in per_task.groupby("task_id"):
        group = group.sort_values("ratio")
        # marker as well as line: with a single compression ratio a trace is one point,
        # which a line-only style would render invisible.
        ax.plot(
            group["ratio"], group[column],
            color="tab:blue" if norm is None else CONTEXT_CMAP(norm(lengths.get(task_id))),
            alpha=0.55 if norm is not None else 0.25, marker="o", markersize=3,
            linewidth=BASE_TRACE_WIDTH if widths is None else widths.get(task_id, BASE_TRACE_WIDTH),
        )

    if zero_line:
        ax.axhline(0, color="black", linewidth=0.5)
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.set_xlabel("Compression ratio")
    ax.set_ylabel(ylabel)
    ax.set_title(title)


def _add_context_colorbar(fig, axes_list, norm) -> None:
    bar = fig.colorbar(ScalarMappable(norm=norm, cmap=CONTEXT_CMAP), ax=axes_list, pad=0.02)
    bar.set_label("context length (prompt tokens)")


def _render_trace_variant(
    per_task: pd.DataFrame,
    output_dir: Path,
    combined_stem: str,
    suffix: str,
    panels: list[tuple[str, str, str, str]],
    suptitle: str,
    zero_line: bool,
    context_column: Optional[str],
) -> None:
    """Draw the combined figure and one standalone figure per panel, for the tasks in `per_task`."""
    lengths = widths = norm = None
    if context_column is not None and context_column in per_task and per_task[context_column].notna().any():
        lengths = per_task.groupby("task_id")[context_column].median()
        widths = _trace_widths(lengths)
        low, high = float(lengths.min()), float(lengths.max())
        # A run truncated to a fixed max_context_length can leave every task the same length;
        # a degenerate Normalize would then paint everything at one end of the ramp.
        norm = Normalize(vmin=low, vmax=high if high > low else low + 1.0)
        # Normalize over the tasks actually drawn, but keep it identical across panels of the
        # same variant so a color means the same context length in every figure of the pair.

    n_tasks = per_task["task_id"].nunique()

    fig, axes = plt.subplots(
        1, len(panels), figsize=FIGSIZE, sharex=True, squeeze=False, constrained_layout=True
    )
    for ax, (column, ylabel, title, _) in zip(axes[0], panels):
        _draw_trace_panel(ax, per_task, column, ylabel, title, zero_line, lengths, widths, norm)
    if norm is not None:
        _add_context_colorbar(fig, axes.ravel().tolist(), norm)
    fig.suptitle(f"{suptitle}  (n={n_tasks} tasks)")
    _save(fig, output_dir, f"{combined_stem}{suffix}")
    plt.close(fig)

    for column, ylabel, title, standalone_stem in panels:
        fig, ax = plt.subplots(figsize=FIGSIZE, constrained_layout=True)
        _draw_trace_panel(ax, per_task, column, ylabel, title, zero_line, lengths, widths, norm)
        if norm is not None:
            _add_context_colorbar(fig, [ax], norm)
        ax.set_title(f"{title}  (n={n_tasks} tasks)")
        _save(fig, output_dir, f"{standalone_stem}{suffix}")
        plt.close(fig)


def plot_quality_traces(
    per_config: pd.DataFrame,
    output_dir: Path,
    primary_metric: str,
    stem: str = "quality_traces_per_task",
) -> None:
    """
    Answer quality and entropy against compression ratio, one line per task (sampled only).

    The per-task companion to `plot_accuracy_vs_ratio`. That figure reports one mean per
    ratio and draws full attention as a horizontal baseline; here each task is a trace, so a
    flat dataset mean can be read for what it is -- genuinely stable, or tasks improving and
    degrading in equal measure and cancelling out.

    Full attention is placed at ratio 0.0 rather than drawn as a separate baseline: ratio 0.0
    *is* no eviction (which is why the ratio loop skips it as a duplicate of `full`), so every
    trace starts from its own uncompressed value and the drop is read along the line.

    Teacher-forced runs never reach this: they generate no answer, so there is no quality axis.
    """
    if per_config.empty or "primary_score" not in per_config or per_config["primary_score"].isna().all():
        return

    frame = per_config.copy()
    frame.loc[frame["kv_caching"] == "full", "ratio"] = 0.0
    frame = frame[frame["ratio"].notna()]

    # Context length is a per-task constant, but `cache_seq_length` is per *config* (the full
    # config's value is the prompt length; the compressed ones are the pruned sizes). Take the
    # full-attention row so the color encodes the prompt, not the surviving cache.
    if "context_length" not in frame and "cache_seq_length" in per_config:
        # groupby().first() rather than set_index(): a results dir that was resumed (or written
        # by two concurrent jobs) can hold the same task twice, and a duplicated index would
        # make the map raise instead of plotting.
        full_lengths = (
            per_config[per_config["kv_caching"] == "full"].groupby("task_id")["cache_seq_length"].first()
        )
        frame["context_length"] = frame["task_id"].map(full_lengths)

    _render_trace_figure(
        frame, output_dir, stem,
        panels=[("primary_score", f"{primary_metric} (per task)", "Answer quality",
                 "answer_quality_traces_per_task"),
                ("H_seq", "Ĥ(Y|x) (bits/sequence)", "Entropy",
                 "entropy_traces_per_task")],
        suptitle="Per-task answer quality and entropy vs compression ratio",
        zero_line=False,
        context_column="context_length",
    )


# Single hue, light -> dark, for encoding a magnitude (never a rainbow: multi-hue ramps imply
# category boundaries that aren't in the data). Truncated at the pale end because Blues' first
# steps are near-invisible against a white figure, which would hide exactly the low-scoring
# tasks this plot exists to find.
SCORE_CMAP = LinearSegmentedColormap.from_list("blues_visible", plt.get_cmap("Blues")(np.linspace(0.25, 1.0, 256)))


def plot_entropy_vs_error(records: pd.DataFrame, output_dir: Path, primary_metric: str) -> None:
    """
    Sequence entropy vs error, one panel per kv_caching config, one point per task.

    The generation analog of the entropy-error scatter in Meronen et al. (WACV 2024)
    Fig. 2/4: points in the upper *left* are the bad ones -- the model got the answer
    wrong while reporting low entropy (confidently wrong). Upper right means it was wrong
    but knew it was uncertain, which is the well-behaved failure mode.

    Points are shaded by the continuous benchmark score rather than split at a correct/incorrect
    cutoff, so no arbitrary threshold enters the figure. Shade and height encode the same
    quantity (error = 1 - primary_score); the redundancy is deliberate -- it makes the quality
    gradient readable along the entropy axis without drawing a cutoff line that isn't real.
    """
    labels = list(dict.fromkeys(records["config_label"]))
    n_cols = min(3, len(labels))
    n_rows = math.ceil(len(labels) / n_cols)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=FIGSIZE, squeeze=False, sharex=True, sharey=True, constrained_layout=True
    )

    # With a shared grid at 18 pt, an axis label on every panel is both redundant and wide
    # enough to run into the neighbouring panel, so label the edges only: the y axis on the
    # first column, the x axis on the lowest *drawn* panel of each column (the bottom row is
    # partly blank whenever the config count is not a multiple of n_cols).
    bottom_of_column = {i % n_cols: i // n_cols for i in range(len(labels))}

    points = None
    for i, label in enumerate(labels):
        row, col = divmod(i, n_cols)
        ax = axes[row][col]
        group = records[records["config_label"] == label]
        points = ax.scatter(
            group["H_seq"], group["error"],
            c=group["primary_score"], cmap=SCORE_CMAP, vmin=0.0, vmax=1.0,
            s=26, edgecolors="white", linewidths=0.4,  # ring keeps overlapping marks separable
        )
        ax.set_title(f"{label}  ({primary_metric}={group['primary_score'].mean():.2f})")
        if row == bottom_of_column[col]:
            ax.set_xlabel("Entropy Ĥ(Y|x) (bits)")
        if col == 0:
            ax.set_ylabel("error = 1 - correctness metric")
    for i in range(len(labels), n_rows * n_cols):
        axes[i // n_cols][i % n_cols].axis("off")

    if points is not None:
        fig.colorbar(points, ax=axes.ravel().tolist(), label=primary_metric, fraction=0.025, pad=0.02)
    fig.suptitle(
        f"Entropy vs error  (n={records['task_id'].nunique()} tasks, "
        f"shaded by {primary_metric})\n"
        "upper-left = confidently wrong; upper-right = wrong but appropriately uncertain"
    )
    # No bbox_inches="tight" here: it would crop the canvas to the drawn content and hand back
    # a figure of some other size, which is exactly what FIGSIZE is pinning down.
    _save(fig, output_dir, "entropy_vs_error")
    plt.close(fig)


def plot_reliability(
    bin_tables: dict[str, pd.DataFrame], eces: dict[str, float], output_dir: Path, primary_metric: str
) -> None:
    """Reliability diagram: one curve per kv_caching config, with its ECE in the legend."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="perfect calibration")

    for label, table in bin_tables.items():
        if table.empty:
            continue
        ax.plot(
            table["mean_confidence"], table["mean_score"], marker="o",
            label=f"{label} (ECE={eces[label]:.3f})",
        )

    ax.set_xlabel("Confidence = exp(-Ĥ / mean_length)")
    ax.set_ylabel(f"Mean {primary_metric}")
    ax.set_title("Calibration of entropy against answer quality")
    ax.legend()
    fig.tight_layout()
    _save(fig, output_dir, "calibration")
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

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE, constrained_layout=True)

    for column in metric_columns:
        mean_col = f"{column}_mean"
        if mean_col in stream and stream[mean_col].notna().any():
            axes[0].plot(stream["ratio"], stream[mean_col], marker="o", label=column)
    primary_mean = f"{primary_metric}_mean"
    if not full.empty and primary_mean in full and full[primary_mean].notna().any():
        axes[0].axhline(
            full[primary_mean].iloc[0], color="gray", linestyle="--", linewidth=1,
            label=f"{primary_metric} (full cache)",
        )
    axes[0].set_xlabel("Compression ratio")
    axes[0].set_ylabel("Mean " + primary_metric + " over tasks")
    axes[0].set_title("Correctness vs compression")
    axes[0].legend()

    axes[1].errorbar(
        stream["ratio"], stream["H_seq_mean"], yerr=stream["H_seq_std"], marker="o", capsize=3,
        label="Ĥ_compressed",
    )
    if not full.empty:
        axes[1].axhline(
            full["H_seq_mean"].iloc[0], color="gray", linestyle="--", linewidth=1, label="Ĥ_full",
        )
    axes[1].set_xlabel("Compression ratio")
    axes[1].set_ylabel("Ĥ(Y|x) (bits/sequence)")
    # Wrapped: at 18 pt the one-line form is wider than a half-figure panel and gets clipped.
    axes[1].set_title("Entropy vs compression\n(mean ± std over tasks)")
    axes[1].legend()

    _save(fig, output_dir, "accuracy_vs_compression_ratio")
    plt.close(fig)
