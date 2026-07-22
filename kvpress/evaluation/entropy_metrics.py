# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Pure information-theory metric math for `entropy_analysis.py`, plus the small
dataclasses that carry the per-task inference outputs those metrics consume.

Kept dependency-free of the model/inference and orchestration code so the metric
functions can be imported and unit-tested on their own. Import direction is
one-way: `entropy_analysis` (inference + orchestration) imports from here; this
module imports nothing from it.
"""

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

LOG2 = torch.log(torch.tensor(2.0)).item()


@dataclass
class TeacherForcedOutput:
    """Per-position teacher-forced quantities for one attention task."""

    log_probs: torch.Tensor  # (ref_len, vocab), natural-log probabilities, float32
    cache_seq_length: int  # number of KV entries retained after prefill


@dataclass
class SampledOutput:
    """Per-sample quantities from free-running (=no teacher forcing) ancestral sampling for one attention task."""

    surprisal: torch.Tensor  # (n_samples,), nats, -log p(y^(i) | prompt)
    lengths: torch.Tensor  # (n_samples,), generated length per sample (tokens, excl. prompt)
    cache_seq_length: int  # number of KV entries retained after prefill


@dataclass
class EntropyEstimate:
    """Plug-in Monte Carlo estimate of sequence-level entropy H(Y | x), in bits."""

    H_hat: float
    se: float
    varentropy: float
    n_samples: int
    mean_length: float
    cache_seq_length: int


def entropy_bits(log_probs: torch.Tensor) -> torch.Tensor:
    """Shannon entropy per row, in bits. log_probs: (seq_len, vocab) natural log."""
    probs = log_probs.exp()
    return -(probs * log_probs).sum(dim=-1) / LOG2


def kl_bits(log_probs: torch.Tensor, log_q: torch.Tensor) -> torch.Tensor:
    """KL(p || q) per row, in bits."""
    p = log_probs.exp()
    return (p * (log_probs - log_q)).sum(dim=-1) / LOG2


def cross_entropy_bits(log_probs: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    """-log p(token) per position, in bits."""
    return -log_probs.gather(1, token_ids.unsqueeze(1)).squeeze(1) / LOG2


def compute_sequence_metrics(full: TeacherForcedOutput, compressed: TeacherForcedOutput, reference_ids: torch.Tensor) -> pd.DataFrame:
    ref = reference_ids[0].to(full.log_probs.device)
    h_full = entropy_bits(full.log_probs)
    h_compressed = entropy_bits(compressed.log_probs)
    return pd.DataFrame(
        {
            "position": range(len(ref)),
            "h_full": h_full.tolist(),
            "h_compressed": h_compressed.tolist(),
            "delta_h": (h_compressed - h_full).tolist(),
            "kl_full_compressed": kl_bits(full.log_probs, compressed.log_probs).tolist(),
            "ce_full": cross_entropy_bits(full.log_probs, ref).tolist(),
            "ce_compressed": cross_entropy_bits(compressed.log_probs, ref).tolist(),
        }
    )


def plugin_entropy_estimate(sample_output: SampledOutput) -> EntropyEstimate:
    """Turn a SampledOutput into H_hat/se/varentropy, converted from nats to bits."""
    surprisal_bits = sample_output.surprisal / LOG2
    n = surprisal_bits.shape[0]
    varentropy = surprisal_bits.var(unbiased=True).item()
    return EntropyEstimate(
        H_hat=surprisal_bits.mean().item(),
        se=(varentropy / n) ** 0.5,
        varentropy=varentropy,
        n_samples=n,
        mean_length=sample_output.lengths.float().mean().item(),
        cache_seq_length=sample_output.cache_seq_length,
    )


def sequence_confidence(estimate: EntropyEstimate) -> float:
    """
    Map a sequence-entropy estimate to a confidence in (0, 1].

        confidence = exp(-H_hat_nats / mean_length)

    i.e. the per-token geometric-mean probability (= 1 / perplexity). H_hat is the mean
    *total* surprisal of a sampled continuation, so dividing by the mean generated length
    gives mean per-token surprisal; exponentiating the negative maps [0, inf) bits of
    uncertainty onto a (0, 1] confidence, which is what a reliability diagram / ECE needs
    on the same scale as an accuracy.
    """
    if not estimate.mean_length > 0:
        return float("nan")
    return math.exp(-(estimate.H_hat * LOG2) / estimate.mean_length)


def compute_ece(
    confidence, correct, n_bins: int = 5, strategy: str = "quantile"
) -> tuple[float, pd.DataFrame]:
    """
    Expected calibration error, plus the per-bin table a reliability diagram needs.

        ECE = sum_b (n_b / N) * |accuracy_b - mean_confidence_b|

    `strategy="quantile"` puts an equal number of examples in each bin, which is the
    right default at the task counts used here (a few dozen): equal-width bins over
    [0, 1] would leave most bins empty and put nearly every example in one or two.
    Pass "uniform" for the classic equal-width bins.
    """
    conf = np.asarray(confidence, dtype=float)
    corr = np.asarray(correct, dtype=float)
    finite = np.isfinite(conf) & np.isfinite(corr)
    conf, corr = conf[finite], corr[finite]

    n = conf.size
    if n == 0:
        return float("nan"), pd.DataFrame(columns=["bin_left", "bin_right", "n", "mean_confidence", "accuracy"])

    if strategy == "quantile":
        edges = np.unique(np.quantile(conf, np.linspace(0.0, 1.0, n_bins + 1)))
        if edges.size < 2:  # every example has the same confidence
            edges = np.array([conf.min() - 1e-9, conf.max() + 1e-9])
    else:
        edges = np.linspace(0.0, 1.0, n_bins + 1)

    # `edges[1:-1]` as the cut points => index i means conf falls in [edges[i], edges[i+1]).
    bin_idx = np.digitize(conf, edges[1:-1], right=False)

    ece = 0.0
    rows = []
    for b in range(edges.size - 1):
        selected = bin_idx == b
        n_b = int(selected.sum())
        if n_b == 0:
            continue
        mean_conf = float(conf[selected].mean())
        accuracy = float(corr[selected].mean())
        ece += (n_b / n) * abs(accuracy - mean_conf)
        rows.append(
            {
                "bin_left": float(edges[b]),
                "bin_right": float(edges[b + 1]),
                "n": n_b,
                "mean_confidence": mean_conf,
                "accuracy": accuracy,
            }
        )

    return ece, pd.DataFrame(rows)
