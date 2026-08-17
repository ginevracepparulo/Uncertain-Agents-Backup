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
    sequences: torch.Tensor  # (n_samples, T_max) int64 sampled token ids; row i is only
    # valid for its first lengths[i] entries (rows that finished
    # early keep decoding, but those tokens are masked out)


@dataclass
class SequenceKLEstimate:
    """Monte Carlo estimate of the sequence-level KL(p_full || p_compressed), in bits."""

    KL: float
    se: float
    n_samples: int


@dataclass
class SequenceEntropyEstimate:
    """Monte Carlo estimate of sequence-level entropy H(Y | x), in bits."""

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


def KL_bits(log_probs: torch.Tensor, log_q: torch.Tensor) -> torch.Tensor:
    """KL(p || q) per row, in bits."""
    p = log_probs.exp()
    return (p * (log_probs - log_q)).sum(dim=-1) / LOG2


def cross_entropy_bits(log_probs: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    """-log p(token) per position, in bits."""
    return -log_probs.gather(1, token_ids.unsqueeze(1)).squeeze(1) / LOG2


def compute_sequence_metrics(full: TeacherForcedOutput, compressed: TeacherForcedOutput, reference_ids: torch.Tensor) -> pd.DataFrame:
    ref = reference_ids[0].to(full.log_probs.device)
    h_full = entropy_bits(full.log_probs)
    h_comp = entropy_bits(compressed.log_probs)
    return pd.DataFrame(
        {
            "position": range(len(ref)),
            "h_full": h_full.tolist(),
            "h_comp": h_comp.tolist(),
            "IG": (h_comp - h_full).tolist(),
            "KL": KL_bits(full.log_probs, compressed.log_probs).tolist(),
            "ce_full": cross_entropy_bits(full.log_probs, ref).tolist(),
            "ce_comp": cross_entropy_bits(compressed.log_probs, ref).tolist(),
        }
    )


def sequence_entropy_estimate(sample_output: SampledOutput) -> SequenceEntropyEstimate:
    """Turns a SampledOutput into H_hat/se/varentropy, converted from nats to bits."""
    surprisal_bits = sample_output.surprisal / LOG2
    n = surprisal_bits.shape[0]
    varentropy = surprisal_bits.var(unbiased=True).item()
    return SequenceEntropyEstimate(
        H_hat=surprisal_bits.mean().item(),
        se=(varentropy / n) ** 0.5,
        varentropy=varentropy,
        n_samples=n,
        mean_length=sample_output.lengths.float().mean().item(),
        cache_seq_length=sample_output.cache_seq_length,
    )


def sequence_KL_estimate(logp_full: torch.Tensor, logp_compressed: torch.Tensor) -> SequenceKLEstimate:
    """
    Monte Carlo estimate of the sequence-level KL divergence, in bits:

        KL(p_f || p_c) = E_{y ~ p_f} [ log p_f(y|x) - log p_c(y|x) ]
        KL_hat         = (1/N) sum_i [ log p_f(y^(i)|x) - log p_c(y^(i)|x) ],  y^(i) ~ p_f(.|x)

    Both arguments are (n_samples,) *natural-log* sequence log-probabilities of the SAME
    sequences y^(i), which must have been drawn from p_f: `logp_full` comes from the full
    config's own sampling pass, `logp_compressed` from re-scoring those very sequences under
    the compressed press. The compressed model never generates here -- it only scores -- which
    is why the two configs producing different free-running samples does not matter.

    Unbiased, but high variance: each position contributes a single scalar (the log-prob of the
    one token that was sampled), discarding the rest of the logit vector, and the log-ratios sum
    over positions so per-token heavy tails compound across the sequence. KL >= 0 by definition,
    so a negative estimate means the sample size is too small for the spread.
    """
    diff_bits = (logp_full - logp_compressed) / LOG2
    n = diff_bits.shape[0]
    return SequenceKLEstimate(
        KL=diff_bits.mean().item(),
        se=(diff_bits.var(unbiased=True).item() / n) ** 0.5,
        n_samples=n,
    )


def sequence_confidence(estimate: SequenceEntropyEstimate) -> float:
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
    confidence, score, n_bins: int = 5, strategy: str = "quantile"
) -> tuple[float, pd.DataFrame]:
    """
    Expected calibration error, plus the per-bin table a reliability diagram needs.

        ECE = sum_b (n_b / N) * |mean_score_b - mean_confidence_b|

    `score` is the benchmark's own per-task quality in [0, 1] (token-F1, ROUGE, ...), used
    directly rather than thresholded into a 0/1 correctness. The textbook ECE is the special
    case where `score` happens to be binary, and the arithmetic below is identical either way
    -- only the reading of a bin changes: "tasks at confidence 0.7 scored 0.7 on average"
    rather than "70% of tasks at confidence 0.7 were correct". Keeping it continuous avoids
    an arbitrary cutoff and uses the full resolution of metrics that are rarely 0 or 1.

    `strategy="quantile"` puts an equal number of examples in each bin, which is the
    right default at the task counts used here (a few dozen): equal-width bins over
    [0, 1] would leave most bins empty and put nearly every example in one or two.
    Pass "uniform" for the classic equal-width bins.
    """
    conf = np.asarray(confidence, dtype=float)
    corr = np.asarray(score, dtype=float)
    finite = np.isfinite(conf) & np.isfinite(corr)
    conf, corr = conf[finite], corr[finite]

    n = conf.size
    if n == 0:
        return float("nan"), pd.DataFrame(columns=["bin_left", "bin_right", "n", "mean_confidence", "mean_score"])

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
        mean_score = float(corr[selected].mean())
        ece += (n_b / n) * abs(mean_score - mean_conf)
        rows.append(
            {
                "bin_left": float(edges[b]),
                "bin_right": float(edges[b + 1]),
                "n": n_b,
                "mean_confidence": mean_conf,
                "mean_score": mean_score,
            }
        )

    return ece, pd.DataFrame(rows)
