# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Correctness tests for the sampled regime's sequence-KL estimator.

The load-bearing check is the zero-KL identity: re-scoring the full model's own draws under
`press=None` must reproduce the log-probabilities the sampling pass already accumulated. That
single equality pins down token alignment, the per-row length masking, and the `position_ids`
numbering all at once -- if any of them were off, the two would disagree and every KL number
the sampled regime reports would be silently wrong.
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))

from entropy_analysis import (  # noqa: E402
    resolve_mc_chunks,
    run_sampled_task,
    score_sequences_under_press,
)
from entropy_metrics import sequence_KL_estimate  # noqa: E402
from kvpress import StreamingLLMPress  # noqa: E402

from tests.fixtures import unit_test_model  # noqa: E402, F401


@pytest.fixture
def prompt_ids(unit_test_model):  # noqa: F811
    torch.manual_seed(0)
    return torch.randint(0, 100, (1, 32), device=unit_test_model.device)


def test_scoring_pass_reproduces_sampling_logprobs(unit_test_model, prompt_ids):  # noqa: F811
    """Re-scoring full's own draws with no press must return the sampling pass's log-probs."""
    torch.manual_seed(0)
    sampled = run_sampled_task(
        unit_test_model, prompt_ids, press=None, n_samples=8, max_new_tokens=12, eos_token_id=-1
    )
    logp = score_sequences_under_press(
        unit_test_model, prompt_ids, sampled.sequences, sampled.lengths, press=None
    )
    assert torch.allclose(logp, -sampled.surprisal, atol=1e-3)


def test_self_kl_is_zero(unit_test_model, prompt_ids):  # noqa: F811
    """KL(p || p) == 0: the estimator must not manufacture divergence out of the same model."""
    torch.manual_seed(0)
    sampled = run_sampled_task(
        unit_test_model, prompt_ids, press=None, n_samples=8, max_new_tokens=12, eos_token_id=-1
    )
    logp = score_sequences_under_press(
        unit_test_model, prompt_ids, sampled.sequences, sampled.lengths, press=None
    )
    estimate = sequence_KL_estimate(-sampled.surprisal, logp)
    assert estimate.n_samples == 8
    assert abs(estimate.KL) < 1e-3


def test_sequences_shape_and_length_masking(unit_test_model, prompt_ids):  # noqa: F811
    """`sequences` must be (n_samples, T_max) with every row's length inside that window."""
    torch.manual_seed(0)
    sampled = run_sampled_task(
        unit_test_model, prompt_ids, press=None, n_samples=8, max_new_tokens=12, eos_token_id=-1
    )
    assert sampled.sequences.shape[0] == 8
    assert sampled.sequences.shape[1] == sampled.lengths.max().item()
    assert (sampled.lengths >= 1).all()


@pytest.mark.parametrize(
    "n_samples,mc_batch_size,expected",
    [
        (8, None, [8]),          # default: one pass over everything
        (8, 16, [8]),            # cap above n_samples is a no-op
        (8, 8, [8]),             # cap equal to n_samples is a no-op
        (8, 4, [4, 4]),          # even split
        (10, 4, [4, 4, 2]),      # ragged final chunk
    ],
)
def test_resolve_mc_chunks(n_samples, mc_batch_size, expected):
    assert resolve_mc_chunks(n_samples, mc_batch_size) == expected
    assert sum(resolve_mc_chunks(n_samples, mc_batch_size)) == n_samples


def test_chunked_sampling_preserves_shapes_and_alignment(unit_test_model, prompt_ids):  # noqa: F811
    """Chunked draws must come back as one (n_samples, ...) block with rows still aligned."""
    torch.manual_seed(0)
    sampled = run_sampled_task(
        unit_test_model, prompt_ids, press=None, n_samples=10, max_new_tokens=12,
        eos_token_id=-1, mc_batch_size=4,
    )
    assert sampled.surprisal.shape == (10,)
    assert sampled.lengths.shape == (10,)
    assert sampled.sequences.shape[0] == 10
    assert sampled.sequences.shape[1] >= sampled.lengths.max().item()

    # The zero-KL identity must still hold row-by-row when BOTH passes are chunked, which is
    # what proves the scoring chunks line up with the sampling chunks.
    logp = score_sequences_under_press(
        unit_test_model, prompt_ids, sampled.sequences, sampled.lengths, press=None, mc_batch_size=4
    )
    assert torch.allclose(logp, -sampled.surprisal, atol=1e-3)


def test_scoring_chunk_size_does_not_change_results(unit_test_model, prompt_ids):  # noqa: F811
    """Re-scoring is deterministic, so its chunking must be exactly invariant."""
    torch.manual_seed(0)
    sampled = run_sampled_task(
        unit_test_model, prompt_ids, press=None, n_samples=8, max_new_tokens=12, eos_token_id=-1
    )
    press = StreamingLLMPress(compression_ratio=0.5, n_sink=4)
    whole = score_sequences_under_press(
        unit_test_model, prompt_ids, sampled.sequences, sampled.lengths, press
    )
    chunked = score_sequences_under_press(
        unit_test_model, prompt_ids, sampled.sequences, sampled.lengths, press, mc_batch_size=3
    )
    assert torch.allclose(whole, chunked, atol=1e-3)


def test_compression_moves_the_estimate(unit_test_model, prompt_ids):  # noqa: F811
    """Scoring under an evicting press must change the log-probs, so KL becomes non-degenerate."""
    torch.manual_seed(0)
    sampled = run_sampled_task(
        unit_test_model, prompt_ids, press=None, n_samples=8, max_new_tokens=12, eos_token_id=-1
    )
    press = StreamingLLMPress(compression_ratio=0.5, n_sink=4)
    logp = score_sequences_under_press(unit_test_model, prompt_ids, sampled.sequences, sampled.lengths, press)
    assert not torch.allclose(logp, -sampled.surprisal, atol=1e-3)
