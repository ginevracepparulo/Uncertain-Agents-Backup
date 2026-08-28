# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Correctness tests for the shared prompt prefill (`PrefilledPrompt`).

The sampled regime used to prefill an `n_samples`-row batch of *identical* prompts once per
pass, per chunk. Prefill is now done once at batch 1 and broadcast (`replicate_prefill`), which
is only legitimate if it is exact. These tests pin that down from three sides:

- a shared prefill reproduces an inline one token for token, with and without a press;
- the broadcast cache attends correctly per row, i.e. batch-n scoring matches batch-1;
- and the prompt really is prefilled once per (task, press) rather than once per pass/chunk,
  which is the whole point of the change.

`tests/test_sequence_kl.py`'s zero-KL identity covers the same code path end to end: it still
has to hold, and it is what would break first if the broadcast cache were misaligned.
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))

from entropy_analysis import (  # noqa: E402
    PrefilledPrompt,
    greedy_generate_output,
    prefill_prompt,
    replicate_prefill,
    run_sampled_task,
    score_sequences_under_press,
)
from kvpress import StreamingLLMPress  # noqa: E402

from tests.fixtures import unit_test_model  # noqa: E402, F401


@pytest.fixture
def prompt_ids(unit_test_model):  # noqa: F811
    torch.manual_seed(0)
    return torch.randint(0, 100, (1, 64), device=unit_test_model.device)


@pytest.fixture(params=[None, 0.5], ids=["no_press", "streamingllm@0.5"])
def press(request):
    """Both regimes matter: an uncompressed cache, and one the press has evicted from."""
    if request.param is None:
        return None
    return StreamingLLMPress(compression_ratio=request.param, n_sink=4)


def test_greedy_output_identical_to_inline_prefill(unit_test_model, prompt_ids, press):  # noqa: F811
    """
    Greedy decoding is deterministic, so a shared prefill must reproduce an inline one exactly.

    This is the sharpest available check: any drift in the broadcast cache -- a misaligned
    position, a dropped sink token, a stale eviction -- changes the argmax and shows up as a
    different token sequence rather than a small numeric difference.
    """
    inline = greedy_generate_output(unit_test_model, prompt_ids, press, max_new_tokens=16, eos_token_id=-1)

    prefilled = prefill_prompt(unit_test_model, prompt_ids, press)
    shared = greedy_generate_output(unit_test_model, prefilled, press, max_new_tokens=16, eos_token_id=-1)

    assert torch.equal(inline, shared)


def test_prefill_survives_reuse_across_passes(unit_test_model, prompt_ids, press):  # noqa: F811
    """
    One `PrefilledPrompt` feeds many passes, so `replicate_prefill` must not consume it.

    Decoding mutates its cache -- every step appends a token, and a `DecodingPress` prunes it --
    so if the copy were shallow the second pass would resume from the first pass's leftovers.
    Re-running the same greedy decode has to give the same answer every time.
    """
    prefilled = prefill_prompt(unit_test_model, prompt_ids, press)
    first = greedy_generate_output(unit_test_model, prefilled, press, max_new_tokens=16, eos_token_id=-1)

    # Interleave a batched sampling pass: it decodes from the same prefill at a different batch
    # size, which is exactly what `process_task_sampled` does between the greedy and KL passes.
    torch.manual_seed(0)
    run_sampled_task(unit_test_model, prefilled, press, 4, 8, eos_token_id=-1)

    second = greedy_generate_output(unit_test_model, prefilled, press, max_new_tokens=16, eos_token_id=-1)
    assert torch.equal(first, second)


def test_broadcast_cache_scores_match_batch_one(unit_test_model, prompt_ids, press):  # noqa: F811
    """
    The broadcast batch dimension must not leak between rows.

    Scoring `n` distinct continuations against one shared prompt cache has to give each row the
    same log-probability it would get alone. Rows are independent in attention, so a mismatch
    would mean the expanded (stride-0) keys/values are being read wrongly once the first decode
    step concatenates onto them.
    """
    torch.manual_seed(0)
    sequences = torch.randint(0, 100, (4, 10), device=unit_test_model.device)
    lengths = torch.tensor([10, 7, 10, 3], device=unit_test_model.device)

    prefilled = prefill_prompt(unit_test_model, prompt_ids, press)
    batched = score_sequences_under_press(unit_test_model, prefilled, sequences, lengths, press)

    for i in range(sequences.shape[0]):
        alone = score_sequences_under_press(
            unit_test_model, prefilled, sequences[i : i + 1], lengths[i : i + 1], press
        )
        assert torch.allclose(batched[i], alone[0], atol=1e-4), f"row {i} differs when batched"


def test_chunking_still_agrees_with_one_pass(unit_test_model, prompt_ids, press):  # noqa: F811
    """
    `mc_batch_size` must stay a pure memory knob now that chunks share one prefill.

    Scoring is deterministic (unlike sampling, whose RNG stream depends on the chunking), so
    splitting the rows across passes must reproduce the unchunked log-probabilities exactly.
    """
    torch.manual_seed(0)
    sequences = torch.randint(0, 100, (6, 10), device=unit_test_model.device)
    lengths = torch.full((6,), 10, device=unit_test_model.device)

    prefilled = prefill_prompt(unit_test_model, prompt_ids, press)
    whole = score_sequences_under_press(unit_test_model, prefilled, sequences, lengths, press)
    chunked = score_sequences_under_press(
        unit_test_model, prefilled, sequences, lengths, press, mc_batch_size=2
    )
    assert torch.allclose(whole, chunked, atol=1e-4)


def count_prefills(monkeypatch, model, prompt_length: int) -> list[int]:
    """
    Record the query length of every forward, so prompt-sized ones can be counted.

    Patched via `monkeypatch` rather than assigned: `unit_test_model` is session-scoped, so a
    permanent wrapper would follow the model into every later test in the run.
    """
    seen: list[int] = []
    original = model.forward

    def counting_forward(*args, **kwargs):
        input_ids = kwargs.get("input_ids", args[0] if args else None)
        if input_ids is not None and input_ids.shape[-1] == prompt_length:
            seen.append(input_ids.shape[-1])
        return original(*args, **kwargs)

    monkeypatch.setattr(model, "forward", counting_forward)
    return seen


def test_prompt_is_prefilled_once_per_press(unit_test_model, prompt_ids, press, monkeypatch):  # noqa: F811
    """
    The saving itself: `n_samples` draws over `k` chunks cost one prefill, not `n_samples`.

    Before the change every chunk prefilled its own `mc_batch_size` identical rows, so a
    50-draw pass at `mc_batch_size=8` paid 50 prompt-length row-prefills where 1 suffices.
    """
    prompt_length = prompt_ids.shape[1]
    seen = count_prefills(monkeypatch, unit_test_model, prompt_length)

    prefilled = prefill_prompt(unit_test_model, prompt_ids, press)
    assert len(seen) == 1, "prefill_prompt must run the prompt exactly once"

    # Three passes over the same prefill, the sampling one deliberately chunked.
    torch.manual_seed(0)
    sampled = run_sampled_task(unit_test_model, prefilled, press, 6, 8, eos_token_id=-1, mc_batch_size=2)
    greedy_generate_output(unit_test_model, prefilled, press, max_new_tokens=8, eos_token_id=-1)
    score_sequences_under_press(
        unit_test_model, prefilled, sampled.sequences, sampled.lengths, press, mc_batch_size=2
    )

    assert len(seen) == 1, f"passes re-prefilled the prompt {len(seen) - 1} extra time(s)"


def test_replicate_prefill_reports_prompt_and_cache_lengths(unit_test_model, prompt_ids):  # noqa: F811
    """
    A press evicts, so the two lengths genuinely differ -- and both are load-bearing.

    `cache_seq_length` is the surviving cache reported in the results; `prompt_length` is the
    pre-eviction length that `position_ids` continue from, because pruned tokens keep the RoPE
    positions they were encoded with.
    """
    press = StreamingLLMPress(compression_ratio=0.5, n_sink=4)
    prefilled = prefill_prompt(unit_test_model, prompt_ids, press)

    assert prefilled.prompt_length == prompt_ids.shape[1]
    assert prefilled.cache_seq_length < prefilled.prompt_length

    cache, next_logits = replicate_prefill(prefilled, 5)
    assert next_logits.shape[0] == 5
    assert cache.get_seq_length() == prefilled.cache_seq_length
    assert cache.layers[0].keys.shape[0] == 5
    # The source cache must be untouched, or the next pass would start from a mutated prompt.
    assert prefilled.cache.layers[0].keys.shape[0] == 1


def test_isinstance_guard_accepts_both_prompt_forms(unit_test_model, prompt_ids):  # noqa: F811
    """Raw token ids stay a valid argument, so existing callers and tests keep working."""
    assert not isinstance(prompt_ids, PrefilledPrompt)
    torch.manual_seed(0)
    from_ids = run_sampled_task(unit_test_model, prompt_ids, None, 3, 6, eos_token_id=-1)
    assert from_ids.sequences.shape[0] == 3
    assert from_ids.cache_seq_length == prompt_ids.shape[1]
