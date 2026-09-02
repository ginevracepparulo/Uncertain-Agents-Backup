# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Scorer for mini-swe-agent turns, used when `entropy_analysis.py` runs with --trajectory_path.

An agent turn's completion has two parts with very different stakes, so they are scored
separately rather than blended:

  * the **action** -- the bash command inside the ```mswea_bash_command fence. This is the
    only part that touches the environment, so it is the closest thing to "did compression
    change what the agent did".
  * the **reasoning** -- the prose outside the fence. It never executes, but it is what the
    model conditions its next turn on, so drift there is worth seeing.

Each gets both an exact match and a token-F1, giving four columns in [0, 1]:
`action_exact_match`, `action_f1`, `reasoning_exact_match`, `reasoning_f1`.

Follows the SCORER_REGISTRY contract: `calculate_metrics(df) -> dict`, reading the greedy
generation from `df["predicted_answer"]` and the gold turn from `df["completion"]`, and
correct on a single-row frame so `benchmark_score_per_example` can reuse it per task.
"""

from collections import Counter

from benchmarks.longbench.calculate_metrics import f1_score, normalize_answer


def _normalize_command(command: str) -> str:
    """Collapse whitespace only.

    Deliberately *not* `normalize_answer`: that strips punctuation and articles, which is
    right for prose and destructive for shell. It would turn `sed -i 's/a/b/g' f.py` and
    `sed -i sabg f.py` into the same string, and erase the difference between `a && b`,
    `a || b` and `a b`.
    """
    return " ".join(command.split())


def _f1(prediction_tokens: list[str], gold_tokens: list[str]) -> float:
    """Token-F1, with the empty cases pinned so it can never disagree with exact match.

    LongBench's `f1_score` returns 0 whenever there is no token overlap, which includes two
    *empty* token lists -- a real case here, since a turn whose completion has no parsable
    action yields an empty gold action, and a compressed model that also emits none has
    reproduced it exactly. Left alone that scores exact_match=1 alongside f1=0, which reads
    as a contradiction in the summary. Both empty is a perfect match; exactly one empty is a
    total miss.
    """
    if not prediction_tokens and not gold_tokens:
        return 1.0
    if not prediction_tokens or not gold_tokens:
        return 0.0
    return float(f1_score(prediction_tokens, gold_tokens))


def _command_f1(prediction: str, gold: str) -> float:
    """Token-F1 over whitespace-separated shell tokens, punctuation preserved."""
    return _f1(_normalize_command(prediction).split(), _normalize_command(gold).split())


def _prose_f1(prediction: str, gold: str) -> float:
    """Token-F1 over prose, using LongBench's own normalisation (lowercase, depunctuate)."""
    return _f1(normalize_answer(prediction).split(), normalize_answer(gold).split())


def score_turn(predicted: str, gold_action: str, gold_reasoning: str) -> dict[str, float]:
    """Four scores for one turn. `predicted` is the whole generated completion."""
    from agent_tasks import split_completion

    action, reasoning = split_completion(predicted)
    return {
        # An unparsable completion scores 0 on both action metrics rather than being skipped:
        # failing to emit a well-formed action *is* a way for compression to break the agent,
        # and silently dropping those turns would flatter the compressed configs.
        "action_exact_match": float(_normalize_command(action) == _normalize_command(gold_action)),
        "action_f1": _command_f1(action, gold_action),
        "reasoning_exact_match": float(normalize_answer(reasoning) == normalize_answer(gold_reasoning)),
        "reasoning_f1": _prose_f1(reasoning, gold_reasoning),
    }


def calculate_metrics(df) -> dict:
    """Mean of each score over the frame's rows (one row = one agent turn)."""
    from agent_tasks import split_completion

    totals: Counter = Counter()
    for _, row in df.iterrows():
        gold = row.get("completion") or ""
        # gold_action/gold_reasoning ride along in Task.raw, but a frame rebuilt from a bare
        # per_config.csv may not carry them; re-splitting the gold completion is equivalent.
        gold_action, gold_reasoning = row.get("gold_action"), row.get("gold_reasoning")
        if gold_action is None or gold_reasoning is None:
            gold_action, gold_reasoning = split_completion(gold)
        totals.update(score_turn(row.get("predicted_answer") or "", gold_action or "", gold_reasoning or ""))
    n = max(len(df), 1)
    return {metric: value / n for metric, value in totals.items()}
