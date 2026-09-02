# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Turn a mini-swe-agent trajectory into `entropy_analysis.Task`s -- one task per agent turn.

A `Task` is one LLM prompt plus a reference continuation to score, and an agent turn is
exactly that: the prompt is the message list the agent sent that turn (system + instance +
every prior assistant/observation pair), and the reference is the assistant text it actually
produced. So the whole entropy pipeline applies unchanged, with the agent's own growing
transcript playing the role a retrieved document plays for a QA benchmark.

Kept separate from `entropy_analysis.py` so trajectory-format knowledge lives in one place:
`entropy_analysis` only ever sees `Task` objects.

Text-based trajectories only
----------------------------
mini-swe-agent's default path uses tool calls, which store the assistant message as
`content: null` with the command inside `tool_calls`, and observations under `role: "tool"`.
The text the model actually emitted is discarded by the provider's parsing before mini ever
sees it, so there is no reference token sequence to teacher-force -- only a structured
summary of what the call meant. Reconstructing one would mean re-serialising the call
through whatever tool-call syntax the *local* model's chat template happens to use, which is
model-specific, unrecoverable from the trajectory (key order, whitespace, escaping all move
the token count) and, when the trajectory came from a hosted API, not even the format the
original model emitted.

So `load_agent_tasks` raises on a toolcall trajectory rather than fabricating a reference.
Run the agent through the text-based path instead:

    mini -y -c mini_textbased.yaml -c model.model_class=litellm_textbased ...
    mini -y -c mini_textbased.yaml -c model.model_class=kvpress.mini_swe_agent_model.KVPressLocalModel ...

where the completion is plain text and the reference is `tokenizer.encode(content)`, exact.
"""

import json
import logging
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Matches the mini-swe-agent text-based action fence. Kept in sync with
# LitellmTextbasedModelConfig.action_regex, plus the plain ```bash fence that
# KVPressLocalModel also accepts (small local models often revert to it).
ACTION_REGEX = r"```(?:mswea_bash_command|bash)\s*\n(.*?)\n```"

# Roles that may appear in a prompt prefix. "exit" is mini's own pseudo-role for the
# terminating message and is never sent to a model; "tool" means a toolcall trajectory.
PROMPT_ROLES = {"system", "user", "assistant"}

# Headroom added to the longest observed completion when deriving the run's generation
# budget, mirroring the 20-token headroom the benchmark HF pushes add to their own column.
MAX_NEW_TOKENS_HEADROOM = 20


def split_completion(completion: str) -> tuple[str, str]:
    """
    Split an assistant completion into (action, reasoning).

    The text-based format is free-form reasoning followed by a fenced command, so the action
    is the fence's capture group and the reasoning is everything outside it. Both are scored
    separately: the action is what actually touches the environment, the reasoning is what
    the model talked itself into. A completion with no parsable fence yields an empty action.
    """
    matches = re.findall(ACTION_REGEX, completion, re.DOTALL)
    action = matches[0].strip() if matches else ""
    reasoning = re.sub(ACTION_REGEX, "", completion, flags=re.DOTALL).strip()
    return action, reasoning


def _clean_messages(messages: list[dict]) -> list[dict]:
    """Strip mini's `extra` sidecar so only what the chat template consumes remains.

    `extra` holds parsed actions, costs, timestamps and the raw API response; mini itself
    drops it before every API call (see litellm_model._prepare_messages_for_api), so keeping
    it here would put text in the prompt that the agent never sent.
    """
    return [{"role": m["role"], "content": m["content"]} for m in messages]


def _check_textbased(messages: list[dict], source: str) -> None:
    """Reject toolcall trajectories loudly. See the module docstring for why."""
    for message in messages:
        if message.get("role") == "tool" or message.get("tool_calls"):
            raise ValueError(
                f"{source} is a toolcall trajectory (found a 'tool' role or a 'tool_calls' field). "
                "The assistant text needed as a teacher-forcing reference is not stored in that "
                "format. Re-run the agent through the text-based path, e.g.\n"
                "  mini -y -c mini_textbased.yaml -c model.model_class=litellm_textbased ..."
            )


def _turns_from_traj(path: Path) -> list[dict]:
    """Extract (messages, completion) pairs from a mini-swe-agent `.traj.json`.

    Walks the transcript and, for every assistant message, takes the prefix before it as the
    prompt and its own content as the completion -- which is exactly the query that produced
    it. Stops at the first `exit` message: it is mini's terminator, carries the submission
    rather than a model turn, and no chat template can render its pseudo-role.
    """
    data = json.loads(path.read_text())
    messages = data if isinstance(data, list) else data["messages"]
    _check_textbased(messages, str(path))

    turns, prefix = [], []
    for message in messages:
        role = message.get("role")
        if role == "exit":
            break
        if role not in PROMPT_ROLES:
            raise ValueError(f"{path}: unexpected message role {role!r}")
        if role == "assistant":
            turns.append({"messages": _clean_messages(prefix), "completion": message["content"] or ""})
        prefix.append(message)
    return turns


def _turns_from_jsonl(path: Path) -> list[dict]:
    """Read the per-turn JSONL log KVPressLocalModel writes via its `log_path` config.

    Richer than the .traj.json: each record already holds the exact `messages`/`completion`
    pair, plus `completion_ids` (the raw generated token ids) when the log came from a
    version that records them, which lets teacher-forcing score the very tokens the agent
    emitted instead of a re-encode of the decoded string.
    """
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    records.sort(key=lambda r: r["turn"])
    turns = []
    for record in records:
        _check_textbased(record["messages"], f"{path} (turn {record['turn']})")
        turns.append(
            {
                "messages": _clean_messages(record["messages"]),
                "completion": record.get("completion", ""),
                "completion_ids": record.get("completion_ids"),
                "prompt_length": record.get("prompt_length"),
            }
        )
    return turns


def load_agent_tasks(task_cls, trajectory_path: str, n_samples: Optional[int] = None) -> list:
    """
    args:
        task_cls: `entropy_analysis.Task`, passed in so this module stays import-free of it
        trajectory_path: a mini-swe-agent `.traj.json`, or the `.jsonl` log KVPressLocalModel
            writes; the format is picked by suffix
        n_samples: keep at most this many turns (chronological), or None for all
    outputs:
        one Task per agent turn, in trajectory order

    Turns with an empty completion are dropped: they carry no reference to teacher-force.

    Every task gets the *same* `max_new_tokens` in `raw` -- the longest completion in the
    trajectory (in characters/4 as a token estimate) plus headroom -- so
    `resolve_max_new_tokens` reads one consistent value off the first task instead of warning
    that tasks disagree. The built-in default of 64 is far too small for a turn of reasoning
    plus a code block.
    """
    path = Path(trajectory_path)
    turns = _turns_from_jsonl(path) if path.suffix == ".jsonl" else _turns_from_traj(path)

    usable = [turn for turn in turns if turn["completion"].strip()]
    if len(usable) < len(turns):
        logger.warning(f"{len(turns) - len(usable)} turn(s) had an empty completion and were dropped")
    if n_samples is not None:
        usable = usable[:n_samples]
    if not usable:
        raise ValueError(f"no usable turns in {path}")

    # A rough token estimate is enough: this only sizes the generation budget, and the
    # sampled regime stops at EOS anyway. Avoids needing a tokenizer at load time.
    budget = max(len(turn["completion"]) // 4 for turn in usable) + MAX_NEW_TOKENS_HEADROOM

    tasks = []
    for i, turn in enumerate(usable):
        action, reasoning = split_completion(turn["completion"])
        tasks.append(
            task_cls(
                task_id=str(i),
                context="",
                question="",
                messages=turn["messages"],
                answer=turn["completion"],
                answer_ids=turn.get("completion_ids"),
                # Kept small on purpose: every key here is merged into per_config.csv, so the
                # raw token ids stay on Task.answer_ids rather than bloating a column.
                raw={
                    "turn": i,
                    "completion": turn["completion"],
                    "gold_action": action,
                    "gold_reasoning": reasoning,
                    "logged_prompt_length": turn.get("prompt_length"),
                    "max_new_tokens": budget,
                },
            )
        )
    logger.info(f"Loaded {len(tasks)} agent turn(s) from {path} (max_new_tokens={budget})")
    return tasks


# The two token spans an agent completion splits into. "all" (the whole completion) is kept
# out of this tuple deliberately: it is the existing, unsplit view rather than a span.
SPANS = ("action", "reasoning")


def token_char_offsets(tokenizer, ids: list[int]) -> list[tuple[int, int]]:
    """
    Character span each token occupies in `tokenizer.decode(ids)`.

    Fast path: decode, re-encode with `return_offsets_mapping`, and use those offsets if the
    round trip reproduced the original ids exactly. That is one tokenizer call instead of n.

    Slow path: decode a growing prefix, one call per token. Needed because the ids are not
    always the encoding of a string -- a trajectory that logged `completion_ids` holds the
    raw generated tokens, and re-encoding their decoded text is not guaranteed to give them
    back. It also covers slow tokenizers, which have no offset mapping at all. The
    `max(end, pos)` guard is for byte-level BPE, where a prefix ending mid-multi-byte
    character decodes to a replacement char that the next token resolves, briefly
    shortening the text.

    The sampled regime calls this once per Monte Carlo draw per config, so the fast path is
    what keeps the span split from dominating the run.
    """
    if getattr(tokenizer, "is_fast", False):
        encoded = tokenizer(
            tokenizer.decode(ids, skip_special_tokens=False),
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        if list(encoded["input_ids"]) == list(ids):
            return [tuple(offset) for offset in encoded["offset_mapping"]]

    offsets, pos = [], 0
    for i in range(len(ids)):
        end = max(len(tokenizer.decode(ids[: i + 1], skip_special_tokens=False)), pos)
        offsets.append((pos, end))
        pos = end
    return offsets


def token_span_labels(tokenizer, ids: list[int]) -> list[str]:
    """
    Label every token of a completion `"action"` or `"reasoning"`.

    The action is the fence's *capture group* -- the command itself. The fence markers
    (```mswea_bash_command and the closing ```) count as reasoning: they are format
    scaffolding, not the decision, and lumping them in would credit the action span with
    tokens the model emits by rote.

    A completion with no parsable fence is all reasoning, so its action span is empty and it
    simply contributes no rows to the action-only outputs.
    """
    text = tokenizer.decode(ids, skip_special_tokens=False)
    match = re.search(ACTION_REGEX, text, re.DOTALL)
    if match is None:
        return ["reasoning"] * len(ids)
    start, end = match.span(1)
    return [
        "action" if (tok_start < end and tok_end > start) else "reasoning"
        for tok_start, tok_end in token_char_offsets(tokenizer, ids)
    ]
