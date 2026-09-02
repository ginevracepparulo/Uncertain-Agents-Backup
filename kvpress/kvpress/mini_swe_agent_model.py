# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
A local-model backend for mini-swe-agent (https://github.com/SWE-agent/mini-swe-agent)
that runs a Hugging Face model in-process -- optionally with KVPress StreamingLLM
compression -- instead of calling a hosted LLM API.

This module is only usable from an environment that has BOTH `kvpress` and
`mini-swe-agent` installed. It is not a dependency of the core `kvpress` package:
`minisweagent` is only imported lazily, inside the methods that need it, so importing
`kvpress` itself never requires mini-swe-agent to be installed.

Usage: point mini-swe-agent's run config at this class via its full import path, e.g.
in a YAML config passed to `mini`:

    model_class: kvpress.mini_swe_agent_model.KVPressLocalModel
    model_name: unsloth/Llama-3.2-1B-Instruct
    press_name: streaming_llm                # any key of PRESS_FACTORY
    compression_ratio: 0.5                   # 0.0 = full attention
    n_sink: 4
    decoding_compression_interval: 4         # omit for prefill-only compression
    log_path: ./agent_run_log.jsonl

It must be paired with the *text-based* agent config, because the analysis needs the
assistant's completion as plain text:

    mini -y -c mini_textbased.yaml \
         -c model.model_class=kvpress.mini_swe_agent_model.KVPressLocalModel \
         -c model.model_name=Qwen/Qwen2.5-0.5B-Instruct \
         -c agent.step_limit=6 -t "..." -o run.traj.json

Two operational notes. Cost is always 0.0 here, so mini's `cost_limit` can never fire --
bound runs with `agent.step_limit` instead. And `mini` defaults to the interactive agent,
so unattended runs need `-y`.

Each turn is appended to `log_path` as one JSON line carrying the exact `messages`,
`completion`, `completion_ids` and `prompt_length`. Replay either that file or the ordinary
`.traj.json` through `evaluation/entropy_analysis.py --trajectory_path`, which turns each
turn into one Task and runs the same full-vs-compressed analysis this project already runs
for static QA -- except now over a live agent trajectory.

Usage
-----
------------------------------------------------------------------------
------------------------------------------------------------------------        
from local
MODEL=unsloth/Llama-3.1-8B-Instruct
MODEL_TAG=llama31_8b
PRESS=streaming_llm
TASK=prime_task

RUN=./results/agent_runs/$MODEL_TAG/$TASK
mkdir -p "$RUN"

MSWEA_SILENT_STARTUP=1 .venv/bin/mini -y \
    -c mini_textbased.yaml \
    -c model.model_class=kvpress.mini_swe_agent_model.KVPressLocalModel \
    -c model.model_name=$MODEL \
    -c model.compression_ratio=0.0 \
    -c model.max_new_tokens=512 \
    -c model.log_path="$RUN/agent_log.jsonl" \
    -c agent.step_limit=30 \
    -t "write a python file prime.py with a function is_prime(n)" \
    -o "$RUN/agent.traj.json" \
    2>&1 | tee "$RUN/agent_run.log"
"""

import dataclasses
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from kvpress import (
    AdaKVPress,
    DecodingPress,
    ExpectedAttentionPress,
    KeyDiffPress,
    KnormPress,
    PrefillDecodingPress,
    RandomPress,
    SnapKVPress,
    StreamingLLMPress,
    TOVAPress,
)

# The presses vetted for this prefill + decode pattern, matching
# evaluation/entropy_analysis.py's SUPPORTED_PRESSES so a live agent run and the replay
# analysis of its trajectory can use the same policy. Constructors rather than shared
# instances: a press is stateful, and _build_press runs once per turn.
PRESS_FACTORY = {
    "streaming_llm": StreamingLLMPress,
    "random": RandomPress,
    "snapkv": SnapKVPress,
    "knorm": KnormPress,
    "tova": TOVAPress,
    "keydiff": KeyDiffPress,
    "expected_attention": lambda **kw: AdaKVPress(ExpectedAttentionPress(**kw)),
}

# Presses that score from a window of decode-time hidden states rather than from keys alone;
# they need that window kept in DecodingPress's buffer. Mirrors entropy_analysis.py's
# HIDDEN_STATE_WINDOW_ATTRS.
HIDDEN_STATE_WINDOW_ATTRS = {"snapkv": "window_size", "expected_attention": "n_sink"}

logger = logging.getLogger(__name__)


@dataclass
class KVPressLocalModelConfig:
    model_name: str
    press_name: str = "streaming_llm"
    compression_ratio: float = 0.0
    n_sink: int = 4
    decoding_compression_interval: Optional[int] = None
    decoding_target_size: Optional[int] = None
    max_new_tokens: int = 512
    do_sample: bool = False
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    device: Optional[str] = None
    # Accept both the harness-specific ```mswea_bash_command fence and a plain ```bash fence:
    # smaller local models often revert to the far more common ```bash convention they saw
    # everywhere in training instead of this harness's custom label. The (?:...) is a
    # non-capturing group, so re.findall still returns exactly the one command capture group.
    action_regex: str = r"```(?:mswea_bash_command|bash)\s*\n(.*?)\n```"
    format_error_template: str = (
        "Please always provide EXACTLY ONE action in triple backticks, found {{actions|length}} actions."
    )
    observation_template: str = (
        "{% if output.exception_info %}<exception>{{output.exception_info}}</exception>\n{% endif %}"
        "<returncode>{{output.returncode}}</returncode>\n<output>\n{{output.output}}</output>"
    )
    multimodal_regex: str = ""
    log_path: Optional[str] = None
    cost_per_call: float = 0.0


def _press_instance(press_name: str, ratio: float, n_sink: int):
    """A fresh press at `ratio`, with `n_sink` applied only where the press has it.

    Mirrors entropy_analysis.py's `press_instance`, including resolving through a wrapper
    (`expected_attention` is an AdaKVPress around the real scorer, which proxies
    compression_ratio but not n_sink).
    """
    assert press_name in PRESS_FACTORY, f"press_name must be one of {sorted(PRESS_FACTORY)}, got {press_name!r}"
    press = PRESS_FACTORY[press_name](compression_ratio=ratio)
    scorer = getattr(press, "press", press)
    if hasattr(scorer, "n_sink"):
        scorer.n_sink = n_sink
    return press



class KVPressLocalModel:
    """
    Drop-in local replacement for mini-swe-agent's text-based models: runs a HF model
    in-process (optionally StreamingLLM-compressed via KVPress) instead of calling an API.
    Implements mini-swe-agent's `Model` protocol via duck typing (no import needed at class
    definition time).
    """

    def __init__(self, **kwargs):
        known_fields = {f.name for f in dataclasses.fields(KVPressLocalModelConfig)}
        ignored = {k: v for k, v in kwargs.items() if k not in known_fields}
        if ignored:
            # Bundled configs (e.g. mini_textbased.yaml) carry fields meant for API-based
            # model classes (e.g. litellm's model_kwargs) under the same `model:` section.
            # Since -c configs merge recursively, those ride along even when model_class
            # points here -- drop anything KVPressLocalModelConfig doesn't declare instead
            # of failing on every field a shared config happens to set.
            logger.info(f"Ignoring config keys not used by KVPressLocalModel: {sorted(ignored)}")
        self.config = KVPressLocalModelConfig(**{k: v for k, v in kwargs.items() if k in known_fields})

        device = self.config.device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device
        dtype = {"cuda": torch.bfloat16, "mps": torch.float16}.get(device, torch.float32)

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.config.model_name, torch_dtype=dtype).to(device)
        self.model.eval()

        self._turn = 0
        self._cost = 0.0
        if self.config.log_path:
            Path(self.config.log_path).parent.mkdir(parents=True, exist_ok=True)

    def _build_press(self, prompt_length: int):
        """Mirrors evaluation/entropy_analysis.py's `build_comp_press`, so a live agent run
        and the replay analysis of its trajectory evict on the same rule.

        ratio=0.0 always disables eviction entirely, at both prefill and decode: otherwise
        a decode-time target_size of prompt_length would still trim the cache once it grew
        past the original prompt.
        """
        ratio = self.config.compression_ratio
        press_name, n_sink = self.config.press_name, self.config.n_sink
        prefill_press = _press_instance(press_name, ratio, n_sink)
        if self.config.decoding_compression_interval is None or ratio == 0.0:
            return prefill_press

        target_size = self.config.decoding_target_size or max(
            getattr(prefill_press, "n_sink", 0) + 1, int(prompt_length * (1 - ratio))
        )
        window_attr = HIDDEN_STATE_WINDOW_ATTRS.get(press_name)
        window = getattr(getattr(prefill_press, "press", prefill_press), window_attr, 0) if window_attr else 0
        if window and self.config.decoding_compression_interval <= window:
            raise ValueError(
                f"press_name {press_name} scores over a {window}-step hidden-state window, so "
                f"decoding_compression_interval must exceed {window} (got "
                f"{self.config.decoding_compression_interval})"
            )
        decode_press = DecodingPress(
            base_press=_press_instance(press_name, 0.0, n_sink),
            compression_interval=self.config.decoding_compression_interval,
            target_size=target_size,
            hidden_states_buffer_size=self.config.decoding_compression_interval if window else 0,
        )
        return PrefillDecodingPress(prefilling_press=prefill_press, decoding_press=decode_press)

    @torch.no_grad()
    def _generate(self, messages: list[dict]) -> tuple[str, list[int], int, int, str]:
        """Returns (completion_text, completion_ids, prompt_length, cache_seq_length, finish_reason).

        Sampling params mirror litellm's model_kwargs passthrough: with do_sample=False
        (default) the model decodes greedily and deterministically -- fine for measurement,
        but a model that falls into a repetition rut can never escape it. Set do_sample=True
        (optionally with temperature/top_p) to allow the model to break out, matching the
        randomness a hosted API uses by default.
        """
        clean_messages = [{"role": m["role"], "content": m["content"]} for m in messages]
        text = self.tokenizer.apply_chat_template(clean_messages, add_generation_prompt=True, tokenize=False)
        prompt_ids = self.tokenizer.encode(text, return_tensors="pt", add_special_tokens=False).to(self.device)
        prompt_length = prompt_ids.shape[1]

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": self.config.max_new_tokens,
            "do_sample": self.config.do_sample,
        }
        if self.config.do_sample:
            # Only pass these when sampling, otherwise transformers warns they're ignored.
            if self.config.temperature is not None:
                gen_kwargs["temperature"] = self.config.temperature
            if self.config.top_p is not None:
                gen_kwargs["top_p"] = self.config.top_p

        press = self._build_press(prompt_length)
        cache = DynamicCache()
        with press(self.model):
            outputs = self.model.generate(input_ids=prompt_ids, past_key_values=cache, **gen_kwargs)
        cache_seq_length = cache.get_seq_length()

        generated_ids = outputs[0, prompt_length:]
        completion_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Match litellm's real finish_reason instead of hardcoding "stop": if generation ran
        # out the token budget without emitting EOS, report "length" so the format-error
        # template can tell the model it was cut off rather than give a generic complaint.
        eos_ids = self.model.generation_config.eos_token_id
        if isinstance(eos_ids, int):
            eos_ids = [eos_ids]
        last_is_eos = len(generated_ids) > 0 and eos_ids is not None and generated_ids[-1].item() in eos_ids
        finish_reason = "length" if (len(generated_ids) >= self.config.max_new_tokens and not last_is_eos) else "stop"

        return completion_text, generated_ids.tolist(), prompt_length, cache_seq_length, finish_reason

    def _log_turn(
        self, messages: list[dict], completion_text: str, completion_ids: list[int],
        prompt_length: int, cache_seq_length: int, finish_reason: str,
    ):
        """Append one turn to `log_path` as JSON, for replay through entropy_analysis.py.

        `completion_ids` is the point of this log over the ordinary `.traj.json`: it records
        the exact tokens generated, so teacher-forcing scores those rather than a re-encode
        of the decoded text. `prompt_length` lets a replay assert it rebuilt the same prompt
        the agent actually ran on -- a mismatch means the chat template rendered differently
        and every number downstream would describe a prompt that never existed.
        """
        if not self.config.log_path:
            return
        record = {
            "turn": self._turn,
            "model_name": self.config.model_name,
            "press_name": self.config.press_name,
            "compression_ratio": self.config.compression_ratio,
            "n_sink": self.config.n_sink,
            "decoding_compression_interval": self.config.decoding_compression_interval,
            "messages": messages,
            "completion": completion_text,
            "completion_ids": completion_ids,
            "prompt_length": prompt_length,
            "cache_seq_length": cache_seq_length,
            "finish_reason": finish_reason,
            "timestamp": time.time(),
        }
        with open(self.config.log_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        self._turn += 1

    def query(self, messages: list[dict[str, str]], **kwargs) -> dict:
        from minisweagent.exceptions import FormatError
        from minisweagent.models import GLOBAL_MODEL_STATS
        from minisweagent.models.utils.actions_text import parse_regex_actions

        completion_text, completion_ids, prompt_length, cache_seq_length, finish_reason = self._generate(messages)
        self._log_turn(messages, completion_text, completion_ids, prompt_length, cache_seq_length, finish_reason)

        GLOBAL_MODEL_STATS.add(self.config.cost_per_call)
        self._cost += self.config.cost_per_call

        try:
            actions = parse_regex_actions(
                completion_text,
                action_regex=self.config.action_regex,
                format_error_template=self.config.format_error_template,
                template_kwargs={"finish_reason": finish_reason},
            )
        except FormatError as e:
            # Contract (see litellm_model.py): all query() implementations must persist
            # the response on FormatError so it isn't silently dropped from the transcript.
            e.messages[0]["extra"]["response"] = completion_text
            raise

        return {
            "role": "assistant",
            "content": completion_text,
            "extra": {"actions": actions, "cost": self.config.cost_per_call, "timestamp": time.time()},
        }

    def format_message(self, **kwargs) -> dict:
        from minisweagent.models.utils.openai_multimodal import expand_multimodal_content

        return expand_multimodal_content(kwargs, pattern=self.config.multimodal_regex)

    def format_observation_messages(
        self, message: dict, outputs: list[dict], template_vars: dict | None = None
    ) -> list[dict]:
        from minisweagent.models.utils.actions_text import format_observation_messages

        return format_observation_messages(
            outputs,
            observation_template=self.config.observation_template,
            template_vars=template_vars,
            multimodal_regex=self.config.multimodal_regex,
        )

    def get_template_vars(self, **kwargs) -> dict[str, Any]:
        return {"model_name": self.config.model_name}

    def serialize(self) -> dict:
        return {
            "info": {
                "config": {
                    "model": {
                        "model_name": self.config.model_name,
                        "press_name": self.config.press_name,
                        "compression_ratio": self.config.compression_ratio,
                        "n_sink": self.config.n_sink,
                        "decoding_compression_interval": self.config.decoding_compression_interval,
                    },
                    "model_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                },
            }
        }
