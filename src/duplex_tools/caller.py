"""Granite's native tool output mapped into conservative caller actions."""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Callable, Mapping
from typing import Any

from .contracts import CallerAction, TranscriptSegment

_TOOL_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def parse_granite_output(raw: str) -> CallerAction:
    matches = _TOOL_CALL.findall(raw)
    if len(matches) > 1:
        return CallerAction("invalid", message="multiple tool calls are unsupported", raw=raw)
    if matches:
        try:
            data = json.loads(matches[0])
            if isinstance(data, dict) and isinstance(data.get("name"), str) and isinstance(data.get("arguments"), dict):
                return CallerAction("call", tool=data["name"], arguments=data["arguments"], raw=raw)
        except json.JSONDecodeError:
            pass
        return CallerAction("invalid", message="malformed tool call", raw=raw)
    stripped = raw.strip()
    if stripped.endswith("<|end_of_text|>"):
        stripped = stripped.removesuffix("<|end_of_text|>").strip()
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        return CallerAction("invalid", message="unstructured caller output", raw=raw)
    if not isinstance(data, dict) or data.get("action") not in {"none", "clarify", "cancel", "amend"}:
        return CallerAction("invalid", message="unknown caller action", raw=raw)
    kind = data["action"]
    if kind == "amend" and (not isinstance(data.get("tool"), str) or not isinstance(data.get("arguments"), dict)):
        return CallerAction("invalid", message="malformed amendment", raw=raw)
    return CallerAction(kind, tool=data.get("tool"), arguments=data.get("arguments", {}), request_id=data.get("request_id"), message=str(data.get("message", "")), raw=raw)


class GraniteToolCaller:
    """A generator is loaded once and called in a worker thread per committed turn."""

    SYSTEM = (
        "Select at most one allowed tool for the latest committed user speech. "
        "Use a native tool call for a new complete request. "
        "For a correction to a pending request return JSON with action amend, request_id, tool, arguments. "
        "For a cancellation return JSON with action cancel and request_id. "
        "If required details are missing return JSON with action clarify and message. "
        "For ordinary chat or incomplete speech return {\"action\":\"none\"}. "
        "Never guess missing arguments. Pending request IDs are controller metadata."
    )

    def __init__(self, generate: Callable[[list[dict[str, str]], list[dict[str, Any]]], str]) -> None:
        self.generate = generate

    async def decide(self, segment: TranscriptSegment, pending: Mapping[str, str], tools: list[dict[str, Any]]) -> CallerAction:
        messages = [
            {"role": "system", "content": self.SYSTEM},
            {"role": "user", "content": f"Pending requests: {json.dumps(dict(pending), sort_keys=True)}\nLatest speech: {segment.text}"},
        ]
        return parse_granite_output(await asyncio.to_thread(self.generate, messages, tools))


class TransformersGraniteGenerator:
    """Optional Kaggle backend; import/load occurs only when constructed."""

    def __init__(self, model_id: str = "ibm-granite/granite-4.0-350m", device: str = "cpu") -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
        self.model.eval()

    def __call__(self, messages: list[dict[str, str]], tools: list[dict[str, Any]]) -> str:
        tokens = self.tokenizer.apply_chat_template(
            messages, tools=tools, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        ).to(self.model.device)
        with self.torch.inference_mode():
            output = self.model.generate(**tokens, do_sample=False, max_new_tokens=128, pad_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.decode(output[0][tokens["input_ids"].shape[-1]:], skip_special_tokens=False).strip()
