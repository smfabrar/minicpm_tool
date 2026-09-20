"""Granite's native tool output mapped into conservative caller actions."""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Callable, Mapping
from typing import Any

from .contracts import CallerAction, TranscriptSegment

_TOOL_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def _arguments(value: Any) -> dict[str, Any] | None:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None
    return value if isinstance(value, dict) else None


def parse_granite_output(raw: str) -> CallerAction:
    stripped = raw.strip()
    for suffix in ("<|end_of_text|>", "<|end_of_turn|>"):
        if stripped.endswith(suffix):
            stripped = stripped.removesuffix(suffix).strip()
    match = _TOOL_CALL.fullmatch(stripped)
    if match:
        try:
            data = json.loads(match.group(1))
            args = _arguments(data.get("arguments")) if isinstance(data, dict) else None
            if isinstance(data, dict) and isinstance(data.get("name"), str) and args is not None:
                return CallerAction("call", tool=data["name"], arguments=args, raw=raw)
        except json.JSONDecodeError:
            pass
        return CallerAction("invalid", message="malformed tool call", raw=raw)
    if "<tool_call>" in stripped:
        return CallerAction("invalid", message="tool call contains extra text or multiple calls", raw=raw)
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        return CallerAction("invalid", message="unstructured caller output", raw=raw)
    if not isinstance(data, dict) or data.get("action") not in {"none", "clarify", "cancel", "amend"}:
        return CallerAction("invalid", message="unknown caller action", raw=raw)
    kind = data["action"]
    args = _arguments(data.get("arguments", {}))
    if kind == "amend" and (not isinstance(data.get("tool"), str) or args is None):
        return CallerAction("invalid", message="malformed amendment", raw=raw)
    return CallerAction(kind, tool=data.get("tool"), arguments=args or {}, request_id=data.get("request_id"), message=str(data.get("message", "")), raw=raw)


class GraniteToolCaller:
    """A generator is loaded once and called in a worker thread per committed turn."""

    SYSTEM = (
        "You route ONLY the last user utterance. Do not answer it. Select at most one allowed tool. "
        "Use room_lookup only for the room or location of a NAMED event. "
        "Use calculator only for an explicit arithmetic question. "
        "Use document_search only when the user explicitly asks to search local documents. "
        "Never substitute document_search for a room lookup. "
        "For a complete request with all required details, emit one native <tool_call> JSON object. "
        "For ordinary conversation, thanks, or unfinished speech emit exactly {\"action\":\"none\"}. "
        "For a complete request missing a required detail emit JSON with action clarify and a short message. "
        "For an explicit correction to a pending request emit JSON with action amend, request_id, tool, arguments. "
        "For an explicit cancellation emit JSON with action cancel and request_id. "
        "Never invent a request ID or missing tool argument. Output only the action."
    )

    def __init__(self, generate: Callable[[list[dict[str, str]], list[dict[str, Any]]], str]) -> None:
        self.generate = generate

    async def decide(self, segment: TranscriptSegment, pending: Mapping[str, str], tools: list[dict[str, Any]]) -> CallerAction:
        available = {tool["function"]["name"] for tool in tools}
        messages = [{"role": "system", "content": self.SYSTEM + " Pending request metadata: " + json.dumps(dict(pending), sort_keys=True)}]
        examples = [
            ("Thanks, that helps.", '{"action":"none"}'),
            ("What room is the", '{"action":"none"}'),
            ("Find the room for my seminar.", '{"action":"clarify","message":"Which seminar?"}'),
        ]
        if "room_lookup" in available:
            examples.append(("Where is the chemistry colloquium?", '<tool_call>{"name":"room_lookup","arguments":{"name":"chemistry colloquium"}}</tool_call>'))
        if "calculator" in available:
            examples.append(("What is 8 times 9?", '<tool_call>{"name":"calculator","arguments":{"expression":"8 * 9"}}</tool_call>'))
        if "document_search" in available:
            examples.append(("Search the local documents for lab access.", '<tool_call>{"name":"document_search","arguments":{"query":"lab access"}}</tool_call>'))
        for user, assistant in examples:
            messages.extend(({"role": "user", "content": user}, {"role": "assistant", "content": assistant}))
        messages.append({"role": "user", "content": segment.text})
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
