"""Granite's native tool output mapped into conservative caller actions."""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Callable, Mapping
from typing import Any

from .contracts import CallerAction, TranscriptSegment

_TOOL_CALL = re.compile(r"<tool_call>\s*(\{.*\})\s*(?:</tool_call>)?", re.DOTALL)


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
    """Use model likelihood for routing, then native generation for arguments."""

    ROUTE_SYSTEM = "Classify the latest user speech for a voice tool router. Return only the requested label letter."

    def __init__(self, generate: Callable[[list[dict[str, str]], list[dict[str, Any]]], str]) -> None:
        self.generate = generate

    async def decide(self, segment: TranscriptSegment, pending: Mapping[str, str], tools: list[dict[str, Any]]) -> CallerAction:
        schemas = {tool["function"]["name"]: tool for tool in tools}
        options = [
            ("A", "no action: ordinary conversation, thanks, or unfinished speech", "none"),
            ("B", "clarify: a tool request is clear but a required detail is missing", "clarify"),
        ]
        tool_labels = {"room_lookup": "C", "calculator": "D", "document_search": "E"}
        descriptions = {
            "room_lookup": "room_lookup: asks for the room or location of a named event",
            "calculator": "calculator: asks for explicit arithmetic",
            "document_search": "document_search: explicitly asks to search local documents",
        }
        for name in ("room_lookup", "calculator", "document_search"):
            if name in schemas:
                options.append((tool_labels[name], descriptions[name], name))
        if pending:
            options.extend((("F", "cancel: explicitly cancels a pending request", "cancel"),
                            ("G", "amend: explicitly corrects a pending request", "amend")))
        examples = (
            "Examples:\n"
            "Thank you, that helps. -> A\n"
            "What room is the -> A\n"
            "Find the room for my seminar. -> B\n"
            "Where is the chemistry colloquium? -> C\n"
            "What is 8 times 9? -> D\n"
            "Search the local documents for lab access. -> E\n"
        )
        menu = "\n".join(f"{label} = {description}" for label, description, _ in options)
        route_messages = [
            {"role": "system", "content": self.ROUTE_SYSTEM},
            {"role": "user", "content": f"{examples}\nLabels:\n{menu}\n\nLatest speech: {segment.text}\nLabel:"},
        ]
        label = await asyncio.to_thread(score_choice, self.generate, route_messages, [item[0] for item in options])
        selected = next(action for candidate, _, action in options if candidate == label)
        if selected == "none":
            return CallerAction("none", raw=f"route={label}")
        if selected == "clarify":
            return CallerAction("clarify", message="Please provide the missing detail.", raw=f"route={label}")
        if selected == "cancel":
            request_id = next(iter(pending)) if len(pending) == 1 else None
            return CallerAction("cancel" if request_id else "clarify", request_id=request_id,
                                message="Which pending request?" if request_id is None else "", raw=f"route={label}")
        if selected == "amend":
            return CallerAction("clarify", message="Please restate the corrected complete request.", raw=f"route={label}")

        argument_messages = [
            {"role": "system", "content": "Call the provided tool for the user speech. Copy only details present in that speech. Output only the native tool call."},
            {"role": "user", "content": segment.text},
        ]
        raw = await asyncio.to_thread(self.generate, argument_messages, [schemas[selected]])
        action = parse_granite_output(raw)
        if action.kind != "call" or action.tool != selected:
            return CallerAction("invalid", message="argument generation did not call the selected tool", raw=f"route={label}\n{raw}")
        return CallerAction("call", tool=action.tool, arguments=action.arguments, raw=f"route={label}\n{raw}")


def score_choice(generator: Any, messages: list[dict[str, str]], choices: list[str]) -> str:
    """Choose a closed label by conditional log likelihood, without free generation."""
    if not hasattr(generator, "model") or not hasattr(generator, "tokenizer") or not hasattr(generator, "torch"):
        raise TypeError("Granite generator must expose model, tokenizer, and torch for closed-label routing")
    torch = generator.torch
    tokenizer = generator.tokenizer
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    prompt_ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"].to(generator.model.device)
    scores: dict[str, float] = {}
    with torch.inference_mode():
        for choice in choices:
            choice_ids = tokenizer(choice, add_special_tokens=False, return_tensors="pt")["input_ids"].to(generator.model.device)
            full = torch.cat((prompt_ids, choice_ids), dim=1)
            logits = generator.model(input_ids=full, use_cache=False).logits
            start = prompt_ids.shape[1] - 1
            token_logits = logits[:, start:start + choice_ids.shape[1], :]
            log_probs = torch.log_softmax(token_logits, dim=-1)
            selected = log_probs.gather(-1, choice_ids.unsqueeze(-1)).squeeze(-1)
            scores[choice] = float(selected.mean().item())
    return max(scores, key=scores.get)


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
