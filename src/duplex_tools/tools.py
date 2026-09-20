"""Allowlisted deterministic tools used by the first experiment."""

from __future__ import annotations

import asyncio
import ast
import operator
from dataclasses import dataclass
from typing import Any, Mapping, Protocol


@dataclass(frozen=True, slots=True)
class ToolResult:
    facts: str
    sources: tuple[str, ...] = ()


class AsyncTool(Protocol):
    async def __call__(self, arguments: Mapping[str, Any]) -> ToolResult: ...


TOOL_SCHEMAS: list[dict[str, Any]] = [
    {"type": "function", "function": {"name": "room_lookup", "description": "Find a room by exact local event or seminar name.", "parameters": {"type": "object", "properties": {"name": {"type": "string", "description": "Exact event name"}}, "required": ["name"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "calculator", "description": "Calculate a basic arithmetic expression.", "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "document_search", "description": "Search local documents for a requested topic.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"], "additionalProperties": False}}},
]


def validate_call(tool: str, arguments: Mapping[str, Any], schemas: list[dict[str, Any]] = TOOL_SCHEMAS) -> dict[str, str]:
    """Reject unknown names, extra keys, missing strings, and huge payloads."""
    choices = {entry["function"]["name"]: entry["function"]["parameters"] for entry in schemas}
    if tool not in choices:
        raise ValueError(f"tool is not allowlisted: {tool}")
    parameters = choices[tool]
    properties = parameters["properties"]
    if set(arguments) != set(parameters["required"]) or set(arguments) - set(properties):
        raise ValueError("arguments must contain exactly the required keys")
    result: dict[str, str] = {}
    for key, value in arguments.items():
        if not isinstance(value, str) or not value.strip() or len(value) > 240:
            raise ValueError(f"{key} must be a non-empty string of at most 240 characters")
        result[key] = value.strip()
    return result


class RoomLookup:
    def __init__(self, records: Mapping[str, str]) -> None:
        self.records = {key.casefold(): value for key, value in records.items()}

    async def __call__(self, arguments: Mapping[str, Any]) -> ToolResult:
        key = str(arguments["name"]).strip().casefold()
        if key not in self.records:
            raise LookupError(f"room not found for {key!r}")
        return ToolResult(self.records[key], (f"local://rooms/{key}",))


class SafeCalculator:
    _binary = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}
    _unary = {ast.UAdd: operator.pos, ast.USub: operator.neg}

    def _eval(self, node: ast.AST, depth: int = 0) -> float:
        if depth > 12:
            raise ValueError("expression too deep")
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            value = float(node.value)
        elif isinstance(node, ast.BinOp) and type(node.op) in self._binary:
            value = self._binary[type(node.op)](self._eval(node.left, depth + 1), self._eval(node.right, depth + 1))
        elif isinstance(node, ast.UnaryOp) and type(node.op) in self._unary:
            value = self._unary[type(node.op)](self._eval(node.operand, depth + 1))
        else:
            raise ValueError("unsupported expression")
        if not abs(value) < 1e12:
            raise ValueError("result out of range")
        return value

    async def __call__(self, arguments: Mapping[str, Any]) -> ToolResult:
        expression = str(arguments["expression"])
        if len(expression) > 120:
            raise ValueError("expression too long")
        value = self._eval(ast.parse(expression, mode="eval").body)
        return ToolResult(f"{expression} = {value:g}.")


class DocumentSearch:
    def __init__(self, documents: Mapping[str, str]) -> None:
        self.documents = dict(documents)

    async def __call__(self, arguments: Mapping[str, Any]) -> ToolResult:
        terms = set(str(arguments["query"]).casefold().split())
        matches = [(sum(term in (title + " " + body).casefold() for term in terms), title, body) for title, body in self.documents.items()]
        matches = sorted((item for item in matches if item[0]), reverse=True)[:2]
        if not matches:
            raise LookupError("no matching local document")
        return ToolResult(" ".join(f"{title}: {body[:240]}" for _, title, body in matches), tuple(f"local://docs/{title}" for _, title, _ in matches))


class LocalLookup:
    def __init__(self, records: Mapping[str, str]) -> None:
        self._records = dict(records)

    async def __call__(self, arguments: Mapping[str, Any]) -> ToolResult:
        await asyncio.sleep(float(arguments.get("delay_s", 0)))
        key = str(arguments.get("key", "")).strip()
        if key not in self._records:
            raise LookupError(f"no local record for {key!r}")
        return ToolResult(self._records[key], (f"local://{key}",))


class FixtureSearch:
    """Repeatable search substitute; live search remains a separate condition."""

    def __init__(self, responses: Mapping[str, ToolResult]) -> None:
        self._responses = dict(responses)

    async def __call__(self, arguments: Mapping[str, Any]) -> ToolResult:
        await asyncio.sleep(float(arguments.get("delay_s", 0)))
        query = str(arguments.get("query", "")).strip()
        if query not in self._responses:
            raise LookupError(f"no fixture search response for {query!r}")
        return self._responses[query]


@dataclass(frozen=True, slots=True)
class RoutedCall:
    tool: str
    arguments: dict[str, str]


def route_explicit(transcript: str) -> RoutedCall | None:
    """Small auditable routing baseline for stable transcript segments."""
    text = " ".join(transcript.strip().split())
    lowered = text.casefold()
    for prefix in ("search for ", "search "):
        if lowered.startswith(prefix):
            query = text[len(prefix) :].strip()
            return RoutedCall("fixture_search", {"query": query}) if query else None
    for prefix in ("look up ", "lookup "):
        if lowered.startswith(prefix):
            key = text[len(prefix) :].strip()
            return RoutedCall("local_lookup", {"key": key}) if key else None
    return None
