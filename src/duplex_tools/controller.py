"""Async tool execution separated from serialized model-state mutation."""

from __future__ import annotations

import asyncio
import hashlib
import json
import uuid
from collections.abc import Mapping
from datetime import timedelta
from pathlib import Path
from typing import Any

from .context_events import (
    Boundary,
    ContextEvent,
    ContextScheduler,
    InjectionPolicy,
    RequestRegistry,
    utc_now,
)
from .tools import AsyncTool


class JsonlEventLog:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self.records: list[dict[str, Any]] = []
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, kind: str, **fields: Any) -> None:
        record = {"at": utc_now().isoformat(), "kind": kind, **fields}
        self.records.append(record)
        if self.path:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")


class ContextController:
    def __init__(
        self,
        tools: Mapping[str, AsyncTool],
        *,
        policy: InjectionPolicy = InjectionPolicy.NEXT_UNIT,
        max_queue: int = 8,
        tool_timeout_s: float = 8.0,
        event_ttl_s: float | None = 30.0,
        log: JsonlEventLog | None = None,
    ) -> None:
        self.tools = dict(tools)
        self.registry = RequestRegistry()
        self.scheduler = ContextScheduler(self.registry, policy=policy, max_queue=max_queue)
        self.tool_timeout_s = tool_timeout_s
        self.event_ttl_s = event_ttl_s
        self.log = log or JsonlEventLog()
        self._tasks: set[asyncio.Task[None]] = set()

    @staticmethod
    def _fingerprint(tool: str, arguments: Mapping[str, Any]) -> str:
        raw = json.dumps([tool, dict(arguments)], sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode()).hexdigest()

    def submit_tool(
        self,
        request_id: str,
        tool: str,
        arguments: Mapping[str, Any],
    ) -> tuple[int, bool]:
        if tool not in self.tools:
            raise ValueError(f"tool is not allowlisted: {tool}")
        record, started = self.registry.submit(
            request_id, self._fingerprint(tool, arguments)
        )
        if not started:
            self.log.write("tool_deduplicated", request_id=request_id, version=record.version)
            return record.version, False
        self.log.write(
            "tool_started",
            request_id=request_id,
            version=record.version,
            tool=tool,
            arguments=dict(arguments),
        )
        task = asyncio.create_task(
            self._execute(request_id, record.version, tool, dict(arguments))
        )
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return record.version, True

    async def _execute(
        self,
        request_id: str,
        version: int,
        tool_name: str,
        arguments: Mapping[str, Any],
    ) -> None:
        created_at = utc_now()
        status = "success"
        facts = ""
        sources: tuple[str, ...] = ()
        error = None
        try:
            result = await asyncio.wait_for(
                self.tools[tool_name](arguments), timeout=self.tool_timeout_s
            )
            facts, sources = result.facts, result.sources
        except Exception as exc:
            status = "failed"
            error = f"{type(exc).__name__}: {exc}"
        completed_at = utc_now()
        expires_at = (
            completed_at + timedelta(seconds=self.event_ttl_s)
            if self.event_ttl_s is not None
            else None
        )
        event = ContextEvent(
            event_id=str(uuid.uuid4()),
            request_id=request_id,
            request_version=version,
            tool=tool_name,
            status=status,
            created_at=created_at,
            completed_at=completed_at,
            expires_at=expires_at,
            facts=facts,
            sources=sources,
        )
        result = self.scheduler.enqueue(event)
        self.log.write(
            "tool_completed",
            event=event.to_dict(),
            decision=result.decision,
            dropped_event_id=result.dropped_event_id,
            error=error,
        )

    def cancel(self, request_id: str) -> bool:
        cancelled = self.registry.cancel(request_id)
        self.log.write("request_cancelled", request_id=request_id, found=cancelled)
        return cancelled

    async def wait_all(self) -> None:
        while self._tasks:
            await asyncio.gather(*tuple(self._tasks), return_exceptions=True)

    def reserve_injection(self, boundary: Boundary, *, model_idle: bool = True) -> ContextEvent | None:
        event = self.scheduler.reserve(boundary, model_idle=model_idle)
        if event:
            self.log.write(
                "injection_reserved",
                event_id=event.event_id,
                request_id=event.request_id,
                version=event.request_version,
                boundary=boundary,
            )
        return event

    def commit_injection(self, event_id: str, *, counter: int) -> ContextEvent:
        event = self.scheduler.commit(event_id)
        self.log.write(
            "context_injected",
            event_id=event.event_id,
            request_id=event.request_id,
            version=event.request_version,
            counter=counter,
        )
        return event

    def release_injection(self, event_id: str, *, reason: str) -> None:
        self.scheduler.release(event_id)
        self.log.write("injection_released", event_id=event_id, reason=reason)
