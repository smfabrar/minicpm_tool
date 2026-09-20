"""Adapter contract test double with explicit, configurable observations."""

from __future__ import annotations

import asyncio

from .contracts import AdapterCapabilities, Observation


class SimulatedAdapter:
    def __init__(self, capabilities: AdapterCapabilities | None = None) -> None:
        self._capabilities = capabilities or AdapterCapabilities(True, True, True, True, True)
        self._observations: asyncio.Queue[Observation] = asyncio.Queue()
        self._pending: dict[str, object] = {}

    def capabilities(self) -> AdapterCapabilities:
        return self._capabilities

    async def observe(self) -> Observation:
        return await self._observations.get()

    async def submit_context(self, event: object) -> None:
        if not self._capabilities.active_session_context:
            raise RuntimeError("backend has no active-session context input")
        event_id = str(getattr(event, "event_id"))
        self._pending[event_id] = event
        await self._observations.put(Observation("context_submitted", event_id=event_id))

    async def evaluate(self, event_id: str) -> None:
        if event_id not in self._pending:
            raise ValueError("unknown pending event")
        self._pending.pop(event_id)
        if self._capabilities.evaluation_ack:
            await self._observations.put(Observation("context_evaluated", event_id=event_id))

    async def cancel_context(self, event_id: str) -> bool:
        if not self._capabilities.pending_context_cancellation:
            return False
        return self._pending.pop(event_id, None) is not None

    async def emit(self, observation: Observation) -> None:
        await self._observations.put(observation)
