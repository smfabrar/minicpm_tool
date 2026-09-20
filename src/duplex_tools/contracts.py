"""Types shared by transcript, caller, controller, and duplex adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal, Mapping, Protocol


@dataclass(frozen=True, slots=True)
class TranscriptSegment:
    segment_id: str
    revision: int
    timestamp: datetime
    text: str
    committed: bool

    def __post_init__(self) -> None:
        if not self.segment_id or self.revision < 1 or self.timestamp.tzinfo is None:
            raise ValueError("segment requires an id, positive revision, and timezone-aware timestamp")


@dataclass(frozen=True, slots=True)
class CallerAction:
    kind: Literal["call", "clarify", "amend", "cancel", "none", "invalid"]
    tool: str | None = None
    arguments: Mapping[str, Any] = field(default_factory=dict)
    request_id: str | None = None
    message: str = ""
    raw: str = ""


@dataclass(frozen=True, slots=True)
class AdapterCapabilities:
    active_session_context: bool
    evaluation_ack: bool
    listening_state: bool
    incremental_output: bool
    pending_context_cancellation: bool


@dataclass(frozen=True, slots=True)
class Observation:
    kind: Literal["assistant_text", "speaking", "listening", "context_submitted", "context_evaluated", "context_rejected", "audio_emitted", "playback_completed", "session_end"]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    event_id: str | None = None
    value: str | bool | None = None


class TranscriptProvider(Protocol):
    async def next_segment(self) -> TranscriptSegment: ...


class ToolCaller(Protocol):
    async def decide(self, segment: TranscriptSegment, pending: Mapping[str, str], tools: list[dict[str, Any]]) -> CallerAction: ...


class DuplexAdapter(Protocol):
    def capabilities(self) -> AdapterCapabilities: ...
    async def observe(self) -> Observation: ...
    async def submit_context(self, event: Any) -> None: ...
    async def cancel_context(self, event_id: str) -> bool: ...
