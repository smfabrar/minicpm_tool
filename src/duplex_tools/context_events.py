"""Context-event schema, request versions, and boundary-aware scheduling."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Iterable


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class RequestStatus(StrEnum):
    RUNNING = "running"
    QUEUED = "queued"
    INJECTED = "injected"
    ADDRESSED = "addressed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class EventDecision(StrEnum):
    ACCEPTED = "accepted"
    DUPLICATE = "duplicate"
    STALE = "stale"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    UNKNOWN_REQUEST = "unknown_request"
    FAILED = "failed"
    OVERFLOW = "overflow"


class Boundary(StrEnum):
    UNIT = "unit"
    LISTENING = "listening"


class InjectionPolicy(StrEnum):
    NEXT_UNIT = "next_unit"
    NEXT_LISTENING = "next_listening"


@dataclass(frozen=True, slots=True)
class ContextEvent:
    event_id: str
    request_id: str
    request_version: int
    tool: str
    status: str
    created_at: datetime
    completed_at: datetime
    facts: str = ""
    sources: tuple[str, ...] = ()
    expires_at: datetime | None = None
    supersedes: str | None = None

    def __post_init__(self) -> None:
        if not self.event_id.strip() or not self.request_id.strip():
            raise ValueError("event_id and request_id must be non-empty")
        if self.request_version < 1:
            raise ValueError("request_version must be positive")
        if self.status not in {"success", "failed"}:
            raise ValueError("status must be 'success' or 'failed'")
        for name in ("created_at", "completed_at", "expires_at"):
            value = getattr(self, name)
            if value is not None and value.tzinfo is None:
                raise ValueError(f"{name} must be timezone-aware")
        if self.completed_at < self.created_at:
            raise ValueError("completed_at cannot precede created_at")
        if self.status == "success" and not self.facts.strip():
            raise ValueError("successful events require facts")

    def is_expired(self, now: datetime | None = None) -> bool:
        return self.expires_at is not None and self.expires_at <= (now or utc_now())

    def injection_text(self, max_chars: int = 640) -> str:
        """Return a compact literal payload without inventing special tokens."""
        if max_chars < 80:
            raise ValueError("max_chars must be at least 80")
        prefix = "External information for the current request: "
        facts = " ".join(self.facts.split()) if self.status == "success" else "The requested lookup failed. Say that the information is unavailable."
        room = max_chars - len(prefix)
        if len(facts) > room:
            facts = facts[: max(0, room - 1)].rstrip() + "…"
        return prefix + facts

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key in ("created_at", "completed_at", "expires_at"):
            value = data[key]
            data[key] = value.isoformat() if value else None
        data["sources"] = list(self.sources)
        return data


@dataclass(slots=True)
class RequestRecord:
    request_id: str
    version: int
    fingerprint: str
    status: RequestStatus
    updated_at: datetime = field(default_factory=utc_now)


class RequestRegistry:
    """Owns request versions so late tool results cannot become current again."""

    def __init__(self) -> None:
        self._records: dict[str, RequestRecord] = {}
        self._accepted_event_ids: set[str] = set()

    def submit(self, request_id: str, fingerprint: str) -> tuple[RequestRecord, bool]:
        request_id = request_id.strip()
        fingerprint = fingerprint.strip()
        if not request_id or not fingerprint:
            raise ValueError("request_id and fingerprint must be non-empty")
        current = self._records.get(request_id)
        if current and current.fingerprint == fingerprint and current.status != RequestStatus.CANCELLED:
            return current, False
        version = 1 if current is None else current.version + 1
        record = RequestRecord(request_id, version, fingerprint, RequestStatus.RUNNING)
        self._records[request_id] = record
        return record, True

    def cancel(self, request_id: str) -> bool:
        record = self._records.get(request_id)
        if record is None:
            return False
        record.status = RequestStatus.CANCELLED
        record.updated_at = utc_now()
        return True

    def assess(self, event: ContextEvent, *, check_duplicate: bool = True) -> EventDecision:
        record = self._records.get(event.request_id)
        if record is None:
            return EventDecision.UNKNOWN_REQUEST
        if check_duplicate and event.event_id in self._accepted_event_ids:
            return EventDecision.DUPLICATE
        if event.request_version != record.version:
            return EventDecision.STALE
        if record.status == RequestStatus.CANCELLED:
            return EventDecision.CANCELLED
        if event.is_expired():
            return EventDecision.EXPIRED
        return EventDecision.ACCEPTED

    def accept(self, event: ContextEvent) -> EventDecision:
        decision = self.assess(event)
        if decision == EventDecision.ACCEPTED:
            self._accepted_event_ids.add(event.event_id)
            record = self._records[event.request_id]
            record.status = RequestStatus.FAILED if event.status == "failed" else RequestStatus.QUEUED
            record.updated_at = utc_now()
        return decision

    def mark_injected(self, event: ContextEvent) -> bool:
        if self.assess(event, check_duplicate=False) != EventDecision.ACCEPTED:
            return False
        record = self._records[event.request_id]
        record.status = RequestStatus.INJECTED
        record.updated_at = utc_now()
        return True

    def mark_addressed(self, request_id: str, version: int) -> bool:
        record = self._records.get(request_id)
        if record is None or record.version != version:
            return False
        record.status = RequestStatus.ADDRESSED
        record.updated_at = utc_now()
        return True

    def get(self, request_id: str) -> RequestRecord | None:
        return self._records.get(request_id)


@dataclass(frozen=True, slots=True)
class EnqueueResult:
    decision: EventDecision
    dropped_event_id: str | None = None


class ContextScheduler:
    """A bounded FIFO consumed only by the speech-model inference owner."""

    def __init__(
        self,
        registry: RequestRegistry,
        *,
        policy: InjectionPolicy = InjectionPolicy.NEXT_UNIT,
        max_queue: int = 8,
    ) -> None:
        if max_queue < 1:
            raise ValueError("max_queue must be positive")
        self.registry = registry
        self.policy = policy
        self.max_queue = max_queue
        self._queue: deque[ContextEvent] = deque()
        self._reserved: ContextEvent | None = None

    def enqueue(self, event: ContextEvent) -> EnqueueResult:
        decision = self.registry.accept(event)
        if decision != EventDecision.ACCEPTED:
            return EnqueueResult(decision)
        dropped = None
        if len(self._queue) >= self.max_queue:
            dropped = self._queue.popleft().event_id
        self._queue.append(event)
        return EnqueueResult(
            EventDecision.OVERFLOW if dropped else EventDecision.ACCEPTED,
            dropped,
        )

    def _boundary_allowed(self, boundary: Boundary) -> bool:
        if self.policy == InjectionPolicy.NEXT_UNIT:
            return boundary in {Boundary.UNIT, Boundary.LISTENING}
        return boundary == Boundary.LISTENING

    def reserve(self, boundary: Boundary, *, model_idle: bool) -> ContextEvent | None:
        if self._reserved is not None or not model_idle or not self._boundary_allowed(boundary):
            return None
        while self._queue:
            event = self._queue.popleft()
            if self.registry.assess(event, check_duplicate=False) == EventDecision.ACCEPTED:
                self._reserved = event
                return event
        return None

    def commit(self, event_id: str) -> ContextEvent:
        if self._reserved is None or self._reserved.event_id != event_id:
            raise ValueError("event is not reserved")
        event = self._reserved
        if not self.registry.mark_injected(event):
            self._reserved = None
            raise ValueError("reserved event became stale before injection")
        self._reserved = None
        return event

    def release(self, event_id: str) -> None:
        if self._reserved is None or self._reserved.event_id != event_id:
            raise ValueError("event is not reserved")
        self._queue.appendleft(self._reserved)
        self._reserved = None

    def withdraw(self, event_id: str) -> bool:
        if self._reserved is not None and self._reserved.event_id == event_id:
            self._reserved = None
            return True
        before = len(self._queue)
        self._queue = deque(event for event in self._queue if event.event_id != event_id)
        return len(self._queue) != before

    def pending(self) -> tuple[ContextEvent, ...]:
        return tuple(self._queue)

    def extend(self, events: Iterable[ContextEvent]) -> list[EnqueueResult]:
        return [self.enqueue(event) for event in events]
