from __future__ import annotations

import unittest
from datetime import timedelta

from duplex_tools.context_events import (
    Boundary,
    ContextEvent,
    ContextScheduler,
    EventDecision,
    InjectionPolicy,
    RequestRegistry,
    utc_now,
)


def event(request_id: str, version: int, event_id: str = "event-1") -> ContextEvent:
    now = utc_now()
    return ContextEvent(
        event_id=event_id,
        request_id=request_id,
        request_version=version,
        tool="local_lookup",
        status="success",
        created_at=now,
        completed_at=now,
        facts="The room is B742.",
    )


class RegistryTests(unittest.TestCase):
    def test_correction_rejects_old_result(self) -> None:
        registry = RequestRegistry()
        first, _ = registry.submit("request", "stockholm")
        second, _ = registry.submit("request", "uppsala")
        self.assertEqual(second.version, first.version + 1)
        self.assertEqual(registry.accept(event("request", first.version)), EventDecision.STALE)
        self.assertEqual(
            registry.accept(event("request", second.version, "event-2")),
            EventDecision.ACCEPTED,
        )

    def test_duplicate_and_cancelled_results_are_rejected(self) -> None:
        registry = RequestRegistry()
        record, _ = registry.submit("request", "lookup")
        result = event("request", record.version)
        self.assertEqual(registry.accept(result), EventDecision.ACCEPTED)
        self.assertEqual(registry.accept(result), EventDecision.DUPLICATE)
        record2, _ = registry.submit("request", "changed")
        registry.cancel("request")
        self.assertEqual(
            registry.accept(event("request", record2.version, "event-2")),
            EventDecision.CANCELLED,
        )

    def test_expired_result_is_rejected(self) -> None:
        registry = RequestRegistry()
        record, _ = registry.submit("request", "lookup")
        now = utc_now()
        expired = ContextEvent(
            event_id="expired",
            request_id="request",
            request_version=record.version,
            tool="lookup",
            status="success",
            created_at=now - timedelta(seconds=2),
            completed_at=now - timedelta(seconds=1),
            expires_at=now - timedelta(milliseconds=1),
            facts="Old fact.",
        )
        self.assertEqual(registry.accept(expired), EventDecision.EXPIRED)


class SchedulerTests(unittest.TestCase):
    def test_listening_policy_waits_for_listening_boundary(self) -> None:
        registry = RequestRegistry()
        record, _ = registry.submit("request", "lookup")
        scheduler = ContextScheduler(registry, policy=InjectionPolicy.NEXT_LISTENING)
        scheduler.enqueue(event("request", record.version))
        self.assertIsNone(scheduler.reserve(Boundary.UNIT, model_idle=True))
        reserved = scheduler.reserve(Boundary.LISTENING, model_idle=True)
        self.assertIsNotNone(reserved)

    def test_failed_prefill_can_release_reservation(self) -> None:
        registry = RequestRegistry()
        record, _ = registry.submit("request", "lookup")
        scheduler = ContextScheduler(registry)
        scheduler.enqueue(event("request", record.version))
        reserved = scheduler.reserve(Boundary.UNIT, model_idle=True)
        assert reserved
        scheduler.release(reserved.event_id)
        self.assertEqual(scheduler.pending()[0].event_id, reserved.event_id)

    def test_queue_overflow_drops_oldest(self) -> None:
        registry = RequestRegistry()
        scheduler = ContextScheduler(registry, max_queue=1)
        first, _ = registry.submit("one", "a")
        second, _ = registry.submit("two", "b")
        scheduler.enqueue(event("one", first.version, "old"))
        result = scheduler.enqueue(event("two", second.version, "new"))
        self.assertEqual(result.decision, EventDecision.OVERFLOW)
        self.assertEqual(result.dropped_event_id, "old")
        self.assertEqual(scheduler.pending()[0].event_id, "new")


if __name__ == "__main__":
    unittest.main()
