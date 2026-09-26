from __future__ import annotations

import asyncio
import unittest

from duplex_tools.context_events import Boundary
from duplex_tools.controller import ContextController
from duplex_tools.tools import LocalLookup


class ControllerTests(unittest.IsolatedAsyncioTestCase):
    async def test_wait_all_removes_an_already_finished_task(self) -> None:
        controller = ContextController({})
        finished = asyncio.create_task(asyncio.sleep(0))
        await finished
        # Reproduce the interval where a task is done but its scheduled discard
        # callback has not removed it from the controller set.
        controller._tasks.add(finished)
        await asyncio.wait_for(controller.wait_all(), timeout=0.1)
        self.assertFalse(controller._tasks)

    async def test_tool_work_does_not_block_caller_and_result_is_injected(self) -> None:
        controller = ContextController(
            {"local_lookup": LocalLookup({"room": "The room is B742."})}
        )
        version, started = controller.submit_tool(
            "request", "local_lookup", {"key": "room", "delay_s": 0.02}
        )
        self.assertTrue(started)
        self.assertEqual(version, 1)
        await asyncio.sleep(0)
        self.assertIsNone(controller.reserve_injection(Boundary.UNIT))
        await controller.wait_all()
        result = controller.reserve_injection(Boundary.UNIT)
        assert result
        controller.commit_injection(result.event_id, counter=4)
        self.assertEqual(result.facts, "The room is B742.")

    async def test_late_old_result_is_stale(self) -> None:
        controller = ContextController(
            {
                "local_lookup": LocalLookup(
                    {"old": "Old result.", "new": "New result."}
                )
            }
        )
        controller.submit_tool(
            "request", "local_lookup", {"key": "old", "delay_s": 0.04}
        )
        await asyncio.sleep(0.005)
        controller.submit_tool(
            "request", "local_lookup", {"key": "new", "delay_s": 0.005}
        )
        await controller.wait_all()
        result = controller.reserve_injection(Boundary.UNIT)
        assert result
        self.assertEqual(result.facts, "New result.")
        stale = [
            record
            for record in controller.log.records
            if record["kind"] == "tool_completed" and record["decision"] == "stale"
        ]
        self.assertEqual(len(stale), 1)

    async def test_tool_failure_is_available_as_status_context(self) -> None:
        controller = ContextController({"local_lookup": LocalLookup({})})
        controller.submit_tool("request", "local_lookup", {"key": "missing"})
        await controller.wait_all()
        event = controller.reserve_injection(Boundary.UNIT)
        assert event
        self.assertEqual(event.status, "failed")
        self.assertIn("lookup failed", event.injection_text())


if __name__ == "__main__":
    unittest.main()
