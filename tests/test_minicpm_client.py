from __future__ import annotations

import json
import unittest

from duplex_tools.context_events import Boundary
from duplex_tools.controller import ContextController
from duplex_tools.minicpm_client import MiniCPMStreamSession
from duplex_tools.tools import LocalLookup


class FakeClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    def post(self, path: str, payload: dict[str, object]) -> tuple[str, str]:
        self.calls.append((path, payload))
        response = {"success": True}
        if path.endswith("omni_init"):
            response["next_cnt"] = 1
        return "application/json", json.dumps(response)


class LostResponseClient(FakeClient):
    def post(self, path: str, payload: dict[str, object]) -> tuple[str, str]:
        if path.endswith("prefill"):
            raise TimeoutError("response lost after request")
        return super().post(path, payload)


class MiniCPMClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_context_is_added_once_to_next_serial_prefill(self) -> None:
        controller = ContextController(
            {"local_lookup": LocalLookup({"room": "The room is B742."})}
        )
        controller.submit_tool("request", "local_lookup", {"key": "room"})
        await controller.wait_all()
        client = FakeClient()
        session = MiniCPMStreamSession(client, controller)  # type: ignore[arg-type]
        await session.initialize(output_dir="output")
        await session.prefill(audio_path="one.wav", counter=1, boundary=Boundary.UNIT)
        await session.prefill(audio_path="two.wav", counter=2, boundary=Boundary.UNIT)
        self.assertEqual(client.calls[0][1]["media_type"], 1)
        self.assertIn("text", client.calls[1][1])
        self.assertNotIn("text", client.calls[2][1])
        self.assertIn("B742", str(client.calls[1][1]["text"]))

    async def test_counter_cannot_reset_within_session(self) -> None:
        controller = ContextController({})
        session = MiniCPMStreamSession(FakeClient(), controller)  # type: ignore[arg-type]
        await session.initialize(output_dir="output")
        with self.assertRaises(ValueError):
            await session.prefill(audio_path="one.wav", counter=2, boundary=Boundary.UNIT)

    async def test_session_cannot_be_reinitialized(self) -> None:
        session = MiniCPMStreamSession(FakeClient(), ContextController({}))  # type: ignore[arg-type]
        await session.initialize(output_dir="output")
        with self.assertRaises(ValueError):
            await session.initialize(output_dir="output")

    async def test_uncertain_prefill_is_not_retried(self) -> None:
        controller = ContextController({"local_lookup": LocalLookup({"room": "B742"})})
        controller.submit_tool("request", "local_lookup", {"key": "room"})
        await controller.wait_all()
        session = MiniCPMStreamSession(LostResponseClient(), controller)  # type: ignore[arg-type]
        await session.initialize(output_dir="output")
        with self.assertRaises(TimeoutError):
            await session.prefill(audio_path="one.wav", counter=1, boundary=Boundary.UNIT)
        self.assertTrue(any(r["kind"] == "context_submission_unknown" for r in controller.log.records))
        with self.assertRaises(RuntimeError):
            await session.prefill(audio_path="one.wav", counter=1, boundary=Boundary.UNIT)


if __name__ == "__main__":
    unittest.main()
