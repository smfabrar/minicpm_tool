import asyncio
import json
import tempfile
import threading
import time
import unittest
from array import array
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from duplex_tools.caller import ChoiceScores, GraniteToolCaller
from duplex_tools.contracts import CallerAction, TranscriptSegment
from duplex_tools.controller import ContextController
from duplex_tools.conversation import ConversationRouter
from duplex_tools.live_duplex import LiveDuplexExperiment, RestartableLiveExperiment, SpeechSegmenter, write_wav
from duplex_tools.official_realtime import OfficialRealtimeExperiment, client_unit_for_tool_ack, native_response_unit
from duplex_tools.tools import RoomLookup


def pcm(amplitude: int, seconds: float) -> bytes:
    return array("h", [amplitude] * round(16000 * seconds)).tobytes()


class SegmenterTests(unittest.TestCase):
    def test_commits_only_after_speech_and_silence(self):
        segmenter = SpeechSegmenter(rms_threshold=0.02, pause_s=0.6)
        self.assertEqual(segmenter.push(pcm(0, 0.3)), [])
        self.assertEqual(segmenter.push(pcm(4000, 0.4)), [])
        completed = segmenter.push(pcm(0, 0.7))
        self.assertEqual(len(completed), 1)
        self.assertGreater(len(completed[0]), len(pcm(4000, 0.4)))
        self.assertIsNone(segmenter.flush())


class RestartTests(unittest.TestCase):
    def test_completed_trial_gets_fresh_session_and_active_start_is_idempotent(self):
        created = []

        class Trial:
            def __init__(self, number):
                self.number = number
                self.status = "ready"
                self.closed = False

            def start(self):
                self.status = "running"
                return f"session-{self.number}"

            def snapshot(self):
                return {"status": self.status, "session_id": f"session-{self.number}"}

            def stop(self, _session_id=None):
                self.status = "complete"
                return "complete"

            def close(self):
                self.closed = True

        def factory():
            trial = Trial(len(created) + 1)
            created.append(trial)
            return trial

        manager = RestartableLiveExperiment(factory)
        self.assertEqual(manager.start(), "session-1")
        self.assertEqual(manager.start(), "session-1")
        self.assertEqual(len(created), 1)
        manager.stop("session-1")
        self.assertEqual(manager.start(), "session-2")
        self.assertEqual(len(created), 2)
        manager.close()
        self.assertTrue(all(trial.closed for trial in created))


class NativeProtocolTests(unittest.TestCase):
    def test_pinned_backend_unit_numbering_matches_exported_session(self):
        self.assertEqual(client_unit_for_tool_ack(19), 18)
        self.assertEqual(client_unit_for_tool_ack(192), 191)
        self.assertEqual(native_response_unit("session_resp_203"), 203)
        self.assertIsNone(native_response_unit("unexpected-format"))

    def test_playback_waits_for_real_speech_without_inserting_silence(self):
        with tempfile.TemporaryDirectory() as directory:
            experiment = LiveDuplexExperiment(None, None, None, Path(directory))
            experiment._state = {"session_id": "trial", "status": "running"}
            experiment.directory = Path(directory)
            experiment._outbound_windows = deque()
            speech = experiment.directory / "speech.wav"
            write_wav(speech, pcm(4000, 0.2))
            result = []
            thread = threading.Thread(target=lambda: result.append(
                experiment.next_playback("trial", timeout_s=0.01)))
            try:
                thread.start()
                time.sleep(0.05)
                self.assertTrue(thread.is_alive())
                experiment._playback.put(speech)
                thread.join(timeout=1)
                self.assertEqual(result, [speech])
            finally:
                experiment.close()


class NativeDrainTests(unittest.IsolatedAsyncioTestCase):
    async def test_backend_ack_events_match_client_tool_results(self):
        class FakeSocket:
            def __aiter__(self):
                self.events = iter([
                    {"type": "tool_context.evaluated", "unit": 19, "ok": True},
                    {"type": "tool_context.evaluated", "unit": 192, "ok": True},
                    {"type": "response.output.delta", "kind": "listen",
                     "response_id": "session_resp_203"},
                    {"type": "session.closed"},
                ])
                return self

            async def __anext__(self):
                try:
                    return json.dumps(next(self.events))
                except StopIteration:
                    raise StopAsyncIteration

        class Receiver:
            _ws = FakeSocket()
            _injected = {18: "room", 191: "calculator"}
            _acknowledged = set()
            _last_native_response_unit = 0
            _max_native_pending_units = 0
            _sent_units = 203
            _error = None
            _closed_event = asyncio.Event()
            records = []

            def _log(self, kind, **fields):
                self.records.append((kind, fields))

        receiver = Receiver()
        with patch.dict("sys.modules", {"numpy": ModuleType("numpy")}):
            await OfficialRealtimeExperiment._read_realtime(receiver)
        self.assertEqual(receiver._acknowledged, {"room", "calculator"})
        self.assertEqual(receiver._last_native_response_unit, 203)
        self.assertEqual(len([kind for kind, _ in receiver.records
                              if kind == "context_evaluated"]), 2)

    async def test_final_unit_and_tool_ack_finish_without_response_done_per_unit(self):
        experiment = object.__new__(OfficialRealtimeExperiment)
        experiment._sent_units = 203
        experiment._last_native_response_unit = 203
        experiment._injected = {18: "room", 191: "calculator"}
        experiment._acknowledged = {"room", "calculator"}
        experiment._last_native_event_ns = time.monotonic_ns() - 3_000_000_000
        experiment._closed_event = asyncio.Event()
        await asyncio.wait_for(experiment._wait_native_drain(), timeout=0.5)


class AmendmentTests(unittest.IsolatedAsyncioTestCase):
    async def test_granite_generates_groundable_arguments_for_pending_amendment(self):
        def generate(_messages, _tools):
            return '<tool_call>{"name":"room_lookup","arguments":{"name":"vision seminar"}}</tool_call>'

        caller = GraniteToolCaller(generate)
        segment = TranscriptSegment("correction", 1,
                                    datetime.now(timezone.utc),
                                    "Actually, the vision seminar.", True)
        schemas = [{"type": "function", "function": {"name": "room_lookup"}}]
        with patch("duplex_tools.caller.score_choices", return_value=ChoiceScores("G", {"G": 0}, 1)):
            action = await caller.decide(segment, {"req-1": "room_lookup"}, schemas)
        self.assertEqual(action.kind, "amend")
        self.assertEqual(action.request_id, "req-1")
        self.assertEqual(action.arguments, {"name": "vision seminar"})

    async def test_late_original_result_is_excluded_after_spoken_correction(self):
        class Caller:
            async def decide(self, segment, pending, _schemas):
                if segment.text.startswith("Actually"):
                    return CallerAction("amend", "room_lookup", {"name": "vision seminar"},
                                        next(iter(pending)))
                return CallerAction("call", "room_lookup", {"name": "robotics seminar"})

        async def room(arguments):
            if arguments["name"] == "robotics seminar":
                await asyncio.sleep(0.05)
            else:
                await asyncio.sleep(0.01)
            return await RoomLookup({"robotics seminar": "B742", "vision seminar": "C314"})(arguments)

        controller = ContextController({"room_lookup": room})
        router = ConversationRouter(Caller(), controller)
        first = await router.ingest(TranscriptSegment("first", 1, datetime.now(timezone.utc),
                                                      "Where is the robotics seminar?", True))
        second = await router.ingest(TranscriptSegment("second", 1, datetime.now(timezone.utc),
                                                       "Actually, the vision seminar.", True))
        await controller.wait_all()
        queued = controller.scheduler.pending()
        self.assertEqual(first.request_id, second.request_id)
        self.assertEqual(len(queued), 1)
        self.assertEqual(queued[0].request_version, 2)
        self.assertEqual(queued[0].facts, "C314")


class FixedCaller:
    async def decide(self, _segment, _pending, _schemas):
        return CallerAction("call", "room_lookup", {"name": "robotics seminar"})


class FakeRecognizer:
    def transcribe(self, _path, **_kwargs):
        return iter([SimpleNamespace(text="Where is the robotics seminar?")]), None


class FakeSession:
    def __init__(self, controller):
        self.controller = controller
        self.counter = 0
        self.submitted = False

    @property
    def next_counter(self):
        return self.counter + 1

    async def prefill(self, *, audio_path, counter, boundary):
        self.counter = counter
        event = self.controller.reserve_injection(boundary)
        if event:
            self.controller.commit_injection(event.event_id, counter=counter)
            self.submitted = True

    async def decode_stream(self, *, debug_dir, round_idx, on_event):
        if self.submitted:
            on_event({"content": "The robotics seminar is in room B742.", "end_of_turn": False})
            on_event({"content": "", "end_of_turn": True, "is_listen": True})
            self.submitted = False


class LiveSessionTests(unittest.TestCase):
    def test_audio_routing_and_result_continue_in_same_session(self):
        with tempfile.TemporaryDirectory() as directory:
            controller = ContextController({"room_lookup": RoomLookup({"robotics seminar": "The robotics seminar is in room B742."})})
            router = ConversationRouter(FixedCaller(), controller)
            session = FakeSession(controller)
            experiment = LiveDuplexExperiment(session, router, FakeRecognizer(), Path(directory),
                                               final_silence_units=2)
            try:
                experiment.start()
                experiment._submit(experiment._accept_pcm(pcm(4000, 0.5) + pcm(0, 0.8)))
                summary = experiment.wait_finished(timeout_s=10)
                self.assertEqual(summary["status"], "complete")
                self.assertEqual(summary["utterance_count"], 1)
                self.assertGreaterEqual(session.counter, 2)
                events = [json.loads(line) for line in (experiment.directory / "live_events.jsonl").read_text().splitlines()]
                self.assertTrue(any(item["kind"] == "transcript_committed" for item in events))
                self.assertTrue(any(item["kind"] == "context_at_input_boundary" for item in events))
                self.assertTrue(any(item["kind"] == "model_event" and "B742" in item["event"].get("content", "") for item in events))
                self.assertTrue((experiment.directory / "microphone.wav").is_file())
            finally:
                experiment.close()


if __name__ == "__main__":
    unittest.main()
