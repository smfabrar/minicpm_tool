import asyncio
import json
import tempfile
import unittest
from array import array
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from duplex_tools.caller import ChoiceScores, GraniteToolCaller
from duplex_tools.contracts import CallerAction, TranscriptSegment
from duplex_tools.controller import ContextController
from duplex_tools.conversation import ConversationRouter
from duplex_tools.live_duplex import LiveDuplexExperiment, SpeechSegmenter
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
