import asyncio
import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

from duplex_tools.contracts import CallerAction
from duplex_tools.controller import ContextController, JsonlEventLog
from duplex_tools.conversation import ConversationRouter
from duplex_tools.tools import RoomLookup
from duplex_tools.voice_demo import VoiceDemo


class _SuccessfulDemo(VoiceDemo):
    async def turn(self, audio_path, progress=None, on_update=None):
        on_update(stage="Transcribing microphone recording with speech recognizer")
        await asyncio.sleep(0.01)
        on_update(transcript="Where is the robotics seminar?")
        on_update(stage="Waiting for Granite tool router (limit 90s)")
        await asyncio.sleep(0.01)
        return "Where is the robotics seminar?", '{"tool":"room_lookup"}', "/tmp/answer.wav"


class _FailedDemo(VoiceDemo):
    async def turn(self, audio_path, progress=None, on_update=None):
        raise RuntimeError("test failure")


def _router():
    return SimpleNamespace(controller=SimpleNamespace(log=JsonlEventLog()))


class VoiceDemoJobTests(unittest.IsolatedAsyncioTestCase):
    async def test_background_turn_returns_immediately_and_can_be_polled(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recording = root / "recording.wav"
            recording.write_bytes(b"audio")
            demo = _SuccessfulDemo(object(), _router(), root / "out", object())
            try:
                initial = await demo.start_turn(str(recording))
                self.assertIn("running:", initial[0])
                self.assertTrue(next((root / "out").glob("submitted_*.wav")))

                for _ in range(20):
                    await asyncio.sleep(0.01)
                    final = demo.poll_turn()
                    if final[0].startswith("complete:"):
                        break
                self.assertIn("complete:", final[0])
                self.assertEqual(final[1], "Where is the robotics seminar?")
                self.assertEqual(final[2], '{"tool":"room_lookup"}')
                self.assertEqual(final[3], "/tmp/answer.wav")
                self.assertEqual(final[4], saved_id := next((root / "out").glob("submitted_*.wav")).stem.removeprefix("submitted_"))
                saved = json.loads((root / "out" / "latest_turn.json").read_text())
                self.assertEqual(saved["status"], "complete")
                self.assertEqual(saved["job_id"], saved_id)
                self.assertNotIn("started_monotonic", saved)
                self.assertIsNot(demo._worker_loop, asyncio.get_running_loop())
            finally:
                demo.close()

    async def test_verdict_uses_displayed_trial_id_and_revision(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recording = root / "recording.wav"
            recording.write_bytes(b"audio")
            demo = _SuccessfulDemo(object(), _router(), root / "out", object())
            try:
                started = await demo.start_turn(str(recording))
                trial_id = started[4]
                self.assertIn("Wait", demo.annotate(trial_id, "correct", "too early"))
                for _ in range(20):
                    await asyncio.sleep(0.01)
                    if demo.poll_turn()[0].startswith("complete:"):
                        break
                self.assertIn("revision 1", demo.annotate(trial_id, "correct", "heard B742"))
                self.assertIn("revision 2", demo.annotate(trial_id, "incorrect", "correcting my note"))
                records = [json.loads(line) for line in (root / "out" / "human_verdicts.jsonl").read_text().splitlines()]
                self.assertEqual([record["revision"] for record in records], [1, 2])
                self.assertEqual({record["trial_id"] for record in records}, {trial_id})
            finally:
                demo.close()

    async def test_background_failure_is_visible_and_logged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recording = root / "recording.wav"
            recording.write_bytes(b"audio")
            router = _router()
            demo = _FailedDemo(object(), router, root / "out", object())
            try:
                await demo.start_turn(str(recording))
                for _ in range(20):
                    await asyncio.sleep(0.01)
                    final = demo.poll_turn()
                    if final[0].startswith("failed:"):
                        break
                self.assertIn("failed:", final[0])
                self.assertIn("RuntimeError: test failure", final[2])
                self.assertEqual(router.controller.log.records[-1]["kind"], "voice_turn_failed")
            finally:
                demo.close()


class _DelayedRoomCaller:
    async def decide(self, segment, pending, tools):
        await asyncio.sleep(0.02)
        return CallerAction("call", "room_lookup", {"name": "robotics seminar"})


class _ResponseSession:
    def __init__(self, controller):
        self.controller = controller
        self.submitted_at = None
        self.decode_count = 0
        self.counters = []

    async def prefill(self, *, audio_path, counter, boundary):
        self.counters.append(counter)
        event = self.controller.reserve_injection(boundary, model_idle=True)
        if event:
            self.controller.commit_injection(event.event_id, counter=counter)
            self.submitted_at = counter

    async def decode(self, *, debug_dir, round_idx):
        self.decode_count += 1
        if self.submitted_at is None and self.decode_count == 1:
            return 'data: {"content":"I do not know.","end_of_turn":true}\n\ndata: [DONE]\n\n'
        if self.submitted_at is not None and self.decode_count == 4:
            return 'data: {"content":"The room is B742.","end_of_turn":false}\n\ndata: {"content":"","end_of_turn":true,"is_listen":true}\n\ndata: [DONE]\n\n'
        return 'data: {"content":"","end_of_turn":true,"is_listen":true}\n\ndata: [DONE]\n\n'


class _NoAnswerSession(_ResponseSession):
    async def decode(self, *, debug_dir, round_idx):
        self.decode_count += 1
        return 'data: {"content":"","end_of_turn":true,"is_listen":true}\n\ndata: [DONE]\n\n'


class _NoAudioDemo(VoiceDemo):
    def _transcribe(self, audio_path):
        return "Where is the robotics seminar?"

    def _split_audio(self, audio_path):
        return [self.output_dir / "input_00001.wav", self.output_dir / "input_00002.wav"]

    def _silence_unit(self):
        return self.output_dir / f"input_{self.counter:05d}.wav"

    async def _new_tts(self, before, flags_before, *, expect_speech, trial_id):
        return None, []


class VoiceDemoCausalTests(unittest.IsolatedAsyncioTestCase):
    async def test_continues_after_submission_and_separates_prior_speech(self):
        with tempfile.TemporaryDirectory() as directory:
            controller = ContextController({"room_lookup": RoomLookup({"robotics seminar": "The room is B742."})})
            router = ConversationRouter(_DelayedRoomCaller(), controller)
            session = _ResponseSession(controller)
            demo = _NoAudioDemo(session, router, Path(directory), object(), continuation_units=4)
            try:
                _, trace_json, _ = await demo.turn("unused.wav")
                trace = json.loads(trace_json)
                self.assertEqual(trace["pre_result_text"], "I do not know.")
                self.assertEqual(trace["post_result_text"], "The room is B742.")
                self.assertEqual(trace["answer_status"], "mixed_prior_and_post_result_speech")
                self.assertTrue(trace["response_finished"])
                self.assertEqual(session.counters, [1, 2, 3, 4])
                self.assertEqual(trace["submitted_events"][0]["counter"], 3)
                self.assertEqual([item["phase"] for item in trace["text_fragments"]],
                                 ["pre_result", "post_result"])
            finally:
                demo.close()

    async def test_no_answer_stops_at_configured_continuation_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            controller = ContextController({"room_lookup": RoomLookup({"robotics seminar": "The room is B742."})})
            router = ConversationRouter(_DelayedRoomCaller(), controller)
            session = _NoAnswerSession(controller)
            demo = _NoAudioDemo(session, router, Path(directory), object(), continuation_units=3)
            try:
                _, trace_json, _ = await demo.turn("unused.wav")
                trace = json.loads(trace_json)
                self.assertEqual(session.counters, [1, 2, 3, 4, 5])
                self.assertEqual(trace["answer_status"], "no_post_result_answer")
                self.assertEqual(trace["post_result_text"], "")
            finally:
                demo.close()

class VoiceDemoLoopOwnershipTests(unittest.TestCase):
    def test_turn_survives_submit_event_loop_closing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recording = root / "recording.wav"
            recording.write_bytes(b"audio")
            demo = _SuccessfulDemo(object(), _router(), root / "out", object())
            try:
                # asyncio.run closes the short-lived submit loop immediately.
                # The actual turn must continue on VoiceDemo's worker loop.
                asyncio.run(demo.start_turn(str(recording)))
                for _ in range(20):
                    time.sleep(0.01)
                    final = demo.poll_turn()
                    if final[0].startswith("complete:"):
                        break
                self.assertIn("complete:", final[0])
            finally:
                demo.close()


if __name__ == "__main__":
    unittest.main()
