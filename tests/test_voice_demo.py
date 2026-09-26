import asyncio
import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

from duplex_tools.controller import JsonlEventLog
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
                saved = json.loads((root / "out" / "latest_turn.json").read_text())
                self.assertEqual(saved["status"], "complete")
                self.assertNotIn("started_monotonic", saved)
                self.assertIsNot(demo._worker_loop, asyncio.get_running_loop())
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
