"""Push-to-talk human gate for a persistent MiniCPM audio-only session."""

from __future__ import annotations

import asyncio
import json
import shutil
import threading
import time
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path

from .context_events import Boundary
from .contracts import TranscriptSegment
from .conversation import ConversationRouter
from .minicpm_client import MiniCPMStreamSession


class VoiceDemo:
    def __init__(self, session: MiniCPMStreamSession, router: ConversationRouter, output_dir: Path,
                 transcriber: object, *, routing_timeout_s: float = 90.0,
                 continuation_units: int = 8,
                 transcription_backend: str = "speech recognizer",
                 routing_backend: str = "Granite tool router") -> None:
        self.session = session
        self.router = router
        self.output_dir = output_dir
        self.transcriber = transcriber
        self.routing_timeout_s = routing_timeout_s
        if continuation_units < 1:
            raise ValueError("continuation_units must be positive")
        self.continuation_units = continuation_units
        self.transcription_backend = transcription_backend
        self.routing_backend = routing_backend
        self.counter = 0
        self._active_trial_id: ContextVar[str | None] = ContextVar("active_voice_trial_id", default=None)
        self._verdict_revisions: dict[str, int] = {}
        self.turn_lock = asyncio.Lock()
        self._jobs: dict[str, dict] = {}
        self._job_tasks: dict[str, object] = {}
        self._latest_job_id: str | None = None
        self._state_lock = threading.RLock()
        self._worker_loop = asyncio.new_event_loop()
        self._worker_thread = threading.Thread(
            target=self._worker_main,
            name="duplex-voice-worker",
            daemon=True,
        )
        self._worker_thread.start()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _worker_main(self) -> None:
        asyncio.set_event_loop(self._worker_loop)
        self._worker_loop.run_forever()
        pending = asyncio.all_tasks(self._worker_loop)
        for task in pending:
            task.cancel()
        if pending:
            self._worker_loop.run_until_complete(
                asyncio.gather(*pending, return_exceptions=True)
            )
        self._worker_loop.close()

    def close(self) -> None:
        """Cancel pending work and stop the demo-owned event loop."""
        with self._state_lock:
            futures = tuple(self._job_tasks.values())
        for future in futures:
            future.cancel()
        if self._worker_loop.is_running():
            self._worker_loop.call_soon_threadsafe(self._worker_loop.stop)
        if self._worker_thread.is_alive():
            self._worker_thread.join(timeout=5)

    def _publish_job(self, job_id: str, **fields) -> None:
        with self._state_lock:
            job = self._jobs[job_id]
            job.update(fields)
            snapshot = {key: value for key, value in job.items() if key != "started_monotonic"}
            temporary = self.output_dir / "latest_turn.json.tmp"
            temporary.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
            temporary.replace(self.output_dir / "latest_turn.json")

    def _job_outputs(self, job_id: str | None = None) -> tuple[str, str, str, str | None, str]:
        with self._state_lock:
            selected = job_id or self._latest_job_id
            if selected is None or selected not in self._jobs:
                return "Ready to record.", "", "{}", None, ""
            job = dict(self._jobs[selected])
        elapsed = time.monotonic() - job["started_monotonic"]
        status = f"{job['status']}: {job['stage']} ({elapsed:.1f}s)"
        trace = job.get("trace") or json.dumps(
            {
                "job_id": selected,
                "status": job["status"],
                "stage": job["stage"],
                "error": job.get("error"),
            },
            indent=2,
        )
        return status, job.get("transcript", ""), trace, job.get("audio"), selected

    async def start_turn(self, audio_path: str | None) -> tuple[str, str, str, str | None, str]:
        """Start a turn quickly so a public UI tunnel need not stay connected."""
        if not audio_path:
            return "Record speech first.", "", "{}", None, ""
        with self._state_lock:
            current_id = self._latest_job_id
            if current_id is not None and self._jobs[current_id]["status"] == "running":
                return self._job_outputs(current_id)
            job_id = uuid.uuid4().hex[:12]
            saved_input = self.output_dir / f"submitted_{job_id}.wav"
            shutil.copy2(audio_path, saved_input)
            self._jobs[job_id] = {
                "job_id": job_id,
                "status": "running",
                "stage": "Queued in the persistent MiniCPM session",
                "transcript": "",
                "trace": "",
                "audio": None,
                "error": None,
                "started_at": datetime.now(timezone.utc).isoformat(),
                "started_monotonic": time.monotonic(),
            }
            self._latest_job_id = job_id
        self._publish_job(job_id)
        future = asyncio.run_coroutine_threadsafe(
            self._run_job(job_id, str(saved_input)), self._worker_loop
        )
        with self._state_lock:
            self._job_tasks[job_id] = future
        def remove_job(_):
            with self._state_lock:
                self._job_tasks.pop(job_id, None)
        future.add_done_callback(remove_job)
        return self._job_outputs(job_id)

    async def _run_job(self, job_id: str, audio_path: str) -> None:
        def update(**fields) -> None:
            self._publish_job(job_id, **fields)

        trial_token = self._active_trial_id.set(job_id)
        try:
            transcript, trace, audio = await self.turn(audio_path, on_update=update)
            self._publish_job(job_id, status="complete", stage="Turn complete",
                              transcript=transcript, trace=trace, audio=audio)
        except asyncio.CancelledError:
            self._publish_job(job_id, status="cancelled", stage="Turn was cancelled")
            raise
        except Exception as exc:
            self.router.controller.log.write(
                "voice_turn_failed", job_id=job_id,
                error=f"{type(exc).__name__}: {exc}",
            )
            self._publish_job(job_id, status="failed", stage="Turn failed",
                              error=f"{type(exc).__name__}: {exc}")
        finally:
            self._active_trial_id.reset(trial_token)

    def poll_turn(self) -> tuple[str, str, str, str | None, str]:
        return self._job_outputs()

    def _transcribe(self, audio_path: str) -> str:
        segments, _ = self.transcriber.transcribe(audio_path, language="en", vad_filter=True)
        return " ".join(segment.text.strip() for segment in segments).strip()

    def _split_audio(self, audio_path: str) -> list[Path]:
        import numpy as np
        import soundfile as sf

        audio, rate = sf.read(audio_path, dtype="float32", always_2d=True)
        mono = audio.mean(axis=1)
        if rate != 16000:
            # Linear interpolation is sufficient for the microphone gate.
            x = np.arange(len(mono)) / rate
            target = np.arange(max(1, round(len(mono) * 16000 / rate))) / 16000
            mono = np.interp(target, x, mono).astype("float32")
        paths = []
        for offset in range(0, len(mono), 16000):
            chunk = mono[offset:offset + 16000]
            chunk = np.pad(chunk, (0, 16000 - len(chunk)))
            path = self.output_dir / f"input_{self.counter + len(paths) + 1:05d}.wav"
            sf.write(path, chunk, 16000)
            paths.append(path)
        return paths

    async def _new_tts(self, before: dict[Path, tuple[int, int]], flags_before: dict[Path, int],
                       *, expect_speech: bool, trial_id: str) -> tuple[str | None, list[str]]:
        import numpy as np
        import soundfile as sf

        deadline = time.monotonic() + (30 if expect_speech else 2)
        last_change = time.monotonic()
        files: list[Path] = []
        while time.monotonic() < deadline:
            current = sorted(
                (path for path in self.output_dir.rglob("wav_*.wav")
                 if (path.stat().st_mtime_ns, path.stat().st_size) != before.get(path)),
                key=lambda path: path.stat().st_mtime_ns,
            )
            if len(current) != len(files):
                files = current
                last_change = time.monotonic()
            finished = any(path.stat().st_mtime_ns > flags_before.get(path, -1)
                           for path in self.output_dir.rglob("generation_done.flag"))
            if files and (finished or time.monotonic() - last_change >= 2):
                break
            await asyncio.sleep(0.25)
        if not files:
            return None, []
        pieces = []
        rate = None
        for path in files:
            data, this_rate = sf.read(path, dtype="float32")
            if rate is not None and this_rate != rate:
                raise ValueError("TTS chunks have inconsistent sample rates")
            rate = this_rate
            pieces.append(data)
        path = self.output_dir / f"heard_{trial_id}.wav"
        sf.write(path, np.concatenate(pieces), rate)
        return str(path), [str(item) for item in files]

    def _silence_unit(self) -> Path:
        import numpy as np
        import soundfile as sf

        path = self.output_dir / f"input_{self.counter:05d}.wav"
        sf.write(path, np.zeros(16000, dtype="float32"), 16000)
        return path

    async def turn(self, audio_path: str | None, progress=None, on_update=None) -> tuple[str, str, str | None]:
        if not audio_path:
            return "", "Record speech first.", None
        async with self.turn_lock:
            started = time.perf_counter()
            trial_id = self._active_trial_id.get() or uuid.uuid4().hex[:12]
            def stage(description):
                self.router.controller.log.write("voice_stage", trial_id=trial_id, stage=description,
                                                 elapsed_s=round(time.perf_counter() - started, 3))
                if progress is not None:
                    progress(0, desc=description)
                if on_update is not None:
                    on_update(stage=description)

            log_start = len(self.router.controller.log.records)
            stage(f"Transcribing microphone recording with {self.transcription_backend}")
            transcript = await asyncio.to_thread(self._transcribe, audio_path)
            if on_update is not None:
                on_update(transcript=transcript)
            segment = TranscriptSegment(uuid.uuid4().hex, 1, datetime.now(timezone.utc), transcript, True)
            route_task = asyncio.create_task(self.router.ingest(segment)) if transcript else None
            before = {path: (path.stat().st_mtime_ns, path.stat().st_size)
                      for path in self.output_dir.rglob("wav_*.wav")}
            flags_before = {path: path.stat().st_mtime_ns for path in self.output_dir.rglob("generation_done.flag")}
            pre_result_text: list[str] = []
            post_result_text: list[str] = []
            text_fragments: list[dict] = []
            submitted_events: list[dict] = []
            response_finished = False

            async def decode_current() -> None:
                nonlocal response_finished
                events = self._decode_events(await self.session.decode(
                    debug_dir=str(self.output_dir), round_idx=self.counter - 1
                ))
                submitted_events[:] = [record for record in self.router.controller.log.records[log_start:]
                                       if record["kind"] == "context_submitted"]
                phase = "post_result" if submitted_events else "pre_result"
                for event in events:
                    content = event.get("content")
                    if content:
                        fragment = str(content)
                        (post_result_text if submitted_events else pre_result_text).append(fragment)
                        text_fragments.append({"at": datetime.now(timezone.utc).isoformat(),
                                               "counter": self.counter, "phase": phase, "text": fragment})
                        self.router.controller.log.write("assistant_text", trial_id=trial_id,
                                                         counter=self.counter, phase=phase, text=fragment)
                    if submitted_events and post_result_text and event.get("end_of_turn"):
                        response_finished = True

            chunks = self._split_audio(audio_path)
            for index, path in enumerate(chunks, 1):
                self.counter += 1
                stage(f"Sending audio chunk {index}/{len(chunks)} (counter {self.counter})")
                await self.session.prefill(audio_path=str(path), counter=self.counter, boundary=Boundary.UNIT)
                stage(f"Waiting for MiniCPM decode {index}/{len(chunks)}")
                await decode_current()
            stage(f"Waiting for {self.routing_backend} (limit {self.routing_timeout_s:g}s)")
            try:
                action = await asyncio.wait_for(route_task, timeout=self.routing_timeout_s) if route_task else None
            except TimeoutError:
                self.router.controller.log.write(
                    "caller_timeout", transcript=transcript,
                    timeout_s=self.routing_timeout_s,
                )
                raise TimeoutError(
                    f"Granite tool routing exceeded {self.routing_timeout_s:g} seconds"
                ) from None
            stage("Waiting for selected tool execution")
            await self.router.controller.wait_all()
            # Keep the model moving after submission. An empty scheduler only
            # means delivery occurred; it does not mean the model has answered.
            for index in range(1, self.continuation_units + 1):
                if action is None or action.kind not in {"call", "amend"}:
                    break
                if submitted_events and response_finished:
                    break
                if not submitted_events and not self.router.controller.scheduler.pending():
                    break
                self.counter += 1
                silence = self._silence_unit()
                stage(f"Post-result continuation {index}/{self.continuation_units} (counter {self.counter})")
                await self.session.prefill(audio_path=str(silence), counter=self.counter, boundary=Boundary.UNIT)
                await decode_current()
            stage("Collecting generated speech (up to 30 seconds)")
            audio, tts_chunks = await self._new_tts(
                before, flags_before,
                expect_speech=bool(pre_result_text or post_result_text), trial_id=trial_id,
            )
            stage("Turn complete")
            if action and action.kind in {"call", "amend"}:
                if not submitted_events:
                    answer_status = "no_context_submitted"
                elif not post_result_text:
                    answer_status = "no_post_result_answer"
                elif pre_result_text:
                    answer_status = "mixed_prior_and_post_result_speech"
                else:
                    answer_status = "post_result_answer_candidate"
            else:
                answer_status = "no_tool_requested"
            trace = {"trial_id": trial_id, "at": datetime.now(timezone.utc).isoformat(), "transcript": transcript,
                     "action": action.kind if action else "none", "tool": action.tool if action else None,
                     "request_id": action.request_id if action else None,
                     "model_text": "".join(pre_result_text + post_result_text),
                     "pre_result_text": "".join(pre_result_text),
                     "post_result_text": "".join(post_result_text),
                     "text_fragments": text_fragments,
                     "submitted_events": submitted_events,
                     "answer_status": answer_status,
                     "response_finished": response_finished,
                     "audio_attribution": ("post_result_candidate" if audio and post_result_text and not pre_result_text
                                           else "mixed_or_unknown" if audio and submitted_events
                                           else "no_result_or_no_audio"),
                     "tts_chunks": tts_chunks,
                     "evaluation": "unknown", "audio_file": audio,
                     "elapsed_s": round(time.perf_counter() - started, 3)}
            with (self.output_dir / "human_trials.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(trace) + "\n")
            return transcript, json.dumps(trace, indent=2), audio

    @staticmethod
    def _decode_events(body: str) -> list[dict]:
        result: list[dict] = []
        for line in body.splitlines():
            if line.startswith("data: ") and line != "data: [DONE]":
                try:
                    value = json.loads(line[6:])
                    if isinstance(value, dict):
                        result.append(value)
                except json.JSONDecodeError:
                    pass
        return result

    @staticmethod
    def _decode_text(body: str) -> list[str]:
        return [str(event["content"]) for event in VoiceDemo._decode_events(body)
                if event.get("content")]

    def annotate(self, trial_id: str, verdict: str, notes: str) -> str:
        if verdict not in {"correct", "incorrect", "no audible answer", "unclear"}:
            return "Select a listening verdict first."
        with self._state_lock:
            if trial_id not in self._jobs or self._jobs[trial_id]["status"] != "complete":
                return "Wait for this trial to finish before saving a verdict."
            revision = self._verdict_revisions.get(trial_id, 0) + 1
            record = {"trial_id": trial_id, "revision": revision,
                      "at": datetime.now(timezone.utc).isoformat(), "verdict": verdict, "notes": notes}
            with (self.output_dir / "human_verdicts.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            self._verdict_revisions[trial_id] = revision
        return f"Verdict saved for trial {trial_id}, revision {revision}."


def make_gradio_ui(demo: VoiceDemo):
    import gradio as gr

    async def start_turn(audio_path):
        return (*await demo.start_turn(audio_path), None, "", None)

    def poll_turn(previous_audio):
        status, transcript, trace, speech, trial_id = demo.poll_turn()
        audio_update = gr.skip() if speech == previous_audio else speech
        return status, transcript, trace, audio_update, trial_id, speech

    with gr.Blocks(title="Duplex Tools human gate") as app:
        gr.Markdown("# Speak with MiniCPM and test its tools\nRecord a short turn, then submit. The trace separates MiniCPM speech before and after the current tool result. Listen to the returned speech and record what you heard. The MiniCPM session stays initialized across turns.")
        microphone = gr.Audio(sources=["microphone"], type="filepath", format="wav", label="Your voice")
        submit = gr.Button("Send turn")
        status = gr.Textbox(label="Turn status", value="Ready to record.", interactive=False)
        transcript = gr.Textbox(label="User transcript")
        trace = gr.Code(label="Tool and delivery trace", language="json")
        speech = gr.Audio(label="MiniCPM speech", autoplay=False)
        previous_audio = gr.State(None)
        trial_id = gr.Textbox(label="Trial ID", interactive=False)
        verdict = gr.Radio(["correct", "incorrect", "no audible answer", "unclear"], label="What did you hear?")
        notes = gr.Textbox(label="Notes or exact spoken words")
        outputs = [status, transcript, trace, speech, trial_id]
        submit.click(start_turn, inputs=[microphone], outputs=outputs + [verdict, notes, previous_audio], queue=False,
                     concurrency_limit=1, show_progress="hidden")
        timer = gr.Timer(value=1.0, active=True)
        timer.tick(poll_turn, inputs=[previous_audio], outputs=outputs + [previous_audio],
                   queue=False, show_progress="hidden")
        save = gr.Button("Save listening verdict")
        saved = gr.Textbox(label="Verdict status")
        save.click(demo.annotate, inputs=[trial_id, verdict, notes], outputs=[saved])
    return app
