"""Paced microphone input and independent speech delivery for a live trial.

The browser transport is provided by Gradio. This module owns the research
session: committed user speech, model input order, tool results, timestamps,
and immutable copies of every emitted audio chunk.
"""

from __future__ import annotations

import asyncio
import json
import math
import queue
import threading
import time
import uuid
import wave
from array import array
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .context_events import Boundary
from .contracts import TranscriptSegment
from .conversation import ConversationRouter
from .minicpm_client import MiniCPMStreamSession


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_wav(path: Path, pcm: bytes, rate: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(pcm)


def microphone_pcm(audio: tuple[int, Any]) -> bytes:
    """Turn one Gradio microphone chunk into mono 16 kHz signed PCM."""
    import numpy as np

    rate, samples = audio
    if rate <= 0:
        raise ValueError("microphone sample rate must be positive")
    data = np.asarray(samples)
    if data.ndim == 2:
        data = data.mean(axis=0 if data.shape[0] <= 2 < data.shape[1] else 1)
    if data.ndim != 1:
        raise ValueError("microphone audio must be mono or stereo")
    if np.issubdtype(data.dtype, np.integer):
        scale = max(abs(np.iinfo(data.dtype).min), np.iinfo(data.dtype).max)
        data = data.astype(np.float32) / scale
    else:
        data = data.astype(np.float32)
    if rate != 16000 and len(data):
        target_length = max(1, round(len(data) * 16000 / rate))
        data = np.interp(
            np.arange(target_length) / 16000,
            np.arange(len(data)) / rate,
            data,
        )
    return (np.clip(data, -1, 1) * 32767).astype("<i2").tobytes()


class SpeechSegmenter:
    """Commit an utterance after speech followed by a real silence interval."""

    FRAME_SAMPLES = 320  # 20 ms at 16 kHz

    def __init__(self, *, rms_threshold: float = 0.018, pause_s: float = 0.65,
                 min_speech_s: float = 0.26, max_utterance_s: float = 12.0) -> None:
        self.rms_threshold = rms_threshold
        self.pause_frames = max(1, math.ceil(pause_s / 0.02))
        self.min_voice_frames = max(1, math.ceil(min_speech_s / 0.02))
        self.max_frames = max(1, math.ceil(max_utterance_s / 0.02))
        self._remainder = bytearray()
        self._preroll: deque[bytes] = deque(maxlen=15)
        self._active: list[bytes] = []
        self._voice_frames = 0
        self._quiet_frames = 0

    @property
    def speech_active(self) -> bool:
        return bool(self._active)

    def _finish(self) -> bytes | None:
        result = b"".join(self._active) if self._voice_frames >= self.min_voice_frames else None
        self._active = []
        self._voice_frames = 0
        self._quiet_frames = 0
        return result

    def push(self, pcm: bytes) -> list[bytes]:
        if len(pcm) % 2:
            raise ValueError("PCM must contain whole 16-bit samples")
        self._remainder.extend(pcm)
        completed: list[bytes] = []
        frame_bytes = self.FRAME_SAMPLES * 2
        while len(self._remainder) >= frame_bytes:
            frame = bytes(self._remainder[:frame_bytes])
            del self._remainder[:frame_bytes]
            values = array("h")
            values.frombytes(frame)
            rms = math.sqrt(sum(int(value) * int(value) for value in values) / len(values)) / 32768
            voiced = rms >= self.rms_threshold
            if not self._active:
                self._preroll.append(frame)
                if voiced:
                    self._active = list(self._preroll)
                    self._voice_frames = 1
                    self._quiet_frames = 0
                    self._preroll.clear()
                continue
            self._active.append(frame)
            if voiced:
                self._voice_frames += 1
                self._quiet_frames = 0
            else:
                self._quiet_frames += 1
            if self._quiet_frames >= self.pause_frames or len(self._active) >= self.max_frames:
                result = self._finish()
                if result:
                    completed.append(result)
        return completed

    def flush(self) -> bytes | None:
        if self._remainder:
            padding = self.FRAME_SAMPLES * 2 - len(self._remainder)
            completed = self.push(bytes(padding))
            self._remainder.clear()
            if completed:
                return completed[-1]
        return self._finish()


class LiveDuplexExperiment:
    """One continuous human session with independently paced input and output."""

    def __init__(self, session: MiniCPMStreamSession, router: ConversationRouter,
                 recognizer: Any, output_root: Path, *,
                 routing_timeout_s: float = 90.0, vad_rms: float = 0.018,
                 vad_pause_s: float = 0.65, max_backlog_s: float = 12.0,
                 final_silence_units: int = 8) -> None:
        self.session = session
        self.router = router
        self.recognizer = recognizer
        self.output_root = Path(output_root)
        self.routing_timeout_s = routing_timeout_s
        self.vad_rms = vad_rms
        self.vad_pause_s = vad_pause_s
        self.max_backlog_s = max_backlog_s
        self.final_silence_units = final_silence_units
        self._state_lock = threading.RLock()
        self._state: dict[str, Any] = {"status": "ready", "session_id": None}
        self._playback: queue.Queue[Path] = queue.Queue()
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, name="live-duplex-worker", daemon=True)
        self._thread.start()
        self._started = False

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()
        tasks = asyncio.all_tasks(self._loop)
        for task in tasks:
            task.cancel()
        if tasks:
            self._loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
        self._loop.close()

    def _submit(self, coro, *, timeout: float = 10.0):
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(timeout=timeout)

    def _log(self, kind: str, **fields: Any) -> None:
        record = {"at": utc_now(), "monotonic_ns": time.monotonic_ns(),
                  "kind": kind, **fields}
        with (self.directory / "live_events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        with self._state_lock:
            self._state["last_event"] = record

    def _set_state(self, **fields: Any) -> None:
        with self._state_lock:
            self._state.update(fields)
            snapshot = dict(self._state)
        temporary = self.directory / "live_status.json.tmp"
        temporary.write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.directory / "live_status.json")

    def start(self) -> str:
        if self._started:
            raise RuntimeError("this live experiment already started; use a new initialized run")
        self._started = True
        return self._submit(self._start())

    async def _start(self) -> str:
        self.id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:6]
        self.directory = self.output_root / "live_sessions" / self.id
        self.directory.mkdir(parents=True, exist_ok=False)
        self._audio_queue: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=40)
        self._speech_queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        self._segmenter = SpeechSegmenter(rms_threshold=self.vad_rms, pause_s=self.vad_pause_s)
        self._input_wav = wave.open(str(self.directory / "microphone.wav"), "wb")
        self._input_wav.setnchannels(1)
        self._input_wav.setsampwidth(2)
        self._input_wav.setframerate(16000)
        self._received_samples = 0
        self._prefilled_samples = 0
        self._max_backlog_s = 0.0
        self._outbound_windows: deque[tuple[int, int]] = deque()
        self._estimated_overlap_s = 0.0
        self._utterances = 0
        self._outbound_chunks: list[Path] = []
        self._assistant_text: list[str] = []
        self._transcripts: list[str] = []
        self._actions: list[dict[str, Any]] = []
        self._latest_transcript = ""
        self._accepting = True
        self._finalizing = False
        self._watcher_stop = False
        self._finish_task: asyncio.Task | None = None
        self._error: str | None = None
        self._awaiting_post_result = False
        self._post_result_text = ""
        self._tts_seen = {path: (path.stat().st_mtime_ns, path.stat().st_size)
                          for path in self.output_root.rglob("wav_*.wav")}
        self._silence_path = self.directory / "one_second_silence.wav"
        write_wav(self._silence_path, bytes(16000 * 2), 16000)
        settings = {"session_id": self.id, "started_at": utc_now(),
                    "input_unit_s": 1.0, "transport_chunk_target_s": 0.5,
                    "vad_rms": self.vad_rms, "vad_pause_s": self.vad_pause_s,
                    "routing_timeout_s": self.routing_timeout_s,
                    "max_backlog_s": self.max_backlog_s,
                    "final_silence_units": self.final_silence_units,
                    "native_context_evaluation_ack": False}
        (self.directory / "live_manifest.json").write_text(json.dumps(settings, indent=2) + "\n")
        self._model_task = asyncio.create_task(self._model_loop())
        self._route_task = asyncio.create_task(self._route_loop())
        self._tts_task = asyncio.create_task(self._watch_tts())
        self._set_state(status="running", session_id=self.id, started_at=settings["started_at"],
                        input_backlog_s=0.0, received_audio_s=0.0, last_transcript="",
                        assistant_text="", error=None)
        self._log("live_started", settings=settings)
        return self.id

    def receive(self, session_id: str, audio: tuple[int, Any] | None) -> str:
        if not session_id or audio is None:
            return "Start the live session first."
        if session_id != self._state.get("session_id"):
            return "This microphone stream belongs to another session."
        pcm = microphone_pcm(audio)
        if pcm:
            self._submit(self._accept_pcm(pcm), timeout=5)
        return f"Microphone connected · {len(pcm) // 2 / 16000:.2f}s chunk"

    async def _accept_pcm(self, pcm: bytes) -> None:
        if not self._accepting:
            return
        arrived_ns = time.monotonic_ns()
        chunk_ns = round((len(pcm) // 2 / 16000) * 1e9)
        while self._outbound_windows and self._outbound_windows[0][1] < arrived_ns - chunk_ns:
            self._outbound_windows.popleft()
        overlap_ns = sum(max(0, min(arrived_ns, end) - max(arrived_ns - chunk_ns, start))
                         for start, end in self._outbound_windows)
        self._estimated_overlap_s += min(chunk_ns, overlap_ns) / 1e9
        self._input_wav.writeframesraw(pcm)
        self._received_samples += len(pcm) // 2
        for utterance in self._segmenter.push(pcm):
            self._speech_queue.put_nowait(utterance)
            self._log("utterance_committed_by_silence", samples=len(utterance) // 2)
        backlog = (self._received_samples - self._prefilled_samples) / 16000
        self._max_backlog_s = max(self._max_backlog_s, backlog)
        self._log("microphone_chunk", samples=len(pcm) // 2,
                  received_audio_s=round(self._received_samples / 16000, 3),
                  input_backlog_s=round(backlog, 3), speech_active=self._segmenter.speech_active,
                  estimated_outbound_overlap_s=round(min(chunk_ns, overlap_ns) / 1e9, 3))
        self._set_state(received_audio_s=round(self._received_samples / 16000, 2),
                        input_backlog_s=round(backlog, 2))
        if backlog > self.max_backlog_s or self._audio_queue.full():
            self._error = f"Input backlog exceeded {self.max_backlog_s:g}s; live timing failed"
            self._log("realtime_overload", error=self._error, backlog_s=backlog)
            self._accepting = False
            self._finish_task = asyncio.create_task(self._finish())
            return
        self._audio_queue.put_nowait(pcm)

    def stop(self, session_id: str | None = None) -> str:
        if not self._started:
            return "No live session has started."
        if session_id and session_id != self._state.get("session_id"):
            return "Session ID does not match."
        self._submit(self._request_stop(), timeout=5)
        return "Stopping capture and finishing queued work. Watch the status below."

    async def _request_stop(self) -> None:
        if self._finalizing or self._finish_task is not None:
            return
        self._accepting = False
        self._finish_task = asyncio.create_task(self._finish())

    async def _route_loop(self) -> None:
        while True:
            pcm = await self._speech_queue.get()
            if pcm is None:
                return
            self._utterances += 1
            path = self.directory / f"utterance_{self._utterances:04d}.wav"
            write_wav(path, pcm)
            self._log("transcription_started", utterance=self._utterances, audio_file=str(path))
            try:
                def transcribe() -> str:
                    segments, _ = self.recognizer.transcribe(
                        str(path), language="en", vad_filter=True)
                    return " ".join(item.text.strip() for item in segments).strip()

                transcript = await asyncio.to_thread(transcribe)
                self._log("transcript_committed", utterance=self._utterances, text=transcript)
                self._latest_transcript = transcript
                self._transcripts.append(transcript)
                self._set_state(last_transcript=transcript)
                if not transcript:
                    continue
                segment = TranscriptSegment(f"{self.id}-{self._utterances}", 1,
                                            datetime.now(timezone.utc), transcript, True)
                action = await asyncio.wait_for(self.router.ingest(segment),
                                                timeout=self.routing_timeout_s)
                self._actions.append({"utterance": self._utterances, "action": action.kind,
                                      "tool": action.tool, "request_id": action.request_id,
                                      "arguments": dict(action.arguments)})
                self._log("caller_action", utterance=self._utterances,
                          action=action.kind, tool=action.tool,
                          arguments=dict(action.arguments), request_id=action.request_id,
                          message=action.message, raw=action.raw)
            except Exception as exc:
                self._log("utterance_failed", utterance=self._utterances,
                          error=f"{type(exc).__name__}: {exc}")

    async def _model_loop(self) -> None:
        buffer = bytearray()
        try:
            while True:
                pcm = await self._audio_queue.get()
                if pcm is None:
                    break
                buffer.extend(pcm)
                while len(buffer) >= 32000:
                    unit = bytes(buffer[:32000])
                    del buffer[:32000]
                    await self._send_unit(unit, microphone=True)
            if buffer:
                await self._send_unit(bytes(buffer).ljust(32000, b"\0"), microphone=True,
                                      actual_samples=len(buffer) // 2)
        except Exception as exc:
            self._error = f"{type(exc).__name__}: {exc}"
            self._log("model_loop_failed", error=self._error)
            self._accepting = False
            if self._finish_task is None:
                self._finish_task = asyncio.create_task(self._finish())

    async def _send_unit(self, pcm: bytes, *, microphone: bool,
                         actual_samples: int = 16000) -> None:
        counter = self.session.next_counter
        path = self.directory / f"input_{counter:06d}.wav"
        write_wav(path, pcm)
        begin = time.monotonic_ns()
        self._log("model_unit_started", counter=counter, microphone=microphone,
                  audio_file=str(path), actual_samples=actual_samples)
        await self.session.prefill(audio_path=str(path), counter=counter, boundary=Boundary.UNIT)
        if microphone:
            self._prefilled_samples += actual_samples
        submitted = [record for record in self.router.controller.log.records
                     if record["kind"] == "context_submitted" and record.get("counter") == counter]
        if submitted:
            self._awaiting_post_result = True
            self._post_result_text = ""
            self._log("context_at_input_boundary", counter=counter,
                      event_ids=[item["event_id"] for item in submitted])

        def text_event(event: dict[str, Any]) -> None:
            self._log("model_event", counter=counter, event=event,
                      submitted_event_ids=[item["event_id"] for item in submitted])
            if event.get("content"):
                self._assistant_text.append(str(event["content"]))
                if self._awaiting_post_result:
                    self._post_result_text += str(event["content"])
                self._set_state(assistant_text="".join(self._assistant_text)[-1000:])
            if self._awaiting_post_result and self._post_result_text and event.get("end_of_turn"):
                self._awaiting_post_result = False

        await self.session.decode_stream(debug_dir=str(self.output_root),
                                         round_idx=counter - 1, on_event=text_event)
        self._log("model_unit_finished", counter=counter,
                  elapsed_ms=round((time.monotonic_ns() - begin) / 1e6, 1),
                  input_backlog_s=round((self._received_samples - self._prefilled_samples) / 16000, 3))

    async def _watch_tts(self) -> None:
        stable: dict[Path, tuple[tuple[int, int], float]] = {}
        last_new = time.monotonic()
        while not self._watcher_stop or time.monotonic() - last_new < 2.0:
            for path in self.output_root.rglob("wav_*.wav"):
                try:
                    signature = (path.stat().st_mtime_ns, path.stat().st_size)
                except FileNotFoundError:
                    continue
                if self._tts_seen.get(path) == signature:
                    continue
                previous = stable.get(path)
                if previous is None or previous[0] != signature:
                    stable[path] = (signature, time.monotonic())
                    continue
                if time.monotonic() - previous[1] < 0.15:
                    continue
                try:
                    import soundfile as sf
                    samples, rate = sf.read(path, dtype="float32", always_2d=True)
                    if len(samples) == 0:
                        continue
                    destination = self.directory / f"speaker_{len(self._outbound_chunks) + 1:04d}.wav"
                    sf.write(destination, samples.mean(axis=1), rate)
                except (OSError, RuntimeError, ValueError):
                    continue
                self._tts_seen[path] = signature
                self._outbound_chunks.append(destination)
                self._playback.put(destination)
                last_new = time.monotonic()
                self._log("speech_chunk_ready", source=str(path), saved_as=str(destination),
                          duration_s=round(len(samples) / rate, 3))
            await asyncio.sleep(0.1)

    async def _finish(self) -> None:
        if self._finalizing:
            return
        self._finalizing = True
        self._set_state(status="finishing")
        try:
            utterance = self._segmenter.flush()
            if utterance:
                self._speech_queue.put_nowait(utterance)
                self._log("utterance_committed_at_stop", samples=len(utterance) // 2)
            self._speech_queue.put_nowait(None)
            await self._route_task
            await self.router.controller.wait_all()
            if not self._model_task.done():
                await self._audio_queue.put(None)
            await self._model_task
            if self._error is None:
                for _ in range(self.final_silence_units):
                    if not self.router.controller.scheduler.pending() and not self._awaiting_post_result:
                        break
                    await self._send_unit(bytes(32000), microphone=False)
            self._watcher_stop = True
            await self._tts_task
            if self._outbound_chunks:
                import numpy as np
                import soundfile as sf
                pieces = [sf.read(path, dtype="float32") for path in self._outbound_chunks]
                rates = {rate for _, rate in pieces}
                if len(rates) == 1:
                    sf.write(self.directory / "speaker_combined.wav",
                             np.concatenate([samples for samples, _ in pieces]), rates.pop())
            final_status = "failed" if self._error else "complete"
            summary = {"session_id": self.id, "status": final_status,
                       "error": self._error, "finished_at": utc_now(),
                       "received_audio_s": round(self._received_samples / 16000, 3),
                       "max_input_backlog_s": round(self._max_backlog_s, 3),
                       "server_estimated_capture_during_output_s": round(self._estimated_overlap_s, 3),
                       "utterance_count": self._utterances,
                       "committed_transcripts": self._transcripts,
                       "caller_actions": self._actions,
                       "assistant_text": "".join(self._assistant_text),
                       "speaker_chunks": len(self._outbound_chunks),
                       "context_evaluation": "unknown"}
            (self.directory / "live_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            self._log("live_finished", summary=summary)
            self._set_state(**summary)
        except Exception as exc:
            self._error = f"{type(exc).__name__}: {exc}"
            self._log("live_finish_failed", error=self._error)
            self._set_state(status="failed", error=self._error)
        finally:
            self._watcher_stop = True
            self._input_wav.close()

    def next_playback(self, session_id: str, timeout_s: float = 1.0) -> Path | None:
        if session_id != self._state.get("session_id"):
            return None
        try:
            path = self._playback.get(timeout=timeout_s)
        except queue.Empty:
            with self._state_lock:
                finished = self._state.get("status") in {"complete", "failed"}
            return None if finished else self._silence_path
        self._loop.call_soon_threadsafe(self._mark_sent, path)
        return path

    def _mark_sent(self, path: Path) -> None:
        with wave.open(str(path), "rb") as handle:
            duration_s = handle.getnframes() / handle.getframerate()
        now_ns = time.monotonic_ns()
        start_ns = max(now_ns, self._outbound_windows[-1][1] if self._outbound_windows else now_ns)
        self._outbound_windows.append((start_ns, start_ns + round(duration_s * 1e9)))
        self._log("speech_chunk_sent_to_browser", audio_file=str(path),
                  duration_s=round(duration_s, 3),
                  estimated_playback_start_ns=start_ns)

    def snapshot(self) -> dict[str, Any]:
        with self._state_lock:
            return dict(self._state)

    def annotate(self, session_id: str, verdict: str, overlap: str, notes: str) -> str:
        if session_id != self._state.get("session_id"):
            return "Session ID does not match."
        if self._state.get("status") not in {"complete", "failed"}:
            return "Stop the session and wait for it to finish before saving a verdict."
        if verdict not in {"correct", "incorrect", "no audible answer", "unclear"}:
            return "Choose what you heard."
        if overlap not in {"yes", "no", "unsure"}:
            return "Record whether you spoke while hearing MiniCPM."
        record = {"session_id": session_id, "at": utc_now(), "verdict": verdict,
                  "spoke_while_assistant_audible": overlap, "notes": notes}
        with (self.directory / "human_verdicts.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        return "Listening verdict saved."

    def wait_finished(self, timeout_s: float = 120.0) -> dict[str, Any]:
        if not self._started:
            raise RuntimeError("no live session was started")
        self.stop()
        self._submit(asyncio.wait_for(self._finish_task, timeout_s), timeout=timeout_s + 5)
        return self.snapshot()

    def close(self) -> None:
        if self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)


class RestartableLiveExperiment:
    """Create a clean native and tool session after each completed trial.

    Model weights are owned by the caller's factory and may be reused. Each
    trial gets a fresh controller, router, native session, and output directory.
    """

    def __init__(self, factory: Callable[[], LiveDuplexExperiment]) -> None:
        self.factory = factory
        self.current: LiveDuplexExperiment | None = None
        self.sessions: dict[str, LiveDuplexExperiment] = {}
        self._lock = threading.RLock()

    def start(self) -> str:
        with self._lock:
            if self.current is not None:
                state = self.current.snapshot()
                if state.get("status") in {"connecting", "running", "finishing"}:
                    return str(state["session_id"])
            experiment = self.factory()
            try:
                session_id = experiment.start()
            except Exception:
                experiment.close()
                raise
            self.current = experiment
            self.sessions[session_id] = experiment
            return session_id

    def _session(self, session_id: str) -> LiveDuplexExperiment | None:
        with self._lock:
            return self.sessions.get(session_id)

    def receive(self, session_id: str, audio: tuple[int, Any] | None) -> str:
        experiment = self._session(session_id)
        return experiment.receive(session_id, audio) if experiment else "Start a live session first."

    def stop(self, session_id: str | None = None) -> str:
        experiment = self._session(session_id) if session_id else self.current
        return experiment.stop(session_id) if experiment else "No live session has started."

    def next_playback(self, session_id: str) -> Path | None:
        experiment = self._session(session_id)
        return experiment.next_playback(session_id) if experiment else None

    def snapshot(self) -> dict[str, Any]:
        return self.current.snapshot() if self.current else {"status": "ready", "session_id": None}

    def annotate(self, session_id: str, verdict: str, overlap: str, notes: str) -> str:
        experiment = self._session(session_id)
        return (experiment.annotate(session_id, verdict, overlap, notes) if experiment
                else "Session ID does not match.")

    def close(self) -> None:
        with self._lock:
            experiments = list(self.sessions.values())
        for experiment in experiments:
            experiment.close()


def make_live_gradio_ui(experiment: LiveDuplexExperiment | RestartableLiveExperiment):
    """Short microphone requests and independent long-lived audio playback."""
    import gradio as gr

    def start():
        session_id = experiment.start()
        return session_id, ("Connecting to MiniCPM. When status says running, click the microphone's own record button."
                            if experiment.snapshot().get("status") == "connecting" else
                            "Session running. Click the microphone's own record button.")

    def microphone_started(session_id):
        return ("Recording microphone audio." if session_id else
                "Start a live session before recording.")

    def receive(audio, session_id):
        return experiment.receive(session_id, audio)

    def stop(session_id):
        return experiment.stop(session_id)

    def play(session_id):
        while True:
            path = experiment.next_playback(session_id)
            if path is None:
                return
            yield str(path)

    def poll():
        snapshot = experiment.snapshot()
        return (snapshot.get("status", "ready"), snapshot.get("last_transcript", ""),
                snapshot.get("assistant_text", ""),
                str(snapshot.get("input_backlog_s", 0)),
                json.dumps(snapshot, indent=2))

    with gr.Blocks(title="Live duplex tool experiment") as app:
        gr.Markdown("# Live MiniCPM tool experiment\nPress **Start live session**. When status says **running**, click the **record button inside the microphone panel**. The separate Begin button has been removed because the browser must start its own microphone. You can pause and resume recording without ending the model session. To finish, stop the microphone in its panel, then press **Stop and finish**. A later Start creates a new native session. Use headphones while testing overlap.")
        session_id = gr.State("")
        with gr.Row():
            microphone = gr.Audio(sources=["microphone"], type="numpy", streaming=True,
                                  label="Live microphone")
            speaker = gr.Audio(streaming=True, autoplay=True, label="MiniCPM live speech")
        start_button = gr.Button("Start live session", variant="primary")
        stop_button = gr.Button("Stop and finish")
        ingest_status = gr.Textbox(label="Microphone status", interactive=False)
        status = gr.Textbox(label="Session status", value="ready", interactive=False)
        transcript = gr.Textbox(label="Latest committed user speech", interactive=False)
        assistant = gr.Textbox(label="Assistant text", interactive=False)
        backlog = gr.Textbox(label="Input backlog in seconds", interactive=False)
        trace = gr.Code(label="Latest event and status", language="json")
        start_event = start_button.click(start, outputs=[session_id, status],
                                         queue=False, concurrency_limit=1)
        start_event.then(play, inputs=[session_id], outputs=[speaker],
                         concurrency_id="live_playback", concurrency_limit=1,
                         show_progress="hidden")
        microphone.start_recording(microphone_started, inputs=[session_id],
                                   outputs=[ingest_status], queue=False, show_progress="hidden")
        microphone.stream(receive, inputs=[microphone, session_id], outputs=[ingest_status],
                          stream_every=0.5, time_limit=3600, queue=False,
                          trigger_mode="multiple", concurrency_limit=1,
                          show_progress="hidden")
        stop_button.click(stop, inputs=[session_id], outputs=[status],
                          queue=False, show_progress="hidden")
        gr.Timer(value=1.0, active=True).tick(
            poll, outputs=[status, transcript, assistant, backlog, trace],
            queue=False, show_progress="hidden")
        verdict = gr.Radio(["correct", "incorrect", "no audible answer", "unclear"],
                           label="Was the spoken answer correct?")
        overlap = gr.Radio(["yes", "no", "unsure"],
                           label="Did you speak while MiniCPM was audible?")
        notes = gr.Textbox(label="Exact words heard and any correction you made")
        saved = gr.Textbox(label="Verdict status")
        gr.Button("Save listening verdict").click(
            experiment.annotate, inputs=[session_id, verdict, overlap, notes], outputs=[saved])
    return app


def make_transport_smoke_ui(directory: Path):
    """CPU-only check of the same simultaneous Gradio mic/speaker transport."""
    import gradio as gr

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    silence = directory / "silence.wav"
    write_wav(silence, bytes(32000))
    playback: queue.Queue[Path] = queue.Queue()
    running = threading.Event()
    buffer = bytearray()
    guard = threading.Lock()
    counter = 0

    def start():
        running.set()
        return gr.Audio(recording=True), "Recording. You should hear your speech returned while recording stays on."

    def receive(audio):
        nonlocal counter
        if audio is None or not running.is_set():
            return "Waiting for microphone."
        pcm = microphone_pcm(audio)
        with guard:
            buffer.extend(pcm)
            while len(buffer) >= 32000:
                counter += 1
                path = directory / f"echo_{counter:04d}.wav"
                write_wav(path, bytes(buffer[:32000]))
                del buffer[:32000]
                playback.put(path)
        return f"Received {len(pcm) // 2 / 16000:.2f}s microphone audio."

    def play():
        while running.is_set():
            try:
                path = playback.get(timeout=1.0)
            except queue.Empty:
                path = silence
            yield str(path)

    def stop():
        running.clear()
        return gr.Audio(recording=False), "Stopped. If echo played without stopping capture, the browser transport worked."

    with gr.Blocks(title="Live audio transport check") as app:
        gr.Markdown("# CPU-only live audio check\nUse headphones. Start once, speak for several seconds, and confirm that audio plays back while the microphone is still recording. This tests the browser transport before loading models or using GPU time.")
        with gr.Row():
            mic = gr.Audio(sources=["microphone"], type="numpy", streaming=True,
                           label="Live microphone")
            speaker = gr.Audio(streaming=True, autoplay=True, label="Echo while recording")
        status = gr.Textbox(label="Status", value="Ready")
        received = gr.Textbox(label="Chunk receipt")
        start_event = gr.Button("Start transport check", variant="primary").click(
            start, outputs=[mic, status], queue=False)
        start_event.then(play, outputs=[speaker], concurrency_id="smoke_playback",
                         concurrency_limit=1, show_progress="hidden")
        mic.stream(receive, inputs=[mic], outputs=[received], stream_every=0.5,
                   time_limit=120, queue=False, trigger_mode="multiple",
                   concurrency_limit=1, show_progress="hidden")
        gr.Button("Stop").click(stop, outputs=[mic, status], queue=False)
        mic.stop_recording(stop, outputs=[mic, status], queue=False)
    return app
