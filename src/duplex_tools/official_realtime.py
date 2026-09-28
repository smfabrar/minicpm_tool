"""Official MiniCPM-o Realtime API client with the project's tool extension.

The model owns speech recognition, listen/speak decisions, and speech synthesis.
Whisper and Granite receive a duplicate microphone stream solely for external
tool selection. A versioned tool result enters the native model on a later
one-second audio unit through the small, documented ``tool_context`` patch.
"""

from __future__ import annotations

import asyncio
import base64
import json
import time
import uuid
import wave
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .context_events import Boundary
from .live_duplex import LiveDuplexExperiment, SpeechSegmenter, utc_now, write_wav


def wav_float32_b64(path: Path) -> str:
    import numpy as np
    import soundfile as sf

    samples, rate = sf.read(path, dtype="float32", always_2d=True)
    mono = samples.mean(axis=1)
    if rate != 16000:
        mono = np.interp(
            np.arange(round(len(mono) * 16000 / rate)) / 16000,
            np.arange(len(mono)) / rate, mono,
        ).astype("<f4")
    return base64.b64encode(np.asarray(mono, dtype="<f4").tobytes()).decode("ascii")


class OfficialRealtimeExperiment(LiveDuplexExperiment):
    """One live session through OpenBMB's ``/v1/realtime?mode=audio``."""

    def __init__(self, *args: Any, gateway_url: str, reference_audio: Path,
                 system_prompt: str = "You are a helpful voice assistant.", **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.gateway_url = gateway_url
        self.reference_audio = Path(reference_audio)
        self.system_prompt = system_prompt

    async def _start(self) -> str:
        import websockets

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
        self._outbound_windows = deque()
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
        self._finish_task = None
        self._error = None
        self._awaiting_post_result = False
        self._post_result_text = ""
        self._sent_units = 0
        self._completed_response_ids: set[str] = set()
        self._max_native_pending_units = 0
        self._injected: dict[int, str] = {}
        self._acknowledged: set[str] = set()
        self._send_lock = asyncio.Lock()
        self._closed_event = asyncio.Event()
        self._silence_path = self.directory / "one_second_silence.wav"
        write_wav(self._silence_path, bytes(32000))
        settings = {"session_id": self.id, "started_at": utc_now(),
                    "protocol": "/v1/realtime?mode=audio", "input_unit_s": 1.0,
                    "transport_chunk_target_s": 0.5, "vad_rms": self.vad_rms,
                    "vad_pause_s": self.vad_pause_s, "max_backlog_s": self.max_backlog_s,
                    "final_silence_units": self.final_silence_units,
                    "reference_audio": str(self.reference_audio),
                    "native_context_evaluation_ack": True}
        (self.directory / "live_manifest.json").write_text(json.dumps(settings, indent=2) + "\n")
        self._set_state(status="connecting", session_id=self.id, started_at=settings["started_at"],
                        input_backlog_s=0.0, received_audio_s=0.0, last_transcript="",
                        assistant_text="", error=None)
        self._log("live_started", settings=settings)
        try:
            self._ws = await websockets.connect(self.gateway_url, max_size=128 * 1024 * 1024,
                                                open_timeout=30, close_timeout=10)
            while True:
                event = json.loads(await asyncio.wait_for(self._ws.recv(), 120))
                self._log("realtime_control", event=event)
                if event.get("type") == "session.queue_done":
                    break
                if event.get("type") in {"session.closed", "error"}:
                    raise RuntimeError(f"realtime queue failed: {event}")
            voice = wav_float32_b64(self.reference_audio)
            await self._ws.send(json.dumps({"type": "session.init", "payload": {
                "system_prompt": self.system_prompt, "config": {"length_penalty": 1.1},
                "voice": {"ref_audio_base64": voice, "tts_ref_audio_base64": voice},
            }}))
            while True:
                event = json.loads(await asyncio.wait_for(self._ws.recv(), 180))
                self._log("realtime_control", event={k: v for k, v in event.items() if k != "audio"})
                if event.get("type") == "session.created":
                    self._native_session_id = event.get("session_id")
                    break
                if event.get("type") in {"session.closed", "error"}:
                    raise RuntimeError(f"realtime initialization failed: {event}")
        except Exception:
            self._input_wav.close()
            raise
        self._model_task = asyncio.create_task(self._model_loop())
        self._route_task = asyncio.create_task(self._route_loop())
        self._tts_task = asyncio.create_task(self._read_realtime())
        self._set_state(status="running", native_session_id=self._native_session_id)
        return self.id

    async def _send_unit(self, pcm: bytes, *, microphone: bool,
                         actual_samples: int = 16000) -> None:
        import numpy as np

        self._sent_units += 1
        counter = self._sent_units
        self._max_native_pending_units = max(
            self._max_native_pending_units,
            self._sent_units - len(self._completed_response_ids),
        )
        path = self.directory / f"input_{counter:06d}.wav"
        write_wav(path, pcm)
        event = self.router.controller.reserve_injection(Boundary.UNIT, model_idle=True)
        payload: dict[str, Any] = {"audio": base64.b64encode(
            (np.frombuffer(pcm, dtype="<i2").astype("<f4") / 32768.0).astype("<f4").tobytes()
        ).decode("ascii")}
        if event is not None:
            context = event.injection_text(600)
            payload["tool_context"] = context.encode("utf-8")[:640].decode("utf-8", "ignore")
        self._log("native_audio_send_started", unit=counter, audio_file=str(path),
                  microphone=microphone, tool_event_id=event.event_id if event else None)
        try:
            async with self._send_lock:
                await self._ws.send(json.dumps({"type": "input.append", "input": payload}))
        except Exception:
            if event is not None:
                self.router.controller.release_injection(event.event_id, reason="websocket_send_failed")
            raise
        if microphone:
            self._prefilled_samples += actual_samples
        if event is not None:
            self.router.controller.commit_injection(event.event_id, counter=counter)
            self._injected[counter] = event.event_id
            self._awaiting_post_result = True
            self._post_result_text = ""
        self._log("native_audio_sent", unit=counter, tool_event_id=event.event_id if event else None)
        self._set_state(input_backlog_s=round(
            (self._received_samples - self._prefilled_samples) / 16000, 2))

    async def _read_realtime(self) -> None:
        import numpy as np

        try:
            async for raw in self._ws:
                event = json.loads(raw)
                kind = event.get("kind")
                response_id = str(event.get("response_id") or "")
                logged = {key: value for key, value in event.items() if key not in {"audio", "audio_data"}}
                self._log("native_event", event=logged)
                if event.get("type") == "tool_context.evaluated":
                    unit = int(event.get("unit", -1))
                    event_id = self._injected.get(unit)
                    if event_id and event.get("ok"):
                        self._acknowledged.add(event_id)
                        self._log("context_evaluated", unit=unit, event_id=event_id)
                    elif event_id:
                        self._error = f"Native tool context evaluation failed at unit {unit}"
                if event.get("type") == "response.output.delta":
                    if kind == "text":
                        fragment = str(event.get("text") or "")
                        self._assistant_text.append(fragment)
                        if self._awaiting_post_result:
                            self._post_result_text += fragment
                        self._set_state(assistant_text="".join(self._assistant_text)[-1000:])
                    elif kind == "audio" and event.get("audio"):
                        samples = np.frombuffer(base64.b64decode(event["audio"]), dtype="<f4")
                        destination = self.directory / f"speaker_{len(self._outbound_chunks) + 1:04d}.wav"
                        write_wav(destination, (np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes(), 24000)
                        self._outbound_chunks.append(destination)
                        self._playback.put(destination)
                        self._log("speech_chunk_ready", audio_file=str(destination),
                                  duration_s=round(len(samples) / 24000, 3), response_id=response_id)
                    elif kind == "listen" and response_id:
                        self._completed_response_ids.add(response_id)
                if event.get("type") == "response.done" and response_id:
                    self._completed_response_ids.add(response_id)
                    if self._awaiting_post_result and self._post_result_text:
                        self._awaiting_post_result = False
                self._max_native_pending_units = max(
                    self._max_native_pending_units,
                    self._sent_units - len(self._completed_response_ids),
                )
                if event.get("type") == "session.closed":
                    self._closed_event.set()
                    return
        except Exception as exc:
            self._error = f"Realtime connection failed: {type(exc).__name__}: {exc}"
            self._log("native_connection_failed", error=self._error)
        finally:
            self._closed_event.set()

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
                try:
                    await asyncio.wait_for(self._wait_native_drain(), timeout=45)
                except asyncio.TimeoutError:
                    self._error = "Native responses did not drain within 45 seconds"
            if not self._closed_event.is_set():
                await asyncio.sleep(1.0)  # Allow delayed audio deltas to arrive.
                async with self._send_lock:
                    await self._ws.send(json.dumps({"type": "session.close", "reason": "user_stop"}))
                try:
                    await asyncio.wait_for(self._closed_event.wait(), timeout=15)
                except asyncio.TimeoutError:
                    self._error = self._error or "Native session close was not acknowledged"
            await self._ws.close()
            await self._tts_task
            if self._outbound_chunks:
                import numpy as np
                import soundfile as sf
                chunks = [sf.read(path, dtype="float32")[0] for path in self._outbound_chunks]
                sf.write(self.directory / "speaker_combined.wav", np.concatenate(chunks), 24000)
            missing_ack = sorted(set(self._injected.values()) - self._acknowledged)
            if missing_ack:
                self._error = self._error or f"No native evaluation acknowledgement for {len(missing_ack)} tool result(s)"
            summary = {"session_id": self.id, "native_session_id": self._native_session_id,
                       "status": "failed" if self._error else "complete", "error": self._error,
                       "finished_at": utc_now(), "received_audio_s": round(self._received_samples / 16000, 3),
                       "max_input_backlog_s": round(self._max_backlog_s, 3),
                       "max_native_pending_units": self._max_native_pending_units,
                       "server_estimated_capture_during_output_s": round(self._estimated_overlap_s, 3),
                       "utterance_count": self._utterances, "committed_transcripts": self._transcripts,
                       "caller_actions": self._actions, "assistant_text": "".join(self._assistant_text),
                       "speaker_chunks": len(self._outbound_chunks),
                       "tool_context_sent": len(self._injected), "tool_context_evaluated": len(self._acknowledged),
                       "missing_evaluation_acks": missing_ack}
            (self.directory / "live_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            self._log("live_finished", summary=summary)
            self._set_state(**summary)
        except Exception as exc:
            self._error = f"{type(exc).__name__}: {exc}"
            self._log("live_finish_failed", error=self._error)
            summary = {"session_id": self.id, "status": "failed", "error": self._error,
                       "finished_at": utc_now(), "received_audio_s": round(self._received_samples / 16000, 3),
                       "tool_context_sent": len(self._injected),
                       "tool_context_evaluated": len(self._acknowledged)}
            (self.directory / "live_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            self._set_state(**summary)
        finally:
            self._input_wav.close()
            if hasattr(self, "_ws"):
                await self._ws.close()

    async def _wait_native_drain(self) -> None:
        while self._sent_units > len(self._completed_response_ids):
            if self._closed_event.is_set():
                raise RuntimeError("Native session closed before all input units were processed")
            await asyncio.sleep(0.1)
