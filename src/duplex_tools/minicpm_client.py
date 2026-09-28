"""Minimal serialized client for llama.cpp-omni's legacy streaming HTTP API."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .context_events import Boundary
from .contracts import AdapterCapabilities, Observation
from .controller import ContextController


class OmniHttpError(RuntimeError):
    def __init__(self, message: str, *, uncertain: bool = True) -> None:
        super().__init__(message)
        self.uncertain = uncertain


@dataclass(slots=True)
class OmniHttpClient:
    base_url: str = "http://127.0.0.1:9060"
    timeout_s: float = 120.0

    def post(self, path: str, payload: dict[str, Any]) -> tuple[str, str]:
        request = Request(
            self.base_url.rstrip("/") + path,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_s) as response:
                return response.headers.get_content_type(), response.read().decode()
        except (HTTPError, URLError, TimeoutError) as exc:
            raise OmniHttpError(str(exc)) from exc

    def post_sse(self, path: str, payload: dict[str, Any],
                 on_event: Callable[[dict[str, Any]], None]) -> None:
        """Read one native decode response as events instead of buffering it."""
        request = Request(
            self.base_url.rstrip("/") + path,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_s) as response:
                if response.headers.get_content_type() != "text/event-stream":
                    raise OmniHttpError("decode did not return an event stream", uncertain=False)
                for raw_line in response:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if line == "data: [DONE]":
                        return
                    if not line.startswith("data: "):
                        continue
                    try:
                        event = json.loads(line[6:])
                    except json.JSONDecodeError:
                        continue
                    if isinstance(event, dict):
                        on_event(event)
        except (HTTPError, URLError, TimeoutError) as exc:
            raise OmniHttpError(str(exc)) from exc


class MiniCPMStreamSession:
    """The only owner allowed to issue prefill/decode state mutations."""

    def __init__(
        self,
        client: OmniHttpClient,
        controller: ContextController,
        *,
        max_context_chars: int = 640,
    ) -> None:
        self.client = client
        self.controller = controller
        self.max_context_chars = max_context_chars
        self._model_lock = asyncio.Lock()
        self._last_counter = 0
        self._initialized = False
        self._indeterminate = False
        self._observations: asyncio.Queue[Observation] = asyncio.Queue()

    def capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities(True, False, True, False, True)

    @property
    def next_counter(self) -> int:
        return self._last_counter + 1

    async def observe(self) -> Observation:
        return await self._observations.get()

    async def submit_context(self, event: Any) -> None:
        # The adapter accepts context for the next prefill boundary; it cannot
        # send a stand-alone text turn to this runtime.
        if event.event_id not in {item.event_id for item in self.controller.scheduler.pending()}:
            result = self.controller.scheduler.enqueue(event)
            if result.decision not in {"accepted", "duplicate"}:
                raise ValueError(f"context was rejected: {result.decision}")

    async def cancel_context(self, event_id: str) -> bool:
        return self.controller.scheduler.withdraw(event_id)

    async def initialize(
        self,
        *,
        output_dir: str,
        model_dir: str = "",
        tts_bin_dir: str = "",
        voice_audio: str = "",
        use_tts: bool = True,
        tts_gpu_layers: int = 99,
        token2wav_device: str = "gpu",
    ) -> dict[str, Any]:
        async with self._model_lock:
            if self._initialized:
                raise ValueError("session is already initialized; reinitialization would reset context")
            content_type, body = await asyncio.to_thread(
                self.client.post,
                "/v1/stream/omni_init",
                {
                    "media_type": 1,
                    "use_tts": use_tts,
                    "duplex_mode": True,
                    "tts_gpu_layers": tts_gpu_layers,
                    "token2wav_device": token2wav_device,
                    "output_dir": output_dir,
                    "voice_audio": voice_audio,
                    **({"model_dir": model_dir} if model_dir else {}),
                    **({"tts_bin_dir": tts_bin_dir} if tts_bin_dir else {}),
                },
            )
            parsed = json.loads(body) if content_type == "application/json" else {"body": body}
            if not parsed.get("success", False) or parsed.get("next_cnt") != 1:
                raise OmniHttpError(f"omni_init was rejected: {parsed}")
            self._initialized = True
            return parsed

    async def prefill(
        self,
        *,
        audio_path: str,
        counter: int,
        boundary: Boundary,
        image_path: str = "",
    ) -> dict[str, Any]:
        if not self._initialized:
            raise ValueError("session must be initialized before prefill")
        async with self._model_lock:
            if self._indeterminate:
                raise RuntimeError("prefill outcome is unknown; session requires manual reconciliation")
            if counter != self._last_counter + 1:
                raise ValueError(f"prefill counter must increase by one: expected {self._last_counter + 1}, got {counter}")
            event = self.controller.reserve_injection(boundary, model_idle=True)
            payload: dict[str, Any] = {
                "audio_path_prefix": audio_path,
                "img_path_prefix": image_path,
                "cnt": counter,
            }
            if event:
                payload["text"] = event.injection_text(self.max_context_chars)
            try:
                content_type, body = await asyncio.to_thread(
                    self.client.post, "/v1/stream/prefill", payload
                )
                parsed = json.loads(body) if content_type == "application/json" else {"body": body}
                if not parsed.get("success", False):
                    raise OmniHttpError(f"prefill was rejected: {parsed}", uncertain=False)
            except Exception as exc:
                if event:
                    if isinstance(exc, OmniHttpError) and not exc.uncertain:
                        self.controller.release_injection(event.event_id, reason="prefill_rejected")
                    else:
                        self.controller.commit_injection(event.event_id, counter=counter)
                        self.controller.log.write("context_submission_unknown", event_id=event.event_id, counter=counter)
                if not (isinstance(exc, OmniHttpError) and not exc.uncertain):
                    self._indeterminate = True
                raise
            self._last_counter = counter
            if event:
                self.controller.commit_injection(event.event_id, counter=counter)
                await self._observations.put(Observation("context_submitted", event_id=event.event_id))
            return parsed

    async def decode(self, *, debug_dir: str, round_idx: int = -1) -> str:
        if not self._initialized:
            raise ValueError("session must be initialized before decode")
        async with self._model_lock:
            _, body = await asyncio.to_thread(
                self.client.post,
                "/v1/stream/decode",
                {"debug_dir": debug_dir, "round_idx": round_idx, "stream": True},
            )
            for line in body.splitlines():
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                try:
                    data = json.loads(line[6:])
                except json.JSONDecodeError:
                    continue
                if data.get("content"):
                    await self._observations.put(Observation("assistant_text", value=str(data["content"])))
                if "is_listen" in data:
                    await self._observations.put(Observation("listening" if data["is_listen"] else "speaking", value=bool(data["is_listen"])))
            return body

    async def decode_stream(self, *, debug_dir: str, round_idx: int = -1,
                            on_event: Callable[[dict[str, Any]], None] | None = None) -> None:
        """Expose text/listen events as they arrive while preserving input order."""
        if not self._initialized:
            raise ValueError("session must be initialized before decode")
        async with self._model_lock:
            loop = asyncio.get_running_loop()
            events: asyncio.Queue[dict[str, Any] | BaseException | None] = asyncio.Queue()

            def worker() -> None:
                try:
                    self.client.post_sse(
                        "/v1/stream/decode",
                        {"debug_dir": debug_dir, "round_idx": round_idx, "stream": True},
                        lambda event: loop.call_soon_threadsafe(events.put_nowait, event),
                    )
                except BaseException as exc:
                    loop.call_soon_threadsafe(events.put_nowait, exc)
                finally:
                    loop.call_soon_threadsafe(events.put_nowait, None)

            task = asyncio.create_task(asyncio.to_thread(worker))
            while True:
                event = await events.get()
                if event is None:
                    break
                if isinstance(event, BaseException):
                    await task
                    raise event
                if event.get("content"):
                    await self._observations.put(Observation("assistant_text", value=str(event["content"])))
                if "is_listen" in event:
                    await self._observations.put(Observation(
                        "listening" if event["is_listen"] else "speaking",
                        value=bool(event["is_listen"]),
                    ))
                if on_event is not None:
                    on_event(event)
            await task
