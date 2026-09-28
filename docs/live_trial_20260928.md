# First exported native live trial: 28 September 2026

The saved run `20260928T152222Z_8a6619` contains session `20260928T160227Z_9dcbbc`. Its manifest records a pinned MiniCPM backend with the tool extension and Python UI adapter `v0.1.23`. The participant reported frequent audible pauses between words. This note separates what the logs prove from what only the participant could hear.

## Evidence from the saved run

| Observation | What it means |
| --- | --- |
| 203 seconds of microphone audio became 203 native one-second input units. | Continuous audio reached the backend during the session. The largest reported adapter input backlog was one second. |
| MiniCPM returned 69 audio chunks totaling 63.64 seconds. | These are the source WAV chunks saved as `speaker_*.wav`; `speaker_combined.wav` joins them without transport silence. |
| Among 59 nearby pairs of source chunks, the median production gap after the previous chunk's playback duration was 0.286 seconds; 31 gaps exceeded 0.2 seconds and 13 exceeded 0.5 seconds. | Native output itself was often slower than uninterrupted playback. Pairs whose start times were at least three seconds apart are excluded as likely pauses between turns. |
| The median interval from saving a source chunk to offering it to Gradio was 0.001 seconds. | The adapter's queue was usually prompt. This does not measure Gradio tunnel transfer or actual browser playback. |
| 406 microphone callbacks each carried 0.5 seconds of audio, but they spanned about 285 seconds of wall time: roughly 71% of real-time pace. The one-second native units were sent at roughly the same pace. | Audio was arriving at the adapter too slowly to keep the model continuously supplied. The saved files do not reveal whether Gradio omitted intervals, delayed them, or whether browser capture was intermittent. |
| The first native response after each sent unit arrived in a median 0.179 seconds; backend audio encoding, LLM prefill, and LLM decode medians were approximately 22, 44, and 97 ms. | The GPU model was often waiting for input or other serial stages. Two GPUs holding model memory do not imply two independently busy inference workers. |
| The old UI returned a one-second silence WAV whenever its queue stayed empty for one second. | This could add audible pauses on top of the native generation gaps. The number of those silence files played was not logged. |
| The backend emitted successful `tool_context.evaluated` events for units 19 and 192. The Python client had placed those contexts on units 18 and 191. | The result reached the C++ model, but the client's off-by-one matching recorded zero acknowledgements. |
| The final native response ID ended in `resp_203`, matching all 203 client input units. There were 140 `listen` events and only 52 `response.done` events. | The old finalizer incorrectly demanded one completed response ID for every audio unit and produced a false 45-second drain error. The [official audio protocol](https://github.com/OpenBMB/MiniCPM-o-Demo/blob/main/docs-app/content/docs/en/realtime-api/audio.md) says `response.done` is not guaranteed for each output turn. |
| The backend log repeatedly reported `n_ctx=0`, `slide FORCE`, and `slide DONE`; after the first tool context was evaluated, it removed recent tokens within seconds. | The notebook omitted an explicit context-size argument. The pinned backend's duplex sliding-window code treated the resulting zero as a trigger to slide on virtually every unit. This could prevent a tool result from remaining available for a later spoken answer. |

The Whisper sidecar committed “Where is that? Or seminar?” and Granite requested `room_lookup` for `seminar`. That lookup failed because it lacked a specific seminar name. MiniCPM then gave an unrelated, invented seminar answer. Later, the sidecar committed “No, I say what is 17 times 3”; the calculator returned `17 * 3 = 51`, its tool context was evaluated by the backend, and MiniCPM said 51. This run does not demonstrate a correct robotics room lookup, a spoken correction to the vision seminar, or the intended 17 × 23 calculation. The speech and tool quality verdict remains a failure despite successful evaluation of two context strings.

## Changes for the next trial

1. Match the backend's tool acknowledgement unit to the client unit with the observed one-unit offset. Record both numbers in `context_evaluated`.
2. Finish after the last numbered native response, all expected tool acknowledgements, and two quiet seconds. Keep a 45-second timeout for genuinely stalled output. The response-ID suffix is an assumption about the pinned backend, recorded in the new run manifest; it is not a general Realtime API promise.
3. Wait for actual speech instead of adding one second of silence to the Gradio output queue. Yield WAV bytes as recommended by [Gradio's Audio documentation](https://gradio.app/docs/gradio/audio). This removes one adapter source of pauses. The native source gaps may remain audible because the trial prioritizes live delivery over holding several seconds of speech in a buffer.
4. Run `python scripts/analyze_live_export.py <session-directory>` or use notebook cell 13. Listen to `speaker_combined.wav` and compare it with the live page. Save a human verdict for the next run.
5. Pass `-c 4096` explicitly when starting the native backend. This is the upstream documented default-sized context and avoids the observed zero reaching duplex sliding-window logic. It changes a server launch argument, so restart the server, but it does not require rebuilding the saved binary.
6. The CPU-only echo page now reports microphone media time divided by wall time. If it is below 90%, treat the browser transport as unfit for a real-time trial before spending GPU time. A WebRTC or direct audio WebSocket path would then need to replace Gradio's event-based microphone streaming.

These changes are in the Python adapter and notebook; the saved C++ runtime bundle and model weights do not need to be rebuilt for them. The updated Gradio playback and finalization still need an actual Kaggle trial.
