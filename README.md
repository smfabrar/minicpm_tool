# Duplex Tools, first extraction

This repository extracts the reusable tool path from the MiniCPM experiments into a Python package. The core has no MiniCPM dependency. MiniCPM is the first real adapter, and `SimulatedAdapter` exercises the portable contract. The original model runtime and weights are separate; the revision-specific input patch is in `fixtures/context-injection.patch` for Kaggle reproduction.

## Live duplex experiment

[`notebooks/kaggle_live_duplex.ipynb`](notebooks/kaggle_live_duplex.ipynb) is the next human experiment. It pulls adapter tag `v0.1.23` and uses the pinned [official MiniCPM-o Demo Realtime API](https://github.com/OpenBMB/MiniCPM-o-Demo/blob/main/docs-app/content/docs/en/realtime-api/audio.md): gateway → worker → C++ MiniCPM backend. The browser sends microphone chunks continuously; the adapter sends one-second 16 kHz float32 units to one native WebSocket session. MiniCPM itself decides when to listen or speak and produces 24 kHz speech. Whisper Tiny and Granite 350M run beside it only to select and execute external tools. A small, separately pinned C++ extension delivers a verified tool result at a native audio boundary and acknowledges evaluation. The notebook adds five seconds to room lookup so a spoken correction can be tested while a request is pending.

The run saves microphone and input-unit WAVs, committed utterances, caller actions, controller decisions, native model text and speech deltas, timing, official service logs, and a human listening verdict in one export ZIP. See [`docs/live_duplex_protocol.md`](docs/live_duplex_protocol.md) for the exact test and acceptance criteria.

This prepares a live overlap test; no Kaggle live result has been observed yet. The browser bridge is Gradio because Kaggle cannot expose the official gateway UI directly. Backlog and timestamps show whether a run kept pace. A native `tool_context.evaluated` event reports when injected text was evaluated; it does not prove the spoken answer was correct or audible, so the listener records that separately.

## Controlled-turn notebook on Kaggle

Open [`notebooks/kaggle_duplex_tools.ipynb`](notebooks/kaggle_duplex_tools.ipynb) in Kaggle. Enable Internet. Set `SOURCE_REF` in the first code cell and run the CPU phase. After enabling a GPU and Kaggle restarts the runtime, rerun the tag and setup cells. The GPU phase builds the pinned patched `llama.cpp-omni`, locates or downloads the audio-only GGUF modules, starts the server, and opens a password-protected microphone UI. Ask about the robotics seminar and listen for **B742**. The UI records the transcript, selected action, HTTP submission, model text, audio file, and your listening verdict.

The notebook clones this repository over HTTPS at tag `v0.1.19`; Kaggle needs no SSH key. A Kaggle Dataset containing the required GGUF folder can be attached to save GPU-time downloads. The fallback downloads the official modules from Hugging Face. During the human experiment, Granite 350M and Whisper Tiny run in FP16 on GPU 1 while the native MiniCPM runtime can use both GPUs.

The human UI submits each turn to a dedicated event-loop thread owned by the voice adapter and polls it with short requests. This keeps a long MiniCPM turn independent of both the public Gradio queue connection and Gradio's request-loop lifecycle. Its latest stage and result are also written to the current run's `latest_turn.json`. Granite routing has a 90-second deadline so a stalled caller is recorded as an experimental failure.

The 2026-09-26 human run is analyzed in [`docs/human_run_2026-09-26.md`](docs/human_run_2026-09-26.md). It showed successful tool selection, execution, and context submission, but assistant speech often preceded the current result and later turns repeated earlier answers. The new voice path logs text before and after submission separately and continues a bounded number of silence input units after delivery. A current-turn answer must appear after the current result and be confirmed by the listener. These changes still require a fresh Kaggle voice run; local tests only verify the ordering and trace logic.

Each native server start now gets a unique directory under `/kaggle/working/duplex_voice_runs/` with a manifest, server log, submitted recordings, context log, TTS chunks, combined audio, and verdicts. Notebook cell 14 stops the writers and creates one downloadable ZIP for that run. The trial ID is shared across the submitted recording, trace, combined audio, and listening verdict. Declared `room_lookup` aliases handle the observed Whisper transcription “robotic seminar” while preserving the canonical room source.

The controller explicitly removes completed tool tasks after waiting for them, avoiding an event-loop spin if a scheduled done callback has not run yet. The Kaggle server starts in its own process session, so interrupting an unrelated notebook cell does not propagate SIGINT to the initialized native runtime.

For an existing Kaggle session, use the updated notebook cells: rerun 1–2, then 9–13 when models and dependencies are already prepared. Cell 9 stops the tracked server and Gradio app, repairs the existing native checkout, and rebuilds incrementally. Cell 10 refreshes the runtime bundle; cells 11–13 start a fresh server, conversation, and UI. Recorded files remain on disk. Changing `SOURCE_REF` alone updates Python modules but does not replace code cells already copied into Kaggle.

The Kaggle CUDA build sets `GGML_CUDA_NO_VMM=ON`. Kaggle can expose the CUDA runtime and cuBLAS without the unversioned driver library needed by CMake's `CUDA::cuda_driver` target. Disabling ggml's virtual memory management removes that direct driver dependency while keeping CUDA kernels and GPU layer offload enabled.

The localhost server is built with `LLAMA_OPENSSL=OFF`: this upstream constructs an SSL listener whenever OpenSSL is enabled, even without certificates. The native context patch also finishes each chunked HTTP decode response with `sink.done()` and captures request parameters by value. Previously the HTTP library could repeatedly invoke the response callback, decoding indefinitely while Python waited for EOF. `scripts/repair_runtime.py` applies these two SSE edits to the pinned existing checkout and rebuilds incrementally using the same build directory. CUDA objects are retained. Re-export the runtime afterward; the updated patch hash intentionally rejects older artifacts. The voice UI reports each stage and audio chunk, with elapsed times in `controller.jsonl`.

Run the real HTTP transport regression without models or a GPU:

```bash
python3 scripts/check_sse_transport.py --upstream /path/to/llama.cpp-omni
```

It applies the full patch to pristine pinned source in a temporary directory, compiles the actual SSE callback against the vendored HTTP library with a simulated decoder, and checks that two HTTP requests finish with exactly two decodes. This verifies the response lifecycle; model inference and audible answers still require Kaggle testing.

After compilation, the notebook creates `/kaggle/working/minicpm_omni_runtime_sm75`. It contains the server, project shared libraries, context patch, provenance, toolchain and Python versions, an `ldd` report, and checksums. Quick Save the notebook output after the experiment, then attach that output and set `ATTACHED_RUNTIME_DIR` in a future session to skip compilation. The build contains T4 `sm_75` machine code; a different GPU architecture requires another artifact. Model weights remain a separate Kaggle Dataset.

The six-case CPU probe characterizes the caller; it is not a demand for perfect model accuracy before the real experiment. GPU admission checks that intended calls remain executable after declared normalization and that invalid or ungrounded proposals cannot reach a tool. Raw caller mistakes are still reported separately and remain failures in the thesis results.

The microphone UI records one turn at a time. It is a human gate for real tool selection and spoken response in a persistent session. It does not establish continuous full-duplex overlap. `context_submitted` means HTTP prefill accepted the payload; the runtime HTTP API has no evaluation acknowledgement. The trace therefore says `evaluation: unknown`.

## Package layout

| Module | Role |
|---|---|
| `contracts.py` | Versioned user transcripts, caller actions, adapter capabilities, observations, protocols |
| `conversation.py` | Committed-transcript gate, request association, call validation |
| `controller.py`, `context_events.py` | Async tool execution, request versions, timeout, expiry, bounded result scheduling, logging |
| `tools.py` | Local room lookup, safe arithmetic, document search and explicit schemas |
| `caller.py` | Granite native `<tool_call>` parsing and optional Transformers backend |
| `minicpm_client.py` | Persistent serialized HTTP session, monotonic counter, context attached at an audio boundary |
| `simulated.py` | Capability-controlled contract test adapter |
| `voice_demo.py` | Human microphone, transcription, audio chunking, result timing trace, listening verdict |

Core installation and tests use no model dependencies:

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
```

The schema validator allows exactly one required string field per tool and rejects extras. A provisional segment never executes. An amendment must name an existing pending request. Late results for old versions and results after cancellation are discarded. The model worker never receives a tool result directly from the tool task; the session owner takes a queued event at a safe audio input boundary.

## Provenance and limits

The MiniCPM patch targets `tc-mb/llama.cpp-omni` commit `64d092c60db4b4ee45768476bd752f03fdcc98ea`. The patch includes the existing server initialization changes required for audio-only startup and the `next_cnt` response. It applies cleanly to a pristine pinned checkout and the resulting runtime builds locally. See [`EVIDENCE.md`](EVIDENCE.md) for the earlier context-injection observation. No MiniCPM source or weights are committed here.

The controlled HTTP notebook above remains the original completed-turn experiment. The separate live notebook now has a native WebSocket microphone loop and a tool-context evaluation acknowledgement, but its GPU build and human interaction still need to be run in Kaggle to establish a research result.

The observed Granite pilot failures, safety boundaries, and measurement policy are recorded in [`docs/caller_failure_analysis.md`](docs/caller_failure_analysis.md).

## Primary references

- [Kaggle notebook settings](https://www.kaggle.com/docs/notebooks)
- [IBM Granite 4.0 350M model card](https://huggingface.co/ibm-granite/granite-4.0-350m)
- [IBM Granite 4.0 tool prompt guide](https://github.com/ibm-granite/granite-4.0-language-models/blob/main/Granite%204.0%20Prompt%20engineering%20guide%20v2.md)
- [OpenBMB MiniCPM-o 4.5 GGUF modules](https://huggingface.co/openbmb/MiniCPM-o-4_5-gguf)
