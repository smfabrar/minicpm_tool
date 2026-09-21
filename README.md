# Duplex Tools, first extraction

This repository extracts the reusable tool path from the MiniCPM experiments into a Python package. The core has no MiniCPM dependency. MiniCPM is the first real adapter, and `SimulatedAdapter` exercises the portable contract. The original model runtime and weights are separate; the revision-specific input patch is in `fixtures/context-injection.patch` for Kaggle reproduction.

## Run on Kaggle

Open [`notebooks/kaggle_duplex_tools.ipynb`](notebooks/kaggle_duplex_tools.ipynb) in Kaggle. Enable Internet. Set `SOURCE_REF` in the first code cell and run the CPU phase. After enabling a GPU and Kaggle restarts the runtime, rerun the tag and setup cells. The GPU phase builds the pinned patched `llama.cpp-omni`, locates or downloads the audio-only GGUF modules, starts the server, and opens a password-protected microphone UI. Ask about the robotics seminar and listen for **B742**. The UI records the transcript, selected action, HTTP submission, model text, audio file, and your listening verdict.

The notebook clones this repository over HTTPS at tag `v0.1.12`; Kaggle needs no SSH key. A Kaggle Dataset containing the required GGUF folder can be attached to save GPU-time downloads. The fallback downloads the official modules from Hugging Face. The caller and speech recognizer run on CPU in the GPU phase, leaving VRAM for MiniCPM.

The Kaggle CUDA build sets `GGML_CUDA_NO_VMM=ON`. Kaggle can expose the CUDA runtime and cuBLAS without the unversioned driver library needed by CMake's `CUDA::cuda_driver` target. Disabling ggml's virtual memory management removes that direct driver dependency while keeping CUDA kernels and GPU layer offload enabled.

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
| `voice_demo.py` | Human microphone, CPU transcription, audio chunking, trace, listening verdict |

Core installation and tests use no model dependencies:

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
```

The schema validator allows exactly one required string field per tool and rejects extras. A provisional segment never executes. An amendment must name an existing pending request. Late results for old versions and results after cancellation are discarded. The model worker never receives a tool result directly from the tool task; the session owner takes a queued event at a safe audio input boundary.

## Provenance and limits

The MiniCPM patch targets `tc-mb/llama.cpp-omni` commit `64d092c60db4b4ee45768476bd752f03fdcc98ea`. The patch includes the existing server initialization changes required for audio-only startup and the `next_cnt` response. It applies cleanly to a pristine pinned checkout and the resulting runtime builds locally. See [`EVIDENCE.md`](EVIDENCE.md) for the earlier context-injection observation. No MiniCPM source or weights are committed here.

The first extraction does not yet have a live streaming microphone loop, incremental SSE playback, native KV evaluation acknowledgement, or a human correction while a tool runs. Those require the subsequent integration and overlap stages. The checked-in notebook is prepared and syntax-checked locally; its GPU and microphone cells must be run in Kaggle to establish the human result.

The observed Granite pilot failures, safety boundaries, and measurement policy are recorded in [`docs/caller_failure_analysis.md`](docs/caller_failure_analysis.md).

## Primary references

- [Kaggle notebook settings](https://www.kaggle.com/docs/notebooks)
- [IBM Granite 4.0 350M model card](https://huggingface.co/ibm-granite/granite-4.0-350m)
- [IBM Granite 4.0 tool prompt guide](https://github.com/ibm-granite/granite-4.0-language-models/blob/main/Granite%204.0%20Prompt%20engineering%20guide%20v2.md)
- [OpenBMB MiniCPM-o 4.5 GGUF modules](https://huggingface.co/openbmb/MiniCPM-o-4_5-gguf)
