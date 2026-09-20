# Context-injection evidence retained from the source workspace

The source workspace tested an audio-only persistent MiniCPM-o 4.5 session using a modified `llama.cpp-omni` runtime. Its duplex request and packet queues carry `user_text` to the single inference worker. The observed processing order was existing conversation, input unit, injected text, audio embeddings, then decode. The fused prefill path declined packets containing text so ordinary prefill handled them.

An injected `Room=B742.` fact produced the spoken answer “The room code is B7 42.” A later correction influenced an answer in the same session. Other trials truncated or omitted requested information. The result proves context injection is feasible, not reliable answer accuracy or real-time overlap.

The patch in `fixtures/context-injection.patch` applies to upstream revision `64d092c60db4b4ee45768476bd752f03fdcc98ea`. The source workspace kept the full evidence index and audio artifacts at `experiments/minicpm_context_injection/evidence/` and `experiments/minicpm_context_injection/artifacts/`; those large research artifacts are not part of this Python package. The native optional context evaluation callback was not exposed by the HTTP API, so successful HTTP prefill establishes submission only.
