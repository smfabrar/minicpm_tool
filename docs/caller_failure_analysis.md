# Caller failure analysis and decision record

## Status of the six-case pilot

The staged Granite 4.0 350M caller produced four strictly correct cases:

| Case | Route | Raw argument | Interpretation |
|---|---|---|---|
| named room | correct | correct | successful call |
| calculator | correct | `17 times 23` | grounded but outside the calculator grammar |
| document search | correct | correct | successful call |
| ordinary thanks | correct | none | successful no-call |
| incomplete room question | incorrect: room lookup | invented `event` | unsafe proposal that must not execute |
| missing seminar name | correct: clarify | none | successful clarification |

This is a pilot and prompt-development set, not thesis evaluation evidence. Repeated prompt changes based on these six cases overfit them. Paraphrase families used for development must be kept out of the frozen evaluation split.

These caller errors do not block the first human MiniCPM experiment. The admission condition is narrower: the three intended calls must be executable after explicitly reported normalization, and unintended or ungrounded proposals must be rejected before tool execution. This lets the end-to-end experiment proceed without relabelling either caller error as correct.

## Why the two failures differ

The calculator failure is a representation mismatch. The model understood the action and numbers, but copied the spoken operator. The safe calculator accepts an explicit arithmetic grammar so that arbitrary code cannot execute. The controller now canonicalizes a small declared vocabulary (`times`, `multiplied by`, `divided by`, `plus`, `minus`) into symbols. Logs retain both raw and canonical arguments. Thesis results must report raw argument correctness and executable argument correctness separately.

The incomplete-speech failure is a routing and grounding error. The label scorer gave lexical evidence such as “room” too much weight and the argument generator invented `event`. Prompting alone is not a safety boundary. The controller now requires room names and document queries to occur in the committed transcript. An ungrounded proposal is logged and becomes a clarification; no tool starts. This lowers unsafe execution independently of caller accuracy and must not be counted as a correct model decision.

## Staged caller design

The original one-pass native tool prompt caused the small model to call tools for ordinary speech and copy few-shot values. The staged design separates two measurements:

1. Granite scores closed action labels from its next-token probabilities.
2. Only for a selected tool, Granite receives that single schema and extracts arguments using its native tool format.
3. The framework validates allowlisting, shape, grounding, and tool-specific grammar before execution.

Closed-label scores now come from one forward pass when every label is one token. This should reduce routing latency substantially relative to one full pass per label. The raw trace includes every label log probability and the winning margin. No confidence threshold is chosen from the six pilot cases. A threshold must be calibrated on a larger development set, then evaluated once on a held-out split.

## Metrics that must remain separate

- Route/action accuracy by class.
- Raw argument exact and semantic correctness.
- Canonical executable argument correctness.
- Proposal rejection rate and reason.
- Unsupported or ungrounded execution rate.
- End-to-end tool result correctness.
- Caller latency, tool latency, context-submission latency, and audible-answer latency.

A safety filter can make unsupported execution zero while model route accuracy remains imperfect. Both numbers must be reported. A correct tool trace also does not establish a correct spoken answer.

## Next model decision

Before selecting a caller, build a development set with complete calls, ordinary speech, incomplete speech, missing details, corrections, and cancellations. Compare the staged 350M caller with the explicit rule baseline and at least one stronger or tool-specialized small model. Keep schemas, prompts, hardware, decoding, and cases fixed. Select based on accuracy, unsafe proposals, latency, and memory. Fine-tuning is justified only after these pretrained comparisons identify persistent errors.
