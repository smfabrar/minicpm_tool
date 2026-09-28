"""Summarize timing from a saved live session without loading any models."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def analyze(session_dir: Path) -> dict:
    events = [json.loads(line) for line in (session_dir / "live_events.jsonl").read_text().splitlines()]
    summary = json.loads((session_dir / "live_summary.json").read_text())
    ready = [event for event in events if event["kind"] == "speech_chunk_ready"]
    dispatched = [event for event in events if event["kind"] == "speech_chunk_sent_to_browser"]
    native = [event["event"] for event in events if event["kind"] == "native_event"]
    injections = [event for event in events if event["kind"] == "native_audio_sent"
                  and event.get("tool_event_id")]
    acks = [event for event in native if event.get("type") == "tool_context.evaluated"]

    nearby_gaps = []
    for earlier, later in zip(ready, ready[1:]):
        start_spacing_s = (later["monotonic_ns"] - earlier["monotonic_ns"]) / 1e9
        gap_s = start_spacing_s - earlier["duration_s"]
        if start_spacing_s < 3.0:  # Exclude pauses between separate spoken turns.
            nearby_gaps.append(gap_s)

    dispatch_delays = []
    if len(ready) == len(dispatched):
        dispatch_delays = [(sent["monotonic_ns"] - made["monotonic_ns"]) / 1e9
                           for made, sent in zip(ready, dispatched)]

    return {
        "session_id": summary.get("session_id"),
        "status": summary.get("status"),
        "error": summary.get("error"),
        "microphone_audio_s": summary.get("received_audio_s"),
        "native_input_units": sum(event["kind"] == "native_audio_sent" for event in events),
        "native_response_ids_seen": len({event.get("response_id") for event in native
                                         if event.get("response_id")}),
        "native_listen_events": sum(event.get("kind") == "listen" for event in native),
        "native_response_done_events": sum(event.get("type") == "response.done" for event in native),
        "native_audio_chunks": len(ready),
        "native_audio_duration_s": round(sum(event["duration_s"] for event in ready), 3),
        "nearby_chunk_gap_count": len(nearby_gaps),
        "nearby_chunk_gap_median_s": (round(statistics.median(nearby_gaps), 3)
                                      if nearby_gaps else None),
        "nearby_chunk_gaps_over_200ms": sum(gap > 0.2 for gap in nearby_gaps),
        "nearby_chunk_gaps_over_500ms": sum(gap > 0.5 for gap in nearby_gaps),
        "browser_dispatch_delay_median_s": (round(statistics.median(dispatch_delays), 3)
                                            if dispatch_delays else None),
        "tool_injection_client_units": [event["unit"] for event in injections],
        "tool_ack_backend_units": [event.get("unit") for event in acks],
        "tool_ack_successes": sum(event.get("ok") is True for event in acks),
        "note": ("Browser dispatch timing is measured at the adapter; it does not prove "
                 "when sound became audible. The nearby-gap metric excludes starts "
                 "at least three seconds apart."),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("session_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(analyze(args.session_dir), indent=2))
