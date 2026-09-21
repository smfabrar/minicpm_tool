from __future__ import annotations

import unittest
from datetime import datetime, timezone
from unittest.mock import patch

from duplex_tools.caller import ChoiceScores, GraniteToolCaller, parse_granite_output
from duplex_tools.contracts import CallerAction, TranscriptSegment
from duplex_tools.controller import ContextController
from duplex_tools.conversation import ConversationRouter
from duplex_tools.tools import RoomLookup, SafeCalculator, TOOL_SCHEMAS, normalize_calculator_expression


class FixedCaller:
    def __init__(self, action: CallerAction) -> None:
        self.action = action

    async def decide(self, segment, pending, tools):
        return self.action


def segment(committed=True):
    return TranscriptSegment("segment-1", 1, datetime.now(timezone.utc), "Find the robotics seminar room", committed)


class RouterTests(unittest.IsolatedAsyncioTestCase):
    async def test_provisional_and_bad_arguments_never_execute(self):
        controller = ContextController({"room_lookup": RoomLookup({"robotics seminar": "B742"})})
        router = ConversationRouter(FixedCaller(CallerAction("call", "room_lookup", {"name": "robotics seminar", "delay_s": "1"})), controller)
        self.assertEqual((await router.ingest(segment(False))).kind, "none")
        self.assertEqual((await router.ingest(segment())).kind, "clarify")
        self.assertFalse(any(item["kind"] == "tool_started" for item in controller.log.records))
        self.assertTrue(any(item["kind"] == "call_rejected" for item in controller.log.records))

    async def test_valid_call_and_cancel_before_result(self):
        controller = ContextController({"room_lookup": RoomLookup({"robotics seminar": "B742"})})
        caller = FixedCaller(CallerAction("call", "room_lookup", {"name": "robotics seminar"}))
        router = ConversationRouter(caller, controller)
        action = await router.ingest(segment())
        self.assertEqual(action.kind, "call")
        self.assertEqual(action.tool, "room_lookup")
        caller.action = CallerAction("cancel", request_id=action.request_id)
        await router.ingest(TranscriptSegment("segment-2", 1, datetime.now(timezone.utc), "Cancel that", True))
        await controller.wait_all()
        self.assertFalse(controller.scheduler.pending())

    async def test_unknown_amendment_is_clarification(self):
        controller = ContextController({"room_lookup": RoomLookup({"robotics seminar": "B742"})})
        router = ConversationRouter(FixedCaller(CallerAction("amend", "room_lookup", {"name": "robotics seminar"}, "missing")), controller)
        self.assertEqual((await router.ingest(segment())).kind, "clarify")

    async def test_ungrounded_room_argument_never_executes(self):
        controller = ContextController({"room_lookup": RoomLookup({"event": "B742"})})
        router = ConversationRouter(FixedCaller(CallerAction("call", "room_lookup", {"name": "event"})), controller)
        action = await router.ingest(TranscriptSegment("incomplete", 1, datetime.now(timezone.utc), "What room is the", True))
        self.assertEqual(action.kind, "clarify")
        self.assertFalse(any(item["kind"] == "tool_started" for item in controller.log.records))
        self.assertTrue(any(item["kind"] == "call_rejected" for item in controller.log.records))

    async def test_spoken_calculator_operator_is_normalized_and_logged(self):
        controller = ContextController({"calculator": SafeCalculator()})
        router = ConversationRouter(FixedCaller(CallerAction("call", "calculator", {"expression": "17 times 23"})), controller)
        action = await router.ingest(TranscriptSegment("math", 1, datetime.now(timezone.utc), "What is 17 times 23?", True))
        self.assertEqual(action.arguments, {"expression": "17 * 23"})
        await controller.wait_all()
        self.assertTrue(any(item["kind"] == "call_normalized" for item in controller.log.records))


class CallerTests(unittest.TestCase):
    def test_native_tool_call_and_non_call(self):
        call = parse_granite_output('<tool_call>{"name":"calculator","arguments":{"expression":"2+3"}}</tool_call><|end_of_text|>')
        self.assertEqual(call.tool, "calculator")
        string_args = parse_granite_output('<tool_call>{"name":"calculator","arguments":"{\\"expression\\":\\"2+3\\"}"}</tool_call><|end_of_text|>')
        self.assertEqual(string_args.arguments, {"expression": "2+3"})
        self.assertEqual(parse_granite_output('{"action":"none"}').kind, "none")
        self.assertEqual(parse_granite_output("I might call a tool").kind, "invalid")
        self.assertEqual(parse_granite_output('<tool_call>{"name":"room_lookup","arguments":{"name":"event"}}</tool_call>. Request ID: made-up').kind, "invalid")
        missing_close = parse_granite_output('<tool_call>{"name":"calculator","arguments":{"expression":"2+3"}}<|end_of_text|>')
        self.assertEqual(missing_close.arguments, {"expression": "2+3"})

    def test_latest_speech_is_the_final_user_turn(self):
        import asyncio
        seen = []

        def generate(messages, tools):
            return '{"action":"none"}'

        def choose(generator, messages, choices):
            seen.extend(messages)
            return ChoiceScores("A", {"A": -0.1, "B": -1.0}, 0.9)

        caller = GraniteToolCaller(generate)
        with patch("duplex_tools.caller.score_choices", side_effect=choose):
            result = asyncio.run(caller.decide(segment(), {}, TOOL_SCHEMAS))
        self.assertEqual(result.kind, "none")
        self.assertIn('"margin": 0.9', result.raw)
        self.assertIn("Latest speech: Find the robotics seminar room", seen[-1]["content"])

    def test_selected_tool_is_only_schema_used_for_arguments(self):
        import asyncio
        seen_tools = []

        def generate(messages, tools):
            seen_tools.append(tools)
            return '<tool_call>{"name":"room_lookup","arguments":{"name":"robotics seminar"}}'

        caller = GraniteToolCaller(generate)
        with patch("duplex_tools.caller.score_choices", return_value=ChoiceScores("C", {"C": -0.1, "A": -1.0}, 0.9)):
            result = asyncio.run(caller.decide(segment(), {}, TOOL_SCHEMAS))
        self.assertEqual(result.tool, "room_lookup")
        self.assertEqual(len(seen_tools[0]), 1)
        self.assertEqual(seen_tools[0][0]["function"]["name"], "room_lookup")

    def test_safe_calculator_rejects_code(self):
        import asyncio
        calc = SafeCalculator()
        self.assertEqual(asyncio.run(calc({"expression": "2 * (3 + 4)"})).facts, "2 * (3 + 4) = 14.")
        with self.assertRaises(ValueError):
            asyncio.run(calc({"expression": "__import__('os').system('id')"}))

    def test_spoken_operator_normalization_remains_restricted(self):
        self.assertEqual(normalize_calculator_expression("17 times 23"), "17 * 23")
        with self.assertRaises(ValueError):
            normalize_calculator_expression("open the calculator")


if __name__ == "__main__":
    unittest.main()
