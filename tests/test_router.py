from __future__ import annotations

import unittest
from datetime import datetime, timezone

from duplex_tools.caller import GraniteToolCaller, parse_granite_output
from duplex_tools.contracts import CallerAction, TranscriptSegment
from duplex_tools.controller import ContextController
from duplex_tools.conversation import ConversationRouter
from duplex_tools.tools import RoomLookup, SafeCalculator, TOOL_SCHEMAS


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
        self.assertFalse(controller.log.records)

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


class CallerTests(unittest.TestCase):
    def test_native_tool_call_and_non_call(self):
        call = parse_granite_output('<tool_call>{"name":"calculator","arguments":{"expression":"2+3"}}</tool_call><|end_of_text|>')
        self.assertEqual(call.tool, "calculator")
        string_args = parse_granite_output('<tool_call>{"name":"calculator","arguments":"{\\"expression\\":\\"2+3\\"}"}</tool_call><|end_of_text|>')
        self.assertEqual(string_args.arguments, {"expression": "2+3"})
        self.assertEqual(parse_granite_output('{"action":"none"}').kind, "none")
        self.assertEqual(parse_granite_output("I might call a tool").kind, "invalid")
        self.assertEqual(parse_granite_output('<tool_call>{"name":"room_lookup","arguments":{"name":"event"}}</tool_call>. Request ID: made-up').kind, "invalid")

    def test_latest_speech_is_the_final_user_turn(self):
        import asyncio
        seen = []

        def generate(messages, tools):
            seen.extend(messages)
            return '{"action":"none"}'

        caller = GraniteToolCaller(generate)
        result = asyncio.run(caller.decide(segment(), {}, TOOL_SCHEMAS))
        self.assertEqual(result.kind, "none")
        self.assertEqual(seen[-1], {"role": "user", "content": "Find the robotics seminar room"})

    def test_safe_calculator_rejects_code(self):
        import asyncio
        calc = SafeCalculator()
        self.assertEqual(asyncio.run(calc({"expression": "2 * (3 + 4)"})).facts, "2 * (3 + 4) = 14.")
        with self.assertRaises(ValueError):
            asyncio.run(calc({"expression": "__import__('os').system('id')"}))


if __name__ == "__main__":
    unittest.main()
