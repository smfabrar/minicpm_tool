"""Versioned user speech routing; only committed speech can execute tools."""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from typing import Any

from .contracts import CallerAction, ToolCaller, TranscriptSegment
from .controller import ContextController
from .tools import TOOL_SCHEMAS, validate_call


class ConversationRouter:
    def __init__(self, caller: ToolCaller, controller: ContextController, schemas: list[dict[str, Any]] = TOOL_SCHEMAS) -> None:
        self.caller = caller
        self.controller = controller
        self.schemas = schemas
        self._revisions: dict[str, int] = {}
        self._pending: dict[str, str] = {}

    @property
    def pending(self) -> Mapping[str, str]:
        return dict(self._pending)

    async def ingest(self, segment: TranscriptSegment) -> CallerAction:
        previous = self._revisions.get(segment.segment_id, 0)
        if segment.revision <= previous:
            return CallerAction("none", message="duplicate or old transcript revision")
        if not segment.committed:
            return CallerAction("none", message="provisional speech")
        self._revisions[segment.segment_id] = segment.revision
        action = await self.caller.decide(segment, self.pending, self.schemas)
        if action.kind in {"call", "amend"}:
            try:
                args = validate_call(action.tool or "", action.arguments, self.schemas)
                if action.tool not in self.controller.tools:
                    raise ValueError("tool has no implementation")
                if action.kind == "amend":
                    if not action.request_id or action.request_id not in self._pending:
                        raise ValueError("amendment requires an unambiguous pending request ID")
                    request_id = action.request_id
                else:
                    request_id = f"req-{uuid.uuid4().hex[:12]}"
                self.controller.submit_tool(request_id, action.tool or "", args)
                self._pending[request_id] = segment.text
                return CallerAction(action.kind, action.tool, args, request_id, raw=action.raw)
            except (ValueError, TypeError) as exc:
                return CallerAction("clarify", message=str(exc), raw=action.raw)
        if action.kind == "cancel":
            if not action.request_id or action.request_id not in self._pending:
                return CallerAction("clarify", message="which pending request should be cancelled?", raw=action.raw)
            self.controller.cancel(action.request_id)
            self._pending.pop(action.request_id)
        return action
