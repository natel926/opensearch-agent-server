"""Strands ``BeforeToolCallEvent`` hook gating destructive MCP tool calls.

Per-request approval state is held in a ``ContextVar`` populated by the
agent orchestrator from ``RunAgentInput.forwarded_props.approved_tool_calls``
and read by ``ToolSafetyHook`` when deciding whether to cancel a destructive
call.
"""

from __future__ import annotations

import contextvars
import json
from collections.abc import Iterable
from typing import Any

from strands.hooks import HookProvider
from strands.hooks.events import BeforeToolCallEvent

from utils.logging_helpers import get_logger, log_info_event
from utils.tool_safety import RiskLevel, canonical_call_hash, classify_tool_call

approval_context: contextvars.ContextVar[frozenset[str]] = contextvars.ContextVar(
    "approval_context",
    default=frozenset(),
)


def set_approved_calls(
    approvals: Iterable[str] | None,
) -> contextvars.Token[frozenset[str]]:
    """Set the approval context for the current async task.

    Returns the ``Token`` that must be passed to :func:`reset_approved_calls`
    to scope the approval to a single agent run.
    """
    return approval_context.set(frozenset(approvals or ()))


def reset_approved_calls(
    token: contextvars.Token[frozenset[str]],
) -> None:
    """Restore the approval context to its prior value."""
    approval_context.reset(token)


logger = get_logger(__name__)

CONFIRMATION_PREFIX = "[CONFIRMATION_REQUIRED] "


class ToolSafetyHook(HookProvider):
    """Cancel destructive tool calls without an approved canonical hash.

    Reads per-request approval state from :data:`approval_context`. The
    orchestrator populates this ContextVar from
    ``RunAgentInput.forwarded_props.approved_tool_calls`` for the duration
    of a single agent run.
    """

    def register_hooks(self, registry: Any, **_: Any) -> None:
        registry.add_callback(BeforeToolCallEvent, self._on_before_tool_call)

    def _on_before_tool_call(self, event: BeforeToolCallEvent) -> None:
        tool_use = event.tool_use or {}
        tool_name: str = tool_use.get("name", "") or ""
        params: dict[str, Any] = tool_use.get("input") or {}
        tool_use_id: str = tool_use.get("toolUseId", "") or ""

        risk = classify_tool_call(tool_name, params)

        if risk is RiskLevel.READ:
            return

        call_hash = canonical_call_hash(tool_name, params)

        if risk is RiskLevel.WRITE:
            log_info_event(
                logger,
                f"Tool call (write) executed: {tool_name}",
                "tool_safety.write_executed",
                tool_name=tool_name,
                tool_use_id=tool_use_id,
                call_hash=call_hash,
            )
            return

        if call_hash in approval_context.get():
            log_info_event(
                logger,
                f"Destructive tool call approved and executed: {tool_name}",
                "tool_safety.destructive_approved",
                tool_name=tool_name,
                tool_use_id=tool_use_id,
                call_hash=call_hash,
            )
            return

        payload = {
            "tool_name": tool_name,
            "tool_use_id": tool_use_id,
            "params": params,
            "approval_hash": call_hash,
            "risk": "destructive",
            "reason": "Mutates or deletes data; explicit user approval required.",
        }
        log_info_event(
            logger,
            f"Destructive tool call cancelled pending approval: {tool_name}",
            "tool_safety.destructive_cancelled",
            tool_name=tool_name,
            tool_use_id=tool_use_id,
            call_hash=call_hash,
        )
        event.cancel_tool = CONFIRMATION_PREFIX + json.dumps(payload)
