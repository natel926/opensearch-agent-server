"""Strands ``BeforeToolCallEvent`` hook gating destructive MCP tool calls.

Per-request approval state is held in a ``ContextVar`` populated by the
agent orchestrator from ``RunAgentInput.forwarded_props.approved_tool_calls``
and read by ``ToolSafetyHook`` when deciding whether to cancel a destructive
call.
"""

from __future__ import annotations

import contextvars
from collections.abc import Iterable

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
