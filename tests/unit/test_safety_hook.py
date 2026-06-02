"""Unit tests for the tool-call safety hook and approval ContextVar."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock

import pytest

from utils.safety_hook import (
    ToolSafetyHook,
    approval_context,
    reset_approved_calls,
    set_approved_calls,
)
from utils.tool_safety import canonical_call_hash

pytestmark = pytest.mark.unit


class TestApprovalContext:
    def test_default_is_empty_frozenset(self) -> None:
        assert approval_context.get() == frozenset()

    def test_set_and_reset(self) -> None:
        token = set_approved_calls(["a", "b"])
        try:
            assert approval_context.get() == frozenset({"a", "b"})
        finally:
            reset_approved_calls(token)
        assert approval_context.get() == frozenset()

    def test_set_with_none_yields_empty(self) -> None:
        token = set_approved_calls(None)
        try:
            assert approval_context.get() == frozenset()
        finally:
            reset_approved_calls(token)

    def test_set_with_empty_list_yields_empty(self) -> None:
        token = set_approved_calls([])
        try:
            assert approval_context.get() == frozenset()
        finally:
            reset_approved_calls(token)

    @pytest.mark.asyncio
    async def test_isolated_per_async_task(self) -> None:
        async def worker(approvals: list[str]) -> frozenset[str]:
            token = set_approved_calls(approvals)
            try:
                await asyncio.sleep(0)
                return approval_context.get()
            finally:
                reset_approved_calls(token)

        a, b = await asyncio.gather(worker(["x"]), worker(["y"]))
        assert a == frozenset({"x"})
        assert b == frozenset({"y"})
        assert approval_context.get() == frozenset()


# ---------------------------------------------------------------------------
# Task 3: ToolSafetyHook behaviour tests
# ---------------------------------------------------------------------------


@dataclass
class _FakeEvent:
    """Stand-in for ``strands.hooks.events.BeforeToolCallEvent``.

    The real event is a frozen dataclass with restricted writes; for the hook
    under test we only need ``tool_use`` and a writable ``cancel_tool``.
    """

    tool_use: dict[str, Any]
    cancel_tool: bool | str = False
    selected_tool: Any = None
    invocation_state: dict[str, Any] = field(default_factory=dict)


def _make_hook_callback() -> tuple[ToolSafetyHook, Any]:
    hook = ToolSafetyHook()
    registry = MagicMock()
    captured: dict[str, Any] = {}

    def add_callback(event_cls, cb):
        captured["cb"] = cb

    registry.add_callback.side_effect = add_callback
    hook.register_hooks(registry)
    return hook, captured["cb"]


class TestToolSafetyHookRead:
    def test_read_tool_left_untouched(self) -> None:
        _, cb = _make_hook_callback()
        event = _FakeEvent(tool_use={"name": "search", "input": {}, "toolUseId": "t1"})
        cb(event)
        assert event.cancel_tool is False


class TestToolSafetyHookWrite:
    def test_write_tool_not_cancelled(self, caplog: Any) -> None:
        caplog.set_level(logging.INFO)
        _, cb = _make_hook_callback()
        event = _FakeEvent(
            tool_use={
                "name": "create_index",
                "input": {"index": "x"},
                "toolUseId": "t2",
            },
        )
        cb(event)
        assert event.cancel_tool is False
        assert any(
            "write_executed" in r.message
            or getattr(r, "event", "") == "tool_safety.write_executed"
            for r in caplog.records
        )


class TestToolSafetyHookDestructiveUnapproved:
    def test_destructive_without_approval_is_cancelled(self) -> None:
        _, cb = _make_hook_callback()
        params = {"index": "logs-2026"}
        event = _FakeEvent(
            tool_use={"name": "delete_index", "input": params, "toolUseId": "t3"},
        )
        cb(event)
        assert isinstance(event.cancel_tool, str)
        assert event.cancel_tool.startswith("[CONFIRMATION_REQUIRED] ")
        payload = json.loads(event.cancel_tool[len("[CONFIRMATION_REQUIRED] ") :])
        assert payload["tool_name"] == "delete_index"
        assert payload["tool_use_id"] == "t3"
        assert payload["params"] == params
        assert payload["risk"] == "destructive"
        assert payload["approval_hash"] == canonical_call_hash("delete_index", params)


class TestToolSafetyHookDestructiveApproved:
    def test_destructive_with_matching_hash_executes(self) -> None:
        _, cb = _make_hook_callback()
        params = {"index": "logs-2026"}
        approved = canonical_call_hash("delete_index", params)
        token = set_approved_calls([approved])
        try:
            event = _FakeEvent(
                tool_use={"name": "delete_index", "input": params, "toolUseId": "t4"},
            )
            cb(event)
        finally:
            reset_approved_calls(token)
        assert event.cancel_tool is False

    def test_destructive_with_mismatched_hash_is_cancelled(self) -> None:
        _, cb = _make_hook_callback()
        wrong_hash = canonical_call_hash("delete_index", {"index": "other"})
        token = set_approved_calls([wrong_hash])
        try:
            event = _FakeEvent(
                tool_use={
                    "name": "delete_index",
                    "input": {"index": "logs-2026"},
                    "toolUseId": "t5",
                },
            )
            cb(event)
        finally:
            reset_approved_calls(token)
        assert isinstance(event.cancel_tool, str)
        assert event.cancel_tool.startswith("[CONFIRMATION_REQUIRED] ")
