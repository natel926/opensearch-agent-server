"""Unit tests for the tool-call safety hook and approval ContextVar."""

from __future__ import annotations

import asyncio

import pytest

from utils.safety_hook import (
    approval_context,
    reset_approved_calls,
    set_approved_calls,
)

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
