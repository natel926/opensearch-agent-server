"""Unit tests verifying the orchestrator sets/resets the approval ContextVar."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from ag_ui.core import RunAgentInput

from orchestrator.registry import AgentRegistration
from orchestrator.router import PageContextRouter
from server.agent_orchestrator import AgentOrchestrator
from utils.safety_hook import approval_context

pytestmark = pytest.mark.unit


def _make_input(approvals: list[str] | None) -> RunAgentInput:
    forwarded: dict[str, Any] = {}
    if approvals is not None:
        forwarded["approved_tool_calls"] = approvals
    return RunAgentInput(
        thread_id="t",
        run_id="r",
        state={},
        messages=[],
        tools=[],
        context=[],
        forwarded_props=forwarded,
    )


def _make_orchestrator(observed: dict[str, Any]) -> AgentOrchestrator:
    """Build an orchestrator whose underlying agent records the ContextVar."""

    class _FakeAGUIAgent:
        async def run(self, input_data: RunAgentInput):
            observed["during_run"] = approval_context.get()
            if False:
                yield None  # pragma: no cover — make this an async generator

    router = MagicMock(spec=PageContextRouter)
    router.route.return_value = AgentRegistration(
        name="default", description="", page_contexts=[]
    )
    orch = AgentOrchestrator(router=router)
    orch._cached_agui_agents["default"] = _FakeAGUIAgent()  # type: ignore[assignment]
    orch._agent_factories["default"] = {
        "factory": lambda: MagicMock(),
        "description": "",
        "config": None,
    }
    return orch


@pytest.mark.asyncio
async def test_approvals_set_during_run_and_reset_after() -> None:
    observed: dict[str, Any] = {}
    orch = _make_orchestrator(observed)
    inp = _make_input(["hash1", "hash2"])

    async for _ in orch.run(inp):
        pass

    assert observed["during_run"] == frozenset({"hash1", "hash2"})
    assert approval_context.get() == frozenset()


@pytest.mark.asyncio
async def test_missing_forwarded_props_yields_empty_set() -> None:
    observed: dict[str, Any] = {}
    orch = _make_orchestrator(observed)
    inp = _make_input(None)

    async for _ in orch.run(inp):
        pass

    assert observed["during_run"] == frozenset()


@pytest.mark.asyncio
async def test_context_reset_even_on_exception() -> None:
    class _ExplodingAGUIAgent:
        async def run(self, input_data: RunAgentInput):
            yield None
            raise RuntimeError("boom")

    router = MagicMock(spec=PageContextRouter)
    router.route.return_value = AgentRegistration(
        name="default", description="", page_contexts=[]
    )
    orch = AgentOrchestrator(router=router)
    orch._cached_agui_agents["default"] = _ExplodingAGUIAgent()  # type: ignore[assignment]
    orch._agent_factories["default"] = {
        "factory": lambda: MagicMock(),
        "description": "",
        "config": None,
    }
    inp = _make_input(["h"])

    with pytest.raises(RuntimeError, match="boom"):
        async for _ in orch.run(inp):
            pass

    assert approval_context.get() == frozenset()
