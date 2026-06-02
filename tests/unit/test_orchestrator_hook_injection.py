"""Unit tests verifying the orchestrator injects ToolSafetyHook into AGUIStrandsAgent.

ag_ui_strands.StrandsAgent does not propagate hooks from the template Strands
Agent into the per-thread StrandsAgentCore it creates per request.  The
orchestrator compensates by injecting ToolSafetyHook directly into
_agent_kwargs after constructing the AGUIStrandsAgent wrapper so that every
per-thread agent executes with the hook registered.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from ag_ui.core import RunAgentInput

from orchestrator.registry import AgentRegistration, AgentRegistry
from orchestrator.router import PageContextRouter
from server.agent_orchestrator import AgentOrchestrator
from utils.safety_hook import ToolSafetyHook

pytestmark = pytest.mark.unit


def _make_orchestrator_with_real_agui_agent() -> AgentOrchestrator:
    """Build an orchestrator whose factory returns a minimal Strands Agent.

    We patch AGUIStrandsAgent so it doesn't actually call a real model, but
    we still want to verify that the orchestrator injects ToolSafetyHook into
    the wrapper's _agent_kwargs.
    """
    registry = AgentRegistry()
    registry.register(
        AgentRegistration(
            name="default",
            description="",
            page_contexts=[],
            is_default=True,
        )
    )
    router = PageContextRouter(registry)
    orch = AgentOrchestrator(router=router)

    # Use a real (but minimal) Strands Agent as the factory product.
    from strands import Agent as StrandsAgentCore

    template = StrandsAgentCore(system_prompt="test", tools=[])
    orch.register_agent_factory(
        name="default",
        factory=lambda: template,
        description="General assistant",
    )
    return orch


def _make_run_input() -> RunAgentInput:
    return RunAgentInput(
        thread_id="t",
        run_id="r",
        state={},
        messages=[],
        tools=[],
        context=[],
        forwarded_props={},
    )


@pytest.mark.asyncio
async def test_orchestrator_injects_safety_hook_into_agui_agent_kwargs() -> None:
    """The hook MUST appear in _agent_kwargs so per-thread agents register it.

    AGUIStrandsAgent.__init__ does not copy hooks from the template Strands
    Agent. The orchestrator compensates by injecting it directly. Without
    this, every destructive MCP tool call on the default agent path would
    execute unguarded.
    """
    orch = _make_orchestrator_with_real_agui_agent()
    inp = _make_run_input()

    # We need to drive the orchestrator generator far enough to create and
    # cache the agui_agent.  The agui_agent creation block executes before
    # agui_agent.run() is called.  We patch agui_agent.run() *after* the
    # agui_agent is created so that the injection (which happens before
    # agui_agent.run() is called) can still be checked.
    #
    # Strategy: patch AGUIStrandsAgent.run at the class level so it's a
    # no-op async generator, then iterate once.
    async def _noop_run(self, input_data):  # type: ignore[override]
        if False:
            yield  # make it an async generator

    with patch(
        "ag_ui_strands.StrandsAgent.run",
        new=_noop_run,
    ):
        gen2 = orch.run(inp, agent_name="default")
        try:
            await gen2.__anext__()
        except StopAsyncIteration:
            pass
        finally:
            await gen2.aclose()

    cached = orch._cached_agui_agents.get("default")
    assert cached is not None, "AGUIStrandsAgent was not cached by orchestrator"

    hooks = cached._agent_kwargs.get("hooks")
    assert hooks is not None, (
        "_agent_kwargs missing 'hooks' key — ToolSafetyHook was not injected. "
        "Every destructive MCP call executes unguarded on this agent path."
    )
    assert any(isinstance(h, ToolSafetyHook) for h in hooks), (
        f"No ToolSafetyHook found in _agent_kwargs['hooks']={hooks!r}. "
        "The per-thread StrandsAgentCore will have no safety gating."
    )


@pytest.mark.asyncio
async def test_safety_hook_present_for_art_agent() -> None:
    """AGUIStrandsAgent created for the ART agent also gets ToolSafetyHook.

    The wrap-stripping bug applies to any agent routed through AGUIStrandsAgent,
    not just the default agent.
    """
    from strands import Agent as StrandsAgentCore

    registry = AgentRegistry()
    registry.register(
        AgentRegistration(
            name="art",
            description="",
            page_contexts=["search_overview"],
            is_default=False,
        )
    )
    router = PageContextRouter(registry)
    orch = AgentOrchestrator(router=router)

    template = StrandsAgentCore(system_prompt="art-test", tools=[])
    orch.register_agent_factory(
        name="art",
        factory=lambda: template,
        description="ART agent",
    )

    inp = _make_run_input()

    async def _noop_run(self, input_data):  # type: ignore[override]
        if False:
            yield

    with patch("ag_ui_strands.StrandsAgent.run", new=_noop_run):
        gen = orch.run(inp, agent_name="art")
        try:
            await gen.__anext__()
        except StopAsyncIteration:
            pass
        finally:
            await gen.aclose()

    cached = orch._cached_agui_agents.get("art")
    assert cached is not None, "AGUIStrandsAgent was not cached for art agent"

    hooks = cached._agent_kwargs.get("hooks")
    assert hooks is not None, "_agent_kwargs missing 'hooks' key for art agent"
    assert any(isinstance(h, ToolSafetyHook) for h in hooks), (
        f"No ToolSafetyHook found in art agent's _agent_kwargs['hooks']={hooks!r}"
    )
