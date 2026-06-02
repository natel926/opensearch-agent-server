"""
Unit tests verifying ToolSafetyHook registration and prompt hardening
on ART specialized sub-agents.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agents.art import specialized_agents
from utils.safety_hook import ToolSafetyHook

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def restore_mcp_tools():
    """Restore module-level _mcp_tools after each test to avoid cross-test pollution."""
    original_tools = specialized_agents._mcp_tools
    yield
    specialized_agents._mcp_tools = original_tools


@pytest.mark.parametrize(
    "agent_func_name, expected_prompt_const",
    [
        ("hypothesis_agent", "HYPOTHESIS_GENERATOR_SYSTEM_PROMPT"),
        ("evaluation_agent", "EVALUATION_AGENT_SYSTEM_PROMPT"),
        ("user_behavior_analysis_agent", "USER_BEHAVIOR_ANALYSIS_AGENT_SYSTEM_PROMPT"),
    ],
)
async def test_sub_agent_registers_safety_hook(agent_func_name, expected_prompt_const):
    # 1. Configure module-level _mcp_tools so the early-return guard doesn't fire.
    specialized_agents._mcp_tools = [MagicMock(name="fake_mcp_tool")]

    # 2. Stub BedrockModel + Agent to capture constructor kwargs.
    with (
        patch.object(specialized_agents, "BedrockModel"),
        patch.object(specialized_agents, "Agent") as mock_agent_cls,
    ):
        mock_agent_instance = MagicMock()

        # invoke_async is awaited — return an awaitable.
        async def _fake_invoke(_q):
            return "ok"

        mock_agent_instance.invoke_async = _fake_invoke
        mock_agent_cls.return_value = mock_agent_instance

        # Find the underlying coroutine (handle the @monitored_tool wrapping).
        # monitored_tool uses functools.wraps, so __wrapped__ is the async_wrapper
        # which calls the original coroutine function.
        decorated = getattr(specialized_agents, agent_func_name)
        underlying = getattr(decorated, "__wrapped__", decorated)
        await underlying("test query")

        # Assert Agent(...) was called with our hook and a hardened prompt.
        assert mock_agent_cls.call_count == 1, (
            f"{agent_func_name}: Agent constructor was not called"
        )
        kwargs = mock_agent_cls.call_args.kwargs
        hooks = kwargs.get("hooks", [])
        assert any(isinstance(h, ToolSafetyHook) for h in hooks), (
            f"{agent_func_name} did not register ToolSafetyHook (hooks={hooks!r})"
        )

        prompt = kwargs.get("system_prompt", "")
        assert "[CONFIRMATION_REQUIRED]" in prompt, (
            f"{agent_func_name} system prompt missing hardening paragraph"
        )
