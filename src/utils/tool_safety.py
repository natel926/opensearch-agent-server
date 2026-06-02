"""Tool-call risk classification and canonical hashing.

Pure functions with no I/O or state. Used by ``ToolSafetyHook`` to gate
destructive MCP tool calls behind explicit user approval.
"""

from __future__ import annotations

import enum
import hashlib
import json
import re
from typing import Any


class RiskLevel(enum.Enum):
    """Risk tier for an MCP tool call.

    READ: latency-neutral, never gated.
    WRITE: audit-logged, executed.
    DESTRUCTIVE: requires per-call approval via canonical hash.
    """

    READ = "read"
    WRITE = "write"
    DESTRUCTIVE = "destructive"


_DESTRUCTIVE_PREFIXES = ("delete_", "drop_", "purge_")
_DESTRUCTIVE_EXACT = frozenset(
    {
        "truncate",
        "force_merge",
        "close_index",
    }
)
_WRITE_PREFIXES = (
    "create_",
    "put_",
    "update_",
    "set_",
    "index_",
    "bulk",
    "restore_",
)
_WRITE_EXACT = frozenset(
    {
        "reindex",
        "snapshot",
    }
)
_READ_PREFIXES = (
    "get_",
    "list_",
    "describe",
    "cat_",
)
_READ_EXACT = frozenset(
    {
        "search",
        "cluster_health",
        "cluster_state",
        "count",
        "msearch",
        "scroll",
        "exists",
    }
)
_GENERIC_REQUEST_TOOLS = frozenset({"opensearch_request", "http_request"})
# Bare \* intentionally has no segment-boundary anchor so index-wildcard
# patterns like /logs-*/_settings are caught regardless of position.
# _all is anchored to segment boundaries to avoid matching path substrings
# like /my-index_all.
_WILDCARD_PATH_RE = re.compile(r"\*|(^|/)_all(/|$)")


def classify_tool_call(tool_name: str, params: dict[str, Any] | None) -> RiskLevel:
    """Classify an MCP tool call by risk tier.

    Order matters: destructive checks come first, then writes, then reads,
    with WRITE as the deny-by-default fallback for unknown names so the
    hook still audits unrecognised tools without latency cost on reads.
    """
    name = (tool_name or "").lower()

    if name in _GENERIC_REQUEST_TOOLS:
        return _classify_generic_request(params or {})

    if any(name.startswith(pfx) for pfx in _DESTRUCTIVE_PREFIXES):
        return RiskLevel.DESTRUCTIVE
    if name in _DESTRUCTIVE_EXACT:
        return RiskLevel.DESTRUCTIVE

    if any(name.startswith(pfx) for pfx in _WRITE_PREFIXES):
        return RiskLevel.WRITE
    if name in _WRITE_EXACT:
        return RiskLevel.WRITE

    if any(name.startswith(pfx) for pfx in _READ_PREFIXES):
        return RiskLevel.READ
    if name in _READ_EXACT:
        return RiskLevel.READ

    return RiskLevel.WRITE


def _classify_generic_request(params: dict[str, Any]) -> RiskLevel:
    method = str(params.get("method", "")).upper()
    path = str(params.get("path", ""))

    if method == "DELETE":
        return RiskLevel.DESTRUCTIVE

    if method in {"PUT", "POST", "PATCH"} and _WILDCARD_PATH_RE.search(path):
        return RiskLevel.DESTRUCTIVE

    if method in {"PUT", "POST", "PATCH"}:
        return RiskLevel.WRITE

    if method in {"GET", "HEAD"}:
        return RiskLevel.READ

    # Unknown or missing method: audit as a write rather than silently
    # treating as read. Conservative default mirrors classify_tool_call's
    # unknown-name fallback.
    return RiskLevel.WRITE


def canonical_call_hash(tool_name: str, params: dict[str, Any] | None) -> str:
    """SHA-256 of a canonical (name, params) JSON encoding.

    The hash is stable under key reordering and pins an approval to
    *this exact call with these exact arguments*, so a re-prompted model
    cannot substitute a different index name and reuse the approval.
    """
    payload = json.dumps(
        {"name": tool_name, "params": params or {}},
        sort_keys=True,
        default=str,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()
