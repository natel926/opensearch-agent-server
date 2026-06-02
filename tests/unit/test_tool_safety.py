"""Unit tests for tool risk classification and canonical hashing."""

from __future__ import annotations

import pytest

from utils.tool_safety import RiskLevel, canonical_call_hash, classify_tool_call

pytestmark = pytest.mark.unit


class TestClassifyReads:
    @pytest.mark.parametrize(
        "name",
        [
            "get_index",
            "cluster_health",
            "search",
            "cat_indices",
            "msearch",
            "list_indices",
            "describe_index",
            "count",
            "scroll",
            "exists",
            "cluster_state",
            "cat_nodes",
        ],
    )
    def test_read_tools(self, name: str) -> None:
        assert classify_tool_call(name, {}) is RiskLevel.READ


class TestClassifyWrites:
    @pytest.mark.parametrize(
        "name",
        [
            "create_index",
            "bulk",
            "update_settings",
            "index_documents",
            "put_mapping",
            "set_setting",
            "reindex",
            "snapshot",
            "restore_snapshot",
        ],
    )
    def test_write_tools(self, name: str) -> None:
        assert classify_tool_call(name, {}) is RiskLevel.WRITE

    def test_unknown_tool_defaults_to_write(self) -> None:
        # Unknown tools execute (with audit) but do not gate on approval.
        assert classify_tool_call("frobnicate_widget", {}) is RiskLevel.WRITE


class TestClassifyDestructive:
    @pytest.mark.parametrize(
        "name",
        [
            "delete_index",
            "delete_by_query",
            "force_merge",
            "close_index",
            "drop_index",
            "purge_index",
            "truncate",
            "delete_pipeline",
            "delete_snapshot",
            "delete_repository",
            "delete_search_pipeline",
            "delete_ingest_pipeline",
        ],
    )
    def test_destructive_by_name(self, name: str) -> None:
        assert classify_tool_call(name, {}) is RiskLevel.DESTRUCTIVE

    def test_generic_request_delete_method(self) -> None:
        assert (
            classify_tool_call(
                "opensearch_request", {"method": "DELETE", "path": "/foo"}
            )
            is RiskLevel.DESTRUCTIVE
        )
        assert (
            classify_tool_call("http_request", {"method": "delete", "path": "/foo"})
            is RiskLevel.DESTRUCTIVE
        )

    @pytest.mark.parametrize("method", ["PUT", "POST", "PATCH"])
    @pytest.mark.parametrize("path", ["/_all/_close", "/*/_settings", "/logs-*/_doc/1"])
    def test_wildcard_or_all_with_write_method(self, method: str, path: str) -> None:
        assert (
            classify_tool_call("opensearch_request", {"method": method, "path": path})
            is RiskLevel.DESTRUCTIVE
        )

    def test_generic_request_get_on_wildcard_is_read(self) -> None:
        assert (
            classify_tool_call(
                "opensearch_request", {"method": "GET", "path": "/*/_search"}
            )
            is RiskLevel.READ
        )

    def test_generic_request_put_safe_path_is_write(self) -> None:
        assert (
            classify_tool_call(
                "opensearch_request", {"method": "PUT", "path": "/my-index/_doc/1"}
            )
            is RiskLevel.WRITE
        )

    def test_generic_request_with_none_params_defaults_to_write(self) -> None:
        # An opensearch_request with no method should fail closed: classified
        # as WRITE (audited) rather than silently treated as READ.
        assert classify_tool_call("opensearch_request", None) is RiskLevel.WRITE


class TestCanonicalCallHash:
    def test_stable_under_key_reordering(self) -> None:
        h1 = canonical_call_hash("delete_index", {"index": "logs", "force": True})
        h2 = canonical_call_hash("delete_index", {"force": True, "index": "logs"})
        assert h1 == h2

    def test_distinct_for_different_params(self) -> None:
        h1 = canonical_call_hash("delete_index", {"index": "logs-2026"})
        h2 = canonical_call_hash("delete_index", {"index": "logs-2025"})
        assert h1 != h2

    def test_distinct_for_different_names(self) -> None:
        h1 = canonical_call_hash("delete_index", {"index": "logs"})
        h2 = canonical_call_hash("close_index", {"index": "logs"})
        assert h1 != h2

    def test_handles_none_params(self) -> None:
        h = canonical_call_hash("delete_index", None)
        assert isinstance(h, str)
        assert len(h) == 64  # SHA-256 hex
