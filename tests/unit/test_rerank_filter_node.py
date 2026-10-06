# LOG-C2-039 — Unit Tests: RerankFilterNode (inner domain node 3)
#
# Invocation canon: node(state) via BaseNode.__call__ with an ANONYMOUS caller.
# NOTE (RULES C2, retired 2026-07-27): execute(self, state) is the ONLY
# signature — config knobs are exercised exclusively by seeding
# state["retrieval_config"], still invoking through node(state).
#
# Mirrors docs/03_test_spec.md section 2.4 (RRF-01..RRF-09).
# Deterministic — no LLM, no network. framework.* / src.* imports only.

import pytest

from framework.schemas.trust_level import TrustLevel

from src.nodes.rerank_filter_node import RerankFilterNode
from src.schemas.state import from_json, to_json


def _doc(doc_id, score, category="hs_code"):
    return {
        "id": doc_id,
        "title": f"entry {doc_id}",
        "category": category,
        "source": "seeded kb",
        "score": score,
        "excerpt": "excerpt text",
    }


def _make_state(candidates, **extra) -> dict:
    state = {
        "retrieved_documents": to_json(candidates),
        "caller_trust_level": TrustLevel.ANONYMOUS.value,
        "node_history": [],
        "error_log": [],
        "session_id": "unit-session",
        "execution_time": {},
    }
    state.update(extra)
    return state


class TestThresholdAndCap:
    def test_rrf_01_default_threshold_drops_weak_candidates(self):
        result = RerankFilterNode()(_make_state([_doc("kb-a", 0.9), _doc("kb-b", 0.1)]))
        kept = from_json(result["ranked_documents"])
        assert [d["id"] for d in kept] == ["kb-a"]  # 0.1 < default 0.25 floor

    def test_rrf_02_execute_with_a_second_argument_type_errors(self):
        # RULES C2: the retired execute(state, config=...) call shape is gone.
        with pytest.raises(TypeError):
            RerankFilterNode().execute(_make_state([_doc("kb-a", 0.9)]), {"configurable": {}})

    def test_rrf_03_state_seeded_score_threshold_override(self):
        state = _make_state(
            [_doc("kb-a", 0.9), _doc("kb-b", 0.3)],
            retrieval_config=to_json({"score_threshold": 0.5}),
        )
        kept = from_json(RerankFilterNode()(state)["ranked_documents"])
        assert [d["id"] for d in kept] == ["kb-a"]

    def test_rrf_04_state_seeded_top_k_override(self):
        state = _make_state(
            [_doc("kb-a", 0.9), _doc("kb-b", 0.8), _doc("kb-c", 0.7)],
            retrieval_config=to_json({"top_k": 1}),
        )
        kept = from_json(RerankFilterNode()(state)["ranked_documents"])
        assert [d["id"] for d in kept] == ["kb-a"]

    def test_ranked_documents_is_json_string(self):
        # List-shaped State fields travel as JSON strings.
        result = RerankFilterNode()(_make_state([_doc("kb-a", 0.9)]))
        assert isinstance(result["ranked_documents"], str)


class TestDomainBoost:
    def test_rrf_05_matching_domain_is_boosted_and_reranked(self):
        state = _make_state(
            [_doc("kb-a", 0.30, category="naccs"), _doc("kb-b", 0.25, category="hs_code")],
            customs_domain="hs_code",
        )
        kept = from_json(RerankFilterNode()(state)["ranked_documents"])
        assert [d["id"] for d in kept] == ["kb-b", "kb-a"]
        assert kept[0]["score"] == 0.35  # 0.25 + 0.1 domain boost

    def test_rrf_06_boost_is_capped_at_one(self):
        state = _make_state(
            [_doc("kb-a", 0.95, category="hs_code")],
            customs_domain="hs_code",
        )
        kept = from_json(RerankFilterNode()(state)["ranked_documents"])
        assert kept[0]["score"] == 1.0

    def test_general_domain_applies_no_boost(self):
        state = _make_state(
            [_doc("kb-a", 0.30, category="naccs"), _doc("kb-b", 0.25, category="hs_code")],
            customs_domain="general",
        )
        kept = from_json(RerankFilterNode()(state)["ranked_documents"])
        assert [d["id"] for d in kept] == ["kb-a", "kb-b"]


class TestCallerTopK:
    def test_rrf_07_stricter_caller_top_k_wins(self):
        state = _make_state(
            [_doc("kb-a", 0.9), _doc("kb-b", 0.8), _doc("kb-c", 0.7)],
            query_filters=to_json({"domain": None, "top_k": 1}),
        )
        kept = from_json(RerankFilterNode()(state)["ranked_documents"])
        assert [d["id"] for d in kept] == ["kb-a"]

    def test_rrf_07_looser_caller_top_k_does_not_widen(self):
        state = _make_state(
            [_doc("kb-a", 0.9), _doc("kb-b", 0.8), _doc("kb-c", 0.7)],
            query_filters=to_json({"domain": None, "top_k": 10}),
            retrieval_config=to_json({"top_k": 2, "score_threshold": 0.25}),
        )
        kept = from_json(RerankFilterNode()(state)["ranked_documents"])
        assert [d["id"] for d in kept] == ["kb-a", "kb-b"]


class TestRobustness:
    def test_rrf_08_garbage_candidates_are_skipped_or_dropped(self):
        candidates = [
            "not-a-dict",
            {"id": "kb-bad", "title": "b", "category": "x", "source": "s", "score": "NaN?", "excerpt": "e"},
            _doc("kb-a", 0.9),
        ]
        kept = from_json(RerankFilterNode()(_make_state(candidates))["ranked_documents"])
        # The string entry is skipped; the uncoercible score becomes 0.0 and
        # falls below the relevance floor.
        assert [d["id"] for d in kept] == ["kb-a"]

    def test_rrf_09_deterministic_tie_break_by_id(self):
        kept = from_json(RerankFilterNode()(_make_state([_doc("kb-b", 0.5), _doc("kb-a", 0.5)]))["ranked_documents"])
        assert [d["id"] for d in kept] == ["kb-a", "kb-b"]
