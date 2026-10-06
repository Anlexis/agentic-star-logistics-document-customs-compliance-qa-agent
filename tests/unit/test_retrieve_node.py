# LOG-C2-039 — Unit Tests: RetrieveNode (inner domain node 2)
#
# Invocation canon: node(state) via BaseNode.__call__ with an ANONYMOUS caller.
# NOTE (RULES C2, retired 2026-07-27): execute(self, state) is the ONLY
# signature on this repo's nodes — a 2nd positional/keyword `config` argument
# TypeErrors (verified against the real installed SDK). Config knobs are
# exercised exclusively by seeding state["retrieval_config"], still invoking
# through node(state) / __call__ — there is no direct-execute config carve-out
# here (unlike older-generation sibling templates).
#
# Mirrors docs/03_test_spec.md section 2.3 (RET-01..RET-08).
# Deterministic — keyword scoring over the seeded config/kb/customs_kb.json;
# no LLM, no network. framework.* / src.* imports only.

import pytest

from framework.schemas.trust_level import TrustLevel

from src.nodes.retrieve_node import RetrieveNode
from src.schemas.state import from_json, to_json

_HS_CODE_QUERY = "hs code classification general interpretive approach for composite goods"


def _make_state(query=_HS_CODE_QUERY, **extra) -> dict:
    state = {
        "search_query": query,
        "caller_trust_level": TrustLevel.ANONYMOUS.value,
        "node_history": [],
        "error_log": [],
        "session_id": "unit-session",
        "execution_time": {},
    }
    state.update(extra)
    return state


class TestRetrieveHappyPath:
    def test_ret_01_top_hit_is_the_general_hs_code_entry(self):
        result = RetrieveNode()(_make_state())
        docs = from_json(result["retrieved_documents"])
        assert docs, "expected candidates for the hs_code query"
        assert docs[0]["id"] == "kb-001"

    def test_ret_02_scores_sorted_descending(self):
        docs = from_json(RetrieveNode()(_make_state())["retrieved_documents"])
        scores = [d["score"] for d in docs]
        assert scores == sorted(scores, reverse=True)
        assert all(s > 0.0 for s in scores)

    def test_ret_03_entry_shape_and_excerpt_cap(self):
        docs = from_json(RetrieveNode()(_make_state())["retrieved_documents"])
        for doc in docs:
            assert set(doc.keys()) == {"id", "title", "category", "source", "score", "excerpt"}
            assert len(doc["excerpt"]) <= 400

    def test_retrieved_documents_is_json_string(self):
        # List-shaped State fields travel as JSON strings.
        result = RetrieveNode()(_make_state())
        assert isinstance(result["retrieved_documents"], str)


class TestRetrieveDomainFilter:
    def test_ret_04_domain_filter_restricts_pool(self):
        state = _make_state(
            query="eligibility and application requirements",
            query_filters=to_json({"domain": "aeo", "top_k": None}),
            customs_domain="aeo",
        )
        docs = from_json(RetrieveNode()(state)["retrieved_documents"])
        assert docs, "aeo domain has seeded entries"
        assert {d["category"] for d in docs} == {"aeo"}

    def test_ret_04_no_cross_domain_leakage(self):
        # Domain filtering must never surface an off-domain KB entry.
        state = _make_state(
            query="incoterms 2020 risk and cost allocation between buyer and seller",
            customs_domain="incoterms",
        )
        docs = from_json(RetrieveNode()(state)["retrieved_documents"])
        assert docs
        assert {d["category"] for d in docs} == {"incoterms"}

    def test_ret_05_empty_query_yields_no_candidates(self):
        docs = from_json(RetrieveNode()(_make_state(query=""))["retrieved_documents"])
        assert docs == []


class TestRetrieveConfigPrecedence:
    """Config plumbing: state-seeded retrieval_config overlays module
    defaults — the ONLY route on this repo's C2-retired node contract."""

    def test_ret_06_execute_with_a_second_argument_type_errors(self):
        # Confirms RULES C2: the retired execute(state, config=...) call shape
        # is gone on this node — a 2nd argument is a hard TypeError, not a
        # silently-ignored no-op.
        with pytest.raises(TypeError):
            RetrieveNode().execute(_make_state(), {"configurable": {}})

    def test_ret_07_state_retrieval_config_kb_path_override(self):
        state = _make_state(retrieval_config=to_json({"kb_path": "config/kb/does_not_exist.json"}))
        result = RetrieveNode()(state)
        assert from_json(result["retrieved_documents"]) == []
        notes = from_json(result.get("intake_notes"), [])
        assert any("not readable" in n for n in notes)

    def test_ret_08_no_retrieval_config_falls_back_to_module_defaults(self):
        # No retrieval_config key in state at all - the real seeded KB is used.
        result = RetrieveNode()(_make_state())
        docs = from_json(result["retrieved_documents"])
        assert docs and docs[0]["id"] == "kb-001"


class TestRetrieveNotesAccumulation:
    def test_ret_09_notes_append_never_clobber(self):
        state = _make_state(
            intake_notes=to_json(["earlier note from input validation"]),
            retrieval_config=to_json({"kb_path": "config/kb/bogus.json"}),
        )
        result = RetrieveNode()(state)
        notes = from_json(result["intake_notes"])
        assert notes[0] == "earlier note from input validation"
        assert len(notes) == 2
