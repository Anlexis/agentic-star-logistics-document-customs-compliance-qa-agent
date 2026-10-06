# LOG-C2-039 — Unit Tests: retrieval quality over the seeded customs KB
#
# Golden-query suite: drives the REAL inner retrieval chain
# (InputValidateNode -> RetrieveNode -> RerankFilterNode) via node(state) /
# __call__ (ANONYMOUS inner nodes) against config/kb/customs_kb.json and pins
# the expected top hit per domain query. The scorer is deterministic (keyword
# field-weights, stable tie-break) and every expectation below was verified
# by actually running this chain against the real seeded KB, so exact top-1
# assertions are safe and catch KB / scorer / threshold regressions.
#
# Mirrors docs/03_test_spec.md section 2.9 (QUAL-01..QUAL-07).
# Deterministic — no LLM, no network. framework.* / src.* imports only.

import json
import pathlib

import pytest

from framework.schemas.trust_level import TrustLevel

from src.nodes.generate_answer_node import GenerateAnswerNode
from src.nodes.input_validate_node import InputValidateNode
from src.nodes.rerank_filter_node import RerankFilterNode
from src.nodes.retrieve_node import RetrieveNode
from src.schemas.state import from_json

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_KB_IDS = {
    entry["id"] for entry in json.loads((_ROOT / "config" / "kb" / "customs_kb.json").read_text(encoding="utf-8"))
}

_DEFAULT_SCORE_THRESHOLD = 0.25  # mirrors config/agent.yaml retrieval block


def _search(payload: str) -> list[dict]:
    """Run the real inner retrieval chain and return the surviving passages."""
    state = {
        "validated_input": payload,
        "caller_trust_level": TrustLevel.ANONYMOUS.value,
        "node_history": [],
        "error_log": [],
        "session_id": "quality-session",
        "execution_time": {},
    }
    state.update(InputValidateNode()(state))
    state.update(RetrieveNode()(state))
    state.update(RerankFilterNode()(state))
    return from_json(state["ranked_documents"], [])


# (query, expected top-1 KB entry id) — verified against the deterministic
# scorer by actually running the chain (see task notes); one query per
# customs domain plus a second hs_code query to cover the domain's two
# distinct entries.
_GOLDEN_QUERIES = [
    ("hs code classification general interpretive approach for composite goods", "kb-001"),
    ("advance classification ruling process for confirming an hs code before filing", "kb-002"),
    ("naccs import declaration required data elements", "kb-003"),
    ("naccs declaration amendment after release", "kb-004"),
    ("incoterms 2020 risk and cost allocation between buyer and seller", "kb-005"),
    ("aeo certification categories and eligibility criteria", "kb-007"),
    ("customs valuation transaction value under customs law", "kb-009"),
]


class TestGoldenQueries:
    @pytest.mark.parametrize(("query", "expected_id"), _GOLDEN_QUERIES)
    def test_qual_01_top_hit_per_golden_query(self, query, expected_id):
        kept = _search(query)
        assert kept, f"no passage cleared the relevance floor for: {query!r}"
        assert kept[0]["id"] == expected_id

    def test_qual_02_all_survivors_clear_the_relevance_floor(self):
        for query, _expected in _GOLDEN_QUERIES:
            for doc in _search(query):
                assert doc["score"] >= _DEFAULT_SCORE_THRESHOLD

    def test_qual_03_survivor_ids_exist_in_the_seeded_kb(self):
        for query, _expected in _GOLDEN_QUERIES:
            for doc in _search(query):
                assert doc["id"] in _KB_IDS


class TestDomainPrecision:
    def test_qual_04_domain_classification_excludes_other_domains(self):
        # Precision: the classified-domain filter must never let a
        # cross-domain KB entry leak into the surviving set.
        by_domain = {
            "hs_code": {"kb-001", "kb-002", "kb-011"},
            "naccs": {"kb-003", "kb-004"},
            "incoterms": {"kb-005", "kb-006"},
            "aeo": {"kb-007", "kb-008"},
            "customs_law": {"kb-009", "kb-010"},
        }
        for query, expected_id in _GOLDEN_QUERIES:
            kept_ids = {d["id"] for d in _search(query)}
            expected_domain = next(dom for dom, ids in by_domain.items() if expected_id in ids)
            assert (
                kept_ids <= by_domain[expected_domain]
            ), f"{query!r} leaked a cross-domain entry: {kept_ids} not subset of {by_domain[expected_domain]}"

    def test_qual_05_explicit_domain_override_restricts_to_that_domain(self):
        payload = json.dumps({"query": "eligibility and application requirements", "domain": "aeo"})
        kept = _search(payload)
        assert kept, "aeo domain carries seeded entries"
        assert {d["category"] for d in kept} == {"aeo"}


class TestNoCoverage:
    def test_qual_06_out_of_domain_query_yields_no_survivors(self):
        assert _search("quantum telepathy sandwich recipes") == []

    def test_qual_07_no_coverage_produces_the_escalation_answer(self):
        state = {
            "ranked_documents": "[]",
            "search_query": "quantum telepathy sandwich recipes",
            "caller_trust_level": TrustLevel.ANONYMOUS.value,
            "node_history": [],
            "error_log": [],
            "session_id": "quality-session",
            "execution_time": {},
        }
        result = GenerateAnswerNode()(state)
        assert "does not contain sufficient coverage" in result["grounded_answer"]
        assert from_json(result["citations"]) == []
