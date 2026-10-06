# LOG-C2-039 — Unit Tests: GenerateAnswerNode (inner domain node 4)
#
# Invocation canon: node(state) via BaseNode.__call__ with an ANONYMOUS caller.
# The grounded answer / citations are DOMAIN fields (the framework's input scan
# does not read them — the
# framework only scans user_input/validated_input/llm_response), so Title-Case
# KB titles inside them are safe to assert on.
#
# Mirrors docs/03_test_spec.md section 2.5 (GEN-01..GEN-06).
# Deterministic — rule-assembled from ranked_documents only (grounded by
# construction; cite-or-refuse; no LLM, no network). framework.* / src.*
# imports only.

from framework.schemas.trust_level import TrustLevel

from src.nodes.generate_answer_node import GenerateAnswerNode
from src.schemas.state import from_json, to_json


def _ranked(*entries):
    return to_json(list(entries))


def _doc(doc_id, title, excerpt, source="seeded customs kb"):
    return {
        "id": doc_id,
        "title": title,
        "category": "hs_code",
        "source": source,
        "score": 0.9,
        "excerpt": excerpt,
    }


def _make_state(ranked_documents, query="hs code classification approach", **extra) -> dict:
    state = {
        "ranked_documents": ranked_documents,
        "search_query": query,
        "caller_trust_level": TrustLevel.ANONYMOUS.value,
        "node_history": [],
        "error_log": [],
        "session_id": "unit-session",
        "execution_time": {},
    }
    state.update(extra)
    return state


class TestGroundedAnswer:
    def test_gen_01_answer_carries_checkbox_style_numbered_citation_markers(self):
        ranked = _ranked(
            _doc("kb-001", "HS code classification — general interpretive approach", "essential character controls."),
            _doc("kb-011", "2026 tariff-schedule revision", "statistical-suffix lines were renumbered."),
        )
        result = GenerateAnswerNode()(_make_state(ranked))
        answer = result["grounded_answer"]
        assert "[ ] [1] HS code classification — general interpretive approach:" in answer
        assert "[ ] [2] 2026 tariff-schedule revision:" in answer

    def test_gen_02_lead_sentence_quotes_the_query(self):
        ranked = _ranked(_doc("kb-001", "HS code classification", "excerpt."))
        result = GenerateAnswerNode()(_make_state(ranked, query="hs code classification approach"))
        assert 'Customs compliance checklist for: "hs code classification approach"' in result["grounded_answer"]

    def test_gen_03_citations_mirror_ranked_order(self):
        ranked = _ranked(
            _doc("kb-001", "HS code classification", "a.", source="General Rules for the Interpretation of the HS"),
            _doc("kb-011", "2026 tariff-schedule revision", "b."),
        )
        citations = from_json(GenerateAnswerNode()(_make_state(ranked))["citations"])
        assert [c["ref"] for c in citations] == [1, 2]
        assert [c["id"] for c in citations] == ["kb-001", "kb-011"]
        assert citations[0]["source"] == "General Rules for the Interpretation of the HS"

    def test_citations_is_json_string(self):
        # List-shaped State fields travel as JSON strings.
        ranked = _ranked(_doc("kb-001", "HS code classification", "a."))
        result = GenerateAnswerNode()(_make_state(ranked))
        assert isinstance(result["citations"], str)

    def test_gen_04_answer_is_grounded_in_ranked_passages_only(self):
        ranked = _ranked(_doc("kb-001", "HS code classification", "essential character controls classification."))
        answer = GenerateAnswerNode()(_make_state(ranked))["grounded_answer"]
        # Every content line traces to the single ranked passage.
        assert "essential character controls classification." in answer
        assert "[ ] [2]" not in answer

    def test_gen_05_currency_note_present_when_coverage_exists(self):
        ranked = _ranked(_doc("kb-001", "HS code classification", "a."))
        result = GenerateAnswerNode()(_make_state(ranked))
        assert result["currency_note"]
        assert "2026" in result["currency_note"]


class TestNoCoverage:
    def test_gen_06_empty_ranked_set_yields_no_coverage_answer(self):
        result = GenerateAnswerNode()(_make_state(_ranked()))
        assert "does not contain sufficient coverage" in result["grounded_answer"]
        assert from_json(result["citations"]) == []
        assert result["currency_note"] == ""

    def test_missing_ranked_field_is_treated_as_no_coverage(self):
        state = _make_state(None)
        del state["ranked_documents"]
        result = GenerateAnswerNode()(state)
        assert "does not contain sufficient coverage" in result["grounded_answer"]

    def test_no_coverage_message_points_to_a_customs_specialist(self):
        result = GenerateAnswerNode()(_make_state(_ranked()))
        assert "通関士" in result["grounded_answer"] or "通関業者" in result["grounded_answer"]
