# LOG-C2-039 — Unit Tests: InputValidateNode (inner domain node 1)
#
# Invocation canon: node(state) via BaseNode.__call__ with an ANONYMOUS caller
# (inner Cat-2 domain node). Payloads are lowercase and free of the shapes the
# framework's input scan masks, so the payload reaches execute() unchanged.
#
# Mirrors docs/03_test_spec.md section 2.2 (VAL-01..VAL-10).
# Deterministic — no LLM, no network. framework.* / src.* imports only.

import json

import pytest

from framework.schemas.trust_level import TrustLevel

from src.nodes.input_validate_node import InputValidateNode
from src.schemas.state import from_json


def _make_state(payload, **extra) -> dict:
    state = {
        "validated_input": payload,
        "caller_trust_level": TrustLevel.ANONYMOUS.value,
        "node_history": [],
        "error_log": [],
        "session_id": "unit-session",
        "execution_time": {},
    }
    state.update(extra)
    return state


class TestPlainTextParsing:
    def test_val_01_plain_text_becomes_query(self):
        result = InputValidateNode()(_make_state("hs code classification for consumer electronics"))
        assert result["search_query"] == "hs code classification for consumer electronics"
        filters = from_json(result["query_filters"])
        assert filters == {"domain": "hs_code", "top_k": None}

    def test_val_02_whitespace_is_collapsed(self):
        result = InputValidateNode()(_make_state("  hs code   classification\n for electronics "))
        assert result["search_query"] == "hs code classification for electronics"

    def test_query_filters_is_json_string(self):
        # Structured State fields travel as JSON strings, never dicts.
        result = InputValidateNode()(_make_state("hs code classification"))
        assert isinstance(result["query_filters"], str)
        assert isinstance(from_json(result["query_filters"]), dict)


class TestJsonEnvelopeParsing:
    def test_val_03_envelope_query_domain_top_k(self):
        payload = json.dumps({"query": "eligibility and application requirements", "domain": "aeo", "top_k": 2})
        result = InputValidateNode()(_make_state(payload))
        assert result["search_query"] == "eligibility and application requirements"
        filters = from_json(result["query_filters"])
        assert filters == {"domain": "aeo", "top_k": 2}

    def test_question_alias_accepted(self):
        payload = json.dumps({"question": "what naccs data elements are required for an import declaration?"})
        result = InputValidateNode()(_make_state(payload))
        assert result["search_query"] == "what naccs data elements are required for an import declaration?"

    def test_domain_override_is_normalised(self):
        payload = json.dumps({"query": "eligibility question", "domain": "  AEO "})
        result = InputValidateNode()(_make_state(payload))
        assert from_json(result["query_filters"])["domain"] == "aeo"

    def test_val_04_malformed_json_falls_back_to_plain_text(self):
        payload = "{ this is not valid json but starts like it"
        result = InputValidateNode()(_make_state(payload))
        assert result["search_query"] == payload
        notes = from_json(result.get("intake_notes"), [])
        assert any("did not parse" in n for n in notes)


class TestDomainOverrideGuard:
    """VAL-05: an invalid domain override is rejected (not silently trusted)
    and the node falls back to keyword auto-classification."""

    def test_val_05_unknown_domain_override_is_rejected_and_auto_classified(self):
        payload = json.dumps({"query": "unrelated shipment paperwork question", "domain": "not_a_real_domain"})
        result = InputValidateNode()(_make_state(payload))
        filters = from_json(result["query_filters"])
        assert filters["domain"] == "general"
        notes = from_json(result.get("intake_notes"), [])
        assert any("unknown domain override" in n for n in notes)
        # The rejected value is never echoed back into the notes.
        assert not any("not_a_real_domain" in n for n in notes)

    def test_val_05_valid_domain_slugs_are_accepted(self):
        for slug in ("hs_code", "naccs", "incoterms", "aeo", "customs_law"):
            payload = json.dumps({"query": "a customs question", "domain": slug})
            result = InputValidateNode()(_make_state(payload))
            assert from_json(result["query_filters"])["domain"] == slug


class TestTopKGuard:
    """VAL-06..08: an envelope-supplied top_k is untrusted.

    Anything unusable is dropped — the configured default then applies — rather
    than clamped to a value the caller never asked for. The option cannot reach
    retrieval out of bounds by either route: on the caller-context channel the
    boundary node rejects the whole request; here the option is simply ignored,
    because a bad search option must not fail an otherwise valid customs
    question.
    """

    @pytest.mark.parametrize("bad", [99, -5, 0, 20.5, "many", "", True, False, [], {}, None])
    def test_val_06_unusable_top_k_is_dropped(self, bad):
        payload = json.dumps({"query": "hs code question", "top_k": bad})
        result = InputValidateNode()(_make_state(payload))
        assert from_json(result["query_filters"])["top_k"] is None

    @pytest.mark.parametrize(
        "raw", ['{"query": "hs code question", "top_k": %s}' % form for form in ("NaN", "Infinity", "-Infinity")]
    )
    def test_val_07_non_finite_top_k_is_dropped_not_raised(self, raw):
        # json.loads accepts bare NaN/Infinity, float() parses them, and int()
        # on an infinite float raises OverflowError — which a plain
        # (TypeError, ValueError) guard does not catch.
        result = InputValidateNode()(_make_state(raw))
        assert from_json(result["query_filters"])["top_k"] is None
        notes = from_json(result.get("intake_notes"), [])
        assert any("unusable top_k" in n for n in notes)

    @pytest.mark.parametrize("good,expected", [(1, 1), (3, 3), (20, 20), ("4", 4), (5.0, 5)])
    def test_usable_top_k_is_kept(self, good, expected):
        payload = json.dumps({"query": "hs code question", "top_k": good})
        result = InputValidateNode()(_make_state(payload))
        assert from_json(result["query_filters"])["top_k"] == expected


class TestSizeAndEmptyGuards:
    def test_val_09_oversize_query_is_truncated(self):
        payload = "customs " * 300  # ~2700 chars after collapse
        result = InputValidateNode()(_make_state(payload))
        assert len(result["search_query"]) == 2000
        notes = from_json(result.get("intake_notes"), [])
        assert any("truncated" in n for n in notes)

    def test_val_10_empty_request_yields_note_not_error(self):
        result = InputValidateNode()(_make_state(""))
        assert result["search_query"] == ""
        notes = from_json(result.get("intake_notes"), [])
        assert any("empty request" in n for n in notes)


class TestDomainClassification:
    """Deterministic keyword-count classifier over the five customs domains."""

    def test_naccs_keywords_classify_to_naccs(self):
        result = InputValidateNode()(_make_state("naccs import declaration required data elements"))
        assert from_json(result["query_filters"])["domain"] == "naccs"

    def test_incoterms_keywords_classify_to_incoterms(self):
        result = InputValidateNode()(_make_state("incoterms fob and cif risk transfer for ocean freight"))
        assert from_json(result["query_filters"])["domain"] == "incoterms"

    def test_customs_law_kanji_keyword_classifies_to_customs_law(self):
        result = InputValidateNode()(_make_state("関税法 post-entry correction and voluntary disclosure"))
        assert from_json(result["query_filters"])["domain"] == "customs_law"

    def test_no_keyword_match_falls_back_to_general(self):
        result = InputValidateNode()(_make_state("what is the weather like today"))
        assert from_json(result["query_filters"])["domain"] == "general"
        notes = from_json(result.get("intake_notes"), [])
        assert any("no domain keywords matched" in n for n in notes)
