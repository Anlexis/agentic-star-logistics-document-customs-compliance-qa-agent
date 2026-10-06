# LOG-C2-039 — Unit Tests: OutputFormatNode (inner domain node 5, terminal)
#
# Invocation canon: node(state) via BaseNode.__call__ with an ANONYMOUS caller.
# formatted_answer is a DOMAIN field (the framework's input scan does not read
# it); the standing
# customs-guidance disclaimer is part of THIS node's output contract and is
# unconditional (every code path appends it — non-suppressible by input
# shaping, incl. the no-coverage refusal path).
#
# Mirrors docs/03_test_spec.md section 2.6 (FMT-01..FMT-06).
# Deterministic — no LLM, no network. framework.* / src.* imports only.

from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel

from src.nodes.output_format_node import OutputFormatNode
from src.schemas.state import to_json

_DISCLAIMER_FRAGMENT = "does not constitute a binding HS classification"


def _make_state(grounded_answer, citations, domain="hs_code", **extra) -> dict:
    state = {
        "grounded_answer": grounded_answer,
        "citations": citations,
        "customs_domain": domain,
        "caller_trust_level": TrustLevel.ANONYMOUS.value,
        "node_history": [],
        "error_log": [],
        "session_id": "unit-session",
        "execution_time": {},
    }
    state.update(extra)
    return state


class TestFormattedAnswer:
    def test_fmt_01_composes_header_domain_body_sources_disclaimer(self):
        citations = to_json(
            [
                {
                    "ref": 1,
                    "id": "kb-001",
                    "title": "HS code classification",
                    "source": "General Rules for the Interpretation of the HS",
                }
            ]
        )
        result = OutputFormatNode()(
            _make_state(
                "[ ] [1] the grounded answer body.", citations, currency_note="verify against the current schedule."
            )
        )
        answer = result["formatted_answer"]
        assert answer.startswith("# Customs Compliance Checklist")
        assert "**Customs domain:** HS Code Classification" in answer
        assert "[ ] [1] the grounded answer body." in answer
        assert "## Sources" in answer
        assert "- [1] HS code classification (General Rules for the Interpretation of the HS)" in answer
        assert "## Tariff & Currency Note" in answer
        assert "verify against the current schedule." in answer
        assert _DISCLAIMER_FRAGMENT in answer
        assert result["status"] == AgentStatus.SUCCESS.value
        # Regression guard: State carries the plain string, never the enum.
        # AgentStatus subclasses str, so a bare isinstance(..., str) would pass
        # for the enum member as well and prove nothing.
        assert isinstance(result["status"], str)
        assert not isinstance(result["status"], AgentStatus)

    def test_domain_label_maps_to_the_human_readable_name(self):
        for domain, label in (
            ("hs_code", "HS Code Classification"),
            ("naccs", "NACCS Declaration"),
            ("incoterms", "Incoterms 2020"),
            ("aeo", "AEO Certification"),
            ("customs_law", "関税法 (Customs Act)"),
            ("general", "General / Multi-domain"),
        ):
            answer = OutputFormatNode()(_make_state("body.", to_json([]), domain=domain))["formatted_answer"]
            assert f"**Customs domain:** {label}" in answer

    def test_fmt_02_source_suffix_omitted_when_blank(self):
        citations = to_json([{"ref": 1, "id": "kb-001", "title": "HS code classification", "source": ""}])
        answer = OutputFormatNode()(_make_state("body.", citations))["formatted_answer"]
        assert "- [1] HS code classification\n" in answer + "\n"
        assert "()" not in answer

    def test_fmt_03_disclaimer_present_on_every_answer(self):
        # The disclaimer must ride WITH the substance, never separately, and
        # is non-suppressible — present on both the covered and no-coverage
        # paths (input shaping cannot omit it).
        for grounded in ("a body.", ""):
            answer = OutputFormatNode()(_make_state(grounded, to_json([])))["formatted_answer"]
            assert _DISCLAIMER_FRAGMENT in answer

    def test_fmt_03_currency_note_section_absent_when_no_coverage(self):
        # currency_note is "" on the no-coverage path (GenerateAnswerNode
        # never annotates a refusal) - OutputFormatNode must not fabricate
        # a Tariff & Currency Note section for it.
        answer = OutputFormatNode()(_make_state("no coverage body.", to_json([]), currency_note=""))["formatted_answer"]
        assert "## Tariff & Currency Note" not in answer


class TestDegradedInputs:
    def test_fmt_04_no_citations_renders_explicit_none_line(self):
        answer = OutputFormatNode()(_make_state("no coverage body.", to_json([])))["formatted_answer"]
        assert "- none (no knowledge-base passage cleared the relevance threshold)" in answer

    def test_fmt_05_missing_grounded_answer_uses_fallback_text(self):
        state = _make_state("", to_json([]))
        del state["grounded_answer"]
        result = OutputFormatNode()(state)
        assert "No answer is available for this request." in result["formatted_answer"]
        assert result["status"] == AgentStatus.SUCCESS.value

    def test_fmt_06_missing_domain_defaults_to_general(self):
        state = _make_state("body.", to_json([]))
        del state["customs_domain"]
        answer = OutputFormatNode()(state)["formatted_answer"]
        assert "**Customs domain:** General / Multi-domain" in answer
