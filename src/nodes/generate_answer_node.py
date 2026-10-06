"""AgentCore Platform v1.0"""

# LOG-C2-039 - GenerateAnswerNode
# Domain node 4: assemble a checklist-format grounded answer from the ranked
# knowledge-base passages, each item carrying a numbered citation marker, plus
# the tariff-schedule annotation. Checklist synthesis and the schedule note are
# one step rather than two: neither needs the other's output, and splitting
# them would only add a state round-trip.
#
# v1 is DETERMINISTIC (no live LLM call): each checklist point is rule-assembled
# from one ranked passage only - nothing outside ranked_documents reaches the
# answer, so the output is grounded by construction (cite-or-refuse: with zero
# ranked passages the node REFUSES with a no-coverage message rather than
# fabricating a checklist item). The model-synthesis upgrade seam is described
# in docs/02_design.md and config/prompts/answer_synthesis_prompt.md.
#
# Wired by the inner graph (DomainWorkflowGraph).
# Returns only changed state keys (partial dict).

from typing import Any, ClassVar, Dict, List

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event
from src.schemas.state import from_json, to_json

# Checklist body used when no KB passage cleared the relevance threshold -
# cite-or-refuse: never answer a customs question without a grounding citation.
_NO_COVERAGE_ANSWER = (
    "The seeded customs knowledge base does not contain sufficient coverage to "
    "answer this question. Rephrase the query with more specific customs-domain "
    "terms (HS code, NACCS declaration, Incoterms 2020, AEO, or 関税法), or "
    "escalate to a licensed 通関士 (customs specialist) / 通関業者 (customs "
    "broker) for a manual review."
)

# Cited excerpt length per checklist point.
_POINT_EXCERPT_CHARS = 240

# Tariff/currency-schedule annotation attached to every coverage-bearing answer.
_CURRENCY_NOTE = (
    "Reflects the 2026 tariff-schedule changes as captured in this knowledge "
    "base's current seed data. HS classifications, duty rates, and NACCS "
    "procedural details are revised periodically - verify against the current "
    "official NACCS / customs authority publication before filing or relying on "
    "this answer for a binding declaration."
)


def _first_sentences(text: str, limit: int) -> str:
    """Trim an excerpt at a sentence boundary where possible, else hard-cap."""
    text = text.strip()
    if len(text) <= limit:
        return text
    cut = text[:limit]
    period = cut.rfind(". ")
    if period > limit // 2:
        return cut[: period + 1]
    return cut.rstrip() + "..."


class GenerateAnswerNode(FunctionNode):
    """Rule-based checklist-format answer assembly with numbered citations.

    Input state keys:
        ranked_documents: JSON list of surviving passages (from RerankFilterNode)
        search_query:     normalised question (for the checklist heading)

    Output state keys (partial dict):
        grounded_answer: checklist body with [n] citation markers (or the
                         cite-or-refuse no-coverage message)
        citations:       JSON list [{ref, id, title, source}]
        checklist_items: JSON list[str] of the individual checklist point
                         strings (structured surface for get_output())
        currency_note:   tariff/currency annotation (empty on no-coverage -
                         nothing to annotate)
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: AgentState) -> Dict[str, Any]:
        ranked: List[Dict[str, Any]] = from_json(state.get("ranked_documents"), []) or []
        query = state.get("search_query") or ""

        citations: List[Dict[str, Any]] = []
        checklist_items: List[str] = []
        currency_note = ""

        if not ranked:
            # Cite-or-refuse: no grounding passage -> refuse, never fabricate.
            grounded_answer = _NO_COVERAGE_ANSWER
        else:
            lines: List[str] = []
            if query:
                lines.append(f'Customs compliance checklist for: "{query}"')
            else:
                lines.append("Customs compliance checklist - most relevant guidance:")
            lines.append("")
            for ref, doc in enumerate(ranked, start=1):
                if not isinstance(doc, dict):
                    continue
                title = str(doc.get("title", "")).strip()
                excerpt = _first_sentences(str(doc.get("excerpt", "")), _POINT_EXCERPT_CHARS)
                point = f"{title}: {excerpt}"
                lines.append(f"[ ] [{ref}] {point}")
                checklist_items.append(point)
                citations.append(
                    {
                        "ref": ref,
                        "id": str(doc.get("id", "")),
                        "title": title,
                        "source": str(doc.get("source", "")),
                    }
                )
            grounded_answer = "\n".join(lines)
            currency_note = _CURRENCY_NOTE

        # Domain audit: checklist answer assembled.
        emit_trace_event(
            "generate_answer_complete",
            {
                "citation_count": len(citations),
                "answer_chars": len(grounded_answer),
                "no_coverage": not ranked,
            },
            state,
        )

        return {
            "grounded_answer": grounded_answer,
            "citations": to_json(citations),
            "checklist_items": to_json(checklist_items),
            "currency_note": currency_note,
        }
