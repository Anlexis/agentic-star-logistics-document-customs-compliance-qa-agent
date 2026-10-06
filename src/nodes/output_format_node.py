"""AgentCore Platform v1.0"""

# LOG-C2-039 - OutputFormatNode
# Domain node 5 (terminal): compose the final formatted answer - the checklist
# body, the customs-domain routing line, the Sources list, the tariff/currency
# annotation, and the standing, non-suppressible customs-guidance disclaimer.
# The disclaimer is part of THIS node's domain output contract, not of the
# outer post_process slot (post_process only gates for disallowed content, it
# does not compose) - and it is unconditional: every path through this node
# (including the no-coverage refusal) appends it, so it can never be omitted by
# input shaping.
#
# Wired by the inner graph (DomainWorkflowGraph). get_output() of the inner
# graph surfaces formatted_answer + the structured fields to the outer
# merge_output(). Returns only changed state keys (partial dict).

from typing import Any, ClassVar, Dict, List

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event
from src.schemas.state import from_json

# Human-readable domain labels for the routing line.
_DOMAIN_LABELS: Dict[str, str] = {
    "hs_code": "HS Code Classification",
    "naccs": "NACCS Declaration",
    "incoterms": "Incoterms 2020",
    "aeo": "AEO Certification",
    "customs_law": "関税法 (Customs Act)",
    "general": "General / Multi-domain",
}

# Standing customs-guidance disclaimer - appended to EVERY answer this template
# emits (non-suppressible: no branch in this node skips it).
_DISCLAIMER = (
    "This answer is generated from a seeded customs-compliance knowledge base "
    "for reference information only and does not constitute a binding HS "
    "classification, customs declaration, or legal ruling. The licensed 通関士 "
    "(customs specialist) / 通関業者 (customs broker) of record remains "
    "responsible for the final classification, declaration, and any filing "
    "decision. Verify against the current NACCS / customs authority publication "
    "before acting on it."
)


class OutputFormatNode(FunctionNode):
    """Compose the final answer: checklist + domain line + sources + currency
    note + the non-suppressible customs-guidance disclaimer.

    Input state keys:
        grounded_answer: checklist body with [n] citation markers
        citations:       JSON list [{ref, id, title, source}]
        customs_domain:  classified domain slug
        currency_note:   tariff/currency annotation (may be empty on no-coverage)

    Output state keys (partial dict):
        formatted_answer: final rendered answer string
        status:           AgentStatus.SUCCESS.value (plain string - never write
                          the bare enum to State, which does not survive
                          checkpointing)
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: AgentState) -> Dict[str, Any]:
        grounded_answer = state.get("grounded_answer") or ("No answer is available for this request.")
        citations: List[Dict[str, Any]] = from_json(state.get("citations"), []) or []
        domain = state.get("customs_domain") or "general"
        currency_note = state.get("currency_note") or ""

        lines: List[str] = []
        lines.append("# Customs Compliance Checklist")
        lines.append("")
        lines.append(f"**Customs domain:** {_DOMAIN_LABELS.get(domain, domain)}")
        lines.append("")
        lines.append(grounded_answer)
        lines.append("")
        lines.append("## Sources")
        if citations:
            for citation in citations:
                if not isinstance(citation, dict):
                    continue
                ref = citation.get("ref", "?")
                title = str(citation.get("title", "")).strip()
                source = str(citation.get("source", "")).strip()
                suffix = f" ({source})" if source else ""
                lines.append(f"- [{ref}] {title}{suffix}")
        else:
            lines.append("- none (no knowledge-base passage cleared the relevance threshold)")
        if currency_note:
            lines.append("")
            lines.append("## Tariff & Currency Note")
            lines.append(currency_note)
        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append(f"*{_DISCLAIMER}*")

        formatted_answer = "\n".join(lines)

        # Domain audit: final checklist answer composed (disclaimer attached).
        emit_trace_event(
            "output_format_complete",
            {
                "answer_chars": len(formatted_answer),
                "citation_count": len(citations),
                "customs_domain": domain,
            },
            state,
        )

        return {
            "formatted_answer": formatted_answer,
            "status": AgentStatus.SUCCESS.value,
        }
