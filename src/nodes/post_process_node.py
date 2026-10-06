"""AgentCore Platform v1.0"""

# LOG-C2-039 - PostProcessNode (outer post_process slot; the output gate)
#
# Reads the final customs-compliance answer from state["result"] (populated by
# CustomsComplianceDocumentQAGraphNode.merge_output(), mapped from the inner
# graph's formatted_answer output) together with the structured fields it
# surfaces (citations, checklist_items, customs_domain, currency_note), and gates
# ALL of them behind the output content-safety scan before
# CustomsComplianceDocumentQAAgent.get_output() is allowed to expose them to the
# caller.
#
# The MODULE-LEVEL `_security_gate_output()` scan below RECURSES into nested
# dict/list/tuple structures - a credential-shaped string nested inside a
# returned payload dict (e.g. inside a citation entry) is caught, not just a
# top-level string. A scan that only reads the top-level answer string misses
# exactly the structured fields this template also publishes. On a violation, the surfaced
# formatted_output/result AND every structured field are replaced with a
# sanitised stub / emptied, and ERROR status is returned -
# CustomsComplianceDocumentQAAgent.get_output() then omits the structured keys
# entirely (SUCCESS-only surface = the fail-closed half of this gate). No
# _extra_security_gate_input/_output instance methods are defined on this node
# (the real SDK auto-wraps such hooks).

import logging
import re
from typing import Any, ClassVar, Dict, List, Optional, Tuple

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event
from src.services.failure_message import EMPTY_INPUT, INPUT_REJECTED, INVALID_VALUE, TOO_LONG
from src.schemas.state import from_json, to_json

logger = logging.getLogger(__name__)

# Disallowed content patterns. Each tuple: (name, compiled regex) - order
# matters (most specific first).
_DISALLOWED_PATTERNS: List[Tuple[str, re.Pattern[str]]] = [
    # API key patterns: sk-..., pk-..., ak-...
    ("api_key", re.compile(r"\b(?:sk|pk|ak)-[A-Za-z0-9]{16,}", re.IGNORECASE)),
    # JWT: three base64url segments separated by dots
    ("jwt", re.compile(r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}")),
    # Bearer token in Authorization-like context
    ("bearer_token", re.compile(r"Bearer\s+[A-Za-z0-9._~+/]{20,}", re.IGNORECASE)),
    # Credential assignment patterns
    (
        "credential_assignment",
        re.compile(
            r"\b(?:password|passwd|secret|api_key|token|access_key|private_key)\s*[:=]\s*\S{8,}",
            re.IGNORECASE,
        ),
    ),
]

_SANITISED_STUB = (
    "[OUTPUT BLOCKED by the output gate - disallowed content detected. Review the "
    "generated customs-compliance answer and retry without credential-like "
    "strings.]"
)


def _security_gate_output(content: Any) -> Optional[str]:
    """Run the output content gate. RECURSES into dict/list/tuple so a
    credential-shaped value nested inside a structured field (citations,
    checklist items, ...) cannot bypass a top-level-string-only scan. Every
    scalar string reachable from `content` is scanned - nothing is implicitly
    whitelisted by depth.

    Returns the name of the first matched violation, or None if clean.
    """
    if content is None:
        return None
    if isinstance(content, str):
        for name, pattern in _DISALLOWED_PATTERNS:
            if pattern.search(content):
                return str(name)
        return None
    if isinstance(content, dict):
        for value in content.values():
            violation = _security_gate_output(value)
            if violation:
                return violation
        return None
    if isinstance(content, (list, tuple)):
        for item in content:
            violation = _security_gate_output(item)
            if violation:
                return violation
        return None
    # int / float / bool / other scalars - nothing to scan.
    return None


# Reason code -> the sentence the caller reads. A code with no entry falls
# back to the generic one rather than leaking the code itself.
_DEGRADED_MESSAGES = {
    "EMPTY_INPUT": EMPTY_INPUT,
    "QUESTION_TOO_LONG": TOO_LONG,
    "INVALID_REQUEST": INVALID_VALUE,
}


class PostProcessNode(FunctionNode):
    """Output gate: recursively scan the final customs answer + every
    structured field for disallowed content before
    CustomsComplianceDocumentQAAgent.get_output() can surface them.

    Outer backbone post_process slot. Reads state["result"] (the merged
    formatted_answer from CustomsComplianceDocumentQAGraphNode.merge_output())
    plus citations / checklist_items / customs_domain / currency_note, and
    applies the content-safety gate to the WHOLE structured payload before
    the response is returned to the caller.

    Input state keys:
        result, citations, checklist_items, customs_domain, currency_note

    Output state keys (partial dict):
        formatted_output: sanitised output (unchanged answer if clean; blocked
                          stub on violation)
        result:           gated alongside formatted_output (blocked stub on
                          violation; unchanged on the clean/empty paths)
        citations, checklist_items, customs_domain, currency_note: emptied on a
                          violation, unchanged when clean
        status:           AgentStatus.SUCCESS.value or AgentStatus.ERROR.value
                          (plain strings - never write the bare enum to State,
                          which does not survive checkpointing)
        error_log:        (on error) list of error messages
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.VERIFIED_EXTERNAL

    def execute(self, state: AgentState) -> Dict[str, Any]:
        # A run declined upstream has nothing to format. Render the reason as
        # the caller-facing body and carry the marker onward.
        marker = state.get("error_code")
        if marker:
            message = _DEGRADED_MESSAGES.get(marker, INPUT_REJECTED)
            emit_trace_event("post_process_degraded", {"reason": marker}, state)
            return {
                "status": AgentStatus.SUCCESS.value,
                "error_code": marker,
                "formatted_output": message,
                "result": message,
            }
        result = state.get("result") or ""

        if not result or not str(result).strip():
            # No answer was generated - forward as-is (non-fatal).
            return {
                "formatted_output": result,
                "status": AgentStatus.SUCCESS.value,
            }

        citations = from_json(state.get("citations"), []) or []
        checklist_items = from_json(state.get("checklist_items"), []) or []
        customs_domain = state.get("customs_domain") or ""
        currency_note = state.get("currency_note") or ""

        # Scan the WHOLE structured payload the caller will ultimately see -
        # never just the top-level result string (the recursive walk covers
        # every nested scalar in citations / checklist_items too).
        payload_to_scan: Dict[str, Any] = {
            "result": result,
            "citations": citations,
            "checklist_items": checklist_items,
            "customs_domain": customs_domain,
            "currency_note": currency_note,
        }
        violation = _security_gate_output(payload_to_scan)
        if violation:
            logger.error(
                "PostProcessNode: OUTPUT BLOCKED - violation type: %s",
                violation,
            )
            return {
                "formatted_output": _SANITISED_STUB,
                "result": _SANITISED_STUB,
                "citations": to_json([]),
                "checklist_items": to_json([]),
                "customs_domain": "",
                "currency_note": "",
                "status": AgentStatus.ERROR.value,
                "error_log": [f"PostProcessNode: output blocked - " f"disallowed content detected ({violation})"],
            }

        # Clean - domain audit: record that a finalized answer was emitted.
        emit_trace_event(
            "post_process_complete",
            {"output_chars": len(str(result))},
            state,
        )

        return {
            "formatted_output": result,
            "status": AgentStatus.SUCCESS.value,
        }
