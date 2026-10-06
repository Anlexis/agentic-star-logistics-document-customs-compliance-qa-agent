"""AgentCore Platform v1.0"""

# LOG-C2-039 — PreProcessNode (outer pre_process slot).
#
# This node owns the caller boundary. Everything a caller can send arrives here
# and nothing reaches the inner domain workflow until this node has accepted it:
#
#   user_input      the natural-language customs question
#   input_context   the per-request options and, optionally, the knowledge
#                   passages to answer FROM:
#                     customs_domain     narrow the search to one domain
#                     top_k              how many passages to cite
#                     score_threshold    the relevance floor
#                     knowledge_entries  caller-supplied passages; when absent
#                                        the seeded reference corpus is used
#
# Three guarantees are enforced HERE rather than delegated:
#
#  1. Refusal of instruction-override content, on the question channel and on
#     every caller-supplied passage. The framework's own input gate is a second
#     layer, not the first: where it is absent or configured off, this check
#     still runs, so the refusal cannot become fail-open. It is deliberately
#     narrow — anchored on whole phrases with the prompt-specific noun required
#     — because customs questions legitimately say "act as an importer of
#     record", "does an advance ruling override the classification", and "ignore
#     the previous declaration". A screen that refuses those blocks real work.
#     There is intentionally no SQL-verb screen: this template opens no database,
#     and "select", "update", "delete" and "drop" are ordinary words in customs
#     declaration language.
#
#  2. Bounds on every caller field. Numbers go through the finite+bounded parser
#     (NaN and Infinity parse as floats but compare False, so an unguarded
#     threshold silently stops filtering); lists have entry caps; strings have
#     length caps; identifiers are inert. A rejected value fails the request
#     CLOSED, naming the FIELD and never echoing the value.
#
#  3. Redaction of shipment / commercial-data identifiers (ISO 6346 container
#     numbers, long booking/AWB references, e-mail addresses) from the question
#     AND from caller-supplied passage text, so raw shipment identifiers never
#     reach the inner nodes or the checkpoint store. HS codes are deliberately
#     NOT redacted — they are public tariff data and exactly what this template
#     answers questions about.

import re
from typing import Any, ClassVar, Dict, List, Optional, Tuple

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event
from src.services.failure_message import INPUT_REJECTED
from src.services.progress import emit_progress
from src.schemas.state import finite_in_range, to_json

# ---------------------------------------------------------------------------
# Caller-field bounds
# ---------------------------------------------------------------------------

MAX_QUESTION_CHARS = 2000
MAX_KNOWLEDGE_ENTRIES = 25
MAX_ENTRY_TITLE_CHARS = 200
MAX_ENTRY_SOURCE_CHARS = 200
MAX_ENTRY_CONTENT_CHARS = 5000

TOP_K_MIN, TOP_K_MAX = 1, 20
SCORE_THRESHOLD_MIN, SCORE_THRESHOLD_MAX = 0.0, 1.0

# The five customs domains the knowledge base spans.
VALID_DOMAINS = ("hs_code", "naccs", "incoterms", "aeo", "customs_law")

# Passage ids are inert: they are matched and echoed, never interpreted.
_INERT_IDENTIFIER_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,63}$")

_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# ---------------------------------------------------------------------------
# Shipment / commercial-data identifier redaction
# ---------------------------------------------------------------------------

_IDENTIFIER_PATTERNS: List[re.Pattern[str]] = [
    # ISO 6346 shipping container number: 4-letter owner/category code + 7 digits.
    re.compile(r"\b[A-Z]{4}\d{7}\b"),
    # Long shipment / booking / AWB reference numbers: 11+ digits, optionally
    # grouped. Deliberately does NOT match 6-10 digit HS codes written without
    # separators (e.g. 8471300000) — those are public tariff data.
    re.compile(r"\b\d{4}[- ]?\d{4}[- ]?\d{3,}\b"),
    # E-mail addresses (broker / compliance-staff correspondence).
    re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
]
_REDACTION = "[REDACTED]"

# ---------------------------------------------------------------------------
# Instruction-override screen
# ---------------------------------------------------------------------------
#
# Anchored on whole phrases, and every alternative requires the prompt-specific
# noun. Ordinary customs language that borrows these verbs ("act as an importer
# of record", "override the classification", "ignore the previous declaration",
# "show me the rules for voluntary disclosure") does not match by construction.

_INSTRUCTION_OVERRIDE_RE = re.compile(
    r"(?:ignore|disregard|forget)\s+(?:all\s+|any\s+|the\s+)?"
    r"(?:previous|prior|above|earlier|preceding)\s+"
    r"(?:instruction|instructions|prompt|prompts|rule|rules)\b"
    r"|(?:reveal|show|print|repeat|output|disclose|dump)\s+(?:me\s+)?(?:your|the)\s+"
    r"(?:system\s+prompt|system\s+message|initial\s+prompt|hidden\s+instructions|"
    r"instructions\s+above)\b"
    r"|\byou\s+are\s+now\s+(?:a|an)\s+(?!importer|exporter|applicant|agent\s+of\s+record)"
    r"|\bact\s+as\s+(?:if\s+you\s+(?:are|were)\s+)?(?:a\s+|an\s+)?"
    r"(?:developer|admin|administrator|root|jailbroken|unrestricted)\s+mode\b"
    r"|\boverride\s+(?:your|the)\s+"
    r"(?:instruction|instructions|rule|rules|safety|guardrail|guardrails|"
    r"restriction|restrictions)\b"
    r"|\b(?:new|updated)\s+system\s+(?:prompt|instructions)\s*[:=]",
    re.IGNORECASE,
)

_REFUSAL_NOTE = "refused — instruction-override content"


def _redact_identifiers(text: str) -> str:
    """Redact shipment/commercial-data identifier tokens from free text."""
    for pattern in _IDENTIFIER_PATTERNS:
        text = pattern.sub(_REDACTION, text)
    return text


def _clean_free_text(value: str, limit: int) -> str:
    """Strip control characters, redact identifiers, cap the length."""
    return _redact_identifiers(_CONTROL_CHARS_RE.sub("", value))[:limit].strip()


def _inert_id(value: Any, field: str) -> Tuple[Optional[str], Optional[str]]:
    """Validate an inert-identifier field. Returns (value, error)."""
    if value is None:
        return None, None
    if not isinstance(value, str):
        return None, f"PreProcessNode: {field} must be a short lowercase identifier."
    candidate = value.strip().lower()
    if not candidate:
        return None, None
    if not _INERT_IDENTIFIER_RE.match(candidate):
        return None, f"PreProcessNode: {field} must be a short lowercase identifier."
    return candidate, None


def _bounded_int(value: Any, field: str, lo: int, hi: int) -> Tuple[Optional[int], Optional[str]]:
    """Validate a caller integer through the finite+bounded parser."""
    if value is None:
        return None, None
    parsed = finite_in_range(value, lo, hi)
    if parsed is None or parsed != int(parsed):
        return None, f"PreProcessNode: {field} must be a whole number between {lo} and {hi}."
    return int(parsed), None


def _bounded_float(value: Any, field: str, lo: float, hi: float) -> Tuple[Optional[float], Optional[str]]:
    """Validate a caller float through the finite+bounded parser."""
    if value is None:
        return None, None
    parsed = finite_in_range(value, lo, hi)
    if parsed is None:
        return None, f"PreProcessNode: {field} must be a number between {lo} and {hi}."
    return parsed, None


def _validate_domain(value: Any) -> Tuple[Optional[str], Optional[str]]:
    """Validate the caller's customs-domain narrowing option."""
    if value is None:
        return None, None
    if not isinstance(value, str):
        return None, "PreProcessNode: input_context.customs_domain must be one of " + ", ".join(VALID_DOMAINS) + "."
    slug = value.strip().lower()
    if not slug:
        return None, None
    if slug not in VALID_DOMAINS:
        return None, "PreProcessNode: input_context.customs_domain must be one of " + ", ".join(VALID_DOMAINS) + "."
    return slug, None


def _validate_entry(raw: Any, index: int) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Validate one caller-supplied knowledge passage. Returns (entry, error).

    Errors name the POSITION in the list and the field — never the content.
    """
    field = f"input_context.knowledge_entries[{index}]"
    if not isinstance(raw, dict):
        return None, f"PreProcessNode: {field} must be an object."

    entry_id, err = _inert_id(raw.get("id"), f"{field}.id")
    if err:
        return None, err
    if not entry_id:
        return None, f"PreProcessNode: {field}.id is required."

    category = raw.get("category")
    if not isinstance(category, str) or category.strip().lower() not in VALID_DOMAINS:
        return None, (
            f"PreProcessNode: {field}.category is required and must be one of " + ", ".join(VALID_DOMAINS) + "."
        )

    content = raw.get("content")
    if not isinstance(content, str) or not content.strip():
        return None, f"PreProcessNode: {field}.content is required."
    if len(content) > MAX_ENTRY_CONTENT_CHARS:
        return None, (f"PreProcessNode: {field}.content exceeds the " f"{MAX_ENTRY_CONTENT_CHARS}-character limit.")
    if _INSTRUCTION_OVERRIDE_RE.search(content):
        return None, f"PreProcessNode: {field}.content {_REFUSAL_NOTE}."

    title = raw.get("title")
    if title is not None and not isinstance(title, str):
        return None, f"PreProcessNode: {field}.title must be text."
    if isinstance(title, str):
        if len(title) > MAX_ENTRY_TITLE_CHARS:
            return None, (f"PreProcessNode: {field}.title exceeds the " f"{MAX_ENTRY_TITLE_CHARS}-character limit.")
        if _INSTRUCTION_OVERRIDE_RE.search(title):
            return None, f"PreProcessNode: {field}.title {_REFUSAL_NOTE}."

    source = raw.get("source")
    if source is not None and not isinstance(source, str):
        return None, f"PreProcessNode: {field}.source must be text."
    if isinstance(source, str):
        if len(source) > MAX_ENTRY_SOURCE_CHARS:
            return None, (f"PreProcessNode: {field}.source exceeds the " f"{MAX_ENTRY_SOURCE_CHARS}-character limit.")
        if _INSTRUCTION_OVERRIDE_RE.search(source):
            return None, f"PreProcessNode: {field}.source {_REFUSAL_NOTE}."

    return (
        {
            "id": entry_id,
            "category": category.strip().lower(),
            "title": _clean_free_text(title, MAX_ENTRY_TITLE_CHARS) if isinstance(title, str) else "",
            "source": _clean_free_text(source, MAX_ENTRY_SOURCE_CHARS) if isinstance(source, str) else "",
            "content": _clean_free_text(content, MAX_ENTRY_CONTENT_CHARS),
        },
        None,
    )


def _validate_entries(raw: Any) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """Validate the caller's knowledge corpus. Returns (entries, error)."""
    if raw is None:
        return [], None
    if not isinstance(raw, list):
        return [], "PreProcessNode: input_context.knowledge_entries must be a list."
    if len(raw) > MAX_KNOWLEDGE_ENTRIES:
        return [], (
            f"PreProcessNode: input_context.knowledge_entries accepts at most " f"{MAX_KNOWLEDGE_ENTRIES} entries."
        )
    entries: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(raw):
        entry, err = _validate_entry(item, index)
        if err or entry is None:
            return [], err or f"PreProcessNode: input_context.knowledge_entries[{index}] is invalid."
        if entry["id"] in seen:
            return [], f"PreProcessNode: input_context.knowledge_entries[{index}].id is a duplicate."
        seen.add(entry["id"])
        entries.append(entry)
    return entries, None


def _validate_caller_context(raw: Any) -> Tuple[Dict[str, Any], Optional[str]]:
    """Validate the whole input_context payload. Returns (validated, error)."""
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        return {}, "PreProcessNode: input_context must be an object."

    domain, err = _validate_domain(raw.get("customs_domain"))
    if err:
        return {}, err
    top_k, err = _bounded_int(raw.get("top_k"), "input_context.top_k", TOP_K_MIN, TOP_K_MAX)
    if err:
        return {}, err
    threshold, err = _bounded_float(
        raw.get("score_threshold"),
        "input_context.score_threshold",
        SCORE_THRESHOLD_MIN,
        SCORE_THRESHOLD_MAX,
    )
    if err:
        return {}, err
    entries, err = _validate_entries(raw.get("knowledge_entries"))
    if err:
        return {}, err

    return (
        {
            "customs_domain": domain,
            "top_k": top_k,
            "score_threshold": threshold,
            "knowledge_entries": entries,
        },
        None,
    )


class PreProcessNode(FunctionNode):
    """Validate and normalise everything the caller sent, or refuse the request.

    Input state keys:
        user_input:    the natural-language customs question
        input_context: per-request options and optional knowledge passages

    Output state keys (partial dict):
        validated_input:   redacted, length-capped question
        validated_context: JSON dict of the validated caller options
        enriched_context:  provenance for downstream observability
        status:            SUCCESS, or ERROR with error_log on any rejection
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.VERIFIED_EXTERNAL

    def _reject(self, message: str, code: str = "INVALID_REQUEST") -> Dict[str, Any]:
        """Fail CLOSED: nothing validated is carried forward."""
        if code:
            # A value the caller can correct: the run COMPLETES carrying the
            # reason so the request can be sent again on the same conversation.
            emit_progress(INPUT_REJECTED)
            return {
                "status": AgentStatus.SUCCESS.value,
                "error_code": code,
                "error_log": [message],
            }
        return {
            "status": AgentStatus.ERROR.value,
            "error_log": [message],
        }

    def execute(self, state: AgentState) -> Dict[str, Any]:
        user_input = state.get("user_input", "")
        raw_context = state.get("input_context", {})  # read-only

        if not user_input or not isinstance(user_input, str) or not user_input.strip():
            return self._reject("PreProcessNode: user_input is empty or missing", code="EMPTY_INPUT")

        question = user_input.strip()
        if len(question) > MAX_QUESTION_CHARS:
            return self._reject(
                f"PreProcessNode: user_input exceeds the {MAX_QUESTION_CHARS}-character limit.",
                code="QUESTION_TOO_LONG",
            )
        if _INSTRUCTION_OVERRIDE_RE.search(question):
            return self._reject(
                f"PreProcessNode: user_input {_REFUSAL_NOTE}.",
                # Not something the caller corrects by rewording: the refusal
                # terminates the run rather than inviting another attempt.
                code="",
            )

        validated_context, err = _validate_caller_context(raw_context)
        if err:
            # A caller-supplied knowledge passage can be refused here for
            # instruction-override content, on the same channel as an
            # out-of-bounds option. That refusal is not something the caller
            # corrects by rewording, so it stays terminating; every other
            # violation on this channel is a value the caller can fix.
            return self._reject(err, code="" if _REFUSAL_NOTE in err else "INVALID_REQUEST")

        validated_input = _clean_free_text(question, MAX_QUESTION_CHARS)

        # Domain audit: a customs question was accepted, redacted, and its caller
        # options bounds-checked. Counts only — never caller values.
        emit_trace_event(
            "pre_process_complete",
            {
                "input_chars": len(validated_input),
                "caller_entries": len(validated_context["knowledge_entries"]),
                "domain_narrowed": validated_context["customs_domain"] is not None,
            },
            state,
        )

        channel = raw_context.get("channel") if isinstance(raw_context, dict) else None
        return {
            "validated_input": validated_input,
            "validated_context": to_json(validated_context),
            "enriched_context": {
                "source": "CustomsComplianceDocumentQAAgent",
                "channel": _clean_free_text(channel, 64) if isinstance(channel, str) else "unknown",
            },
            "status": AgentStatus.SUCCESS.value,
        }
