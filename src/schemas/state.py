"""AgentCore Platform v1.0"""

# State must be a flat TypedDict — never a Pydantic BaseModel. LangGraph
# checkpoints use msgpack serialization; Pydantic objects cause silent
# corruption. Extend AgentState with agent-specific fields only. Do NOT add
# credentials, secrets, or Pydantic models.
#
# msgpack safety: structured fields (dict / list[dict]) are stored as JSON
# STRINGS, not bare Python containers — a bare dict/list in a checkpointed State
# field is a state-safety violation. Producers serialize with to_json() on
# write; consumers deserialize with from_json() on read.
#
# LOG-C2-039 — Customs Compliance Document Q&A Agent (Cat 2 RAG). Two-layer
# nested Cat 2 graph: outer backbone (AgentBaseGraph) + inner domain workflow
# (BaseGraph). Fields below cover both layers.
#
# Confidentiality note: shipment / commercial-data identifiers (ISO 6346
# container numbers, long booking/AWB reference numbers, e-mail addresses) in
# the request payload are surface-stripped by PreProcessNode before any field is
# written to State — on the question channel AND on the caller-context channel.
# Only the normalised customs question, knowledge-base passage summaries, and the
# final checklist answer are persisted — never raw shipment identifiers. HS codes
# themselves are intentionally NOT treated as confidential: they are public
# tariff data and are exactly what this template answers questions about.

import json
import math
from typing import Any, NotRequired, Optional

from framework.schemas.agent_state import AgentState


def to_json(value: Any) -> Optional[str]:
    """Serialize a dict/list State field to a JSON string (msgpack safety).

    None passes through unchanged so an 'unset' field stays distinguishable from
    an empty container.
    """
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False)


def from_json(value: Optional[str], default: Any = None) -> Any:
    """Deserialize a JSON-string State field back to its dict/list.

    None / empty / malformed input -> the supplied ``default`` so a missing or
    corrupt field is non-fatal for the consuming node.
    """
    if not value:
        return default
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return default


def finite_in_range(value: Any, lo: float, hi: float) -> Optional[float]:
    """Parse a caller-controlled numeric: FINITE float within [lo, hi], else None.

    Rejects bools, non-numerics, and — critically — non-finite values. float()
    happily parses "NaN"/"Infinity", Python's json accepts bare NaN and Infinity
    in request bodies, and IEEE NaN comparisons are always False, which turns a
    threshold check into a silent pass. int() on an infinite float raises
    OverflowError, which a plain (TypeError, ValueError) guard does not catch.
    Every caller-supplied number comes through here so an unusable value fails
    CLOSED instead of disabling the check it was meant to control.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed) or not lo <= parsed <= hi:
        return None
    return parsed


class State(AgentState):
    """Flat TypedDict for LOG-C2-039.

    All shared fields (user_input, input_context, status, session_id,
    node_history, error_log, hitl_*, etc.) are inherited from AgentState. Domain
    fields are NotRequired so the TypedDict is valid at graph initialisation,
    before any node has written a value.
    """

    # ------------------------------------------------------------------
    # Outer layer — set by PreProcessNode / the main-slot GraphNode's
    # merge_output
    # ------------------------------------------------------------------

    # Shipment/commercial-data-stripped, validated question produced by
    # PreProcessNode. Raw input is NOT persisted beyond PreProcessNode.
    validated_input: NotRequired[str]

    # JSON STRING (to_json) of the VALIDATED caller options taken from
    # input_context. PreProcessNode is the only writer: every field is
    # bounds-checked there, so inner nodes never see raw caller data.
    # Deserialised shape: {"customs_domain": str | None, "top_k": int | None,
    # "score_threshold": float | None, "knowledge_entries": list[dict]}.
    validated_context: NotRequired[Optional[str]]

    # Final customs-compliance checklist answer, mapped from the inner graph's
    # formatted_answer output via merge_output.
    customs_answer: NotRequired[str]

    # ------------------------------------------------------------------
    # Inner layer — domain nodes (DomainWorkflowGraph)
    # ------------------------------------------------------------------

    # InputValidateNode outputs
    # Normalised free-text customs question (whitespace-collapsed, length-capped).
    search_query: NotRequired[str]

    # Classified/validated customs-domain slug: "hs_code" | "naccs" | "incoterms"
    # | "aeo" | "customs_law" | "general" (no domain resolved — RetrieveNode then
    # searches across all five domains instead of narrowing).
    customs_domain: NotRequired[str]

    # JSON STRING (to_json) of the effective query params. Deserialised shape:
    # {"domain": str | None, "top_k": int | None}. Consumers (RerankFilterNode)
    # read it back via from_json().
    query_filters: NotRequired[Optional[str]]

    # Runtime `retrieval` block forwarded from config/config.yaml by the
    # main-slot GraphNode's _parent_config() -> DomainWorkflowGraph.
    # _extra_initial_state(). JSON STRING (to_json) of
    # {"top_k": int, "score_threshold": float, "kb_path": str}. Consumers
    # (RetrieveNode, RerankFilterNode) read it back via from_json().
    retrieval_config: NotRequired[Optional[str]]

    # RetrieveNode output
    # JSON STRING (to_json) of scored knowledge-base candidates. Deserialised
    # shape: list[dict], each entry {"id": str, "title": str, "category": str,
    # "source": str, "score": float, "excerpt": str}. Consumers
    # (RerankFilterNode) read it back via from_json().
    retrieved_documents: NotRequired[Optional[str]]

    # RerankFilterNode output
    # JSON STRING (to_json) of domain-boosted + threshold-filtered passages,
    # capped at top_k. Same entry shape as retrieved_documents. Consumers
    # (GenerateAnswerNode) read it back via from_json().
    ranked_documents: NotRequired[Optional[str]]

    # GenerateAnswerNode outputs
    # Rule-assembled checklist answer body with numbered [n] citation markers (or
    # the cite-or-refuse no-coverage message).
    grounded_answer: NotRequired[str]

    # JSON STRING (to_json) of citations. Deserialised shape: list[dict], each
    # entry {"ref": int, "id": str, "title": str, "source": str}. Consumers
    # (OutputFormatNode, the outer get_output()) read it back via from_json().
    citations: NotRequired[Optional[str]]

    # JSON STRING (to_json) of the individual checklist point strings (the
    # structured surface for get_output()). Deserialised shape: list[str].
    checklist_items: NotRequired[Optional[str]]

    # Tariff/currency-schedule annotation attached to a coverage-bearing answer
    # (empty string on the no-coverage / cite-or-refuse path — nothing to
    # annotate).
    currency_note: NotRequired[str]

    # OutputFormatNode output
    # Final formatted answer (checklist + domain line + sources + currency note +
    # the non-suppressible customs-guidance disclaimer). Written by
    # OutputFormatNode; surfaced to the outer graph via get_output() ->
    # merge_output().
    formatted_answer: NotRequired[str]

    # Validation / parse notes accumulated during intake (no caller values).
    # JSON STRING (to_json) of list[str].
    intake_notes: NotRequired[Optional[str]]

    # ------------------------------------------------------------------
    # Tracing / audit — framework-managed; do NOT write from node code
    # ------------------------------------------------------------------

    trace_id: Optional[str]
    correlation_id: Optional[str]
    error_code: Optional[str]
    # node_history inherited from AgentState; listed here for clarity
    # node_history: Optional[List[str]]
