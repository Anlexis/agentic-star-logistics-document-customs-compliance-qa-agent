"""AgentCore Platform v1.0"""

# LOG-C2-039 — InputValidateNode
#
# Domain node 1: settle the question and the search options for this request,
# and resolve which of the five customs domains the knowledge base spans it
# belongs to (hs_code | naccs | incoterms | aeo | customs_law).
#
# Two request shapes are accepted:
#
#   plain text                    the whole string is the question
#   {"query": "...",              a question plus per-request options, carried
#    "domain": "...",             inside the string the outer graph passes in
#    "top_k": N}
#
# Per-request options also arrive on the caller-context channel, already
# bounds-checked by PreProcessNode. Those WIN over the envelope: they are the
# only form that has been through the boundary node's validation. Envelope
# numbers are re-validated here through the same finite+bounded parser, so
# neither channel can smuggle a NaN or an out-of-range value past the checks.
#
# Wired by the inner graph (DomainWorkflowGraph).
# Returns only changed state keys (partial dict).

import json
import re
from typing import Any, ClassVar, Dict, List, Optional, Tuple

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event
from src.schemas.state import finite_in_range, from_json, to_json

# Hard cap on the normalised query length (defence in depth on input size).
_MAX_QUERY_CHARS = 2000

# Bounds for the caller-supplied top_k override.
_TOP_K_MIN = 1
_TOP_K_MAX = 20

_WHITESPACE_RE = re.compile(r"\s+")

# The five customs domains this template's knowledge base spans.
_VALID_DOMAINS = ("hs_code", "naccs", "incoterms", "aeo", "customs_law")

# Deterministic keyword classifier — no model, no network. Each phrase is
# matched as a WHOLE word or phrase, never as a bare substring: three-letter
# Incoterms codes are the content of this domain and they hide inside ordinary
# customs words. Unanchored, "cif" matches inside "specific", "cip" inside
# "principle" and "dap" inside "adapt" — so "what is the specific duty rate"
# classified as incoterms, and RetrieveNode then filtered the customs-law
# passages that actually answer it out of the candidate set.
_DOMAIN_KEYWORDS: Dict[str, List[str]] = {
    "hs_code": [
        "hs code",
        "harmonized system",
        "harmonised system",
        "tariff classification",
        "tariff code",
        "advance classification",
        "事前教示",
        "classification ruling",
    ],
    "naccs": [
        "naccs",
        "import declaration",
        "export declaration",
        "customs entry",
        "customs clearance",
        "declaration form",
        "entry amendment",
    ],
    "incoterms": [
        "incoterms",
        "fob",
        "cif",
        "exw",
        "dap",
        "ddp",
        "fca",
        "cpt",
        "cip",
        "delivery term",
        "shipping term",
        "risk transfer",
    ],
    "aeo": [
        "aeo",
        "authorized economic operator",
        "authorised economic operator",
        "broker accreditation",
        "aeo certification",
        "aeo application",
    ],
    "customs_law": [
        "関税法",
        "customs act",
        "customs law",
        "customs valuation",
        "duty rate",
        "tariff schedule",
        "post-entry correction",
        "voluntary disclosure",
    ],
}


def _compile_phrase(phrase: str) -> Optional[re.Pattern[str]]:
    """Compile one keyword phrase into a whole-word matcher.

    ASCII phrases get alphanumeric guards on both ends and tolerate any run of
    spaces or hyphens between words, so "hs code", "hs-code" and "hs  code" all
    match while "specific" never yields "cif". Phrases with no word boundaries
    to anchor on (Japanese terms such as 関税法) are matched as substrings, which
    is correct for a script that does not separate words.
    """
    if not phrase.isascii():
        return None
    body = r"[\s\-]+".join(re.escape(word) for word in phrase.split())
    return re.compile(rf"(?<![a-z0-9])(?:{body})(?![a-z0-9])", re.IGNORECASE)


_DOMAIN_MATCHERS: Dict[str, List[Tuple[str, Optional[re.Pattern[str]]]]] = {
    domain: [(phrase, _compile_phrase(phrase)) for phrase in phrases] for domain, phrases in _DOMAIN_KEYWORDS.items()
}


def _classify_domain(text: str, notes: List[str]) -> str:
    """Whole-word keyword-count classifier over the five customs domains.

    Returns the single highest-scoring domain. With no match, or with two or
    more domains tied at the top, returns "general" so RetrieveNode searches
    every domain: narrowing to one arbitrarily chosen domain filters out the
    passages that answer the question, which is the expensive mistake here.
    """
    scores = {domain: 0 for domain in _VALID_DOMAINS}
    for domain, matchers in _DOMAIN_MATCHERS.items():
        for phrase, pattern in matchers:
            if pattern.search(text) if pattern is not None else (phrase in text):
                scores[domain] += 1
    best = max(scores.values())
    if best == 0:
        notes.append("InputValidateNode: no domain keywords matched — searching across " "all customs domains.")
        return "general"
    leaders = [domain for domain in _VALID_DOMAINS if scores[domain] == best]
    if len(leaders) > 1:
        notes.append(
            "InputValidateNode: the question matched several customs domains — " "searching across all of them."
        )
        return "general"
    return leaders[0]


def _coerce_top_k(value: Any, notes: List[str]) -> Optional[int]:
    """Guarded coercion of an envelope-supplied top_k.

    Runs through the finite+bounded parser: NaN and Infinity parse as floats
    (and arrive intact through raw JSON), and int() on an infinite float raises
    OverflowError, which a plain (TypeError, ValueError) guard does not catch.
    An unusable value is dropped with a note rather than raised — a bad search
    option must not fail an otherwise valid customs question — and the option
    then falls back to the configured default.
    """
    if value is None:
        return None
    parsed = finite_in_range(value, _TOP_K_MIN, _TOP_K_MAX)
    if parsed is None or parsed != int(parsed):
        notes.append("InputValidateNode: unusable top_k ignored.")
        return None
    return int(parsed)


def _coerce_domain(value: Any, notes: List[str]) -> Optional[str]:
    """Validate an envelope-supplied domain override against the known slugs."""
    if not isinstance(value, str) or not value.strip():
        return None
    slug = value.strip().lower()
    if slug in _VALID_DOMAINS:
        return slug
    notes.append("InputValidateNode: unknown domain override ignored — auto-classifying instead.")
    return None


class InputValidateNode(FunctionNode):
    """Settle the question, the search options, and the customs domain.

    Input state keys:
        validated_input | user_input: the request payload
        validated_context:            bounds-checked caller options (JSON)

    Output state keys (partial dict):
        search_query:   normalised free-text customs question
        customs_domain: resolved domain slug (hs_code | naccs | incoterms | aeo
                        | customs_law | general)
        query_filters:  JSON dict {"domain": str, "top_k": int|None}
        intake_notes:   (when anomalies were seen) JSON list[str]
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: AgentState) -> Dict[str, Any]:
        raw = state.get("validated_input") or state.get("user_input", "")
        caller = from_json(state.get("validated_context"), {}) or {}
        notes: List[str] = []

        query = ""
        domain_override: Optional[str] = None
        top_k: Optional[int] = None

        if isinstance(raw, str) and raw.strip():
            payload: Any = None
            text = raw.strip()
            if text.startswith("{"):
                try:
                    payload = json.loads(text)
                except (json.JSONDecodeError, ValueError):
                    notes.append(
                        "InputValidateNode: JSON-looking input did not parse — " "treated as plain text query."
                    )
            if isinstance(payload, dict):
                query = str(payload.get("query") or payload.get("question") or "")
                domain_override = _coerce_domain(payload.get("domain"), notes)
                top_k = _coerce_top_k(payload.get("top_k"), notes)
            else:
                query = text
        else:
            notes.append("InputValidateNode: empty request — no question to search.")

        # The caller-context channel went through PreProcessNode's bounds checks,
        # so it wins over anything carried in the envelope.
        if isinstance(caller.get("customs_domain"), str):
            domain_override = caller["customs_domain"]
        if isinstance(caller.get("top_k"), int) and not isinstance(caller.get("top_k"), bool):
            top_k = caller["top_k"]

        # Normalise whitespace and cap length.
        query = _WHITESPACE_RE.sub(" ", query).strip()
        if len(query) > _MAX_QUERY_CHARS:
            query = query[:_MAX_QUERY_CHARS]
            notes.append(f"InputValidateNode: query truncated to {_MAX_QUERY_CHARS} chars.")

        customs_domain = domain_override or _classify_domain(query, notes)
        filters = {"domain": customs_domain, "top_k": top_k}

        # Domain audit: request parsed, normalised, and domain-resolved.
        emit_trace_event(
            "input_validate_complete",
            {
                "query_chars": len(query),
                "customs_domain": customs_domain,
                "domain_was_explicit": domain_override is not None,
                "has_top_k_override": top_k is not None,
            },
            state,
        )

        out: Dict[str, Any] = {
            "search_query": query,
            "customs_domain": customs_domain,
            "query_filters": to_json(filters),
        }
        if notes:
            out["intake_notes"] = to_json(notes)
        return out
