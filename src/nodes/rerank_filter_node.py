"""AgentCore Platform v1.0"""

# LOG-C2-039 - RerankFilterNode
# Domain node 3: rerank the retrieval candidates and enforce the relevance floor.
# Deterministic: a small domain-match boost on top of the retrieval score
# (candidates are already domain-filtered by RetrieveNode when a specific
# customs_domain was classified; the boost still helps when customs_domain ==
# "general" and candidates span multiple domains), drop everything below
# `score_threshold`, cap the survivors at `top_k`.
#
# Runtime tuning (top_k, score_threshold) comes from the state field
# retrieval_config, which the inner graph seeds from config/config.yaml. Module
# defaults mirror that file so the node still works if the config is unreadable.
# A caller-supplied top_k wins when it is stricter, and a caller-supplied
# score_threshold — bounds-checked at the boundary node — replaces the
# configured floor. execute() takes no config parameter; seeding through State
# is the sanctioned route.
#
# Wired by the inner graph (DomainWorkflowGraph).
# Returns only changed state keys (partial dict).

import math
from typing import Any, ClassVar, Dict, List

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event
from src.schemas.state import finite_in_range, from_json, to_json

# Defaults mirror the `retrieval` block in config/config.yaml.
_DEFAULT_RETRIEVAL: Dict[str, Any] = {
    "top_k": 4,
    "score_threshold": 0.25,
}

# Boost applied when a candidate's category matches the caller's classified domain.
_DOMAIN_BOOST = 0.1


def _resolve_retrieval_config(state: AgentState) -> Dict[str, Any]:
    """Effective retrieval config: the seeded runtime block over module defaults."""
    effective = dict(_DEFAULT_RETRIEVAL)  # local copy - never mutate the module default
    from_state = from_json(state.get("retrieval_config"), None)
    if isinstance(from_state, dict):
        effective.update(from_state)
    return effective


class RerankFilterNode(FunctionNode):
    """Rerank candidates, apply the score threshold, cap at top_k.

    Input state keys:
        retrieved_documents: JSON list of scored candidates (from RetrieveNode)
        query_filters:       JSON dict with the classified/overridden domain +
                             optional top_k
        customs_domain:      resolved domain filter ("general" = no boost target)
        validated_context:   bounds-checked caller options (JSON)
        retrieval_config:    runtime retrieval block (JSON)

    Output state keys (partial dict):
        ranked_documents: JSON list of surviving passages (score desc, <= top_k)
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: AgentState) -> Dict[str, Any]:
        candidates: List[Dict[str, Any]] = from_json(state.get("retrieved_documents"), []) or []
        filters = from_json(state.get("query_filters"), {}) or {}
        retrieval_cfg = _resolve_retrieval_config(state)

        try:
            top_k = int(retrieval_cfg.get("top_k", _DEFAULT_RETRIEVAL["top_k"]))
        except (TypeError, ValueError, OverflowError):
            top_k = int(_DEFAULT_RETRIEVAL["top_k"])
        top_k = max(1, min(20, top_k))
        # A stricter caller override (validated by InputValidateNode) wins.
        caller_top_k = filters.get("top_k")
        if isinstance(caller_top_k, int) and 1 <= caller_top_k < top_k:
            top_k = caller_top_k

        try:
            score_threshold = float(retrieval_cfg.get("score_threshold", _DEFAULT_RETRIEVAL["score_threshold"]))
        except (TypeError, ValueError, OverflowError):
            score_threshold = float(_DEFAULT_RETRIEVAL["score_threshold"])
        if not math.isfinite(score_threshold):
            score_threshold = float(_DEFAULT_RETRIEVAL["score_threshold"])
        score_threshold = max(0.0, min(1.0, score_threshold))

        # A caller-supplied relevance floor, already bounds-checked at the
        # boundary node, replaces the configured one.
        caller = from_json(state.get("validated_context"), {}) or {}
        caller_threshold = caller.get("score_threshold")
        if isinstance(caller_threshold, (int, float)) and not isinstance(caller_threshold, bool):
            checked = finite_in_range(caller_threshold, 0.0, 1.0)
            if checked is not None:
                score_threshold = checked

        domain = state.get("customs_domain") or filters.get("domain")

        reranked: List[Dict[str, Any]] = []
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            entry = dict(candidate)  # local copy - inputs stay immutable
            try:
                score = float(entry.get("score", 0.0))
            except (TypeError, ValueError):
                score = 0.0
            if domain and domain != "general" and str(entry.get("category", "")).lower() == str(domain).lower():
                score = min(1.0, score + _DOMAIN_BOOST)
            entry["score"] = round(score, 4)
            reranked.append(entry)

        # Deterministic ordering: score desc, then id asc for stable ties.
        reranked.sort(key=lambda c: (-c.get("score", 0.0), str(c.get("id", ""))))

        kept = [c for c in reranked if c.get("score", 0.0) >= score_threshold][:top_k]
        dropped = len(reranked) - len(kept)

        # Domain audit: rerank + relevance floor applied.
        emit_trace_event(
            "rerank_filter_complete",
            {
                "kept": len(kept),
                "dropped": dropped,
                "score_threshold": score_threshold,
                "top_k": top_k,
            },
            state,
        )

        return {"ranked_documents": to_json(kept)}
