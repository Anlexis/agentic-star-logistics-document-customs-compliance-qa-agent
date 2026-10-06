"""AgentCore Platform v1.0"""

# LOG-C2-039 — RetrieveNode
#
# Domain node 2: score the customs-compliance knowledge base against the
# question. Deterministic keyword retrieval — no embedding model, no vector
# store — so the same question over the same corpus always returns the same
# passages in the same order and an answer can be audited after the fact. The
# retrieved_documents contract is store-agnostic, so swapping in a vector or
# hybrid client only changes this node's internals.
#
# Corpus: the passages the caller supplied for this request, when there are any.
# With none supplied, the seeded reference corpus in config/kb/ is searched
# instead, so a fresh checkout answers real customs questions before anything is
# wired in. Caller passages arrive already bounds-checked and redacted by
# PreProcessNode; this node never sees raw caller data.
#
# Runtime tuning (top_k, kb_path) comes from the state field retrieval_config,
# which the inner graph seeds from config/config.yaml. Module defaults mirror
# that file so the node still works if the config is unreadable. execute() takes
# no config parameter — seeding through State is the sanctioned route.
#
# Domain routing: candidates are narrowed to the resolved customs_domain unless
# it is "general", in which case all five domains are searched.
#
# Wired by the inner graph (DomainWorkflowGraph).
# Returns only changed state keys (partial dict).

import json
import re
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Tuple

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event
from src.schemas.state import from_json, to_json

# Defaults mirror the `retrieval` block in config/config.yaml.
_DEFAULT_RETRIEVAL: Dict[str, Any] = {
    "top_k": 4,
    "score_threshold": 0.25,
    "kb_path": "config/kb/customs_kb.json",
}

# Repo root: src/nodes/retrieve_node.py -> parents[2].
_REPO_ROOT = Path(__file__).resolve().parents[2]

# Minimal stopword set for query tokenisation (deterministic, no NLP deps).
_STOPWORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "and",
        "or",
        "of",
        "to",
        "in",
        "on",
        "for",
        "is",
        "are",
        "be",
        "with",
        "under",
        "what",
        "which",
        "when",
        "how",
        "do",
        "does",
        "must",
        "should",
        "before",
        "after",
        "by",
        "at",
        "from",
        "that",
        "this",
        "it",
        "as",
        "was",
        "were",
        "can",
        "may",
        "any",
    }
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Per-field match weights: a query token found in the title counts more than one
# found only in the body content.
_TITLE_WEIGHT = 1.0
_TAG_WEIGHT = 0.8
_CONTENT_WEIGHT = 0.5

# Excerpt length carried into retrieved_documents (keeps State small).
_EXCERPT_CHARS = 400


def _tokenize(text: str) -> List[str]:
    """Lowercase alphanumeric tokens, stopwords and 1-2 char noise removed."""
    return [t for t in _TOKEN_RE.findall(text.lower()) if len(t) > 2 and t not in _STOPWORDS]


def _resolve_retrieval_config(state: AgentState) -> Dict[str, Any]:
    """Effective retrieval config: the seeded runtime block over module defaults."""
    effective = dict(_DEFAULT_RETRIEVAL)  # local copy — never mutate the module default
    from_state = from_json(state.get("retrieval_config"), None)
    if isinstance(from_state, dict):
        effective.update(from_state)
    return effective


def _load_seeded_corpus(kb_path: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Load the seeded reference corpus. Missing / malformed file degrades."""
    notes: List[str] = []
    path = Path(kb_path)
    if not path.is_absolute():
        path = _REPO_ROOT / path
    try:
        entries = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        notes.append("RetrieveNode: the seeded knowledge base is not readable.")
        return [], notes
    if not isinstance(entries, list):
        notes.append("RetrieveNode: the seeded knowledge base root must be a JSON list.")
        return [], notes
    return [e for e in entries if isinstance(e, dict)], notes


def _score_entry(entry: Dict[str, Any], query_tokens: List[str]) -> float:
    """Per-entry relevance: best field-weight per query token, averaged."""
    if not query_tokens:
        return 0.0
    title_tokens = set(_tokenize(str(entry.get("title", ""))))
    tag_tokens = set(_tokenize(" ".join(str(t) for t in entry.get("tags", []))))
    content_tokens = set(_tokenize(str(entry.get("content", ""))))
    total = 0.0
    for token in query_tokens:
        if token in title_tokens:
            total += _TITLE_WEIGHT
        elif token in tag_tokens:
            total += _TAG_WEIGHT
        elif token in content_tokens:
            total += _CONTENT_WEIGHT
    return round(total / len(query_tokens), 4)


class RetrieveNode(FunctionNode):
    """Score the customs knowledge base against the query and emit candidates.

    Input state keys:
        search_query:      normalised question (from InputValidateNode)
        customs_domain:    resolved domain filter ("general" = no filter)
        validated_context: bounds-checked caller options, including any
                           caller-supplied knowledge passages (JSON)
        retrieval_config:  runtime retrieval block (JSON)

    Output state keys (partial dict):
        retrieved_documents: JSON list of scored candidates (score desc)
        intake_notes:        (on corpus anomalies) JSON list[str]
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: AgentState) -> Dict[str, Any]:
        query = state.get("search_query") or state.get("validated_input") or state.get("user_input", "")
        domain = state.get("customs_domain") or "general"
        retrieval_cfg = _resolve_retrieval_config(state)
        caller = from_json(state.get("validated_context"), {}) or {}

        try:
            top_k = int(retrieval_cfg.get("top_k", _DEFAULT_RETRIEVAL["top_k"]))
        except (TypeError, ValueError, OverflowError):
            top_k = int(_DEFAULT_RETRIEVAL["top_k"])
        top_k = max(1, min(20, top_k))

        caller_entries = caller.get("knowledge_entries") or []
        if isinstance(caller_entries, list) and caller_entries:
            entries: List[Dict[str, Any]] = [e for e in caller_entries if isinstance(e, dict)]
            notes: List[str] = []
            corpus = "caller"
        else:
            entries, notes = _load_seeded_corpus(str(retrieval_cfg.get("kb_path", _DEFAULT_RETRIEVAL["kb_path"])))
            corpus = "seeded"

        if domain and domain != "general":
            entries = [e for e in entries if str(e.get("category", "")).lower() == domain]

        query_tokens = _tokenize(query if isinstance(query, str) else "")

        candidates: List[Dict[str, Any]] = []
        for entry in entries:
            score = _score_entry(entry, query_tokens)
            if score <= 0.0:
                continue
            candidates.append(
                {
                    "id": str(entry.get("id", "")),
                    "title": str(entry.get("title", "")),
                    "category": str(entry.get("category", "")),
                    "source": str(entry.get("source", "")),
                    "score": score,
                    "excerpt": str(entry.get("content", ""))[:_EXCERPT_CHARS],
                }
            )

        # Deterministic ordering: score desc, then id asc for stable ties.
        candidates.sort(key=lambda c: (-c["score"], c["id"]))
        # Keep a candidate pool wider than top_k — RerankFilterNode makes the
        # final cut after the domain boost + threshold.
        pool_size = max(top_k * 3, 10)
        candidates = candidates[:pool_size]

        # Domain audit: retrieval pass completed. Counts only — never content.
        emit_trace_event(
            "retrieve_complete",
            {
                "candidates": len(candidates),
                "kb_entries": len(entries),
                "query_tokens": len(query_tokens),
                "customs_domain": domain,
                "top_k": top_k,
                "corpus": corpus,
            },
            state,
        )

        out: Dict[str, Any] = {"retrieved_documents": to_json(candidates)}
        if notes:
            prior = from_json(state.get("intake_notes"), []) or []
            out["intake_notes"] = to_json(list(prior) + notes)
        return out
