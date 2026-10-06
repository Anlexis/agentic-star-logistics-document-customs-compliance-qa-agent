"""AgentCore Platform v1.0"""

# src/graph/context_bridge.py — carries the caller's validated options across
# the outer→inner graph boundary.
#
# Why this exists: the framework's GraphNode.execute() invokes the inner graph
# as `subgraph.invoke(user_input, session_id=..., ctx=...)`. It forwards no
# caller context, so an inner-node read of the caller's per-request options
# would see nothing at all through the whole nested graph — the search options
# and the caller-supplied knowledge passages would silently have no effect.
#
# The sanctioned subclass hooks carry it across:
#
#   CustomsComplianceDocumentQAGraphNode.extract_input(state)
#       [runs BEFORE subgraph.invoke]  -> set_caller_context(...)
#   DomainWorkflowGraph._extra_initial_state()
#       [runs INSIDE subgraph.invoke]  -> {"validated_context": get_caller_context()}
#
# What crosses is the VALIDATED context PreProcessNode produced — the bounds
# checks happen once, at the boundary node, and raw caller data never reaches
# the inner graph. A ContextVar keeps the hand-off correct per thread/task, so
# concurrent invocations in one process cannot observe each other's context.

from contextvars import ContextVar
from typing import Optional

_CALLER_CONTEXT: ContextVar[Optional[str]] = ContextVar("log_c2_039_caller_context", default=None)


def set_caller_context(validated_context: Optional[str]) -> None:
    """Stash the validated caller context for the imminent inner-graph invoke."""
    _CALLER_CONTEXT.set(validated_context if isinstance(validated_context, str) else None)


def get_caller_context() -> Optional[str]:
    """Read (without consuming) the stashed caller context; None when unset."""
    return _CALLER_CONTEXT.get()
