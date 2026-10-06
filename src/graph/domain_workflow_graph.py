"""AgentCore Platform v1.0"""

# LOG-C2-039 - DomainWorkflowGraph (inner BaseGraph)
#
# This is the INNER graph for the Cat 2 two-layer nested architecture. It
# encapsulates the full customs-compliance document Q&A domain workflow:
#
#   START -> input_validate -> retrieve -> rerank_filter
#         -> generate_answer -> output_format -> END
#
# Called by CustomsComplianceDocumentQAGraphNode.get_subgraph() (graph.py).
# get_output() shapes the sub_result dict consumed by merge_output() there.
#
# Rules enforced:
#   - Inherits BaseGraph (fully custom topology - no forced backbone)
#   - Implements all 7 BaseGraph ABC methods
#   - register_nodes() does NOT call super() (abstract in BaseGraph)
#   - register_nodes() instantiates every domain node with NO ctor args
#   - Does NOT register initialize / finalize (outer backbone concerns)
#   - get_output() designed together with CustomsComplianceDocumentQAGraphNode.merge_output()

from typing import Any, Dict

from langgraph.graph import END, START

from framework.graph.base_graph import BaseGraph
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from src.graph.context_bridge import get_caller_context
from src.nodes.generate_answer_node import GenerateAnswerNode
from src.nodes.input_validate_node import InputValidateNode
from src.nodes.output_format_node import OutputFormatNode
from src.nodes.rerank_filter_node import RerankFilterNode
from src.nodes.retrieve_node import RetrieveNode
from src.schemas.state import State, to_json


class DomainWorkflowGraph(BaseGraph):
    """Inner domain workflow graph for LOG-C2-039.

    Inherits BaseGraph directly for a fully custom node topology. Called by
    CustomsComplianceDocumentQAGraphNode.get_subgraph() in graph.py, which
    passes the runtime config (`_parent_config()`) into the ctor.

    Pipeline (linear):
        START
          -> input_validate  (InputValidateNode)  - parse + classify customs domain
          -> retrieve        (RetrieveNode)       - keyword-score the seeded customs KB
          -> rerank_filter   (RerankFilterNode)   - domain-boost / threshold / top_k cut
          -> generate_answer (GenerateAnswerNode) - checklist answer + citations + currency note
          -> output_format   (OutputFormatNode)   - final format + non-suppressible disclaimer
          -> END

    All nodes are FunctionNode subclasses returning partial-dict state updates.
    initialize / finalize are outer backbone concerns - not registered here.
    """

    # -- Identity --------------------------------------------------------------

    @property
    def name(self) -> str:
        """Unique identifier for this inner graph."""
        return "log_c2_039_customs_compliance_qa_workflow"

    @property
    def state_schema(self) -> type:
        """TypedDict subclass shared across inner and outer graph."""
        return State

    # -- Config validation -----------------------------------------------------

    def _validate_config(self) -> None:
        """Validate inner graph config before compilation.

        The forwarded `retrieval` block (top_k / score_threshold / kb_path) is
        read per-call by the domain nodes with safe defaults, so absence is
        non-fatal. Validation is permissive here rather than raising.
        """
        pass

    # -- Config + caller context forwarding into state --------------------------

    def _extra_initial_state(self) -> Dict[str, Any]:
        """Seed the inner state with the runtime config and the caller context.

        Two things have to reach the domain nodes and neither arrives on its
        own:

        `retrieval_config` — the `retrieval` block the outer GraphNode
        forwarded from config/config.yaml under config["configurable"]. It is
        seeded as a JSON STRING, not a bare dict, so it stays checkpoint-safe.
        RetrieveNode / RerankFilterNode read it directly, since execute() takes
        no config parameter.

        `validated_context` — the caller's bounds-checked options and knowledge
        passages, handed over by the outer graph through the context bridge.
        The framework does not forward caller context into a nested graph, so
        without this the caller's search options would silently do nothing.
        """
        retrieval = (self.config or {}).get("configurable", {}).get("retrieval") or {}
        extra: Dict[str, Any] = {"retrieval_config": to_json(retrieval)}
        caller_context = get_caller_context()
        if caller_context is not None:
            extra["validated_context"] = caller_context
        return extra

    # -- Node registration -----------------------------------------------------

    def register_nodes(self) -> None:
        """Register all 5 domain nodes.

        No super() call - BaseGraph.register_nodes() is abstract. Do NOT
        register initialize or finalize; those are outer backbone concerns
        handled by AgentBaseGraph in graph.py.

        Every node is instantiated with NO constructor arguments - SDK
        v1.0.0rc1 FunctionNode subclasses take no required __init__; config
        flows in via State (execute(self, state) -> dict, no config
        parameter). Every key registered here is referenced in add_edges().
        """
        self._nodes["input_validate"] = InputValidateNode()
        self._nodes["retrieve"] = RetrieveNode()
        self._nodes["rerank_filter"] = RerankFilterNode()
        self._nodes["generate_answer"] = GenerateAnswerNode()
        self._nodes["output_format"] = OutputFormatNode()

    # -- Edge wiring -----------------------------------------------------------

    def add_edges(self) -> None:
        """Wire the linear customs Q&A domain topology.

        Each step passes its partial-dict output into the shared State. For
        this template the topology is intentionally linear - no conditional
        branching between domain nodes. route() is implemented as required by
        the ABC but add_conditional_edges() is not used.
        """
        self._sg.add_edge(START, "input_validate")
        self._sg.add_edge("input_validate", "retrieve")
        self._sg.add_edge("retrieve", "rerank_filter")
        self._sg.add_edge("rerank_filter", "generate_answer")
        self._sg.add_edge("generate_answer", "output_format")
        self._sg.add_edge("output_format", END)

    # -- Routing ---------------------------------------------------------------

    def route(self, state: AgentState) -> str:
        """Conditional routing - required by BaseGraph ABC.

        For this linear topology add_conditional_edges() is not used, so this
        method is never called at runtime. It is implemented to satisfy the
        ABC contract. Returns END on error so an unexpected call does not
        re-enter a processing node.
        """
        if state.get("status") == AgentStatus.ERROR.value:
            return END
        return "output_format"

    # -- Output shape ----------------------------------------------------------

    def get_output(self, state: AgentState) -> Dict[str, Any]:
        """Shape the output dict returned to the outer graph as sub_result.

        This dict is received by CustomsComplianceDocumentQAGraphNode.merge_output()
        in graph.py as the `sub_result` argument. Both methods are designed
        together to guarantee field-name consistency:

            Inner get_output()  emits: "formatted_answer", "citations",
                                       "checklist_items", "customs_domain",
                                       "currency_note", "status", ...
            Outer merge_output() reads: the same keys from sub_result

        Additional fields (intake_notes, trace_id, correlation_id,
        node_history) are surfaced for observability / downstream extension.
        """
        return {
            # the reason must leave the subgraph or the outer graph cannot report it
            "error_code": state.get("error_code"),
            "formatted_answer": state.get("formatted_answer"),
            "citations": state.get("citations"),
            "checklist_items": state.get("checklist_items"),
            "customs_domain": state.get("customs_domain"),
            "currency_note": state.get("currency_note"),
            "status": state.get("status"),
            "intake_notes": state.get("intake_notes"),
            "trace_id": state.get("trace_id"),
            "correlation_id": state.get("correlation_id"),
            "node_history": state.get("node_history", []),
        }
