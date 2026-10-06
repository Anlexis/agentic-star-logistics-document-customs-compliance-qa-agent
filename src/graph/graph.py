"""AgentCore Platform v1.0"""

# LOG-C2-039 - Outer graph (AgentBaseGraph; Cat 2 two-layer nested architecture)
#
# Customs Compliance Document Q&A Agent (Cat 2 RAG domain workflow).
#
# Architecture (Cat 2):
#
#   Outer backbone (fixed - identical to Cat 1, do NOT override add_edges()):
#     START -> initialize -> pre_process -> main -> {route} -> post_process -> finalize -> END
#                                             |  (RETRY, max 3)
#                                             -> pre_process
#
#   `main` slot is a GraphNode subclass (CustomsComplianceDocumentQAGraphNode)
#   that delegates the full customs-compliance Q&A domain workflow to
#   DomainWorkflowGraph (inner BaseGraph: input_validate -> retrieve ->
#   rerank_filter -> generate_answer -> output_format).
#
#   Domain complexity is fully encapsulated inside the inner graph. The outer
#   backbone is never modified.
#
# Directory layout:
#   src/graph/graph.py                 <- outer graph (this file)
#   src/graph/domain_workflow_graph.py <- inner graph (multi-step topology)
#
# Class-name contract — these three must agree or the agent will not load:
#   graph.py class:           CustomsComplianceDocumentQAAgent (this file)
#   config/agent.yaml class:  "src.graph.graph.CustomsComplianceDocumentQAAgent"
#   src/api/server.py import: from src.graph.graph import CustomsComplianceDocumentQAAgent
#
# Rules enforced:
#   - CustomsComplianceDocumentQAAgent inherits AgentBaseGraph directly
#   - super().register_nodes() called first (fills initialize + finalize)
#   - CustomsComplianceDocumentQAGraphNode assigned to self._nodes["main"]
#   - _parent_config() forwards the runtime retrieval/llm keys (never {})
#   - merge_output() returns only changed keys
#   - add_edges() NOT overridden on the outer graph
#   - get_output() overridden to surface the structured fields (checklist /
#     citations / domain / currency) on SUCCESS only — fail-closed, mirroring
#     the output gate

from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Dict

from framework.graph.agent_base_graph import AgentBaseGraph
from framework.nodes.graph_node import GraphNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from src.graph.context_bridge import set_caller_context
from src.nodes.post_process_node import PostProcessNode
from src.nodes.pre_process_node import PreProcessNode
from src.schemas.state import State, from_json

if TYPE_CHECKING:  # import cycle at runtime; the checker needs the name
    from src.graph.domain_workflow_graph import DomainWorkflowGraph

# Runtime config path: src/graph/graph.py -> parents[2] = repo root.
# config/agent.yaml is the static manifest and carries no runtime block; every
# tunable value lives in config/config.yaml, which is what this file reads.
_RUNTIME_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "config.yaml"

# Fallbacks mirror the `retrieval` / `llm` blocks in config/config.yaml so
# _parent_config() never forwards an empty config even if that file is
# unreadable in an exotic deployment layout.
_FALLBACK_RETRIEVAL = {
    "top_k": 4,
    "score_threshold": 0.25,
    "kb_path": "config/kb/customs_kb.json",
}
_FALLBACK_LLM = {
    "temperature": 0.0,
    "max_tokens": 1500,
}


class CustomsComplianceDocumentQAGraphNode(GraphNode):
    """GraphNode subclass assigned to the `main` slot of the outer agent.

    Wraps DomainWorkflowGraph (inner Cat 2 BaseGraph customs Q&A pipeline).
    Called by AgentBaseGraph backbone after pre_process and before post_process.

    Contracts:
      get_subgraph()    - instantiate DomainWorkflowGraph with the forwarded
                          manifest config (_parent_config())
      extract_input()   - pull validated_input (shipment-identifier redacted)
                          from outer state and stash the validated caller
                          context for the inner graph
      merge_output()    - map sub_result fields into outer state delta
                          (changed keys only)
      error_strategy    - "propagate": re-raise inner errors as SubgraphError
                          (fail-fast)
    """

    # "propagate": re-raise inner graph exceptions as SubgraphError (default - fail fast).
    error_strategy: ClassVar[str] = "propagate"

    # False: interrupts are handled inside the inner graph only — this
    # template has no human-in-the-loop gate.
    propagate_hitl: ClassVar[bool] = False

    def _parent_config(self) -> Dict[str, Any]:
        """Forward the runtime `retrieval` + `llm` blocks to the inner graph.

        Loads config/config.yaml and returns both tuning blocks under
        config["configurable"] — never an empty dict. The inner graph
        republishes the `retrieval` block into inner state
        (DomainWorkflowGraph._extra_initial_state()) so RetrieveNode /
        RerankFilterNode read live top_k / score_threshold values rather than
        declarations nothing reads. The `llm` block is forwarded verbatim for
        the documented synthesis upgrade (unused by the deterministic nodes).
        """
        runtime: Dict[str, Any] = {}
        try:
            import yaml

            loaded = yaml.safe_load(_RUNTIME_CONFIG_PATH.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                runtime = loaded
        except Exception:
            runtime = {}
        retrieval = runtime.get("retrieval")
        if not isinstance(retrieval, dict) or not retrieval:
            retrieval = dict(_FALLBACK_RETRIEVAL)
        llm = runtime.get("llm")
        if not isinstance(llm, dict) or not llm:
            llm = dict(_FALLBACK_LLM)
        return {"configurable": {"retrieval": retrieval, "llm": llm}}

    def get_subgraph(self) -> "DomainWorkflowGraph":
        """Instantiate and return the inner domain workflow graph.

        DomainWorkflowGraph is imported lazily (inside the method) to avoid
        circular-import risk at module load time.

        The inner graph receives the runtime config via its BaseGraph
        constructor; its domain NODES still take no constructor arguments and
        read config purely from State — execute(self, state) -> dict takes no
        config parameter.
        """
        from src.graph.domain_workflow_graph import DomainWorkflowGraph

        return DomainWorkflowGraph(config=self._parent_config())

    def execute(self, state: AgentState) -> dict[str, Any]:
        """Skip the inner graph when the request was already found unacceptable.

        A request declined by pre_process has no validated input to act on, so
        running the inner graph would only produce a second, vaguer reason for
        the same rejection - and overwrite the specific one already settled.
        """
        marker = state.get("error_code")
        if marker:
            return {"status": AgentStatus.SUCCESS.value, "error_code": marker}
        result: dict[str, Any] = super().execute(state)
        return result

    def extract_input(self, state: AgentState) -> str:
        """Return the string input passed into inner_graph.invoke().

        PreProcessNode validates the request, redacts shipment identifiers and
        writes the result to validated_input. Prefer that; fall back to
        user_input if validated_input is absent (e.g. in unit tests).

        This is also the last hook that runs before the framework invokes the
        inner graph, and the framework does not forward input_context across
        that boundary — so the validated caller options are stashed here for
        DomainWorkflowGraph._extra_initial_state() to pick up. Only the
        VALIDATED form is carried: raw caller data never crosses into the
        inner graph.
        """
        set_caller_context(state.get("validated_context"))
        return str(state.get("validated_input") or state.get("user_input", ""))

    def merge_output(self, state: AgentState, sub_result: Dict[str, Any]) -> Dict[str, Any]:
        """Map inner graph sub_result back into the outer state delta.

        sub_result is the dict returned by DomainWorkflowGraph.get_output().
        Returns ONLY changed keys - never the full state.

        Key coupling (designed together with DomainWorkflowGraph.get_output()):
          Inner get_output() emits  -> "formatted_answer", "citations",
                                       "checklist_items", "customs_domain",
                                       "currency_note", "status", ...
          This merge_output() reads -> the same keys from sub_result

        customs_answer (str | None): final rendered customs answer; written
          by OutputFormatNode inside the inner graph.
        result: PostProcessNode (outer post_process slot) reads
          state.get("result") - the inner graph emits the rendered answer
          under "formatted_answer", so map it to "result" as well;
          otherwise the final output surfaced by PostProcessNode (and its
          output gate) is always empty.
        citations / checklist_items / customs_domain / currency_note: the
          structured fields CustomsComplianceDocumentQAAgent.get_output()
          surfaces on SUCCESS.
        status (str | None): terminal AgentStatus value from the inner graph run.
        """
        return {
            # Outer reason wins: a reason settled before the inner run is the real
            # one, and a plain sub_result.get() would erase it.
            "error_code": state.get("error_code") or sub_result.get("error_code", ""),
            "customs_answer": sub_result.get("formatted_answer"),
            "result": sub_result.get("formatted_answer"),
            "citations": sub_result.get("citations"),
            "checklist_items": sub_result.get("checklist_items"),
            "customs_domain": sub_result.get("customs_domain"),
            "currency_note": sub_result.get("currency_note"),
            "status": sub_result.get("status"),
        }


class CustomsComplianceDocumentQAAgent(AgentBaseGraph):
    """Outer graph for LOG-C2-039 (Cat 2 RAG).

    Inherits AgentBaseGraph directly. Domain logic is fully encapsulated in
    CustomsComplianceDocumentQAGraphNode (main slot), which delegates to
    DomainWorkflowGraph (inner BaseGraph).

    Backbone (fixed - identical to Cat 1):
        START -> initialize -> pre_process -> main -> post_process -> finalize -> END

    register_nodes() is the ONLY structural override:
      - super().register_nodes() fills: initialize, finalize (framework defaults)
      - pre_process:  PreProcessNode (caller contract: refusal screen, field
                      bounds, shipment-identifier redaction)
      - main:         CustomsComplianceDocumentQAGraphNode (delegates to
                      DomainWorkflowGraph)
      - post_process: PostProcessNode (recursive output gate)

    add_edges() is NOT overridden - backbone wiring belongs to the framework.

    get_output() IS overridden: this template's product is structured — a
    checklist-format answer plus citations, domain routing and a currency
    annotation. It EXTENDS super().get_output() (never replaces —
    status/node_history/trace_id stay intact) and adds the structured keys ONLY
    when status == SUCCESS. That is the fail-closed half of the output gate:
    PostProcessNode has already scanned and, on a violation, emptied
    citations/checklist_items/customs_domain/currency_note and set status
    ERROR, so a violation surfaces only the sanitised `output` string — never
    the structured fields.
    """

    @property
    def name(self) -> str:
        """Agent identifier registered with AgentRegistry."""
        return "CustomsComplianceDocumentQAAgent"

    @property
    def state_schema(self) -> type:
        return State

    def register_nodes(self) -> None:
        """Fill all 5 backbone slots.

        super().register_nodes() MUST be called first - it injects the
        framework's default InitializeNode (sets schema_version, session_id,
        trust_level) and FinalizeNode (builds response_metadata, total_time_ms).
        """
        super().register_nodes()  # fills: initialize, finalize

        self._nodes["pre_process"] = PreProcessNode()
        self._nodes["main"] = CustomsComplianceDocumentQAGraphNode()
        self._nodes["post_process"] = PostProcessNode()

    # add_edges() is NOT overridden - backbone wiring belongs to the framework.

    def get_output(self, state: AgentState) -> Dict[str, Any]:
        """Extend the base envelope with the structured customs-answer fields.

        Surfaces citations / checklist_items / customs_domain / currency_note
        ONLY when status == AgentStatus.SUCCESS.value — fail-closed, mirroring
        the gate outcome in PostProcessNode. On ERROR (including a blocked
        output), only the base envelope (output/status/trace_id/
        correlation_id/node_history) is returned — the sanitised stub already
        lives in `output`; nothing structured leaks alongside it.
        """
        base: Dict[str, Any] = super().get_output(state)
        # A run that completed WITHOUT carrying out the request holds the
        # sentence saying what to correct, not a product: none of the
        # structured fields below were produced, so none is released.
        if state.get("error_code"):
            return base
        if state.get("status") == AgentStatus.SUCCESS.value:
            base["citations"] = from_json(state.get("citations"), [])
            base["checklist_items"] = from_json(state.get("checklist_items"), [])
            base["customs_domain"] = state.get("customs_domain")
            base["currency_note"] = state.get("currency_note")
        return base


# Back-compat alias — some loaders look for a module-level `Graph`. Keep both
# names pointing at the same agent.
Graph = CustomsComplianceDocumentQAAgent
