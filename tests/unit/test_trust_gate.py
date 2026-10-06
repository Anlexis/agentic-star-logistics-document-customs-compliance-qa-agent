# LOG-C2-039 — Unit Tests: the caller trust gate
#
# Trust is decided ONCE, at the two outer boundary slots (PreProcessNode and
# PostProcessNode), which both require VERIFIED_EXTERNAL — the level the
# manifest publishes. The five inner domain nodes require ANONYMOUS, which any
# caller satisfies. That split is deliberate: an inner node demanding MORE than
# the manifest publishes would admit a caller at the door and then deny them
# mid-pipeline, and the failure would only show up on a real external call.
# Both directions are pinned below — the denial AND the pass.
#
# Every invocation goes through node(state) — BaseNode.__call__, which runs the
# trust gate before execute() — never node.execute(state) directly, which would
# bypass the gate this file exists to test. A denial RETURNS an error dict
# (never raises) with status ERROR; execute() never runs, so execute-only
# output keys are ABSENT from the returned dict.

from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel

from src.nodes.generate_answer_node import GenerateAnswerNode
from src.nodes.input_validate_node import InputValidateNode
from src.nodes.output_format_node import OutputFormatNode
from src.nodes.post_process_node import PostProcessNode
from src.nodes.pre_process_node import PreProcessNode
from src.nodes.rerank_filter_node import RerankFilterNode
from src.nodes.retrieve_node import RetrieveNode


def _make_state(
    trust_value: str,
    user_input: str = "what hs code applies to this shipment of consumer electronics with embedded ai chips?",
    **extra,
) -> dict:
    state = {
        "user_input": user_input,
        "caller_trust_level": trust_value,
        "node_history": [],
        "error_log": [],
        "session_id": "test-session",
        "execution_time": {},
    }
    state.update(extra)
    return state


class TestTrustGate:
    """All invocations go through node(state) / __call__."""

    def test_anonymous_caller_allowed_on_inner_node(self):
        """: an ANONYMOUS caller passes an ANONYMOUS inner domain node."""
        node = InputValidateNode()  # required_trust_level = ANONYMOUS
        result = node(_make_state(TrustLevel.ANONYMOUS.value, validated_input="hs code question"))
        assert "trust gate denied" not in str(result.get("error_log", []))
        assert result.get("search_query") is not None, "inner node should produce a normalised query"

    def test_anonymous_caller_denied_on_pre_process(self):
        """rejection: ANONYMOUS caller on the VERIFIED_EXTERNAL PreProcessNode.

        __call__ must RETURN an error dict (never raise) with status ERROR and
        'trust gate denied' in the error_log. execute() never ran, so the
        execute-only output key (validated_input) must be ABSENT.
        """
        node = PreProcessNode()  # required_trust_level = VERIFIED_EXTERNAL
        result = node(_make_state(TrustLevel.ANONYMOUS.value))
        assert result.get("status") == AgentStatus.ERROR.value
        error_log = result.get("error_log", [])
        assert any(
            "trust gate denied" in str(e) for e in error_log
        ), f"Expected 'trust gate denied' in error_log, got: {error_log}"
        assert "validated_input" not in result, "execute() must not run on a trust denial — validated_input leaked"

    def test_verified_external_caller_passes_pre_process(self):
        """: a VERIFIED_EXTERNAL caller clears the pre_process gate and
        the node writes the identifier-screened validated_input."""
        node = PreProcessNode()
        result = node(_make_state(TrustLevel.VERIFIED_EXTERNAL.value))
        assert result.get("status") == AgentStatus.SUCCESS.value
        assert result.get("validated_input")

    def test_pre_process_empty_input_rejected_after_gate(self):
        """passes, then the node's own input validation rejects empty input."""
        node = PreProcessNode()
        result = node(_make_state(TrustLevel.VERIFIED_EXTERNAL.value, user_input=""))
        assert result.get("status") == AgentStatus.SUCCESS.value
        assert any("empty" in str(e) for e in result.get("error_log", []))

    def test_anonymous_caller_denied_on_post_process(self):
        """rejection on the other VERIFIED_EXTERNAL outer slot (post_process).

        The denial dict carries no execute-only key (formatted_output ABSENT).
        """
        node = PostProcessNode()  # required_trust_level = VERIFIED_EXTERNAL
        result = node(
            _make_state(
                TrustLevel.ANONYMOUS.value,
                result="a clean customs compliance checklist answer",
            )
        )
        assert result.get("status") == AgentStatus.ERROR.value
        assert any("trust gate denied" in str(e) for e in result.get("error_log", []))
        assert "formatted_output" not in result, "execute() must not run on a trust denial — formatted_output leaked"

    def test_verified_external_caller_passes_post_process(self):
        """: a VERIFIED_EXTERNAL caller clears the post_process gate."""
        node = PostProcessNode()
        result = node(
            _make_state(
                TrustLevel.VERIFIED_EXTERNAL.value,
                result="a clean customs compliance checklist answer",
            )
        )
        assert result.get("status") == AgentStatus.SUCCESS.value
        assert result.get("formatted_output")


class TestTrustLevelMatrix:
    """The template's declared trust matrix (docs/02_design.md Security Gates).

    Outer S-gate slots require VERIFIED_EXTERNAL (the manifest's
    required_trust_level); the five inner domain nodes run behind that outer
    boundary and are declared ANONYMOUS per the Cat-2 nested convention.
    """

    def test_outer_gate_nodes_require_verified_external(self):
        assert PreProcessNode.required_trust_level is TrustLevel.VERIFIED_EXTERNAL
        assert PostProcessNode.required_trust_level is TrustLevel.VERIFIED_EXTERNAL

    def test_inner_domain_nodes_admit_anonymous(self):
        for node_cls in (
            InputValidateNode,
            RetrieveNode,
            RerankFilterNode,
            GenerateAnswerNode,
            OutputFormatNode,
        ):
            assert node_cls.required_trust_level is TrustLevel.ANONYMOUS, (
                f"{node_cls.__name__} must declare TrustLevel.ANONYMOUS " "(inner Cat-2 domain node)"
            )
