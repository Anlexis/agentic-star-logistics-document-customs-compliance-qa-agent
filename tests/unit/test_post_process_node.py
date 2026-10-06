# LOG-C2-039 — Unit Tests: PostProcessNode (outer post_process slot; output gate)
#
# Invocation canon: node(state) via BaseNode.__call__. PostProcessNode is the
# second outer S-gate slot and requires VERIFIED_EXTERNAL (like PreProcessNode),
# so its behavioural tests build the state at that level; the ANONYMOUS
# rejection lives in test_trust_gate.py.
#
# Gate layering: the node's own module-level _security_gate_output() scan runs
# INSIDE execute() and RECURSES into nested dict/list/tuple structures — not
# just the top-level `result` string. It replaces a violating answer with the sanitised stub
# (returned dict — no exception) and empties every structured field
# (citations/checklist_items/customs_domain/currency_note). The framework's
# OWN post-execute framework credential scan then sees only the clean
# stub, so it never raises for these cases. Intentional-credential tests
# assert the raw secret never survives into formatted_output, result, OR any
# nested structured field.
#
# Mirrors docs/03_test_spec.md section 2.7 (POST-01..POST-08).
# Deterministic — no LLM, no network. framework.* / src.* imports only.

from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel

from src.nodes.post_process_node import PostProcessNode
from src.schemas.state import from_json, to_json

_CLEAN_REPORT = (
    "# Customs Compliance Checklist\n\n"
    "[ ] [1] hs code classification: essential character controls classification.\n"
)

# JWT-shaped token built at runtime so no credential-shaped literal ever sits
# in the repository (credential-scan hygiene).
_FAKE_JWT = "eyJ" + "a" * 12 + "." + "b" * 12 + "." + "c" * 12


def _make_state(result_text, **extra) -> dict:
    state = {
        "result": result_text,
        "caller_trust_level": TrustLevel.VERIFIED_EXTERNAL.value,
        "node_history": [],
        "error_log": [],
        "session_id": "unit-session",
        "execution_time": {},
    }
    state.update(extra)
    return state


class TestPostProcessClean:
    def test_post_01_clean_output_passes_through(self):
        result = PostProcessNode()(_make_state(_CLEAN_REPORT))
        assert result["status"] == AgentStatus.SUCCESS.value
        # Regression guard: State carries the plain string, never the enum.
        # AgentStatus subclasses str, so a bare isinstance(..., str) would pass
        # for the enum member as well and prove nothing.
        assert isinstance(result["status"], str)
        assert not isinstance(result["status"], AgentStatus)
        assert result["formatted_output"] == _CLEAN_REPORT

    def test_post_02_empty_result_is_non_fatal(self):
        result = PostProcessNode()(_make_state(""))
        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["formatted_output"] == ""


class TestPostProcessS3Gate:
    def _assert_blocked(self, result, secret):
        assert result["status"] == AgentStatus.ERROR.value
        assert any("output blocked" in str(e) for e in result["error_log"])
        # The raw secret must not survive into either surfaced field.
        assert secret not in str(result.get("formatted_output", ""))
        assert secret not in str(result.get("result", ""))
        assert "[OUTPUT BLOCKED by the output gate" in result["formatted_output"]

    def test_post_03_api_key_is_blocked(self):
        secret = "sk-ABCDEF0123456789abcdef"
        result = PostProcessNode()(_make_state(f"# Report\n\n<!-- debug api_key={secret} -->\n"))
        self._assert_blocked(result, secret)

    def test_post_04_credential_assignment_is_blocked(self):
        secret = "password=super_secret_value_123"
        result = PostProcessNode()(_make_state(f"# Report\n\ninternal note: {secret}\n"))
        self._assert_blocked(result, "super_secret_value_123")

    def test_post_05_jwt_is_blocked(self):
        result = PostProcessNode()(_make_state(f"# Report\n\nsession token {_FAKE_JWT}\n"))
        self._assert_blocked(result, _FAKE_JWT)

    def test_post_06_bearer_token_is_blocked(self):
        secret = "Bearer abcdefghijklmnopqrstuvwxyz0123456789"
        result = PostProcessNode()(_make_state(f"# Report\n\nauthorization: {secret}\n"))
        self._assert_blocked(result, secret)


class TestPostProcessRecursiveGate:
    """The gate scan is RECURSIVE: a credential-shaped string nested inside
    a structured field must be caught even when the top-level `result`
    string is completely clean."""

    def test_post_07_credential_nested_in_citations_is_caught_and_blocked(self):
        # result itself is clean; the violation is nested two levels deep
        # inside citations[0]["source"] — only a recursive scan finds it.
        secret = "sk-" + "A" * 24
        citations = to_json([{"ref": 1, "id": "kb-001", "title": "hs code classification", "source": secret}])
        state = _make_state(
            _CLEAN_REPORT,
            citations=citations,
            checklist_items=to_json(["hs code classification: essential character controls."]),
            customs_domain="hs_code",
            currency_note="verify against the current schedule.",
        )
        result = PostProcessNode()(state)
        assert result["status"] == AgentStatus.ERROR.value
        assert any("output blocked" in str(e) for e in result["error_log"])
        # Fail-closed: the secret is gone from EVERY surfaced field, and every
        # structured field is emptied — not just the top-level result/output.
        assert secret not in str(result.get("formatted_output", ""))
        assert secret not in str(result.get("result", ""))
        assert secret not in str(result.get("citations", ""))
        assert from_json(result.get("citations")) == []
        assert from_json(result.get("checklist_items")) == []
        assert result.get("customs_domain") == ""
        assert result.get("currency_note") == ""

    def test_post_08_credential_nested_in_checklist_items_is_caught(self):
        secret = "Bearer " + "b" * 20
        state = _make_state(
            _CLEAN_REPORT,
            citations=to_json([]),
            checklist_items=to_json([f"hs code classification: {secret} embedded mid-sentence."]),
        )
        result = PostProcessNode()(state)
        assert result["status"] == AgentStatus.ERROR.value
        assert secret not in str(result.get("formatted_output", ""))
        assert from_json(result.get("checklist_items")) == []

    def test_clean_structured_payload_passes_through_untouched(self):
        # Sanity counterpart: a genuinely clean structured payload is NOT
        # false-flagged by the recursive scan.
        citations = to_json(
            [
                {
                    "ref": 1,
                    "id": "kb-001",
                    "title": "hs code classification",
                    "source": "General Rules for the Interpretation of the HS",
                }
            ]
        )
        state = _make_state(
            _CLEAN_REPORT,
            citations=citations,
            checklist_items=to_json(["hs code classification: essential character controls."]),
            customs_domain="hs_code",
            currency_note="verify against the current schedule.",
        )
        result = PostProcessNode()(state)
        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["formatted_output"] == _CLEAN_REPORT
