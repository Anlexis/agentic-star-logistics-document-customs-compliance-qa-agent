# LOG-C2-039 — Unit Tests: manifest / runtime-config consistency
#
# Two files, two jobs, and the tests below pin both against the code:
#
#   config/agent.yaml   the static manifest — identity, entry point, the trust
#                       level the caller must clear. Flat: every key at root.
#   config/config.yaml  live runtime parameters — max_retry, timeout_s, and the
#                       retrieval/llm blocks the outer graph forwards into the
#                       inner graph.
#
# The distinction matters because a reader pointed at the wrong file gets {}
# and silently falls back to defaults, so declared values look configured while
# doing nothing. These tests fail fast on that drift.
#
# Deterministic — no model, no network.

import json
import pathlib

import yaml

from framework.schemas.trust_level import TrustLevel

from src.graph.graph import CustomsComplianceDocumentQAAgent, CustomsComplianceDocumentQAGraphNode
from src.nodes.post_process_node import PostProcessNode
from src.nodes.pre_process_node import PreProcessNode

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_MANIFEST = yaml.safe_load((_ROOT / "config" / "agent.yaml").read_text(encoding="utf-8"))
_RUNTIME = yaml.safe_load((_ROOT / "config" / "config.yaml").read_text(encoding="utf-8"))


class TestManifestIdentity:
    def test_cfg_01_template_id_matches_the_repo(self):
        assert _MANIFEST["id"] == "LOG-C2-039"

    def test_cfg_02_declared_class_is_the_graph_class(self):
        # The manifest entry point is a single dotted import path and must
        # resolve to the class server.py imports.
        assert _MANIFEST["class"] == ("src.graph.graph." + CustomsComplianceDocumentQAAgent.__name__)
        assert _MANIFEST["name"] == CustomsComplianceDocumentQAAgent().name

    def test_cfg_03_category_and_industry(self):
        assert _MANIFEST["category"] == "Cat 2"
        assert _MANIFEST["industry"] == "LOG"
        assert _MANIFEST["base_type"] == "RAGAgent"

    def test_manifest_is_flat(self):
        # A nested `agent:` block is the retired shape; the registry reads root
        # keys, so a nested manifest loads as an agent with no id at all.
        assert "agent" not in _MANIFEST

    def test_declared_requirements_match_the_code(self):
        # This template invokes no model and requires no secret. Declaring
        # either would make the agent fail to start on a runtime that has not
        # provisioned it.
        assert _MANIFEST["requires"]["secrets"] == []
        assert _MANIFEST["requires"]["extras"] == []
        assert _MANIFEST["generation_mode"] == "deterministic"


class TestManifestSecurity:
    def test_cfg_04_required_trust_level_matches_outer_gate_nodes(self):
        declared = TrustLevel(_MANIFEST["required_trust_level"])
        assert declared is TrustLevel.VERIFIED_EXTERNAL
        assert PreProcessNode.required_trust_level is declared
        assert PostProcessNode.required_trust_level is declared

    def test_every_node_declares_a_trust_level(self):
        # A node class with no required_trust_level raises at class-definition
        # time on the installed framework; pin the whole set so a new node
        # cannot be added without one.
        from src.nodes.generate_answer_node import GenerateAnswerNode
        from src.nodes.input_validate_node import InputValidateNode
        from src.nodes.output_format_node import OutputFormatNode
        from src.nodes.rerank_filter_node import RerankFilterNode
        from src.nodes.retrieve_node import RetrieveNode

        inner_nodes = [
            InputValidateNode,
            RetrieveNode,
            RerankFilterNode,
            GenerateAnswerNode,
            OutputFormatNode,
        ]
        for node in inner_nodes:
            assert isinstance(node.required_trust_level, TrustLevel)
            # The inner nodes must not demand MORE than the manifest publishes,
            # or an external caller the manifest admits is denied mid-pipeline.
            assert node.required_trust_level is TrustLevel.ANONYMOUS


class TestRuntimeConfig:
    def test_cfg_05_max_retry_within_framework_ceiling(self):
        max_retry = _RUNTIME["max_retry"]
        assert isinstance(max_retry, int)
        assert 0 <= max_retry < 10  # the framework's retry ceiling

    def test_timeout_uses_the_key_the_framework_reads(self):
        assert "timeout_s" in _RUNTIME
        assert "timeout_seconds" not in _RUNTIME

    def test_hitl_is_not_enabled(self):
        # No interrupt gate in this template, so no checkpointer is required.
        assert (_RUNTIME.get("hitl") or {}).get("enabled", False) is False

    def test_the_server_builds_the_agent_with_the_runtime_config(self):
        # An agent constructed with no config silently drops every value in
        # config.yaml; the entry point must load it.
        from src.api.server import _load_runtime_config

        loaded = _load_runtime_config()
        assert loaded["max_retry"] == _RUNTIME["max_retry"]
        assert loaded["timeout_s"] == _RUNTIME["timeout_s"]


class TestRetrievalBlock:
    def test_cfg_06_retrieval_block_matches_node_defaults(self):
        # Node module defaults mirror the runtime config — a drift silently
        # changes tuning.
        retrieval = _RUNTIME["retrieval"]
        from src.nodes.rerank_filter_node import _DEFAULT_RETRIEVAL as rerank_defaults
        from src.nodes.retrieve_node import _DEFAULT_RETRIEVAL as retrieve_defaults

        assert retrieval["top_k"] == retrieve_defaults["top_k"] == rerank_defaults["top_k"]
        assert (
            retrieval["score_threshold"] == retrieve_defaults["score_threshold"] == rerank_defaults["score_threshold"]
        )
        assert retrieval["kb_path"] == retrieve_defaults["kb_path"]
        assert (_ROOT / retrieval["kb_path"]).is_file()

    def test_cfg_07_parent_config_forwards_the_runtime_blocks(self):
        cfg = CustomsComplianceDocumentQAGraphNode()._parent_config()
        assert cfg["configurable"]["retrieval"] == _RUNTIME["retrieval"]
        assert cfg["configurable"]["llm"] == _RUNTIME["llm"]
        assert cfg["configurable"]["retrieval"], "_parent_config() must never forward an empty retrieval block"

    def test_parent_config_reads_the_runtime_file_not_the_manifest(self):
        # Regression guard for the migration: the manifest carries no runtime
        # block, so a reader still pointed at it returns {} and every declared
        # value goes dead behind the module fallbacks.
        from src.graph import graph as graph_module

        assert graph_module._RUNTIME_CONFIG_PATH.name == "config.yaml"
        assert "retrieval" not in _MANIFEST and "llm" not in _MANIFEST


class TestSeededKnowledgeBase:
    def _entries(self):
        return json.loads((_ROOT / _RUNTIME["retrieval"]["kb_path"]).read_text(encoding="utf-8"))

    def test_kb_is_a_well_formed_entry_list(self):
        entries = self._entries()
        assert isinstance(entries, list)
        assert len(entries) >= 5, "the seeded corpus must carry a usable set of passages"
        for entry in entries:
            assert set(entry.keys()) == {"id", "title", "category", "source", "tags", "content"}
            assert entry["id"] and entry["title"] and entry["content"]

    def test_kb_ids_are_unique(self):
        ids = [e["id"] for e in self._entries()]
        assert len(ids) == len(set(ids))

    def test_kb_categories_are_the_five_declared_domains(self):
        categories = {e["category"] for e in self._entries()}
        assert categories <= {"hs_code", "naccs", "incoterms", "aeo", "customs_law"}

    def test_kb_does_not_reproduce_verbatim_incoterms_text(self):
        # The Incoterms 2020 rules text is copyrighted: this corpus carries
        # paraphrase and orientation only, never the rules wording.
        incoterms_entries = [e for e in self._entries() if e["category"] == "incoterms"]
        assert incoterms_entries, "expected at least one incoterms passage"
        for entry in incoterms_entries:
            blob = (entry["source"] + " " + entry["content"]).lower()
            assert "paraphrase" in blob or "not the icc rules text" in blob or "orientation" in blob
