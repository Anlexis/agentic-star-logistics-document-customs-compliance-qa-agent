# Test Specification — LOG-C2-039

**Template ID:** LOG-C2-039
**Template Name:** CustomsComplianceDocumentQAAgent
**Category:** Cat 2 (nested retrieval workflow)

This document describes the tests that ship with the template. Every file named below exists in
`tests/`, and the counts are what the suite collects.

The pipeline is deterministic and network-free: retrieval scores a bundled JSON corpus
(`config/kb/customs_kb.json`, 11 passages across 関税法 / NACCS / Incoterms paraphrase / HS-code /
AEO) or the passages the caller sends. There is no vector store, no model client and no external
connection anywhere in the suite or the code it exercises, so the whole suite runs offline and
deterministically.

## 1. Scope and conventions

| Area | Files |
|---|---|
| Per-node unit tests | `test_pre_process_node.py`, `test_input_validate_node.py`, `test_retrieve_node.py`, `test_rerank_filter_node.py`, `test_generate_answer_node.py`, `test_output_format_node.py`, `test_post_process_node.py` |
| Trust gate | `test_trust_gate.py` |
| Manifest / runtime config / corpus | `test_config_manifest.py` |
| Retrieval quality | `test_retrieval_quality.py` |
| Graph composition | `test_domain_workflow_graph.py`, `test_graph_composition.py` |
| Framework contract | `test_framework_compliance_tc06_tc07.py` |
| Refusal contract — a correctable refusal completes, an unfixable one terminates | `test_pre_process_node.py`, `test_graph_composition.py`, `test_invoke_e2e.py` |
| Boundary (PoB) | `test_import_isolation.py`, `test_state_safety.py`, `test_pb_invoke_order.py`, `test_pb7_hitl_interrupt_propagation.py`, `test_server_boot.py`, `test_invoke_e2e.py` |

**Two invocation styles, chosen deliberately.**

`node(state)` runs the full path — trust gate, then the framework's input scan, then `execute()`.
It is used for the accept/reject behaviour a caller actually experiences, and it is the only
correct way to test the trust gate itself.

`node.execute(state)` runs the node alone, with no framework wrapper in front. It is used for
every guarantee the template owns — the instruction-override refusal, the field bounds, the
identifier redaction. A test that only ever runs the wrapped path proves nothing about a
deployment where the framework scan is absent or configured off: the payload would reach the
answer path and the run would succeed. Running the node bare is what proves the template refuses
on its own.

**Trust levels in fixtures.** State builders set `caller_trust_level` to `VERIFIED_EXTERNAL` for
the two outer boundary slots (the level the manifest publishes) and `ANONYMOUS` for the five inner
domain nodes.

**Node signature.** Every node has the single signature `execute(self, state) -> dict`; a second
argument raises `TypeError` (asserted in `test_retrieve_node.py` and the `RerankFilterNode`
equivalent). Runtime tuning is exercised by seeding `state["retrieval_config"]`, still invoking
through `node(state)`.

**Audit muting.** `shared.*` is never stubbed out of `sys.modules` — the framework imports
`shared.security` at load time. The domain audit emitter is muted by an autouse fixture in
`tests/unit/conftest.py`; the audit assertion re-patches the same attribute with a spy and
asserts on the event payload.

## 2. What each area pins

### 2.1 PreProcessNode — the caller boundary · `test_pre_process_node.py` (100)

| Group | What it pins |
|---|---|
| Success | valid question accepted; `validated_context` is a JSON string; channel provenance |
| Correctable refusal | empty / whitespace / missing / non-string input, and an oversize question: each **completes** — `status` is SUCCESS, `error_log` carries the reason, and no `validated_input` is written. Asserting SUCCESS here is the point of the group: the run must end reportably rather than terminate, while still having done no work |
| Instruction-override screen | every attack form refused **by `execute()` directly**, with `status` ERROR and neither `validated_input` nor `validated_context` carried forward — and 12 real customs questions carrying the same verbs ("act as an importer of record", "override the classification", "ignore the previous declaration", "show me the rules", "select the correct heading", "delete a line", "drop shipments") all accepted at SUCCESS. Both directions, because a screen only ever tested with attacks looks perfect and quietly blocks real work |
| Caller passages | an override inside a caller-supplied passage **terminates** — `status` ERROR with no reason code set — even though it arrives on the same channel as the bounded options that complete; the error names the position and field |
| Identifier redaction | ISO 6346 container numbers and 11+ digit booking references redacted; e-mail redacted **by the node itself**; the same redaction proven on the caller-context channel; HS codes (6–10 bare digits) survive untouched; ordinary customs terminology round-trips |
| Field bounds | a parametrized non-finite matrix per numeric field (`"NaN"`, `"Infinity"`, `"-Infinity"`, `float('nan')`, `float('inf')`, `float('-inf')`) plus out-of-range and wrong-type values: all fail CLOSED and name the field in `error_log`, and all complete at SUCCESS rather than terminating; usable values accepted and present in `validated_context` |
| No echo | a credential-shaped rejected value never appears in the error |
| Caller passages, structure | entry cap (25), duplicate ids, and ten malformed entry shapes each complete at SUCCESS with the position named in `error_log`; control characters stripped from text that renders |
| Degradation | absent caller data leaves the baseline in place |
| Audit | `pre_process_complete` emitted with counts only, no caller text |

### 2.2 InputValidateNode · `test_input_validate_node.py` (34)

Plain-text and JSON-envelope parsing; whitespace normalisation and length cap; domain override
validation with no echo of the rejected value; the area classifier; and the envelope `top_k`
guard — unusable values (out of range, non-numeric, bool, non-finite) dropped so the configured
default applies, usable values kept. The non-finite cases are driven through raw JSON, because
`json.loads` accepts bare `NaN`/`Infinity` and `int()` on an infinite float raises `OverflowError`
that a plain `(TypeError, ValueError)` guard does not catch.

### 2.3 RetrieveNode · `test_retrieve_node.py` (11) and RerankFilterNode · `test_rerank_filter_node.py` (12)

Tokenisation and scoring; area narrowing; corpus selection (caller passages when supplied, the
bundled corpus otherwise); unreadable-corpus degradation; the area boost, relevance floor and
`top_k` cap; the caller's bounds-checked `score_threshold` replacing the configured floor;
JSON-string State discipline; and the single-argument `execute()` signature.

### 2.4 GenerateAnswerNode · `test_generate_answer_node.py` (9) and OutputFormatNode · `test_output_format_node.py` (8)

Checklist assembly with one point per ranked passage and a `[n]` marker each; cite-or-refuse with
zero ranked passages; the schedule annotation present on a cited answer and empty on the refusal;
and composition of the final answer with the standing disclaimer on every path through the node.

### 2.5 PostProcessNode — the output gate · `test_post_process_node.py` (9)

A clean answer passes through; credential-shaped content in the top-level answer blocks; and —
the case a shallow scan misses — a credential nested inside a citation entry or a checklist item
is caught by the recursive scan, with the structured fields emptied and status ERROR.

### 2.6 Trust gate · `test_trust_gate.py` (8)

Both directions at both boundary slots: an ANONYMOUS caller denied on `PreProcessNode` and on
`PostProcessNode` with `execute()` demonstrably not run, and a `VERIFIED_EXTERNAL` caller passing
both. Plus the class-level declarations: the outer slots require `VERIFIED_EXTERNAL`, the inner
nodes require `ANONYMOUS`.

### 2.7 Manifest and runtime config · `test_config_manifest.py` (18)

Identity and entry point resolve to the real class; the manifest is flat (a nested `agent:` block
is the retired shape and would load as an agent with no id); `requires.secrets` and
`requires.extras` are empty and match the code; the trust level matches the boundary nodes; every
node declares a trust level and no inner node demands more than the manifest publishes;
`max_retry` is within the framework ceiling and `timeout_s` uses the key the framework reads; the
entry point builds the agent WITH the runtime config; `_parent_config()` reads `config/config.yaml`
and not the manifest; and the bundled corpus is well-formed, uniquely keyed, within the five
areas, and carries no verbatim Incoterms rules text.

### 2.8 Composition · `test_domain_workflow_graph.py` (11), `test_graph_composition.py` (19)

Inner topology and the seven abstract members; `_extra_initial_state()` seeding; outer slot
registration; `add_edges()` not overridden; the `GraphNode` contract, whose `merge_output()` delta
is asserted whole — the mapped answer and structured fields plus the reason marker, blank on a run
that produced an answer; `_parent_config()` never forwarding an empty block even with an unreadable
config file; and end-to-end invoke through the compiled agent.

The terminating path is pinned on the compiled agent too: an ANONYMOUS caller is denied at the
`VERIFIED_EXTERNAL` `pre_process` slot, the main slot never delegates to the inner graph, and
`node_history` shows the run routed past `post_process` to `finalize` with no output produced. That
last assertion is what distinguishes terminating from completing — a run that reached
`post_process` would have composed a body.

### 2.9 Framework contract · `test_framework_compliance_tc06_tc07.py` (2)

Overriding the framework's default input or output gate raises at class definition — domain nodes
extend them only through the `_extra_security_gate_*` hooks.

## 3. Boundary tests

| ID | File | What it proves |
|---|---|---|
| PB-IMPORT | `test_import_isolation.py` (1) | template code does not import the platform SDK |
| PB-STATE | `test_state_safety.py` (1) | State is msgpack-safe — no bare dict/list in a checkpointed field |
| PB-6 | `test_pb_invoke_order.py` (6) | slot order on a real `invoke()` over the deployment sign-off payload: pre_process before main before post_process |
| PB-7 | `test_pb7_hitl_interrupt_propagation.py` (1) | skip stub — no graph class declares `propagate_hitl=True` and there is no cross-boundary interrupt |
| PB-BOOT | `test_server_boot.py` (4) | `import src.api.server` builds and compiles the agent without raising |
| PB-E2E | `test_invoke_e2e.py` (35) | see below |

### 3.1 End-to-end through `POST /invoke` · `test_invoke_e2e.py`

Driven through the app's real ASGI interface. There is no test client: `httpx` is only a
transitive dependency, so a hand-rolled ASGI call keeps this test dependency-free and unable to
silently skip.

| Group | What it proves |
|---|---|
| Auth boundary | a missing or wrong Bearer token is rejected with the same generic body |
| Real work from caller data | the caller's own passages are retrieved and cited **by id** — if the bundled corpus answered instead, the caller's data never crossed into the inner graph, which is the bridge failure this catches; the bundled corpus answers when nothing is sent; area narrowing and `top_k` take effect; an unanswerable question refuses instead of inventing an answer |
| Validation rejection | the full non-finite matrix through **raw JSON** for each numeric field, plus out-of-range, unknown area, and malformed passage shapes: every one fails closed and comes back with `"status": "success"` and no `citations` — the run reports the refusal in its own envelope instead of ending the turn on an exception. The counterpart is the terminating side: an instruction-override question comes back `"status": "error"` with neither `citations` nor `checklist_items`, while an ordinary question using the same words is answered. Every case here is driven through the real HTTP path and asserts a 200 response, so the two outcomes are distinguished by the envelope status, not by the transport |
| Output contract | the disclaimer on every answered path including the no-coverage refusal; cited answers list their sources; a credential reaching the rendered answer through a caller passage returns `"status": "error"` with the structured fields absent and the secret nowhere in the body |
| Identifiers | this domain's identifier forms come out byte-identical: `6109.10`, `8471300000`, `MSKU 4512345`, `UN 1263`, `REF-2026-0041`, `NRT-LAX`, `CIF`, `17:00`, `30 kg`, `1200pcs`, and `4512345` — a pure-numeric reference with no letters to protect it. A numbered heading following a three-letter code keeps its number. These pin the deliberate absence of a numeric rounding grid (see `docs/02_design.md`, "On numeric rounding grids"); they were confirmed to fail when such a grid is applied to this output |

## 4. Running the suite

```bash
pip install -e ".[dev]"
python -m pytest tests/ -v
```

No platform connection, network access or credentials are needed.
