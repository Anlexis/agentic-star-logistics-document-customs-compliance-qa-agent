# Template Design Specification — LOG-C2-039

**Template ID:** LOG-C2-039
**Template Name:** CustomsComplianceDocumentQAAgent
**Category:** Cat 2 (multi-step domain workflow — retrieval-augmented Q&A)
**Industry:** LOG

## Position in the framework

| Aspect | Value |
|---|---|
| Agent class | `CustomsComplianceDocumentQAAgent` (alias `Graph`) |
| L1 Base (framework base class) | `AgentBaseGraph` — direct framework inheritance |
| Inner graph base | `BaseGraph` — `DomainWorkflowGraph` |
| Pattern | Cat 2 two-layer nested: the fixed five-slot outer backbone, with a `GraphNode` in the `main` slot wrapping an inner `BaseGraph` domain workflow |

Three-layer separation:

- **State** — a flat `TypedDict`. Never Pydantic: checkpoints are msgpack-serialised and
  model objects corrupt silently. Structured fields travel as JSON strings via `to_json()` /
  `from_json()`.
- **Node** — `FunctionNode` subclasses overriding `execute(self, state) -> dict` only. There is
  no `config` parameter on `execute()`; configuration reaches a node through State.
- **Graph** — composition via `register_nodes()`. The outer `add_edges()` is not overridden.

## Purpose

Retrieval-augmented Q&A over a customs-compliance knowledge base spanning five areas —
関税法 (Customs Act) summaries, NACCS declaration guidance, Incoterms 2020 (paraphrase and
orientation, never the copyrighted ICC rules text), HS-code reference, and AEO certification
guidance. Logistics operators, customs brokers (通関業者), and import/export compliance staff ask
a natural-language customs question and receive a cited, checklist-format answer routed by
customs area and annotated against the current tariff schedule.

The pipeline is deterministic end to end — keyword retrieval and rule-based checklist assembly,
no model call and no network — so the same question over the same corpus always produces the same
answer, and an answer can be audited after the fact.

## Configuration: two files, two jobs

| File | Contents | Read by |
|---|---|---|
| `config/agent.yaml` | The static manifest, flat at root: id, name, namespace, entry-point class, `required_trust_level`, and the `requires` block declaring which secrets and extras the agent needs at compile time. | The registry, at load |
| `config/config.yaml` | Live runtime parameters: `max_retry`, `timeout_s`, and the `retrieval` and `llm` blocks. | `_parent_config()` and `src/api/server.py` |

The split matters operationally. A reader pointed at the wrong file gets `{}` and falls back to
module defaults, so every value in the file it should have read looks configured while doing
nothing at all. `config/agent.yaml` carries no runtime block; anything tunable lives in
`config/config.yaml`, and both readers — the graph node and the HTTP entry point — load that file.
`test_config_manifest.py` pins this, including that the entry point constructs the agent WITH the
runtime config rather than with no arguments.

`requires.secrets` and `requires.extras` are both empty, which is a claim about the code: this
template constructs no model client and calls no `secrets.require()`. Declaring either would make
the agent fail to start on a runtime that has not provisioned it.

## Architecture

### Outer backbone (`AgentBaseGraph`)

```
START → initialize → pre_process → main → {route} → post_process → finalize → END
                                     ↓ (retry, max_retry from config/config.yaml)
                                   pre_process
```

| Slot | Class | Responsibility | required_trust_level |
|------|-------|----------------|----------------------|
| initialize | `InitializeNode` (framework) | session_id, trust_level, schema_version | — |
| pre_process | `PreProcessNode` | the caller contract: refuse instruction-override content, bounds-check every caller field, redact shipment identifiers → `validated_input` + `validated_context` | `VERIFIED_EXTERNAL` |
| main | `CustomsComplianceDocumentQAGraphNode` (`GraphNode`) | delegate to the inner `DomainWorkflowGraph`; map inner `formatted_answer` → outer `result` plus the structured fields; skip the delegation when `error_code` is already set | — (delegates) |
| post_process | `PostProcessNode` | the output gate: recursively scan `result` and every structured field for credential-shaped content; render the refusal sentence instead when `error_code` is set | `VERIFIED_EXTERNAL` |
| finalize | `FinalizeNode` (framework) | response_metadata, total_time_ms | — |

`{route}` is the framework's own `AgentBaseGraph.route()` — the outer `add_edges()` is not
overridden — and it reads **`status` alone**. A run that reaches that edge carrying
`AgentStatus.ERROR` routes straight to `finalize`, so `post_process` never runs and no body is
composed. A refusal the caller can correct therefore keeps `AgentStatus.SUCCESS` and stays on the
main line; what it carries instead is `error_code`, which routing never looks at. See "The refusal
contract" below.

### Inner graph (`DomainWorkflowGraph` — `BaseGraph`, linear)

```
START → input_validate → retrieve → rerank_filter → generate_answer → output_format → END
```

| Node | Responsibility | Input State | Output State |
|------|----------------|-------------|--------------|
| `InputValidateNode` | Settle the question and the search options; resolve the customs area | `validated_input`, `validated_context` | `search_query`, `customs_domain`, `query_filters`, `intake_notes` |
| `RetrieveNode` | Deterministic keyword retrieval: tokenise, score title/tag/content overlap, narrow to the resolved area unless it is `general` | `search_query`, `customs_domain`, `validated_context`, `retrieval_config` | `retrieved_documents`, `intake_notes` |
| `RerankFilterNode` | Area-match boost, drop below `score_threshold`, cap at `top_k` | `retrieved_documents`, `query_filters`, `validated_context`, `retrieval_config` | `ranked_documents` |
| `GenerateAnswerNode` | Rule-based checklist assembly from ranked passages only, with numbered citation markers and the schedule annotation | `ranked_documents`, `search_query` | `grounded_answer`, `citations`, `checklist_items`, `currency_note` |
| `OutputFormatNode` | Compose the final answer: checklist, area line, Sources, schedule note, and the standing disclaimer | `grounded_answer`, `citations`, `customs_domain`, `currency_note` | `formatted_answer`, `status` |

Trust is decided **once**, at the two outer boundary slots, which both require
`VERIFIED_EXTERNAL` — the level the manifest publishes. All five inner nodes require
`ANONYMOUS`, which any caller satisfies. An inner node demanding more than the manifest publishes
would admit a caller at the door and then deny them mid-pipeline, and that failure only shows up
on a real external call. Both directions are pinned in `test_trust_gate.py`.

### Data flow

```
user_input + input_context
  → PreProcessNode                                 → validated_input, validated_context
  → CustomsComplianceDocumentQAGraphNode
        .extract_input   stash validated_context   → inner DomainWorkflowGraph.invoke(validated_input)
        → input_validate                           → search_query / customs_domain / query_filters
        → retrieve                                 → retrieved_documents
        → rerank_filter                            → ranked_documents
        → generate_answer                          → grounded_answer / citations / checklist_items / currency_note
        → output_format                            → formatted_answer (+ the standing disclaimer)
     get_output()                                  → {formatted_answer, citations, checklist_items, ...}
  → .merge_output                                  → result / citations / checklist_items / customs_domain / currency_note
  → PostProcessNode (recursive output gate)        → formatted_output (gated) + structured fields (gated)
  → CustomsComplianceDocumentQAAgent.get_output()  → structured envelope, SUCCESS-only (fail-closed)
```

The two refusal paths leave that flow at different points:

```
correctable refusal (status stays SUCCESS, error_code set)
  → PreProcessNode                                 → error_code, error_log  (no validated_* written)
  → CustomsComplianceDocumentQAGraphNode.execute   → marker read, inner graph NOT invoked
  → PostProcessNode                                → the fixed sentence for that code → formatted_output
  → get_output()                                   → base envelope only (no structured fields)

terminating refusal (status becomes ERROR)
  → PreProcessNode, or the main slot               → status ERROR, error_log
  → {route} reads status                           → finalize, skipping post_process entirely
  → get_output()                                   → base envelope, output empty
     (the output gate fires after routing instead — it replaces an already
      composed body with the sanitised stub and empties the structured fields)
```

### The caller-context bridge

The framework's `GraphNode.execute()` invokes the inner graph as
`subgraph.invoke(user_input, session_id=..., ctx=...)`. It forwards no caller context — so
without a bridge, the caller's search options and knowledge passages would silently have no
effect on a nested template like this one, and every run would answer from the bundled corpus
regardless of what was sent.

`src/graph/context_bridge.py` closes that gap through the sanctioned subclass hooks: the outer
`extract_input()` stashes the validated context in a `ContextVar` immediately before the inner
invoke, and the inner `_extra_initial_state()` reads it back. A `ContextVar` keeps the hand-off
per thread and per task, so concurrent invocations in one process cannot observe each other's
context. Only the **validated** form crosses; raw caller data never reaches the inner graph.
`tests/proof_of_boundary/test_invoke_e2e.py` proves it end to end by asserting that, when a
caller corpus is supplied, the citations come from it.

### Runtime config forwarding (`_parent_config`)

`CustomsComplianceDocumentQAGraphNode._parent_config()` loads `config/config.yaml` and forwards
both tuning blocks under `config["configurable"]`, never `{}`:

```
{"configurable": {"retrieval": {top_k, score_threshold, kb_path}, "llm": {...}}}
```

`get_subgraph()` passes that into `DomainWorkflowGraph(config=...)`, and the inner graph
republishes the `retrieval` block into the inner initial state as the JSON-string field
`retrieval_config`. `RetrieveNode` and `RerankFilterNode` read it from State, falling back to
module defaults that mirror the file.

## The caller contract

`POST /invoke` accepts the question plus an optional `input_context`. `PreProcessNode` owns that
boundary: nothing reaches the inner workflow until it has accepted it.

| Field | Type | Bounds |
|---|---|---|
| `customs_domain` | string | one of `hs_code`, `naccs`, `incoterms`, `aeo`, `customs_law` |
| `top_k` | integer | 1–20 |
| `score_threshold` | number | 0.0–1.0 |
| `knowledge_entries` | list of objects | at most 25; unique inert `id`; `category` from the five areas; `content` required, ≤5000 chars; `title` / `source` ≤200 chars |

Three rules hold across all of it:

**Numbers go through a finite+bounded parser.** `float()` parses `"NaN"` and `"Infinity"`
happily, Python's `json` accepts bare `NaN` and `Infinity` in a request body, IEEE NaN compares
False against every bound — so an unguarded threshold silently stops filtering — and `int()` on
an infinite float raises `OverflowError`, which a plain `(TypeError, ValueError)` guard does not
catch. `finite_in_range()` in `src/schemas/state.py` rejects bools, non-numerics, non-finite
values and out-of-range magnitudes, and the request fails CLOSED — nothing validated is carried
forward and no search runs. How that refusal is *reported* is the next section.

**Rejected values are never echoed.** An error names the field and, for list entries, the
position — never the content.

**Absent data degrades to the baseline.** Send no `knowledge_entries` and the bundled corpus
answers, so a fresh checkout works before anything is wired in.

There are two channels for the search options, and they do not disagree about bounds. The
contract surface is `input_context`, which fails closed. A question may also arrive as a JSON
envelope (`{"query": ..., "domain": ..., "top_k": ...}`) parsed by `InputValidateNode`; an
unusable option there is dropped with a note and the configured default applies, because a bad
search option should not fail an otherwise valid customs question. Both channels run every number
through the same parser, so neither can pass a NaN or an out-of-range value through, and the
context channel wins where both are present.

## The refusal contract

Refusing is not one behaviour. A caller who sent one value in the wrong shape needs the run to end
with a sentence saying what to correct; a caller whose payload tried to rewrite the agent's
instructions needs the run to stop. Both refuse identically — no retrieval, no answer, nothing
validated carried forward, the same audit record — and they differ only in how the outcome is
reported.

**Correctable — the run COMPLETES carrying a reason.** `status` stays `AgentStatus.SUCCESS`,
`error_code` names what went wrong, and the caller reads a fixed sentence from
`src/services/failure_message.py`. Terminating here instead would end the calling surface's turn
and surface only an exception type, leaving the reason reachable solely from the audit trail;
completing lets the caller fix the value and send the question again on the same conversation.
A progress event is emitted so a caller watching the run sees it settle rather than stall.

| Trigger | `error_code` | Refused by |
|---|---|---|
| `user_input` absent, not a string, empty or whitespace-only | `EMPTY_INPUT` | `PreProcessNode` |
| `user_input` longer than 2000 characters | `QUESTION_TOO_LONG` | `PreProcessNode` |
| any `input_context` bound violation that is not an override refusal — `input_context` not an object; unknown `customs_domain`; `top_k` outside 1–20 or not a whole number; `score_threshold` outside 0.0–1.0 or non-finite; `knowledge_entries` not a list, over the 25 cap, an entry that is not an object, a missing/non-inert/duplicate `id`, a missing or unknown `category`, missing/empty/oversize `content`, oversize or non-text `title`/`source` | `INVALID_REQUEST` | `PreProcessNode` |

**Unfixable — the run TERMINATES.** `status` becomes `AgentStatus.ERROR` and no `error_code` is
set. A refusal settled before the conditional edge — at `pre_process`, or at the main slot — routes
straight to `finalize`, so `post_process` never runs and no body is composed at all. The output
gate is the one that fires *after* routing, with a body already composed, so it replaces that body
with the sanitised stub and empties the structured fields instead. Either way nothing the request
asked for is published. Rewording is not a remedy for any of these, so inviting another attempt
would be misleading.

| Trigger | Refused by |
|---|---|
| instruction-override content in the question | `PreProcessNode` |
| instruction-override content in a caller-supplied passage's `content` / `title` / `source` | `PreProcessNode` |
| credential-shaped content anywhere in the rendered answer or the structured fields | `PostProcessNode` (output gate) |
| caller below `VERIFIED_EXTERNAL` at either outer boundary slot | framework S-1 trust gate |
| high-confidence injection finding on `user_input` / `validated_input` | framework S-2 input gate |
| credential pattern in a node's returned dict | framework S-3 output gate |
| any inner-graph failure | `error_strategy: propagate` → `SubgraphError` |

**The marker travels; the nodes downstream of it do nothing.** Once `error_code` is set, each
remaining node reads it and passes through rather than acting:

| Reader | Behaviour when `error_code` is set |
|---|---|
| `CustomsComplianceDocumentQAGraphNode.execute()` | returns `{status: SUCCESS, error_code}`; the inner graph is never invoked, so no domain field is ever written |
| `CustomsComplianceDocumentQAGraphNode.merge_output()` | a reason settled at the outer layer wins over one from `sub_result` — a plain `get()` would erase the specific reason with a vaguer one |
| `DomainWorkflowGraph.get_output()` | surfaces `error_code` out of the subgraph, so the outer layer can report a reason the inner run settled |
| `PostProcessNode.execute()` | renders the sentence for that code into `formatted_output` / `result` and returns; the output gate has nothing to scan |
| `CustomsComplianceDocumentQAAgent.get_output()` | returns the base envelope only — a run that did not carry out the request must not publish citations, checklist items, an area or a schedule note |

`error_code` is an internal marker. It is never published: `get_output()` returns the base
envelope's keys, and the code itself is not among them — the caller reads the sentence, not the
code.

## Screens, and the direction that actually costs you

`PreProcessNode` refuses instruction-override content on the question channel and on every
caller-supplied passage. It does this itself rather than relying on the framework's input scan:
where that scan is absent or configured off, a template that delegates the guarantee fails
**open** — the payload reaches the answer path and the run succeeds. `test_pre_process_node.py`
proves the refusal by calling `execute()` directly, with no framework wrapper in front.

This refusal **terminates**: it is on the unfixable side of the contract above, including when it
arrives on the caller-passage channel alongside the bounded options that complete with a reason
code. Rewording the payload is not a correction.

The screen is deliberately narrow, and every alternative requires the prompt-specific noun,
because the fail-closed direction is the one that blocks real work. Customs questions legitimately
say "can a broker **act as an** importer of record", "does an advance ruling **override the**
classification", "can I **ignore the previous** declaration", and "**show me the rules** for
voluntary disclosure". An unanchored screen refuses all four. The suite probes both directions:
attack forms refused, and a set of real customs questions carrying the same verbs answered.

There is deliberately **no SQL-verb screen**. This template opens no database, and `select`,
`update`, `delete` and `drop` are ordinary words in declaration language — "how do I *select* the
correct heading", "*delete* a line from a NACCS entry", "*drop* shipments". A screen with no
protective value here would only ever produce false refusals.

## Identifier redaction

`PreProcessNode` redacts ISO 6346 container numbers, booking/AWB references of 11+ digits, and
e-mail addresses, on the question channel **and** on caller-supplied passage text — caller passages
are a second free-text route into the same answer, so the same redaction has to run on both.

HS codes are deliberately **not** redacted. Six-to-ten bare digits are public tariff data and are
exactly what this template answers questions about; the booking-reference pattern is shaped to
clear that range. `test_pre_process_node.py` asserts that boundary in both directions.

## On numeric rounding grids

Some templates in this family render monetary aggregates and snap every monetary token onto a
rounding grid on the way out. **That grid is deliberately not used here.** This agent renders no
monetary aggregate — it answers classification and procedure questions and quotes references — and
the grammar such a grid uses reads any standalone three-letter uppercase word as a currency
marker.

In customs work those words are the content. Incoterms codes are literally three-letter uppercase
words (`FOB`, `CIF`, `EXW`, `DDP`, `DAP`), and they sit next to numbers constantly. So do route
codes, container prefixes and dangerous-goods numbers. Running the family's grammar over this
domain's own text rewrites `MSKU 4512345` to `MSKU 4,512,000`, `REF-2026-0041` to
`REF-2,000-0041`, and `CIF 1200 pcs` to `CIF 1,000 pcs` — it would mangle the very references the
answer exists to quote.

The invariant enforced instead is the one below: cited or refused, disclaimed, and free of
credential-shaped content. `tests/proof_of_boundary/test_invoke_e2e.py` pins the absence of a grid
directly, asserting that this domain's identifier forms — HS subheadings, container numbers, entry
references, route codes, bare Incoterms codes, times, weights, piece counts, and a pure-numeric
reference with no letters to protect it — come out of the real output path byte-identical, and
that a numbered heading following a three-letter code keeps its number.

## The output invariant

Every answer this template emits satisfies all four of these, on every path:

1. **Cited or refused.** `GenerateAnswerNode` assembles each checklist point from exactly one
   ranked passage and gives it a `[n]` marker traceable to that passage. With zero ranked
   passages it returns a no-coverage refusal and no citations rather than fabricating a point.
   Nothing outside `ranked_documents` reaches the answer, so the answer is grounded by
   construction.
2. **Disclaimed.** `OutputFormatNode` appends the standing customs-guidance disclaimer on every
   code path through the node, including the refusal path. No branch skips it, so it cannot be
   removed by shaping the input. The disclaimer belongs to this node's output contract — it is
   not injected by `post_process`, which only gates.
3. **Free of credential-shaped content.** `PostProcessNode` runs a module-level
   `_security_gate_output()` scan that RECURSES into nested dict/list/tuple structures rather
   than reading only the top-level `result` string — a credential nested inside a citation entry
   or a checklist item is exactly what a shallow scan misses, and those structured fields are
   published too. On a violation, `formatted_output` and `result` become a sanitised stub, the
   structured fields are emptied, and status becomes ERROR.
4. **Structured only when an answer was actually produced.** `get_output()` EXTENDS
   `super().get_output()` — status, node_history and trace_id stay intact — and adds `citations` /
   `checklist_items` / `customs_domain` / `currency_note` only when `error_code` is unset **and**
   status is SUCCESS. That is the fail-closed half of the gate, and it covers both refusal paths: a
   blocked output surfaces the sanitised string alone, and a run that completed carrying a reason
   code surfaces the sentence alone. Neither publishes anything structured alongside it.

## State definition

| Field | Type | Purpose | Layer |
|-------|------|---------|-------|
| `validated_input` | `NotRequired[str]` | redacted, length-capped question | outer |
| `validated_context` | `NotRequired[Optional[str]]` (JSON) | bounds-checked caller options and passages | outer |
| `customs_answer` | `NotRequired[str]` | final answer, mapped from inner `formatted_answer` | outer |
| `search_query` | `NotRequired[str]` | normalised customs question | inner |
| `customs_domain` | `NotRequired[str]` | resolved area slug (`hs_code`\|`naccs`\|`incoterms`\|`aeo`\|`customs_law`\|`general`) | inner |
| `query_filters` | `NotRequired[Optional[str]]` (JSON) | effective query params (`domain`, `top_k`) | inner |
| `retrieval_config` | `NotRequired[Optional[str]]` (JSON) | forwarded `retrieval` block | inner |
| `retrieved_documents` | `NotRequired[Optional[str]]` (JSON) | scored candidates | inner |
| `ranked_documents` | `NotRequired[Optional[str]]` (JSON) | reranked + threshold-filtered passages | inner |
| `grounded_answer` | `NotRequired[str]` | checklist answer body | inner |
| `citations` | `NotRequired[Optional[str]]` (JSON) | `[{ref, id, title, source}]` | inner |
| `checklist_items` | `NotRequired[Optional[str]]` (JSON) | `list[str]` of checklist points | inner |
| `currency_note` | `NotRequired[str]` | tariff-schedule annotation | inner |
| `formatted_answer` | `NotRequired[str]` | final composed answer | inner |
| `intake_notes` | `NotRequired[Optional[str]]` (JSON) | validation / parse notes, no caller values | inner |
| `error_code` | `Optional[str]` | internal marker for a refusal the caller can correct (`EMPTY_INPUT` \| `QUESTION_TOO_LONG` \| `INVALID_REQUEST`); read by every node downstream of the refusal and never published in the envelope | both |
| `trace_id` / `correlation_id` | `Optional[str]` | framework-managed tracing | both |

State constraints:

- Flat `TypedDict` only — primitives and JSON-serialisable types.
- Structured fields stored as JSON strings via `to_json()` / `from_json()`, used consistently by
  every producer AND consumer.
- Domain fields are `NotRequired[...]`, so the TypedDict is valid before any node has written.
- `formatted_output` is not re-declared — that backbone field stays framework-owned.
- No credentials, tokens or raw personal identifiers in State.
- No Pydantic models, dataclasses, or arbitrary Python objects.

## Audit events

Every node's `execute()` emits exactly one domain event on its success path, carrying counts and
flags only — never caller text. Nodes do not emit `node_start` / `node_complete` / `node_error`;
the framework's `BaseNode.__call__()` does that.

`pre_process_complete` · `input_validate_complete` · `retrieve_complete` ·
`rerank_filter_complete` · `generate_answer_complete` · `output_format_complete` ·
`post_process_complete`

A run that completes carrying a reason code emits `post_process_degraded` in place of
`post_process_complete`, with the reason code as its only payload. A refusal is still an audited
outcome — reporting it as a completion must not make it invisible to the audit trail.

## Model synthesis seam

The pipeline is deterministic: retrieval is keyword scoring and `GenerateAnswerNode` assembles the
checklist rule-based, one point per ranked passage. There is no model call and no model client
dependency, which is why `requires.extras` is empty. The `llm` block in `config/config.yaml` is
forwarded through `_parent_config()` for forward compatibility and is consumed by no node today.

`config/prompts/answer_synthesis_prompt.md` describes the upgrade seam: a synthesising
`GenerateAnswerNode` would call a model over the same `ranked_documents` input and emit the same
`grounded_answer` / `citations` / `checklist_items` / `currency_note` State contract, so no other
node changes. Adding it means declaring the extra and the secret in `config/agent.yaml`.

## Composition pattern

- **Pattern:** `GraphNode` (subgraph) in the outer `main` slot.
- **Composition target:** `DomainWorkflowGraph` (inner `BaseGraph`).
- **Error strategy:** `propagate` — inner errors re-raised as `SubgraphError`.
- Inner domain nodes run at `ANONYMOUS`; the outer boundary slots at `VERIFIED_EXTERNAL`.

## Import isolation

- [x] Template code does not import the platform SDK.
- [x] Import targets: `framework/` and `shared/` only.
- [x] The base class is the framework base class directly.

## Design decision record

| Decision | Option A | Option B | Chosen | Rationale |
|----------|----------|----------|--------|-----------|
| Base type | `AgentBaseGraph` | `AutonomousBaseGraph` | **`AgentBaseGraph`** | A fixed multi-step retrieval workflow, not an autonomous loop |
| Composition | Standalone slots | `GraphNode` → inner `BaseGraph` | **`GraphNode` → inner `BaseGraph`** | A five-step domain workflow exceeds a single `main` node; nesting keeps the outer backbone untouched |
| Answer synthesis | Rule-based assembly | Model call | **Rule-based** | Deterministic and auditable, testable offline; the seam above swaps a model in without touching the rest |
| Corpus | External vector store | Bundled JSON corpus + caller-supplied passages | **Both** | Self-contained and deterministic out of the box; callers answer from their own corpus without deploying a store. The `retrieved_documents` contract is store-agnostic |
| Output shape | Plain prose | Structured (`get_output()` override) | **Structured** | The product is a checklist plus citations, area routing and a schedule note — machine-readable fields, not just prose |
| Area routing | Separate classify node | Folded into `InputValidateNode` | **Folded in** | Classification needs only the normalised question; a separate node would add a State round-trip and nothing else |
| Keyword matching | Substring | Whole word / phrase | **Whole word / phrase** | Three-letter Incoterms codes hide inside ordinary customs words — `cif` in "specific", `cip` in "principle", `dap` in "adapt" — and an unanchored classifier routed "what is the specific duty rate" to Incoterms, filtering out the Customs Act passages that answer it |
| Rounding grid | Apply the family's grid | No grid; pin identifiers instead | **No grid** | No monetary aggregate is rendered, and the grid's grammar mangles this domain's identifiers (see above) |
| Error routing | Terminate every refusal | Complete a correctable refusal with a reason code | **Complete with a reason code** | The backbone's conditional edge is the framework's and reads `status` alone, so terminating routes past `post_process` and composes no body — the caller sees an exception type and the reason survives only in the audit trail. Keeping SUCCESS holds the run on the main line, and `error_code` carries what to correct to a node that can render it. Refusals rewording cannot fix still terminate |
| Refusal marker | Branch the graph on it | Carry it in State; each node reads and passes through | **Carry it in State** | `route()` is framework-owned and must not be overridden for this, and a conditional edge per refusal would have to be repeated in both layers. A marker each downstream node reads keeps the topology fixed while guaranteeing no domain field is written after a refusal |
