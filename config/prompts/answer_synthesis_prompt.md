# Answer Synthesis Prompt — LOG-C2-039 (v2 LLM upgrade seam)

> **v1 does NOT use this prompt at runtime.** v1 of `GenerateAnswerNode` is
> deterministic (rule-based checklist assembly over `ranked_documents`); no node
> reads this file. It documents the synthesis contract for the v2 LLM upgrade
> described in `docs/02_design.md` ("v1 Implementation Note — LLM synthesis"), so
> the v2 swap changes only the inside of `GenerateAnswerNode.execute()`.

## Contract (v2 GenerateAnswerNode)

- **Input:** the same `ranked_documents` JSON (id / title / category / source /
  score / excerpt) and `search_query` the v1 node reads, plus `customs_domain`
  (the classified routing domain from `InputValidateNode`).
- **Output:** the same state contract — `grounded_answer` (str, checklist format
  with numbered `[n]` citation markers), `citations` (JSON list of
  `{ref, id, title, source}`), `checklist_items` (JSON list[str]), and
  `currency_note` (str).
- **Grounding rule (cite-or-refuse):** every checklist point must be traceable
  to one of the supplied passages via a `[n]` marker; content not present in
  the passages must not be asserted. When no passage clears the relevance
  threshold, refuse — recommend rephrasing or escalating to a licensed 通関士 /
  通関業者 — never answer from parametric knowledge.
- **Currency rule:** every coverage-bearing answer carries a tariff/currency
  annotation noting that HS classifications, duty rates, and NACCS procedural
  details are revised periodically and must be verified against the current
  official publication before filing.
- **Tone:** neutral, compliance-appropriate reference information — never a
  binding classification or declaration ruling (the non-suppressible
  disclaimer is appended downstream by `OutputFormatNode`, not by this
  prompt).

## Prompt template

```
You answer Japan-customs compliance questions strictly from the knowledge-base
passages provided below, spanning five domains: HS-code classification, NACCS
declaration guidance, Incoterms 2020 (public paraphrase/summary), AEO
certification, and 関税法 (Customs Act).

Question:
{search_query}

Classified customs domain:
{customs_domain}

Passages (each with a reference number):
{ranked_documents}

Rules:
1. Use ONLY the passages above. If they do not answer the question, say the
   customs knowledge base has insufficient coverage and stop — do not guess.
2. Render the answer as a checklist; mark every point with the [n] reference
   of its passage.
3. Do not issue a binding HS classification, customs declaration, or legal
   ruling — this is reference information only.
4. Note that tariff schedules and HS codes change periodically and should be
   verified before filing.
5. Keep the answer under 300 words.
```

## Manifest coupling

The `llm` block in `config/agent.yaml` (`temperature`, `max_tokens`) is already
forwarded to the inner graph via
`CustomsComplianceDocumentQAGraphNode._parent_config()` under
`config["configurable"]["llm"]`; the v2 node reads it from there.
