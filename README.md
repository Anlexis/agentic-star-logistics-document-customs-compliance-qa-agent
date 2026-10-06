# Logistics Document & Customs Compliance Q&A Agent

AI agent for answering logistics document and Japanese customs compliance questions, built with Agentic Star.

> **Category**: Cat 2 (a domain pipeline — several steps orchestrated for one specific job)
> **Industry**: Logistics
> **Template ID**: LOG-C2-039

## Overview

Answers natural-language customs-compliance questions from a curated knowledge base and returns
a cited, checklist-format answer rather than a paragraph of prose.

The corpus spans the five areas a customs question usually falls into — HS-code classification,
NACCS declaration procedure, Incoterms 2020 (paraphrase and orientation, never the copyrighted
rules text), AEO certification, and Japan's Customs Act (関税法). A question is routed to the
matching area, the relevant passages are scored and ranked, and each checklist point is assembled
from exactly one passage and carries its citation. Nothing outside the retrieved passages reaches
the answer, so when the corpus does not cover a question the agent says so and points to a
licensed customs specialist (通関士) instead of inventing a classification.

Callers can answer from their own passages instead of the bundled corpus by sending them with the
request, which is what makes the template useful against a real internal knowledge base. Every
answer carries a standing disclaimer: this is reference information, and the licensed broker of
record still owns the binding classification and filing decision.

Typical users are logistics operators, customs brokers (通関業者), and import/export compliance
staff, for whom the same lookup otherwise means cross-reading several separate regulatory sources.

This is an agent template built with the **AGENTIC STAR** development platform and the
**AgentCore Framework**. It is intended to be taken as a starting point: fork it, adapt it to
your own data and policies, and run it inside your own AGENTIC STAR deployment.

## Requirements

**This template does not run standalone.** It requires:

| Requirement | Notes |
|---|---|
| **AGENTIC STAR platform** | The agent connects to the platform at start-up. Without it, start-up fails immediately (see *Behaviour without the platform* below). Deployment guides and API documentation: [AGENTIC STAR Developers](https://developers.fd.agenticstar.tm.softbank.jp/) |
| **AgentCore Framework** (`agenticstar-agentcore`) | Installed from PyPI as a dependency. |
| Python | >=3.11 |

```bash
pip install -e .
```

### Behaviour without the platform

The framework is designed to run **only** on AGENTIC STAR. There is no fallback or degraded mode.
The agent resolves its platform services while the graph is being compiled and its secrets
provider is being bound — so if the platform is unreachable or the installed SDK does not match,
start-up raises there and the process stops, rather than serving requests in a partially working
state. This is intentional: a half-running agent is worse than one that refuses to start.

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest tests/ -v
```

Tests run without a platform connection. Running the agent itself does not.

Worked examples of the graph patterns live in `src/examples/`; `graph_cat2_sample.py` compiles
this agent and answers one question end to end.

## Calling it

`POST /invoke` takes the question plus an optional `input_context`:

```json
{
  "input": "Which HS subheading applies to knitted cotton t-shirts?",
  "input_context": {
    "customs_domain": "hs_code",
    "top_k": 3,
    "knowledge_entries": [
      {
        "id": "knitted_cotton_shirts",
        "category": "hs_code",
        "title": "Knitted cotton shirt classification note",
        "source": "Internal broker handbook, rev 4",
        "content": "Knitted cotton t-shirts are classified under subheading 6109.10."
      }
    ]
  }
}
```

Every `input_context` field is optional and every one is bounds-checked before anything downstream
reads it; send none of them and the bundled corpus answers. `customs_domain` is one of `hs_code`,
`naccs`, `incoterms`, `aeo`, `customs_law`. A rejected field fails the request and names the field
without echoing the value.

## Project Structure

```
src/          agent implementation (nodes, services, schemas) and runnable examples
tests/        unit and boundary tests
config/       agent manifest, runtime parameters, and the seeded knowledge base
docs/         design and test documentation
```

`config/agent.yaml` is the static manifest — identity, entry point, required caller trust level.
`config/config.yaml` holds the live runtime parameters (retry, timeout, retrieval tuning). See
`docs/02_design.md` for the design and `docs/03_test_spec.md` for the test specification.

## Customising

1. Adjust `config/config.yaml` for your own retrieval tuning.
2. Replace `config/kb/customs_kb.json` with your own passages, or send them per request.
3. Review the node implementations under `src/nodes/` for domain-specific logic.
4. Re-run the test suite.

## License

MIT — see [LICENSE](LICENSE).

## Status of this repository

This template is published **as is**, by its individual author, under the MIT license. It carries
**no warranty and no support commitment**, and no organisation stands behind its behaviour or
fitness for any purpose. Issues and pull requests may or may not receive a response; that is at
the sole discretion of the repository owner.
