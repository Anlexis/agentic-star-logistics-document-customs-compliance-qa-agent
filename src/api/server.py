"""AgentCore Platform v1.0"""

# Standalone HTTP entry point for the agent.
# Entry points are adapters only — no business logic here.
# For platform-level routing, the gateway calls agent.invoke() directly.

import os
import secrets
from pathlib import Path
from typing import Any, Dict
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from framework.schemas.invocation_context import InvocationContext
from framework.schemas.trust_level import TrustLevel
from framework.secrets.context import bound_secrets
from shared.secrets import factory as secrets_factory
from src.graph.graph import CustomsComplianceDocumentQAAgent

app = FastAPI(title="Agent")

# Runtime parameters live in config/config.yaml. Construct the graph WITH them:
# an agent built with no config falls back to framework defaults, so every value
# declared in that file — max_retry, timeout_s, the retrieval block — would be
# dead on this path while appearing to be configured.
_RUNTIME_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "config.yaml"


def _load_runtime_config() -> Dict[str, Any]:
    """Read config/config.yaml; an unreadable file degrades to framework defaults."""
    try:
        import yaml

        loaded = yaml.safe_load(_RUNTIME_CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return loaded if isinstance(loaded, dict) else {}


agent = CustomsComplianceDocumentQAAgent(config=_load_runtime_config())
agent.compile()
# The namespace matches the `namespace` value in config/agent.yaml.
agent.provision_secrets(secrets_factory(namespace="log", agent_name="CustomsComplianceDocumentQAAgent"))


class InvokeRequest(BaseModel):
    input: str
    session_id: str = ""
    # Per-request options and, optionally, the knowledge passages to answer FROM.
    # Every field is validated against explicit bounds by PreProcessNode before
    # anything downstream reads it.
    input_context: Dict[str, Any] = Field(default_factory=dict)


@app.post("/invoke")
async def invoke(req: InvokeRequest, request: Request) -> Any:
    trust = getattr(request.state, "trust_level", TrustLevel.ANONYMOUS)
    # Standalone caller auth: when INVOKE_AUTH_TOKEN is set on the server
    # environment, callers that no upstream middleware vouched for (still
    # ANONYMOUS) must present it as a Bearer token and run at
    # VERIFIED_EXTERNAL. Middleware-established trust is never demoted. This
    # adapter is the entry-point auth boundary — a deployment-level caller
    # credential, not an agent secret, so the secrets provider does not apply
    # (no InvocationContext exists before auth).
    expected = os.environ.get("INVOKE_AUTH_TOKEN")
    if expected and trust is TrustLevel.ANONYMOUS:
        supplied = request.headers.get("authorization", "")
        # Compare bytes: compare_digest raises TypeError on non-ASCII str input
        # (headers decode as latin-1), which would 500 instead of the generic 401.
        if not secrets.compare_digest(supplied.encode(), f"Bearer {expected}".encode()):
            # Generic body on purpose — do not leak whether the token was absent,
            # malformed, or wrong.
            raise HTTPException(status_code=401, detail="Token is invalid or expired.")
        trust = TrustLevel.VERIFIED_EXTERNAL
    with bound_secrets(agent._secrets_provider):
        ctx = InvocationContext(
            session_id=req.session_id or str(uuid4()),
            caller_trust_level=trust,
            caller_id=getattr(request.state, "caller_id", ""),
        )
        return agent.invoke(req.input, ctx=ctx, input_context=req.input_context)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "agent": "CustomsComplianceDocumentQAAgent"}
