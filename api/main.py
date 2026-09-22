"""
api/main.py

FastAPI application entry point for the Evidence-Gated Self-Healing Agent.

Architecture notes
------------------
* All state lives in api.dependencies.AppState (in-memory singleton).
* Routes are modular — three new evidence-focused routers plus legacy ones.
* Verification stays 100% deterministic — no LLM is consulted for evidence.
* Startup/shutdown lifespan logs confirm the architectural guarantees.

Zero-trust distinction exposed in the API
-----------------------------------------
  EXECUTOR_CLAIM    — tool self-report (unverified, may be false)
  MACHINE_EVIDENCE  — sandbox ground truth (cannot be fabricated)
  VERIFICATION_RESULT — deterministic PASS/FAIL/INCONCLUSIVE verdict

Run with:
    uvicorn api.main:app --reload --port 8000

Interactive docs:
    http://localhost:8000/docs   (Swagger UI)
    http://localhost:8000/redoc  (ReDoc)
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.routes.sandbox import router as sandbox_router
from api.routes.workflows import router as workflows_router
from api.routes.workflow import router as workflow_router
from api.routes.evidence import router as evidence_router
from api.routes.audit import router as audit_router
from api.schemas import ErrorResponse, HealthResponse
from api.dependencies import get_app_state, llm_is_configured

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("=" * 60)
    log.info("Evidence-Gated Self-Healing Agent API starting")
    log.info("Verification: DETERMINISTIC (no LLM in evidence pipeline)")
    log.info("Architectural guarantee: LLM never fabricates evidence")
    log.info("Zero-trust: EXECUTOR_CLAIM / MACHINE_EVIDENCE / VERIFICATION_RESULT")
    llm_status = "CONFIGURED" if llm_is_configured() else "NOT CONFIGURED (mock mode)"
    log.info("LLM provider: %s", llm_status)
    log.info("=" * 60)
    yield
    log.info("API shutting down.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Evidence-Gated Self-Healing Agent",
    description=(
        "A zero-trust AI agent workflow system.\n\n"
        "## Zero-Trust Evidence Model\n\n"
        "Every action produces a **3-layer evidence record**:\n\n"
        "| Layer | Source | Trust Level |\n"
        "|-------|--------|-------------|\n"
        "| **EXECUTOR_CLAIM** | Tool self-report | UNVERIFIED — may be false |\n"
        "| **MACHINE_EVIDENCE** | `sandbox.state_store` | GROUND TRUTH — cannot be fabricated |\n"
        "| **VERIFICATION_RESULT** | Deterministic engine | AUTHORITATIVE — PASS/FAIL/INCONCLUSIVE |\n\n"
        "**Key guarantees:**\n"
        "- Evidence ALWAYS originates from `sandbox.state_store`, never from any LLM\n"
        "- INCONCLUSIVE is **never** silently promoted to PASS\n"
        "- State transitions are **explicit** — invalid jumps are rejected\n"
        "- Retries are **bounded** — infinite loops are impossible\n"
        "- Atomic rollback reverts **all** mutations on failure (when enabled)\n"
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ---------------------------------------------------------------------------
# CORS (permissive for local frontend dev; tighten in production via env var)
# ---------------------------------------------------------------------------

_cors_origins = os.getenv("CORS_ORIGINS", "*").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Global exception handler
# ---------------------------------------------------------------------------


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    log.exception("Unhandled exception on %s %s", request.method, request.url)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc),
        ).model_dump(),
    )


# ---------------------------------------------------------------------------
# Routers — new evidence-focused routes at /workflow/
# ---------------------------------------------------------------------------

app.include_router(workflow_router)    # POST /workflow/run, GET /workflow/{id}, ...
app.include_router(evidence_router)    # GET /workflow/{id}/evidence
app.include_router(audit_router)       # GET /workflow/{id}/audit

# ---------------------------------------------------------------------------
# Legacy routers (preserved for backward compatibility)
# ---------------------------------------------------------------------------

app.include_router(workflows_router)   # /workflows/run, /workflows/{id}, ...
app.include_router(sandbox_router)     # /sandbox/state, /sandbox/reset, ...

# ---------------------------------------------------------------------------
# Health / root
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health() -> HealthResponse:
    """
    Liveness check — always returns 200 if the server is up.

    Reports total workflow run count and LLM configuration status.
    """
    state = get_app_state()
    return HealthResponse(workflow_count=len(state.list_results()))


@app.get("/", tags=["System"], include_in_schema=False)
def root() -> dict[str, str]:
    return {
        "service": "Evidence-Gated Self-Healing Agent",
        "docs": "/docs",
        "health": "/health",
        "run_workflow": "POST /workflow/run",
        "evidence": "GET /workflow/{id}/evidence",
        "audit": "GET /workflow/{id}/audit",
        "llm_configured": str(llm_is_configured()),
    }
