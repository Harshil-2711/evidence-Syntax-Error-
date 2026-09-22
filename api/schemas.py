"""
api/schemas.py

Pydantic request/response models for the FastAPI layer.

These are the API contracts — separate from core domain models so the
REST surface can evolve independently.  Serialisation helpers convert
between domain objects and API objects.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from core.models import ActionStatus, ActionType, VerificationStatus, WorkflowStatus


# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------


class RunWorkflowRequest(BaseModel):
    """POST /workflows/run"""

    goal: str = Field(
        min_length=1,
        max_length=2000,
        examples=["Create project Hackathon Alpha, add Harshit, make it private, generate a report."],
    )
    max_retries: int = Field(default=3, ge=0, le=10)
    max_replan_cycles: int = Field(default=2, ge=0, le=5)
    planner_mode: str = Field(default="mock", pattern="^(mock|llm)$")


class ResetSandboxRequest(BaseModel):
    """POST /sandbox/reset — clears all sandbox state."""

    confirm: bool = Field(
        description="Must be True to confirm destructive reset."
    )


# ---------------------------------------------------------------------------
# Response: Action
# ---------------------------------------------------------------------------


class ActionResponse(BaseModel):
    action_id: str
    action_type: str
    parameters: dict[str, Any]
    expected_postcondition: dict[str, Any]
    status: str
    execution_result: dict[str, Any] | None
    evidence_id: str | None
    retry_count: int
    error: str | None
    timestamp: str

    @classmethod
    def from_action(cls, action) -> "ActionResponse":
        return cls(
            action_id=action.action_id,
            action_type=action.action_type.value,
            parameters=action.parameters,
            expected_postcondition=action.expected_postcondition,
            status=action.status.value,
            execution_result=action.execution_result,
            evidence_id=action.evidence_id,
            retry_count=action.retry_count,
            error=action.error,
            timestamp=action.timestamp.isoformat(),
        )


# ---------------------------------------------------------------------------
# Response: Evidence
# ---------------------------------------------------------------------------


class EvidenceResponse(BaseModel):
    evidence_id: str
    action_id: str
    source: str
    observed_state: dict[str, Any]
    expected_state: dict[str, Any]
    verification_status: str
    timestamp: str

    @classmethod
    def from_evidence(cls, evidence) -> "EvidenceResponse":
        return cls(
            evidence_id=evidence.evidence_id,
            action_id=evidence.action_id,
            source=evidence.source,
            observed_state=evidence.observed_state,
            expected_state=evidence.expected_state,
            verification_status=evidence.verification_status.value,
            timestamp=evidence.timestamp.isoformat(),
        )


# ---------------------------------------------------------------------------
# Response: Workflow
# ---------------------------------------------------------------------------


class WorkflowSummaryResponse(BaseModel):
    """Lightweight summary returned in list endpoints."""

    run_id: str
    name: str
    status: str
    action_count: int
    succeeded_count: int
    failed_count: int
    created_at: str
    updated_at: str


class WorkflowDetailResponse(BaseModel):
    """Full detail returned for a single workflow."""

    run_id: str
    name: str
    status: str
    actions: list[ActionResponse]
    evidence_log: list[EvidenceResponse]
    transition_history: list[dict[str, str]]
    summary: str
    created_at: str
    updated_at: str

    @classmethod
    def from_result(cls, result, summary: str = "") -> "WorkflowDetailResponse":
        run = result.run
        history = run.metadata.get("transition_history", [])
        succeeded = sum(1 for a in run.actions if a.status == ActionStatus.SUCCEEDED)
        failed = sum(1 for a in run.actions if a.status == ActionStatus.FAILED)
        return cls(
            run_id=run.run_id,
            name=run.name,
            status=run.status.value,
            actions=[ActionResponse.from_action(a) for a in run.actions],
            evidence_log=[EvidenceResponse.from_evidence(e) for e in run.evidence_log],
            transition_history=history,
            summary=summary or result.summary,
            created_at=run.created_at.isoformat(),
            updated_at=run.updated_at.isoformat(),
        )


class RunWorkflowResponse(BaseModel):
    """Response returned immediately after triggering a workflow run."""

    run_id: str
    status: str
    summary: str
    goal: str
    action_count: int
    succeeded_count: int
    failed_count: int
    evidence_count: int
    detail: WorkflowDetailResponse


# ---------------------------------------------------------------------------
# Response: Sandbox state
# ---------------------------------------------------------------------------


class SandboxStateResponse(BaseModel):
    """Current snapshot of all sandbox entities."""

    projects: dict[str, Any]
    members: dict[str, Any]
    permissions: dict[str, Any]
    files: dict[str, Any]
    notifications: dict[str, Any]
    reports: dict[str, Any]
    journal_length: int


# ---------------------------------------------------------------------------
# Response: Health
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "0.1.0"
    layer: str = "evidence-gated-agent"
    workflow_count: int = 0


# ---------------------------------------------------------------------------
# Response: Error
# ---------------------------------------------------------------------------


class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None
