"""
api/routes/workflow.py

Core workflow routes — POST /workflow/run, GET status/actions,
and POST /workflow/{id}/failure-injection for deterministic demo failures.

Zero-trust distinction exposed in every response
-------------------------------------------------
EXECUTOR_CLAIM  — the tool's unilateral self-report  (unverified)
MACHINE_EVIDENCE — sandbox-observed ground truth     (cannot be fabricated)
VERIFICATION_RESULT — deterministic PASS/FAIL/INCONCLUSIVE verdict

The /actions endpoint makes this distinction explicit for every action.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from api.dependencies import (
    AppState,
    WorkflowRunRecord,
    _app_state,
    build_and_run,
    llm_is_configured,
)
from core.models import ActionStatus, ActionType, VerificationStatus
from sandbox.failure_injection import FailureInjector, FailureMode, FailureRule

log = logging.getLogger(__name__)

router = APIRouter(prefix="/workflow", tags=["Workflow"])


# ===========================================================================
# Request / response schemas
# ===========================================================================


class RunWorkflowRequest(BaseModel):
    """
    Request body for POST /workflow/run.

    goal:             Natural-language task description.
    max_retries:      Per-action retry budget (default 3, max 10).
    max_replan_cycles: Global replan budget (default 2).
    atomic_completion: If True, any failure triggers full rollback.
    planner_mode:     "mock" (no API key required) | "llm" (requires key).
    """

    goal: str = Field(..., min_length=1, description="Natural-language task goal")
    max_retries: int = Field(3, ge=0, le=10)
    max_replan_cycles: int = Field(2, ge=0, le=5)
    atomic_completion: bool = Field(False, description="Roll back all actions on any failure")
    planner_mode: str = Field("mock", pattern="^(mock|llm)$")


class WorkflowSummaryResponse(BaseModel):
    workflow_id: str
    goal: str
    final_status: str
    total_actions: int
    verified_actions: int
    failed_actions: int
    rolled_back_actions: int
    recovery_count: int
    retry_count: int
    evidence_coverage: float
    blocked_reason: str | None
    created_at: str
    parent_run_id: str | None
    injector_active: bool
    coordinator_summary: str

    @classmethod
    def from_record(cls, record: WorkflowRunRecord) -> "WorkflowSummaryResponse":
        s = record.summary
        return cls(
            workflow_id=record.run_id,
            goal=record.goal,
            final_status=record.result.run.status.value,
            total_actions=s.total_actions if s else 0,
            verified_actions=s.verified_actions if s else 0,
            failed_actions=s.failed_actions if s else 0,
            rolled_back_actions=s.rolled_back_actions if s else 0,
            recovery_count=s.recovery_count if s else 0,
            retry_count=s.retry_count if s else 0,
            evidence_coverage=round(s.evidence_coverage, 3) if s else 0.0,
            blocked_reason=s.blocked_reason if s else None,
            created_at=record.created_at.isoformat(),
            parent_run_id=record.parent_run_id,
            injector_active=bool(record.injector_config),
            coordinator_summary=record.result.summary,
        )


# ---------------------------------------------------------------------------
# Actions schema — explicitly shows the 3-way distinction
# ---------------------------------------------------------------------------


class ExecutorClaimInAction(BaseModel):
    """
    The tool's self-reported outcome.

    CAUTION: This is NOT verified truth. The tool could be lying
    (FALSE_SUCCESS injection) or simply wrong. Only MACHINE_EVIDENCE
    confirms what actually happened.
    """

    self_reported_success: bool | None
    failure_mode_detected: str | None
    CAUTION: str = Field(
        default="UNVERIFIED — executor self-report only. Not machine-checked.",
        description="This claim has NOT been independently verified.",
    )


class ActionResponse(BaseModel):
    """
    Per-action status with the 3-way zero-trust distinction.

    executor_claim       → what the tool said (may be false)
    verification_verdict → what the verifier decided (authoritative)
    final_status         → SUCCEEDED only if verifier said PASS
    contradiction        → True if executor claimed success but verifier said FAIL
    """

    action_id: str
    action_index: int
    action_type: str
    parameters: dict[str, Any]
    expected_postcondition: dict[str, Any]
    retry_count: int

    # ── The 3-way distinction ──────────────────────────────────────────
    executor_claim: ExecutorClaimInAction
    verification_verdict: str  # PASS | FAIL | INCONCLUSIVE | PENDING
    final_status: str

    # ── Derived signals ────────────────────────────────────────────────
    contradiction_detected: bool = Field(
        description="True if executor claimed success but verifier produced FAIL. "
                    "This is a zero-trust violation caught by machine evidence."
    )
    evidence_id: str | None

    @classmethod
    def from_action_and_record(
        cls,
        action: Any,
        record: Any,
        index: int,
    ) -> "ActionResponse":
        exec_res = action.execution_result or {}
        self_reported = exec_res.get("tool_success")
        failure_mode = exec_res.get("failure_mode")

        # Derive verdict from the LAST vresult in record (final attempt)
        vresult = record.vresult
        verdict = vresult.status.value if vresult else "PENDING"

        # Contradiction = executor said success but verifier said FAIL
        contradiction = (
            self_reported is True
            and vresult is not None
            and vresult.status == VerificationStatus.FAIL
        )

        return cls(
            action_id=action.action_id,
            action_index=index,
            action_type=action.action_type.value,
            parameters=action.parameters,
            expected_postcondition=action.expected_postcondition,
            retry_count=action.retry_count,
            executor_claim=ExecutorClaimInAction(
                self_reported_success=self_reported,
                failure_mode_detected=str(failure_mode) if failure_mode else None,
            ),
            verification_verdict=verdict,
            final_status=action.status.value,
            contradiction_detected=contradiction,
            evidence_id=action.evidence_id,
        )


# ---------------------------------------------------------------------------
# Failure injection schema
# ---------------------------------------------------------------------------


class FailureInjectionRequest(BaseModel):
    """
    Configure deterministic failure injection and replay the workflow.

    The workflow is re-run with a fresh sandbox store but the same goal.
    A new workflow_id is returned (the original run is preserved).

    mode options
    ------------
    FALSE_SUCCESS         — executor claims success without mutating state
    EXECUTION_FAILURE     — tool raises an error / returns failure
    TEMPORARY_FAILURE     — fails first N times, then succeeds
    PERMISSION_FAILURE    — tool rejected due to missing permissions
    MISSING_EVIDENCE      — evidence collection returns empty
    CONTRADICTORY_STATE   — state contradicts expected postcondition
    """

    tool_name: str = Field(..., description="Which tool to inject failures on")
    mode: str = Field(..., description="Failure mode (see schema description)")
    param_filter: dict[str, Any] = Field(
        default_factory=dict,
        description="Only apply injection when these params match",
    )
    max_fires: int = Field(
        0,
        ge=0,
        description="0 = unlimited; N = inject at most N times (use 1 for one-shot FALSE_SUCCESS)",
    )
    fail_count: int = Field(1, ge=1, description="For TEMPORARY_FAILURE: how many times to fail")
    reason: str = Field("", description="Human-readable reason for the injection")
    max_retries: int = Field(3, ge=0, le=10)
    max_replan_cycles: int = Field(2, ge=0, le=5)
    atomic_completion: bool = False


class FailureInjectionResponse(BaseModel):
    replay_workflow_id: str
    original_workflow_id: str
    goal: str
    injected_failure: str
    final_status: str
    message: str


# ===========================================================================
# Endpoints
# ===========================================================================


@router.post(
    "/run",
    summary="Run a workflow goal",
    description=(
        "Plans and executes the goal through the full evidence-gated loop.\n\n"
        "**Returns immediately** with the completed run result.\n\n"
        "Zero-trust guarantees:\n"
        "- Evidence is ALWAYS collected from the sandbox, never from the LLM\n"
        "- INCONCLUSIVE is NEVER promoted to PASS\n"
        "- Retries are bounded by max_retries"
    ),
)
def run_workflow(body: RunWorkflowRequest) -> WorkflowSummaryResponse:
    if not body.goal.strip():
        raise HTTPException(status_code=400, detail="Goal must not be blank")

    if body.planner_mode == "llm" and not llm_is_configured():
        raise HTTPException(
            status_code=400,
            detail=(
                "planner_mode='llm' requires an LLM API key. "
                "Set OPENAI_API_KEY, ANTHROPIC_API_KEY, or GOOGLE_API_KEY in your environment. "
                "Use planner_mode='mock' for deterministic demo mode."
            ),
        )

    try:
        record = build_and_run(
            goal=body.goal,
            max_retries=body.max_retries,
            max_replan_cycles=body.max_replan_cycles,
            atomic_completion=body.atomic_completion,
            planner_mode=body.planner_mode,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    _app_state.register_record(record)
    log.info(
        "Workflow run completed: id=%s status=%s",
        record.run_id,
        record.result.run.status.value,
    )
    return WorkflowSummaryResponse.from_record(record)


@router.get(
    "/{workflow_id}",
    summary="Get workflow state",
    description="Returns the current (or final) state of a workflow run.",
)
def get_workflow(workflow_id: str) -> WorkflowSummaryResponse:
    record = _app_state.get_record(workflow_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"Workflow '{workflow_id}' not found")
    return WorkflowSummaryResponse.from_record(record)


@router.get(
    "/{workflow_id}/actions",
    summary="Get all actions with executor claims and verification verdicts",
    description=(
        "Returns every action in the workflow with the full **3-way zero-trust distinction**:\n\n"
        "1. **EXECUTOR_CLAIM** — what the tool self-reported (may be false)\n"
        "2. **VERIFICATION_VERDICT** — what the verifier determined (authoritative)\n"
        "3. **FINAL_STATUS** — only SUCCEEDED if verifier said PASS\n\n"
        "`contradiction_detected=true` means the executor lied — it claimed success "
        "but machine evidence proved the action did not complete."
    ),
)
def get_actions(workflow_id: str) -> list[ActionResponse]:
    record = _app_state.get_record(workflow_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"Workflow '{workflow_id}' not found")

    run = record.result.run
    result = record.result
    action_to_record = {r.action.action_id: r for r in result.records}

    return [
        ActionResponse.from_action_and_record(
            action=action,
            record=action_to_record.get(action.action_id),
            index=i,
        )
        for i, action in enumerate(run.actions)
        if action_to_record.get(action.action_id) is not None
    ]


@router.post(
    "/{workflow_id}/failure-injection",
    summary="Re-run this workflow with deterministic failure injection",
    description=(
        "Creates a **fresh isolated run** of the same goal but with failure injection active.\n\n"
        "The original workflow is preserved — a new `replay_workflow_id` is returned.\n\n"
        "Use this to demonstrate:\n"
        "- **FALSE_SUCCESS**: executor lies, verifier catches it\n"
        "- **EXECUTION_FAILURE**: tool fails, recovery retries\n"
        "- **TEMPORARY_FAILURE**: transient failure recovers on retry\n\n"
        "Set `max_fires=1` with `mode=FALSE_SUCCESS` for the classic demo scenario."
    ),
)
def inject_and_replay(
    workflow_id: str,
    body: FailureInjectionRequest,
) -> FailureInjectionResponse:
    original_record = _app_state.get_record(workflow_id)
    if not original_record:
        raise HTTPException(status_code=404, detail=f"Workflow '{workflow_id}' not found")

    # Validate failure mode
    try:
        failure_mode = FailureMode[body.mode.upper()]
    except KeyError:
        valid_modes = [m.name for m in FailureMode]
        raise HTTPException(
            status_code=400,
            detail=f"Unknown failure mode '{body.mode}'. Valid: {valid_modes}",
        )

    # Build injector
    injector = FailureInjector()
    rule = FailureRule(
        tool_name=body.tool_name,
        mode=failure_mode,
        reason=body.reason or f"Injected via API: {failure_mode.name} on {body.tool_name}",
        param_filter=body.param_filter,
        max_fires=body.max_fires,
        fail_count=body.fail_count,
    )
    injector.register(rule)

    injector_config = [body.model_dump()]

    try:
        replay_record = build_and_run(
            goal=original_record.goal,
            max_retries=body.max_retries,
            max_replan_cycles=body.max_replan_cycles,
            atomic_completion=body.atomic_completion,
            injector=injector,
            parent_run_id=workflow_id,
            injector_config=injector_config,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    _app_state.register_record(replay_record)
    log.info(
        "Failure-injection replay: original=%s replay=%s mode=%s status=%s",
        workflow_id,
        replay_record.run_id,
        failure_mode.name,
        replay_record.result.run.status.value,
    )

    return FailureInjectionResponse(
        replay_workflow_id=replay_record.run_id,
        original_workflow_id=workflow_id,
        goal=original_record.goal,
        injected_failure=f"{failure_mode.name} on '{body.tool_name}'",
        final_status=replay_record.result.run.status.value,
        message=(
            f"Replay completed with {failure_mode.name} injected on '{body.tool_name}'. "
            f"Fetch /workflow/{replay_record.run_id}/evidence to see the 3-way verification."
        ),
    )
