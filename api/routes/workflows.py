"""
api/routes/workflows.py

Workflow routes — trigger, inspect, and list evidence-gated workflow runs.

Endpoints
---------
POST /workflows/run        — Parse goal, plan, execute, return full result
GET  /workflows/           — List all workflow runs (summary)
GET  /workflows/{run_id}   — Get full detail of one run
GET  /workflows/{run_id}/actions  — Actions for a run
GET  /workflows/{run_id}/evidence — Evidence log for a run
GET  /workflows/{run_id}/audit    — State-machine transition history
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from agents.planner import Planner, PlannerConfig, PlannerMode
from api.dependencies import AppState, get_app_state, make_coordinator
from api.schemas import (
    ActionResponse,
    EvidenceResponse,
    RunWorkflowRequest,
    RunWorkflowResponse,
    WorkflowDetailResponse,
    WorkflowSummaryResponse,
)
from core.models import ActionStatus

router = APIRouter(prefix="/workflows", tags=["Workflows"])

AppStateDep = Annotated[AppState, Depends(get_app_state)]


# ---------------------------------------------------------------------------
# POST /workflows/run
# ---------------------------------------------------------------------------


@router.post(
    "/run",
    response_model=RunWorkflowResponse,
    status_code=status.HTTP_200_OK,
    summary="Run a workflow from a natural-language goal",
    description=(
        "Plans and executes a multi-step workflow. Every action is verified "
        "against sandbox evidence before being marked complete. "
        "The executor's own success claim is NEVER sufficient — the verifier "
        "inspects actual sandbox state independently."
    ),
)
def run_workflow(
    body: RunWorkflowRequest,
    state: AppStateDep,
) -> RunWorkflowResponse:
    # 1. Plan
    planner_mode = PlannerMode.LLM if body.planner_mode == "llm" else PlannerMode.MOCK
    planner = Planner(PlannerConfig(mode=planner_mode))

    try:
        plan = planner.plan(body.goal)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not plan.steps:
        raise HTTPException(
            status_code=422,
            detail=f"Planner could not derive any actions from goal: '{body.goal}'",
        )

    # 2. Execute with evidence gating
    coordinator = make_coordinator(
        store=state.store,
        max_retries=body.max_retries,
        max_replan_cycles=body.max_replan_cycles,
    )
    result = coordinator.execute_plan(plan)

    # 3. Persist result
    state.register_result(result)

    # 4. Build response
    run = result.run
    succeeded = sum(1 for a in run.actions if a.status == ActionStatus.SUCCEEDED)
    failed = sum(1 for a in run.actions if a.status == ActionStatus.FAILED)

    return RunWorkflowResponse(
        run_id=run.run_id,
        status=run.status.value,
        summary=result.summary,
        goal=body.goal,
        action_count=len(run.actions),
        succeeded_count=succeeded,
        failed_count=failed,
        evidence_count=len(run.evidence_log),
        detail=WorkflowDetailResponse.from_result(result),
    )


# ---------------------------------------------------------------------------
# GET /workflows/
# ---------------------------------------------------------------------------


@router.get(
    "/",
    response_model=list[WorkflowSummaryResponse],
    summary="List all workflow runs",
)
def list_workflows(state: AppStateDep) -> list[WorkflowSummaryResponse]:
    summaries = []
    for result in state.list_results():
        run = result.run
        succeeded = sum(1 for a in run.actions if a.status == ActionStatus.SUCCEEDED)
        failed = sum(1 for a in run.actions if a.status == ActionStatus.FAILED)
        summaries.append(
            WorkflowSummaryResponse(
                run_id=run.run_id,
                name=run.name,
                status=run.status.value,
                action_count=len(run.actions),
                succeeded_count=succeeded,
                failed_count=failed,
                created_at=run.created_at.isoformat(),
                updated_at=run.updated_at.isoformat(),
            )
        )
    return summaries


# ---------------------------------------------------------------------------
# GET /workflows/{run_id}
# ---------------------------------------------------------------------------


@router.get(
    "/{run_id}",
    response_model=WorkflowDetailResponse,
    summary="Get full detail of a workflow run",
)
def get_workflow(run_id: str, state: AppStateDep) -> WorkflowDetailResponse:
    result = state.get_result(run_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Workflow run '{run_id}' not found.")
    return WorkflowDetailResponse.from_result(result)


# ---------------------------------------------------------------------------
# GET /workflows/{run_id}/actions
# ---------------------------------------------------------------------------


@router.get(
    "/{run_id}/actions",
    response_model=list[ActionResponse],
    summary="Get all actions for a workflow run",
)
def get_workflow_actions(run_id: str, state: AppStateDep) -> list[ActionResponse]:
    result = state.get_result(run_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Workflow run '{run_id}' not found.")
    return [ActionResponse.from_action(a) for a in result.run.actions]


# ---------------------------------------------------------------------------
# GET /workflows/{run_id}/evidence
# ---------------------------------------------------------------------------


@router.get(
    "/{run_id}/evidence",
    response_model=list[EvidenceResponse],
    summary="Get evidence log for a workflow run",
    description=(
        "Returns the machine-collected evidence for every action. "
        "Evidence is ALWAYS sourced from sandbox state — never fabricated."
    ),
)
def get_workflow_evidence(run_id: str, state: AppStateDep) -> list[EvidenceResponse]:
    result = state.get_result(run_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Workflow run '{run_id}' not found.")
    return [EvidenceResponse.from_evidence(e) for e in result.run.evidence_log]


# ---------------------------------------------------------------------------
# GET /workflows/{run_id}/audit
# ---------------------------------------------------------------------------


@router.get(
    "/{run_id}/audit",
    response_model=list[dict],
    summary="Get state-machine transition history for a workflow run",
    description=(
        "Returns the ordered list of all workflow state transitions with "
        "timestamps and reasons. Provides a full audit trail."
    ),
)
def get_workflow_audit(run_id: str, state: AppStateDep) -> list[dict]:
    result = state.get_result(run_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Workflow run '{run_id}' not found.")
    return result.run.metadata.get("transition_history", [])
