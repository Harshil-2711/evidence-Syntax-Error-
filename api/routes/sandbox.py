"""
api/routes/sandbox.py

Sandbox inspection and control routes.

Endpoints
---------
GET  /sandbox/state          — Full snapshot of all sandbox entities
GET  /sandbox/journal        — Raw mutation journal (audit log)
POST /sandbox/reset          — Wipe all state (requires confirm=true)
GET  /sandbox/projects       — List all projects
GET  /sandbox/projects/{pid} — Get single project
GET  /sandbox/projects/{pid}/members     — List members of a project
GET  /sandbox/projects/{pid}/files       — List files in a project
GET  /sandbox/projects/{pid}/permissions — List permissions for a project
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, status

from api.dependencies import AppState, get_app_state
from api.schemas import ResetSandboxRequest, SandboxStateResponse

router = APIRouter(prefix="/sandbox", tags=["Sandbox"])

AppStateDep = Annotated[AppState, Depends(get_app_state)]


# ---------------------------------------------------------------------------
# GET /sandbox/state
# ---------------------------------------------------------------------------


@router.get(
    "/state",
    response_model=SandboxStateResponse,
    summary="Full snapshot of all sandbox state",
)
def get_sandbox_state(state: AppStateDep) -> SandboxStateResponse:
    snap = state.store.snapshot()
    return SandboxStateResponse(
        projects=snap["projects"],
        members=snap["members"],
        permissions=snap["permissions"],
        files=snap["files"],
        notifications=snap["notifications"],
        reports=snap["reports"],
        journal_length=len(state.store.journal),
    )


# ---------------------------------------------------------------------------
# GET /sandbox/journal
# ---------------------------------------------------------------------------


@router.get(
    "/journal",
    response_model=list[dict[str, Any]],
    summary="Raw mutation journal (ordered audit log of all state changes)",
)
def get_journal(state: AppStateDep) -> list[dict[str, Any]]:
    return [
        {
            "operation": e.operation,
            "entity_type": e.entity_type,
            "entity_id": e.entity_id,
            "snapshot_before": e.snapshot_before,
            "snapshot_after": e.snapshot_after,
            "timestamp": e.timestamp.isoformat(),
        }
        for e in state.store.journal
    ]


# ---------------------------------------------------------------------------
# POST /sandbox/reset
# ---------------------------------------------------------------------------


@router.post(
    "/reset",
    status_code=status.HTTP_200_OK,
    summary="Reset all sandbox state and workflow history",
    description="Destructive — wipes all projects, members, files, evidence, and run history.",
)
def reset_sandbox(body: ResetSandboxRequest, state: AppStateDep) -> dict[str, str]:
    if not body.confirm:
        raise HTTPException(
            status_code=400,
            detail="Reset requires confirm=true in request body.",
        )
    state.reset()
    return {"status": "reset", "message": "All sandbox state and workflow history cleared."}


# ---------------------------------------------------------------------------
# GET /sandbox/projects
# ---------------------------------------------------------------------------


@router.get(
    "/projects",
    response_model=list[dict[str, Any]],
    summary="List all projects in the sandbox",
)
def list_projects(state: AppStateDep) -> list[dict[str, Any]]:
    snap = state.store.snapshot()
    return list(snap["projects"].values())


# ---------------------------------------------------------------------------
# GET /sandbox/projects/{project_id}
# ---------------------------------------------------------------------------


@router.get(
    "/projects/{project_id}",
    response_model=dict[str, Any],
    summary="Get a single project by ID",
)
def get_project(project_id: str, state: AppStateDep) -> dict[str, Any]:
    project = state.store.get_project(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found.")
    return project


# ---------------------------------------------------------------------------
# GET /sandbox/projects/{project_id}/members
# ---------------------------------------------------------------------------


@router.get(
    "/projects/{project_id}/members",
    response_model=list[dict[str, Any]],
    summary="List members of a project",
)
def get_members(project_id: str, state: AppStateDep) -> list[dict[str, Any]]:
    project = state.store.get_project(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found.")
    snap = state.store.snapshot()
    return list(snap["members"].get(project_id, {}).values())


# ---------------------------------------------------------------------------
# GET /sandbox/projects/{project_id}/files
# ---------------------------------------------------------------------------


@router.get(
    "/projects/{project_id}/files",
    response_model=list[dict[str, Any]],
    summary="List files in a project",
)
def get_files(project_id: str, state: AppStateDep) -> list[dict[str, Any]]:
    project = state.store.get_project(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found.")
    snap = state.store.snapshot()
    return list(snap["files"].get(project_id, {}).values())


# ---------------------------------------------------------------------------
# GET /sandbox/projects/{project_id}/permissions
# ---------------------------------------------------------------------------


@router.get(
    "/projects/{project_id}/permissions",
    response_model=dict[str, str],
    summary="Get permission roles for all users in a project",
)
def get_permissions(project_id: str, state: AppStateDep) -> dict[str, str]:
    project = state.store.get_project(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found.")
    snap = state.store.snapshot()
    return snap["permissions"].get(project_id, {})
