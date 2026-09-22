"""
scenarios/scenario4_rollback.py

SCENARIO 4 — ATOMIC ROLLBACK

Workflow steps:
  1. create_project  → VERIFIED (project exists in sandbox)
  2. add_member      → VERIFIED (member exists in sandbox)
  3. set_permission  → FAIL    (injected permanent failure)

With atomic_completion=True enabled:
  ANY failure triggers a full rollback of all prior verified actions.

Rollback steps (via journal rewind):
  Journal entry for add_member   → undone
  Journal entry for create_project → undone

Post-rollback state:
  project_exists  = False    (verified)
  member_exists   = False    (verified)

Final status: ROLLED_BACK

Every action produces audit events. The audit log includes a
rollback event showing all undone actions and the clean sandbox state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agents.coordinator import CoordinatorConfig
from core.audit import AuditEvent, WorkflowSummary, get_audit_log, get_workflow_summary
from core.evidence import EvidenceCollector
from core.models import Action, ActionType, ActionStatus, WorkflowStatus
from core.verifier import VerificationEngine
from sandbox.failure_injection import FailureInjector, FailureMode, FailureRule
from sandbox.state_store import SandboxStateStore
from scenarios.runner import ScenarioResult, ScenarioRunner

GOAL = (
    "Create project Hackathon Alpha, add Harshit as a member, "
    "make the project private, and generate a report."
)

SCENARIO_NAME = "Scenario 4 — Atomic Rollback"

_PROJECT_ID = "hackathon-alpha"
_MEMBER_ID  = "harshit"


def _build_injector() -> FailureInjector:
    """
    Permanent EXECUTION_FAILURE on set_permission.

    create_project and add_member succeed; set_permission always fails.
    atomic_completion=True triggers rollback of both verified actions.
    """
    injector = FailureInjector()
    injector.register(FailureRule(
        tool_name="set_permission",
        mode=FailureMode.EXECUTION_FAILURE,
        reason=(
            "INJECTED: Permission service unavailable. "
            "Cannot modify project visibility — rolling back entire workflow."
        ),
        param_filter={"project_id": _PROJECT_ID},
        fail_count=999,  # permanent
    ))
    return injector


def _verify_rollback_state(store: SandboxStateStore) -> list[dict[str, Any]]:
    """
    Independently verify that the sandbox is clean after rollback.

    Runs the VerificationEngine against the post-rollback state to produce
    machine-checkable evidence that entities no longer exist.
    """
    verifications = []

    # Verify project no longer exists
    project_exists = store.get_project(_PROJECT_ID) is not None
    verifications.append({
        "entity": "project",
        "entity_id": _PROJECT_ID,
        "check": "exists",
        "expected": False,
        "observed": project_exists,
        "verified": project_exists is False,
        "verdict": "PASS" if not project_exists else "FAIL",
        "note": (
            "Project was rolled back — does not exist in sandbox."
            if not project_exists
            else "ERROR: Project still exists after rollback!"
        ),
    })

    # Verify member no longer exists
    member_exists = store.get_member(_PROJECT_ID, _MEMBER_ID) is not None
    verifications.append({
        "entity": "member",
        "entity_id": f"{_PROJECT_ID}:{_MEMBER_ID}",
        "check": "member_exists",
        "expected": False,
        "observed": member_exists,
        "verified": member_exists is False,
        "verdict": "PASS" if not member_exists else "FAIL",
        "note": (
            "Member was rolled back — does not exist in sandbox."
            if not member_exists
            else "ERROR: Member still exists after rollback!"
        ),
    })

    return verifications


def run() -> ScenarioResult:
    """
    Execute the atomic-rollback scenario deterministically.

    Returns a ScenarioResult with:
      - final_status == ROLLED_BACK
      - rollback_verifications: project_exists=False, member_exists=False
      - audit_log contains: execute→evidence→verify(PASS)×2,
                            execute→evidence→verify(FAIL),
                            recovery_agent decision,
                            atomic_rollback event
      - summary.rolled_back_actions == 2 (project + member)
    """
    injector = _build_injector()
    runner = ScenarioRunner(
        injector=injector,
        config=CoordinatorConfig(
            max_retries=0,           # fail fast — no retries in atomic mode
            max_replan_cycles=0,
            atomic_completion=True,  # ← enables rollback on any failure
        ),
    )
    result = runner.run(
        goal=GOAL,
        scenario_name=SCENARIO_NAME,
        expected_status="ROLLED_BACK",
    )

    # Post-rollback: verify sandbox is clean
    rollback_verifications = _verify_rollback_state(runner.store)
    result.rollback_verifications = rollback_verifications

    return result
