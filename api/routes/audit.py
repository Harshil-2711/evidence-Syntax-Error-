"""
api/routes/audit.py

Audit endpoint — GET /workflow/{id}/audit

Returns the complete chronological audit trail for a workflow run.

Each AuditEvent records one agent action with:
  - timestamp
  - agent (executor | evidence_collector | verifier | recovery_agent | rollback)
  - action
  - parameters
  - execution_result
  - evidence
  - verification
  - decision
  - retry_number

The audit trail is the tamper-evident record of everything the system did.
It is built deterministically from ActionRecord objects — not from logs.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from api.dependencies import _app_state

router = APIRouter(prefix="/workflow", tags=["Audit"])


# ===========================================================================
# Schema
# ===========================================================================


class AuditEventResponse(BaseModel):
    """
    One entry in the workflow audit trail.

    agent identifies which component emitted the event:
      executor          — tool dispatch and self-report
      evidence_collector — sandbox snapshot
      verifier          — PASS/FAIL/INCONCLUSIVE verdict
      recovery_agent    — recovery strategy decision
      rollback          — atomic rollback event (Scenario 4)
      coordinator       — workflow-level state transitions
    """

    timestamp: str
    agent: str
    action: str
    parameters: dict[str, Any]
    execution_result: dict[str, Any] | None
    evidence: dict[str, Any] | None
    verification: str | None = Field(
        description="PASS | FAIL | INCONCLUSIVE | None (if not a verification event)"
    )
    decision: str | None
    retry_number: int
    action_id: str | None
    action_type: str | None

    @classmethod
    def from_audit_event(cls, ev: Any) -> "AuditEventResponse":
        ts = ev.timestamp
        if isinstance(ts, datetime):
            ts_str = ts.isoformat()
        else:
            ts_str = str(ts)
        return cls(
            timestamp=ts_str,
            agent=ev.agent,
            action=ev.action,
            parameters=ev.parameters,
            execution_result=ev.execution_result,
            evidence=ev.evidence,
            verification=ev.verification,
            decision=ev.decision,
            retry_number=ev.retry_number,
            action_id=ev.action_id,
            action_type=ev.action_type,
        )


class AuditSummaryResponse(BaseModel):
    executor_events: int
    evidence_events: int
    verifier_events: int
    recovery_events: int
    rollback_events: int
    total_events: int
    pass_verdicts: int
    fail_verdicts: int
    inconclusive_verdicts: int
    contradictions_caught: int = Field(
        description="FAIL verdicts where executor had claimed success"
    )


class AuditTrailResponse(BaseModel):
    """
    Complete audit trail for a workflow run.

    Events are ordered chronologically by action sequence and then
    by agent pipeline order: executor → evidence_collector → verifier
    → recovery_agent (if triggered) → rollback (if triggered).

    The state_machine_history field shows every workflow-level state
    transition with timestamps and reasons — the tamper-evident record
    of everything the state machine did.
    """

    workflow_id: str
    goal: str
    final_status: str
    events: list[AuditEventResponse]
    summary: AuditSummaryResponse
    state_machine_history: list[dict[str, Any]] = Field(
        description="Ordered list of state machine transitions with reasons."
    )


# ===========================================================================
# Endpoint
# ===========================================================================


@router.get(
    "/{workflow_id}/audit",
    summary="Get the complete audit trail",
    description=(
        "Returns every agent event in chronological order.\n\n"
        "**Agent pipeline per action:**\n"
        "`executor` → `evidence_collector` → `verifier` → `recovery_agent` (if failed)\n\n"
        "**Event types:**\n"
        "- `executor` — tool dispatch; shows self-reported success\n"
        "- `evidence_collector` — sandbox snapshot; shows observed state\n"
        "- `verifier` — PASS/FAIL/INCONCLUSIVE with per-condition reasons\n"
        "- `recovery_agent` — strategy chosen when verification fails\n"
        "- `rollback` — atomic rollback event (Scenario 4)\n\n"
        "**state_machine_history** shows every workflow-level transition — "
        "a complete, tamper-evident record."
    ),
    response_model=AuditTrailResponse,
)
def get_audit(workflow_id: str) -> AuditTrailResponse:
    record = _app_state.get_record(workflow_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"Workflow '{workflow_id}' not found")

    events = [AuditEventResponse.from_audit_event(ev) for ev in record.audit_log]

    # Compute summary counts
    executor_count = sum(1 for e in events if e.agent == "executor")
    evidence_count = sum(1 for e in events if e.agent == "evidence_collector")
    verifier_count = sum(1 for e in events if e.agent == "verifier")
    recovery_count = sum(1 for e in events if e.agent == "recovery_agent")
    rollback_count = sum(1 for e in events if e.agent == "rollback")

    pass_count = sum(1 for e in events if e.verification == "PASS")
    fail_count = sum(1 for e in events if e.verification == "FAIL")
    inconc_count = sum(1 for e in events if e.verification == "INCONCLUSIVE")

    # Count contradictions (verifier FAIL where executor had said success)
    # These appear as verifier events with "VERIFICATION FAILED" in decision
    contradiction_count = 0
    action_to_record = {r.action.action_id: r for r in record.result.records}
    for ev in events:
        if ev.agent == "verifier" and ev.verification == "FAIL" and ev.action_id:
            rec = action_to_record.get(ev.action_id)
            if rec:
                # Find matching snapshot
                for snap in rec.attempt_history:
                    if snap.vresult and snap.vresult.status.value == "FAIL":
                        if snap.outcome and snap.outcome.success:
                            contradiction_count += 1
                            break

    summary_obj = AuditSummaryResponse(
        executor_events=executor_count,
        evidence_events=evidence_count,
        verifier_events=verifier_count,
        recovery_events=recovery_count,
        rollback_events=rollback_count,
        total_events=len(events),
        pass_verdicts=pass_count,
        fail_verdicts=fail_count,
        inconclusive_verdicts=inconc_count,
        contradictions_caught=contradiction_count,
    )

    # State machine transition history from run metadata
    history = record.result.run.metadata.get("transition_history", [])

    return AuditTrailResponse(
        workflow_id=workflow_id,
        goal=record.goal,
        final_status=record.result.run.status.value,
        events=events,
        summary=summary_obj,
        state_machine_history=history,
    )
