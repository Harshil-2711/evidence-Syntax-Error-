"""
core/audit.py

Audit event model and workflow summary utilities.

Every significant moment in an evidence-gated workflow run produces an
AuditEvent.  Events are derived deterministically from the ActionRecord
objects already stored in CoordinatorResult — no extra state required.

Public API
----------
  AuditEvent          — Immutable record of one agent action
  WorkflowSummary     — Aggregate metrics for a completed run
  get_audit_log(result)       -> list[AuditEvent]
  get_workflow_summary(result) -> WorkflowSummary
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from core.models import ActionStatus, WorkflowStatus

if TYPE_CHECKING:
    from agents.coordinator import CoordinatorResult


# ---------------------------------------------------------------------------
# AuditEvent
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AuditEvent:
    """
    An immutable record of one agent action within a workflow run.

    Fields
    ------
    timestamp:         UTC time the event occurred.
    agent:             Which agent emitted the event.
                       One of: coordinator, executor, evidence_collector,
                                verifier, recovery_agent, replanner, rollback
    action:            What the agent did (verb).
    parameters:        Relevant parameters or inputs.
    execution_result:  Tool's self-reported result (NOT verified truth).
    evidence:          Observed state + expected state collected from sandbox.
    verification:      Verifier verdict: PASS | FAIL | INCONCLUSIVE | None.
    decision:          Recovery or coordination decision made.
    retry_number:      0 = first attempt; 1 = first retry; etc.
    action_id:         ID of the Action this event relates to (if any).
    action_type:       ActionType.value string (e.g., "create_project").
    """

    timestamp: datetime
    agent: str
    action: str
    parameters: dict[str, Any]
    execution_result: dict[str, Any] | None
    evidence: dict[str, Any] | None
    verification: str | None
    decision: str | None
    retry_number: int
    action_id: str | None = None
    action_type: str | None = None


# ---------------------------------------------------------------------------
# WorkflowSummary
# ---------------------------------------------------------------------------


@dataclass
class WorkflowSummary:
    """
    Aggregate metrics for a completed (or terminated) workflow run.

    Fields
    ------
    goal:               The original natural-language goal.
    total_actions:      Total number of planned actions.
    verified_actions:   Actions that reached SUCCEEDED status.
    failed_actions:     Actions that reached FAILED status.
    rolled_back_actions: Actions that were undone by rollback.
    recovery_count:     Number of times recovery was triggered.
    retry_count:        Total retry attempts across all actions.
    evidence_coverage:  Fraction of actions that produced at least one Evidence.
    final_status:       Terminal workflow status string.
    blocked_reason:     Why the workflow was blocked (if applicable).
    """

    goal: str
    total_actions: int
    verified_actions: int
    failed_actions: int
    rolled_back_actions: int
    recovery_count: int
    retry_count: int
    evidence_coverage: float
    final_status: str
    blocked_reason: str | None = None

    @property
    def all_verified(self) -> bool:
        return self.verified_actions == self.total_actions

    @property
    def any_failed(self) -> bool:
        return self.failed_actions > 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "total_actions": self.total_actions,
            "verified_actions": self.verified_actions,
            "failed_actions": self.failed_actions,
            "rolled_back_actions": self.rolled_back_actions,
            "recovery_count": self.recovery_count,
            "retry_count": self.retry_count,
            "evidence_coverage": round(self.evidence_coverage, 3),
            "final_status": self.final_status,
            "blocked_reason": self.blocked_reason,
        }


# ---------------------------------------------------------------------------
# get_audit_log
# ---------------------------------------------------------------------------


def get_audit_log(result: "CoordinatorResult") -> list[AuditEvent]:
    """
    Build a structured audit log from a CoordinatorResult.

    One or more AuditEvents are generated per ActionRecord:
      1. executor       — tool dispatch + self-reported outcome
      2. evidence_collector — observed sandbox state
      3. verifier       — PASS / FAIL / INCONCLUSIVE verdict
      4. recovery_agent — recovery decision (if triggered)
      5. rollback       — if atomic rollback was performed

    Events are ordered chronologically by action sequence.
    """
    events: list[AuditEvent] = []
    run = result.run

    for record in result.records:
        action = record.action

        # Collect all attempts to emit: historical ones first, then the final state
        # (unless the final state was already snapshotted — avoid duplication)
        attempts_to_emit = list(record.attempt_history)  # failed/retried attempts

        # Always add the final state as a synthetic snapshot if it wasn't captured
        # A snapshot is captured on FAIL; PASS results are NOT snapshotted, so add them
        final_already_snapshotted = any(
            s.vresult is record.vresult for s in attempts_to_emit
        )
        if not final_already_snapshotted:
            from agents.coordinator import AttemptSnapshot
            attempts_to_emit.append(AttemptSnapshot(
                attempt_number=record.attempts,
                outcome=record.outcome,
                evidence=record.evidence,
                vresult=record.vresult,
                recovery=None,  # no recovery on successful final attempt
            ))

        for snap in attempts_to_emit:
            retry_number = action.retry_count if snap.vresult is record.vresult else max(0, snap.attempt_number - 1)

            # ── 1. Executor event ──────────────────────────────────────
            if snap.outcome is not None:
                events.append(AuditEvent(
                    timestamp=action.timestamp,
                    agent="executor",
                    action=f"execute:{action.action_type.value}",
                    parameters=action.parameters,
                    execution_result={
                        "self_reported_success": snap.outcome.success,
                        "data": snap.outcome.tool_result.data,
                        "error": snap.outcome.error,
                        "failure_mode": action.execution_result.get("failure_mode")
                        if action.execution_result else None,
                    },
                    evidence=None,
                    verification=None,
                    decision="EXECUTOR_CLAIM_NOT_PROOF: verifier will check independently",
                    retry_number=retry_number,
                    action_id=action.action_id,
                    action_type=action.action_type.value,
                ))

            # ── 2. Evidence collector event ────────────────────────────
            if snap.evidence is not None:
                ev = snap.evidence
                events.append(AuditEvent(
                    timestamp=ev.timestamp,
                    agent="evidence_collector",
                    action="collect_evidence",
                    parameters={"source": ev.source},
                    execution_result=None,
                    evidence={
                        "observed_state": ev.observed_state,
                        "expected_state": ev.expected_state,
                    },
                    verification=ev.verification_status.value,
                    decision=None,
                    retry_number=retry_number,
                    action_id=action.action_id,
                    action_type=action.action_type.value,
                ))

            # ── 3. Verifier event ──────────────────────────────────────
            if snap.vresult is not None:
                vr = snap.vresult
                ts = snap.evidence.timestamp if snap.evidence else action.timestamp
                events.append(AuditEvent(
                    timestamp=ts,
                    agent="verifier",
                    action="verify",
                    parameters={"postconditions": list(action.expected_postcondition.keys())},
                    execution_result=None,
                    evidence={
                        "verdict": vr.status.value,
                        "reasons": vr.reasons,
                        "observed": snap.evidence.observed_state if snap.evidence else {},
                        "expected": action.expected_postcondition,
                    },
                    verification=vr.status.value,
                    decision=_verdict_explanation(vr.status.value, vr.reasons),
                    retry_number=retry_number,
                    action_id=action.action_id,
                    action_type=action.action_type.value,
                ))

            # ── 4. Recovery event ──────────────────────────────────────
            if snap.recovery is not None:
                rd = snap.recovery
                events.append(AuditEvent(
                    timestamp=datetime.now(timezone.utc),
                    agent="recovery_agent",
                    action="decide_recovery",
                    parameters={
                        "category": rd.category.value,
                        "retries_left": rd.metadata.get("retries_left", "?"),
                    },
                    execution_result=None,
                    evidence=None,
                    verification=None,
                    decision=f"STRATEGY={rd.strategy.value}: {rd.reason}",
                    retry_number=retry_number,
                    action_id=action.action_id,
                    action_type=action.action_type.value,
                ))

    # ── 5. Rollback event (one global event if rollback occurred) ─────
    if run.status == WorkflowStatus.ROLLED_BACK:
        rolled = [a for a in run.actions if a.status == ActionStatus.ROLLED_BACK]
        events.append(AuditEvent(
            timestamp=run.updated_at,
            agent="rollback",
            action="atomic_rollback",
            parameters={
                "rolled_back_actions": [a.action_type.value for a in rolled],
            },
            execution_result=None,
            evidence=None,
            verification=None,
            decision=(
                f"All {len(rolled)} verified action(s) undone. "
                "Sandbox restored to pre-run state."
            ),
            retry_number=0,
            action_id=None,
            action_type=None,
        ))

    return events


# ---------------------------------------------------------------------------
# get_workflow_summary
# ---------------------------------------------------------------------------


def get_workflow_summary(result: "CoordinatorResult") -> WorkflowSummary:
    """
    Compute aggregate metrics from a CoordinatorResult.

    Returns a WorkflowSummary with counts, coverage, and final status.
    """
    run = result.run
    actions = run.actions

    succeeded = sum(1 for a in actions if a.status == ActionStatus.SUCCEEDED)
    failed = sum(1 for a in actions if a.status == ActionStatus.FAILED)
    rolled_back = sum(1 for a in actions if a.status == ActionStatus.ROLLED_BACK)
    recovery_count = sum(1 for r in result.records if r.recovery is not None)
    retry_count = sum(a.retry_count for a in actions)

    # Evidence coverage = fraction of actions that produced at least one evidence
    action_ids_with_evidence = {e.action_id for e in run.evidence_log}
    covered = sum(1 for a in actions if a.action_id in action_ids_with_evidence)
    coverage = covered / len(actions) if actions else 0.0

    # Extract blocked reason from transition history
    history = run.metadata.get("transition_history", [])
    blocked_reason: str | None = None
    for entry in reversed(history):
        if entry.get("to") == "BLOCKED":
            blocked_reason = entry.get("reason")
            break

    return WorkflowSummary(
        goal=run.name,
        total_actions=len(actions),
        verified_actions=succeeded,
        failed_actions=failed,
        rolled_back_actions=rolled_back,
        recovery_count=recovery_count,
        retry_count=retry_count,
        evidence_coverage=coverage,
        final_status=run.status.value,
        blocked_reason=blocked_reason,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _verdict_explanation(status: str, reasons: list[str]) -> str:
    if status == "PASS":
        return "All expected postconditions satisfied by machine-observed state."
    if status == "FAIL":
        first = reasons[0] if reasons else "condition not met"
        return f"VERIFICATION FAILED: {first}"
    if status == "INCONCLUSIVE":
        return "Evidence collection incomplete — cannot verify postconditions."
    return status
