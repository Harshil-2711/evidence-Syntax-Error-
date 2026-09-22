"""
core/state.py

Deterministic workflow state machine.

All state transitions are explicit and validated against an allowed-transitions
table. No transition can be made outside the allowed set, preventing
accidental status corruption (e.g., jumping from FAILED → COMPLETED).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

from core.models import ActionStatus, WorkflowRun, WorkflowStatus

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Allowed state transitions (zero-trust: only listed transitions are valid)
# ---------------------------------------------------------------------------

ALLOWED_TRANSITIONS: dict[WorkflowStatus, set[WorkflowStatus]] = {
    WorkflowStatus.PLANNED: {
        WorkflowStatus.EXECUTING,
        WorkflowStatus.BLOCKED,
    },
    WorkflowStatus.EXECUTING: {
        WorkflowStatus.AWAITING_EVIDENCE,
        WorkflowStatus.FAILED,
        WorkflowStatus.BLOCKED,
        WorkflowStatus.COMPLETED,
    },
    WorkflowStatus.AWAITING_EVIDENCE: {
        WorkflowStatus.VERIFYING,
        WorkflowStatus.FAILED,          # evidence never arrived
    },
    WorkflowStatus.VERIFYING: {
        WorkflowStatus.VERIFIED,
        WorkflowStatus.FAILED,
    },
    WorkflowStatus.VERIFIED: {
        WorkflowStatus.EXECUTING,       # proceed to next action
        WorkflowStatus.COMPLETED,       # last action verified
    },
    WorkflowStatus.FAILED: {
        WorkflowStatus.RECOVERING,
        WorkflowStatus.BLOCKED,         # unrecoverable
    },
    WorkflowStatus.RECOVERING: {
        WorkflowStatus.ROLLED_BACK,
        WorkflowStatus.REPLANNING,
        WorkflowStatus.EXECUTING,       # direct retry without rollback
        WorkflowStatus.VERIFIED,        # re-query resolved directly to PASS
        WorkflowStatus.BLOCKED,
    },
    WorkflowStatus.REPLANNING: {
        WorkflowStatus.EXECUTING,
        WorkflowStatus.BLOCKED,
    },
    WorkflowStatus.ROLLED_BACK: {
        WorkflowStatus.REPLANNING,
        WorkflowStatus.BLOCKED,         # abort after rollback when replan exhausted
        WorkflowStatus.COMPLETED,       # rolled back to clean state = done
    },
    WorkflowStatus.BLOCKED: set(),      # terminal — no outgoing transitions
    WorkflowStatus.COMPLETED: set(),    # terminal
}


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------


class WorkflowStateMachine:
    """
    Manages lifecycle transitions for a WorkflowRun.

    Raises TransitionError for invalid moves, ensuring the system can never
    silently skip to a misleading state (e.g., COMPLETED after a silent FAIL).
    """

    def __init__(self, run: WorkflowRun) -> None:
        self.run = run

    # ------------------------------------------------------------------
    # Transition helpers
    # ------------------------------------------------------------------

    def transition(self, new_status: WorkflowStatus, reason: str = "") -> None:
        """
        Apply a validated status transition to the workflow run.

        Args:
            new_status: The target WorkflowStatus.
            reason:     Optional human-readable explanation (stored in metadata).

        Raises:
            TransitionError: If the transition is not in ALLOWED_TRANSITIONS.
        """
        current = self.run.status
        allowed = ALLOWED_TRANSITIONS.get(current, set())

        if new_status not in allowed:
            raise TransitionError(
                f"Invalid transition: {current.value} → {new_status.value}. "
                f"Allowed from {current.value}: "
                f"{[s.value for s in allowed] or 'none (terminal state)'}"
            )

        self.run.status = new_status
        self.run.touch()

        # Record the transition in metadata for audit
        history: list[dict] = self.run.metadata.setdefault("transition_history", [])
        history.append(
            {
                "from": current.value,
                "to": new_status.value,
                "reason": reason,
                "at": datetime.now(timezone.utc).isoformat(),
            }
        )

    # ------------------------------------------------------------------
    # Convenience transition methods
    # ------------------------------------------------------------------

    def start_execution(self) -> None:
        self.transition(WorkflowStatus.EXECUTING, "workflow execution started")

    def await_evidence(self) -> None:
        self.transition(WorkflowStatus.AWAITING_EVIDENCE, "action dispatched, awaiting sandbox evidence")

    def start_verification(self) -> None:
        self.transition(WorkflowStatus.VERIFYING, "evidence received, starting deterministic verification")

    def mark_verified(self) -> None:
        self.transition(WorkflowStatus.VERIFIED, "all postconditions satisfied")

    def mark_failed(self, reason: str = "") -> None:
        self.transition(WorkflowStatus.FAILED, reason or "action failed or evidence contradicts postcondition")

    def start_recovery(self) -> None:
        self.transition(WorkflowStatus.RECOVERING, "entering recovery mode")

    def start_replanning(self) -> None:
        self.transition(WorkflowStatus.REPLANNING, "generating revised plan")

    def mark_rolled_back(self) -> None:
        self.transition(WorkflowStatus.ROLLED_BACK, "compensating actions applied, state restored")

    def block(self, reason: str = "") -> None:
        self.transition(WorkflowStatus.BLOCKED, reason or "unrecoverable condition")

    def complete(self) -> None:
        self.transition(WorkflowStatus.COMPLETED, "all actions verified successfully")

    # ------------------------------------------------------------------
    # Derived queries
    # ------------------------------------------------------------------

    @property
    def is_terminal(self) -> bool:
        return self.run.status in {WorkflowStatus.COMPLETED, WorkflowStatus.BLOCKED}

    @property
    def can_retry(self) -> bool:
        return self.run.status == WorkflowStatus.FAILED

    def all_actions_done(self) -> bool:
        """True when every action in the plan has reached a terminal action status."""
        terminal = {ActionStatus.SUCCEEDED, ActionStatus.FAILED, ActionStatus.ROLLED_BACK, ActionStatus.SKIPPED}
        return all(a.status in terminal for a in self.run.actions)


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------


class TransitionError(Exception):
    """Raised when an invalid workflow state transition is attempted."""
