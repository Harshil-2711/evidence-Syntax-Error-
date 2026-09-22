"""
core/models.py

Defines the canonical data models for the Evidence-Gated Self-Healing Agent.
All models are pure Python dataclasses with no LLM dependency.
Evidence is NEVER fabricated — it must originate from the sandbox state store.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class WorkflowStatus(str, Enum):
    """Top-level lifecycle states of a workflow run."""

    PLANNED = "PLANNED"
    EXECUTING = "EXECUTING"
    AWAITING_EVIDENCE = "AWAITING_EVIDENCE"
    VERIFYING = "VERIFYING"
    VERIFIED = "VERIFIED"
    FAILED = "FAILED"
    RECOVERING = "RECOVERING"
    REPLANNING = "REPLANNING"
    ROLLED_BACK = "ROLLED_BACK"
    BLOCKED = "BLOCKED"
    COMPLETED = "COMPLETED"


class ActionStatus(str, Enum):
    """Execution lifecycle states of a single action."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    AWAITING_EVIDENCE = "AWAITING_EVIDENCE"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    ROLLED_BACK = "ROLLED_BACK"
    SKIPPED = "SKIPPED"


class VerificationStatus(str, Enum):
    """
    Result of comparing expected postconditions against observed sandbox state.

    INVARIANT: INCONCLUSIVE must NEVER be promoted to PASS automatically.
    """

    PASS = "PASS"
    FAIL = "FAIL"
    INCONCLUSIVE = "INCONCLUSIVE"


class ActionType(str, Enum):
    """Enumeration of all tool actions available in the sandbox."""

    CREATE_PROJECT = "create_project"
    DELETE_PROJECT = "delete_project"
    ADD_MEMBER = "add_member"
    REMOVE_MEMBER = "remove_member"
    SET_PERMISSION = "set_permission"
    UPLOAD_FILE = "upload_file"
    DELETE_FILE = "delete_file"
    GENERATE_REPORT = "generate_report"
    SEND_NOTIFICATION = "send_notification"


# ---------------------------------------------------------------------------
# Core data models
# ---------------------------------------------------------------------------


@dataclass
class Action:
    """
    Represents a single step in a workflow plan.

    Every field is required for machine-checkable evidence gating.
    The LLM never fills in evidence fields — those come from the sandbox.
    """

    action_id: str
    action_type: ActionType
    parameters: dict[str, Any]
    expected_postcondition: dict[str, Any]  # machine-checkable predicate dict
    status: ActionStatus = ActionStatus.PENDING
    execution_result: dict[str, Any] | None = None
    evidence_id: str | None = None
    retry_count: int = 0
    error: str | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @classmethod
    def create(
        cls,
        action_type: ActionType,
        parameters: dict[str, Any],
        expected_postcondition: dict[str, Any],
        action_id: str | None = None,
    ) -> "Action":
        """Factory that assigns a UUID action_id unless one is supplied."""
        return cls(
            action_id=action_id or str(uuid.uuid4()),
            action_type=action_type,
            parameters=parameters,
            expected_postcondition=expected_postcondition,
        )


@dataclass
class Evidence:
    """
    Machine-generated snapshot produced by the sandbox AFTER an action executes.

    ARCHITECTURAL RULE: Evidence must originate from the sandbox/state store.
    An LLM must never generate or fabricate this object.
    """

    evidence_id: str
    action_id: str
    source: str                      # e.g. "sandbox.state_store"
    observed_state: dict[str, Any]   # actual slice of sandbox state
    expected_state: dict[str, Any]   # copy of action.expected_postcondition
    verification_status: VerificationStatus = VerificationStatus.INCONCLUSIVE
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @classmethod
    def create(
        cls,
        action_id: str,
        source: str,
        observed_state: dict[str, Any],
        expected_state: dict[str, Any],
        evidence_id: str | None = None,
    ) -> "Evidence":
        """Factory that assigns a UUID evidence_id unless one is supplied."""
        return cls(
            evidence_id=evidence_id or str(uuid.uuid4()),
            action_id=action_id,
            source=source,
            observed_state=observed_state,
            expected_state=expected_state,
        )


@dataclass
class WorkflowRun:
    """
    Tracks the full execution state of a multi-step workflow.

    Contains the ordered plan (list of Actions) and a log of all Evidence
    collected. Status transitions are driven by the verifier, never by the LLM.
    """

    run_id: str
    name: str
    actions: list[Action] = field(default_factory=list)
    evidence_log: list[Evidence] = field(default_factory=list)
    status: WorkflowStatus = WorkflowStatus.PLANNED
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(cls, name: str, run_id: str | None = None) -> "WorkflowRun":
        return cls(run_id=run_id or str(uuid.uuid4()), name=name)

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def get_action(self, action_id: str) -> Action | None:
        """Look up an action by its ID."""
        return next((a for a in self.actions if a.action_id == action_id), None)

    def get_evidence(self, evidence_id: str) -> Evidence | None:
        """Look up evidence by its ID."""
        return next((e for e in self.evidence_log if e.evidence_id == evidence_id), None)

    def pending_actions(self) -> list[Action]:
        return [a for a in self.actions if a.status == ActionStatus.PENDING]

    def failed_actions(self) -> list[Action]:
        return [a for a in self.actions if a.status == ActionStatus.FAILED]

    def touch(self) -> None:
        """Update the updated_at timestamp."""
        self.updated_at = datetime.now(timezone.utc)
