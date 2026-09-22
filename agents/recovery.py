"""
agents/recovery.py

Recovery Agent — classifies failures and selects a recovery strategy.

Recovery classification
-----------------------
TRANSIENT_FAILURE   — Likely to succeed on retry (network blip, lock contention)
STATE_MISMATCH      — Observed state contradicts expected (entity exists/missing)
INVALID_INPUT       — Parameters were structurally wrong
PERMISSION_FAILURE  — Authorization denied
MISSING_EVIDENCE    — Evidence could not be collected (INCONCLUSIVE result)
UNKNOWN_FAILURE     — Anything else

Recovery strategies
-------------------
RETRY              — Re-execute the same action (bounded)
ALTERNATIVE_ACTION — Execute a different action to reach the same goal
REQUERY_STATE      — Re-snapshot sandbox state to resolve INCONCLUSIVE
REPLAN             — Hand off to Replanner for a revised plan
ROLLBACK           — Undo previous actions and replan
ABORT              — Stop; emit BLOCKED

Deterministic classification logic
-----------------------------------
Classification is based on:
  * The VerificationStatus (FAIL vs INCONCLUSIVE)
  * error message patterns
  * retry_count vs max_retries
  * Injected failure_mode metadata if present

No LLM is used for classification — it is rule-based and testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from core.models import Action, ActionStatus, VerificationStatus
from core.verifier import VerificationResult
from sandbox.failure_injection import FailureMode


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class FailureCategory(str, Enum):
    TRANSIENT_FAILURE  = "TRANSIENT_FAILURE"
    STATE_MISMATCH     = "STATE_MISMATCH"
    INVALID_INPUT      = "INVALID_INPUT"
    PERMISSION_FAILURE = "PERMISSION_FAILURE"
    MISSING_EVIDENCE   = "MISSING_EVIDENCE"
    UNKNOWN_FAILURE    = "UNKNOWN_FAILURE"


class RecoveryStrategy(str, Enum):
    RETRY              = "RETRY"
    ALTERNATIVE_ACTION = "ALTERNATIVE_ACTION"
    REQUERY_STATE      = "REQUERY_STATE"
    REPLAN             = "REPLAN"
    ROLLBACK           = "ROLLBACK"
    ABORT              = "ABORT"


# ---------------------------------------------------------------------------
# Recovery decision
# ---------------------------------------------------------------------------


@dataclass
class RecoveryDecision:
    """
    Encapsulates the recovery agent's output for one failed action.

    Fields
    ------
    category:          Classified failure type.
    strategy:          Chosen recovery strategy.
    retry_allowed:     True if the action can be retried immediately.
    rollback_required: True if prior state changes must be undone first.
    abort:             True if no recovery is possible (BLOCKED).
    reason:            Human-readable explanation.
    metadata:          Extra context (e.g., retry_count, failure_mode).
    """

    category: FailureCategory
    strategy: RecoveryStrategy
    retry_allowed: bool
    rollback_required: bool
    abort: bool
    reason: str
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Recovery agent
# ---------------------------------------------------------------------------


class RecoveryAgent:
    """
    Classifies a failure and selects a bounded, deterministic recovery strategy.

    Parameters
    ----------
    max_retries:   Maximum times a single action may be retried before ABORT.
    """

    def __init__(self, max_retries: int = 3) -> None:
        self.max_retries = max_retries

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def decide(
        self,
        action: Action,
        vresult: VerificationResult,
    ) -> RecoveryDecision:
        """
        Produce a RecoveryDecision for a failed/inconclusive action.

        Steps
        -----
        1. Classify failure into FailureCategory.
        2. Check retry budget.
        3. Select RecoveryStrategy based on category + budget.
        4. Return RecoveryDecision.
        """
        category = self._classify(action, vresult)
        return self._select_strategy(action, vresult, category)

    # ------------------------------------------------------------------
    # Classification (deterministic, rule-based)
    # ------------------------------------------------------------------

    def _classify(self, action: Action, vresult: VerificationResult) -> FailureCategory:
        """
        Classify the failure using observable signals only.
        Priority: injected failure_mode > verification status > error string.
        """
        # Check injected failure_mode metadata from tool result
        exec_result = action.execution_result or {}

        # Try to read failure_mode from execution_result metadata chain
        failure_mode_raw = None
        if isinstance(exec_result, dict):
            failure_mode_raw = exec_result.get("failure_mode")

        if failure_mode_raw is not None:
            fm = str(failure_mode_raw)
            if "TEMPORARY" in fm:
                return FailureCategory.TRANSIENT_FAILURE
            if "PERMISSION" in fm:
                return FailureCategory.PERMISSION_FAILURE
            if "MISSING_EVIDENCE" in fm:
                return FailureCategory.MISSING_EVIDENCE
            if "FALSE_SUCCESS" in fm or "CONTRADICTORY" in fm:
                return FailureCategory.STATE_MISMATCH

        # INCONCLUSIVE → MISSING_EVIDENCE
        if vresult.status == VerificationStatus.INCONCLUSIVE:
            return FailureCategory.MISSING_EVIDENCE

        # Error message pattern matching
        err = (action.error or "").lower()
        if "permission" in err or "denied" in err or "unauthorized" in err:
            return FailureCategory.PERMISSION_FAILURE
        if "temporary" in err or "transient" in err or "retry" in err:
            return FailureCategory.TRANSIENT_FAILURE
        if "not found" in err or "does not exist" in err or "missing" in err:
            return FailureCategory.STATE_MISMATCH
        if "invalid" in err or "parameter" in err or "required" in err:
            return FailureCategory.INVALID_INPUT

        # Verification FAIL with no other signal → STATE_MISMATCH
        if vresult.status == VerificationStatus.FAIL:
            return FailureCategory.STATE_MISMATCH

        return FailureCategory.UNKNOWN_FAILURE

    # ------------------------------------------------------------------
    # Strategy selection (deterministic decision table)
    # ------------------------------------------------------------------

    def _select_strategy(
        self,
        action: Action,
        vresult: VerificationResult,
        category: FailureCategory,
    ) -> RecoveryDecision:
        """
        Map (category, retry_count, max_retries) → RecoveryStrategy.

        Decision table
        --------------
        TRANSIENT_FAILURE  + retries left  → RETRY
        TRANSIENT_FAILURE  + no retries    → REPLAN
        STATE_MISMATCH     + retries left  → ROLLBACK → REPLAN
        STATE_MISMATCH     + no retries    → ABORT
        PERMISSION_FAILURE + retries left  → RETRY (config may change)
        PERMISSION_FAILURE + no retries    → ABORT
        MISSING_EVIDENCE   + retries left  → REQUERY_STATE
        MISSING_EVIDENCE   + no retries    → ABORT
        INVALID_INPUT                      → REPLAN (retrying won't help)
        UNKNOWN_FAILURE    + retries left  → RETRY
        UNKNOWN_FAILURE    + no retries    → ABORT
        """
        retries_left = self.max_retries - action.retry_count

        if category == FailureCategory.TRANSIENT_FAILURE:
            if retries_left > 0:
                return RecoveryDecision(
                    category=category,
                    strategy=RecoveryStrategy.RETRY,
                    retry_allowed=True,
                    rollback_required=False,
                    abort=False,
                    reason=f"Transient failure; {retries_left} retries remaining.",
                    metadata={"retries_left": retries_left},
                )
            return RecoveryDecision(
                category=category,
                strategy=RecoveryStrategy.REPLAN,
                retry_allowed=False,
                rollback_required=False,
                abort=False,
                reason="Transient failure exhausted retries; requesting replan.",
                metadata={"retries_left": 0},
            )

        if category == FailureCategory.STATE_MISMATCH:
            if retries_left > 0:
                return RecoveryDecision(
                    category=category,
                    strategy=RecoveryStrategy.RETRY,
                    retry_allowed=True,
                    rollback_required=False,
                    abort=False,
                    reason=f"State mismatch; retrying action ({retries_left} retries left).",
                    metadata={"retries_left": retries_left},
                )
            return RecoveryDecision(
                category=category,
                strategy=RecoveryStrategy.ROLLBACK,
                retry_allowed=False,
                rollback_required=True,
                abort=False,
                reason="State mismatch exhausted retries; rolling back and replanning.",
                metadata={"retries_left": 0},
            )

        if category == FailureCategory.PERMISSION_FAILURE:
            if retries_left > 0:
                return RecoveryDecision(
                    category=category,
                    strategy=RecoveryStrategy.RETRY,
                    retry_allowed=True,
                    rollback_required=False,
                    abort=False,
                    reason=f"Permission failure; retrying ({retries_left} left).",
                    metadata={"retries_left": retries_left},
                )
            return RecoveryDecision(
                category=category,
                strategy=RecoveryStrategy.ABORT,
                retry_allowed=False,
                rollback_required=False,
                abort=True,
                reason="Permission failure — no retries left; aborting.",
                metadata={"retries_left": 0},
            )

        if category == FailureCategory.MISSING_EVIDENCE:
            if retries_left > 0:
                return RecoveryDecision(
                    category=category,
                    strategy=RecoveryStrategy.REQUERY_STATE,
                    retry_allowed=True,
                    rollback_required=False,
                    abort=False,
                    reason=f"Evidence missing; re-querying state ({retries_left} left).",
                    metadata={"retries_left": retries_left},
                )
            return RecoveryDecision(
                category=category,
                strategy=RecoveryStrategy.ABORT,
                retry_allowed=False,
                rollback_required=False,
                abort=True,
                reason="Evidence missing and retries exhausted; aborting.",
                metadata={"retries_left": 0},
            )

        if category == FailureCategory.INVALID_INPUT:
            return RecoveryDecision(
                category=category,
                strategy=RecoveryStrategy.REPLAN,
                retry_allowed=False,
                rollback_required=False,
                abort=False,
                reason="Invalid input — retry won't help; requesting replan.",
                metadata={"retries_left": retries_left},
            )

        # UNKNOWN_FAILURE
        if retries_left > 0:
            return RecoveryDecision(
                category=category,
                strategy=RecoveryStrategy.RETRY,
                retry_allowed=True,
                rollback_required=False,
                abort=False,
                reason=f"Unknown failure; attempting retry ({retries_left} left).",
                metadata={"retries_left": retries_left},
            )
        return RecoveryDecision(
            category=category,
            strategy=RecoveryStrategy.ABORT,
            retry_allowed=False,
            rollback_required=False,
            abort=True,
            reason="Unknown failure and retries exhausted; aborting.",
            metadata={"retries_left": 0},
        )
