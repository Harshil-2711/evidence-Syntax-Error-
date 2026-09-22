"""
agents/replanner.py

Replanner Agent — generates a revised action plan when the original plan fails.

Given:
  - original goal
  - completed and VERIFIED actions (already applied to sandbox)
  - failed action + failure reason
  - current sandbox state snapshot

The replanner returns a new TaskPlan that:
  1. Does NOT repeat already-verified steps.
  2. Addresses the failure (skip bad step, substitute alternative, etc.).
  3. Can be fed back into the coordinator for execution.

In mock mode (default) the replanner uses deterministic rules.
In LLM mode it would produce structured JSON validated by TaskPlan.
"""

from __future__ import annotations

from typing import Any

from agents.planner import MockPlanner, PlannerMode, TaskPlan
from core.models import Action, ActionStatus, ActionType


# ---------------------------------------------------------------------------
# Replanner
# ---------------------------------------------------------------------------


class Replanner:
    """
    Generates a revised plan after a failure.

    Parameters
    ----------
    max_retries:  Copied from RecoveryAgent — used to annotate plan notes.
    mode:         PlannerMode.MOCK (default) or PlannerMode.LLM (not yet wired).
    """

    def __init__(self, max_retries: int = 3, mode: PlannerMode = PlannerMode.MOCK) -> None:
        self.max_retries = max_retries
        self.mode = mode
        self._mock_planner = MockPlanner()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def replan(
        self,
        original_goal: str,
        verified_actions: list[Action],
        failed_action: Action,
        failure_reason: str,
        sandbox_snapshot: dict[str, Any],
    ) -> TaskPlan:
        """
        Produce a revised TaskPlan that accounts for what already happened.

        Strategy
        --------
        1. Re-run the mock planner on the original goal.
        2. Remove steps whose action_type + parameters already succeeded
           (present in verified_actions with SUCCEEDED status).
        3. If the failed action cannot be safely retried (e.g., it would
           create a duplicate), skip it or substitute an alternative.
        4. Renumber step indices and rewrite dependencies.
        5. Return the pruned plan.
        """
        # Generate a fresh full plan
        full_plan = self._mock_planner.plan(original_goal)

        # Build set of already-verified (action_type, frozenset(params)) tuples
        verified_fingerprints = {
            (a.action_type, _param_key(a.parameters))
            for a in verified_actions
            if a.status == ActionStatus.SUCCEEDED
        }

        # Build fingerprint of failed action to potentially skip it
        failed_fingerprint = (failed_action.action_type, _param_key(failed_action.parameters))

        # Determine if the failed action is safe to retry
        skip_failed = self._should_skip_failed(failed_action, sandbox_snapshot, failure_reason)

        # Filter steps
        remaining_steps = []
        new_idx = 0
        old_to_new_idx: dict[int, int] = {}

        for step in full_plan.steps:
            fp = (step.action_type, _param_key(step.parameters))

            if fp in verified_fingerprints:
                # Already done and verified — skip
                continue

            if fp == failed_fingerprint and skip_failed:
                # Skip the permanently failed step
                continue

            old_to_new_idx[step.step_index] = new_idx
            step = step.model_copy(
                update={
                    "step_index": new_idx,
                    "dependencies": [
                        old_to_new_idx[d]
                        for d in step.dependencies
                        if d in old_to_new_idx
                    ],
                }
            )
            remaining_steps.append(step)
            new_idx += 1

        return TaskPlan(
            goal=original_goal,
            steps=remaining_steps,
            planner_mode=self.mode,
            notes=(
                f"Revised plan after failure of '{failed_action.action_type.value}'. "
                f"Reason: {failure_reason}. "
                f"Skipped {len(verified_fingerprints)} already-verified step(s). "
                f"Skip failed: {skip_failed}."
            ),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _should_skip_failed(
        self,
        action: Action,
        snapshot: dict[str, Any],
        reason: str,
    ) -> bool:
        """
        Determine whether to skip the failed action entirely.

        Rules:
        - If the entity already exists in the sandbox (despite failure),
          skip create-type actions to avoid duplicate errors.
        - If the reason indicates INVALID_INPUT, skip (replan can't fix params).
        - Otherwise, include the action so it gets another chance.
        """
        reason_lower = reason.lower()
        if "invalid" in reason_lower or "invalid_input" in reason_lower:
            return True

        # For create_project: if project already exists, skip
        if action.action_type == ActionType.CREATE_PROJECT:
            pid = action.parameters.get("project_id")
            if pid and pid in snapshot.get("projects", {}):
                return True

        return False


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


# Keys that uniquely identify an entity — used for "already done" matching
_IDENTITY_KEYS = frozenset({"project_id", "user_id", "file_id", "report_id", "notification_id"})


def _param_key(params: dict) -> frozenset:
    """
    Stable, hashable identity fingerprint of a parameter dict.

    Only keys that uniquely identify the target entity are included, so
    variations in 'content', 'role', 'message' etc. don't prevent matching.
    """
    identity = {k: v for k, v in params.items() if k in _IDENTITY_KEYS}
    if identity:
        return frozenset(identity.items())
    # Fallback: use all string/int/bool values
    return frozenset(
        (k, v) for k, v in params.items() if isinstance(v, str | int | float | bool)
    )
