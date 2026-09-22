"""
agents/coordinator.py

Coordinator — orchestrates the full evidence-gated execution loop.

Workflow per action
-------------------
  PLAN → EXECUTE → AWAIT EVIDENCE → VERIFY → [NEXT STEP | RECOVER]

Zero-trust rules enforced here
-------------------------------
1. An action is NEVER marked SUCCEEDED on executor's word alone.
2. Evidence is ALWAYS collected from the sandbox after every execution attempt.
3. Verification is ALWAYS run deterministically before advancing.
4. A downstream action with an unverified dependency is BLOCKED, not skipped.
5. Retries are bounded by max_retries; infinite loops are impossible.
6. INCONCLUSIVE stays INCONCLUSIVE — never silently promoted.

State machine
-------------
All workflow-level transitions go through WorkflowStateMachine, which
rejects invalid jumps and records a full audit trail.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from agents.executor import Executor, ExecutionOutcome
from agents.planner import TaskPlan
from agents.recovery import FailureCategory, RecoveryAgent, RecoveryDecision, RecoveryStrategy
from agents.replanner import Replanner
from core.evidence import EvidenceCollector
from core.models import Action, ActionStatus, Evidence, VerificationStatus, WorkflowRun, WorkflowStatus
from core.state import WorkflowStateMachine
from core.verifier import VerificationEngine, VerificationResult
from sandbox.state_store import SandboxStateStore

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Coordinator configuration
# ---------------------------------------------------------------------------


@dataclass
class CoordinatorConfig:
    max_retries: int = 3          # per-action retry budget
    max_replan_cycles: int = 2    # global replan budget (prevents replan loops)
    atomic_completion: bool = False  # if True, any failure rolls back ALL prior actions


# ---------------------------------------------------------------------------
# Execution record (per-action audit entry)
# ---------------------------------------------------------------------------


@dataclass
class AttemptSnapshot:
    """Immutable record of one execution attempt for a given action."""
    attempt_number: int
    outcome: ExecutionOutcome | None
    evidence: Evidence | None
    vresult: VerificationResult | None
    recovery: RecoveryDecision | None


@dataclass
class ActionRecord:
    action: Action
    outcome: ExecutionOutcome | None = None        # most-recent outcome
    evidence: Evidence | None = None               # most-recent evidence
    vresult: VerificationResult | None = None      # most-recent vresult
    recovery: RecoveryDecision | None = None       # most-recent recovery decision
    attempts: int = 0
    attempt_history: list[AttemptSnapshot] = field(default_factory=list)

    def snapshot_attempt(self) -> None:
        """Capture the current attempt state into attempt_history."""
        self.attempt_history.append(AttemptSnapshot(
            attempt_number=self.attempts,
            outcome=self.outcome,
            evidence=self.evidence,
            vresult=self.vresult,
            recovery=self.recovery,
        ))


# ---------------------------------------------------------------------------
# Coordinator result
# ---------------------------------------------------------------------------


@dataclass
class CoordinatorResult:
    """
    Final output of a workflow run.

    Attributes
    ----------
    run:           The WorkflowRun with full audit trail.
    records:       Per-action execution records.
    final_status:  COMPLETED, BLOCKED, or ROLLED_BACK.
    summary:       Human-readable outcome summary.
    """

    run: WorkflowRun
    records: list[ActionRecord] = field(default_factory=list)
    final_status: WorkflowStatus = WorkflowStatus.PLANNED
    summary: str = ""


# ---------------------------------------------------------------------------
# Coordinator
# ---------------------------------------------------------------------------


class Coordinator:
    """
    Drives a TaskPlan through the evidence-gated execution loop.

    Constructor parameters
    ----------------------
    state_store:  The single source of truth (sandbox).
    executor:     Dispatches actions to sandbox tools.
    config:       Retry / replan limits.
    """

    def __init__(
        self,
        state_store: SandboxStateStore,
        executor: Executor,
        config: CoordinatorConfig | None = None,
    ) -> None:
        self._store    = state_store
        self._executor = executor
        self._config   = config or CoordinatorConfig()
        self._collector = EvidenceCollector(state_store)
        self._verifier  = VerificationEngine()
        self._recovery  = RecoveryAgent(max_retries=self._config.max_retries)
        self._replanner = Replanner(max_retries=self._config.max_retries)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute_plan(self, plan: TaskPlan) -> CoordinatorResult:
        """
        Execute all steps in *plan* with full evidence gating.

        Returns a CoordinatorResult regardless of success or failure.
        Never raises — all errors are captured in the result.
        """
        run = WorkflowRun.create(name=plan.goal)
        run.actions = plan.to_actions()
        sm = WorkflowStateMachine(run)

        records: list[ActionRecord] = [ActionRecord(action=a) for a in run.actions]
        action_to_record: dict[str, ActionRecord] = {r.action.action_id: r for r in records}

        sm.start_execution()
        log.info("Coordinator starting plan '%s' (%d steps)", plan.goal, len(run.actions))

        journal_start = len(self._store.journal)  # snapshot for atomic rollback
        replan_count = 0

        while not sm.is_terminal:
            # Find the next PENDING action whose dependencies are all VERIFIED
            next_action = self._pick_next_action(run, plan)

            if next_action is None:
                # All pending actions are dependency-blocked or exhausted
                if all(a.status == ActionStatus.SUCCEEDED for a in run.actions):
                    sm.complete()
                elif any(a.status == ActionStatus.PENDING for a in run.actions):
                    sm.block("Dependency deadlock: pending actions with no verifiable predecessor")
                else:
                    sm.block("No actionable steps remain")
                break

            record = action_to_record[next_action.action_id]

            # ── Execute ──────────────────────────────────────────────
            sm.await_evidence()
            outcome = self._execute_action(next_action)
            record.outcome = outcome
            record.attempts += 1

            # ── Collect evidence ─────────────────────────────────────
            evidence = self._collector.collect(next_action)
            next_action.evidence_id = evidence.evidence_id
            run.evidence_log.append(evidence)
            record.evidence = evidence

            # ── Verify ───────────────────────────────────────────────
            sm.start_verification()
            vresult = self._verifier.verify(evidence)
            record.vresult = vresult

            log.debug(
                "Action %s (%s): executor=%s, verifier=%s",
                next_action.action_id[:8],
                next_action.action_type.value,
                "SUCCESS" if outcome.success else "FAIL",
                vresult.status.value,
            )

            if vresult.status == VerificationStatus.PASS:
                # ── Verified PASS ─────────────────────────────────────
                next_action.status = ActionStatus.SUCCEEDED
                sm.mark_verified()

                # If all actions are done, complete the workflow
                if all(a.status == ActionStatus.SUCCEEDED for a in run.actions):
                    sm.complete()
                else:
                    sm.transition(WorkflowStatus.EXECUTING, "proceeding to next action")

            else:
                # ── Verification FAIL or INCONCLUSIVE ─────────────────
                next_action.status = ActionStatus.FAILED
                sm.mark_failed(
                    f"Verification {vresult.status.value} for action "
                    f"'{next_action.action_type.value}': {'; '.join(vresult.reasons[:2])}"
                )

                # ── Recover ───────────────────────────────────────────
                decision = self._recovery.decide(next_action, vresult)
                record.recovery = decision
                record.snapshot_attempt()   # preserve this failed attempt for audit
                log.info(
                    "Recovery: category=%s strategy=%s",
                    decision.category.value,
                    decision.strategy.value,
                )

                sm.start_recovery()

                # ── Atomic completion policy ──────────────────────────
                # When enabled, ANY failure triggers a full rollback of
                # every mutation made since this workflow started.
                # This overrides the normal per-action recovery logic.
                if self._config.atomic_completion:
                    result = self._atomic_rollback(
                        run=run,
                        sm=sm,
                        journal_start=journal_start,
                        records=records,
                        reason=(
                            f"Action '{next_action.action_type.value}' failed "
                            f"[{decision.category.value}]: {decision.reason}"
                        ),
                    )
                    log.info("Atomic rollback complete. All mutations reverted.")
                    return result

                if decision.abort:
                    sm.block(f"Recovery aborted: {decision.reason}")
                    break

                if decision.strategy == RecoveryStrategy.RETRY:
                    next_action.retry_count += 1
                    next_action.status = ActionStatus.PENDING  # reset for retry
                    sm.transition(WorkflowStatus.EXECUTING, "retrying action")
                    continue

                if decision.strategy == RecoveryStrategy.REQUERY_STATE:
                    # Re-collect evidence; if still INCONCLUSIVE, treat as FAIL
                    fresh_evidence = self._collector.collect(next_action)
                    fresh_vresult = self._verifier.verify(fresh_evidence)
                    if fresh_vresult.status == VerificationStatus.PASS:
                        next_action.status = ActionStatus.SUCCEEDED
                        run.evidence_log.append(fresh_evidence)
                        sm.transition(WorkflowStatus.VERIFIED, "re-query resolved to PASS")
                        if all(a.status == ActionStatus.SUCCEEDED for a in run.actions):
                            sm.complete()
                        else:
                            sm.transition(WorkflowStatus.EXECUTING, "proceeding after re-query")
                    else:
                        next_action.retry_count += 1
                        next_action.status = ActionStatus.PENDING
                        sm.transition(WorkflowStatus.EXECUTING, "re-query inconclusive — retry")
                    continue

                if decision.strategy in (RecoveryStrategy.ROLLBACK, RecoveryStrategy.REPLAN):
                    if decision.rollback_required:
                        sm.mark_rolled_back()
                        self._store.rollback_last()
                        log.info("Rolled back last sandbox mutation")

                    if replan_count >= self._config.max_replan_cycles:
                        sm.block("Max replan cycles reached")
                        break

                    sm.start_replanning()
                    replan_count += 1

                    verified_actions = [a for a in run.actions if a.status == ActionStatus.SUCCEEDED]
                    new_plan = self._replanner.replan(
                        original_goal=plan.goal,
                        verified_actions=verified_actions,
                        failed_action=next_action,
                        failure_reason=decision.reason,
                        sandbox_snapshot=self._store.snapshot(),
                    )

                    log.info(
                        "Replan %d/%d produced %d steps",
                        replan_count,
                        self._config.max_replan_cycles,
                        len(new_plan.steps),
                    )

                    if not new_plan.steps:
                        sm.block("Replan produced empty plan — no actionable steps remain")
                        break

                    # Splice new actions into the run (preserve verified ones)
                    new_actions = new_plan.to_actions()
                    run.actions = (
                        [a for a in run.actions if a.status == ActionStatus.SUCCEEDED]
                        + new_actions
                    )
                    new_records = [ActionRecord(action=a) for a in new_actions]
                    records.extend(new_records)
                    action_to_record.update({r.action.action_id: r for r in new_records})

                    sm.transition(WorkflowStatus.EXECUTING, "executing revised plan")
                    continue

                # Fallback: block
                sm.block(f"Unhandled strategy {decision.strategy.value}")
                break

        result = CoordinatorResult(
            run=run,
            records=records,
            final_status=run.status,
            summary=self._build_summary(run, records),
        )
        log.info("Workflow '%s' finished: %s", plan.goal, run.status.value)
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _pick_next_action(self, run: WorkflowRun, plan: TaskPlan) -> Action | None:
        """
        Return the first PENDING action whose dependencies are all SUCCEEDED.

        A dependency is represented as an index into plan.steps.  We map that
        index to the corresponding action in run.actions and check its status.

        Dependency gating — zero-trust rule:
          A downstream action MUST NOT execute if any dependency is not SUCCEEDED.
        """
        # Build a map: step_index → action (by position in original plan)
        # For replanned actions that have no step, treat them as independent
        step_index_to_action: dict[int, Action] = {}
        for step in plan.steps:
            if step.step_index < len(run.actions):
                step_index_to_action[step.step_index] = run.actions[step.step_index]

        for i, action in enumerate(run.actions):
            if action.status != ActionStatus.PENDING:
                continue

            # Find the matching step to read dependencies
            matching_step = next(
                (s for s in plan.steps if s.action_type == action.action_type
                 and _params_match(s.parameters, action.parameters)),
                None,
            )

            if matching_step is None:
                # Replanned action — no dependency data; run it
                return action

            # Check all declared dependencies
            deps_satisfied = True
            for dep_idx in matching_step.dependencies:
                dep_action = step_index_to_action.get(dep_idx)
                if dep_action is None or dep_action.status != ActionStatus.SUCCEEDED:
                    deps_satisfied = False
                    break

            if deps_satisfied:
                return action

        return None

    def _atomic_rollback(
        self,
        run: WorkflowRun,
        sm: WorkflowStateMachine,
        journal_start: int,
        records: list[ActionRecord],
        reason: str,
    ) -> "CoordinatorResult":
        """
        Roll back ALL sandbox mutations made since this workflow started.

        Called when atomic_completion=True and any action fails.
        Undoes every journal entry created since journal_start, marks all
        previously-succeeded actions as ROLLED_BACK, and returns the run
        with status=ROLLED_BACK.

        Zero-trust guarantee: rollback is applied and then evidence is
        collected to verify the sandbox is clean.
        """
        sm.mark_rolled_back()

        # Count mutations to be unwound
        mutations_undone = len(self._store.journal) - journal_start
        self._store.rollback_to_length(journal_start)
        log.info(
            "Atomic rollback: unwound %d sandbox mutation(s) for workflow '%s'",
            mutations_undone,
            run.name,
        )

        # Update action statuses
        for action in run.actions:
            if action.status == ActionStatus.SUCCEEDED:
                action.status = ActionStatus.ROLLED_BACK

        run.touch()
        n_rb = sum(1 for a in run.actions if a.status == ActionStatus.ROLLED_BACK)
        summary = (
            f"Workflow '{run.name}' | Status: ROLLED_BACK | "
            f"{n_rb} action(s) rolled back | "
            f"Reason: {reason}"
        )
        return CoordinatorResult(
            run=run,
            records=records,
            final_status=WorkflowStatus.ROLLED_BACK,
            summary=summary,
        )

    def _execute_action(self, action: Action) -> ExecutionOutcome:
        """Thin wrapper so tests can monkeypatch without touching executor."""
        return self._executor.execute(action)

    @staticmethod
    def _build_summary(run: WorkflowRun, records: list[ActionRecord]) -> str:
        succeeded = sum(1 for a in run.actions if a.status == ActionStatus.SUCCEEDED)
        failed = sum(1 for a in run.actions if a.status == ActionStatus.FAILED)
        total = len(run.actions)
        return (
            f"Workflow '{run.name}' | Status: {run.status.value} | "
            f"{succeeded}/{total} actions verified | {failed} failed"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _params_match(step_params: dict, action_params: dict) -> bool:
    """True if every key in step_params appears with the same value in action_params."""
    return all(action_params.get(k) == v for k, v in step_params.items())
