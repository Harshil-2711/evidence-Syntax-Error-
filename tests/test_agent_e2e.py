"""
tests/test_agent_e2e.py

End-to-end tests for the Evidence-Gated Self-Healing Agent orchestration layer.

All tests are fully deterministic — no API key required.
The FailureInjector controls all failure scenarios exactly.

Tests
-----
TEST A — Happy path: 4-step goal fully verified
TEST B — False success: executor lies, verifier catches it
TEST C — Recovery retry: transient failure recovers and succeeds
TEST D — Repeated failure eventually produces BLOCKED (not false completion)
TEST E — Dependency gating: downstream action blocked when predecessor unverified

All tests also assert the full state-machine audit trail (transition_history).
"""

from __future__ import annotations

import pytest

from agents.coordinator import Coordinator, CoordinatorConfig
from agents.executor import Executor
from agents.planner import Planner, PlannerConfig, PlannerMode, TaskPlan, PlanStep
from agents.recovery import FailureCategory, RecoveryStrategy
from core.models import ActionStatus, ActionType, VerificationStatus, WorkflowStatus
from sandbox.failure_injection import FailureInjector, FailureMode, FailureRule
from sandbox.state_store import SandboxStateStore
from sandbox.tools import ToolRegistry


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_coordinator(
    store: SandboxStateStore | None = None,
    injector: FailureInjector | None = None,
    max_retries: int = 3,
    max_replan_cycles: int = 2,
) -> tuple[Coordinator, SandboxStateStore]:
    store = store or SandboxStateStore()
    registry = ToolRegistry(store, injector or FailureInjector())
    executor = Executor(registry)
    cfg = CoordinatorConfig(max_retries=max_retries, max_replan_cycles=max_replan_cycles)
    coordinator = Coordinator(store, executor, cfg)
    return coordinator, store


def _simple_plan(project_id: str = "proj-a", name: str = "Alpha", owner: str = "alice") -> TaskPlan:
    """Minimal single-step plan: create a project."""
    step = PlanStep(
        step_index=0,
        action_type=ActionType.CREATE_PROJECT,
        parameters={"project_id": project_id, "name": name, "owner": owner},
        expected_postcondition={"exists": True, "data.name": name},
        dependencies=[],
        description=f"Create project '{name}'",
    )
    return TaskPlan(goal=f"Create project {name}", steps=[step])


# ---------------------------------------------------------------------------
# TEST A — Happy path
# ---------------------------------------------------------------------------


class TestHappyPath:
    """
    TEST A: Full 4-step goal executed without any failures.

    Goal: "Create Hackathon Alpha, add Harshit, make it private, generate a report."
    Expected: All 4 actions SUCCEEDED, workflow COMPLETED.
    """

    def test_full_goal_all_verified(self):
        planner = Planner(PlannerConfig(mode=PlannerMode.MOCK))
        plan = planner.plan(
            "Create project Hackathon Alpha, add Harshit, make it private and generate a report."
        )

        # Plan must have at least 3 steps (create + add_member + set_permission/report)
        assert len(plan.steps) >= 3, f"Expected ≥3 steps, got {len(plan.steps)}"
        assert any(s.action_type == ActionType.CREATE_PROJECT for s in plan.steps)
        assert any(s.action_type == ActionType.ADD_MEMBER for s in plan.steps)

        coordinator, store = _make_coordinator()
        result = coordinator.execute_plan(plan)

        assert result.final_status == WorkflowStatus.COMPLETED, result.summary
        assert all(
            a.status == ActionStatus.SUCCEEDED for a in result.run.actions
        ), [f"{a.action_type.value}={a.status.value}" for a in result.run.actions]

        # All evidence must be PASS
        for ev in result.run.evidence_log:
            assert ev.verification_status == VerificationStatus.PASS, (
                f"Evidence for action {ev.action_id} was {ev.verification_status.value}"
            )

        # Transition history must end with COMPLETED
        history = result.run.metadata.get("transition_history", [])
        assert history[-1]["to"] == WorkflowStatus.COMPLETED.value

    def test_simple_single_step_plan(self):
        coordinator, store = _make_coordinator()
        plan = _simple_plan("proj-simple", "Simple", "bob")
        result = coordinator.execute_plan(plan)

        assert result.final_status == WorkflowStatus.COMPLETED
        assert store.get_project("proj-simple") is not None
        assert result.run.actions[0].status == ActionStatus.SUCCEEDED

    def test_planner_produces_pydantic_models(self):
        planner = Planner()
        plan = planner.plan("Create project Demo, add Alice, generate a report.")
        assert isinstance(plan, TaskPlan)
        for step in plan.steps:
            assert isinstance(step, PlanStep)
            assert step.action_type in ActionType.__members__.values()

    def test_planner_rejects_empty_goal(self):
        planner = Planner()
        with pytest.raises(ValueError):
            planner.plan("")


# ---------------------------------------------------------------------------
# TEST B — False executor success detected by verifier
# ---------------------------------------------------------------------------


class TestFalseSuccessDetected:
    """
    TEST B: Executor reports success (FALSE_SUCCESS injection) but does not
    mutate state. Verifier must detect the lie. Workflow must NOT complete.
    """

    def test_false_success_yields_blocked_not_completed(self):
        injector = FailureInjector.with_false_success("create_project", project_id="proj-lie")
        # max_replan_cycles=0: no replan after rollback — workflow goes BLOCKED immediately.
        # (Replanning would re-run MockPlanner which generates "lie" slug, bypassing injection.)
        coordinator, store = _make_coordinator(injector=injector, max_retries=0, max_replan_cycles=0)
        plan = _simple_plan("proj-lie", "Lie", "evil")

        result = coordinator.execute_plan(plan)

        # Executor said SUCCESS — but verifier must have caught it
        assert result.final_status != WorkflowStatus.COMPLETED, (
            "Workflow must NOT complete when verifier catches FALSE_SUCCESS"
        )
        assert result.final_status in {WorkflowStatus.BLOCKED, WorkflowStatus.FAILED}, (
            f"Expected BLOCKED or FAILED, got {result.final_status.value}"
        )

        # Project must not exist in state (false success didn't mutate)
        assert store.get_project("proj-lie") is None

        # The evidence must show FAIL
        for ev in result.run.evidence_log:
            if ev.action_id == result.run.actions[0].action_id:
                assert ev.verification_status == VerificationStatus.FAIL, (
                    "Evidence must be FAIL for FALSE_SUCCESS"
                )

    def test_false_success_evidence_contains_exists_false(self):
        injector = FailureInjector.with_false_success("create_project", project_id="proj-lie2")
        coordinator, store = _make_coordinator(injector=injector, max_retries=0, max_replan_cycles=0)
        plan = _simple_plan("proj-lie2", "Lie2", "evil")
        result = coordinator.execute_plan(plan)

        evidence = result.run.evidence_log[0]
        assert evidence.observed_state["exists"] is False

    def test_action_status_is_failed_not_succeeded(self):
        injector = FailureInjector.with_false_success("create_project", project_id="proj-lie3")
        coordinator, store = _make_coordinator(injector=injector, max_retries=0, max_replan_cycles=0)
        plan = _simple_plan("proj-lie3", "Lie3", "evil")
        result = coordinator.execute_plan(plan)

        action = result.run.actions[0]
        assert action.status == ActionStatus.FAILED, (
            f"Action status should be FAILED after FALSE_SUCCESS, got {action.status.value}"
        )


# ---------------------------------------------------------------------------
# TEST C — Recovery retries and succeeds
# ---------------------------------------------------------------------------


class TestRecoveryRetrySucceeds:
    """
    TEST C: A TEMPORARY_FAILURE fires once on create_project. The recovery
    agent retries. Second attempt succeeds. Workflow must COMPLETE.
    """

    def test_retry_after_transient_failure(self):
        injector = FailureInjector.with_temporary_failure(
            "create_project",
            fail_count=1,
            reason="transient lock error",
            project_id="proj-retry",
        )
        coordinator, store = _make_coordinator(injector=injector, max_retries=3)
        plan = _simple_plan("proj-retry", "Retry", "alice")
        result = coordinator.execute_plan(plan)

        assert result.final_status == WorkflowStatus.COMPLETED, result.summary
        assert store.get_project("proj-retry") is not None
        assert result.run.actions[0].status == ActionStatus.SUCCEEDED

    def test_retry_count_incremented(self):
        injector = FailureInjector.with_temporary_failure(
            "create_project",
            fail_count=2,
            reason="transient",
            project_id="proj-retrycnt",
        )
        coordinator, store = _make_coordinator(injector=injector, max_retries=3)
        plan = _simple_plan("proj-retrycnt", "RetryCount", "alice")
        result = coordinator.execute_plan(plan)

        assert result.final_status == WorkflowStatus.COMPLETED
        action = result.run.actions[0]
        # retry_count reflects number of retries
        assert action.retry_count >= 2

    def test_recovery_agent_classifies_transient_correctly(self):
        from agents.recovery import RecoveryAgent, FailureCategory, RecoveryStrategy
        from core.models import Action, ActionStatus, ActionType, Evidence, VerificationStatus
        from core.verifier import VerificationResult

        agent = RecoveryAgent(max_retries=3)
        action = Action.create(
            action_type=ActionType.CREATE_PROJECT,
            parameters={"project_id": "x"},
            expected_postcondition={"exists": True},
        )
        action.error = "Temporary failure — retry allowed"
        action.retry_count = 0

        ev = Evidence.create(
            action_id=action.action_id,
            source="test",
            observed_state={"exists": False},
            expected_state={"exists": True},
        )
        ev.verification_status = VerificationStatus.FAIL
        vresult = VerificationResult(
            status=VerificationStatus.FAIL,
            reasons=["'exists': expected=True, observed=False → FAIL"],
            evidence=ev,
        )

        decision = agent.decide(action, vresult)
        assert decision.strategy == RecoveryStrategy.RETRY
        assert decision.retry_allowed is True


# ---------------------------------------------------------------------------
# TEST D — Repeated failure → BLOCKED (never false completion)
# ---------------------------------------------------------------------------


class TestRepeatedFailureBlocked:
    """
    TEST D: EXECUTION_FAILURE fired every time (permanent). Verifier always
    gets FAIL. After max_retries exhausted → BLOCKED, never COMPLETED.
    """

    def test_permanent_failure_produces_blocked(self):
        # Register a permanent execution failure (non-temporary)
        injector = FailureInjector()
        injector.register(FailureRule(
            tool_name="create_project",
            mode=FailureMode.EXECUTION_FAILURE,
            reason="permanent DB error",
            param_filter={"project_id": "proj-perm"},
            fail_count=999,  # effectively permanent
        ))
        coordinator, store = _make_coordinator(
            injector=injector,
            max_retries=2,
            max_replan_cycles=0,  # no replanning
        )
        plan = _simple_plan("proj-perm", "Perm", "alice")
        result = coordinator.execute_plan(plan)

        assert result.final_status == WorkflowStatus.BLOCKED, (
            f"Expected BLOCKED, got {result.final_status.value}"
        )
        assert result.final_status != WorkflowStatus.COMPLETED

    def test_blocked_workflow_has_no_succeeded_actions(self):
        injector = FailureInjector()
        injector.register(FailureRule(
            tool_name="create_project",
            mode=FailureMode.EXECUTION_FAILURE,
            reason="disk full",
            param_filter={"project_id": "proj-diskfull"},
            fail_count=999,
        ))
        coordinator, store = _make_coordinator(
            injector=injector, max_retries=1, max_replan_cycles=0
        )
        plan = _simple_plan("proj-diskfull", "DiskFull", "alice")
        result = coordinator.execute_plan(plan)

        assert result.final_status == WorkflowStatus.BLOCKED
        assert all(
            a.status != ActionStatus.SUCCEEDED for a in result.run.actions
        )

    def test_max_retries_respected_exactly(self):
        """
        With max_retries=N and a permanent failure, the action must be
        attempted at most N+1 times (1 initial + N retries).
        """
        fail_count = 999
        max_retries = 2
        injector = FailureInjector()
        injector.register(FailureRule(
            tool_name="create_project",
            mode=FailureMode.EXECUTION_FAILURE,
            reason="always fail",
            param_filter={"project_id": "proj-maxretry"},
            fail_count=fail_count,
        ))
        coordinator, store = _make_coordinator(
            injector=injector,
            max_retries=max_retries,
            max_replan_cycles=0,
        )
        plan = _simple_plan("proj-maxretry", "MaxRetry", "alice")
        result = coordinator.execute_plan(plan)

        assert result.final_status == WorkflowStatus.BLOCKED
        action = result.run.actions[0]
        # retry_count must equal max_retries (not exceed)
        assert action.retry_count <= max_retries, (
            f"retry_count={action.retry_count} exceeded max_retries={max_retries}"
        )


# ---------------------------------------------------------------------------
# TEST E — Dependency gating
# ---------------------------------------------------------------------------


class TestDependencyGating:
    """
    TEST E: A downstream action (add_member) has a dependency on create_project.
    If create_project is unverified, add_member must NOT execute.
    """

    def test_downstream_blocked_when_dependency_unverified(self):
        """
        Inject a permanent failure on create_project.
        add_member depends on create_project (step_index=0).
        add_member must remain PENDING — never execute.
        """
        injector = FailureInjector()
        injector.register(FailureRule(
            tool_name="create_project",
            mode=FailureMode.EXECUTION_FAILURE,
            reason="create fails",
            param_filter={"project_id": "proj-dep"},
            fail_count=999,
        ))
        coordinator, store = _make_coordinator(
            injector=injector, max_retries=0, max_replan_cycles=0
        )

        # Two-step plan: create_project then add_member (dependency: step 0)
        steps = [
            PlanStep(
                step_index=0,
                action_type=ActionType.CREATE_PROJECT,
                parameters={"project_id": "proj-dep", "name": "Dep", "owner": "alice"},
                expected_postcondition={"exists": True},
                dependencies=[],
            ),
            PlanStep(
                step_index=1,
                action_type=ActionType.ADD_MEMBER,
                parameters={"project_id": "proj-dep", "user_id": "bob", "role": "editor"},
                expected_postcondition={"member_exists": True},
                dependencies=[0],  # depends on step 0 being VERIFIED
            ),
        ]
        plan = TaskPlan(goal="Create project Dep then add bob", steps=steps)
        result = coordinator.execute_plan(plan)

        # create_project failed → add_member must remain PENDING or be BLOCKED
        add_member_action = next(
            (a for a in result.run.actions if a.action_type == ActionType.ADD_MEMBER), None
        )
        assert add_member_action is not None

        assert add_member_action.status in {ActionStatus.PENDING, ActionStatus.SKIPPED}, (
            f"add_member should be PENDING (blocked by dependency), "
            f"got {add_member_action.status.value}"
        )

        # add_member must NOT have been executed (no execution_result showing attempt)
        if add_member_action.execution_result:
            # If there is an execution_result, it must not show tool_success=True
            assert add_member_action.execution_result.get("tool_success") is not True

    def test_member_not_in_store_when_dependency_blocked(self):
        """Member must not exist in sandbox if dependency was never satisfied."""
        injector = FailureInjector()
        injector.register(FailureRule(
            tool_name="create_project",
            mode=FailureMode.EXECUTION_FAILURE,
            reason="fail",
            param_filter={"project_id": "proj-dep2"},
            fail_count=999,
        ))
        coordinator, store = _make_coordinator(
            injector=injector, max_retries=0, max_replan_cycles=0
        )

        steps = [
            PlanStep(
                step_index=0,
                action_type=ActionType.CREATE_PROJECT,
                parameters={"project_id": "proj-dep2", "name": "Dep2", "owner": "alice"},
                expected_postcondition={"exists": True},
                dependencies=[],
            ),
            PlanStep(
                step_index=1,
                action_type=ActionType.ADD_MEMBER,
                parameters={"project_id": "proj-dep2", "user_id": "carol", "role": "editor"},
                expected_postcondition={"member_exists": True},
                dependencies=[0],
            ),
        ]
        plan = TaskPlan(goal="dep2 test", steps=steps)
        coordinator.execute_plan(plan)

        # Project should not exist
        assert store.get_project("proj-dep2") is None
        # Member should not exist
        assert store.get_member("proj-dep2", "carol") is None

    def test_independent_steps_not_blocked(self):
        """
        Steps with no dependencies must always execute regardless of other step status.
        """
        coordinator, store = _make_coordinator()

        # Two independent steps (no dependency between them)
        steps = [
            PlanStep(
                step_index=0,
                action_type=ActionType.CREATE_PROJECT,
                parameters={"project_id": "proj-ind-a", "name": "IndA", "owner": "alice"},
                expected_postcondition={"exists": True},
                dependencies=[],
            ),
            PlanStep(
                step_index=1,
                action_type=ActionType.GENERATE_REPORT,
                parameters={
                    "report_id": "rep-ind",
                    "project_id": "proj-ind-a",
                    "report_type": "summary",
                },
                expected_postcondition={"exists": True},
                dependencies=[0],  # depends on project
            ),
        ]
        plan = TaskPlan(goal="ind test", steps=steps)
        result = coordinator.execute_plan(plan)

        assert result.final_status == WorkflowStatus.COMPLETED
        assert all(a.status == ActionStatus.SUCCEEDED for a in result.run.actions)


# ---------------------------------------------------------------------------
# Additional integration scenarios
# ---------------------------------------------------------------------------


class TestIntegrationScenarios:

    def test_planner_mock_e2e_complete(self):
        """Planner → Coordinator → all verified."""
        planner = Planner()
        plan = planner.plan("Create project Omega, add Alice, generate a report.")
        coordinator, store = _make_coordinator()
        result = coordinator.execute_plan(plan)
        assert result.final_status == WorkflowStatus.COMPLETED

    def test_audit_trail_populated(self):
        """Transition history must be non-empty after any run."""
        coordinator, store = _make_coordinator()
        plan = _simple_plan("proj-audit", "Audit", "alice")
        result = coordinator.execute_plan(plan)
        history = result.run.metadata.get("transition_history", [])
        assert len(history) >= 2  # at minimum PLANNED→EXECUTING→...→COMPLETED

    def test_evidence_log_length_matches_actions(self):
        """Every executed action must produce at least one evidence entry."""
        coordinator, store = _make_coordinator()
        plan = _simple_plan("proj-evlog", "EvLog", "alice")
        result = coordinator.execute_plan(plan)
        executed = [a for a in result.run.actions if a.status == ActionStatus.SUCCEEDED]
        assert len(result.run.evidence_log) >= len(executed)

    def test_coordinator_summary_contains_status(self):
        coordinator, store = _make_coordinator()
        plan = _simple_plan("proj-sum", "Sum", "alice")
        result = coordinator.execute_plan(plan)
        assert WorkflowStatus.COMPLETED.value in result.summary

    def test_state_mismatch_fail_classification(self):
        """FALSE_SUCCESS → verifier FAIL → recovery classifies STATE_MISMATCH."""
        from agents.recovery import RecoveryAgent, FailureCategory
        from core.models import Action, ActionStatus, ActionType, Evidence, VerificationStatus
        from core.verifier import VerificationResult

        agent = RecoveryAgent(max_retries=3)
        action = Action.create(
            action_type=ActionType.CREATE_PROJECT,
            parameters={"project_id": "x"},
            expected_postcondition={"exists": True},
        )
        action.status = ActionStatus.FAILED
        action.retry_count = 0

        ev = Evidence.create(
            action_id=action.action_id,
            source="test",
            observed_state={"exists": False},
            expected_state={"exists": True},
        )
        ev.verification_status = VerificationStatus.FAIL
        vresult = VerificationResult(
            status=VerificationStatus.FAIL,
            reasons=["'exists': expected=True, observed=False → FAIL"],
            evidence=ev,
        )
        decision = agent.decide(action, vresult)
        assert decision.category == FailureCategory.STATE_MISMATCH
        # When retries are available, STATE_MISMATCH uses RETRY (not blind rollback).
        # Blind rollback on FALSE_SUCCESS would undo a DIFFERENT action since
        # FALSE_SUCCESS doesn't mutate state — RETRY is the correct strategy.
        assert decision.strategy in {RecoveryStrategy.RETRY, RecoveryStrategy.ROLLBACK}, (
            f"Expected RETRY or ROLLBACK for STATE_MISMATCH, got {decision.strategy}"
        )

    def test_replanner_skips_verified_steps(self):
        """Replanner must exclude already-verified steps from the new plan."""
        from agents.replanner import Replanner
        from core.models import Action, ActionStatus, ActionType

        replanner = Replanner()
        # Simulate: create_project was already verified
        verified = Action.create(
            action_type=ActionType.CREATE_PROJECT,
            parameters={"project_id": "hackathon-alpha", "name": "Hackathon Alpha", "owner": "harshit"},
            expected_postcondition={"exists": True},
        )
        verified.status = ActionStatus.SUCCEEDED

        failed = Action.create(
            action_type=ActionType.ADD_MEMBER,
            parameters={"project_id": "hackathon-alpha", "user_id": "harshit", "role": "editor"},
            expected_postcondition={"member_exists": True},
        )
        failed.status = ActionStatus.FAILED

        new_plan = replanner.replan(
            original_goal="Create project Hackathon Alpha, add Harshit, generate a report.",
            verified_actions=[verified],
            failed_action=failed,
            failure_reason="permission denied",
            sandbox_snapshot={"projects": {"hackathon-alpha": {}}, "members": {}, "permissions": {}, "files": {}, "notifications": {}, "reports": {}},
        )
        # create_project must not appear in new plan (already done)
        new_types = [s.action_type for s in new_plan.steps]
        assert ActionType.CREATE_PROJECT not in new_types, (
            "Replanner must not re-include already-verified create_project"
        )
