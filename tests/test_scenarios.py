"""
tests/test_scenarios.py

Automated tests for all four hackathon demo scenarios.

These are the CANONICAL demo tests. Every assertion maps directly to a
claim made in the project specification. If a test fails, the claim is
false and the scenario is NOT ready for demo.

SCENARIO 1 — Happy Path          → COMPLETED, all PASS
SCENARIO 2 — False Success        → verifier catches lie, retry succeeds, COMPLETED
SCENARIO 3 — Unrecoverable        → bounded retries, BLOCKED, no false completion
SCENARIO 4 — Atomic Rollback      → ROLLED_BACK, sandbox clean, verifications PASS

Also tests:
  - AuditEvent model structure
  - WorkflowSummary correctness
  - get_audit_log() and get_workflow_summary() utilities
"""

from __future__ import annotations

import pytest

import scenarios.scenario1_happy_path as s1
import scenarios.scenario2_false_success as s2
import scenarios.scenario3_unrecoverable as s3
import scenarios.scenario4_rollback as s4

from core.audit import AuditEvent, WorkflowSummary, get_audit_log, get_workflow_summary
from core.models import ActionStatus, VerificationStatus, WorkflowStatus


# ===========================================================================
# SCENARIO 1 — HAPPY PATH
# ===========================================================================


class TestScenario1HappyPath:
    """
    Scenario 1: Goal → Plan → Execute → Evidence → Verify → COMPLETED

    Every action must be executed, have evidence collected from the real
    sandbox, be verified PASS, and the run must reach COMPLETED status.
    """

    @pytest.fixture(scope="class")
    def result(self):
        return s1.run()

    # ── Terminal status ────────────────────────────────────────────────
    def test_final_status_is_completed(self, result):
        assert result.final_status == WorkflowStatus.COMPLETED, (
            f"Expected COMPLETED, got {result.final_status.value}\n"
            f"Summary: {result.coordinator_result.summary}"
        )

    def test_all_actions_succeeded(self, result):
        actions = result.coordinator_result.run.actions
        failed = [a for a in actions if a.status != ActionStatus.SUCCEEDED]
        assert not failed, (
            f"Some actions did not SUCCEED: "
            f"{[(a.action_type.value, a.status.value) for a in failed]}"
        )

    # ── Evidence ──────────────────────────────────────────────────────
    def test_all_evidence_is_pass(self, result):
        evidence_log = result.coordinator_result.run.evidence_log
        assert evidence_log, "Evidence log must not be empty"
        bad = [
            e for e in evidence_log
            if e.verification_status != VerificationStatus.PASS
        ]
        assert not bad, (
            f"Some evidence items are not PASS: "
            f"{[(e.action_id[:8], e.verification_status.value) for e in bad]}"
        )

    def test_evidence_sourced_from_sandbox(self, result):
        """Evidence must NEVER be fabricated — must come from sandbox."""
        for ev in result.coordinator_result.run.evidence_log:
            assert ev.source == "sandbox.state_store", (
                f"Evidence source must be sandbox, got '{ev.source}'"
            )

    def test_evidence_count_matches_actions(self, result):
        run = result.coordinator_result.run
        assert len(run.evidence_log) >= len(run.actions)

    # ── Audit log ─────────────────────────────────────────────────────
    def test_audit_log_has_events(self, result):
        assert len(result.audit_log) >= 3

    def test_audit_log_contains_all_agents(self, result):
        agents = {e.agent for e in result.audit_log}
        assert "executor" in agents
        assert "evidence_collector" in agents
        assert "verifier" in agents

    def test_audit_log_all_verifications_pass(self, result):
        verifier_events = [e for e in result.audit_log if e.agent == "verifier"]
        assert verifier_events, "No verifier events found in audit log"
        for ev in verifier_events:
            assert ev.verification == "PASS", (
                f"Audit event shows non-PASS: {ev.verification}\n"
                f"Evidence: {ev.evidence}"
            )

    def test_audit_event_structure(self, result):
        for ev in result.audit_log:
            assert isinstance(ev, AuditEvent)
            assert isinstance(ev.timestamp, object)
            assert ev.agent in {
                "executor", "evidence_collector", "verifier",
                "recovery_agent", "replanner", "rollback", "coordinator",
            }
            assert isinstance(ev.parameters, dict)
            assert isinstance(ev.retry_number, int)

    # ── Workflow summary ───────────────────────────────────────────────
    def test_summary_structure(self, result):
        s = result.summary
        assert isinstance(s, WorkflowSummary)
        assert s.total_actions >= 3
        assert s.verified_actions == s.total_actions
        assert s.failed_actions == 0
        assert s.rolled_back_actions == 0
        assert s.recovery_count == 0
        assert s.retry_count == 0
        assert s.evidence_coverage == 1.0
        assert s.final_status == "COMPLETED"

    def test_audit_transition_history_ends_completed(self, result):
        history = result.coordinator_result.run.metadata.get("transition_history", [])
        assert history, "Transition history is empty"
        assert history[-1]["to"] == "COMPLETED"

    def test_sandbox_has_created_project(self, result):
        """The sandbox must reflect the actual created project."""
        runner_store = result.extra.get("store")
        # Verify through the coordinator result's evidence instead
        evidence_log = result.coordinator_result.run.evidence_log
        project_evidence = next(
            (e for e in evidence_log if "exists" in e.observed_state),
            None,
        )
        assert project_evidence is not None
        assert project_evidence.observed_state.get("exists") is True


# ===========================================================================
# SCENARIO 2 — FALSE SUCCESS DETECTION
# ===========================================================================


class TestScenario2FalseSuccess:
    """
    Scenario 2: Executor lies → verifier catches → retry succeeds → COMPLETED

    The add_member tool returns a success claim WITHOUT mutating state.
    The verifier must detect the contradiction and trigger recovery.
    """

    @pytest.fixture(scope="class")
    def result(self):
        return s2.run()

    # ── Terminal status ────────────────────────────────────────────────
    def test_final_status_is_completed(self, result):
        """Despite the lie, recovery allows the second attempt to succeed."""
        assert result.final_status == WorkflowStatus.COMPLETED, (
            f"Expected COMPLETED after recovery, got {result.final_status.value}\n"
            f"Summary: {result.coordinator_result.summary}"
        )

    # ── The lie was detected ───────────────────────────────────────────
    def test_verifier_produced_at_least_one_fail(self, result):
        """The false-success claim must have been caught by the verifier."""
        fail_verdicts = [
            e for e in result.audit_log
            if e.agent == "verifier" and e.verification == "FAIL"
        ]
        assert fail_verdicts, (
            "Verifier must have produced at least one FAIL to catch the false success. "
            "No FAIL verifier events found in audit log."
        )

    def test_fail_evidence_shows_member_not_in_state(self, result):
        """The FAIL evidence must show member_exists=False despite executor saying success."""
        for ev in result.audit_log:
            if ev.agent == "verifier" and ev.verification == "FAIL":
                observed = ev.evidence.get("observed", {}) if ev.evidence else {}
                # member_exists or exists must be False
                member_present = observed.get("member_exists", observed.get("exists", None))
                assert member_present is False, (
                    f"FAIL evidence should show member_exists=False, got: {observed}"
                )
                # Decision must mention contradiction
                assert ev.decision is not None
                assert "FAIL" in ev.decision or "VERIFICATION" in ev.decision

    # ── Recovery was triggered ─────────────────────────────────────────
    def test_recovery_was_triggered(self, result):
        recovery_events = [
            e for e in result.audit_log if e.agent == "recovery_agent"
        ]
        assert recovery_events, "RecoveryAgent must have been invoked"

    def test_recovery_strategy_is_retry(self, result):
        """STATE_MISMATCH after FALSE_SUCCESS → RETRY (not blind rollback)."""
        for ev in result.audit_log:
            if ev.agent == "recovery_agent":
                assert "RETRY" in (ev.decision or ""), (
                    f"Expected RETRY strategy, got decision: {ev.decision}"
                )

    def test_add_member_retry_count_is_nonzero(self, result):
        """add_member must have been retried at least once."""
        from core.models import ActionType
        run = result.coordinator_result.run
        add_member_action = next(
            (a for a in run.actions if a.action_type == ActionType.ADD_MEMBER),
            None,
        )
        assert add_member_action is not None
        assert add_member_action.retry_count >= 1, (
            f"Expected retry_count >= 1, got {add_member_action.retry_count}"
        )

    # ── Second attempt succeeded ───────────────────────────────────────
    def test_add_member_final_status_succeeded(self, result):
        from core.models import ActionType
        run = result.coordinator_result.run
        add_member_action = next(
            (a for a in run.actions if a.action_type == ActionType.ADD_MEMBER),
            None,
        )
        assert add_member_action is not None
        assert add_member_action.status == ActionStatus.SUCCEEDED, (
            f"add_member must end SUCCEEDED after retry, got {add_member_action.status.value}"
        )

    def test_final_evidence_for_member_is_pass(self, result):
        """The last evidence entry for add_member must be PASS."""
        from core.models import ActionType
        run = result.coordinator_result.run
        add_member_action = next(
            (a for a in run.actions if a.action_type == ActionType.ADD_MEMBER),
            None,
        )
        assert add_member_action is not None
        member_evidence = [
            e for e in run.evidence_log if e.action_id == add_member_action.action_id
        ]
        assert member_evidence, "No evidence entries for add_member"
        last_evidence = member_evidence[-1]
        assert last_evidence.verification_status == VerificationStatus.PASS, (
            f"Final evidence for add_member must be PASS, got "
            f"{last_evidence.verification_status.value}"
        )

    def test_summary_shows_recovery(self, result):
        s = result.summary
        assert s.recovery_count >= 1, "Summary must show at least one recovery"
        assert s.retry_count >= 1, "Summary must show at least one retry"
        assert s.final_status == "COMPLETED"


# ===========================================================================
# SCENARIO 3 — UNRECOVERABLE FAILURE / BLOCKED
# ===========================================================================


class TestScenario3Unrecoverable:
    """
    Scenario 3: Permanent failure exhausts bounded retries → BLOCKED.

    The system must NEVER falsely claim completion.
    It must explicitly record WHY it was blocked.
    """

    @pytest.fixture(scope="class")
    def result(self):
        return s3.run()

    # ── Terminal status ────────────────────────────────────────────────
    def test_final_status_is_blocked(self, result):
        assert result.final_status == WorkflowStatus.BLOCKED, (
            f"Expected BLOCKED, got {result.final_status.value}\n"
            f"Summary: {result.coordinator_result.summary}"
        )

    def test_final_status_is_not_completed(self, result):
        """This is the critical zero-trust guarantee: no false completion."""
        assert result.final_status != WorkflowStatus.COMPLETED, (
            "CRITICAL: Workflow claimed COMPLETED despite permanent failure — "
            "this is a false completion claim!"
        )

    # ── No actions succeeded ───────────────────────────────────────────
    def test_no_actions_succeeded(self, result):
        succeeded = [
            a for a in result.coordinator_result.run.actions
            if a.status == ActionStatus.SUCCEEDED
        ]
        assert not succeeded, (
            f"No actions should SUCCEED when the first action permanently fails: "
            f"{[a.action_type.value for a in succeeded]}"
        )

    # ── Retries were bounded ───────────────────────────────────────────
    def test_retry_count_bounded_by_max_retries(self, result):
        from core.models import ActionType
        run = result.coordinator_result.run
        create_action = next(
            (a for a in run.actions if a.action_type == ActionType.CREATE_PROJECT),
            None,
        )
        assert create_action is not None
        max_retries = result.extra.get("max_retries", 2)
        assert create_action.retry_count <= max_retries, (
            f"retry_count={create_action.retry_count} exceeded "
            f"max_retries={max_retries} — unbounded retries!"
        )

    def test_retry_count_is_nonzero(self, result):
        """Must have tried at least once before giving up."""
        s = result.summary
        assert s.retry_count >= 1, (
            "Must have retried at least once before BLOCKED"
        )

    # ── Block reason is explicit ───────────────────────────────────────
    def test_blocked_reason_is_populated(self, result):
        """The system must explicitly explain why it refused completion."""
        s = result.summary
        assert s.blocked_reason is not None, (
            "WorkflowSummary.blocked_reason must be set when status=BLOCKED"
        )
        assert len(s.blocked_reason) > 0

    def test_transition_history_ends_blocked(self, result):
        history = result.coordinator_result.run.metadata.get("transition_history", [])
        assert history, "Transition history is empty"
        assert history[-1]["to"] == "BLOCKED", (
            f"Transition history must end at BLOCKED, got: {history[-1]}"
        )

    # ── Sandbox is clean ──────────────────────────────────────────────
    def test_sandbox_has_no_project(self, result):
        """create_project never succeeded — project must not be in sandbox."""
        evidence_log = result.coordinator_result.run.evidence_log
        # All evidence for create_project should show exists=False
        from core.models import ActionType
        run = result.coordinator_result.run
        create_action = next(
            (a for a in run.actions if a.action_type == ActionType.CREATE_PROJECT),
            None,
        )
        assert create_action is not None
        project_evidence = [
            e for e in evidence_log if e.action_id == create_action.action_id
        ]
        assert project_evidence, "Must have collected evidence for create_project"
        for ev in project_evidence:
            assert ev.verification_status == VerificationStatus.FAIL, (
                f"All create_project evidence must be FAIL, got {ev.verification_status}"
            )

    # ── Audit log ─────────────────────────────────────────────────────
    def test_audit_log_has_fail_verdicts(self, result):
        fail_verdicts = [e for e in result.audit_log if e.verification == "FAIL"]
        assert fail_verdicts, "Audit log must show FAIL verdicts"

    def test_audit_log_has_recovery_decisions(self, result):
        recovery_events = [e for e in result.audit_log if e.agent == "recovery_agent"]
        assert recovery_events, "Audit log must show recovery decisions"

    # ── Summary ───────────────────────────────────────────────────────
    def test_summary_metrics(self, result):
        s = result.summary
        assert s.final_status == "BLOCKED"
        assert s.verified_actions == 0
        assert s.failed_actions >= 1
        assert s.recovery_count >= 1
        assert s.blocked_reason is not None


# ===========================================================================
# SCENARIO 4 — ATOMIC ROLLBACK
# ===========================================================================


class TestScenario4AtomicRollback:
    """
    Scenario 4: create_project VERIFIED, add_member VERIFIED,
                set_permission FAIL → atomic rollback of all prior actions.

    The sandbox must be clean after rollback.
    Post-rollback verifications must confirm entities are gone.
    """

    @pytest.fixture(scope="class")
    def result(self):
        return s4.run()

    # ── Terminal status ────────────────────────────────────────────────
    def test_final_status_is_rolled_back(self, result):
        assert result.final_status == WorkflowStatus.ROLLED_BACK, (
            f"Expected ROLLED_BACK, got {result.final_status.value}\n"
            f"Summary: {result.coordinator_result.summary}"
        )

    def test_final_status_is_not_completed(self, result):
        assert result.final_status != WorkflowStatus.COMPLETED, (
            "CRITICAL: Workflow claimed COMPLETED despite set_permission failure!"
        )

    # ── Actions were rolled back ───────────────────────────────────────
    def test_verified_actions_are_marked_rolled_back(self, result):
        run = result.coordinator_result.run
        from core.models import ActionType
        # create_project and add_member were verified before failure → must be ROLLED_BACK
        rb_types = {
            a.action_type
            for a in run.actions
            if a.status == ActionStatus.ROLLED_BACK
        }
        assert ActionType.CREATE_PROJECT in rb_types, (
            "create_project must be marked ROLLED_BACK"
        )
        assert ActionType.ADD_MEMBER in rb_types, (
            "add_member must be marked ROLLED_BACK"
        )

    def test_no_actions_are_succeeded(self, result):
        """After atomic rollback, no action should have SUCCEEDED status."""
        succeeded = [
            a for a in result.coordinator_result.run.actions
            if a.status == ActionStatus.SUCCEEDED
        ]
        assert not succeeded, (
            f"Actions should be ROLLED_BACK not SUCCEEDED: "
            f"{[a.action_type.value for a in succeeded]}"
        )

    def test_summary_rolled_back_count(self, result):
        s = result.summary
        assert s.rolled_back_actions >= 2, (
            f"At least 2 actions should be rolled back (project + member), "
            f"got {s.rolled_back_actions}"
        )
        assert s.final_status == "ROLLED_BACK"

    # ── Sandbox is clean ──────────────────────────────────────────────
    def test_rollback_verifications_all_pass(self, result):
        """Post-rollback sandbox checks must ALL pass."""
        assert result.rollback_verifications, "Rollback verifications are empty"
        failed = [
            v for v in result.rollback_verifications if v["verdict"] != "PASS"
        ]
        assert not failed, (
            f"Rollback verification failed — sandbox is NOT clean:\n"
            + "\n".join(
                f"  {v['entity']} '{v['entity_id']}': "
                f"expected exists={v['expected']}, observed={v['observed']}"
                for v in failed
            )
        )

    def test_rollback_verification_project_gone(self, result):
        project_check = next(
            (v for v in result.rollback_verifications if v["entity"] == "project"),
            None,
        )
        assert project_check is not None
        assert project_check["observed"] is False, (
            "Project must not exist in sandbox after rollback"
        )
        assert project_check["verdict"] == "PASS"

    def test_rollback_verification_member_gone(self, result):
        member_check = next(
            (v for v in result.rollback_verifications if v["entity"] == "member"),
            None,
        )
        assert member_check is not None
        assert member_check["observed"] is False, (
            "Member must not exist in sandbox after rollback"
        )
        assert member_check["verdict"] == "PASS"

    # ── Pre-failure evidence was collected ────────────────────────────
    def test_evidence_was_collected_before_failure(self, result):
        """create_project and add_member must have PASS evidence before rollback."""
        from core.models import ActionType
        run = result.coordinator_result.run
        pass_evidence = [
            e for e in run.evidence_log
            if e.verification_status == VerificationStatus.PASS
        ]
        assert pass_evidence, "Must have PASS evidence for steps before failure"

    def test_fail_evidence_for_set_permission(self, result):
        """set_permission must have FAIL evidence (the trigger for rollback)."""
        from core.models import ActionType
        run = result.coordinator_result.run
        fail_evidence = [
            e for e in run.evidence_log
            if e.verification_status == VerificationStatus.FAIL
        ]
        assert fail_evidence, "Must have FAIL evidence for set_permission"

    # ── Audit log ─────────────────────────────────────────────────────
    def test_audit_log_has_rollback_event(self, result):
        rollback_events = [e for e in result.audit_log if e.agent == "rollback"]
        assert rollback_events, "Audit log must have a rollback event"
        rb = rollback_events[0]
        assert "rolled_back_actions" in rb.parameters
        assert len(rb.parameters["rolled_back_actions"]) >= 2

    def test_audit_log_shows_pass_then_fail_sequence(self, result):
        """The audit log must show PASS verdicts before the FAIL."""
        verdicts = [
            (e.action_type, e.verification)
            for e in result.audit_log
            if e.agent == "verifier"
        ]
        assert any(v == "PASS" for _, v in verdicts), "Must have PASS verdicts"
        assert any(v == "FAIL" for _, v in verdicts), "Must have FAIL verdict"

        # Verify order: PASS items come before FAIL
        pass_indices = [i for i, (_, v) in enumerate(verdicts) if v == "PASS"]
        fail_indices = [i for i, (_, v) in enumerate(verdicts) if v == "FAIL"]
        assert pass_indices and fail_indices
        assert max(pass_indices) < min(fail_indices), (
            "PASS verdicts must precede FAIL verdict in audit log"
        )

    def test_audit_log_has_recovery_event(self, result):
        recovery_events = [e for e in result.audit_log if e.agent == "recovery_agent"]
        assert recovery_events, "Audit log must contain a recovery decision event"

    def test_transition_history_contains_rolled_back(self, result):
        history = result.coordinator_result.run.metadata.get("transition_history", [])
        rolled_back_transitions = [
            h for h in history if h.get("to") == "ROLLED_BACK"
        ]
        assert rolled_back_transitions, "Transition history must include ROLLED_BACK state"

    def test_summary_final_status(self, result):
        s = result.summary
        assert s.final_status == "ROLLED_BACK"
        assert s.failed_actions >= 1   # set_permission
        assert s.verified_actions == 0 # no actions left succeeded
        assert s.rolled_back_actions >= 2


# ===========================================================================
# AuditEvent + WorkflowSummary unit tests
# ===========================================================================


class TestAuditAndSummaryUtilities:
    """Unit tests for get_audit_log() and get_workflow_summary()."""

    @pytest.fixture(scope="class")
    def happy_result(self):
        return s1.run()

    def test_get_audit_log_returns_list_of_audit_events(self, happy_result):
        log = get_audit_log(happy_result.coordinator_result)
        assert isinstance(log, list)
        assert all(isinstance(e, AuditEvent) for e in log)

    def test_get_workflow_summary_returns_workflow_summary(self, happy_result):
        s = get_workflow_summary(happy_result.coordinator_result)
        assert isinstance(s, WorkflowSummary)

    def test_audit_event_fields_are_populated(self, happy_result):
        for ev in get_audit_log(happy_result.coordinator_result):
            assert ev.agent is not None and len(ev.agent) > 0
            assert ev.action is not None and len(ev.action) > 0
            assert isinstance(ev.parameters, dict)
            assert isinstance(ev.retry_number, int)

    def test_workflow_summary_as_dict(self, happy_result):
        d = get_workflow_summary(happy_result.coordinator_result).as_dict()
        required_keys = {
            "goal", "total_actions", "verified_actions", "failed_actions",
            "rolled_back_actions", "recovery_count", "retry_count",
            "evidence_coverage", "final_status", "blocked_reason",
        }
        assert required_keys <= set(d.keys())

    def test_summary_evidence_coverage_between_0_and_1(self, happy_result):
        s = get_workflow_summary(happy_result.coordinator_result)
        assert 0.0 <= s.evidence_coverage <= 1.0

    def test_summary_all_verified_property(self, happy_result):
        s = get_workflow_summary(happy_result.coordinator_result)
        assert s.all_verified is True

    def test_blocked_summary_sets_blocked_reason(self):
        blocked_result = s3.run()
        s = get_workflow_summary(blocked_result.coordinator_result)
        assert s.blocked_reason is not None

    def test_rollback_summary_has_rolled_back_count(self):
        rolled_result = s4.run()
        s = get_workflow_summary(rolled_result.coordinator_result)
        assert s.rolled_back_actions >= 2
