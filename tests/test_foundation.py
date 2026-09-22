"""
tests/test_foundation.py

Unit tests for the Evidence-Gated Self-Healing Agent deterministic foundation.

All tests run WITHOUT any LLM API.  Every outcome is fully deterministic.

Test coverage:
  1.  Successful project creation + evidence PASS
  2.  Failed project creation (tool error) + evidence FAIL
  3.  Successful member addition + evidence PASS
  4.  False executor success detected by verifier (FALSE_SUCCESS injection)
  5.  Missing evidence → INCONCLUSIVE
  6.  State mismatch → FAIL
  7.  Rollback state changes verified
  8.  State machine valid transitions
  9.  State machine rejects invalid transitions
  10. Temporary failure recovers on retry
  11. Permission failure injection
  12. INCONCLUSIVE never promoted to PASS
"""

from __future__ import annotations

import pytest

from core.evidence import EvidenceCollector
from core.models import (
    Action,
    ActionStatus,
    ActionType,
    Evidence,
    VerificationStatus,
    WorkflowRun,
    WorkflowStatus,
)
from core.state import TransitionError, WorkflowStateMachine
from core.verifier import VerificationEngine
from sandbox.failure_injection import FailureInjector, FailureMode, FailureRule
from sandbox.state_store import SandboxStateStore
from sandbox.tools import ToolRegistry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def store() -> SandboxStateStore:
    return SandboxStateStore()


@pytest.fixture()
def tools(store: SandboxStateStore) -> ToolRegistry:
    return ToolRegistry(store)


@pytest.fixture()
def collector(store: SandboxStateStore) -> EvidenceCollector:
    return EvidenceCollector(store)


@pytest.fixture()
def engine() -> VerificationEngine:
    return VerificationEngine()


# ---------------------------------------------------------------------------
# Helper: build an action + collect evidence + verify
# ---------------------------------------------------------------------------


def run_and_verify(
    tools: ToolRegistry,
    collector: EvidenceCollector,
    engine: VerificationEngine,
    action: Action,
    tool_call_fn,
) -> tuple:
    """
    Execute tool_call_fn, collect evidence, run verifier, return (result, evidence, vresult).
    """
    result = tool_call_fn()
    evidence = collector.collect(action)
    vresult = engine.verify(evidence)
    return result, evidence, vresult


# ===========================================================================
# 1. Successful project creation
# ===========================================================================


def test_successful_project_creation(tools, collector, engine):
    """A successful create_project must yield PASS verification."""
    action = Action.create(
        action_type=ActionType.CREATE_PROJECT,
        parameters={"project_id": "proj-1", "name": "Alpha", "owner": "alice"},
        expected_postcondition={"exists": True, "data.name": "Alpha", "data.owner": "alice"},
    )

    result, evidence, vresult = run_and_verify(
        tools, collector, engine, action,
        lambda: tools.create_project("proj-1", "Alpha", "alice"),
    )

    assert result.success is True, f"Tool should succeed: {result.error}"
    assert evidence.observed_state["exists"] is True
    assert vresult.status == VerificationStatus.PASS, "\n".join(vresult.reasons)
    assert evidence.verification_status == VerificationStatus.PASS


# ===========================================================================
# 2. Failed project creation
# ===========================================================================


def test_failed_project_creation(store, collector, engine):
    """An injected execution failure must yield FAIL verification."""
    injector = FailureInjector.with_execution_failure(
        "create_project",
        reason="Simulated DB write error",
        project_id="proj-bad",
    )
    tools = ToolRegistry(store, injector)

    action = Action.create(
        action_type=ActionType.CREATE_PROJECT,
        parameters={"project_id": "proj-bad", "name": "Bad", "owner": "bob"},
        expected_postcondition={"exists": True},
    )

    result = tools.create_project("proj-bad", "Bad", "bob")
    assert result.success is False
    assert "INJECTED" in result.error

    # Collect evidence — project was never created, so exists=False
    evidence = collector.collect(action)
    vresult = engine.verify(evidence)

    assert vresult.status == VerificationStatus.FAIL
    assert evidence.verification_status == VerificationStatus.FAIL


# ===========================================================================
# 3. Successful member addition
# ===========================================================================


def test_successful_member_addition(tools, collector, engine):
    """Add a member to an existing project → PASS."""
    tools.create_project("proj-2", "Beta", "alice")

    action = Action.create(
        action_type=ActionType.ADD_MEMBER,
        parameters={"project_id": "proj-2", "user_id": "bob", "role": "editor"},
        expected_postcondition={"member_exists": True, "project_exists": True},
    )

    result, evidence, vresult = run_and_verify(
        tools, collector, engine, action,
        lambda: tools.add_member("proj-2", "bob", "editor"),
    )

    assert result.success is True
    assert evidence.observed_state["member_exists"] is True
    assert vresult.status == VerificationStatus.PASS, "\n".join(vresult.reasons)


# ===========================================================================
# 4. False executor success detected by verifier
# ===========================================================================


def test_false_executor_success_detected(store, collector, engine):
    """
    FALSE_SUCCESS injection: tool claims success but doesn't mutate state.
    The verifier must catch the lie and produce FAIL.
    """
    injector = FailureInjector.with_false_success("create_project", project_id="proj-lie")
    tools = ToolRegistry(store, injector)

    action = Action.create(
        action_type=ActionType.CREATE_PROJECT,
        parameters={"project_id": "proj-lie", "name": "Lie", "owner": "evil"},
        expected_postcondition={"exists": True},
    )

    result = tools.create_project("proj-lie", "Lie", "evil")

    # Tool reports success (the lie)
    assert result.success is True

    # But evidence reveals the truth: project doesn't exist in state
    evidence = collector.collect(action)
    assert evidence.observed_state["exists"] is False, (
        "State should show project does NOT exist after FALSE_SUCCESS"
    )

    vresult = engine.verify(evidence)
    assert vresult.status == VerificationStatus.FAIL, (
        "Verifier must catch FALSE_SUCCESS as FAIL"
    )


# ===========================================================================
# 5. Missing evidence → INCONCLUSIVE
# ===========================================================================


def test_missing_evidence_produces_inconclusive(store, collector, engine):
    """
    An action with a missing mandatory parameter (project_id) causes the
    evidence collector to fail observation → INCONCLUSIVE.
    INCONCLUSIVE must NEVER become PASS.
    """
    action = Action.create(
        action_type=ActionType.CREATE_PROJECT,
        # Deliberately omit "project_id" to trigger INCONCLUSIVE
        parameters={"name": "Ghost"},
        expected_postcondition={"exists": True},
    )

    evidence = collector.collect(action)
    vresult = engine.verify(evidence)

    assert vresult.status == VerificationStatus.INCONCLUSIVE
    assert evidence.verification_status == VerificationStatus.INCONCLUSIVE
    # The invariant: INCONCLUSIVE ≠ PASS
    assert vresult.status != VerificationStatus.PASS


# ===========================================================================
# 6. State mismatch → FAIL
# ===========================================================================


def test_state_mismatch_produces_fail(store, collector, engine):
    """
    Create a project as "Alice" but postcondition expects owner "bob".
    Verifier must return FAIL.
    """
    store.create_project("proj-mismatch", "Mismatch", "alice")

    action = Action.create(
        action_type=ActionType.CREATE_PROJECT,
        parameters={"project_id": "proj-mismatch", "name": "Mismatch", "owner": "alice"},
        expected_postcondition={"exists": True, "data.owner": "bob"},  # wrong owner
    )

    evidence = collector.collect(action)
    vresult = engine.verify(evidence)

    assert vresult.status == VerificationStatus.FAIL
    assert any("FAIL" in r for r in vresult.reasons)


# ===========================================================================
# 7. Rollback state changes verified
# ===========================================================================


def test_rollback_state_changes_verified(store, collector, engine):
    """
    Create a project, then roll it back.
    Postcondition 'exists=False' must PASS after rollback.
    """
    store.create_project("proj-rb", "Rollback", "alice")
    assert store.get_project("proj-rb") is not None

    # Simulate rollback by undoing the last journal entry
    store.rollback_last()
    assert store.get_project("proj-rb") is None, "Rollback should remove the project"

    # Verify the post-rollback state with evidence
    action = Action.create(
        action_type=ActionType.DELETE_PROJECT,
        parameters={"project_id": "proj-rb"},
        expected_postcondition={"exists": False},
    )

    evidence = collector.collect(action)
    vresult = engine.verify(evidence)

    assert vresult.status == VerificationStatus.PASS, "\n".join(vresult.reasons)


# ===========================================================================
# 8. State machine valid transitions
# ===========================================================================


def test_state_machine_valid_transitions():
    """Walk through a normal happy-path transition sequence."""
    run = WorkflowRun.create("test-workflow")
    sm = WorkflowStateMachine(run)

    assert run.status == WorkflowStatus.PLANNED
    sm.start_execution()
    assert run.status == WorkflowStatus.EXECUTING
    sm.await_evidence()
    assert run.status == WorkflowStatus.AWAITING_EVIDENCE
    sm.start_verification()
    assert run.status == WorkflowStatus.VERIFYING
    sm.mark_verified()
    assert run.status == WorkflowStatus.VERIFIED
    sm.complete()
    assert run.status == WorkflowStatus.COMPLETED


# ===========================================================================
# 9. State machine rejects invalid transitions
# ===========================================================================


def test_state_machine_rejects_invalid_transitions():
    """PLANNED → COMPLETED is forbidden; must raise TransitionError."""
    run = WorkflowRun.create("invalid-jump")
    sm = WorkflowStateMachine(run)

    with pytest.raises(TransitionError):
        sm.complete()  # Cannot jump from PLANNED directly to COMPLETED


def test_state_machine_terminal_state_is_locked():
    """No transitions are allowed from COMPLETED or BLOCKED."""
    run = WorkflowRun.create("terminal")
    sm = WorkflowStateMachine(run)
    sm.start_execution()
    sm.await_evidence()
    sm.start_verification()
    sm.mark_verified()
    sm.complete()

    assert run.status == WorkflowStatus.COMPLETED
    with pytest.raises(TransitionError):
        sm.start_execution()  # terminal — no escape


# ===========================================================================
# 10. Temporary failure recovers on retry
# ===========================================================================


def test_temporary_failure_recovers(store, collector, engine):
    """
    A TEMPORARY_FAILURE rule fires once; the second call succeeds.
    """
    injector = FailureInjector.with_temporary_failure(
        "create_project",
        fail_count=1,
        reason="transient disk error",
        project_id="proj-temp",
    )
    tools = ToolRegistry(store, injector)

    action = Action.create(
        action_type=ActionType.CREATE_PROJECT,
        parameters={"project_id": "proj-temp", "name": "Temp", "owner": "alice"},
        expected_postcondition={"exists": True},
    )

    # First attempt → FAIL
    first = tools.create_project("proj-temp", "Temp", "alice")
    assert first.success is False
    assert "TEMPORARY" in first.error

    # Second attempt → SUCCESS (rule exhausted)
    second = tools.create_project("proj-temp", "Temp", "alice")
    assert second.success is True

    evidence = collector.collect(action)
    vresult = engine.verify(evidence)
    assert vresult.status == VerificationStatus.PASS


# ===========================================================================
# 11. Permission failure injection
# ===========================================================================


def test_permission_failure_injection(store, collector, engine):
    """Permission failure must prevent tool execution and yield FAIL."""
    store.create_project("proj-perm", "Perm", "alice")

    injector = FailureInjector.with_permission_failure(
        "add_member",
        reason="User lacks WRITE role",
        project_id="proj-perm",
    )
    tools = ToolRegistry(store, injector)

    action = Action.create(
        action_type=ActionType.ADD_MEMBER,
        parameters={"project_id": "proj-perm", "user_id": "eve", "role": "editor"},
        expected_postcondition={"member_exists": True},
    )

    result = tools.add_member("proj-perm", "eve", "editor")
    assert result.success is False
    assert "Permission denied" in result.error

    evidence = collector.collect(action)
    vresult = engine.verify(evidence)
    assert vresult.status == VerificationStatus.FAIL


# ===========================================================================
# 12. INCONCLUSIVE never promoted to PASS (invariant enforcement)
# ===========================================================================


def test_inconclusive_never_becomes_pass(engine):
    """
    An Evidence object with no observed_state keys matching expected_state
    must stay INCONCLUSIVE — never PASS.
    """
    evidence = Evidence.create(
        action_id="dummy",
        source="test",
        observed_state={},          # Empty — nothing to observe
        expected_state={"exists": True},
    )
    vresult = engine.verify(evidence)

    assert vresult.status == VerificationStatus.INCONCLUSIVE
    assert vresult.status != VerificationStatus.PASS  # invariant


def test_empty_expected_state_is_inconclusive(engine):
    """No postconditions defined → INCONCLUSIVE (not PASS by default)."""
    evidence = Evidence.create(
        action_id="dummy",
        source="test",
        observed_state={"exists": True},
        expected_state={},  # No conditions to check
    )
    vresult = engine.verify(evidence)
    assert vresult.status == VerificationStatus.INCONCLUSIVE
