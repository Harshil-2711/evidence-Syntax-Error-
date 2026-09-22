"""
tests/test_api_v2.py

API integration tests for the new /workflow/ endpoints.

Tests cover all four demo scenarios through the HTTP API:

  POST /workflow/run               — start a workflow
  GET  /workflow/{id}              — check state
  GET  /workflow/{id}/actions      — 3-way distinction (claim/evidence/verdict)
  GET  /workflow/{id}/evidence     — EvidenceTriple with contradiction detection
  GET  /workflow/{id}/audit        — full audit trail
  POST /workflow/{id}/failure-injection — replay with deterministic failures
  GET  /health                     — service status

Every assertion about the evidence response validates the zero-trust
distinction between EXECUTOR_CLAIM, MACHINE_EVIDENCE, and VERIFICATION_RESULT.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.dependencies import _app_state

HAPPY_GOAL = (
    "Create project Hackathon Alpha, add Harshit as a member, "
    "make the project private, and generate a report."
)

client = TestClient(app)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_state():
    """Isolate every test with a clean AppState."""
    _app_state.reset()
    yield
    _app_state.reset()


def _run_happy() -> dict:
    """Helper: run the happy-path goal, return parsed JSON response."""
    resp = client.post("/workflow/run", json={"goal": HAPPY_GOAL})
    assert resp.status_code == 200, resp.text
    return resp.json()


# ===========================================================================
# POST /workflow/run
# ===========================================================================


class TestRunWorkflow:

    def test_run_returns_200(self):
        resp = client.post("/workflow/run", json={"goal": HAPPY_GOAL})
        assert resp.status_code == 200

    def test_run_returns_workflow_id(self):
        data = _run_happy()
        assert "workflow_id" in data
        assert len(data["workflow_id"]) > 0

    def test_run_happy_path_completes(self):
        data = _run_happy()
        assert data["final_status"] == "COMPLETED"

    def test_run_returns_summary_fields(self):
        data = _run_happy()
        required = {
            "workflow_id", "goal", "final_status", "total_actions",
            "verified_actions", "failed_actions", "rolled_back_actions",
            "recovery_count", "retry_count", "evidence_coverage",
        }
        assert required <= set(data.keys())

    def test_run_empty_goal_rejected(self):
        resp = client.post("/workflow/run", json={"goal": ""})
        assert resp.status_code == 422  # Pydantic min_length

    def test_run_whitespace_goal_rejected(self):
        resp = client.post("/workflow/run", json={"goal": "   "})
        assert resp.status_code == 400

    def test_run_invalid_planner_mode_rejected(self):
        resp = client.post("/workflow/run", json={"goal": HAPPY_GOAL, "planner_mode": "invalid"})
        assert resp.status_code == 422

    def test_run_llm_mode_without_key_rejected(self):
        resp = client.post("/workflow/run", json={"goal": HAPPY_GOAL, "planner_mode": "llm"})
        assert resp.status_code == 400
        assert "API key" in resp.json()["detail"]

    def test_run_evidence_coverage_is_1_on_happy_path(self):
        data = _run_happy()
        assert data["evidence_coverage"] == 1.0

    def test_run_no_recovery_on_happy_path(self):
        data = _run_happy()
        assert data["recovery_count"] == 0
        assert data["retry_count"] == 0

    def test_run_injector_not_active_by_default(self):
        data = _run_happy()
        assert data["injector_active"] is False


# ===========================================================================
# GET /workflow/{id}
# ===========================================================================


class TestGetWorkflow:

    def test_get_returns_200(self):
        run = _run_happy()
        wid = run["workflow_id"]
        resp = client.get(f"/workflow/{wid}")
        assert resp.status_code == 200

    def test_get_unknown_returns_404(self):
        resp = client.get("/workflow/nonexistent-id-xyz")
        assert resp.status_code == 404

    def test_get_returns_same_status_as_run(self):
        run = _run_happy()
        wid = run["workflow_id"]
        get_data = client.get(f"/workflow/{wid}").json()
        assert get_data["final_status"] == run["final_status"]

    def test_get_preserves_goal(self):
        run = _run_happy()
        wid = run["workflow_id"]
        get_data = client.get(f"/workflow/{wid}").json()
        assert get_data["goal"] == HAPPY_GOAL


# ===========================================================================
# GET /workflow/{id}/actions — 3-way zero-trust distinction
# ===========================================================================


class TestGetActions:

    def test_actions_returns_list(self):
        run = _run_happy()
        resp = client.get(f"/workflow/{run['workflow_id']}/actions")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_actions_unknown_returns_404(self):
        resp = client.get("/workflow/nope/actions")
        assert resp.status_code == 404

    def test_actions_each_has_3_layer_fields(self):
        """Every action response must expose all three zero-trust layers."""
        run = _run_happy()
        actions = client.get(f"/workflow/{run['workflow_id']}/actions").json()
        assert actions
        for a in actions:
            # Layer 1: executor claim
            assert "executor_claim" in a
            assert "self_reported_success" in a["executor_claim"]
            assert "CAUTION" in a["executor_claim"]
            # Layer 2: verdict (from machine evidence)
            assert "verification_verdict" in a
            # Layer 3: final status (only SUCCEEDED if PASS)
            assert "final_status" in a

    def test_actions_happy_path_all_pass(self):
        run = _run_happy()
        actions = client.get(f"/workflow/{run['workflow_id']}/actions").json()
        for a in actions:
            assert a["verification_verdict"] == "PASS"
            assert a["final_status"] == "SUCCEEDED"

    def test_actions_no_contradictions_on_happy_path(self):
        run = _run_happy()
        actions = client.get(f"/workflow/{run['workflow_id']}/actions").json()
        for a in actions:
            assert a["contradiction_detected"] is False

    def test_actions_executor_claimed_success_on_happy_path(self):
        run = _run_happy()
        actions = client.get(f"/workflow/{run['workflow_id']}/actions").json()
        for a in actions:
            assert a["executor_claim"]["self_reported_success"] is True

    def test_actions_has_evidence_id(self):
        run = _run_happy()
        actions = client.get(f"/workflow/{run['workflow_id']}/actions").json()
        for a in actions:
            assert a["evidence_id"] is not None


# ===========================================================================
# GET /workflow/{id}/evidence — EvidenceTriple 3-layer schema
# ===========================================================================


class TestGetEvidence:

    def test_evidence_returns_200(self):
        run = _run_happy()
        resp = client.get(f"/workflow/{run['workflow_id']}/evidence")
        assert resp.status_code == 200

    def test_evidence_unknown_returns_404(self):
        resp = client.get("/workflow/nope/evidence")
        assert resp.status_code == 404

    def test_evidence_has_all_top_level_fields(self):
        run = _run_happy()
        data = client.get(f"/workflow/{run['workflow_id']}/evidence").json()
        required = {
            "workflow_id", "goal", "final_status",
            "total_evidence_records", "contradictions_detected",
            "evidence", "zero_trust_guarantee",
        }
        assert required <= set(data.keys())

    def test_evidence_zero_trust_guarantee_field_present(self):
        run = _run_happy()
        data = client.get(f"/workflow/{run['workflow_id']}/evidence").json()
        assert "zero_trust_guarantee" in data
        assert "sandbox.state_store" in data["zero_trust_guarantee"]

    def test_evidence_triples_have_all_three_layers(self):
        """The 3-layer schema must be present on every evidence record."""
        run = _run_happy()
        data = client.get(f"/workflow/{run['workflow_id']}/evidence").json()
        assert data["evidence"]
        for triple in data["evidence"]:
            # Layer 1
            assert "executor_claim" in triple
            assert "LAYER" in triple["executor_claim"]
            assert triple["executor_claim"]["LAYER"] == "EXECUTOR_CLAIM"
            assert "CAUTION" in triple["executor_claim"]
            # Layer 2
            assert "machine_evidence" in triple
            if triple["machine_evidence"]:
                assert "LAYER" in triple["machine_evidence"]
                assert triple["machine_evidence"]["LAYER"] == "MACHINE_EVIDENCE"
                assert "SOURCE_GUARANTEE" in triple["machine_evidence"]
            # Layer 3
            assert "verification_result" in triple
            assert "LAYER" in triple["verification_result"]
            assert triple["verification_result"]["LAYER"] == "VERIFICATION_RESULT"
            assert "ZERO_TRUST_INVARIANT" in triple["verification_result"]

    def test_evidence_source_is_always_sandbox(self):
        """Evidence MUST originate from sandbox.state_store — NEVER from LLM."""
        run = _run_happy()
        data = client.get(f"/workflow/{run['workflow_id']}/evidence").json()
        for triple in data["evidence"]:
            if triple["machine_evidence"]:
                assert triple["machine_evidence"]["source"] == "sandbox.state_store", (
                    "Evidence source must be 'sandbox.state_store' — no LLM fabrication allowed"
                )

    def test_evidence_happy_path_all_pass(self):
        run = _run_happy()
        data = client.get(f"/workflow/{run['workflow_id']}/evidence").json()
        for triple in data["evidence"]:
            assert triple["verification_result"]["verdict"] == "PASS"

    def test_evidence_no_contradictions_on_happy_path(self):
        run = _run_happy()
        data = client.get(f"/workflow/{run['workflow_id']}/evidence").json()
        assert data["contradictions_detected"] == 0
        for triple in data["evidence"]:
            assert triple["contradiction_detected"] is False


# ===========================================================================
# GET /workflow/{id}/audit
# ===========================================================================


class TestGetAudit:

    def test_audit_returns_200(self):
        run = _run_happy()
        resp = client.get(f"/workflow/{run['workflow_id']}/audit")
        assert resp.status_code == 200

    def test_audit_unknown_returns_404(self):
        resp = client.get("/workflow/nope/audit")
        assert resp.status_code == 404

    def test_audit_has_required_fields(self):
        run = _run_happy()
        data = client.get(f"/workflow/{run['workflow_id']}/audit").json()
        required = {
            "workflow_id", "goal", "final_status",
            "events", "summary", "state_machine_history",
        }
        assert required <= set(data.keys())

    def test_audit_has_events(self):
        run = _run_happy()
        data = client.get(f"/workflow/{run['workflow_id']}/audit").json()
        assert len(data["events"]) >= 3

    def test_audit_event_fields(self):
        run = _run_happy()
        data = client.get(f"/workflow/{run['workflow_id']}/audit").json()
        for ev in data["events"]:
            assert "timestamp" in ev
            assert "agent" in ev
            assert "action" in ev
            assert "parameters" in ev
            assert "retry_number" in ev

    def test_audit_summary_counts(self):
        run = _run_happy()
        data = client.get(f"/workflow/{run['workflow_id']}/audit").json()
        s = data["summary"]
        assert s["executor_events"] >= 1
        assert s["evidence_events"] >= 1
        assert s["verifier_events"] >= 1
        assert s["total_events"] >= 3
        assert s["pass_verdicts"] >= 1

    def test_audit_state_machine_history_ends_completed(self):
        run = _run_happy()
        data = client.get(f"/workflow/{run['workflow_id']}/audit").json()
        history = data["state_machine_history"]
        assert history
        assert history[-1]["to"] == "COMPLETED"


# ===========================================================================
# POST /workflow/{id}/failure-injection — deterministic demo failures
# ===========================================================================


class TestFailureInjection:

    def test_injection_returns_replay_id(self):
        run = _run_happy()
        resp = client.post(
            f"/workflow/{run['workflow_id']}/failure-injection",
            json={
                "tool_name": "add_member",
                "mode": "FALSE_SUCCESS",
                "max_fires": 1,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "replay_workflow_id" in data
        assert data["replay_workflow_id"] != run["workflow_id"]

    def test_injection_false_success_completes(self):
        """FALSE_SUCCESS × 1 → verifier catches → retry → COMPLETED."""
        run = _run_happy()
        resp = client.post(
            f"/workflow/{run['workflow_id']}/failure-injection",
            json={
                "tool_name": "add_member",
                "mode": "FALSE_SUCCESS",
                "max_fires": 1,
                "max_retries": 3,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["final_status"] == "COMPLETED"
        assert "FALSE_SUCCESS" in data["injected_failure"]

    def test_injection_execution_failure_blocks(self):
        """Permanent EXECUTION_FAILURE → BLOCKED after max_retries."""
        run = _run_happy()
        resp = client.post(
            f"/workflow/{run['workflow_id']}/failure-injection",
            json={
                "tool_name": "create_project",
                "mode": "EXECUTION_FAILURE",
                "max_fires": 0,   # unlimited = permanent
                "max_retries": 1,
                "max_replan_cycles": 0,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["final_status"] == "BLOCKED"

    def test_injection_atomic_rollback_produces_rolled_back(self):
        """set_permission FAIL + atomic_completion → ROLLED_BACK."""
        run = _run_happy()
        resp = client.post(
            f"/workflow/{run['workflow_id']}/failure-injection",
            json={
                "tool_name": "set_permission",
                "mode": "EXECUTION_FAILURE",
                "param_filter": {"project_id": "hackathon-alpha"},
                "max_fires": 0,
                "max_retries": 0,
                "max_replan_cycles": 0,
                "atomic_completion": True,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["final_status"] == "ROLLED_BACK"

    def test_injection_replay_has_separate_audit_trail(self):
        """The replay run should have its own audit trail accessible by replay_id."""
        run = _run_happy()
        inj_resp = client.post(
            f"/workflow/{run['workflow_id']}/failure-injection",
            json={
                "tool_name": "add_member",
                "mode": "FALSE_SUCCESS",
                "max_fires": 1,
                "max_retries": 3,
            },
        ).json()
        replay_id = inj_resp["replay_workflow_id"]

        # Should be fetchable as an independent run
        get_resp = client.get(f"/workflow/{replay_id}")
        assert get_resp.status_code == 200

        audit_resp = client.get(f"/workflow/{replay_id}/audit")
        assert audit_resp.status_code == 200

    def test_injection_replay_evidence_shows_contradiction(self):
        """FALSE_SUCCESS replay evidence must show contradiction_detected=True."""
        run = _run_happy()
        inj_resp = client.post(
            f"/workflow/{run['workflow_id']}/failure-injection",
            json={
                "tool_name": "add_member",
                "mode": "FALSE_SUCCESS",
                "max_fires": 1,
                "max_retries": 3,
            },
        ).json()
        replay_id = inj_resp["replay_workflow_id"]

        ev_data = client.get(f"/workflow/{replay_id}/evidence").json()
        contradictions = [
            t for t in ev_data["evidence"] if t["contradiction_detected"]
        ]
        assert contradictions, (
            "Replay with FALSE_SUCCESS must have at least one contradiction "
            "(executor claimed success but machine evidence proved otherwise)"
        )
        assert ev_data["contradictions_detected"] >= 1

    def test_injection_unknown_workflow_returns_404(self):
        resp = client.post(
            "/workflow/nonexistent/failure-injection",
            json={"tool_name": "add_member", "mode": "FALSE_SUCCESS"},
        )
        assert resp.status_code == 404

    def test_injection_invalid_mode_returns_400(self):
        run = _run_happy()
        resp = client.post(
            f"/workflow/{run['workflow_id']}/failure-injection",
            json={"tool_name": "add_member", "mode": "NOT_A_MODE"},
        )
        assert resp.status_code == 400
        assert "Unknown failure mode" in resp.json()["detail"]

    def test_injection_replay_shows_parent_run_id(self):
        run = _run_happy()
        inj_resp = client.post(
            f"/workflow/{run['workflow_id']}/failure-injection",
            json={
                "tool_name": "add_member",
                "mode": "FALSE_SUCCESS",
                "max_fires": 1,
            },
        ).json()
        replay_id = inj_resp["replay_workflow_id"]

        get_data = client.get(f"/workflow/{replay_id}").json()
        assert get_data["parent_run_id"] == run["workflow_id"]
        assert get_data["injector_active"] is True


# ===========================================================================
# GET /health
# ===========================================================================


class TestHealth:

    def test_health_returns_200(self):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_has_workflow_count(self):
        resp = client.get("/health")
        assert "workflow_count" in resp.json()

    def test_health_count_increases_after_run(self):
        before = client.get("/health").json()["workflow_count"]
        _run_happy()
        after = client.get("/health").json()["workflow_count"]
        assert after > before
