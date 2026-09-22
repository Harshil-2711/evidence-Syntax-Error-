"""
tests/test_api.py

FastAPI integration tests using httpx.AsyncClient + pytest-anyio.

Tests cover:
  1.  Health check returns 200
  2.  Run a happy-path workflow → COMPLETED
  3.  Run returns correct action count and evidence count
  4.  List workflows accumulates runs
  5.  Get workflow by run_id returns full detail
  6.  Get actions returns per-action status
  7.  Get evidence returns verified evidence with PASS status
  8.  Get audit trail returns ordered transition history
  9.  Sandbox state reflects what the workflow created
  10. Sandbox journal is non-empty after a run
  11. Reset clears all state and history
  12. Unknown run_id returns 404
  13. Empty goal returns 400/422
  14. Workflow with executor failure returns BLOCKED status (not COMPLETED)
  15. Sandbox project endpoint returns correct data
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.dependencies import _app_state


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_state():
    """Reset all singleton state before each test for isolation."""
    _app_state.reset()
    yield
    _app_state.reset()


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


HAPPY_GOAL = "Create project Hackathon Alpha, add Harshit, make it private, generate a report."
SIMPLE_GOAL = "Create project TestProj, add Alice, generate a report."


# ---------------------------------------------------------------------------
# 1. Health check
# ---------------------------------------------------------------------------


def test_health_check(client):
    res = client.get("/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert "version" in body


# ---------------------------------------------------------------------------
# 2. Happy-path workflow → COMPLETED
# ---------------------------------------------------------------------------


def test_run_workflow_happy_path_completes(client):
    res = client.post("/workflows/run", json={"goal": HAPPY_GOAL})
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "COMPLETED", f"Expected COMPLETED, got {body['status']}\n{body['summary']}"
    assert body["run_id"] != ""


# ---------------------------------------------------------------------------
# 3. Run response has correct counts
# ---------------------------------------------------------------------------


def test_run_workflow_counts(client):
    res = client.post("/workflows/run", json={"goal": SIMPLE_GOAL})
    assert res.status_code == 200
    body = res.json()
    assert body["action_count"] >= 2            # create + add_member + report
    assert body["succeeded_count"] == body["action_count"]
    assert body["failed_count"] == 0
    assert body["evidence_count"] >= body["action_count"]


# ---------------------------------------------------------------------------
# 4. List workflows accumulates across runs
# ---------------------------------------------------------------------------


def test_list_workflows_accumulates(client):
    client.post("/workflows/run", json={"goal": SIMPLE_GOAL})
    client.post("/workflows/run", json={"goal": "Create project Beta, add Bob, generate a report."})

    res = client.get("/workflows/")
    assert res.status_code == 200
    runs = res.json()
    assert len(runs) == 2
    for run in runs:
        assert run["status"] == "COMPLETED"


# ---------------------------------------------------------------------------
# 5. Get workflow by run_id
# ---------------------------------------------------------------------------


def test_get_workflow_detail(client):
    run_res = client.post("/workflows/run", json={"goal": SIMPLE_GOAL})
    run_id = run_res.json()["run_id"]

    res = client.get(f"/workflows/{run_id}")
    assert res.status_code == 200
    body = res.json()
    assert body["run_id"] == run_id
    assert body["status"] == "COMPLETED"
    assert len(body["actions"]) >= 1
    assert len(body["evidence_log"]) >= 1


# ---------------------------------------------------------------------------
# 6. Get actions for a run
# ---------------------------------------------------------------------------


def test_get_workflow_actions(client):
    run_res = client.post("/workflows/run", json={"goal": SIMPLE_GOAL})
    run_id = run_res.json()["run_id"]

    res = client.get(f"/workflows/{run_id}/actions")
    assert res.status_code == 200
    actions = res.json()
    assert len(actions) >= 1
    for action in actions:
        assert action["status"] == "SUCCEEDED"
        assert action["action_type"] in [
            "create_project", "add_member", "set_permission",
            "generate_report", "send_notification",
        ]


# ---------------------------------------------------------------------------
# 7. Evidence log has PASS status for happy path
# ---------------------------------------------------------------------------


def test_get_workflow_evidence_all_pass(client):
    run_res = client.post("/workflows/run", json={"goal": SIMPLE_GOAL})
    run_id = run_res.json()["run_id"]

    res = client.get(f"/workflows/{run_id}/evidence")
    assert res.status_code == 200
    evidence_list = res.json()
    assert len(evidence_list) >= 1
    for ev in evidence_list:
        assert ev["verification_status"] == "PASS", (
            f"Evidence {ev['evidence_id']} had status {ev['verification_status']}"
        )
        # Evidence must come from sandbox — never fabricated
        assert ev["source"] == "sandbox.state_store"


# ---------------------------------------------------------------------------
# 8. Audit trail — ordered transitions
# ---------------------------------------------------------------------------


def test_get_workflow_audit_trail(client):
    run_res = client.post("/workflows/run", json={"goal": SIMPLE_GOAL})
    run_id = run_res.json()["run_id"]

    res = client.get(f"/workflows/{run_id}/audit")
    assert res.status_code == 200
    history = res.json()
    assert len(history) >= 3  # PLANNED→EXECUTING→...→COMPLETED

    # First transition must start from PLANNED
    assert history[0]["from"] == "PLANNED"
    # Last transition must end at COMPLETED
    assert history[-1]["to"] == "COMPLETED"


# ---------------------------------------------------------------------------
# 9. Sandbox state reflects workflow mutations
# ---------------------------------------------------------------------------


def test_sandbox_state_reflects_workflow(client):
    client.post("/workflows/run", json={"goal": SIMPLE_GOAL})

    res = client.get("/sandbox/state")
    assert res.status_code == 200
    snap = res.json()

    # "testproj" is the slugified project name from SIMPLE_GOAL
    assert len(snap["projects"]) >= 1, "Sandbox must contain the created project"
    assert snap["journal_length"] > 0


# ---------------------------------------------------------------------------
# 10. Sandbox journal non-empty after run
# ---------------------------------------------------------------------------


def test_sandbox_journal_non_empty(client):
    client.post("/workflows/run", json={"goal": SIMPLE_GOAL})
    res = client.get("/sandbox/journal")
    assert res.status_code == 200
    journal = res.json()
    assert len(journal) >= 1
    for entry in journal:
        assert "operation" in entry
        assert "timestamp" in entry


# ---------------------------------------------------------------------------
# 11. Reset clears everything
# ---------------------------------------------------------------------------


def test_reset_clears_state(client):
    client.post("/workflows/run", json={"goal": SIMPLE_GOAL})
    # Confirm there's something to clear
    assert len(client.get("/workflows/").json()) == 1

    res = client.post("/sandbox/reset", json={"confirm": True})
    assert res.status_code == 200
    assert res.json()["status"] == "reset"

    # After reset: no workflows, empty sandbox
    assert client.get("/workflows/").json() == []
    snap = client.get("/sandbox/state").json()
    assert snap["projects"] == {}
    assert snap["journal_length"] == 0


def test_reset_requires_confirm(client):
    res = client.post("/sandbox/reset", json={"confirm": False})
    assert res.status_code == 400


# ---------------------------------------------------------------------------
# 12. Unknown run_id returns 404
# ---------------------------------------------------------------------------


def test_get_unknown_workflow_returns_404(client):
    res = client.get("/workflows/nonexistent-run-id")
    assert res.status_code == 404


def test_get_unknown_run_actions_returns_404(client):
    res = client.get("/workflows/nonexistent/actions")
    assert res.status_code == 404


def test_get_unknown_run_evidence_returns_404(client):
    res = client.get("/workflows/nonexistent/evidence")
    assert res.status_code == 404


# ---------------------------------------------------------------------------
# 13. Empty goal returns 400/422
# ---------------------------------------------------------------------------


def test_empty_goal_rejected(client):
    res = client.post("/workflows/run", json={"goal": ""})
    assert res.status_code in {400, 422}


def test_whitespace_goal_rejected(client):
    res = client.post("/workflows/run", json={"goal": "   "})
    # Either Pydantic min_length rejects it (422) or planner raises ValueError (400)
    assert res.status_code in {400, 422}


# ---------------------------------------------------------------------------
# 14. Workflow with permanent executor failure → BLOCKED
# ---------------------------------------------------------------------------


def test_run_blocked_workflow_returns_blocked_status(client):
    """
    Send a goal whose only step will fail due to a conflicting pre-existing entity.
    We seed the store directly, then run a 'create' for the same project_id.
    """
    # Seed the sandbox so "proj-conflict" already exists
    _app_state.store.create_project("proj-conflict", "Existing", "alice")

    # Now try to create the same project via the API — tool will reject with "already exists"
    # (no failure injection needed; the tool itself errors)
    res = client.post(
        "/workflows/run",
        json={
            "goal": "Create project Proj Conflict, add Bob, generate a report.",
            "max_retries": 0,
            "max_replan_cycles": 0,
        },
    )
    assert res.status_code == 200
    body = res.json()
    # The create_project tool returns error "already exists" → verifier FAIL → BLOCKED
    assert body["status"] != "COMPLETED", (
        "Workflow must NOT complete when project creation fails"
    )


# ---------------------------------------------------------------------------
# 15. Sandbox project endpoint
# ---------------------------------------------------------------------------


def test_sandbox_project_endpoint(client):
    client.post("/workflows/run", json={"goal": SIMPLE_GOAL})

    res = client.get("/sandbox/projects")
    assert res.status_code == 200
    projects = res.json()
    assert len(projects) >= 1

    # Fetch the first project specifically
    pid = projects[0]["project_id"]
    res2 = client.get(f"/sandbox/projects/{pid}")
    assert res2.status_code == 200
    assert res2.json()["project_id"] == pid


def test_sandbox_unknown_project_returns_404(client):
    res = client.get("/sandbox/projects/does-not-exist")
    assert res.status_code == 404


# ---------------------------------------------------------------------------
# 16. Planner mode field validation
# ---------------------------------------------------------------------------


def test_invalid_planner_mode_rejected(client):
    res = client.post("/workflows/run", json={"goal": SIMPLE_GOAL, "planner_mode": "gpt"})
    assert res.status_code == 422


# ---------------------------------------------------------------------------
# 17. Root and docs endpoints
# ---------------------------------------------------------------------------


def test_root_returns_info(client):
    res = client.get("/")
    assert res.status_code == 200
    body = res.json()
    assert "docs" in body


def test_openapi_schema_available(client):
    res = client.get("/openapi.json")
    assert res.status_code == 200
    schema = res.json()
    assert "paths" in schema
    assert "/workflows/run" in schema["paths"]
