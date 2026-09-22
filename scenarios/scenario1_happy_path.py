"""
scenarios/scenario1_happy_path.py

SCENARIO 1 — HAPPY PATH

Goal: "Create Hackathon Alpha, add Harshit as a member, make the project
       private, and generate a report."

Expected flow:
  Plan  → Execute → Evidence → Verify (PASS) → Next action → … → COMPLETED

All four actions must be executed, have evidence collected from the sandbox,
be verified PASS, and the workflow must reach COMPLETED.

No failure injection — standard clean-room execution.
"""

from __future__ import annotations

from scenarios.runner import ScenarioResult, ScenarioRunner
from sandbox.failure_injection import FailureInjector

GOAL = (
    "Create project Hackathon Alpha, add Harshit as a member, "
    "make the project private, and generate a report."
)

SCENARIO_NAME = "Scenario 1 — Happy Path"


def run() -> ScenarioResult:
    """
    Execute the happy-path scenario deterministically.

    Returns a ScenarioResult with:
      - final_status == COMPLETED
      - all actions SUCCEEDED
      - all evidence PASS
      - audit log showing full PLAN→EXECUTE→EVIDENCE→VERIFY chain
    """
    runner = ScenarioRunner(
        injector=FailureInjector(),  # no failures
    )
    return runner.run(
        goal=GOAL,
        scenario_name=SCENARIO_NAME,
        expected_status="COMPLETED",
    )
