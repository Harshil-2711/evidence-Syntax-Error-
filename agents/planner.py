"""
agents/planner.py

Planner Agent — converts a natural-language goal into a structured,
machine-checkable action plan.

Design principles
-----------------
* In MOCK mode (default, no API key required) a deterministic rule-based
  parser produces the plan.  Every test runs deterministically.
* In LLM mode (optional, requires provider config) an LLM generates the plan
  JSON, which is then validated by the same Pydantic models.
* The LLM is NEVER allowed to produce Evidence or mutate sandbox state.
  It only fills in action_type, parameters, and expected_postconditions.

Pydantic models
---------------
  PlanStep         — one step in the plan (serialisable to/from JSON)
  TaskPlan         — ordered list of PlanSteps
  PlannerConfig    — controls LLM vs mock mode + model settings
"""

from __future__ import annotations

import re
import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from core.models import Action, ActionType


# ---------------------------------------------------------------------------
# Planner mode
# ---------------------------------------------------------------------------


class PlannerMode(str, Enum):
    MOCK = "mock"   # deterministic rule-based, no API key needed
    LLM  = "llm"   # real LLM call (placeholder, not wired up yet)


# ---------------------------------------------------------------------------
# Pydantic plan models
# ---------------------------------------------------------------------------


class PlanStep(BaseModel):
    """
    One step of the planner's output.

    This is the ONLY object the LLM is allowed to produce.
    Evidence fields are always left empty — they are filled by the sandbox.
    """

    step_index: int = Field(ge=0, description="Zero-based position in the plan")
    action_type: ActionType
    parameters: dict[str, Any] = Field(default_factory=dict)
    expected_postcondition: dict[str, Any] = Field(
        default_factory=dict,
        description="Machine-checkable predicates verified against sandbox state",
    )
    dependencies: list[int] = Field(
        default_factory=list,
        description="step_index values that must be VERIFIED before this step runs",
    )
    description: str = Field(default="", description="Human-readable step label")

    def to_action(self, action_id: str | None = None) -> Action:
        """Convert this PlanStep into a core Action ready for execution."""
        return Action.create(
            action_type=self.action_type,
            parameters=self.parameters,
            expected_postcondition=self.expected_postcondition,
            action_id=action_id,
        )


class TaskPlan(BaseModel):
    """
    Ordered list of PlanSteps produced by the planner.

    The coordinator consumes this plan sequentially, gating each step on its
    dependency steps being in VERIFIED state.
    """

    plan_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    goal: str
    steps: list[PlanStep] = Field(default_factory=list)
    planner_mode: PlannerMode = PlannerMode.MOCK
    notes: str = ""

    def to_actions(self) -> list[Action]:
        """Convert all steps to Action objects (preserving order)."""
        return [step.to_action() for step in self.steps]


# ---------------------------------------------------------------------------
# Planner config
# ---------------------------------------------------------------------------


class PlannerConfig(BaseModel):
    mode: PlannerMode = PlannerMode.MOCK
    llm_provider: str = ""   # e.g. "openai", "anthropic", "google"
    model_name: str = ""     # e.g. "gpt-4o", "claude-3-5-sonnet"
    api_key: str = ""        # loaded from env in LLM mode
    temperature: float = 0.0 # deterministic LLM output when possible


# ---------------------------------------------------------------------------
# Mock planner — deterministic, keyword-based
# ---------------------------------------------------------------------------

# Token → canonical action type mapping (order matters for matching)
_TOKEN_MAP: list[tuple[re.Pattern, ActionType]] = [
    (re.compile(r"\b(create|new|init|setup)\b.*\b(project|workspace|repo)\b", re.I), ActionType.CREATE_PROJECT),
    (re.compile(r"\b(delete|remove|destroy)\b.*\b(project|workspace)\b", re.I),      ActionType.DELETE_PROJECT),
    (re.compile(r"\b(add|invite|include)\b.*\b(member|user|person|harshit|alice|bob|eve|carol)\b", re.I), ActionType.ADD_MEMBER),
    (re.compile(r"\b(remove|kick|exclude)\b.*\b(member|user)\b", re.I),              ActionType.REMOVE_MEMBER),
    (re.compile(r"\b(set|change|make|mark)\b.*\b(permission|private|public|role|access)\b", re.I), ActionType.SET_PERMISSION),
    (re.compile(r"\b(upload|attach)\b.*\b(file|document|attachment)\b", re.I),       ActionType.UPLOAD_FILE),
    (re.compile(r"\b(delete|remove)\b.*\b(file|document)\b", re.I),                  ActionType.DELETE_FILE),
    (re.compile(r"\b(generate|create|produce|make)\b.*\b(report|summary|analysis)\b", re.I), ActionType.GENERATE_REPORT),
    (re.compile(r"\b(send|notify|alert|email)\b.*\b(notification|message|alert)\b", re.I), ActionType.SEND_NOTIFICATION),
]

# Extract project/user names from goal
_PROJECT_RE  = re.compile(r"project\s+['\"]?([A-Za-z0-9_\-][A-Za-z0-9_\- ]*)['\"]?(?=\s*(?:,|\.|$|\band\b|\badd\b|\bmake\b|\bgenerate\b|\bset\b|\bsend\b|\bupload\b))", re.I)
_MEMBER_RE   = re.compile(r"\b(Harshit|Alice|Bob|Eve|Carol|Dave)\b")
_PRIVATE_RE  = re.compile(r"\b(private|restricted)\b", re.I)
_PUBLIC_RE   = re.compile(r"\b(public|open)\b", re.I)


def _extract_project_name(goal: str) -> str:
    m = _PROJECT_RE.search(goal)
    return m.group(1).strip().title() if m else "DefaultProject"


def _extract_members(goal: str) -> list[str]:
    return list(dict.fromkeys(m.lower() for m in _MEMBER_RE.findall(goal)))


def _extract_privacy(goal: str) -> str:
    if _PRIVATE_RE.search(goal):
        return "private"
    if _PUBLIC_RE.search(goal):
        return "public"
    return "private"


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


class MockPlanner:
    """
    Rule-based deterministic planner. Produces a TaskPlan without any API call.

    Supports goals like:
      "Create Hackathon Alpha, add Harshit, make it private, generate a report."
    """

    def plan(self, goal: str) -> TaskPlan:
        project_name = _extract_project_name(goal)
        project_id   = _slug(project_name) or "proj-default"
        members      = _extract_members(goal)
        privacy      = _extract_privacy(goal)

        steps: list[PlanStep] = []
        idx = 0

        # --- Detect which actions are needed ---
        action_types_needed: list[ActionType] = []
        for pattern, atype in _TOKEN_MAP:
            if pattern.search(goal):
                if atype not in action_types_needed:
                    action_types_needed.append(atype)

        # Always start with create_project if it's in the goal
        create_idx: int | None = None
        if ActionType.CREATE_PROJECT in action_types_needed:
            steps.append(PlanStep(
                step_index=idx,
                action_type=ActionType.CREATE_PROJECT,
                parameters={
                    "project_id": project_id,
                    "name": project_name,
                    "owner": members[0] if members else "owner",
                },
                expected_postcondition={"exists": True, "data.name": project_name},
                dependencies=[],
                description=f"Create project '{project_name}'",
            ))
            create_idx = idx
            idx += 1

        # add_member for each detected member
        if ActionType.ADD_MEMBER in action_types_needed:
            for user in members:
                deps = [create_idx] if create_idx is not None else []
                steps.append(PlanStep(
                    step_index=idx,
                    action_type=ActionType.ADD_MEMBER,
                    parameters={
                        "project_id": project_id,
                        "user_id": user,
                        "role": "editor",
                    },
                    expected_postcondition={
                        "member_exists": True,
                        "project_exists": True,
                    },
                    dependencies=deps,
                    description=f"Add member '{user}' to '{project_name}'",
                ))
                idx += 1

        # set_permission
        if ActionType.SET_PERMISSION in action_types_needed:
            deps = [create_idx] if create_idx is not None else []
            user = members[0] if members else "owner"
            steps.append(PlanStep(
                step_index=idx,
                action_type=ActionType.SET_PERMISSION,
                parameters={
                    "project_id": project_id,
                    "user_id": user,
                    "role": privacy,
                },
                expected_postcondition={"permission": privacy},
                dependencies=deps,
                description=f"Set project visibility to '{privacy}'",
            ))
            idx += 1

        # generate_report
        if ActionType.GENERATE_REPORT in action_types_needed:
            report_id = f"report-{project_id}"
            deps = [create_idx] if create_idx is not None else []
            steps.append(PlanStep(
                step_index=idx,
                action_type=ActionType.GENERATE_REPORT,
                parameters={
                    "report_id": report_id,
                    "project_id": project_id,
                    "report_type": "summary",
                    "content": f"Auto-generated report for {project_name}",
                },
                expected_postcondition={"exists": True},
                dependencies=deps,
                description=f"Generate summary report for '{project_name}'",
            ))
            idx += 1

        # send_notification
        if ActionType.SEND_NOTIFICATION in action_types_needed:
            notif_id = f"notif-{project_id}"
            recipient = members[0] if members else "owner"
            deps = [create_idx] if create_idx is not None else []
            steps.append(PlanStep(
                step_index=idx,
                action_type=ActionType.SEND_NOTIFICATION,
                parameters={
                    "notification_id": notif_id,
                    "recipient": recipient,
                    "message": f"Project '{project_name}' is ready.",
                },
                expected_postcondition={"exists": True},
                dependencies=deps,
                description=f"Notify '{recipient}' of project creation",
            ))
            idx += 1

        return TaskPlan(
            goal=goal,
            steps=steps,
            planner_mode=PlannerMode.MOCK,
            notes="Generated by MockPlanner (deterministic, no API required)",
        )


# ---------------------------------------------------------------------------
# LLM planner stub (wired when provider is configured)
# ---------------------------------------------------------------------------


class LLMPlanner:
    """
    Placeholder for LLM-backed planning.

    The LLM receives the goal and produces JSON matching TaskPlan schema.
    This class validates and wraps that output — it never trusts raw text.

    NOT YET IMPLEMENTED — will be wired in the agent layer phase.
    """

    def __init__(self, config: PlannerConfig) -> None:
        self._config = config

    def plan(self, goal: str) -> TaskPlan:
        raise NotImplementedError(
            "LLM planner is not yet implemented. "
            "Set PlannerConfig(mode=PlannerMode.MOCK) to use deterministic planning."
        )


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


class Planner:
    """
    Unified planner entry point.

    Selects MockPlanner or LLMPlanner based on config.
    All tests use MockPlanner by default.
    """

    def __init__(self, config: PlannerConfig | None = None) -> None:
        self._config = config or PlannerConfig()
        if self._config.mode == PlannerMode.MOCK:
            self._impl: MockPlanner | LLMPlanner = MockPlanner()
        else:
            self._impl = LLMPlanner(self._config)

    def plan(self, goal: str) -> TaskPlan:
        """Convert goal string into a validated TaskPlan."""
        if not goal.strip():
            raise ValueError("Goal must not be empty.")
        return self._impl.plan(goal)
