"""
sandbox/failure_injection.py

Deterministic failure injection for demos and reproducible tests.

Failures are NEVER random.  They are configured explicitly via a rule table
before execution begins, making demo scenarios 100% reproducible.

Failure modes
-------------
EXECUTION_FAILURE    — Tool raises / returns error immediately.
PARTIAL_EXECUTION    — Tool starts but does not complete cleanly.
FALSE_SUCCESS        — Tool reports success but does NOT mutate state.
MISSING_EVIDENCE     — Tool succeeds but evidence record is absent.
CONTRADICTORY_STATE  — State is mutated to contradict postcondition.
PERMISSION_FAILURE   — Tool rejected due to permission check.
TEMPORARY_FAILURE    — First N calls fail; subsequent calls succeed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Failure mode enum
# ---------------------------------------------------------------------------


class FailureMode(str, Enum):
    EXECUTION_FAILURE   = "EXECUTION_FAILURE"
    PARTIAL_EXECUTION   = "PARTIAL_EXECUTION"
    FALSE_SUCCESS       = "FALSE_SUCCESS"
    MISSING_EVIDENCE    = "MISSING_EVIDENCE"
    CONTRADICTORY_STATE = "CONTRADICTORY_STATE"
    PERMISSION_FAILURE  = "PERMISSION_FAILURE"
    TEMPORARY_FAILURE   = "TEMPORARY_FAILURE"


# ---------------------------------------------------------------------------
# Failure rule
# ---------------------------------------------------------------------------


@dataclass
class FailureRule:
    """
    A rule that makes a specific tool call fail.

    Matching logic:
      - tool_name must match exactly (case-insensitive).
      - param_filter (optional) must be a subset of the call's parameters.
        All key/value pairs in param_filter must appear in the actual params.
      - For TEMPORARY_FAILURE, fail_count controls how many calls fail before
        the injector stops triggering this rule.
    """

    tool_name: str
    mode: FailureMode
    reason: str = ""
    param_filter: dict[str, Any] = field(default_factory=dict)
    fail_count: int = 1           # For TEMPORARY_FAILURE: how many calls fail
    max_fires: int = 0            # For any mode: 0 = unlimited, N = fire at most N times
    _triggered: int = field(default=0, init=False, repr=False, compare=False)

    def matches(self, tool_name: str, params: dict) -> bool:
        if tool_name.lower() != self.tool_name.lower():
            return False
        for k, v in self.param_filter.items():
            if params.get(k) != v:
                return False
        return True

    def consume(self) -> bool:
        """
        Returns True if this rule should fire for the current call.

        TEMPORARY_FAILURE: fires up to fail_count times, then is silent.
        Any mode with max_fires > 0: fires at most max_fires times.
        Any other mode: fires every time (unlimited).
        """
        self._triggered += 1
        if self.mode == FailureMode.TEMPORARY_FAILURE:
            return self._triggered <= self.fail_count
        if self.max_fires > 0:
            return self._triggered <= self.max_fires
        return True  # unlimited


# ---------------------------------------------------------------------------
# Failure injector
# ---------------------------------------------------------------------------


class FailureInjector:
    """
    Consults the registered rule table before each tool call.

    Usage (in tests / demo setup)::

        injector = FailureInjector()
        injector.register(FailureRule(
            tool_name="create_project",
            mode=FailureMode.EXECUTION_FAILURE,
            reason="Simulated DB write error",
            param_filter={"project_id": "proj-bad"},
        ))

    Then pass `injector` to ToolRegistry.
    """

    def __init__(self) -> None:
        self._rules: list[FailureRule] = []

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, rule: FailureRule) -> None:
        """Add a failure rule to the injector."""
        self._rules.append(rule)

    def clear(self) -> None:
        """Remove all registered rules (useful between test cases)."""
        self._rules.clear()

    # ------------------------------------------------------------------
    # Check
    # ------------------------------------------------------------------

    def should_fail(self, tool_name: str, params: dict) -> dict | None:
        """
        Check if any registered rule matches this tool invocation.

        Returns a dict {"mode": FailureMode, "reason": str} if a failure
        should be injected, or None if the call should proceed normally.

        Rules are evaluated in registration order; the FIRST match wins.
        """
        for rule in self._rules:
            if rule.matches(tool_name, params):
                fired = rule.consume()
                if fired:
                    return {"mode": rule.mode, "reason": rule.reason}
        return None

    # ------------------------------------------------------------------
    # Convenience factory methods for common scenarios
    # ------------------------------------------------------------------

    @classmethod
    def with_execution_failure(cls, tool_name: str, reason: str = "", **param_filter) -> "FailureInjector":
        inj = cls()
        inj.register(FailureRule(tool_name=tool_name, mode=FailureMode.EXECUTION_FAILURE,
                                  reason=reason, param_filter=dict(param_filter)))
        return inj

    @classmethod
    def with_false_success(cls, tool_name: str, **param_filter) -> "FailureInjector":
        inj = cls()
        inj.register(FailureRule(tool_name=tool_name, mode=FailureMode.FALSE_SUCCESS,
                                  param_filter=dict(param_filter)))
        return inj

    @classmethod
    def with_missing_evidence(cls, tool_name: str, **param_filter) -> "FailureInjector":
        inj = cls()
        inj.register(FailureRule(tool_name=tool_name, mode=FailureMode.MISSING_EVIDENCE,
                                  param_filter=dict(param_filter)))
        return inj

    @classmethod
    def with_permission_failure(cls, tool_name: str, reason: str = "", **param_filter) -> "FailureInjector":
        inj = cls()
        inj.register(FailureRule(tool_name=tool_name, mode=FailureMode.PERMISSION_FAILURE,
                                  reason=reason, param_filter=dict(param_filter)))
        return inj

    @classmethod
    def with_temporary_failure(cls, tool_name: str, fail_count: int = 1, reason: str = "", **param_filter) -> "FailureInjector":
        inj = cls()
        inj.register(FailureRule(tool_name=tool_name, mode=FailureMode.TEMPORARY_FAILURE,
                                  reason=reason, fail_count=fail_count,
                                  param_filter=dict(param_filter)))
        return inj

    @classmethod
    def with_contradictory_state(cls, tool_name: str, **param_filter) -> "FailureInjector":
        inj = cls()
        inj.register(FailureRule(tool_name=tool_name, mode=FailureMode.CONTRADICTORY_STATE,
                                  param_filter=dict(param_filter)))
        return inj
