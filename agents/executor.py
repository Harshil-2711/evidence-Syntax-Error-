"""
agents/executor.py

Executor Agent — dispatches a single Action to the appropriate sandbox tool.

Architectural contract
----------------------
* The executor calls the tool and returns an ExecutionOutcome.
* ExecutionOutcome.success is the tool's SELF-REPORT — it is NOT proof.
* The coordinator must ALWAYS collect Evidence and run the verifier independently.
* The executor never touches Evidence objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from core.models import Action, ActionStatus, ActionType
from sandbox.tools import ToolRegistry, ToolResult


# ---------------------------------------------------------------------------
# Execution outcome
# ---------------------------------------------------------------------------


@dataclass
class ExecutionOutcome:
    """
    The executor's self-report.

    IMPORTANT: success=True here is NOT sufficient to mark an action VERIFIED.
    The coordinator must collect Evidence from the sandbox and run the verifier.
    """

    action_id: str
    success: bool
    tool_result: ToolResult
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------


class Executor:
    """
    Dispatches Actions to the ToolRegistry.

    Maps ActionType → ToolRegistry method deterministically.
    Raises ExecutorError for unknown action types (programming errors).
    """

    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry

        # Static dispatch table: ActionType → (method_name, param_extractor)
        self._dispatch: dict[ActionType, str] = {
            ActionType.CREATE_PROJECT:    "create_project",
            ActionType.DELETE_PROJECT:    "delete_project",
            ActionType.ADD_MEMBER:        "add_member",
            ActionType.REMOVE_MEMBER:     "remove_member",
            ActionType.SET_PERMISSION:    "set_permission",
            ActionType.UPLOAD_FILE:       "upload_file",
            ActionType.DELETE_FILE:       "delete_file",
            ActionType.GENERATE_REPORT:   "generate_report",
            ActionType.SEND_NOTIFICATION: "send_notification",
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute(self, action: Action) -> ExecutionOutcome:
        """
        Execute *action* by calling the matching sandbox tool.

        Sets action.status to RUNNING before calling, then to SUCCEEDED or
        FAILED based solely on the tool's self-report — the verifier decides
        the ground truth.
        """
        action.status = ActionStatus.RUNNING

        method_name = self._dispatch.get(action.action_type)
        if method_name is None:
            err = f"No tool registered for action_type={action.action_type.value}"
            action.status = ActionStatus.FAILED
            action.error = err
            return ExecutionOutcome(
                action_id=action.action_id,
                success=False,
                tool_result=ToolResult(success=False, error=err),
                error=err,
            )

        tool_method = getattr(self._registry, method_name)

        try:
            result: ToolResult = tool_method(**action.parameters)
        except Exception as exc:  # noqa: BLE001
            err = f"Unhandled exception in tool '{method_name}': {exc}"
            action.status = ActionStatus.FAILED
            action.error = err
            action.execution_result = {"exception": str(exc)}
            return ExecutionOutcome(
                action_id=action.action_id,
                success=False,
                tool_result=ToolResult(success=False, error=err),
                error=err,
            )

        # Store self-reported result on action (NOT treated as verified truth).
        # Include failure_mode so RecoveryAgent can classify accurately.
        action.execution_result = {
            "tool_success": result.success,
            "data": result.data,
            "failure_mode": result.metadata.get("failure_mode"),  # e.g. FailureMode.FALSE_SUCCESS
        }

        if result.success:
            # Tentatively mark as running until verifier confirms
            # (coordinator will update to SUCCEEDED or FAILED after verification)
            action.status = ActionStatus.RUNNING
        else:
            action.status = ActionStatus.FAILED
            action.error = result.error

        return ExecutionOutcome(
            action_id=action.action_id,
            success=result.success,
            tool_result=result,
            error=result.error if not result.success else None,
        )


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------


class ExecutorError(Exception):
    """Raised by the executor for programming-level errors (unknown tool etc.)."""
