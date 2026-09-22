"""
sandbox/tools.py

Tool registry for the sandbox.

Each tool:
  1. Validates inputs deterministically.
  2. Updates the SandboxStateStore (the single source of truth).
  3. Returns a ToolResult that includes the mutation outcome.

Tools never fabricate evidence or modify Evidence objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from sandbox.failure_injection import FailureInjector, FailureMode
from sandbox.state_store import SandboxStateStore


# ---------------------------------------------------------------------------
# Tool result
# ---------------------------------------------------------------------------


@dataclass
class ToolResult:
    """
    Outcome of a single tool invocation.

    success:   True if the tool completed without error.
    data:      The entity returned by the tool (or None on failure).
    error:     Human-readable error message if success is False.
    metadata:  Extra context (e.g., injected failure mode).
    """

    success: bool
    data: dict | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------


class ToolRegistry:
    """
    Provides all sandbox tools.  Tools are plain methods that mutate the
    SandboxStateStore and return a ToolResult.

    A FailureInjector is consulted BEFORE executing any tool to allow
    deterministic failure simulation for demos and tests.
    """

    def __init__(
        self,
        state_store: SandboxStateStore,
        failure_injector: FailureInjector | None = None,
    ) -> None:
        self._store = state_store
        self._injector: FailureInjector = failure_injector or FailureInjector()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_failure(self, tool_name: str, params: dict) -> ToolResult | None:
        """
        Ask the FailureInjector whether this call should fail.

        Returns a ToolResult with success=False if a failure is injected,
        otherwise returns None (proceed normally).
        """
        failure = self._injector.should_fail(tool_name, params)
        if failure is None:
            return None

        mode = failure["mode"]
        reason = failure.get("reason", "Injected failure")

        if mode == FailureMode.EXECUTION_FAILURE:
            return ToolResult(
                success=False,
                error=f"[INJECTED:{mode}] {reason}",
                metadata={"failure_mode": mode, "injected": True},
            )
        if mode == FailureMode.PERMISSION_FAILURE:
            return ToolResult(
                success=False,
                error=f"[INJECTED:{mode}] Permission denied: {reason}",
                metadata={"failure_mode": mode, "injected": True},
            )
        if mode == FailureMode.PARTIAL_EXECUTION:
            # Caller gets a success=False with partial data hint
            return ToolResult(
                success=False,
                error=f"[INJECTED:{mode}] {reason}",
                metadata={"failure_mode": mode, "injected": True, "partial": True},
            )
        if mode == FailureMode.FALSE_SUCCESS:
            # Return success=True but do NOT actually mutate state
            return ToolResult(
                success=True,
                data={"injected": True, "warning": "FALSE_SUCCESS — state was NOT actually modified"},
                metadata={"failure_mode": mode, "injected": True},
            )
        if mode == FailureMode.MISSING_EVIDENCE:
            # Execute normally but poison the result so evidence won't exist
            return ToolResult(
                success=True,
                data=None,  # returning None triggers INCONCLUSIVE in evidence collection
                metadata={"failure_mode": mode, "injected": True},
            )
        if mode == FailureMode.CONTRADICTORY_STATE:
            # Execute normally — the contradictory state is set up separately
            return None
        if mode == FailureMode.TEMPORARY_FAILURE:
            return ToolResult(
                success=False,
                error=f"[INJECTED:{mode}] Temporary failure: {reason}",
                metadata={"failure_mode": mode, "injected": True, "retry_allowed": True},
            )
        return None

    # ------------------------------------------------------------------
    # Tools
    # ------------------------------------------------------------------

    def create_project(
        self,
        project_id: str,
        name: str,
        owner: str,
        **kwargs,
    ) -> ToolResult:
        """Create a new project in the sandbox."""
        params = {"project_id": project_id, "name": name, "owner": owner, **kwargs}
        injected = self._check_failure("create_project", params)
        if injected is not None:
            return injected

        if self._store.get_project(project_id) is not None:
            return ToolResult(
                success=False,
                error=f"Project '{project_id}' already exists.",
            )

        project = self._store.create_project(project_id, name, owner, **kwargs)
        return ToolResult(success=True, data=project)

    def delete_project(self, project_id: str, **kwargs) -> ToolResult:
        """Delete an existing project."""
        params = {"project_id": project_id, **kwargs}
        injected = self._check_failure("delete_project", params)
        if injected is not None:
            return injected

        deleted = self._store.delete_project(project_id)
        if not deleted:
            return ToolResult(success=False, error=f"Project '{project_id}' not found.")
        return ToolResult(success=True, data={"project_id": project_id, "deleted": True})

    def add_member(
        self,
        project_id: str,
        user_id: str,
        role: str = "member",
        **kwargs,
    ) -> ToolResult:
        """Add a user to a project."""
        params = {"project_id": project_id, "user_id": user_id, "role": role, **kwargs}
        injected = self._check_failure("add_member", params)
        if injected is not None:
            return injected

        if self._store.get_project(project_id) is None:
            return ToolResult(success=False, error=f"Project '{project_id}' does not exist.")

        member = self._store.add_member(project_id, user_id, role, **kwargs)
        return ToolResult(success=True, data=member)

    def remove_member(self, project_id: str, user_id: str, **kwargs) -> ToolResult:
        """Remove a user from a project."""
        params = {"project_id": project_id, "user_id": user_id, **kwargs}
        injected = self._check_failure("remove_member", params)
        if injected is not None:
            return injected

        removed = self._store.remove_member(project_id, user_id)
        if not removed:
            return ToolResult(
                success=False,
                error=f"Member '{user_id}' not found in project '{project_id}'.",
            )
        return ToolResult(success=True, data={"project_id": project_id, "user_id": user_id, "removed": True})

    def set_permission(
        self,
        project_id: str,
        user_id: str,
        role: str,
        **kwargs,
    ) -> ToolResult:
        """Set a user's permission role on a project."""
        params = {"project_id": project_id, "user_id": user_id, "role": role, **kwargs}
        injected = self._check_failure("set_permission", params)
        if injected is not None:
            return injected

        if self._store.get_project(project_id) is None:
            return ToolResult(success=False, error=f"Project '{project_id}' does not exist.")

        perm = self._store.set_permission(project_id, user_id, role)
        return ToolResult(success=True, data=perm)

    def upload_file(
        self,
        project_id: str,
        file_id: str,
        filename: str,
        content: str = "",
        **kwargs,
    ) -> ToolResult:
        """Upload a file to a project."""
        params = {"project_id": project_id, "file_id": file_id, "filename": filename, **kwargs}
        injected = self._check_failure("upload_file", params)
        if injected is not None:
            return injected

        if self._store.get_project(project_id) is None:
            return ToolResult(success=False, error=f"Project '{project_id}' does not exist.")

        file_data = self._store.upload_file(project_id, file_id, filename, content, **kwargs)
        return ToolResult(success=True, data=file_data)

    def delete_file(self, project_id: str, file_id: str, **kwargs) -> ToolResult:
        """Delete a file from a project."""
        params = {"project_id": project_id, "file_id": file_id, **kwargs}
        injected = self._check_failure("delete_file", params)
        if injected is not None:
            return injected

        deleted = self._store.delete_file(project_id, file_id)
        if not deleted:
            return ToolResult(
                success=False,
                error=f"File '{file_id}' not found in project '{project_id}'.",
            )
        return ToolResult(success=True, data={"project_id": project_id, "file_id": file_id, "deleted": True})

    def generate_report(
        self,
        report_id: str,
        project_id: str,
        report_type: str,
        content: str = "",
        **kwargs,
    ) -> ToolResult:
        """Generate and store a report for a project."""
        params = {"report_id": report_id, "project_id": project_id, "report_type": report_type, **kwargs}
        injected = self._check_failure("generate_report", params)
        if injected is not None:
            return injected

        report = self._store.create_report(report_id, project_id, report_type, content, **kwargs)
        return ToolResult(success=True, data=report)

    def send_notification(
        self,
        notification_id: str,
        recipient: str,
        message: str,
        **kwargs,
    ) -> ToolResult:
        """Send (record) a notification."""
        params = {"notification_id": notification_id, "recipient": recipient, "message": message, **kwargs}
        injected = self._check_failure("send_notification", params)
        if injected is not None:
            return injected

        notif = self._store.create_notification(notification_id, recipient, message, **kwargs)
        return ToolResult(success=True, data=notif)
