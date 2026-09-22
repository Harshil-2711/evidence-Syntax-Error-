"""
core/evidence.py

Evidence collection layer.

Responsible for snapshotting the relevant slice of sandbox state AFTER an
action executes, and packaging it as an Evidence object ready for the verifier.

ARCHITECTURAL RULE:
  Evidence is ALWAYS pulled from the sandbox state store.
  This module never constructs fake or synthetic state.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from core.models import Action, Evidence, VerificationStatus

if TYPE_CHECKING:
    from sandbox.state_store import SandboxStateStore


# ---------------------------------------------------------------------------
# Evidence source constant
# ---------------------------------------------------------------------------

SOURCE_SANDBOX = "sandbox.state_store"
SOURCE_MISSING = "sandbox.state_store.MISSING"


# ---------------------------------------------------------------------------
# Evidence collector
# ---------------------------------------------------------------------------


class EvidenceCollector:
    """
    Extracts relevant state from the sandbox and wraps it in an Evidence object.

    The collector is keyed on the action_type so it knows which slice of the
    sandbox to snapshot for each kind of tool call.
    """

    def __init__(self, state_store: "SandboxStateStore") -> None:
        self._store = state_store

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def collect(self, action: Action) -> Evidence:
        """
        Snapshot sandbox state that is relevant to *action* and return Evidence.

        If the relevant state cannot be determined (e.g., entity key is absent
        from the action parameters), the Evidence is marked INCONCLUSIVE with
        an empty observed_state. INCONCLUSIVE must NEVER become PASS.
        """
        try:
            observed = self._snapshot(action)
            source = SOURCE_SANDBOX
        except KeyError as exc:
            # Missing parameter — state observation is impossible
            observed = {"error": f"Missing parameter in action: {exc}"}
            source = SOURCE_MISSING

        evidence = Evidence.create(
            action_id=action.action_id,
            source=source,
            observed_state=observed,
            expected_state=action.expected_postcondition,
        )
        return evidence

    # ------------------------------------------------------------------
    # Snapshot routing (deterministic, no LLM involvement)
    # ------------------------------------------------------------------

    def _snapshot(self, action: Action) -> dict[str, Any]:
        """
        Return the slice of sandbox state relevant to the given action.

        Raises KeyError if a mandatory parameter is absent.
        """
        p = action.parameters
        at = action.action_type.value  # e.g. "create_project"

        dispatch = {
            "create_project":    self._snap_project,
            "delete_project":    self._snap_project,
            "add_member":        self._snap_member,
            "remove_member":     self._snap_member,
            "set_permission":    self._snap_permission,
            "upload_file":       self._snap_file,
            "delete_file":       self._snap_file,
            "generate_report":   self._snap_report,
            "send_notification": self._snap_notification,
        }

        handler = dispatch.get(at)
        if handler is None:
            return {"error": f"No snapshot handler for action_type={at}"}

        return handler(p)

    # ------------------------------------------------------------------
    # Per-entity snapshot helpers
    # ------------------------------------------------------------------

    def _snap_project(self, p: dict) -> dict:
        project_id = p["project_id"]
        project = self._store.get_project(project_id)
        return {
            "entity": "project",
            "project_id": project_id,
            "exists": project is not None,
            "data": project,
        }

    def _snap_member(self, p: dict) -> dict:
        project_id = p["project_id"]
        user_id = p["user_id"]
        member = self._store.get_member(project_id, user_id)
        project = self._store.get_project(project_id)
        return {
            "entity": "member",
            "project_id": project_id,
            "user_id": user_id,
            "project_exists": project is not None,
            "member_exists": member is not None,
            "member_data": member,
        }

    def _snap_permission(self, p: dict) -> dict:
        project_id = p["project_id"]
        user_id = p["user_id"]
        perm = self._store.get_permission(project_id, user_id)
        return {
            "entity": "permission",
            "project_id": project_id,
            "user_id": user_id,
            "permission": perm,
        }

    def _snap_file(self, p: dict) -> dict:
        project_id = p["project_id"]
        file_id = p["file_id"]
        file_data = self._store.get_file(project_id, file_id)
        return {
            "entity": "file",
            "project_id": project_id,
            "file_id": file_id,
            "exists": file_data is not None,
            "data": file_data,
        }

    def _snap_report(self, p: dict) -> dict:
        report_id = p["report_id"]
        report = self._store.get_report(report_id)
        return {
            "entity": "report",
            "report_id": report_id,
            "exists": report is not None,
            "data": report,
        }

    def _snap_notification(self, p: dict) -> dict:
        notification_id = p["notification_id"]
        notif = self._store.get_notification(notification_id)
        return {
            "entity": "notification",
            "notification_id": notification_id,
            "exists": notif is not None,
            "data": notif,
        }
