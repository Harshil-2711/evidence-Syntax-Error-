"""
sandbox/state_store.py

In-memory sandbox state store.

Entities managed:
  - projects
  - members      (per project)
  - permissions  (per project/user)
  - files        (per project)
  - notifications
  - reports

All mutations are journaled so callers can inspect history, and so
rollback operations can be applied deterministically.

This store is the SINGLE SOURCE OF TRUTH for evidence collection.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


# ---------------------------------------------------------------------------
# Journal entry (immutable audit record)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class JournalEntry:
    operation: str           # e.g. "create_project"
    entity_type: str         # e.g. "project"
    entity_id: str
    snapshot_before: Any     # deep-copy of state before mutation
    snapshot_after: Any      # deep-copy of state after mutation
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# State store
# ---------------------------------------------------------------------------


class SandboxStateStore:
    """
    Thread-unsafe (single-threaded) in-memory store for the agent sandbox.

    Every mutating method appends a JournalEntry so actions can be
    replayed or reversed during rollback.
    """

    def __init__(self) -> None:
        # Primary data stores
        self._projects: dict[str, dict] = {}
        # members[project_id][user_id] = member dict
        self._members: dict[str, dict[str, dict]] = {}
        # permissions[project_id][user_id] = role string
        self._permissions: dict[str, dict[str, str]] = {}
        # files[project_id][file_id] = file dict
        self._files: dict[str, dict[str, dict]] = {}
        self._notifications: dict[str, dict] = {}
        self._reports: dict[str, dict] = {}

        # Audit journal
        self._journal: list[JournalEntry] = []

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def journal(self) -> list[JournalEntry]:
        return list(self._journal)

    def snapshot(self) -> dict:
        """Return a deep copy of the entire state (useful for testing)."""
        return {
            "projects": copy.deepcopy(self._projects),
            "members": copy.deepcopy(self._members),
            "permissions": copy.deepcopy(self._permissions),
            "files": copy.deepcopy(self._files),
            "notifications": copy.deepcopy(self._notifications),
            "reports": copy.deepcopy(self._reports),
        }

    # ------------------------------------------------------------------
    # Projects
    # ------------------------------------------------------------------

    def get_project(self, project_id: str) -> dict | None:
        return copy.deepcopy(self._projects.get(project_id))

    def create_project(self, project_id: str, name: str, owner: str, **kwargs) -> dict:
        before = copy.deepcopy(self._projects.get(project_id))
        project = {
            "project_id": project_id,
            "name": name,
            "owner": owner,
            "created_at": datetime.now(timezone.utc).isoformat(),
            **kwargs,
        }
        self._projects[project_id] = project
        self._members.setdefault(project_id, {})
        self._permissions.setdefault(project_id, {})
        self._files.setdefault(project_id, {})
        self._journal.append(JournalEntry(
            operation="create_project",
            entity_type="project",
            entity_id=project_id,
            snapshot_before=before,
            snapshot_after=copy.deepcopy(project),
        ))
        return copy.deepcopy(project)

    def delete_project(self, project_id: str) -> bool:
        before = copy.deepcopy(self._projects.get(project_id))
        if project_id not in self._projects:
            return False
        del self._projects[project_id]
        self._members.pop(project_id, None)
        self._permissions.pop(project_id, None)
        self._files.pop(project_id, None)
        self._journal.append(JournalEntry(
            operation="delete_project",
            entity_type="project",
            entity_id=project_id,
            snapshot_before=before,
            snapshot_after=None,
        ))
        return True

    # ------------------------------------------------------------------
    # Members
    # ------------------------------------------------------------------

    def get_member(self, project_id: str, user_id: str) -> dict | None:
        return copy.deepcopy(self._members.get(project_id, {}).get(user_id))

    def add_member(self, project_id: str, user_id: str, role: str = "member", **kwargs) -> dict:
        before = copy.deepcopy(self._members.get(project_id, {}).get(user_id))
        member = {
            "user_id": user_id,
            "project_id": project_id,
            "role": role,
            "joined_at": datetime.now(timezone.utc).isoformat(),
            **kwargs,
        }
        self._members.setdefault(project_id, {})[user_id] = member
        self._journal.append(JournalEntry(
            operation="add_member",
            entity_type="member",
            entity_id=f"{project_id}/{user_id}",
            snapshot_before=before,
            snapshot_after=copy.deepcopy(member),
        ))
        return copy.deepcopy(member)

    def remove_member(self, project_id: str, user_id: str) -> bool:
        before = copy.deepcopy(self._members.get(project_id, {}).get(user_id))
        if user_id not in self._members.get(project_id, {}):
            return False
        del self._members[project_id][user_id]
        self._permissions.get(project_id, {}).pop(user_id, None)
        self._journal.append(JournalEntry(
            operation="remove_member",
            entity_type="member",
            entity_id=f"{project_id}/{user_id}",
            snapshot_before=before,
            snapshot_after=None,
        ))
        return True

    # ------------------------------------------------------------------
    # Permissions
    # ------------------------------------------------------------------

    def get_permission(self, project_id: str, user_id: str) -> str | None:
        return self._permissions.get(project_id, {}).get(user_id)

    def set_permission(self, project_id: str, user_id: str, role: str) -> dict:
        before = self._permissions.get(project_id, {}).get(user_id)
        self._permissions.setdefault(project_id, {})[user_id] = role
        perm = {"project_id": project_id, "user_id": user_id, "role": role}
        self._journal.append(JournalEntry(
            operation="set_permission",
            entity_type="permission",
            entity_id=f"{project_id}/{user_id}",
            snapshot_before=before,
            snapshot_after=role,
        ))
        return perm

    # ------------------------------------------------------------------
    # Files
    # ------------------------------------------------------------------

    def get_file(self, project_id: str, file_id: str) -> dict | None:
        return copy.deepcopy(self._files.get(project_id, {}).get(file_id))

    def upload_file(self, project_id: str, file_id: str, filename: str, content: str = "", **kwargs) -> dict:
        before = copy.deepcopy(self._files.get(project_id, {}).get(file_id))
        file_data = {
            "file_id": file_id,
            "project_id": project_id,
            "filename": filename,
            "content": content,
            "uploaded_at": datetime.now(timezone.utc).isoformat(),
            **kwargs,
        }
        self._files.setdefault(project_id, {})[file_id] = file_data
        self._journal.append(JournalEntry(
            operation="upload_file",
            entity_type="file",
            entity_id=f"{project_id}/{file_id}",
            snapshot_before=before,
            snapshot_after=copy.deepcopy(file_data),
        ))
        return copy.deepcopy(file_data)

    def delete_file(self, project_id: str, file_id: str) -> bool:
        before = copy.deepcopy(self._files.get(project_id, {}).get(file_id))
        if file_id not in self._files.get(project_id, {}):
            return False
        del self._files[project_id][file_id]
        self._journal.append(JournalEntry(
            operation="delete_file",
            entity_type="file",
            entity_id=f"{project_id}/{file_id}",
            snapshot_before=before,
            snapshot_after=None,
        ))
        return True

    # ------------------------------------------------------------------
    # Reports
    # ------------------------------------------------------------------

    def get_report(self, report_id: str) -> dict | None:
        return copy.deepcopy(self._reports.get(report_id))

    def create_report(self, report_id: str, project_id: str, report_type: str, content: str = "", **kwargs) -> dict:
        before = copy.deepcopy(self._reports.get(report_id))
        report = {
            "report_id": report_id,
            "project_id": project_id,
            "report_type": report_type,
            "content": content,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            **kwargs,
        }
        self._reports[report_id] = report
        self._journal.append(JournalEntry(
            operation="create_report",
            entity_type="report",
            entity_id=report_id,
            snapshot_before=before,
            snapshot_after=copy.deepcopy(report),
        ))
        return copy.deepcopy(report)

    # ------------------------------------------------------------------
    # Notifications
    # ------------------------------------------------------------------

    def get_notification(self, notification_id: str) -> dict | None:
        return copy.deepcopy(self._notifications.get(notification_id))

    def create_notification(self, notification_id: str, recipient: str, message: str, **kwargs) -> dict:
        before = copy.deepcopy(self._notifications.get(notification_id))
        notif = {
            "notification_id": notification_id,
            "recipient": recipient,
            "message": message,
            "sent_at": datetime.now(timezone.utc).isoformat(),
            **kwargs,
        }
        self._notifications[notification_id] = notif
        self._journal.append(JournalEntry(
            operation="create_notification",
            entity_type="notification",
            entity_id=notification_id,
            snapshot_before=before,
            snapshot_after=copy.deepcopy(notif),
        ))
        return copy.deepcopy(notif)

    # ------------------------------------------------------------------
    # Rollback helpers
    # ------------------------------------------------------------------

    def rollback_last(self) -> JournalEntry | None:
        """
        Undo the most recent mutation by restoring the snapshot_before.

        Returns the journal entry that was reversed, or None if journal empty.
        """
        if not self._journal:
            return None
        entry = self._journal.pop()
        self._restore(entry)
        return entry

    def rollback_to_length(self, target_length: int) -> list[JournalEntry]:
        """
        Roll back journal entries until len(journal) == target_length.

        Returns the list of reversed entries (most-recent first).

        Raises ValueError if target_length is negative (programming error).
        """
        if target_length < 0:
            raise ValueError(
                f"rollback_to_length: target_length must be >= 0, got {target_length}"
            )
        reversed_entries: list[JournalEntry] = []
        while len(self._journal) > target_length:
            entry = self._journal.pop()
            self._restore(entry)
            reversed_entries.append(entry)
        return reversed_entries

    def _restore(self, entry: JournalEntry) -> None:
        """Apply snapshot_before to undo a single mutation."""
        et = entry.entity_type
        eid = entry.entity_id

        if et == "project":
            if entry.snapshot_before is None:
                self._projects.pop(eid, None)
            else:
                self._projects[eid] = entry.snapshot_before

        elif et == "member":
            project_id, user_id = eid.split("/", 1)
            proj_members = self._members.setdefault(project_id, {})
            if entry.snapshot_before is None:
                proj_members.pop(user_id, None)
            else:
                proj_members[user_id] = entry.snapshot_before

        elif et == "permission":
            project_id, user_id = eid.split("/", 1)
            proj_perms = self._permissions.setdefault(project_id, {})
            if entry.snapshot_before is None:
                proj_perms.pop(user_id, None)
            else:
                proj_perms[user_id] = entry.snapshot_before

        elif et == "file":
            project_id, file_id = eid.split("/", 1)
            proj_files = self._files.setdefault(project_id, {})
            if entry.snapshot_before is None:
                proj_files.pop(file_id, None)
            else:
                proj_files[file_id] = entry.snapshot_before

        elif et == "report":
            if entry.snapshot_before is None:
                self._reports.pop(eid, None)
            else:
                self._reports[eid] = entry.snapshot_before

        elif et == "notification":
            if entry.snapshot_before is None:
                self._notifications.pop(eid, None)
            else:
                self._notifications[eid] = entry.snapshot_before
