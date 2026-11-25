"""Compatibility spelling for upstream schedule revision logs."""

from .versions import (
    ACTIVE,
    NOT_PUBLISHED,
    NOT_RECEIVED,
    NO_VERSION,
    RESTORED,
    RESTORE_SOURCE_MISSING,
    REVOKED,
    REVOKED_SELECTED,
    VISIBLE,
    ScheduleBook,
    ScheduleSelection,
    ScheduleVersion,
    canonical_status,
    selection_record,
)


ScheduleRevision = ScheduleVersion
RevisionSelection = ScheduleSelection
RevisionBook = ScheduleBook

__all__ = [
    "ACTIVE", "RESTORED", "REVOKED", "VISIBLE", "NO_VERSION", "NOT_PUBLISHED",
    "NOT_RECEIVED", "REVOKED_SELECTED", "RESTORE_SOURCE_MISSING",
    "ScheduleBook", "ScheduleSelection", "ScheduleVersion",
    "ScheduleRevision", "RevisionSelection", "RevisionBook",
    "canonical_status", "selection_record",
]
