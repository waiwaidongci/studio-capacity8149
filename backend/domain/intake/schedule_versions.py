"""Naming-compatibility wrapper for schedule version selection."""

from .versions import ScheduleBook, ScheduleSelection, ScheduleVersion, selection_record


ScheduleVersions = ScheduleBook
VersionSelection = ScheduleSelection

__all__ = [
    "ScheduleBook", "ScheduleSelection", "ScheduleVersion",
    "ScheduleVersions", "VersionSelection", "selection_record",
]
