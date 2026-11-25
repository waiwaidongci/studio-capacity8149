"""Naming-compatibility wrapper for calendar version selection."""

from .versions import ScheduleBook, ScheduleSelection, ScheduleVersion, selection_record


CalendarBook = ScheduleBook
CalendarSelection = ScheduleSelection
CalendarVersion = ScheduleVersion

__all__ = [
    "CalendarBook", "CalendarSelection", "CalendarVersion", "selection_record",
]
