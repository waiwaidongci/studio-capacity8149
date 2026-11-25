from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from typing import Any

from pydantic import ValidationError

from .schemas import ListeningDay
from .versions import ScheduleBook, ScheduleSelection, ScheduleVersion, make_checksum


SHANGHAI = timezone(timedelta(hours=8))
ONE_DAY = timedelta(days=1)


def closing_cutoff(day: date) -> datetime:
    return datetime.combine(day, time(20, tzinfo=SHANGHAI))


def record_key(row: ListeningDay | Mapping | object) -> str:
    if isinstance(row, Mapping):
        value = row.get("service_date")
        if value is None:
            for alias in ("date", "target_date", "day", "schedule_date"):
                if row.get(alias) is not None:
                    value = row[alias]
                    break
        return value if isinstance(value, str) else value.isoformat()
    return row.service_date.isoformat()


def _as_day(row: ListeningDay | Mapping | object, *, allow_shell: bool = False) -> ListeningDay:
    if isinstance(row, ListeningDay):
        return row
    if isinstance(row, Mapping):
        try:
            return ListeningDay.model_validate(row)
        except ValidationError:
            if not allow_shell:
                raise
            # A separate schedule-version log may be supplied, so the base rows
            # only need to identify dates and optional closing observations.
            service_date = row.get('service_date')
            if isinstance(service_date, str):
                service_date = date.fromisoformat(service_date)
            closing = row.get('closing_occupied_seats')
            if isinstance(closing, bool):
                raise ValueError('counts must be integers')
            if closing is not None:
                closing = int(closing)
                if not 0 <= closing <= 1_000_000:
                    raise ValueError('closing_occupied_seats is outside the allowed range')
            return ListeningDay.model_construct(
                service_date=service_date,
                closing_occupied_seats=closing,
                schedule_versions=None,
            )
    # Last-resort support for test doubles/objects with the documented attrs.
    return row  # type: ignore[return-value]


def _normalize_version_argument(schedule_versions: Any) -> list[Any]:
    if schedule_versions is None:
        return []
    if isinstance(schedule_versions, Mapping):
        result = []
        for service_date, value in schedule_versions.items():
            items = value if isinstance(value, (list, tuple)) else [value]
            for item in items:
                if isinstance(item, Mapping):
                    entry = dict(item)
                    entry.setdefault('service_date', service_date)
                    result.append(entry)
                else:
                    result.append(item)
        return result
    if isinstance(schedule_versions, (list, tuple)):
        return list(schedule_versions)
    raise TypeError('schedule_versions must be a list or a date-keyed mapping')


def version_from_revision(revision, service_date: date, source_index: int) -> ScheduleVersion:
    published = revision.calendar_known_at
    received = revision.received_at or published
    checksum = make_checksum(
        service_date=service_date,
        version=getattr(revision, "schedule_version", None),
        published_at=published,
        received_at=received,
        status=revision.schedule_status or ("revoked" if revision.revoked else "active"),
        public_playback=revision.public_playback,
        review_batches=revision.review_batches,
        capacity_seats=revision.capacity_seats,
        restores_version=getattr(revision, "restores_version", None),
    )
    return ScheduleVersion(
        service_date=service_date,
        published_at=published,
        received_at=received,
        status=revision.schedule_status or "active",
        public_playback=revision.public_playback,
        review_batches=revision.review_batches,
        capacity_seats=revision.capacity_seats,
        version=getattr(revision, "schedule_version", None),
        source_index=source_index,
        checksum=checksum,
        restored=bool(getattr(revision, "restored", False)),
        restores_version=getattr(revision, "restores_version", None),
    )


def book_from_rows(rows: Sequence[ListeningDay]) -> ScheduleBook:
    versions: list[ScheduleVersion] = []
    for index, row in enumerate(rows):
        day = row.service_date
        if getattr(row, 'calendar_known_at', None) is not None:
            versions.append(version_from_revision(row, day, index))
        for nested_index, revision in enumerate(row.schedule_versions or ()):
            if revision.calendar_known_at is None:
                continue
            versions.append(version_from_revision(
                revision, day, index * 10_000 + nested_index + 1
            ))
    return ScheduleBook(versions)


def _is_versioned_group(group: Sequence[ListeningDay]) -> bool:
    if len(group) < 2:
        return False
    if any(row.schedule_version is not None for row in group):
        return True
    if any(row.schedule_status == "revoked" or row.revoked for row in group):
        return True
    published = {row.calendar_known_at for row in group}
    received = {row.received_at for row in group}
    return len(published) > 1 or len(received) > 1


@dataclass(frozen=True)
class Timeline:
    days: tuple[ListeningDay, ...]
    actuals: tuple[int, ...]
    schedule_book: ScheduleBook | None = None
    date_index: Mapping[date, int] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        if self.schedule_book is None:
            object.__setattr__(self, "schedule_book", book_from_rows(self.days))
        if not self.date_index:
            object.__setattr__(
                self, "date_index", {day.service_date: index for index, day in enumerate(self.days)}
            )

    @property
    def origin_index(self) -> int:
        return len(self.actuals) - 1

    def index_of(self, day: date) -> int:
        return self.date_index[day]

    def selection_for(self, origin: int, target: int) -> ScheduleSelection:
        as_of = closing_cutoff(self.days[origin].service_date)
        return self.schedule_book.select(self.days[target].service_date, as_of)

    def selection_for_date(self, service_date: date, as_of: datetime) -> ScheduleSelection:
        return self.schedule_book.select(service_date, as_of)

    def select_version(self, origin: int, target: int) -> ScheduleSelection:
        return self.selection_for(origin, target)

    def visible_schedule(self, origin: int, target: int) -> ScheduleVersion | None:
        selection = self.selection_for(origin, target)
        return selection.effective_version if selection.visible else None

    def knows_calendar(self, origin: int, target: int) -> bool:
        return self.selection_for(origin, target).visible


def build_timeline(
    rows: Sequence[ListeningDay | Mapping],
    schedule_versions: Any | None = None,
) -> Timeline:
    if not rows and not schedule_versions:
        raise ValueError("Provide a continuous sequence of service dates")

    has_external_versions = schedule_versions is not None
    normalized = [_as_day(row, allow_shell=has_external_versions) for row in rows]
    normalized.extend(_as_day(item) for item in _normalize_version_argument(schedule_versions))
    if not normalized:
        raise ValueError("Provide a continuous sequence of service dates")

    groups: dict[date, list[ListeningDay]] = {}
    for row in normalized:
        groups.setdefault(row.service_date, []).append(row)

    service_dates = tuple(sorted(groups))
    previous_date = None
    actuals: list[int] = []
    unique_days: list[ListeningDay] = []

    for index, day in enumerate(service_dates):
        group = groups[day]
        if len(group) > 1 and not _is_versioned_group(group):
            raise ValueError(f"Duplicate service date: {day}")

        if previous_date is not None and day - previous_date != ONE_DAY:
            raise ValueError(f"Missing service date: {previous_date + ONE_DAY}")

        # Every publication, including nested revisions, must satisfy the
        # original business bound of being before the target day's 20:00 close.
        for row in group:
            revisions = [row, *(row.schedule_versions or ())]
            for revision in revisions:
                published_at = getattr(revision, 'calendar_known_at', None)
                if published_at is not None and published_at >= closing_cutoff(day):
                    raise ValueError(
                        f"Calendar for {day} must be published before that day closes"
                    )

        observed = {
            row.closing_occupied_seats
            for row in group
            if row.closing_occupied_seats is not None
        }
        if len(observed) > 1:
            raise ValueError(f"Conflicting closing seat occupancy for {day}")
        if observed:
            if len(actuals) != index:
                raise ValueError(
                    f"Missing closing seat occupancy before {day}; "
                    "blanks are only allowed in the future tail"
                )
            actuals.append(next(iter(observed)))

        # The representative only supplies the date/index.  All schedule values
        # must be obtained through ScheduleBook and its point-in-time selection.
        unique_days.append(group[0])
        previous_date = day

    if not actuals:
        raise ValueError("Provide at least one observed closing seat occupancy")

    book = book_from_rows(normalized)
    return Timeline(tuple(unique_days), tuple(actuals), book)


def validate_rows(
    rows: Sequence[ListeningDay | Mapping],
    schedule_versions: Any | None = None,
) -> None:
    build_timeline(rows, schedule_versions)
