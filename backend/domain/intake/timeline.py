from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone

from .schemas import ListeningDay


SHANGHAI = timezone(timedelta(hours=8))
ONE_DAY = timedelta(days=1)


def closing_cutoff(day: date) -> datetime:
    return datetime.combine(day, time(20, tzinfo=SHANGHAI))


def record_key(row: ListeningDay) -> str:
    return row.service_date.isoformat()


@dataclass(frozen=True)
class Timeline:
    days: tuple[ListeningDay, ...]
    actuals: tuple[int, ...]

    @property
    def origin_index(self) -> int:
        return len(self.actuals) - 1

    def knows_calendar(self, origin: int, target: int) -> bool:
        return self.days[target].calendar_known_at <= closing_cutoff(self.days[origin].service_date)


def build_timeline(rows: Sequence[ListeningDay]) -> Timeline:
    if not rows:
        raise ValueError('Provide a continuous sequence of service dates')
    days = tuple(sorted(rows, key=lambda row: row.service_date))
    actuals = []
    previous_date = None
    for index, row in enumerate(days):
        day = row.service_date
        if day == previous_date:
            raise ValueError(f'Duplicate service date: {day}')
        if previous_date is not None and day - previous_date != ONE_DAY:
            raise ValueError(f'Missing service date: {previous_date + ONE_DAY}')
        if row.calendar_known_at >= closing_cutoff(day):
            raise ValueError(f'Calendar for {day} must be published before that day closes')
        if row.closing_occupied_seats is not None:
            if len(actuals) != index:
                raise ValueError(f'Missing closing seat occupancy before {day}; blanks are only allowed in the future tail')
            actuals.append(row.closing_occupied_seats)
        previous_date = day
    if not actuals:
        raise ValueError('Provide at least one observed closing seat occupancy')
    return Timeline(days, tuple(actuals))


def validate_rows(rows: Sequence[ListeningDay]) -> None:
    build_timeline(rows)
