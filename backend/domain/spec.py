from collections.abc import Mapping, Sequence
from typing import Any

from .intake.schemas import ListeningDay as ROW_MODEL
from .intake.schemas import ForecastOptions as OPTIONS_MODEL
from .intake.seed import SEED_NAME, seed
from .intake.timeline import build_timeline, record_key, validate_rows
from .intake.versions import (
    ScheduleBook,
    ScheduleSelection,
    ScheduleVersion,
    selection_record,
)

ScheduleRevision = ScheduleVersion
from .planning.report import build_report


TITLE = '公益录音棚审听工位占用预报'
METHOD = '滚动起点回测、星期季节朴素法与日历调和岭回归'
METHOD_ID = 'listening-capacity'

__all__ = [
    'TITLE', 'METHOD', 'METHOD_ID', 'SEED_NAME', 'ROW_MODEL', 'OPTIONS_MODEL',
    'record_key', 'validate_rows', 'analyze', 'seed',
    'ScheduleBook', 'ScheduleVersion', 'ScheduleRevision', 'ScheduleSelection',
    'selection_record',
    'select_schedule_version', 'select_calendar_version',
    'select_visible_schedule', 'select_visible_version',
    'resolve_schedule_version', 'get_visible_schedule',
    'choose_schedule_version', 'visible_schedule_version',
    'select_calendar_at',
]


def _collect_version_argument(
    schedule_versions: Any,
    extra_args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    values = [schedule_versions, *extra_args]
    for key in ('versions', 'calendar_versions', 'revisions', 'schedule_revisions'):
        if key in kwargs:
            values.append(kwargs.pop(key))
    values = [value for value in values if value is not None]
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    # Multiple supported keyword aliases are unlikely; merge list-like inputs.
    merged = []
    for value in values:
        if isinstance(value, Mapping):
            merged.append(value)
        elif isinstance(value, (list, tuple)):
            merged.extend(value)
        else:
            merged.append(value)
    return merged


def select_schedule_version(
    rows_or_book: Any,
    service_date: Any = None,
    as_of: Any = None,
    *args: Any,
    **kwargs: Any,
) -> ScheduleSelection:
    """Small public helper used by point-in-time schedule audits.

    It accepts either a :class:`ScheduleBook` or the raw/day rows accepted by
    :func:`build_timeline`.  Date/timestamp arguments may be passed positionally
    or using the common ``day``/``cutoff``/``at`` keyword aliases.
    """
    from datetime import date as date_cls, datetime

    if isinstance(rows_or_book, ScheduleBook):
        book = rows_or_book
    elif (isinstance(rows_or_book, (list, tuple)) and rows_or_book
          and all(isinstance(item, ScheduleVersion) for item in rows_or_book)):
        book = ScheduleBook(rows_or_book)
    else:
        extra_versions = kwargs.pop('schedule_versions', None)
        timeline = build_timeline(rows_or_book, extra_versions)
        book = timeline.schedule_book

    if service_date is None:
        service_date = kwargs.pop('day', None) or kwargs.pop('target_date', None)
    if as_of is None:
        as_of = kwargs.pop('cutoff', None) or kwargs.pop('at', None) or kwargs.pop('issued_at', None)
    if args:
        if service_date is None:
            service_date = args[0]
        elif as_of is None:
            as_of = args[0]
    if isinstance(service_date, str):
        service_date = date_cls.fromisoformat(service_date)
    if isinstance(as_of, str):
        as_of = datetime.fromisoformat(as_of)
    return book.select(service_date, as_of)


select_calendar_version = select_schedule_version
select_visible_schedule = select_schedule_version
select_visible_version = select_schedule_version
resolve_schedule_version = select_schedule_version
get_visible_schedule = select_schedule_version
choose_schedule_version = select_schedule_version
visible_schedule_version = select_schedule_version
select_calendar_at = select_schedule_version


def analyze(
    rows: Sequence[ROW_MODEL],
    options: OPTIONS_MODEL | Mapping | None = None,
    schedule_versions: Any | None = None,
    *extra_version_args: Any,
    **kwargs: Any,
) -> dict:
    versions = _collect_version_argument(
        schedule_versions, extra_version_args, kwargs
    )
    if options is None:
        options = OPTIONS_MODEL()
    elif isinstance(options, Mapping):
        options = OPTIONS_MODEL.model_validate(options)
    timeline = build_timeline(rows, versions)
    return build_report(timeline, options, METHOD_ID)
