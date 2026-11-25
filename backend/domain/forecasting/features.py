from math import cos, pi, sin

from ..intake.timeline import Timeline
from ..intake.versions import ScheduleSelection
from .seasonal import WEEK_LENGTH, predict_weekly


FEATURE_NAMES = (
    'origin_occupied_seats', 'previous_week_occupied_seats', 'weekday_sin', 'weekday_cos',
    'public_playback', 'review_batches',
)


def features_with_selection(
    timeline: Timeline, origin: int, target: int
) -> tuple[tuple[float, ...], ScheduleSelection] | None:
    previous_week = predict_weekly(timeline, origin, target)
    selection = timeline.selection_for(origin, target)
    if previous_week is None or not selection.visible:
        return None
    day = timeline.days[target]
    version = selection.effective_version or selection.version
    assert version is not None
    angle = 2 * pi * day.service_date.weekday() / WEEK_LENGTH
    return (
        timeline.actuals[origin], previous_week, sin(angle), cos(angle),
        int(version.public_playback), version.review_batches,
    ), selection


def features_for(timeline: Timeline, origin: int, target: int) -> tuple[float, ...] | None:
    result = features_with_selection(timeline, origin, target)
    return None if result is None else result[0]
