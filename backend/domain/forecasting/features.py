from math import cos, pi, sin

from ..intake.timeline import Timeline
from .seasonal import WEEK_LENGTH, predict_weekly


FEATURE_NAMES = (
    'origin_occupied_seats', 'previous_week_occupied_seats', 'weekday_sin', 'weekday_cos',
    'public_playback', 'review_batches',
)


def features_for(timeline: Timeline, origin: int, target: int) -> tuple[float, ...] | None:
    previous_week = predict_weekly(timeline, origin, target)
    if previous_week is None or not timeline.knows_calendar(origin, target):
        return None
    day = timeline.days[target]
    angle = 2 * pi * day.service_date.weekday() / WEEK_LENGTH
    return (
        timeline.actuals[origin], previous_week, sin(angle), cos(angle),
        int(day.public_playback), day.review_batches,
    )
