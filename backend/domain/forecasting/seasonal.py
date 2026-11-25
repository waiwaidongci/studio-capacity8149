from ..intake.timeline import Timeline


WEEK_LENGTH = 7
WEEKLY_MODEL = 'weekly_naive'


def weekly_source_index(target: int) -> int:
    return target - WEEK_LENGTH


def predict_weekly(timeline: Timeline, origin: int, target: int) -> int | None:
    source = weekly_source_index(target)
    if 0 <= source <= origin:
        return timeline.actuals[source]
    return None
