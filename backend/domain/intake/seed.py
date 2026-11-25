from datetime import date, timedelta
from math import cos, pi, sin
from random import Random

from .timeline import closing_cutoff


SEED_NAME = '回声桥公益录音棚_B室审听排期与闭室_20261124.csv'
FIRST_DATE = date(2026, 8, 5)
OBSERVED_DAYS = 112
PLAYBACK_DAY = date(2026, 11, 27)


def seed() -> list[dict]:
    random = Random(88025)
    rows = []
    for index in range(OBSERVED_DAYS + 7):
        day = FIRST_DATE + timedelta(days=index)
        weekday = day.weekday()
        public_playback = index % 19 in (3, 11) or day == PLAYBACK_DAY
        batches = random.randint(5, 23) + 3 * public_playback
        if day == PLAYBACK_DAY:
            batches = 31
        weekly_angle = 2 * pi * weekday / 7
        occupied = round(
            5 + 0.65 * batches + 1.5 * sin(weekly_angle) - cos(weekly_angle)
            + 3 * public_playback + random.randint(-1, 1)
        )
        rows.append({
            'service_date': day.isoformat(),
            'calendar_known_at': (closing_cutoff(day) - timedelta(days=10, hours=5)).isoformat(),
            'public_playback': public_playback,
            'review_batches': batches,
            'capacity_seats': 20 if day == PLAYBACK_DAY else 24,
            'closing_occupied_seats': occupied if index < OBSERVED_DAYS else None,
        })
    return rows
