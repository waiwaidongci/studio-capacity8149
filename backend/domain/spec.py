from collections.abc import Sequence

from .intake.schemas import ListeningDay as ROW_MODEL
from .intake.schemas import ForecastOptions as OPTIONS_MODEL
from .intake.seed import SEED_NAME, seed
from .intake.timeline import build_timeline, record_key, validate_rows
from .planning.report import build_report


TITLE = '公益录音棚审听工位占用预报'
METHOD = '滚动起点回测、星期季节朴素法与日历调和岭回归'
METHOD_ID = 'listening-capacity'

__all__ = [
    'TITLE', 'METHOD', 'SEED_NAME', 'ROW_MODEL', 'OPTIONS_MODEL',
    'record_key', 'validate_rows', 'analyze', 'seed',
]


def analyze(rows: Sequence[ROW_MODEL], options: OPTIONS_MODEL) -> dict:
    return build_report(build_timeline(rows), options, METHOD_ID)
