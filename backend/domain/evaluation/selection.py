from dataclasses import dataclass
from statistics import mean

from ..forecasting.ridge import RIDGE_MODEL
from ..forecasting.seasonal import WEEKLY_MODEL
from .backtest import Fold


MIN_PAIRED_ORIGINS = 4
TIE_TOLERANCE = 1e-9


@dataclass(frozen=True)
class Selection:
    model: str
    reason: str
    weekly_mae: float | None
    ridge_mae: float | None


def select_model(folds: list[Fold]) -> Selection:
    weekly_mae = float(mean(fold.weekly_error for fold in folds)) if folds else None
    ridge_mae = mean(fold.ridge_error for fold in folds) if folds else None
    if len(folds) < MIN_PAIRED_ORIGINS:
        return Selection(WEEKLY_MODEL, 'insufficient_paired_origins', weekly_mae, ridge_mae)
    if ridge_mae + TIE_TOLERANCE < weekly_mae:
        return Selection(RIDGE_MODEL, 'lower_mae', weekly_mae, ridge_mae)
    reason = 'mae_tie' if abs(ridge_mae - weekly_mae) <= TIE_TOLERANCE else 'lower_mae'
    return Selection(WEEKLY_MODEL, reason, weekly_mae, ridge_mae)
