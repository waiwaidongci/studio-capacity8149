from dataclasses import dataclass

from ..intake.timeline import Timeline
from ..intake.versions import ScheduleSelection
from ..forecasting.features import features_with_selection
from ..forecasting.ridge import RidgeTraining
from ..forecasting.seasonal import predict_weekly


@dataclass(frozen=True)
class Fold:
    origin: int
    target: int
    actual: int
    weekly_prediction: int
    ridge_prediction: float
    ridge_training_rows: int
    ridge_last_target: int
    schedule_selection: ScheduleSelection | None = None

    @property
    def weekly_error(self) -> int:
        return abs(self.actual - self.weekly_prediction)

    @property
    def ridge_error(self) -> float:
        return abs(self.actual - self.ridge_prediction)

    @property
    def selection(self) -> ScheduleSelection | None:
        return self.schedule_selection

    @property
    def calendar_selection(self) -> ScheduleSelection | None:
        return self.schedule_selection


def rolling_folds(timeline: Timeline, lead: int, training: RidgeTraining,
                  options) -> tuple[int, list[Fold]]:
    end = max(0, len(timeline.actuals) - lead)
    start = max(0, end - options.backtest_origins)
    folds = []
    for origin in range(start, end):
        target = origin + lead
        result = features_with_selection(timeline, origin, target)
        if result is None:
            continue
        _features, target_selection = result
        fit = training.fit_through(origin, options.ridge_alpha)
        if fit is None:
            continue
        weekly = predict_weekly(timeline, origin, target)
        assert weekly is not None  # Feature eligibility includes an observed seasonal source.
        folds.append(Fold(
            origin, target, timeline.actuals[target], weekly,
            max(0.0, fit.predict(_features)), fit.training_rows, fit.last_target_index,
            target_selection,
        ))
    return end - start, folds
