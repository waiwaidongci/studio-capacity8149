from dataclasses import dataclass

from ..forecasting.features import features_for
from ..forecasting.ridge import RidgeTraining
from ..forecasting.seasonal import predict_weekly
from ..intake.schemas import ForecastOptions
from ..intake.timeline import Timeline


@dataclass(frozen=True)
class Fold:
    origin: int
    target: int
    actual: int
    weekly_prediction: int
    ridge_prediction: float
    ridge_training_rows: int
    ridge_last_target: int

    @property
    def weekly_error(self) -> int:
        return abs(self.actual - self.weekly_prediction)

    @property
    def ridge_error(self) -> float:
        return abs(self.actual - self.ridge_prediction)


def rolling_folds(timeline: Timeline, lead: int, training: RidgeTraining,
                  options: ForecastOptions) -> tuple[int, list[Fold]]:
    end = max(0, len(timeline.actuals) - lead)
    start = max(0, end - options.backtest_origins)
    folds = []
    for origin in range(start, end):
        target = origin + lead
        features = features_for(timeline, origin, target)
        if features is None:
            continue
        fit = training.fit_through(origin, options.ridge_alpha)
        if fit is None:
            continue
        weekly = predict_weekly(timeline, origin, target)
        assert weekly is not None  # Feature eligibility includes an observed seasonal source.
        folds.append(Fold(
            origin, target, timeline.actuals[target], weekly,
            max(0.0, fit.predict(features)), fit.training_rows, fit.last_target_index,
        ))
    return end - start, folds
