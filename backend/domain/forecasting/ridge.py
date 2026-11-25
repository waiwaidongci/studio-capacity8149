from bisect import bisect_right
from dataclasses import dataclass

import numpy as np

from ..intake.timeline import Timeline
from ..intake.versions import ScheduleSelection
from .features import FEATURE_NAMES, features_with_selection


RIDGE_MODEL = 'calendar_ridge'
MIN_TRAINING_ROWS = 21


@dataclass(frozen=True)
class RidgeFit:
    intercept: float
    coefficients: tuple[float, ...]
    training_rows: int
    last_target_index: int
    schedule_selections: tuple[ScheduleSelection, ...] = ()

    def predict(self, features: tuple[float, ...]) -> float:
        return self.intercept + sum(weight * value for weight, value in zip(self.coefficients, features))

    @property
    def schedule_versions(self) -> tuple[ScheduleSelection, ...]:
        return self.schedule_selections


@dataclass(frozen=True)
class RidgeTraining:
    target_indices: tuple[int, ...]
    predictors: np.ndarray
    outcomes: np.ndarray
    schedule_selections: tuple[ScheduleSelection, ...] = ()

    @classmethod
    def from_timeline(cls, timeline: Timeline, lead: int) -> 'RidgeTraining':
        indices = []
        predictors = []
        selections = []
        for target in range(lead, len(timeline.actuals)):
            result = features_with_selection(timeline, target - lead, target)
            if result is not None:
                features, selection = result
                indices.append(target)
                predictors.append(features)
                selections.append(selection)
        return cls(
            tuple(indices),
            np.asarray(predictors, dtype=float).reshape(-1, len(FEATURE_NAMES)),
            np.asarray([timeline.actuals[index] for index in indices], dtype=float),
            tuple(selections),
        )

    def fit_through(self, origin: int, alpha: float) -> RidgeFit | None:
        count = bisect_right(self.target_indices, origin)
        if count < MIN_TRAINING_ROWS:
            return None
        predictors = self.predictors[:count]
        outcomes = self.outcomes[:count]
        center = predictors.mean(axis=0)
        scale = predictors.std(axis=0)
        constant = predictors.min(axis=0) == predictors.max(axis=0)
        center[constant] = predictors[0, constant]
        scale[constant] = 1.0
        standardized = (predictors - center) / scale
        penalty = alpha * np.eye(len(FEATURE_NAMES))
        standardized_weights = np.linalg.solve(
            standardized.T @ standardized + penalty,
            standardized.T @ (outcomes - outcomes.mean()),
        )
        coefficients = standardized_weights / scale
        intercept = outcomes.mean() - center @ coefficients
        return RidgeFit(
            float(intercept), tuple(map(float, coefficients)), count,
            self.target_indices[count - 1], self.schedule_selections[:count],
        )

    @property
    def schedule_versions(self) -> tuple[ScheduleSelection, ...]:
        return self.schedule_selections
