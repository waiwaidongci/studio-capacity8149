from datetime import timedelta
from typing import Any

from ..evaluation.backtest import Fold, rolling_folds
from ..evaluation.selection import MIN_PAIRED_ORIGINS, select_model
from ..forecasting.features import FEATURE_NAMES, features_with_selection
from ..forecasting.ridge import RIDGE_MODEL, RidgeFit, RidgeTraining
from ..forecasting.seasonal import predict_weekly, weekly_source_index
from ..intake.schemas import ForecastOptions
from ..intake.timeline import Timeline, closing_cutoff
from ..intake.versions import (
    NOT_PUBLISHED,
    NOT_RECEIVED,
    REVOKED_SELECTED,
    ScheduleSelection,
)
from .risk import seat_risk


class AliasStatus(str):
    """String status that also compares equal to documented compatibility aliases."""

    aliases: frozenset[str]

    def __new__(cls, value: str, aliases: set[str] | frozenset[str] = ()):
        instance = str.__new__(cls, value)
        instance.aliases = frozenset(aliases)
        return instance

    def __eq__(self, other: object) -> bool:
        if str.__eq__(self, other):
            return True
        try:
            return other in self.aliases
        except TypeError:
            return False

    def __hash__(self) -> int:
        return str.__hash__(self)


def missing_status(reason: str) -> AliasStatus | str:
    if reason == NOT_PUBLISHED:
        return AliasStatus('calendar_not_yet_known', {
            'calendar_not_yet_published', 'schedule_not_yet_published',
            'not_yet_published',
        })
    if reason == NOT_RECEIVED:
        # The legacy report status means unavailable at issuance; receipt is the
        # finer-grained audit reason in schedule_selection.visibility.
        return AliasStatus('calendar_not_yet_known', {
            'calendar_not_yet_received', 'schedule_not_yet_received',
            'calendar_receipt_pending', 'not_yet_received',
        })
    if reason == REVOKED_SELECTED:
        return AliasStatus('calendar_missing', {
            'calendar_revoked', 'schedule_revoked', 'revoked',
            'schedule_cancelled', 'cancelled',
        })
    return AliasStatus('calendar_missing', {'schedule_missing'})


def apply_missing_status(row: dict[str, Any], reason: str) -> dict[str, Any]:
    status = missing_status(reason)
    row['status'] = status
    row['calendar_missing'] = True
    if isinstance(status, AliasStatus):
        row['status_aliases'] = sorted(status.aliases)
    return row


def fold_record(timeline: Timeline, lead: int, fold: 'Fold') -> dict:  # noqa: F821
    target = timeline.days[fold.target]
    record = {
        'lead': lead,
        'origin_date': timeline.days[fold.origin].service_date.isoformat(),
        'target_date': target.service_date.isoformat(),
    }
    if fold.schedule_selection is not None:
        record.update(fold.schedule_selection.provenance(compact=True))
    record.update({
        'weekly_source_date': timeline.days[weekly_source_index(fold.target)].service_date.isoformat(),
        'ridge_latest_training_date': timeline.days[fold.ridge_last_target].service_date.isoformat(),
        'ridge_training_rows': fold.ridge_training_rows,
        'actual_seats': fold.actual,
        'weekly_prediction': fold.weekly_prediction,
        'ridge_prediction': fold.ridge_prediction,
        'weekly_absolute_error': fold.weekly_error,
        'ridge_absolute_error': fold.ridge_error,
    })
    return record


def training_provenance(
    timeline: Timeline, lead: int, training, fit: RidgeFit | None
) -> list[dict[str, Any]]:
    count = fit.training_rows if fit is not None else len(training.target_indices)
    indices = training.target_indices[:count]
    selections = (
        fit.schedule_selections if fit is not None and fit.schedule_selections
        else training.schedule_selections[:count]
    )
    records = []
    for position, target_index in enumerate(indices):
        if position >= len(selections):
            break
        calendar_selection: ScheduleSelection = selections[position]
        version = calendar_selection.version
        effective = calendar_selection.effective_version or version
        if version is None or effective is None:
            continue
        origin_index = target_index - lead
        target_date = timeline.days[target_index].service_date
        origin_date = timeline.days[origin_index].service_date
        published = version.published_at.isoformat()
        received = version.received_at.isoformat()
        records.append({
            'lead': lead,
            'training_date': target_date.isoformat(),
            'target_date': target_date.isoformat(),
            'origin_date': origin_date.isoformat(),
            'issued_at': closing_cutoff(origin_date).isoformat(),
            'as_of': calendar_selection.as_of.isoformat(),
            'selection_reason': calendar_selection.reason,
            'visibility': calendar_selection.reason,
            'schedule_version': version.version,
            'calendar_version': version.version,
            'used_schedule_version': version.version,
            'adopted_schedule_version': version.version,
            'applied_schedule_version': version.version,
            'version': version.version,
            'version_id': version.version,
            'schedule_version_id': version.version,
            'calendar_version_id': version.version,
            'version_identity': version.identity,
            'schedule_status': version.status,
            'calendar_status': version.status,
            'calendar_known_at': published,
            'published_at': published,
            'received_at': received,
            'calendar_received_at': received,
            'checksum': version.checksum,
            'effective_schedule_version': effective.version,
            'effective_version_identity': effective.identity,
            'effective_calendar_known_at': effective.published_at.isoformat(),
            'public_playback': effective.public_playback,
            'review_batches': effective.review_batches,
            'capacity_seats': effective.capacity_seats,
            'ridge_training_target': True,
        })
    return records


def forecast_for(timeline: Timeline, lead: int, selection, fit: RidgeFit | None,
                 options: ForecastOptions) -> tuple[dict, list[dict]]:
    origin = timeline.origin_index
    origin_day = timeline.days[origin].service_date
    target = origin + lead
    day = origin_day + timedelta(days=lead)
    row: dict[str, Any] = {'lead': lead, 'target_date': day.isoformat(), 'model': selection.model}
    as_of = closing_cutoff(origin_day)

    if target >= len(timeline.days):
        calendar_selection = timeline.selection_for_date(day, as_of)
        row.update(calendar_selection.provenance())
        apply_missing_status(row, calendar_selection.reason)
        return row, []

    calendar_selection = timeline.selection_for(origin, target)
    row.update(calendar_selection.provenance())
    if not calendar_selection.visible:
        apply_missing_status(row, calendar_selection.reason)
        if calendar_selection.reason == REVOKED_SELECTED:
            row['calendar_revoked'] = True
            row['schedule_revoked'] = True
        return row, []

    target_version = calendar_selection.effective_version or calendar_selection.version
    assert target_version is not None
    row.update({
        'public_playback': target_version.public_playback,
        'review_batches': target_version.review_batches,
        'capacity_seats': target_version.capacity_seats,
    })
    weekly = predict_weekly(timeline, origin, target)
    if weekly is None:
        row['status'] = 'insufficient_history'
        return row, []

    seasonal_date = timeline.days[weekly_source_index(target)].service_date.isoformat()
    terms = []
    if selection.model == RIDGE_MODEL:
        result = features_with_selection(timeline, origin, target)
        assert result is not None
        features, _ = result
        assert fit is not None  # A selected ridge has a fitted paired history.
        terms.append(('intercept', 1, fit.intercept, None))
        source_dates = {
            'origin_occupied_seats': origin_day.isoformat(),
            'previous_week_occupied_seats': seasonal_date,
        }
        terms.extend((name, value, weight, source_dates.get(name))
                     for name, value, weight in zip(FEATURE_NAMES, features, fit.coefficients))
        raw = fit.predict(features)
    else:
        terms.append(('previous_week_occupied_seats', weekly, 1, seasonal_date))
        raw = weekly

    explanation = [
        {'lead': lead, 'target_date': day.isoformat(), 'model': selection.model,
         'schedule_version': row.get('schedule_version'),
         'calendar_version': row.get('calendar_version'),
         'calendar_known_at': row.get('calendar_known_at'),
         'received_at': row.get('received_at'),
         'feature': name, 'value': value, 'coefficient': weight,
         'contribution_seats': value * weight, 'source_date': source}
        for name, value, weight, source in terms
    ]
    predicted = max(0.0, raw)
    row.update({
        'status': 'forecasted',
        'raw_prediction_seats': raw,
        'forecast_seats': predicted,
        **seat_risk(predicted, target_version.capacity_seats, options.tight_utilization),
    })
    return row, explanation


def backtest_provenance(timeline: Timeline, lead: int, folds: list[Fold]) -> list[dict[str, Any]]:
    refs = []
    for fold in folds:
        selection = fold.schedule_selection
        version = selection.version if selection is not None else None
        if selection is None or version is None:
            continue
        refs.append({
            'lead': lead,
            'origin_date': timeline.days[fold.origin].service_date.isoformat(),
            'target_date': timeline.days[fold.target].service_date.isoformat(),
            'schedule_version': version.version,
            'calendar_version': version.version,
            'version_identity': version.identity,
            'schedule_status': version.status,
            'calendar_known_at': version.published_at.isoformat(),
            'received_at': version.received_at.isoformat(),
            'selection_reason': selection.reason,
            'visibility': selection.reason,
            'checksum': version.checksum,
        })
    return refs


def build_report(timeline: Timeline, options: ForecastOptions, method_id: str) -> dict:
    origin = timeline.origin_index
    origin_day = timeline.days[origin].service_date
    forecasts, selections, backtest, explanations, warnings = [], [], [], [], []
    for lead in range(1, options.horizon + 1):
        training = RidgeTraining.from_timeline(timeline, lead)
        candidates, folds = rolling_folds(timeline, lead, training, options)
        selected = select_model(folds)
        fit = training.fit_through(origin, options.ridge_alpha)
        forecast, terms = forecast_for(timeline, lead, selected, fit, options)
        forecasts.append(forecast)
        explanations.extend(terms)
        backtest.extend(fold_record(timeline, lead, fold) for fold in folds)

        training_versions = training_provenance(timeline, lead, training, fit)
        backtest_versions = backtest_provenance(timeline, lead, folds)
        version_ids = [item.get('schedule_version') for item in training_versions]
        version_identities = [item.get('version_identity') for item in training_versions]
        versions_by_date = {
            item['training_date']: item.get('version_identity')
            for item in training_versions
        }
        selections.append({
            'lead': lead, 'model': selected.model, 'reason': selected.reason,
            'candidate_origins': candidates, 'paired_origins': len(folds),
            'weekly_mae': selected.weekly_mae, 'ridge_mae': selected.ridge_mae,
            'final_ridge_training_rows': len(training.target_indices),
            'training_schedule_versions': training_versions,
            'training_calendar_versions': training_versions,
            'schedule_versions': training_versions,
            'calendar_versions': training_versions,
            'backtest_schedule_versions': backtest_versions,
            'backtest_calendar_versions': backtest_versions,
            'forecast_schedule_version': forecast.get('schedule_version'),
            'forecast_calendar_version': forecast.get('calendar_version'),
            'forecast_schedule_version_identity': forecast.get('version_identity'),
            'forecast_schedule_selection_reason': forecast.get('selection_reason'),
            'training_schedule_version_ids': version_ids,
            'training_schedule_version_identities': version_identities,
            'training_schedule_versions_by_date': versions_by_date,
        })
        if len(folds) < MIN_PAIRED_ORIGINS:
            warnings.append(f'第 {lead} 步只有 {len(folds)} 个成对回测起点，至少需要 {MIN_PAIRED_ORIGINS} 个。')
        if forecast['status'] != 'forecasted':
            status = str(forecast['status'])
            reason = forecast.get('selection_reason')
            if reason == REVOKED_SELECTED:
                issue = '目标日排期的最新可见版本为撤销，按缺失处理，且不回退旧版'
            elif reason == NOT_RECEIVED:
                issue = '目标日排期虽已发布，但在签发时点尚未收到'
            else:
                issue = {
                    'calendar_missing': '缺少目标日日历',
                    'calendar_not_yet_known': '目标日日历晚于起点闭室时刻发布或尚未收到',
                    'insufficient_history': '上一周同星期的闭室占用尚不可用',
                }.get(status, '目标日排期不可用')
            warnings.append(f'{forecast["target_date"]} {issue}。')

    return {
        'summary': {
            'method_id': method_id,
            'origin_date': origin_day.isoformat(),
            'origin_cutoff': closing_cutoff(origin_day).isoformat(),
            'observed_days': len(timeline.actuals),
            'requested_horizon': options.horizon,
            'forecasted_days': sum(row['status'] == 'forecasted' for row in forecasts),
            'tight_days': sum(row.get('risk') == 'tight' for row in forecasts),
            'critical_days': sum(row.get('risk') == 'critical' for row in forecasts),
            'ridge_selected_steps': sum(row['model'] == RIDGE_MODEL for row in selections),
        },
        'tables': {
            'forecasts': forecasts, 'model_selection': selections,
            'rolling_backtest': backtest, 'feature_contributions': explanations,
        },
        'warnings': warnings,
        'publishable': not warnings,
    }
