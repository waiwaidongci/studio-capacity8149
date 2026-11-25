from datetime import timedelta

from ..evaluation.backtest import Fold, rolling_folds
from ..evaluation.selection import MIN_PAIRED_ORIGINS, Selection, select_model
from ..forecasting.features import FEATURE_NAMES, features_for
from ..forecasting.ridge import RIDGE_MODEL, RidgeFit, RidgeTraining
from ..forecasting.seasonal import predict_weekly, weekly_source_index
from ..intake.schemas import ForecastOptions
from ..intake.timeline import Timeline, closing_cutoff
from .risk import seat_risk


def fold_record(timeline: Timeline, lead: int, fold: Fold) -> dict:
    target = timeline.days[fold.target]
    return {
        'lead': lead,
        'origin_date': timeline.days[fold.origin].service_date.isoformat(),
        'target_date': target.service_date.isoformat(),
        'calendar_known_at': target.calendar_known_at.isoformat(),
        'weekly_source_date': timeline.days[weekly_source_index(fold.target)].service_date.isoformat(),
        'ridge_latest_training_date': timeline.days[fold.ridge_last_target].service_date.isoformat(),
        'ridge_training_rows': fold.ridge_training_rows,
        'actual_seats': fold.actual,
        'weekly_prediction': fold.weekly_prediction,
        'ridge_prediction': fold.ridge_prediction,
        'weekly_absolute_error': fold.weekly_error,
        'ridge_absolute_error': fold.ridge_error,
    }


def forecast_for(timeline: Timeline, lead: int, selection: Selection, fit: RidgeFit | None,
                 options: ForecastOptions) -> tuple[dict, list[dict]]:
    origin = timeline.origin_index
    target = origin + lead
    day = timeline.days[origin].service_date + timedelta(days=lead)
    row = {'lead': lead, 'target_date': day.isoformat(), 'model': selection.model}
    if target >= len(timeline.days):
        return {**row, 'status': 'calendar_missing'}, []
    if not timeline.knows_calendar(origin, target):
        return {**row, 'status': 'calendar_not_yet_known'}, []
    target_day = timeline.days[target]
    row.update({
        'calendar_known_at': target_day.calendar_known_at.isoformat(),
        'public_playback': target_day.public_playback,
        'review_batches': target_day.review_batches,
        'capacity_seats': target_day.capacity_seats,
    })
    weekly = predict_weekly(timeline, origin, target)
    if weekly is None:
        return {**row, 'status': 'insufficient_history'}, []

    seasonal_date = timeline.days[weekly_source_index(target)].service_date.isoformat()
    terms = []
    if selection.model == RIDGE_MODEL:
        features = features_for(timeline, origin, target)
        assert fit is not None and features is not None  # A selected ridge has a fitted paired history.
        terms.append(('intercept', 1, fit.intercept, None))
        source_dates = {
            'origin_occupied_seats': timeline.days[origin].service_date.isoformat(),
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
         'feature': name, 'value': value, 'coefficient': weight,
         'contribution_seats': value * weight, 'source_date': source}
        for name, value, weight, source in terms
    ]
    predicted = max(0.0, raw)
    return {
        **row, 'status': 'forecasted', 'raw_prediction_seats': raw, 'forecast_seats': predicted,
        **seat_risk(predicted, target_day.capacity_seats, options.tight_utilization),
    }, explanation


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
        selections.append({
            'lead': lead, 'model': selected.model, 'reason': selected.reason,
            'candidate_origins': candidates, 'paired_origins': len(folds),
            'weekly_mae': selected.weekly_mae, 'ridge_mae': selected.ridge_mae,
            'final_ridge_training_rows': len(training.target_indices),
        })
        if len(folds) < MIN_PAIRED_ORIGINS:
            warnings.append(f'第 {lead} 步只有 {len(folds)} 个成对回测起点，至少需要 {MIN_PAIRED_ORIGINS} 个。')
        if forecast['status'] != 'forecasted':
            issue = {
                'calendar_missing': '缺少目标日日历',
                'calendar_not_yet_known': '目标日日历晚于起点闭室时刻发布',
                'insufficient_history': '上一周同星期的闭室占用尚不可用',
            }[forecast['status']]
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
