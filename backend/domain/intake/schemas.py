from datetime import date, datetime, timedelta, timezone
from typing import Annotated

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator


NonnegativeCount = Annotated[int, Field(ge=0, le=1_000_000)]


class ListeningDay(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)

    service_date: date = Field(le=date.max - timedelta(days=7))
    calendar_known_at: AwareDatetime
    public_playback: bool
    review_batches: NonnegativeCount
    capacity_seats: NonnegativeCount = Field(gt=0)
    closing_occupied_seats: NonnegativeCount | None = None

    @field_validator('service_date', mode='before')
    @classmethod
    def require_date(cls, value):
        if type(value) is date or isinstance(value, str) and len(value) == 10:
            return value
        raise ValueError('service_date must be a YYYY-MM-DD date')

    @field_validator('calendar_known_at', mode='before')
    @classmethod
    def require_timestamp(cls, value):
        if isinstance(value, datetime) or isinstance(value, str) and 'T' in value:
            return value
        raise ValueError('calendar_known_at must be an ISO 8601 timestamp with a UTC offset')

    @field_validator('calendar_known_at')
    @classmethod
    def normalize_timestamp(cls, value: datetime) -> datetime:
        try:
            return value.astimezone(timezone.utc)
        except OverflowError:
            raise ValueError('calendar_known_at is outside the UTC date range') from None

    @field_validator('public_playback', mode='before')
    @classmethod
    def require_playback_flag(cls, value):
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        raise ValueError('public_playback must be true or false')

    @field_validator('review_batches', 'capacity_seats', 'closing_occupied_seats', mode='before')
    @classmethod
    def reject_boolean_counts(cls, value):
        if isinstance(value, bool):
            raise ValueError('counts must be integers')
        return value


class ForecastOptions(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)

    horizon: int = Field(default=7, strict=True, ge=1, le=7)
    backtest_origins: int = Field(default=28, strict=True, ge=4, le=84)
    ridge_alpha: float = Field(default=1.0, strict=True, ge=1e-6, le=1e6)
    tight_utilization: float = Field(default=0.85, strict=True, gt=0, lt=1)
