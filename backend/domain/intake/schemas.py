from datetime import date, datetime, timedelta, timezone
from typing import Annotated, Any

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator, model_validator

from .versions import (
    ACTIVE,
    NESTED_VERSION_KEYS,
    PUBLISHED_KEYS,
    RECEIVED_KEYS,
    RESTORED,
    RESTORES_KEYS,
    RESTORED_BOOL_KEYS,
    REVOKED,
    REVOKED_BOOL_KEYS,
    STATUS_KEYS,
    VERSION_KEYS,
    canonical_status,
    first_value,
    fuzzy_first,
    truthy,
)


NonnegativeCount = Annotated[int, Field(ge=0, le=1_000_000)]
PositiveCount = Annotated[int, Field(gt=0, le=1_000_000)]


def _pop_known(data: dict[str, Any], keys: tuple[str, ...], canonical: str) -> None:
    for key in keys:
        if key != canonical:
            data.pop(key, None)


def normalize_revision_aliases(value: Any, *, allow_nested: bool = False) -> Any:
    """Map upstream version/timestamp naming variants to canonical fields."""
    if not isinstance(value, dict):
        return value
    data = dict(value)

    published = first_value(data, PUBLISHED_KEYS)
    if published is None:
        published = fuzzy_first(
            data, contains=("published", "publication"), exclude=("received",)
        )
    if published is not None:
        data["calendar_known_at"] = published
    _pop_known(data, PUBLISHED_KEYS, "calendar_known_at")
    for key in list(data):
        lowered = key.lower()
        if "publish" in lowered and key != "calendar_known_at":
            data.pop(key, None)

    received = first_value(data, RECEIVED_KEYS)
    if received is None:
        received = fuzzy_first(data, contains=("received", "receipt", "arriv"))
    if received is not None:
        data["received_at"] = received
    _pop_known(data, RECEIVED_KEYS, "received_at")
    for key in list(data):
        lowered = key.lower()
        if ("received" in lowered or "receipt" in lowered or "arriv" in lowered) and key != "received_at":
            data.pop(key, None)

    version = first_value(data, VERSION_KEYS)
    if version is None:
        version = fuzzy_first(data, contains=("version", "revision", "snapshot"),
                              exclude=("status", "count", "restore", "source",
                                       "based", "previous", "note", "comment",
                                       "description", "author", "message"))
        if isinstance(version, (list, tuple)):
            version = None
    if version is not None:
        data["schedule_version"] = version
    _pop_known(data, VERSION_KEYS, "schedule_version")
    restores = first_value(data, RESTORES_KEYS)
    if restores is not None:
        data["restores_version"] = restores
    _pop_known(data, RESTORES_KEYS, "restores_version")
    for key in list(data):
        lowered = key.lower()
        if (("version" in lowered or "revision" in lowered or "snapshot" in lowered)
                and not isinstance(data.get(key), (list, tuple))
                and key not in {"schedule_version", "restores_version"}):
            data.pop(key, None)

    raw_status = first_value(data, STATUS_KEYS)
    revoked_flag = first_value(data, REVOKED_BOOL_KEYS)
    restored_flag = first_value(data, RESTORED_BOOL_KEYS)
    if revoked_flag is None:
        for key, value in data.items():
            lowered = str(key).lower()
            if (any(token in lowered for token in ("revok", "cancel", "withdraw", "delet", "remov"))
                    and "restore" not in lowered and truthy(value)):
                revoked_flag = True
                break
    if restored_flag is None:
        for key, value in data.items():
            lowered = str(key).lower()
            if (any(token in lowered for token in ("restor", "recover", "reinstat", "reactiv"))
                    and truthy(value)):
                restored_flag = True
                break
    # Timestamp fields such as revoked_at carry meaning just by being present.
    if isinstance(revoked_flag, str) and revoked_flag.strip().lower() not in {
        "false", "0", "no", "off", "",
    } and not truthy(revoked_flag):
        revoked_flag = True
    if isinstance(restored_flag, str) and restored_flag.strip().lower() not in {
        "false", "0", "no", "off", "",
    } and not truthy(restored_flag):
        restored_flag = True
    status, restored = canonical_status(raw_status, revoked_flag, restored_flag)
    data["schedule_status"] = status
    data["revoked"] = status == REVOKED
    data["restored"] = restored
    _pop_known(data, STATUS_KEYS, "schedule_status")
    _pop_known(data, REVOKED_BOOL_KEYS, "revoked")
    _pop_known(data, RESTORED_BOOL_KEYS, "restored")
    for key in list(data):
        lowered = key.lower()
        if key in {"schedule_status", "revoked", "restored", "restores_version"}:
            continue
        if ("status" in lowered or lowered in {"state", "op"}
                or "change_type" in lowered or "record_type" in lowered):
            data.pop(key, None)
        elif "revok" in lowered or "cancel" in lowered or "withdraw" in lowered:
            data.pop(key, None)
        elif any(token in lowered for token in ("restor", "recover", "reinstat", "reactiv")):
            data.pop(key, None)

    if allow_nested:
        nested = first_value(data, NESTED_VERSION_KEYS)
        if nested is None:
            for key, item in list(data.items()):
                if isinstance(item, (list, tuple)) and any(
                    token in key.lower() for token in ("version", "revision")
                ):
                    nested = item
                    data.pop(key, None)
                    break
        if nested is not None:
            data["schedule_versions"] = list(nested)
        for key in NESTED_VERSION_KEYS:
            if key != "schedule_versions":
                data.pop(key, None)

    return data


class ScheduleRevision(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    # Nested revision entries may repeat the target date/observation fields.
    service_date: date | None = None
    closing_occupied_seats: NonnegativeCount | None = None
    calendar_known_at: AwareDatetime | None = None
    received_at: AwareDatetime | None = None
    schedule_version: str | int | None = None
    restores_version: str | int | None = None
    schedule_status: str | None = None
    revoked: bool | None = None
    restored: bool | None = None
    public_playback: bool | None = None
    review_batches: NonnegativeCount | None = None
    capacity_seats: NonnegativeCount | None = None

    @model_validator(mode="before")
    @classmethod
    def normalize_aliases(cls, data: Any) -> Any:
        return normalize_revision_aliases(data)

    @field_validator("calendar_known_at", "received_at", mode="before")
    @classmethod
    def require_timestamp(cls, value):
        if value is None:
            return None
        if isinstance(value, datetime) or isinstance(value, str) and "T" in value:
            return value
        raise ValueError("timestamp must be an ISO 8601 timestamp with a UTC offset")

    @field_validator("calendar_known_at", "received_at")
    @classmethod
    def normalize_timestamp(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        try:
            return value.astimezone(timezone.utc)
        except OverflowError:
            raise ValueError("timestamp is outside the UTC date range") from None

    @field_validator("public_playback", mode="before")
    @classmethod
    def require_playback_flag(cls, value):
        if value is None:
            return None
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.lower() in ("true", "false"):
            return value.lower() == "true"
        raise ValueError("public_playback must be true or false")

    @field_validator("review_batches", "capacity_seats", "closing_occupied_seats", mode="before")
    @classmethod
    def reject_boolean_counts(cls, value):
        if isinstance(value, bool):
            raise ValueError("counts must be integers")
        return value

    @model_validator(mode="after")
    def validate_revision(self):
        nested_versions = getattr(self, "schedule_versions", None)
        if self.calendar_known_at is None:
            if nested_versions:
                # The day row carries only the observation/date while its
                # traceable schedule payloads live in nested version records.
                self.schedule_status = "observation_only"
                self.revoked = False
                self.restored = False
                return self
            raise ValueError(
                "calendar_known_at (or calendar_published_at/published_at) is required"
            )
        if self.received_at is None:
            # Legacy snapshots were effectively received when published.
            self.received_at = self.calendar_known_at
        elif self.received_at < self.calendar_known_at:
            raise ValueError("received_at must not be earlier than calendar_known_at")

        if self.schedule_status is None:
            self.schedule_status = REVOKED if self.revoked else ACTIVE
        if self.revoked is None:
            self.revoked = self.schedule_status == REVOKED
        if self.restored is None:
            self.restored = False

        if self.schedule_status == REVOKED or self.revoked:
            self.schedule_status = REVOKED
            self.revoked = True
            return self

        is_restored = (
            self.schedule_status == RESTORED
            or bool(self.restored)
        )
        if is_restored:
            self.schedule_status = RESTORED
            self.revoked = False
            self.restored = True
            # A restoration may either carry full values or explicitly point at
            # an earlier received version; ScheduleBook resolves the latter.
            return self

        missing = [
            name for name in ("public_playback", "review_batches", "capacity_seats")
            if getattr(self, name) is None
        ]
        if missing:
            raise ValueError(
                f"Active schedule version is missing: {', '.join(missing)}; "
                "send a revoked version if the schedule was withdrawn"
            )
        if self.capacity_seats <= 0:
            raise ValueError("capacity_seats must be positive for an active schedule")
        self.schedule_status = ACTIVE
        self.revoked = False
        self.restored = bool(self.restored)
        return self

    @property
    def version(self):
        return self.schedule_version

    @property
    def calendar_version(self):
        return self.schedule_version

    @property
    def published_at(self):
        return self.calendar_known_at

    @property
    def calendar_published_at(self):
        return self.calendar_known_at

    @property
    def calendar_received_at(self):
        return self.received_at


class ListeningDay(ScheduleRevision):
    service_date: date = Field(le=date.max - timedelta(days=7))
    closing_occupied_seats: NonnegativeCount | None = None
    schedule_versions: list[ScheduleRevision] | None = None

    @model_validator(mode="before")
    @classmethod
    def normalize_day_aliases(cls, data: Any) -> Any:
        if isinstance(data, dict) and data.get("service_date") is None:
            for key in ("date", "day", "target_date", "schedule_date"):
                if data.get(key) is not None:
                    data = dict(data)
                    data["service_date"] = data[key]
                    data.pop(key, None)
                    break
        normalized = normalize_revision_aliases(data, allow_nested=True)
        if isinstance(normalized, dict):
            # The parent before-validator also runs, but keeping this explicit
            # makes the nested alias handling independent of validator order.
            normalized = normalize_revision_aliases(normalized, allow_nested=True)
        return normalized

    @field_validator("service_date", mode="before")
    @classmethod
    def require_date(cls, value):
        if type(value) is date or isinstance(value, str) and len(value) == 10:
            return value
        raise ValueError("service_date must be a YYYY-MM-DD date")

    @field_validator("closing_occupied_seats", mode="before")
    @classmethod
    def reject_boolean_occupancy(cls, value):
        if isinstance(value, bool):
            raise ValueError("counts must be integers")
        return value


class ForecastOptions(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    horizon: int = Field(default=7, strict=True, ge=1, le=7)
    backtest_origins: int = Field(default=28, strict=True, ge=4, le=84)
    ridge_alpha: float = Field(default=1.0, strict=True, ge=1e-6, le=1e6)
    tight_utilization: float = Field(default=0.85, strict=True, gt=0, lt=1)


# Flat CSV accepts the canonical model fields plus every common upstream alias.
CSV_COLUMN_ALIASES = frozenset(
    {
        "service_date", "date", "day", "target_date", "schedule_date",
        "closing_occupied_seats",
        *PUBLISHED_KEYS, *RECEIVED_KEYS, *VERSION_KEYS, *STATUS_KEYS,
        *REVOKED_BOOL_KEYS, *RESTORED_BOOL_KEYS,
        "public_playback", "review_batches", "capacity_seats",
    }
)
LEGACY_REQUIRED_CSV_COLUMNS = frozenset({
    "service_date", "calendar_known_at", "public_playback",
    "review_batches", "capacity_seats",
})
DATE_CSV_ALIASES = frozenset({"service_date", "date", "day", "target_date", "schedule_date"})
DATE_REQUIRED_CSV_COLUMNS = frozenset({"service_date"})
PUBLICATION_CSV_ALIASES = frozenset(PUBLISHED_KEYS)
SCHEDULE_VALUE_COLUMNS = frozenset({"public_playback", "review_batches", "capacity_seats"})
STATUS_CSV_COLUMNS = frozenset(STATUS_KEYS) | frozenset(REVOKED_BOOL_KEYS)
