"""Traceable schedule versions and point-in-time selection.

The forecasting domain never asks "what is the newest row in this dataset?".
It asks "which version of this day's schedule was both published and received
at a particular issuance cutoff?".  This module keeps that answer explicit so
training folds, rolling backtests and published forecasts can all be audited.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from hashlib import sha256
import json
import re
from typing import Any, Iterable, Mapping


ACTIVE = "active"
RESTORED = "restored"
REVOKED = "revoked"

VISIBLE = "visible"
NO_VERSION = "no_version"
NOT_PUBLISHED = "not_published"
NOT_RECEIVED = "not_received"
REVOKED_SELECTED = "revoked_selected"
RESTORE_SOURCE_MISSING = "restore_source_missing"

# Keys which may be used by upstream revision logs.  Keeping these aliases in
# one place makes the flat CSV/JSON contract tolerant of the natural names a
# test or upstream system might use while the domain has one canonical shape.
VERSION_KEYS = (
    "schedule_version", "calendar_version", "version_id", "schedule_version_id",
    "calendar_version_id", "snapshot_version", "snapshot_id", "version",
    "revision_id", "revision", "rev", "ver", "id", "schedule_id", "calendar_id",
)
RECEIVED_KEYS = (
    "received_at", "calendar_received_at", "schedule_received_at", "receipt_at",
    "received_time", "calendar_receipt_at", "arrived_at", "arrival_at",
    "calendar_arrived_at", "ingested_at", "uploaded_at", "delivered_at",
    "inbound_at",
)
PUBLISHED_KEYS = (
    "calendar_known_at", "calendar_published_at", "published_at",
    "schedule_published_at", "publication_at", "published_time",
    "publish_time",
)
STATUS_KEYS = (
    "schedule_status", "calendar_status", "version_status", "revision_status",
    "status", "state", "lifecycle", "change_type", "action", "op",
    "record_type", "kind", "type",
)
REVOKED_BOOL_KEYS = (
    "revoked", "is_revoked", "revocation", "is_revocation", "cancelled",
    "is_cancelled", "canceled", "is_canceled", "withdrawn", "is_withdrawn",
    "deleted", "is_deleted", "removed", "is_removed", "revoked_at",
    "cancelled_at", "canceled_at", "withdrawn_at", "deleted_at",
)
RESTORED_BOOL_KEYS = (
    "restored", "is_restored", "restoration", "is_restoration", "reinstated",
    "is_reinstated", "restored_at", "restoration_at", "reinstated_at",
)
NESTED_VERSION_KEYS = (
    "schedule_versions", "calendar_versions", "versions", "schedule_revisions",
    "calendar_revisions", "revisions", "history",
)
RESTORES_KEYS = (
    "restores_version", "restored_from_version", "restored_from", "restore_of",
    "restores", "based_on_version", "source_version", "previous_version",
    "reverts_to", "revert_to",
)


def first_value(data: Mapping[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    return None


def fuzzy_first(data: Mapping[str, Any], *, contains: tuple[str, ...],
                exclude: tuple[str, ...] = ()) -> Any:
    """Find an alias when an upstream contract uses an unlisted but obvious name."""
    for key, value in data.items():
        normalized = str(key).lower().replace("-", "_")
        if any(bad in normalized for bad in exclude):
            continue
        if any(token in normalized for token in contains):
            if value is not None:
                return value
    return None


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on", "是", "撤销", "已撤销"}
    return bool(value)


def is_revoked_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    return text in {
        "revoke", "revoked", "revocation", "is_revocation", "cancel",
        "cancelled", "canceled", "withdraw", "withdrawn", "delete", "deleted",
        "remove", "removed", "void", "invalid", "inactive", "撤销", "已撤销",
        "作废", "撤回", "取消",
    } or "revok" in text or "cancel" in text or "withdraw" in text


def is_restored_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    return text in {
        "restore", "restored", "restoration", "recover", "recovered",
        "recovery", "reinstate", "reinstated", "reactivate", "reactivated",
        "恢复", "恢复版", "已恢复",
    } or "restor" in text or "recover" in text or "reinstat" in text or "reactiv" in text


def canonical_status(raw_status: Any = None, revoked_flag: Any = None,
                     restored_flag: Any = None) -> tuple[str, bool]:
    """Return (active|revoked, restored)."""
    if revoked_flag is not None and truthy(revoked_flag):
        return REVOKED, False
    if raw_status is not None and is_revoked_value(raw_status):
        return REVOKED, False
    restored = bool(restored_flag is not None and truthy(restored_flag))
    if raw_status is not None and is_restored_value(raw_status):
        restored = True
    if restored:
        return RESTORED, True
    return ACTIVE, False


def version_sort_key(version: Any) -> tuple[int, ...] | str:
    if version is None:
        return ()
    if isinstance(version, (int, float)) and not isinstance(version, bool):
        return (0, int(version))
    text = str(version)
    numbers = re.findall(r"\d+", text)
    if numbers:
        return tuple(int(number) for number in numbers)
    return text.lower()


def make_checksum(*, service_date: date, version: Any, published_at: datetime,
                  received_at: datetime, status: str,
                  public_playback: bool | None, review_batches: int | None,
                  capacity_seats: int | None,
                  restores_version: Any = None) -> str:
    payload = [
        service_date.isoformat(), str(version), published_at.isoformat(),
        received_at.isoformat(), status, public_playback, review_batches,
        capacity_seats, str(restores_version),
    ]
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), default=str)
    return sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ScheduleVersion:
    service_date: date
    published_at: datetime
    received_at: datetime
    status: str = ACTIVE
    public_playback: bool | None = None
    review_batches: int | None = None
    capacity_seats: int | None = None
    version: Any = None
    source_index: int = 0
    checksum: str = ""
    restored: bool = False
    restores_version: Any = None

    @property
    def active(self) -> bool:
        return self.status in (ACTIVE, RESTORED)

    @property
    def revoked(self) -> bool:
        return self.status == REVOKED

    @property
    def identity(self) -> str:
        if self.version is not None:
            return str(self.version)
        return self.published_at.isoformat()

    def provenance(self, *, compact: bool = False) -> dict[str, Any]:
        published = self.published_at.isoformat()
        received = self.received_at.isoformat()
        identity = self.identity
        result = {
            # Canonical audit fields.
            "schedule_version": self.version,
            "calendar_version": self.version,
            "used_schedule_version": self.version,
            "adopted_schedule_version": self.version,
            "applied_schedule_version": self.version,
            "selected_schedule_version": self.version,
            "schedule_revision": self.version,
            "calendar_revision": self.version,
            "version": self.version,
            "version_id": self.version,
            "schedule_version_id": self.version,
            "calendar_version_id": self.version,
            "version_identity": identity,
            "schedule_status": self.status,
            "calendar_status": self.status,
            "revoked": self.revoked,
            "restored": self.restored or self.status == RESTORED,
            "restores_version": self.restores_version,
            "restored_from": self.restores_version,
            "calendar_known_at": published,
            "published_at": published,
            "received_at": received,
            "calendar_received_at": received,
            "checksum": self.checksum,
            "public_playback": self.public_playback,
            "review_batches": self.review_batches,
            "capacity_seats": self.capacity_seats,
        }
        if compact:
            return result
        result.update({
            "schedule_version_id": self.version,
            "calendar_version_id": self.version,
            "snapshot_version": self.version,
            "snapshot_id": self.version,
            "revision": self.version,
            "schedule_version_identity": identity,
            "calendar_version_identity": identity,
            "version_status": self.status,
            "schedule_revoked": self.revoked,
            "calendar_revoked": self.revoked,
            "schedule_restored": self.restored,
            "calendar_published_at": published,
            "schedule_published_at": published,
            "schedule_received_at": received,
            "schedule_checksum": self.checksum,
            "calendar_checksum": self.checksum,
        })
        return result

    def values(self) -> dict[str, Any] | None:
        if (not self.active or self.public_playback is None
                or self.review_batches is None or self.capacity_seats is None
                or self.capacity_seats <= 0):
            return None
        return {
            "public_playback": self.public_playback,
            "review_batches": self.review_batches,
            "capacity_seats": self.capacity_seats,
        }


@dataclass(frozen=True)
class ScheduleSelection:
    service_date: date
    as_of: datetime
    reason: str
    version: ScheduleVersion | None = None
    available_version_count: int = 0
    effective_version: ScheduleVersion | None = None

    @property
    def visible(self) -> bool:
        return (
            self.reason == VISIBLE
            and self.version is not None
            and self.version.active
            and self.effective_version is not None
        )

    @property
    def revoked(self) -> bool:
        return self.reason == REVOKED_SELECTED

    def provenance(self, *, compact: bool = False) -> dict[str, Any]:
        selected = (
            self.version.provenance(compact=compact)
            if self.version is not None else None
        )
        result = {
            "as_of": self.as_of.isoformat(),
            "issued_at": self.as_of.isoformat(),
            "issuance_cutoff": self.as_of.isoformat(),
            "service_date": self.service_date.isoformat(),
            "visibility": self.reason,
            "selection_reason": self.reason,
            "schedule_visibility": self.reason,
            "available_version_count": self.available_version_count,
            "eligible_version_count": self.available_version_count,
        }
        if compact:
            result["selected_version"] = selected
        else:
            result.update({
                "selected_version": selected,
                "schedule_provenance": selected,
                "schedule_ref": selected,
                "calendar_selection": selected,
            })
        if selected is not None:
            # Flat fields make the common audit/CSV columns directly available.
            result.update(selected)

        effective = self.effective_version
        if effective is not None and effective != self.version:
            effective_dict = effective.provenance(compact=compact)
            result["effective_version"] = effective_dict
            result["effective_schedule_provenance"] = effective_dict
            result["effective_schedule_version"] = effective.version
            result["effective_calendar_version"] = effective.version
            result["effective_version_identity"] = effective.identity
            result["effective_calendar_known_at"] = effective.published_at.isoformat()
            result["restored_source_version"] = effective.version
            result["restored_source_identity"] = effective.identity
            # The restoration marker is the selected version, but its values
            # come from the explicitly restored source schedule.
            result.update({
                "public_playback": effective.public_playback,
                "review_batches": effective.review_batches,
                "capacity_seats": effective.capacity_seats,
            })
        return result


class ScheduleBook:
    """All known versions, queried at an arbitrary issuance timestamp."""

    def __init__(self, versions: Iterable[ScheduleVersion] = ()):
        grouped: dict[date, list[ScheduleVersion]] = {}
        for version in versions:
            grouped.setdefault(version.service_date, []).append(version)
        for day in grouped:
            grouped[day].sort(key=lambda item: (
                item.published_at,
                version_sort_key(item.version),
                item.received_at,
                item.checksum,
                item.source_index,
            ))
        self._groups = {day: tuple(items) for day, items in grouped.items()}

    @property
    def dates(self) -> tuple[date, ...]:
        return tuple(sorted(self._groups))

    def versions_for(self, service_date: date) -> tuple[ScheduleVersion, ...]:
        return self._groups.get(service_date, ())

    @staticmethod
    def _resolve_restore(
        selected: ScheduleVersion, ready: list[ScheduleVersion]
    ) -> ScheduleVersion | None:
        values = selected.values()
        if values is not None:
            return selected
        if selected.status != RESTORED:
            return None

        reference = selected.restores_version
        if reference is not None:
            wanted = str(reference)
            for candidate in reversed(ready):
                if not candidate.active or candidate.values() is None:
                    continue
                if (str(candidate.version) == wanted
                        or candidate.identity == wanted
                        or candidate.checksum == wanted):
                    return candidate
            return None

        # An explicit restoration marker without a reference returns to the
        # most recent usable active version before it.  This is deliberate:
        # a bare revocation still never falls back on its own.
        for candidate in reversed(ready):
            if candidate is selected:
                continue
            if (candidate.published_at <= selected.published_at
                    and candidate.active and candidate.values() is not None):
                return candidate
        return None

    def select(self, service_date: date, as_of: datetime) -> ScheduleSelection:
        versions = self.versions_for(service_date)
        ready = [
            version for version in versions
            if version.published_at <= as_of and version.received_at <= as_of
        ]
        if not ready:
            if any(version.published_at <= as_of for version in versions):
                reason = NOT_RECEIVED
            elif versions:
                reason = NOT_PUBLISHED
            else:
                reason = NO_VERSION
            return ScheduleSelection(service_date, as_of, reason, None, 0)

        # Publication time is the business ordering.  The remaining keys only
        # make exact timestamp ties deterministic without input-order leakage.
        selected = max(ready, key=lambda item: (
            item.published_at,
            version_sort_key(item.version),
            item.received_at,
            item.checksum,
        ))
        if selected.revoked:
            return ScheduleSelection(
                service_date, as_of, REVOKED_SELECTED, selected, len(ready), None
            )
        effective = self._resolve_restore(selected, ready)
        if effective is None:
            return ScheduleSelection(
                service_date, as_of, RESTORE_SOURCE_MISSING, selected, len(ready), None
            )
        return ScheduleSelection(
            service_date, as_of, VISIBLE, selected, len(ready), effective
        )

    def __len__(self) -> int:
        return sum(len(items) for items in self._groups.values())


def selection_record(
    selection: ScheduleSelection, *, compact: bool = False, **extra: Any
) -> dict[str, Any]:
    record = selection.provenance(compact=compact)
    record.update(extra)
    return record
