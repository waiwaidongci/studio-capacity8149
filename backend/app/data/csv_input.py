import csv
from io import StringIO

from app.errors import Problem
from app.settings import settings
from domain.intake.schemas import (
    CSV_COLUMN_ALIASES,
    DATE_CSV_ALIASES,
    PUBLICATION_CSV_ALIASES,
    SCHEDULE_VALUE_COLUMNS,
    STATUS_CSV_COLUMNS,
)


def parse_csv(body: bytes) -> list[dict]:
    try:
        reader = csv.DictReader(StringIO(body.decode('utf-8-sig'), newline=''), strict=True)
        fields = reader.fieldnames or []
        known = CSV_COLUMN_ALIASES
        if not any(name in DATE_CSV_ALIASES for name in fields):
            raise ValueError('CSV columns must use schema names. Required: service_date')
        if not any(name in PUBLICATION_CSV_ALIASES for name in fields):
            raise ValueError(
                'CSV must contain calendar_known_at or another recognized publication timestamp'
            )
        versioned = any(name in STATUS_CSV_COLUMNS for name in fields)
        if not versioned and not SCHEDULE_VALUE_COLUMNS.issubset(fields):
            missing = sorted(SCHEDULE_VALUE_COLUMNS.difference(fields))
            raise ValueError(
                f'CSV columns must use schema names. Missing schedule columns: {missing}'
            )
        def recognized(column: str) -> bool:
            lowered = column.lower()
            if column in known:
                return True
            return any(token in lowered for token in (
                'version', 'revision', 'snapshot', 'publish', 'known_at',
                'received', 'receipt', 'arriv', 'status', 'state',
                'revok', 'cancel', 'withdraw', 'delete', 'remove',
                'restor', 'recover', 'reinstat', 'reactiv',
            ))

        if not all(recognized(name) for name in fields) or len(fields) != len(set(fields)):
            raise ValueError(f'CSV columns must use schema names. Known columns: {sorted(known)}')
        result = []
        for index, row in enumerate(reader, 1):
            if None in row or None in row.values():
                raise ValueError(f'CSV row {index} has a different number of columns')
            if index > settings.max_rows:
                raise ValueError(f'At most {settings.max_rows} rows are accepted')
            result.append({key: value for key, value in row.items() if value != ''})
        return result
    except (UnicodeError, csv.Error, ValueError) as exc:
        raise Problem(422, 'invalid_csv', str(exc)) from None
