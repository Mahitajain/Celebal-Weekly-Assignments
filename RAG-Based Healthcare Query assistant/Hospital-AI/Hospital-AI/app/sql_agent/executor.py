from __future__ import annotations

import time
from dataclasses import dataclass, field

from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from app.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class ExecutionResult:
    success: bool
    columns: list[str] = field(default_factory=list)
    rows: list[dict] = field(default_factory=list)
    row_count: int = 0
    elapsed_ms: float = 0.0
    error: str | None = None


def execute_sql(engine: Engine, sql: str) -> ExecutionResult:
    """Execute an already-validated, read-only SQL statement.

    This function trusts that `sql` has already passed
    `app.sql_agent.validator.validate_and_sanitize` -- it does not
    re-validate, it only defends against runtime failures (bad column
    combinations the parser couldn't catch, SQLite errors, etc.).
    """
    t0 = time.time()
    try:
        with engine.connect() as conn:
            # SQLite has no native statement timeout; a `LIMIT` was already
            # enforced by the validator which bounds worst-case scan cost
            # for this dataset size. For a Postgres deployment this is
            # where `SET LOCAL statement_timeout` would go.
            result = conn.execute(text(sql))
            columns = list(result.keys())
            rows = [dict(zip(columns, row)) for row in result.fetchall()]

        elapsed_ms = (time.time() - t0) * 1000
        logger.info("Executed SQL in %.1fms, %d rows returned", elapsed_ms, len(rows))
        return ExecutionResult(
            success=True,
            columns=columns,
            rows=rows,
            row_count=len(rows),
            elapsed_ms=round(elapsed_ms, 1),
        )
    except SQLAlchemyError as exc:
        logger.warning("SQL execution failed: %s", exc)
        return ExecutionResult(success=False, error=str(exc.__cause__ or exc), elapsed_ms=(time.time() - t0) * 1000)
