"""
SQL safety validation.

An LLM-generated query must never be executed blind. This module enforces,
independent of what the LLM was asked to do:

1. Exactly one statement (no `; DROP TABLE ...` stacked after a semicolon).
2. The statement is a SELECT (or CTE feeding a SELECT) -- no INSERT, UPDATE,
   DELETE, DROP, ALTER, ATTACH, PRAGMA, etc.
3. Every referenced table is in an explicit whitelist (the four tables this
   app actually has).
4. A row limit is enforced by rewriting the query, not by trusting the LLM
   to have included one.

We use `sqlglot` to parse into an AST rather than regex/string matching,
because regex is trivially defeated by comments, string literals containing
keywords, or alternate casing.
"""
from __future__ import annotations

from dataclasses import dataclass

import sqlglot
from sqlglot import exp

ALLOWED_TABLES = {"admissions", "doctors", "hospitals", "insurance_providers"}

# Statement types that are never acceptable from a natural-language query
# assistant, regardless of what the user asked for.
FORBIDDEN_EXPRESSIONS = tuple(
    getattr(exp, name)
    for name in ("Insert", "Update", "Delete", "Drop", "Alter", "Create", "Attach", "Command", "TruncateTable")
    if hasattr(exp, name)
)


@dataclass
class ValidationResult:
    is_valid: bool
    safe_sql: str | None
    error: str | None = None


def validate_and_sanitize(raw_sql: str, row_limit: int) -> ValidationResult:
    sql = raw_sql.strip().rstrip(";")

    if not sql:
        return ValidationResult(False, None, "Empty SQL.")

    # Reject stacked statements outright (defense in depth even though
    # sqlglot.parse_one would only look at the first statement anyway).
    try:
        statements = sqlglot.parse(sql, read="sqlite")
    except Exception as exc:  # sqlglot raises its own ParseError subtypes
        return ValidationResult(False, None, f"SQL failed to parse: {exc}")

    statements = [s for s in statements if s is not None]
    if len(statements) != 1:
        return ValidationResult(False, None, "Only a single SQL statement is allowed.")

    stmt = statements[0]

    if not isinstance(stmt, (exp.Select, exp.Union)):
        return ValidationResult(False, None, "Only SELECT queries are permitted.")

    # Walk the full AST for any forbidden node types (covers CTEs/subqueries).
    for node in stmt.walk():
        if isinstance(node, tuple):  # older sqlglot yields (node, parent, key)
            node = node[0]
        if isinstance(node, FORBIDDEN_EXPRESSIONS):
            return ValidationResult(False, None, f"Statement type '{type(node).__name__}' is not permitted.")

    tables_used = {t.name.lower() for t in stmt.find_all(exp.Table)}
    disallowed = tables_used - ALLOWED_TABLES
    if disallowed:
        return ValidationResult(False, None, f"Query references unknown table(s): {sorted(disallowed)}")

    # Enforce a row cap by rewriting/inserting LIMIT, rather than trusting
    # the model to have added one -- an unbounded scan over the fact table
    # returned to a chat UI is both a UX and a resource-usage problem.
    existing_limit = stmt.args.get("limit")
    if existing_limit is None:
        stmt = stmt.limit(row_limit)
    else:
        try:
            current = int(existing_limit.expression.this)
            if current > row_limit:
                stmt.set("limit", exp.Limit(expression=exp.Literal.number(row_limit)))
        except Exception:
            stmt.set("limit", exp.Limit(expression=exp.Literal.number(row_limit)))

    safe_sql = stmt.sql(dialect="sqlite")
    return ValidationResult(True, safe_sql, None)
