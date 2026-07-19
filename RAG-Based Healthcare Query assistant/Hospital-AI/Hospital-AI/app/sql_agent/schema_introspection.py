"""
Schema introspection for prompt grounding.

Rather than hand-maintaining a text description of the schema (which
silently drifts out of sync with the actual tables), we introspect the
live database with SQLAlchemy's Inspector and a few cheap DISTINCT
queries. This is what "analyze the database schema automatically" means
in practice: the NL2SQL prompt is always describing the database that is
actually running, not a stale doc.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine


@dataclass
class ColumnInfo:
    name: str
    type: str
    sample_values: list[str] = field(default_factory=list)


@dataclass
class TableInfo:
    name: str
    columns: list[ColumnInfo]
    foreign_keys: list[tuple[str, str, str]]  # (column, ref_table, ref_column)


# Columns cheap and useful enough to sample distinct values for, so the LLM
# knows the exact literal spellings ("Elective" vs "elective") to use in
# WHERE clauses instead of guessing.
_ENUM_LIKE_COLUMNS = {
    "medical_condition",
    "admission_type",
    "test_results",
    "gender",
    "blood_type",
    "medication",
}


def introspect_schema(engine: Engine) -> list[TableInfo]:
    inspector = inspect(engine)
    tables: list[TableInfo] = []

    with engine.connect() as conn:
        for table_name in inspector.get_table_names():
            columns = []
            for col in inspector.get_columns(table_name):
                sample_values: list[str] = []
                if col["name"] in _ENUM_LIKE_COLUMNS:
                    rows = conn.execute(
                        text(f'SELECT DISTINCT "{col["name"]}" FROM "{table_name}" LIMIT 10')
                    ).all()
                    sample_values = [str(r[0]) for r in rows]
                columns.append(ColumnInfo(name=col["name"], type=str(col["type"]), sample_values=sample_values))

            fks = [
                (fk["constrained_columns"][0], fk["referred_table"], fk["referred_columns"][0])
                for fk in inspector.get_foreign_keys(table_name)
                if fk["constrained_columns"]
            ]
            tables.append(TableInfo(name=table_name, columns=columns, foreign_keys=fks))

    return tables


def schema_to_prompt_text(tables: list[TableInfo]) -> str:
    """Render the introspected schema as compact DDL-ish text for the LLM prompt."""
    lines = []
    for t in tables:
        lines.append(f"TABLE {t.name} (")
        for c in t.columns:
            enum_hint = f"  -- example values: {c.sample_values}" if c.sample_values else ""
            lines.append(f"    {c.name} {c.type}{enum_hint}")
        for col, ref_table, ref_col in t.foreign_keys:
            lines.append(f"    FOREIGN KEY ({col}) REFERENCES {ref_table}({ref_col})")
        lines.append(")")
    return "\n".join(lines)
