from __future__ import annotations

from app.sql_agent.validator import validate_and_sanitize


def test_valid_select_passes():
    result = validate_and_sanitize("SELECT * FROM admissions WHERE medical_condition = 'Diabetes'", row_limit=200)
    assert result.is_valid
    assert "LIMIT 200" in result.safe_sql


def test_rejects_stacked_statements():
    result = validate_and_sanitize("SELECT * FROM admissions; DROP TABLE admissions;", row_limit=200)
    assert not result.is_valid


def test_rejects_delete():
    result = validate_and_sanitize("DELETE FROM admissions WHERE age > 60", row_limit=200)
    assert not result.is_valid


def test_rejects_drop():
    result = validate_and_sanitize("DROP TABLE admissions", row_limit=200)
    assert not result.is_valid


def test_rejects_unknown_table():
    result = validate_and_sanitize("SELECT * FROM users", row_limit=200)
    assert not result.is_valid
    assert "unknown table" in result.error.lower()


def test_rejects_pragma():
    result = validate_and_sanitize("PRAGMA table_info(admissions)", row_limit=200)
    assert not result.is_valid


def test_caps_row_limit_even_if_llm_requested_more():
    result = validate_and_sanitize("SELECT * FROM admissions LIMIT 50000", row_limit=200)
    assert result.is_valid
    assert "LIMIT 200" in result.safe_sql


def test_allows_join_across_whitelisted_tables():
    sql = "SELECT a.patient_name, d.name FROM admissions a JOIN doctors d ON a.doctor_id = d.doctor_id"
    result = validate_and_sanitize(sql, row_limit=200)
    assert result.is_valid


def test_rejects_empty_sql():
    result = validate_and_sanitize("", row_limit=200)
    assert not result.is_valid


def test_rejects_garbage_sql():
    result = validate_and_sanitize("this is not sql at all !!!", row_limit=200)
    assert not result.is_valid
