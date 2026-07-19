from __future__ import annotations

from app.sql_agent.agent import SQLAgent


def test_count_query(db_engine):
    agent = SQLAgent(db_engine)
    result = agent.answer("How many emergency admissions are there?")
    assert result.success
    assert result.execution.row_count == 1
    assert result.execution.rows[0]["patient_count"] > 0


def test_average_length_of_stay(db_engine):
    agent = SQLAgent(db_engine)
    result = agent.answer("What is the average hospital stay for cancer patients?")
    assert result.success
    value = result.execution.rows[0]["avg_length_of_stay_days"]
    assert value is not None and value > 0


def test_condition_and_age_filter(db_engine):
    agent = SQLAgent(db_engine)
    result = agent.answer("Show diabetic patients older than 60")
    assert result.success
    assert result.execution.row_count > 0
    for row in result.execution.rows[:5]:
        assert row["age"] > 60


def test_generated_sql_is_read_only(db_engine):
    agent = SQLAgent(db_engine)
    result = agent.answer("List all patients assigned to Dr Smith")
    assert result.success
    assert result.sql.strip().upper().startswith("SELECT")
    assert "LIMIT" in result.sql.upper()
