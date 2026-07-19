"""
NL-to-SQL Agent.

Pipeline: introspect schema -> build prompt -> call LLM -> validate/sanitize
-> execute -> (retry once on failure with the error fed back) -> explain in
plain English.

If no LLM is configured, `_fallback_sql()` provides a small set of
deterministic, regex-based templates covering the most common question
shapes (condition filters, doctor filters, admission-type filters, average
length of stay, counts). This keeps the project demonstrable without an
API key, while the documented/expected path is LLM-driven NL2SQL.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from sqlalchemy.engine import Engine

from app.config import get_settings
from app.llm.client import LLMClient, get_llm_client
from app.sql_agent import prompts
from app.sql_agent.executor import ExecutionResult, execute_sql
from app.sql_agent.schema_introspection import introspect_schema, schema_to_prompt_text
from app.sql_agent.validator import validate_and_sanitize
from app.utils.logging_config import get_logger

logger = get_logger(__name__)

_CONDITIONS = ["Cancer", "Obesity", "Diabetes", "Asthma", "Hypertension", "Arthritis"]
_ADMISSION_TYPES = ["Urgent", "Emergency", "Elective"]
_TEST_RESULTS = ["Normal", "Inconclusive", "Abnormal"]


@dataclass
class SQLAgentResult:
    success: bool
    sql: str | None = None
    explanation: str | None = None
    execution: ExecutionResult | None = None
    answer: str = ""
    used_llm: bool = False
    retries: int = 0
    elapsed_ms: float = 0.0
    error: str | None = None


class SQLAgent:
    def __init__(self, engine: Engine, llm: LLMClient | None = None):
        self.engine = engine
        self.llm = llm or get_llm_client()
        self.settings = get_settings()
        self._schema_text: str | None = None

    def _schema(self) -> str:
        if self._schema_text is None:
            self._schema_text = schema_to_prompt_text(introspect_schema(self.engine))
        return self._schema_text

    # -- LLM-driven generation ------------------------------------------------
    def _generate_sql_llm(self, question: str, history: str, prior_error: str | None = None) -> str | None:
        system = prompts.NL2SQL_SYSTEM_TEMPLATE.format(schema=self._schema())
        user = prompts.NL2SQL_USER_TEMPLATE.format(history=history or "(none)", question=question)
        if prior_error:
            user += f"\n\nYour previous attempt failed with this database error:\n{prior_error}\nFix the query."

        response = self.llm.complete(system=system, user=user, temperature=0.0)
        if response is None:
            return None
        sql = response.text.strip()
        sql = re.sub(r"^```(sql)?|```$", "", sql, flags=re.MULTILINE).strip()
        return sql

    def _explain_llm(self, question: str, sql: str) -> str | None:
        response = self.llm.complete(
            system=prompts.EXPLAIN_SYSTEM_TEMPLATE,
            user=prompts.EXPLAIN_USER_TEMPLATE.format(question=question, sql=sql),
            temperature=0.2,
            max_tokens=200,
        )
        return response.text.strip() if response else None

    # -- Deterministic fallback (no LLM configured) ---------------------------
    def _fallback_sql(self, question: str) -> tuple[str, str] | None:
        """Very small pattern library covering common question shapes.

        Returns (sql, explanation) or None if nothing matched, in which
        case the caller reports that an LLM is required for this question.
        """
        q = question.lower()

        # A few common adjectival/plural forms that won't literal-match the
        # stored enum spelling ("diabetic" -> "Diabetes").
        _ALIASES = {
            "diabetic": "Diabetes", "diabetes": "Diabetes",
            "asthmatic": "Asthma",
            "obese": "Obesity",
            "hypertensive": "Hypertension",
            "arthritic": "Arthritis",
        }

        def _match_enum(options: list[str]) -> str | None:
            for alias, canonical in _ALIASES.items():
                if alias in q and canonical in options:
                    return canonical
            for opt in options:
                if opt.lower() in q:
                    return opt
            return None

        condition = _match_enum(_CONDITIONS)
        admission_type = _match_enum(_ADMISSION_TYPES)
        test_result = _match_enum(_TEST_RESULTS)

        age_gt = re.search(r"older than (\d+)|above age (\d+)|age(?:d)? (?:over|>) ?(\d+)", q)
        age_lt = re.search(r"younger than (\d+)|below age (\d+)|age(?:d)? (?:under|<) ?(\d+)", q)

        doctor_match = re.search(r"dr\.?\s+([a-z]+)", q)

        if "average" in q and ("stay" in q or "length of stay" in q):
            where = []
            if condition:
                where.append(f"medical_condition = '{condition}'")
            if admission_type:
                where.append(f"admission_type = '{admission_type}'")
            clause = f" WHERE {' AND '.join(where)}" if where else ""
            sql = f"SELECT AVG(length_of_stay_days) AS avg_length_of_stay_days FROM admissions{clause}"
            return sql, "Computes the average length of stay in days" + (
                f" for {condition} patients" if condition else ""
            )

        if q.strip().startswith(("how many", "count")) or "how many" in q:
            where = []
            if condition:
                where.append(f"medical_condition = '{condition}'")
            if admission_type:
                where.append(f"admission_type = '{admission_type}'")
            if test_result:
                where.append(f"test_results = '{test_result}'")
            clause = f" WHERE {' AND '.join(where)}" if where else ""
            sql = f"SELECT COUNT(*) AS patient_count FROM admissions{clause}"
            return sql, "Counts admissions matching the requested filters."

        if doctor_match:
            name_fragment = doctor_match.group(1)
            sql = (
                "SELECT a.patient_name, a.medical_condition, a.date_of_admission, a.admission_type "
                "FROM admissions a JOIN doctors d ON a.doctor_id = d.doctor_id "
                f"WHERE d.name LIKE '%{name_fragment.title()}%' "
                "ORDER BY a.date_of_admission DESC"
            )
            return sql, f"Lists patients whose doctor's name contains '{name_fragment.title()}'."

        if condition or admission_type or age_gt or age_lt or "this week" in q or "recent" in q:
            where = []
            if condition:
                where.append(f"medical_condition = '{condition}'")
            if admission_type:
                where.append(f"admission_type = '{admission_type}'")
            if test_result:
                where.append(f"test_results = '{test_result}'")
            for m, op in ((age_gt, ">"), (age_lt, "<")):
                if m:
                    val = next(g for g in m.groups() if g)
                    where.append(f"age {op} {int(val)}")
            if "this week" in q:
                where.append("date_of_admission >= date('now', '-7 day')")
            clause = f" WHERE {' AND '.join(where)}" if where else ""
            sql = (
                "SELECT patient_name, age, gender, medical_condition, admission_type, "
                f"date_of_admission FROM admissions{clause} ORDER BY date_of_admission DESC"
            )
            return sql, "Lists admissions matching the requested filters."

        return None

    # -- Public entry point ----------------------------------------------------
    def answer(self, question: str, history: str = "") -> SQLAgentResult:
        t0 = time.time()

        if self.llm.available:
            sql = self._generate_sql_llm(question, history)
            used_llm = True
            explanation = None
        else:
            fallback = self._fallback_sql(question)
            if fallback is None:
                return SQLAgentResult(
                    success=False,
                    error=(
                        "No LLM is configured and this question doesn't match a known pattern. "
                        "Set ANTHROPIC_API_KEY / OPENAI_API_KEY / GROQ_API_KEY and LLM_PROVIDER to "
                        "enable full natural-language SQL generation."
                    ),
                    elapsed_ms=(time.time() - t0) * 1000,
                )
            sql, explanation = fallback
            used_llm = False

        if sql is None or sql.strip().upper() == "NO_QUERY":
            return SQLAgentResult(
                success=False,
                error="This question doesn't appear to be answerable from the hospital admissions data.",
                used_llm=used_llm,
                elapsed_ms=(time.time() - t0) * 1000,
            )

        retries = 0
        validation = validate_and_sanitize(sql, self.settings.sql_row_limit)
        if not validation.is_valid and used_llm:
            # Give the model one chance to self-correct against the validator's error.
            retries += 1
            sql = self._generate_sql_llm(question, history, prior_error=validation.error) or sql
            validation = validate_and_sanitize(sql, self.settings.sql_row_limit)

        if not validation.is_valid:
            return SQLAgentResult(
                success=False,
                sql=sql,
                error=f"Generated SQL failed safety validation: {validation.error}",
                used_llm=used_llm,
                retries=retries,
                elapsed_ms=(time.time() - t0) * 1000,
            )

        execution = execute_sql(self.engine, validation.safe_sql)

        if not execution.success and used_llm and retries == 0:
            retries += 1
            sql = self._generate_sql_llm(question, history, prior_error=execution.error) or sql
            validation = validate_and_sanitize(sql, self.settings.sql_row_limit)
            if validation.is_valid:
                execution = execute_sql(self.engine, validation.safe_sql)

        if not execution.success:
            return SQLAgentResult(
                success=False,
                sql=validation.safe_sql,
                error=execution.error,
                used_llm=used_llm,
                retries=retries,
                elapsed_ms=(time.time() - t0) * 1000,
            )

        if used_llm:
            explanation = self._explain_llm(question, validation.safe_sql) or "Query executed successfully."

        answer_text = self._summarize(execution, explanation)

        return SQLAgentResult(
            success=True,
            sql=validation.safe_sql,
            explanation=explanation,
            execution=execution,
            answer=answer_text,
            used_llm=used_llm,
            retries=retries,
            elapsed_ms=round((time.time() - t0) * 1000, 1),
        )

    @staticmethod
    def _summarize(execution: ExecutionResult, explanation: str | None) -> str:
        if execution.row_count == 0:
            return "No matching records were found."
        if execution.row_count == 1 and len(execution.columns) == 1:
            value = execution.rows[0][execution.columns[0]]
            return f"{explanation + ' ' if explanation else ''}Result: {value}"
        return f"{explanation + ' ' if explanation else ''}Found {execution.row_count} matching record(s)."
