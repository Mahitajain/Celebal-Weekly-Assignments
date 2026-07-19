from __future__ import annotations

import pytest

from app.database.load_data import load
from app.database.session import engine
from app.rag.retriever import Retriever


@pytest.fixture(scope="session", autouse=True)
def ensure_database():
    """Ensure the SQLite DB is populated once for the whole test session."""
    from sqlalchemy import text

    from app.database.session import init_db

    init_db()
    with engine.connect() as conn:
        try:
            count = conn.execute(text("SELECT COUNT(*) FROM admissions")).scalar_one()
        except Exception:
            count = 0
    if count == 0:
        load()
    yield


@pytest.fixture(scope="session")
def db_engine(ensure_database):
    return engine


@pytest.fixture(scope="session")
def retriever(ensure_database):
    r = Retriever()
    r.build(persist=False)
    return r
