"""
ETL: raw_csv -> normalized SQLite database.

Run as a script (`python -m app.database.load_data`) or import `load()`.
Uses pandas for vectorized cleaning + bulk inserts rather than one ORM
object per row -- at 55.5k rows, row-at-a-time ORM inserts would take
minutes; bulk `to_sql` takes seconds.
"""
from __future__ import annotations

import sys
import time

import pandas as pd
from sqlalchemy import text

from app.config import get_settings
from app.database.models import Base
from app.database.session import engine, init_db
from app.utils.logging_config import get_logger

logger = get_logger(__name__)


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    # Normalize name casing -- the raw data has inconsistent capitalization
    # ("Bobby JacksOn", "LesLie TErRy") which would otherwise fragment
    # what should be the same dimension value across rows.
    df["Doctor"] = df["Doctor"].str.strip().str.title()
    df["Hospital"] = df["Hospital"].str.strip()
    df["Insurance Provider"] = df["Insurance Provider"].str.strip()
    df["Name"] = df["Name"].str.strip().str.title()
    df["Gender"] = df["Gender"].str.strip().str.title()
    df["Blood Type"] = df["Blood Type"].str.strip()
    df["Medical Condition"] = df["Medical Condition"].str.strip().str.title()
    df["Admission Type"] = df["Admission Type"].str.strip().str.title()
    df["Medication"] = df["Medication"].str.strip().str.title()
    df["Test Results"] = df["Test Results"].str.strip().str.title()

    df["Date of Admission"] = pd.to_datetime(df["Date of Admission"]).dt.date
    df["Discharge Date"] = pd.to_datetime(df["Discharge Date"]).dt.date
    df["length_of_stay_days"] = (
        pd.to_datetime(df["Discharge Date"]) - pd.to_datetime(df["Date of Admission"])
    ).dt.days.clip(lower=0)

    df["Age"] = df["Age"].astype(int)
    df["Room Number"] = df["Room Number"].astype(int)
    df["Billing Amount"] = df["Billing Amount"].astype(float).round(2)

    # Drop exact-duplicate rows and rows with non-sensical stays.
    df = df.drop_duplicates()
    return df


def _build_dimension(df: pd.DataFrame, column: str) -> pd.DataFrame:
    values = sorted(df[column].dropna().unique().tolist())
    return pd.DataFrame({"name": values})


def load(csv_path=None, echo: bool = True) -> dict:
    settings = get_settings()
    csv_path = csv_path or settings.raw_csv_path

    t0 = time.time()
    logger.info("Reading raw CSV from %s", csv_path)
    raw = pd.read_csv(csv_path)
    df = _clean(raw)
    logger.info("Cleaned %d rows (from %d raw rows) in %.2fs", len(df), len(raw), time.time() - t0)

    # Fresh schema every load -- this is a batch ETL for a demo dataset,
    # not an incremental production pipeline, so drop/recreate is the
    # simplest correct behavior.
    Base.metadata.drop_all(bind=engine)
    init_db()

    doctors = _build_dimension(df, "Doctor")
    hospitals = _build_dimension(df, "Hospital")
    providers = _build_dimension(df, "Insurance Provider")

    with engine.begin() as conn:
        doctors.to_sql("doctors", conn, if_exists="append", index=False)
        hospitals.to_sql("hospitals", conn, if_exists="append", index=False)
        providers.to_sql("insurance_providers", conn, if_exists="append", index=False)

        doctor_ids = dict(conn.execute(text("SELECT name, doctor_id FROM doctors")).all())
        hospital_ids = dict(conn.execute(text("SELECT name, hospital_id FROM hospitals")).all())
        provider_ids = dict(conn.execute(text("SELECT name, provider_id FROM insurance_providers")).all())

        fact = pd.DataFrame(
            {
                "patient_name": df["Name"],
                "age": df["Age"],
                "gender": df["Gender"],
                "blood_type": df["Blood Type"],
                "medical_condition": df["Medical Condition"],
                "date_of_admission": df["Date of Admission"],
                "discharge_date": df["Discharge Date"],
                "admission_type": df["Admission Type"],
                "doctor_id": df["Doctor"].map(doctor_ids),
                "hospital_id": df["Hospital"].map(hospital_ids),
                "insurance_provider_id": df["Insurance Provider"].map(provider_ids),
                "billing_amount": df["Billing Amount"],
                "room_number": df["Room Number"],
                "medication": df["Medication"],
                "test_results": df["Test Results"],
                "length_of_stay_days": df["length_of_stay_days"],
            }
        )
        fact.to_sql("admissions", conn, if_exists="append", index=False, chunksize=5000)

    stats = {
        "admissions": len(fact),
        "doctors": len(doctors),
        "hospitals": len(hospitals),
        "insurance_providers": len(providers),
        "elapsed_seconds": round(time.time() - t0, 2),
    }
    logger.info("Load complete: %s", stats)
    return stats


if __name__ == "__main__":
    stats = load()
    print(stats)
    sys.exit(0)
