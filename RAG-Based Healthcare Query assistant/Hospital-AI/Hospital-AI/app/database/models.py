"""
Database schema.

The raw CSV is a single flat table: one row per admission, with doctor,
hospital, and insurance provider repeated as free text on every row
(40k+ distinct doctors across 55.5k rows -- almost 1:1, but still worth
normalizing so the schema doesn't imply "one doctor row == one doctor
identity" is meaningless, and so lookups/joins have something to index).

Design decision: this dataset has no real patient identifier (`Name` is
not unique -- the same synthetic name can recur across unrelated
admissions). Rather than invent a false patient entity, we model an
`Admission` as the fact/grain of the table (one row = one hospital stay)
and keep patient demographic attributes (name, age, gender, blood type)
on that fact row. Doctor, Hospital, and InsurancePlan are proper
dimension tables since they are referenced by many admissions and
benefit from indexed foreign keys instead of repeated VARCHAR scans.

This is a conventional star-schema: one fact table (admissions) with
several small dimension tables, which is the standard pattern for
analytical / reporting workloads like "average stay by condition" or
"admissions per doctor".
"""
from __future__ import annotations

import datetime as dt

from sqlalchemy import (
    Date,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Doctor(Base):
    __tablename__ = "doctors"

    doctor_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False)

    admissions: Mapped[list["Admission"]] = relationship(back_populates="doctor")

    __table_args__ = (UniqueConstraint("name", name="uq_doctor_name"),)


class Hospital(Base):
    __tablename__ = "hospitals"

    hospital_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False)

    admissions: Mapped[list["Admission"]] = relationship(back_populates="hospital")

    __table_args__ = (UniqueConstraint("name", name="uq_hospital_name"),)


class InsuranceProvider(Base):
    __tablename__ = "insurance_providers"

    provider_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(64), nullable=False)

    admissions: Mapped[list["Admission"]] = relationship(back_populates="insurance_provider")

    __table_args__ = (UniqueConstraint("name", name="uq_provider_name"),)


class Admission(Base):
    """One row per hospital stay -- the fact table / grain of this schema."""

    __tablename__ = "admissions"

    admission_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Patient demographics (denormalized onto the fact row -- see module docstring)
    patient_name: Mapped[str] = mapped_column(String(128), nullable=False)
    age: Mapped[int] = mapped_column(Integer, nullable=False)
    gender: Mapped[str] = mapped_column(String(16), nullable=False)
    blood_type: Mapped[str] = mapped_column(String(8), nullable=False)

    medical_condition: Mapped[str] = mapped_column(String(64), nullable=False)
    date_of_admission: Mapped[dt.date] = mapped_column(Date, nullable=False)
    discharge_date: Mapped[dt.date] = mapped_column(Date, nullable=False)
    admission_type: Mapped[str] = mapped_column(String(32), nullable=False)

    doctor_id: Mapped[int] = mapped_column(ForeignKey("doctors.doctor_id"), nullable=False)
    hospital_id: Mapped[int] = mapped_column(ForeignKey("hospitals.hospital_id"), nullable=False)
    insurance_provider_id: Mapped[int] = mapped_column(
        ForeignKey("insurance_providers.provider_id"), nullable=False
    )

    billing_amount: Mapped[float] = mapped_column(Float, nullable=False)
    room_number: Mapped[int] = mapped_column(Integer, nullable=False)
    medication: Mapped[str] = mapped_column(String(64), nullable=False)
    test_results: Mapped[str] = mapped_column(String(32), nullable=False)

    # Derived at load time so "average length of stay" queries don't need
    # to compute a date-diff in every query (cheap to store, expensive-ish
    # to recompute across 55k rows repeatedly for an interactive assistant).
    length_of_stay_days: Mapped[int] = mapped_column(Integer, nullable=False)

    doctor: Mapped["Doctor"] = relationship(back_populates="admissions")
    hospital: Mapped["Hospital"] = relationship(back_populates="admissions")
    insurance_provider: Mapped["InsuranceProvider"] = relationship(back_populates="admissions")

    __table_args__ = (
        Index("ix_admissions_condition", "medical_condition"),
        Index("ix_admissions_admission_type", "admission_type"),
        Index("ix_admissions_date_of_admission", "date_of_admission"),
        Index("ix_admissions_test_results", "test_results"),
        Index("ix_admissions_doctor_id", "doctor_id"),
        Index("ix_admissions_age", "age"),
    )
