# Database Schema

## Source data

The raw dataset is a single flat CSV, one row per hospital admission:
`Name, Age, Gender, Blood Type, Medical Condition, Date of Admission,
Doctor, Hospital, Insurance Provider, Billing Amount, Room Number,
Admission Type, Discharge Date, Medication, Test Results`. 55,500 raw
rows; after dropping exact duplicates, 54,966 admissions are loaded.

## Column analysis (why the schema looks like this)

- **`Name` is not a patient identifier.** It's a synthetic display name
  with no guarantee of uniqueness across unrelated admissions, and the
  dataset has no separate patient ID field. Modeling a `patients` table
  keyed by name would silently and incorrectly merge different people
  who happen to share a generated name. Instead, the fact table's grain
  is **one row per admission**, with demographic fields (name, age,
  gender, blood type) attached directly to that row -- this is both more
  honest about what the data actually supports and matches how the
  source system (a hospital admissions feed) would realistically emit
  data before patient-matching/MPI logic is applied.
- **`Doctor`, `Hospital`, `Insurance Provider` are highly repetitive free
  text** (40,341 distinct doctor values, 39,876 distinct hospital values,
  only 5 distinct insurance providers, across 54,966 rows). These are
  classic dimension-table candidates: normalizing them (a) avoids
  repeating long strings 50k+ times, (b) gives the query planner an
  indexed integer foreign key to join on instead of a `VARCHAR` scan, and
  (c) is what "study relationships, normalize if needed" concretely means
  for this dataset -- there isn't a natural patient dimension to extract,
  but there clearly is a doctor/hospital/insurer one.
- **`length_of_stay_days` is computed and stored at load time**, not
  derived per-query. "Average length of stay" is one of the example
  questions in the brief; computing `discharge_date - date_of_admission`
  inline in every query is cheap per-row but adds up when the assistant
  is asked variations of that question repeatedly across a chat session.
  Storing it is a small, clearly-justified denormalization.

## Entity-relationship diagram

```mermaid
erDiagram
    ADMISSIONS }o--|| DOCTORS : "doctor_id"
    ADMISSIONS }o--|| HOSPITALS : "hospital_id"
    ADMISSIONS }o--|| INSURANCE_PROVIDERS : "insurance_provider_id"

    ADMISSIONS {
        int admission_id PK
        string patient_name
        int age
        string gender
        string blood_type
        string medical_condition
        date date_of_admission
        date discharge_date
        string admission_type
        int doctor_id FK
        int hospital_id FK
        int insurance_provider_id FK
        float billing_amount
        int room_number
        string medication
        string test_results
        int length_of_stay_days
    }
    DOCTORS {
        int doctor_id PK
        string name
    }
    HOSPITALS {
        int hospital_id PK
        string name
    }
    INSURANCE_PROVIDERS {
        int provider_id PK
        string name
    }
```

## Indexes

`admissions` carries indexes on `medical_condition`, `admission_type`,
`date_of_admission`, `test_results`, `doctor_id`, and `age` -- the
columns that show up as `WHERE`/`ORDER BY` predicates in the example
questions from the brief ("diabetic patients older than 60", "admitted
this week", "ICU/emergency admissions"). Doctor, hospital, and insurance
provider names carry a `UNIQUE` constraint, which SQLite backs with an
index automatically, making the ETL's dimension-table lookups and any
future name-based joins fast.

## Data cleaning applied at load time (`app/database/load_data.py`)

- Name/title casing normalized (`"Bobby JacksOn"` -> `"Bobby Jackson"`)
  for doctor, hospital, patient-name, and all enum-like text columns --
  the raw data has inconsistent capitalization that would otherwise
  fragment what should be identical dimension values.
- Exact duplicate rows dropped.
- Dates parsed and length-of-stay clipped at zero (defensive; guards
  against a discharge date recorded before admission in source data).
- Billing amounts rounded to 2 decimal places.

## Why SQLite (and how to move to Postgres)

SQLite keeps the project's setup to `pip install -r requirements.txt`
and nothing else -- no separate database server to run, which matters
for a project meant to be cloned and run locally in minutes. The schema
and every query pattern here are standard ANSI SQL; moving to Postgres
is a one-line change to `DATABASE_URL` in `.env`
(`postgresql://user:pass@host/db`) plus adding `psycopg2-binary` to
`requirements.txt` -- `app/database/session.py` already branches on the
URL scheme for SQLite-specific connect args, so nothing else in the
codebase assumes SQLite.
