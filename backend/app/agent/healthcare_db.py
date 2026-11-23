"""Tiny SQLite patient database backing the healthcare-domain query tool (demo only).

Mirrors ``bank_db`` / ``ecomm_db``: re-seeded in the container's ephemeral
storage before every query, so a destructive statement only affects that one
call. Three tables:

- ``patients`` has name/email/phone/date of birth, so it drives the output-PII
  control demo.
- ``prescriptions`` is a deliberately PII-free ledger (48 rows), so the SQL
  row-limit steer can be demoed without an output-PII control withholding the
  answer.
- ``drugs`` is the formulary: the 15 drugs in the knowledge base with class,
  indication, typical dose, and pharmacy stock.

Fabricated data only. There is no clinician table: doctor data is out of scope.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from .demo_db import run_on_fresh_db

_DB_PATH = Path("/tmp/healthcare_patients.db")

# (id, name, email, phone, date_of_birth, plan) — fabricated demo data only.
_PATIENTS = [
    ("P-3001", "Grace Holloway", "grace.holloway@example.com", "+1-415-555-0201", "1958-03-14", "Medicare Advantage"),
    ("P-3002", "Marcus Webb", "marcus.webb@example.com", "+1-206-555-0202", "1971-11-02", "PPO"),
    ("P-3003", "Aiko Tanaka", "aiko.tanaka@example.com", "+1-312-555-0203", "1985-07-21", "HMO"),
    ("P-3004", "Daniel Okafor", "daniel.okafor@example.com", "+1-617-555-0204", "1964-01-30", "PPO"),
    ("P-3005", "Elena Petrova", "elena.petrova@example.com", "+1-512-555-0205", "1990-09-09", "HMO"),
    ("P-3006", "Samuel Ortiz", "samuel.ortiz@example.com", "+1-303-555-0206", "1949-05-17", "Medicare Advantage"),
    ("P-3007", "Fatima Rahman", "fatima.rahman@example.com", "+1-646-555-0207", "1978-12-05", "PPO"),
    ("P-3008", "Liam O'Connor", "liam.oconnor@example.com", "+1-503-555-0208", "1995-04-26", "HMO"),
]

# (name, drug_class, indication, typical_dose, stock_qty, requires_rx)
_DRUGS = [
    ("Lisinopril", "ACE Inhibitor", "Hypertension and Heart Failure", "10-40 mg once daily", 420, 1),
    ("Metformin", "Biguanide", "Type 2 Diabetes", "500-2000 mg daily (divided doses)", 610, 1),
    ("Atorvastatin", "Statin", "Hyperlipidemia and ASCVD prevention", "10-80 mg once daily", 380, 1),
    ("Amlodipine", "Calcium Channel Blocker", "Hypertension and Angina", "2.5-10 mg once daily", 290, 1),
    ("Levothyroxine", "Thyroid Hormone", "Hypothyroidism", "25-200 mcg once daily", 0, 1),
    ("Aspirin", "NSAID/Antiplatelet", "Pain; fever; cardiovascular prevention", "81-325 mg daily (low-dose)", 900, 0),
    ("Losartan", "Angiotensin Receptor Blocker", "Hypertension; diabetic nephropathy", "25-100 mg once or twice daily", 175, 1),
    ("Metoprolol", "Beta-Blocker", "Hypertension; angina; heart failure", "25-200 mg twice daily", 240, 1),
    ("Omeprazole", "Proton Pump Inhibitor", "GERD; peptic ulcers", "20-40 mg once daily", 530, 0),
    ("Albuterol", "Short-Acting Beta-2 Agonist", "Asthma; COPD", "2 puffs q4-6h as needed", 64, 1),
    ("Gabapentin", "Anticonvulsant", "Neuropathic pain; partial seizures", "300-3600 mg daily in 3 doses", 12, 1),
    ("Sertraline", "SSRI", "Depression; anxiety; OCD; PTSD", "25-200 mg once daily", 310, 1),
    ("Hydrochlorothiazide", "Thiazide Diuretic", "Hypertension; edema", "12.5-50 mg once daily", 205, 1),
    ("Warfarin", "Anticoagulant", "DVT/PE; atrial fibrillation", "Individualized by INR (2-10 mg daily)", 40, 1),
    ("Prednisone", "Corticosteroid", "Inflammatory and autoimmune conditions", "5-60 mg daily", 150, 1),
]

_STATUSES = ["Active", "Active", "Refill due", "Active", "Discontinued", "Ready for pickup"]


def _frequency(typical_dose: str) -> str:
    if "needed" in typical_dose:
        return "as needed"
    if "3 doses" in typical_dose:
        return "three times daily"
    if "twice" in typical_dose:
        return "twice daily"
    return "once daily"


def _seed_prescriptions() -> list[tuple[str, str, str, str, str, str, int, str]]:
    """48 deterministic prescription lines (6 per patient) — enough that a LIMIT matters."""
    rows = []
    per_patient = 6
    for p_idx, patient in enumerate(_PATIENTS):
        for n in range(per_patient):
            i = p_idx * per_patient + n
            drug = _DRUGS[(p_idx * 2 + n * 7) % len(_DRUGS)]
            dose = drug[3].split(" ")[0].split("-")[0]
            unit = "mcg" if "mcg" in drug[3] else "mg"
            rows.append(
                (
                    f"RX-{50001 + i}",
                    patient[0],
                    drug[0],
                    f"{dose} {unit}" if dose[0].isdigit() else drug[3],
                    _frequency(drug[3]),
                    f"2026-{1 + (p_idx + n) % 10:02d}-{1 + (p_idx * 6 + n * 4) % 28:02d}",
                    (p_idx + n * 3) % 6,
                    _STATUSES[(p_idx + 2 * n) % len(_STATUSES)],
                )
            )
    return rows


def _seed_db(conn: sqlite3.Connection) -> None:
    """Drop and recreate the three tables with the fabricated demo rows."""
    conn.executescript(
        """
        DROP TABLE IF EXISTS patients;
        CREATE TABLE patients (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT NOT NULL,
            phone TEXT NOT NULL,
            date_of_birth TEXT NOT NULL,
            plan TEXT NOT NULL
        );
        DROP TABLE IF EXISTS drugs;
        CREATE TABLE drugs (
            name TEXT PRIMARY KEY,
            drug_class TEXT NOT NULL,
            indication TEXT NOT NULL,
            typical_dose TEXT NOT NULL,
            stock_qty INTEGER NOT NULL,
            requires_rx INTEGER NOT NULL
        );
        DROP TABLE IF EXISTS prescriptions;
        CREATE TABLE prescriptions (
            id TEXT PRIMARY KEY,
            patient_id TEXT NOT NULL,
            drug TEXT NOT NULL,
            dose TEXT NOT NULL,
            frequency TEXT NOT NULL,
            prescribed_date TEXT NOT NULL,
            refills_left INTEGER NOT NULL,
            status TEXT NOT NULL
        );
        """
    )
    conn.executemany(
        "INSERT INTO patients (id, name, email, phone, date_of_birth, plan) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        _PATIENTS,
    )
    conn.executemany(
        "INSERT INTO drugs "
        "(name, drug_class, indication, typical_dose, stock_qty, requires_rx) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        _DRUGS,
    )
    conn.executemany(
        "INSERT INTO prescriptions "
        "(id, patient_id, drug, dose, frequency, prescribed_date, refills_left, status) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        _seed_prescriptions(),
    )


def query_patients(sql: str) -> str:
    """Execute the given SQL against the demo healthcare tables, whatever it is.

    No statement allow-list: Galileo Agent Control is the only thing that may
    refuse a query. Returns ``{"success", "row_count", "data": [...]}`` as JSON.
    """
    return run_on_fresh_db(_DB_PATH, _seed_db, sql)
