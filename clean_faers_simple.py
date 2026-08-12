"""
clean_faers_simple.py

The SIMPLE version: one quarter, three files (DEMO, DRUG, REAC).
Goal is to understand what's happening before scaling up to more
quarters or more files.

What this does, in plain terms:
  1. Loads three tables: who the patient was (DEMO), what drug they took
     (DRUG), and what reaction happened (REAC).
  2. Removes duplicate versions of the same report (a report can get
     revised/re-filed; we only want the latest version of each).
  3. Cleans up messy drug names (e.g. strips dosage numbers so
     "ASPIRIN 81MG" and "ASPIRIN 325MG" can be grouped as "ASPIRIN").
  4. Saves one clean, combined file: faers_clean.db (a SQLite database
     you can query, or open in a free tool like DB Browser for SQLite
     to look at like a spreadsheet).
"""

import re
import sqlite3
from pathlib import Path

import pandas as pd

# ---- These paths match YOUR folder structure exactly ----
ASCII_DIR = Path("./raw_data/faers_ascii_2026q1/ASCII")
QUARTER_CODE = "26Q1"
OUT_DB = Path("./faers_clean.db")
# -----------------------------------------------------------


def load_table(table_name: str) -> pd.DataFrame:
    """Load one .txt file, e.g. DEMO26Q1.txt, into a pandas table."""
    path = ASCII_DIR / f"{table_name}{QUARTER_CODE}.txt"
    print(f"Loading {path} ...")
    df = pd.read_csv(path, sep="$", dtype=str, low_memory=False, encoding="latin-1")
    df.columns = [c.strip().lower() for c in df.columns]
    print(f"  -> {len(df):,} rows, {len(df.columns)} columns")
    return df


def dedupe_to_latest_case(demo: pd.DataFrame) -> pd.DataFrame:
    """
    Some reports get revised. Each version has its own primaryid, but
    shares the same caseid. We only want the newest version per caseid.
    """
    demo = demo.copy()
    demo["primaryid_num"] = pd.to_numeric(demo["primaryid"], errors="coerce")
    demo = demo.sort_values("primaryid_num")
    latest = demo.groupby("caseid", as_index=False).tail(1)
    print(f"Deduped: kept {len(latest):,} of {len(demo):,} DEMO rows")
    return latest.drop(columns="primaryid_num")


DOSAGE_PATTERN = re.compile(r"\b\d+(\.\d+)?\s*(MG|MCG|G|ML|IU|MEQ|%)\b", re.IGNORECASE)


def normalize_drug_name(raw_name: str) -> str:
    """Uppercase, strip dosage numbers/units, strip stray punctuation."""
    if not isinstance(raw_name, str):
        return ""
    name = raw_name.upper().strip()
    name = DOSAGE_PATTERN.sub("", name)
    name = re.sub(r"[^A-Z0-9 \-]", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def main():
    print(f"=== Cleaning FAERS quarter {QUARTER_CODE} ===\n")

    demo = load_table("DEMO")
    drug = load_table("DRUG")
    reac = load_table("REAC")

    print()
    demo_clean = dedupe_to_latest_case(demo)

    # Only keep DRUG/REAC rows whose primaryid survived the dedup step
    keep_ids = set(demo_clean["primaryid"])
    drug_clean = drug[drug["primaryid"].isin(keep_ids)].copy()
    reac_clean = reac[reac["primaryid"].isin(keep_ids)].copy()

    print("\nCleaning drug names...")
    drug_clean["drugname_clean"] = drug_clean["drugname"].apply(normalize_drug_name)

    # Show a sanity-check example so you can SEE the cleaning working
    example = drug_clean[drug_clean["drugname_clean"].str.contains("ASPIRIN", na=False)]
    if not example.empty:
        print("Example — raw names that got cleaned into 'ASPIRIN...':")
        for raw in example["drugname"].unique()[:5]:
            print(f"    {raw!r}")

    print(f"\nSaving to {OUT_DB} ...")
    conn = sqlite3.connect(OUT_DB)
    demo_clean.to_sql("demo", conn, if_exists="replace", index=False)
    drug_clean.to_sql("drug", conn, if_exists="replace", index=False)
    reac_clean.to_sql("reac", conn, if_exists="replace", index=False)
    conn.commit()
    conn.close()

    print("\nDone! You now have faers_clean.db with 3 tables: demo, drug, reac.")
    print("Next: we'll query it to make sure it looks right.")


if __name__ == "__main__":
    main()
