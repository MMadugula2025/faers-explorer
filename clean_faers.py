"""
clean_faers.py — multi-quarter version, matching your actual folder layout:

    raw_data/faers_ascii_2026q1/ASCII/DEMO26Q1.txt
    raw_data/faers_ascii_2025q4/ASCII/DEMO25Q4.txt
    ...

Loads DEMO, DRUG, REAC across every quarter listed in QUARTERS, combines
them, deduplicates case versions GLOBALLY (a case's latest version can be
in a different quarter than its original), cleans drug names, and writes
one combined faers_clean.db.

Usage:
    1. Make sure QUARTERS below lists every quarter folder you downloaded
       (using the "2026q1" style, matching your raw_data folder names).
    2. Run: python3 clean_faers.py
"""

import re
import sqlite3
from pathlib import Path

import pandas as pd

# ---- CONFIG: edit this list to match what you've downloaded ----
BASE_DIR = Path("./raw_data")
QUARTERS = [
    "2025q2",
    "2025q3",
    "2025Q4",
    "2026q1",
]
OUT_DB = Path("./faers_clean.db")
# --------------------------------------------------------------------


def quarter_to_file_code(quarter_folder: str) -> str:
    """
    Converts a folder-style quarter like '2026q1' into the file-suffix
    style used inside the FDA's files, e.g. '26Q1'.
    """
    year_short = quarter_folder[2:4]       # "2026q1" -> "26"
    q_part = quarter_folder[4:].upper()    # "2026q1" -> "Q1"
    return f"{year_short}{q_part}"


def load_table_for_quarter(table_name: str, quarter_folder: str) -> pd.DataFrame:
    file_code = quarter_to_file_code(quarter_folder)
    path = BASE_DIR / f"faers_ascii_{quarter_folder}" / "ASCII" / f"{table_name}{file_code}.txt"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Check that {quarter_folder} was unzipped into "
            f"raw_data/faers_ascii_{quarter_folder}/ASCII/ with files like "
            f"{table_name}{file_code}.txt"
        )
    df = pd.read_csv(path, sep="$", dtype=str, low_memory=False, encoding="latin-1")
    df.columns = [c.strip().lower() for c in df.columns]
    df["source_quarter"] = quarter_folder
    return df


def load_all_quarters(table_name: str) -> pd.DataFrame:
    frames = []
    for q in QUARTERS:
        print(f"  loading {table_name} for {q} ...")
        frames.append(load_table_for_quarter(table_name, q))
    combined = pd.concat(frames, ignore_index=True)
    print(f"  {table_name}: {len(combined):,} rows combined across {len(QUARTERS)} quarters")
    return combined


def dedupe_to_latest_case(demo: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only the most recent primaryid per caseid, across ALL quarters
    combined — must happen after combining, since a case's latest version
    can land in a later quarter than its original report.
    """
    demo = demo.copy()
    demo["primaryid_num"] = pd.to_numeric(demo["primaryid"], errors="coerce")
    demo = demo.sort_values("primaryid_num")
    latest = demo.groupby("caseid", as_index=False).tail(1)
    return latest.drop(columns="primaryid_num")


DOSAGE_PATTERN = re.compile(r"\b\d+(\.\d+)?\s*(MG|MCG|G|ML|IU|MEQ|%)\b", re.IGNORECASE)


def normalize_drug_name(raw_name: str) -> str:
    if not isinstance(raw_name, str):
        return ""
    name = raw_name.upper().strip()
    name = DOSAGE_PATTERN.sub("", name)
    name = re.sub(r"[^A-Z0-9 \-]", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def build_database():
    print(f"Loading {len(QUARTERS)} quarters: {', '.join(QUARTERS)}\n")

    print("Loading DEMO across all quarters...")
    demo = load_all_quarters("DEMO")
    print("Loading DRUG across all quarters...")
    drug = load_all_quarters("DRUG")
    print("Loading REAC across all quarters...")
    reac = load_all_quarters("REAC")

    print("\nDeduplicating case versions GLOBALLY (across all quarters)...")
    demo_clean = dedupe_to_latest_case(demo)
    keep_ids = set(demo_clean["primaryid"])
    print(f"  Kept {len(demo_clean):,} of {len(demo):,} total DEMO rows after global dedup")

    drug_clean = drug[drug["primaryid"].isin(keep_ids)].copy()
    reac_clean = reac[reac["primaryid"].isin(keep_ids)].copy()

    print("\nNormalizing drug names...")
    drug_clean["drugname_clean"] = drug_clean["drugname"].apply(normalize_drug_name)

    print(f"\nWriting combined, cleaned data to {OUT_DB} ...")
    conn = sqlite3.connect(OUT_DB)
    demo_clean.to_sql("demo", conn, if_exists="replace", index=False)
    drug_clean.to_sql("drug", conn, if_exists="replace", index=False)
    reac_clean.to_sql("reac", conn, if_exists="replace", index=False)

    conn.execute("CREATE INDEX IF NOT EXISTS idx_drug_primaryid ON drug(primaryid)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_drug_name ON drug(drugname_clean)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_reac_primaryid ON reac(primaryid)")
    conn.commit()
    conn.close()

    print(f"\nDone. All {len(QUARTERS)} quarters combined into {OUT_DB}.")
    print("Each row has a 'source_quarter' column so you can trace it back if needed.")


if __name__ == "__main__":
    build_database()
