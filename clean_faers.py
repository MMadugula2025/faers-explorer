"""
clean_faers.py (multi-quarter version)

Loads MULTIPLE quarters of raw FAERS ASCII data, combines them into one
dataset, deduplicates case versions GLOBALLY across all quarters (since a
case's follow-up report can land in a later quarter than the original),
normalizes drug names, and builds one clean SQLite database.

Expected folder layout:

    raw_data/
        23Q1/ASCII/DEMO23Q1.txt, DRUG23Q1.txt, REAC23Q1.txt, OUTC23Q1.txt
        23Q2/ASCII/DEMO23Q2.txt, ...
        ...
        26Q2/ASCII/...

Usage:
    1. Update QUARTERS below to list every quarter you downloaded.
    2. Update BASE_DIR if your folder isn't named "raw_data".
    3. Run: python clean_faers.py
"""

import re
import sqlite3
from pathlib import Path

import pandas as pd

# ---- CONFIG: edit these for your download ----
BASE_DIR = Path("./raw_data")
QUARTERS = [
    "23Q1", "23Q2", "23Q3", "23Q4",
    "24Q1", "24Q2", "24Q3", "24Q4",
    "25Q1", "25Q2", "25Q3", "25Q4",
    "26Q1", "26Q2",
]
OUT_DB = Path("./faers_clean.db")
# ------------------------------------------------


def load_table_for_quarter(table_name: str, quarter: str) -> pd.DataFrame:
    """Load one table (e.g. DEMO) for one quarter (e.g. 23Q1)."""
    path = BASE_DIR / quarter / "ASCII" / f"{table_name}{quarter}.txt"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Check that you unzipped {quarter} into "
            f"raw_data/{quarter}/ASCII/ with files named like {table_name}{quarter}.txt"
        )
    df = pd.read_csv(path, sep="$", dtype=str, low_memory=False, encoding="latin-1")
    df.columns = [c.strip().lower() for c in df.columns]
    df["source_quarter"] = quarter  # keep track of which quarter each row came from
    return df


def load_all_quarters(table_name: str) -> pd.DataFrame:
    """Load and concatenate one table across every quarter in QUARTERS."""
    frames = []
    for q in QUARTERS:
        print(f"  loading {table_name}{q}.txt ...")
        frames.append(load_table_for_quarter(table_name, q))
    combined = pd.concat(frames, ignore_index=True)
    print(f"  {table_name}: {len(combined):,} rows combined across {len(QUARTERS)} quarters")
    return combined


def dedupe_to_latest_case(demo: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only the single most recent primaryid per caseid, across ALL
    quarters combined. This must happen AFTER combining every quarter,
    because a case's latest version can appear in a later quarter's file
    than its original report.
    """
    demo = demo.copy()
    demo["primaryid_num"] = pd.to_numeric(demo["primaryid"], errors="coerce")
    demo = demo.sort_values("primaryid_num")
    latest = demo.groupby("caseid", as_index=False).tail(1)
    return latest.drop(columns="primaryid_num")


DOSAGE_PATTERN = re.compile(
    r"\b\d+(\.\d+)?\s*(MG|MCG|G|ML|IU|MEQ|%)\b", re.IGNORECASE
)


def normalize_drug_name(raw_name: str) -> str:
    if not isinstance(raw_name, str):
        return ""
    name = raw_name.upper().strip()
    name = DOSAGE_PATTERN.sub("", name)
    name = re.sub(r"[^A-Z0-9 \-]", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def build_database():
    print(f"Loading {len(QUARTERS)} quarters: {QUARTERS[0]} through {QUARTERS[-1]}\n")

    print("Loading DEMO across all quarters...")
    demo = load_all_quarters("DEMO")
    print("Loading DRUG across all quarters...")
    drug = load_all_quarters("DRUG")
    print("Loading REAC across all quarters...")
    reac = load_all_quarters("REAC")
    print("Loading OUTC across all quarters...")
    outc = load_all_quarters("OUTC")

    print("\nDeduplicating case versions GLOBALLY (across all 14 quarters)...")
    demo_clean = dedupe_to_latest_case(demo)
    keep_ids = set(demo_clean["primaryid"])
    print(f"  Kept {len(demo_clean):,} of {len(demo):,} total DEMO rows after global dedup")

    drug_clean = drug[drug["primaryid"].isin(keep_ids)].copy()
    reac_clean = reac[reac["primaryid"].isin(keep_ids)].copy()
    outc_clean = outc[outc["primaryid"].isin(keep_ids)].copy()

    print("\nNormalizing drug names...")
    drug_clean["drugname_clean"] = drug_clean["drugname"].apply(normalize_drug_name)

    # Add a real calendar year column derived from receipt date, so you can
    # group by year for trend charts regardless of which quarter file a
    # row originally came from.
    demo_clean["event_year"] = pd.to_datetime(
        demo_clean["fda_dt"], format="%Y%m%d", errors="coerce"
    ).dt.year

    print(f"\nWriting combined, cleaned data to {OUT_DB} ...")
    conn = sqlite3.connect(OUT_DB)
    demo_clean.to_sql("demo", conn, if_exists="replace", index=False)
    drug_clean.to_sql("drug", conn, if_exists="replace", index=False)
    reac_clean.to_sql("reac", conn, if_exists="replace", index=False)
    outc_clean.to_sql("outc", conn, if_exists="replace", index=False)

    conn.execute("CREATE INDEX IF NOT EXISTS idx_drug_primaryid ON drug(primaryid)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_drug_name ON drug(drugname_clean)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_reac_primaryid ON reac(primaryid)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_demo_year ON demo(event_year)")
    conn.commit()
    conn.close()

    print("\nDone. All 14 quarters are now combined into one clean database:")
    print(f"  {OUT_DB}")
    print("Every table has a 'source_quarter' column if you ever need to trace")
    print("a row back to which download it came from.")


if __name__ == "__main__":
    build_database()
