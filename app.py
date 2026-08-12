"""
FAERS Adverse Event Explorer — local database version

Reads from your own cleaned faers_clean.db (built by clean_faers_simple.py
or clean_faers.py) instead of calling the live openFDA API. This means it
only knows about whatever quarters you've loaded into that database.

Note: FAERS is voluntary, self-reported data. A reported association
between a drug and an event does NOT mean the drug caused the event.
"""

import re
import sqlite3
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DB_PATH = Path("./faers_clean.db")

st.set_page_config(page_title="FAERS Adverse Event Explorer", layout="wide")

DOSAGE_PATTERN = re.compile(r"\b\d+(\.\d+)?\s*(MG|MCG|G|ML|IU|MEQ|%)\b", re.IGNORECASE)


def normalize_drug_name(raw_name: str) -> str:
    """Same normalization used when the database was built, so searches
    match what's actually stored in drugname_clean."""
    if not isinstance(raw_name, str):
        return ""
    name = raw_name.upper().strip()
    name = DOSAGE_PATTERN.sub("", name)
    name = re.sub(r"[^A-Z0-9 \-]", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def get_connection():
    if not DB_PATH.exists():
        st.error(
            f"Couldn't find {DB_PATH}. Run clean_faers_simple.py (or clean_faers.py) "
            "first to build the database."
        )
        st.stop()
    return sqlite3.connect(DB_PATH)


@st.cache_data(show_spinner=False)
def get_available_drugs(limit: int = 5000) -> list:
    """Pull distinct cleaned drug names for the search box's suggestions."""
    conn = get_connection()
    df = pd.read_sql(
        f"SELECT DISTINCT drugname_clean FROM drug WHERE drugname_clean != '' LIMIT {limit}",
        conn,
    )
    conn.close()
    return sorted(df["drugname_clean"].tolist())


@st.cache_data(show_spinner=False)
def get_total_reports(drug_clean: str) -> int:
    conn = get_connection()
    df = pd.read_sql(
        "SELECT COUNT(DISTINCT primaryid) as n FROM drug WHERE drugname_clean = ?",
        conn,
        params=(drug_clean,),
    )
    conn.close()
    return int(df["n"].iloc[0])


@st.cache_data(show_spinner=False)
def get_reports_by_year(drug_clean: str) -> pd.DataFrame:
    """
    Report counts grouped by year, using DEMO's fda_dt (the date FDA
    received the report). Only shows years actually present in your
    database — with one quarter loaded, you'll see one year/bar.
    """
    conn = get_connection()
    query = """
        SELECT demo.fda_dt, drug.primaryid
        FROM drug
        JOIN demo ON drug.primaryid = demo.primaryid
        WHERE drug.drugname_clean = ?
    """
    df = pd.read_sql(query, conn, params=(drug_clean,))
    conn.close()

    if df.empty:
        return pd.DataFrame(columns=["year", "reports"])

    df["year"] = pd.to_datetime(df["fda_dt"], format="%Y%m%d", errors="coerce").dt.year
    yearly = df.dropna(subset=["year"]).groupby("year", as_index=False)["primaryid"].nunique()
    yearly.columns = ["year", "reports"]
    yearly["year"] = yearly["year"].astype(int)
    return yearly.sort_values("year")


@st.cache_data(show_spinner=False)
def get_top_reactions(drug_clean: str, limit: int = 15) -> pd.DataFrame:
    conn = get_connection()
    query = """
        SELECT reac.pt as reaction, COUNT(*) as report_count
        FROM drug
        JOIN reac ON drug.primaryid = reac.primaryid
        WHERE drug.drugname_clean = ?
        GROUP BY reac.pt
        ORDER BY report_count DESC
        LIMIT ?
    """
    df = pd.read_sql(query, conn, params=(drug_clean, limit))
    conn.close()
    return df


def main():
    st.title("FAERS Adverse Event Explorer")
    st.caption(
        "Built on your own locally cleaned FAERS data (faers_clean.db). "
        "FAERS is voluntary, self-reported data — a reported association does "
        "not mean the drug caused the event."
    )

    drug_input = st.text_input(
        "Drug name",
        value="ASPIRIN",
        help="Try a drug from your loaded quarter(s), e.g. MOUNJARO, PREDNISONE, ASPIRIN",
    )

    if not drug_input:
        st.info("Enter a drug name to begin.")
        return

    drug_clean = normalize_drug_name(drug_input)

    total = get_total_reports(drug_clean)

    if total == 0:
        st.warning(
            f"No reports found for '{drug_input}' (normalized to '{drug_clean}'). "
            "It may not be in the quarter(s) you've loaded, or the spelling doesn't "
            "match exactly — FAERS drug names aren't standardized."
        )
        with st.expander("See some drug names that ARE in your database"):
            st.write(get_available_drugs(limit=50))
        return

    st.metric("Total reports (in your loaded data)", f"{total:,}")

    yearly_df = get_reports_by_year(drug_clean)
    reactions_df = get_top_reactions(drug_clean)

    tab1, tab2 = st.tabs(["Reports over time", "Top reported reactions"])

    with tab1:
        if len(yearly_df) <= 1:
            st.info(
                "Only one year of data is currently loaded, so there's not much of "
                "a trend to show yet. Add more quarters with clean_faers.py to see "
                "a real trend line here."
            )
        fig = go.Figure()
        fig.add_trace(
            go.Bar(x=yearly_df["year"], y=yearly_df["reports"], name=drug_clean)
        )
        fig.update_layout(
            title=f"Reports by year — {drug_clean}",
            xaxis_title="Year",
            yaxis_title="Number of reports",
            xaxis=dict(type="category"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        if reactions_df.empty:
            st.write("No reaction data found.")
        else:
            fig2 = go.Figure(
                go.Bar(
                    x=reactions_df["report_count"],
                    y=reactions_df["reaction"],
                    orientation="h",
                )
            )
            fig2.update_layout(
                title=f"Top {len(reactions_df)} reported reactions — {drug_clean}",
                xaxis_title="Number of reports",
                yaxis={"categoryorder": "total ascending"},
            )
            st.plotly_chart(fig2, use_container_width=True)


if __name__ == "__main__":
    main()
