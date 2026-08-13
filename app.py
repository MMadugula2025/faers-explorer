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
import requests
import streamlit as st

DB_PATH = Path("./faers_clean.db")

# Direct-download URL for the full database, hosted as a GitHub Release
# asset (git commits are capped at 100MB, but Release assets allow up to
# 2GB — this lets the deployed app use the full multi-quarter dataset
# without it living in the git repo itself).
# UPDATE THIS to your actual release asset URL once you've uploaded it.
DB_DOWNLOAD_URL = "https://github.com/MMadugula2025/faers-explorer/releases/download/v1-data/faers_clean.db"

st.set_page_config(page_title="FAERS Adverse Event Explorer", layout="wide")

# Shared color palette so every chart in the app feels like one designed
# whole rather than each using Plotly's default colors independently.
ACCENT = "#0D9488"          # used for single-series bar charts
QUALITATIVE_PALETTE = [
    "#0D9488", "#2563EB", "#7C3AED", "#DB2777", "#EA580C",
    "#65A30D", "#0891B2", "#9333EA", "#DC2626", "#CA8A04",
]

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


def ensure_database_downloaded():
    """
    If faers_clean.db isn't present locally (as will be true on a fresh
    Streamlit Cloud deploy, since it's too large to live in the git repo),
    download it from the GitHub Release asset instead. Runs once per cold
    start — the file persists on disk for the life of that container.
    """
    if DB_PATH.exists():
        return

    if DB_DOWNLOAD_URL == "PASTE_YOUR_RELEASE_URL_HERE":
        st.error(
            "Database download URL isn't configured yet. Set DB_DOWNLOAD_URL "
            "at the top of app.py to your GitHub Release asset URL."
        )
        st.stop()

    with st.spinner(
        "Downloading FAERS database (about 1.7GB) — this only happens once "
        "per app restart and may take a few minutes..."
    ):
        try:
            response = requests.get(DB_DOWNLOAD_URL, stream=True, timeout=600)
            response.raise_for_status()
            with open(DB_PATH, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    f.write(chunk)
        except requests.RequestException as e:
            st.error(f"Failed to download the database: {e}")
            st.stop()


def get_connection():
    ensure_database_downloaded()
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
    received the report). Grouping happens in SQL rather than pandas,
    since SQLite can do this much faster than pulling every row into
    Python first.
    """
    conn = get_connection()
    query = """
        SELECT
            substr(demo.fda_dt, 1, 4) as year,
            COUNT(DISTINCT drug.primaryid) as reports
        FROM drug
        JOIN demo ON drug.primaryid = demo.primaryid
        WHERE drug.drugname_clean = ?
          AND demo.fda_dt IS NOT NULL
        GROUP BY year
        ORDER BY year
    """
    df = pd.read_sql(query, conn, params=(drug_clean,))
    conn.close()

    if df.empty:
        return pd.DataFrame(columns=["year", "reports"])

    df["year"] = df["year"].astype(int)
    return df


@st.cache_data(show_spinner=False)
def get_total_reports_by_year() -> pd.DataFrame:
    """
    Total FAERS report volume per year, across ALL drugs — used to
    normalize a single drug's yearly counts against overall reporting
    volume, since FAERS as a whole gets more reports every year
    regardless of any single drug's safety profile.
    """
    conn = get_connection()
    query = """
        SELECT
            substr(fda_dt, 1, 4) as year,
            COUNT(DISTINCT primaryid) as total_reports
        FROM demo
        WHERE fda_dt IS NOT NULL
        GROUP BY year
        ORDER BY year
    """
    df = pd.read_sql(query, conn)
    conn.close()

    if df.empty:
        return df

    df["year"] = df["year"].astype(int)
    return df


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


@st.cache_data(show_spinner=False)
def get_co_occurring_drugs(drug_clean: str, limit: int = 10) -> pd.DataFrame:
    """
    Which OTHER drugs most often appear on the same reports as this one.
    Since one FAERS report can list multiple drugs, this self-joins the
    drug table against itself on primaryid to find co-mentions.
    """
    conn = get_connection()
    query = """
        SELECT d2.drugname_clean as other_drug, COUNT(DISTINCT d1.primaryid) as co_reports
        FROM drug d1
        JOIN drug d2
            ON d1.primaryid = d2.primaryid
            AND d2.drugname_clean != d1.drugname_clean
        WHERE d1.drugname_clean = ?
          AND d2.drugname_clean != ''
        GROUP BY d2.drugname_clean
        ORDER BY co_reports DESC
        LIMIT ?
    """
    df = pd.read_sql(query, conn, params=(drug_clean, limit))
    conn.close()
    return df



OUTCOME_LABELS = {
    "DE": "Death",
    "LT": "Life-Threatening",
    "HO": "Hospitalization",
    "DS": "Disability",
    "CA": "Congenital Anomaly",
    "RI": "Required Intervention",
    "OT": "Other Serious Event",
}


@st.cache_data(show_spinner=False)
def get_outcome_breakdown(drug_clean: str) -> pd.DataFrame:
    """
    How serious were the outcomes reported alongside this drug?
    A single report can have more than one outcome code (e.g. both
    Hospitalization AND Life-Threatening), so these counts describe how
    often each outcome TYPE appears, not a partition of all reports.
    """
    conn = get_connection()
    query = """
        SELECT outc.outc_cod as code, COUNT(DISTINCT outc.primaryid) as n
        FROM drug
        JOIN outc ON drug.primaryid = outc.primaryid
        WHERE drug.drugname_clean = ?
        GROUP BY outc.outc_cod
        ORDER BY n DESC
    """
    df = pd.read_sql(query, conn, params=(drug_clean,))
    conn.close()

    if df.empty:
        return df

    df["outcome"] = df["code"].map(OUTCOME_LABELS).fillna(df["code"])
    return df


@st.cache_data(show_spinner=False)
def get_drug_serious_rate(drug_clean: str) -> float:
    """
    % of this drug's reports that include at least one serious outcome
    code (death, hospitalization, life-threatening, etc.) in OUTC.
    """
    conn = get_connection()
    total = get_total_reports(drug_clean)
    if total == 0:
        conn.close()
        return 0.0

    df = pd.read_sql(
        """
        SELECT COUNT(DISTINCT drug.primaryid) as serious
        FROM drug
        JOIN outc ON drug.primaryid = outc.primaryid
        WHERE drug.drugname_clean = ?
        """,
        conn,
        params=(drug_clean,),
    )
    conn.close()
    serious = int(df["serious"].iloc[0])
    return (serious / total) * 100


@st.cache_data(show_spinner=False)
def get_overall_serious_rate() -> float:
    """
    % of ALL reports in the loaded data (any drug) that include at least
    one serious outcome — the baseline to compare a single drug against.
    """
    conn = get_connection()
    total_df = pd.read_sql("SELECT COUNT(DISTINCT primaryid) as n FROM demo", conn)
    serious_df = pd.read_sql("SELECT COUNT(DISTINCT primaryid) as n FROM outc", conn)
    conn.close()

    total = int(total_df["n"].iloc[0])
    if total == 0:
        return 0.0
    serious = int(serious_df["n"].iloc[0])
    return (serious / total) * 100


@st.cache_data(show_spinner=False)
def get_top_drugs_by_year(top_n: int = 5) -> pd.DataFrame:
    """
    For each year present in the data, find the N drugs with the most
    unique report mentions that year. Uses a SQL window function
    (ROW_NUMBER) so the ranking is computed by the database, not pandas.
    """
    conn = get_connection()
    query = """
        SELECT year, drugname_clean, mentions
        FROM (
            SELECT
                substr(demo.fda_dt, 1, 4) as year,
                drug.drugname_clean as drugname_clean,
                COUNT(DISTINCT drug.primaryid) as mentions,
                ROW_NUMBER() OVER (
                    PARTITION BY substr(demo.fda_dt, 1, 4)
                    ORDER BY COUNT(DISTINCT drug.primaryid) DESC
                ) as rn
            FROM drug
            JOIN demo ON drug.primaryid = demo.primaryid
            WHERE demo.fda_dt IS NOT NULL AND drug.drugname_clean != ''
            GROUP BY year, drugname_clean
        )
        WHERE rn <= ?
        ORDER BY year, mentions DESC
    """
    df = pd.read_sql(query, conn, params=(top_n,))
    conn.close()

    if df.empty:
        return df

    df["year"] = df["year"].astype(int)
    return df


def render_top_drugs_by_year_chart(top_n: int = 5):
    """
    Bump chart (a ranking/slope chart): one line per drug, showing its
    rank position (1 = most mentioned) across each year. Deliberately a
    different chart type from the bar charts used elsewhere on the page,
    since rank-over-time is what we're trying to communicate here, not
    raw magnitude.
    """
    df = get_top_drugs_by_year(top_n=top_n)

    if df.empty:
        st.info("No data available yet — load at least one quarter to see this chart.")
        return

    # Rank within each year: 1 = most mentions that year
    df = df.sort_values(["year", "mentions"], ascending=[True, False]).copy()
    df["rank"] = df.groupby("year")["mentions"].rank(method="first", ascending=False).astype(int)

    fig = go.Figure()
    for i, drug_name in enumerate(df["drugname_clean"].unique()):
        drug_df = df[df["drugname_clean"] == drug_name].sort_values("year")
        color = QUALITATIVE_PALETTE[i % len(QUALITATIVE_PALETTE)]
        fig.add_trace(
            go.Scatter(
                x=drug_df["year"],
                y=drug_df["rank"],
                mode="lines+markers+text",
                name=drug_name,
                text=[f"{m:,}" for m in drug_df["mentions"]],
                textposition="middle right",
                hovertemplate=(
                    f"<b>{drug_name}</b><br>"
                    "Year: %{x}<br>Rank: %{y}<br>Mentions: %{text}<extra></extra>"
                ),
                marker=dict(size=10, color=color),
                line=dict(width=2, color=color),
            )
        )

    n_years = df["year"].nunique()
    fig.update_layout(
        title=f"Top {top_n} Most Mentioned Drugs by Year",
        xaxis_title="Year",
        yaxis_title="Rank (1 = most mentioned)",
        xaxis=dict(type="category"),
        yaxis=dict(autorange="reversed", dtick=1, tick0=1, range=[top_n + 0.5, 0.5]),
        legend_title="Drug",
        height=max(450, 90 * top_n),
    )
    st.plotly_chart(fig, use_container_width=True)

    if n_years <= 1:
        st.caption(
            "Only one year of data is currently loaded — add more quarters with "
            "clean_faers.py to see how rankings shift across years."
        )



def render_footer():
    """Shown at the bottom of every page in the app."""
    st.divider()
    st.caption(
        "FAERS is a **self-reported** surveillance system — anyone (patients, "
        "doctors, drug manufacturers, lawyers) can submit a report, and the FDA "
        "does not verify that the drug actually caused what was reported. A "
        "reaction or outcome appearing here alongside a drug reflects what someone "
        "*reported*, not a confirmed medical finding. These figures should be read "
        "as a starting point for questions, not as medical guidance."
    )


@st.cache_data(show_spinner=False)
def get_data_coverage() -> dict:
    """
    Basic stats about what's actually loaded into faers_clean.db, so
    anyone viewing the dashboard can see the scope of the data at a
    glance instead of guessing.
    """
    conn = get_connection()

    total_reports = pd.read_sql("SELECT COUNT(DISTINCT primaryid) as n FROM demo", conn)
    total_drugs = pd.read_sql(
        "SELECT COUNT(DISTINCT drugname_clean) as n FROM drug WHERE drugname_clean != ''",
        conn,
    )

    quarters = []
    try:
        cols = pd.read_sql("PRAGMA table_info(demo)", conn)
        if "source_quarter" in cols["name"].values:
            q_df = pd.read_sql(
                "SELECT DISTINCT source_quarter FROM demo ORDER BY source_quarter", conn
            )
            quarters = q_df["source_quarter"].tolist()
    except Exception:
        quarters = []

    conn.close()

    return {
        "total_reports": int(total_reports["n"].iloc[0]),
        "total_drugs": int(total_drugs["n"].iloc[0]),
        "quarters": quarters,
    }


def render_data_coverage_sidebar():
    coverage = get_data_coverage()
    with st.sidebar:
        st.subheader("Data currently loaded")
        st.metric("Total unique reports", f"{coverage['total_reports']:,}")
        st.metric("Unique drugs", f"{coverage['total_drugs']:,}")
        if coverage["quarters"]:
            st.write(f"**Quarters loaded ({len(coverage['quarters'])}):**")
            st.write(", ".join(coverage["quarters"]))
        else:
            st.caption(
                "Quarter labels not available for this database — rebuild with "
                "clean_faers.py to see them here."
            )


def main():
    render_data_coverage_sidebar()

    st.title("FAERS Adverse Event Explorer")
    st.caption(
        "Built on your own locally cleaned FAERS data (faers_clean.db). "
        "FAERS is voluntary, self-reported data — a reported association does "
        "not mean the drug caused the event."
    )

    drug_input = st.text_input(
        "Drug name",
        value="",
        placeholder="e.g. ASPIRIN, MOUNJARO, PREDNISONE, METHOTREXATE...",
        help="Enter a drug name as it might appear on a prescription. "
             "Try MOUNJARO, PREDNISONE, METHOTREXATE, ASPIRIN, or ACTEMRA "
             "if you're not sure what to search.",
    )

    if not drug_input:
        st.info(
            "Enter a drug name above to explore its reported adverse events. "
            "Not sure what to try? A few examples from the loaded data: "
            "**MOUNJARO**, **PREDNISONE**, **METHOTREXATE**, **ASPIRIN**, **ACTEMRA**."
        )
        st.divider()
        st.subheader("Top 5 Most Mentioned Drugs by Year")
        st.caption(
            "An overview of which drugs generated the most adverse event reports "
            "each year, across the quarters you've loaded — a starting point before "
            "you search for a specific drug above."
        )
        render_top_drugs_by_year_chart(top_n=5)
        render_footer()
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
        render_footer()
        return

    st.metric("Total reports (in your loaded data)", f"{total:,}")

    yearly_df = get_reports_by_year(drug_clean)
    reactions_df = get_top_reactions(drug_clean)
    outcome_df = get_outcome_breakdown(drug_clean)
    co_drugs_df = get_co_occurring_drugs(drug_clean)

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "Reports over time",
            "Top reported reactions",
            "Outcome severity",
            "Commonly reported with",
        ]
    )

    with tab1:
        if len(yearly_df) <= 1:
            st.info(
                "Only one year of data is currently loaded, so there's not much of "
                "a trend to show yet. Add more quarters with clean_faers.py to see "
                "a real trend line here."
            )

        normalize = st.checkbox(
            "Show as % of total FAERS reports that year (normalized)",
            value=False,
            help="FAERS' overall reporting volume grows every year regardless of "
                 "any single drug's safety profile. Normalizing shows this drug's "
                 "share of total reports each year, which controls for that growth.",
        )

        if normalize:
            totals_df = get_total_reports_by_year()
            merged = yearly_df.merge(totals_df, on="year", how="left")
            merged["pct"] = (merged["reports"] / merged["total_reports"]) * 100

            fig = go.Figure()
            fig.add_trace(
                go.Bar(x=merged["year"], y=merged["pct"], name=drug_clean, marker_color=ACCENT)
            )
            fig.update_layout(
                title=f"Reports by year, as % of all FAERS reports — {drug_clean}",
                xaxis_title="Year",
                yaxis_title="% of total FAERS reports that year",
                xaxis=dict(type="category"),
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "Each bar = (this drug's reports that year) ÷ (all drugs' reports "
                "that year). This controls for FAERS' overall growth in reporting "
                "volume, which the raw count view doesn't."
            )
        else:
            fig = go.Figure()
            fig.add_trace(
                go.Bar(x=yearly_df["year"], y=yearly_df["reports"], name=drug_clean, marker_color=ACCENT)
            )
            fig.update_layout(
                title=f"Reports by year — {drug_clean}",
                xaxis_title="Year",
                yaxis_title="Number of reports",
                xaxis=dict(type="category"),
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "Raw counts aren't adjusted for FAERS' overall reporting volume, "
                "which grows every year. Check the box above to see this drug's "
                "share of total reports instead."
            )

    with tab2:
        if reactions_df.empty:
            st.write("No reaction data found.")
        else:
            fig2 = go.Figure(
                go.Bar(
                    x=reactions_df["report_count"],
                    y=reactions_df["reaction"],
                    orientation="h",
                    marker_color=ACCENT,
                )
            )
            fig2.update_layout(
                title=f"Top {len(reactions_df)} reported reactions — {drug_clean}",
                xaxis_title="Number of reports",
                yaxis={"categoryorder": "total ascending"},
            )
            st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        drug_rate = get_drug_serious_rate(drug_clean)
        overall_rate = get_overall_serious_rate()
        delta = drug_rate - overall_rate

        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                f"{drug_clean}: reports with a serious outcome",
                f"{drug_rate:.1f}%",
                delta=f"{delta:+.1f} pts vs. overall average",
                delta_color="off",
            )
        with col2:
            st.metric("Overall average across all loaded drugs", f"{overall_rate:.1f}%")
        st.caption(
            "'Serious' means the report included at least one FAERS outcome code "
            "(death, hospitalization, life-threatening, disability, etc.). This "
            "compares this drug's reports against the average across every drug "
            "in your loaded data — both numbers come from the same self-reporting "
            "system, so it's a fair like-for-like comparison, not a claim about "
            "real-world risk."
        )
        st.divider()

        if outcome_df.empty:
            st.write(
                "No outcome data found for this drug — either no serious outcomes "
                "were coded, or OUTC data isn't loaded (run the cleaning scripts "
                "again after this update to include it)."
            )
        else:
            fig3 = go.Figure(
                go.Pie(
                    labels=outcome_df["outcome"],
                    values=outcome_df["n"],
                    hole=0.45,
                    textinfo="label+percent",
                    marker=dict(colors=QUALITATIVE_PALETTE),
                )
            )
            fig3.update_layout(
                title=f"Reported outcome types — {drug_clean}",
            )
            st.plotly_chart(fig3, use_container_width=True)
            st.caption(
                "A single report can list more than one outcome (e.g. both "
                "Hospitalization and Life-Threatening), so these counts reflect "
                "how often each outcome type was mentioned — they won't necessarily "
                "add up to your total report count above. Most FAERS reports have "
                "no serious outcome coded at all."
            )

    with tab4:
        if co_drugs_df.empty:
            st.write(
                f"No other drugs found co-mentioned with {drug_clean} in your "
                "loaded data — most of its reports may list it as the only drug."
            )
        else:
            fig4 = go.Figure(
                go.Bar(
                    x=co_drugs_df["co_reports"],
                    y=co_drugs_df["other_drug"],
                    orientation="h",
                    marker_color=ACCENT,
                )
            )
            fig4.update_layout(
                title=f"Drugs most often reported alongside {drug_clean}",
                xaxis_title="Number of shared reports",
                yaxis={"categoryorder": "total ascending"},
            )
            st.plotly_chart(fig4, use_container_width=True)
            st.caption(
                "Counts how often each drug appears on the SAME report as "
                f"{drug_clean}. This can reflect common co-prescribing (e.g. a "
                "drug given alongside another to offset a known side effect), "
                "unrelated medications the patient happened to also be taking, "
                "or a mix of both — the data alone can't distinguish which."
            )

    render_footer()


if __name__ == "__main__":
    main()
