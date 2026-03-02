# app.py
# Streamlit Community Cloud-ready dashboard (end-to-end)
#
# Upload CSV/XLSX -> select sheet -> apply your rules -> monthly/overall weighted aggregation
# -> visuals + tables -> download CSVs (zip + individual).
#
# Locked rules (as implemented):
# - Exclude primary World rows (reporterDesc != "World" and partnerDesc != "World") [toggleable]
# - Keep ONLY partner2Desc == "World" [toggleable]
# - qty >= min_qty
# - value_used = fobvalue if present else primaryValue
# - weighted_avg_price ($/ton) = (sum(value_used) / sum(qty)) * ton_factor
#   where ton_factor = 1000 if qtyUnitAbbr == "kg" else 1
#
# Streamlit Cloud:
# - Put this file at repo root as app.py
# - requirements.txt at repo root:
#     streamlit
#     pandas
#     numpy
#     openpyxl
#     plotly
#
# Local run:
#   python3 -m pip install -r requirements.txt
#   python3 -m streamlit run app.py

from __future__ import annotations

import io
import re
import zipfile
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

try:
    import plotly.express as px
except Exception:
    px = None

Key = Tuple[str, str, str]  # (month, sender, receiver)


# -----------------------
# Config & helpers
# -----------------------

@dataclass(frozen=True)
class Config:
    min_qty: float
    threshold: float
    exclude_primary_world: bool
    keep_partner2_world_only: bool


def _to_float_series(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
        .str.replace(",", "", regex=False)
        .replace({"None": "", "nan": ""})
        .pipe(pd.to_numeric, errors="coerce")
        .fillna(0.0)
    )


def _normalize_period_to_month(period: str) -> str:
    p = (period or "").strip()
    if not p:
        return ""

    m = re.fullmatch(r"(\d{4})(\d{2})", p)  # YYYYMM
    if m:
        return f"{m.group(1)}-{m.group(2)}"

    m = re.fullmatch(r"(\d{4})[-/](\d{1,2})", p)  # YYYY-MM or YYYY/MM
    if m:
        return f"{int(m.group(1)):04d}-{int(m.group(2)):02d}"

    m = re.fullmatch(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})", p)  # YYYY-MM-DD / YYYY/MM/DD
    if m:
        return f"{int(m.group(1)):04d}-{int(m.group(2)):02d}"

    return p


def add_month_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in ["refYear", "refMonth", "period"]:
        if c not in df.columns:
            df[c] = np.nan

    ref_year = pd.to_numeric(df["refYear"], errors="coerce")
    ref_month = pd.to_numeric(df["refMonth"], errors="coerce")

    month = pd.Series(["Unknown"] * len(df), index=df.index)
    mask = ref_year.notna() & ref_month.notna()
    month.loc[mask] = (
        ref_year.loc[mask].astype(int).map(lambda y: f"{y:04d}")
        + "-"
        + ref_month.loc[mask].astype(int).map(lambda m: f"{m:02d}")
    )

    period_str = df["period"].astype(str).replace({"nan": "", "None": ""}).fillna("")
    mask2 = (month == "Unknown") & period_str.ne("")
    month.loc[mask2] = period_str.loc[mask2].map(_normalize_period_to_month).replace({"": "Unknown"})

    df["month"] = month
    return df


def compute_value_used(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "fobvalue" not in df.columns:
        df["fobvalue"] = 0.0
    if "primaryValue" not in df.columns:
        df["primaryValue"] = 0.0

    df["fobvalue_num"] = _to_float_series(df["fobvalue"])
    df["primaryValue_num"] = _to_float_series(df["primaryValue"])
    df["value_used"] = np.where(df["fobvalue_num"] != 0.0, df["fobvalue_num"], df["primaryValue_num"])
    return df


def compute_ton_factor(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "qtyUnitAbbr" not in df.columns:
        df["qtyUnitAbbr"] = ""
    df["qty_unit"] = df["qtyUnitAbbr"].astype(str).str.strip().str.lower()
    df["ton_factor"] = np.where(df["qty_unit"] == "kg", 1000.0, 1.0)
    return df


def apply_filters(df: pd.DataFrame, cfg: Config, flow: str) -> pd.DataFrame:
    df = df.copy()
    for c in ["flowDesc", "reporterDesc", "partnerDesc", "partner2Desc"]:
        if c not in df.columns:
            df[c] = ""

    df["flowDesc"] = df["flowDesc"].astype(str).str.strip()
    df["reporterDesc"] = df["reporterDesc"].astype(str).str.strip()
    df["partnerDesc"] = df["partnerDesc"].astype(str).str.strip()
    df["partner2Desc"] = df["partner2Desc"].astype(str).str.strip()

    if "qty" not in df.columns:
        df["qty"] = 0.0
    df["qty_num"] = _to_float_series(df["qty"])

    mask = df["flowDesc"].eq(flow)

    if cfg.exclude_primary_world:
        mask &= df["reporterDesc"].str.lower().ne("world")
        mask &= df["partnerDesc"].str.lower().ne("world")

    if cfg.keep_partner2_world_only:
        mask &= df["partner2Desc"].str.lower().eq("world")

    mask &= df["qty_num"] >= float(cfg.min_qty)

    return df.loc[mask].copy()


def _classify(avg: pd.Series, threshold: float) -> pd.Series:
    return np.where(avg > threshold, "Porsche 911", np.where(avg < threshold, "Cayman", "Boundary"))


def aggregate_monthly(df_filtered: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    g = (
        df_filtered.groupby(["month", "reporterDesc", "partnerDesc"], dropna=False)
        .agg(
            shipment_count=("qty_num", "size"),
            total_qty=("qty_num", "sum"),
            total_value=("value_used", "sum"),
            ton_factor=("ton_factor", "first"),
        )
        .reset_index()
    )
    g["weighted_avg_price"] = np.where(
        g["total_qty"] > 0, (g["total_value"] / g["total_qty"]) * g["ton_factor"], 0.0
    )
    g["class"] = _classify(g["weighted_avg_price"], float(cfg.threshold))
    g = g.rename(columns={"reporterDesc": "sender_country", "partnerDesc": "receiver_country"})
    return g


def aggregate_overall(df_filtered: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    g = (
        df_filtered.groupby(["reporterDesc", "partnerDesc"], dropna=False)
        .agg(
            shipment_count=("qty_num", "size"),
            total_qty=("qty_num", "sum"),
            total_value=("value_used", "sum"),
            ton_factor=("ton_factor", "first"),
        )
        .reset_index()
    )
    g["weighted_avg_price"] = np.where(
        g["total_qty"] > 0, (g["total_value"] / g["total_qty"]) * g["ton_factor"], 0.0
    )
    g["class"] = _classify(g["weighted_avg_price"], float(cfg.threshold))
    g["month"] = "(All)"
    g = g.rename(columns={"reporterDesc": "sender_country", "partnerDesc": "receiver_country"})
    # keep same column order as monthly
    g = g[["month", "sender_country", "receiver_country", "shipment_count", "total_qty", "total_value", "ton_factor", "weighted_avg_price", "class"]]
    return g


def make_pivot_views(groups: pd.DataFrame, klass: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    subset = groups[groups["class"] == klass].copy()

    sender_first = subset.sort_values(["month", "sender_country", "receiver_country"]).reset_index(drop=True)
    receiver_first = subset.sort_values(["month", "receiver_country", "sender_country"]).reset_index(drop=True)

    cols_sender = ["month", "sender_country", "receiver_country", "shipment_count", "total_qty", "total_value", "weighted_avg_price"]
    cols_receiver = ["month", "receiver_country", "sender_country", "shipment_count", "total_qty", "total_value", "weighted_avg_price"]
    return sender_first[cols_sender], receiver_first[cols_receiver]


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


def to_zip_bytes(files: Dict[str, bytes]) -> bytes:
    zbuf = io.BytesIO()
    with zipfile.ZipFile(zbuf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, content in files.items():
            zf.writestr(name, content)
    return zbuf.getvalue()


def require_columns(df: pd.DataFrame, cols: set[str]) -> Optional[list[str]]:
    missing = sorted([c for c in cols if c not in df.columns])
    return missing if missing else None


# -----------------------
# Streamlit UI
# -----------------------

st.set_page_config(page_title="Trade Pivot Dashboard", layout="wide")

# Use a raw string for CSS to avoid any accidental escape issues.
st.markdown(
    r"""
    <style>
      .kpi-card { padding: 12px 14px; border-radius: 14px; border: 1px solid rgba(125,125,125,.25); }
      .muted { opacity: .75; font-size: 0.9rem; }
      .small { font-size: 0.85rem; opacity: .8; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Trade Pivot Dashboard")
st.caption("Upload trade data → apply rules → explore pivots & trends → download outputs.")

with st.sidebar:
    st.header("Data")
    uploaded = st.file_uploader("Upload XLSX or CSV", type=["xlsx", "xlsm", "csv"])

    st.header("Rules")
    min_qty = st.number_input("Minimum qty", min_value=0.0, value=5000.0, step=1000.0)
    threshold = st.number_input("Threshold ($/ton)", min_value=0.0, value=14000.0, step=500.0)

    exclude_primary_world = st.checkbox("Exclude primary World rows", value=True)
    keep_partner2_world_only = st.checkbox('Keep ONLY partner2Desc == "World"', value=True)

    cfg = Config(
        min_qty=float(min_qty),
        threshold=float(threshold),
        exclude_primary_world=exclude_primary_world,
        keep_partner2_world_only=keep_partner2_world_only,
    )

    st.header("Aggregation")
    agg_mode = st.radio("Mode", ["Monthly (by month)", "Overall (across months)"], index=0)

    flows = st.multiselect("Flows", options=["Export", "Import"], default=["Export", "Import"])

    st.header("Display")
    default_class = st.selectbox("Default class", ["Porsche 911", "Cayman", "Boundary"], index=0)
    show_raw_preview = st.checkbox("Show raw preview", value=False)
    show_debug = st.checkbox("Show debug", value=False)

if not uploaded:
    st.info("Upload an XLSX or CSV to begin.")
    st.stop()

# Load file
file_name = uploaded.name.lower()
sheet: Optional[str] = None

if file_name.endswith((".xlsx", ".xlsm")):
    xls = pd.ExcelFile(uploaded)
    sheet = st.selectbox("Select sheet", options=xls.sheet_names, index=0)
    df = pd.read_excel(xls, sheet_name=sheet, dtype=object)
else:
    df = pd.read_csv(uploaded, dtype=object)

# Validate schema
required = {"flowDesc", "reporterDesc", "partnerDesc", "partner2Desc", "qty", "qtyUnitAbbr", "fobvalue", "primaryValue"}
missing = require_columns(df, required)
if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

# Derived columns
df2 = add_month_column(df)
df2 = compute_value_used(df2)
df2 = compute_ton_factor(df2)

if show_raw_preview:
    with st.expander("Raw data preview", expanded=False):
        st.dataframe(df2.head(50), use_container_width=True)

# Process flows
outputs: Dict[str, Dict[str, pd.DataFrame]] = {}
groups_by_flow: Dict[str, pd.DataFrame] = {}
filtered_rows_count: Dict[str, int] = {}

for flow in flows:
    dff = apply_filters(df2, cfg, flow=flow)
    filtered_rows_count[flow] = int(len(dff))

    if agg_mode == "Overall (across months)":
        groups = aggregate_overall(dff, cfg)
    else:
        groups = aggregate_monthly(dff, cfg)

    groups_by_flow[flow] = groups

    p_sender, p_receiver = make_pivot_views(groups, "Porsche 911")
    c_sender, c_receiver = make_pivot_views(groups, "Cayman")
    b_sender, b_receiver = make_pivot_views(groups, "Boundary")

    outputs[flow] = {
        "groups": groups,
        "porsche_sender": p_sender,
        "porsche_receiver": p_receiver,
        "cayman_sender": c_sender,
        "cayman_receiver": c_receiver,
        "boundary_sender": b_sender,
        "boundary_receiver": b_receiver,
    }

# KPIs
st.subheader("Overview")
kpi_cols = st.columns(max(len(flows), 1))
for i, flow in enumerate(flows if flows else ["(none)"]):
    with kpi_cols[i]:
        if flow == "(none)":
            st.warning("Select at least one flow.")
            continue
        g = groups_by_flow[flow]
        st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
        st.markdown(f"### {flow}")
        st.markdown(f'<div class="muted">Filtered rows: {filtered_rows_count[flow]:,}</div>', unsafe_allow_html=True)
        st.metric("Groups", int(len(g)))
        st.metric("Porsche", int((g["class"] == "Porsche 911").sum()))
        st.metric("Cayman", int((g["class"] == "Cayman").sum()))
        st.metric("Boundary", int((g["class"] == "Boundary").sum()))
        st.markdown("</div>", unsafe_allow_html=True)

tabs = st.tabs(["Explore", "Trends", "Matrix", "Downloads", "Debug"])

# ---------------- Explore ----------------
with tabs[0]:
    st.subheader("Explore")
    if not flows:
        st.warning("Select at least one flow in the sidebar.")
        st.stop()

    flow = st.selectbox("Flow", options=flows, index=0)
    klass = st.selectbox("Class", options=["Porsche 911", "Cayman", "Boundary"],
                         index=["Porsche 911", "Cayman", "Boundary"].index(default_class))

    g = outputs[flow]["groups"].copy()
    g = g[g["class"] == klass].copy()

    colA, colB, colC = st.columns(3)
    with colA:
        month_opts = sorted(outputs[flow]["groups"]["month"].unique())
        month_filter = st.multiselect("Months", options=month_opts, default=[])
    with colB:
        sender_filter = st.multiselect("Senders", options=sorted(outputs[flow]["groups"]["sender_country"].unique()), default=[])
    with colC:
        receiver_filter = st.multiselect("Receivers", options=sorted(outputs[flow]["groups"]["receiver_country"].unique()), default=[])

    if month_filter:
        g = g[g["month"].isin(month_filter)]
    if sender_filter:
        g = g[g["sender_country"].isin(sender_filter)]
    if receiver_filter:
        g = g[g["receiver_country"].isin(receiver_filter)]

    col1, col2 = st.columns([1, 2])
    with col1:
        sort_by = st.selectbox("Sort by", ["total_value", "total_qty", "weighted_avg_price", "shipment_count"], index=0)
        top_n = st.slider("Top N", 10, 500, 200)
    with col2:
        st.markdown('<div class="small">Group-level metrics. weighted_avg_price is computed from totals (not avg-of-avgs).</div>',
                    unsafe_allow_html=True)

    g_sorted = g.sort_values(sort_by, ascending=False).head(top_n).copy()

    # reconciliation columns (prove weighted avg is computed from totals)
    g_sorted["recalc_weighted_avg"] = np.where(
        g_sorted["total_qty"] > 0,
        (g_sorted["total_value"] / g_sorted["total_qty"]) * g_sorted.get("ton_factor", 1.0),
        0.0
    )
    g_sorted["delta"] = g_sorted["weighted_avg_price"] - g_sorted["recalc_weighted_avg"]

    st.dataframe(
        g_sorted[["month", "sender_country", "receiver_country", "shipment_count",
                  "total_qty", "total_value", "weighted_avg_price",
                  "recalc_weighted_avg", "delta", "class"]],
        use_container_width=True,
    )

# ---------------- Trends ----------------
with tabs[1]:
    st.subheader("Trends")
    if agg_mode == "Overall (across months)":
        st.info("Trends are only available in Monthly mode.")
        st.stop()

    if not flows:
        st.warning("Select at least one flow in the sidebar.")
        st.stop()

    flow = st.selectbox("Flow for trends", options=flows, index=0, key="trend_flow")
    groups = outputs[flow]["groups"].copy()

    metric = st.selectbox("Metric", ["groups", "avg_price_mean", "avg_price_median", "total_value_sum", "total_qty_sum"], index=0)

    by_month = groups.groupby("month", dropna=False).agg(
        groups=("weighted_avg_price", "size"),
        avg_price_mean=("weighted_avg_price", "mean"),
        avg_price_median=("weighted_avg_price", "median"),
        total_value_sum=("total_value", "sum"),
        total_qty_sum=("total_qty", "sum"),
    ).reset_index().sort_values("month")

    if px:
        fig = px.line(by_month, x="month", y=metric, markers=True)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.dataframe(by_month, use_container_width=True)

    st.write("Class counts by month")
    by_month_class = groups.groupby(["month", "class"], dropna=False).size().reset_index(name="count").sort_values(["month", "class"])
    if px:
        fig = px.bar(by_month_class, x="month", y="count", color="class", barmode="stack")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.dataframe(by_month_class, use_container_width=True)

# ---------------- Matrix ----------------
with tabs[2]:
    st.subheader("Matrix (Top 20 x Top 20)")
    if not flows:
        st.warning("Select at least one flow in the sidebar.")
        st.stop()

    flow = st.selectbox("Flow for matrix", options=flows, index=0, key="matrix_flow")
    groups = outputs[flow]["groups"].copy()

    klass = st.selectbox("Class for matrix", ["Porsche 911", "Cayman", "Boundary"], index=0, key="matrix_class")
    if agg_mode == "Monthly (by month)":
        month = st.selectbox("Month", options=["(All)"] + sorted(groups["month"].unique()), index=0)
    else:
        month = "(All)"

    gg = groups[groups["class"] == klass].copy()
    if month != "(All)":
        gg = gg[gg["month"] == month]

    value_metric = st.selectbox("Matrix metric", ["total_value", "total_qty", "weighted_avg_price", "shipment_count"], index=0)

    pivot = gg.pivot_table(
        index="sender_country",
        columns="receiver_country",
        values=value_metric,
        aggfunc="sum",
        fill_value=0,
    )

    # reduce to top N for readability
    top_senders = gg.groupby("sender_country")[value_metric].sum().sort_values(ascending=False).head(20).index
    top_receivers = gg.groupby("receiver_country")[value_metric].sum().sort_values(ascending=False).head(20).index
    pivot = pivot.loc[pivot.index.intersection(top_senders), pivot.columns.intersection(top_receivers)]

    if px:
        fig = px.imshow(pivot, aspect="auto")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.dataframe(pivot, use_container_width=True)

# ---------------- Downloads ----------------
with tabs[3]:
    st.subheader("Downloads")
    if not flows:
        st.warning("Select at least one flow in the sidebar.")
        st.stop()

    cols = st.columns(len(flows))
    for i, flow in enumerate(flows):
        with cols[i]:
            st.markdown(f"### {flow}")
            out = outputs[flow]

            files = {
                f"{flow.lower()}/pivot_sender_receiver_porsche.csv": to_csv_bytes(out["porsche_sender"]),
                f"{flow.lower()}/pivot_receiver_sender_porsche.csv": to_csv_bytes(out["porsche_receiver"]),
                f"{flow.lower()}/pivot_sender_receiver_cayman.csv": to_csv_bytes(out["cayman_sender"]),
                f"{flow.lower()}/pivot_receiver_sender_cayman.csv": to_csv_bytes(out["cayman_receiver"]),
                f"{flow.lower()}/groups_all_classes.csv": to_csv_bytes(out["groups"]),
            }
            zip_bytes = to_zip_bytes(files)

            st.download_button(
                label=f"Download {flow} outputs (ZIP)",
                data=zip_bytes,
                file_name=f"{flow.lower()}_outputs.zip",
                mime="application/zip",
                use_container_width=True,
            )

# ---------------- Debug ----------------
with tabs[4]:
    if not show_debug:
        st.info("Enable “Show debug” in the sidebar.")
    else:
        st.subheader("Debug counts")
        for flow in flows:
            st.write(f"**{flow}** filtered rows: {filtered_rows_count[flow]:,} | groups: {len(groups_by_flow[flow]):,}")
        flow = st.selectbox("Flow (debug)", options=flows, index=0, key="debug_flow")
        st.dataframe(outputs[flow]["groups"].head(200), use_container_width=True)
