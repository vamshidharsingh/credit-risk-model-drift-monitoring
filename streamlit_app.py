"""Credit Risk Model Drift Monitoring — Streamlit Dashboard.

Tabs (dynamic based on available results):
1. Overview — KPI cards, alert timeline, default rate (SBA)
2. Drift Analysis — Score PSI, feature PSI heatmap (SBA)
3. Model Performance — AUC over time by strategy (SBA)
4. Strategy Comparison — side-by-side table, retrain events (SBA)
5. Freddie Mac — Full mortgage drift replay and strategy results
6. UCI Benchmark — synthetic drift benchmark results
7. About — methodology, datasets, decisions
"""

from __future__ import annotations

import json

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from drift.charts import (
    add_alert_bands,
    add_event_annotations,
    alert_color,
    readable_name,
)
from drift.config import (
    ALERT_AMBER_SCORE_PSI,
    ALERT_GREEN_SCORE_PSI,
    BRAND,
    CHART_PALETTE,
    INK,
    REPORTS_DIR,
)

st.set_page_config(
    page_title="Credit Risk Drift Monitor",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Custom CSS ─────────────────────────────────────────────────────────────

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* Global overrides */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Hide default Streamlit elements for cleaner look */
#MainMenu {visibility: hidden;}
header {visibility: hidden;}
footer {visibility: hidden;}

/* Main container spacing */
.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 2rem !important;
    max-width: 1200px;
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
}
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3,
section[data-testid="stSidebar"] .stMarkdown li {
    color: #e2e8f0 !important;
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background-color: transparent;
    border-bottom: 2px solid #e2e8f0;
    padding-bottom: 0;
}
.stTabs [data-baseweb="tab"] {
    height: 48px;
    padding: 0 24px;
    background-color: transparent;
    border: none;
    border-radius: 8px 8px 0 0;
    font-weight: 500;
    font-size: 0.9rem;
    color: #64748b;
    transition: all 0.2s ease;
}
.stTabs [data-baseweb="tab"]:hover {
    color: #1e40af;
    background-color: #eff6ff;
}
.stTabs [aria-selected="true"] {
    background-color: #eff6ff !important;
    color: #1e40af !important;
    font-weight: 600 !important;
    border-bottom: 3px solid #1e40af !important;
}
.stTabs [data-baseweb="tab-highlight"] {
    display: none;
}
.stTabs [data-baseweb="tab-border"] {
    display: none;
}

/* KPI card container */
.kpi-card {
    background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 20px 16px;
    text-align: center;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04), 0 4px 12px rgba(0,0,0,0.03);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
    min-height: 110px;
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 4px;
    background: linear-gradient(90deg, #1e40af, #3b82f6, #60a5fa);
    border-radius: 16px 16px 0 0;
}
.kpi-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.08), 0 8px 24px rgba(0,0,0,0.06);
    border-color: #bfdbfe;
}
.kpi-label {
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #64748b;
    margin-bottom: 6px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.kpi-value {
    font-size: 1.3rem;
    font-weight: 700;
    color: #0f172a;
    line-height: 1.3;
    margin-bottom: 4px;
    white-space: nowrap;
}
.kpi-delta {
    font-size: 0.72rem;
    font-weight: 500;
    padding: 2px 8px;
    border-radius: 12px;
    display: inline-block;
    white-space: nowrap;
}
.kpi-delta.positive {
    color: #15803d;
    background: #f0fdf4;
}
.kpi-delta.negative {
    color: #dc2626;
    background: #fef2f2;
}
.kpi-delta.neutral {
    color: #64748b;
    background: #f1f5f9;
}

/* Accent KPI cards */
.kpi-card.accent-red::before {
    background: linear-gradient(90deg, #dc2626, #ef4444, #f87171);
}
.kpi-card.accent-green::before {
    background: linear-gradient(90deg, #15803d, #22c55e, #4ade80);
}
.kpi-card.accent-amber::before {
    background: linear-gradient(90deg, #d97706, #f59e0b, #fbbf24);
}

/* Section headers */
.section-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 32px 0 16px 0;
    padding-bottom: 12px;
    border-bottom: 2px solid #e2e8f0;
}
.section-header h3 {
    font-size: 1.15rem;
    font-weight: 600;
    color: #0f172a;
    margin: 0;
}
.section-header .section-icon {
    width: 36px;
    height: 36px;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.1rem;
    background: #eff6ff;
}
.section-caption {
    font-size: 0.85rem;
    color: #64748b;
    margin: -8px 0 20px 0;
    line-height: 1.5;
}

/* Data tables */
div[data-testid="stDataFrame"] {
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    overflow: hidden;
}

/* Alert badge */
.alert-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.03em;
}
.alert-green { background: #dcfce7; color: #15803d; }
.alert-amber { background: #fef3c7; color: #92400e; }
.alert-red { background: #fee2e2; color: #991b1b; }

/* Summary cards */
.summary-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 20px;
}

/* Metric highlight */
.metric-highlight {
    background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
    border: 1px solid #bfdbfe;
    border-radius: 12px;
    padding: 16px 20px;
    margin: 8px 0;
}

/* Divider override */
hr {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, #cbd5e1, transparent);
    margin: 32px 0;
}

/* Hero header */
.hero-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #1e40af 100%);
    border-radius: 20px;
    padding: 32px 36px;
    margin-bottom: 28px;
    color: white;
    position: relative;
    overflow: hidden;
}
.hero-header::after {
    content: '';
    position: absolute;
    top: -50%;
    right: -20%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(59,130,246,0.15) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-size: 1.6rem;
    font-weight: 700;
    margin-bottom: 6px;
    position: relative;
    z-index: 1;
}
.hero-subtitle {
    font-size: 0.9rem;
    color: #93c5fd;
    font-weight: 400;
    position: relative;
    z-index: 1;
}

/* About page */
.about-block {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 28px;
    margin-bottom: 20px;
}
.about-block h4 {
    color: #1e40af;
    font-weight: 600;
    margin-bottom: 12px;
}

/* Animated fade-in */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(12px); }
    to { opacity: 1; transform: translateY(0); }
}
.stTabs [data-baseweb="tab-panel"] {
    animation: fadeInUp 0.4s ease-out;
}

/* Plotly chart container */
div[data-testid="stPlotlyChart"] {
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 8px;
    background: white;
    box-shadow: 0 1px 2px rgba(0,0,0,0.03);
}

/* Expander styling */
div[data-testid="stExpander"] {
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    overflow: hidden;
    margin-bottom: 12px;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ── Load data ───────────────────────────────────────────────────────────────


@st.cache_data
def load_drift_results() -> dict:
    path = REPORTS_DIR / "sba_drift_replay.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


@st.cache_data
def load_strategy_results() -> dict:
    path = REPORTS_DIR / "sba_strategy_comparison.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


@st.cache_data
def load_uci_results() -> dict:
    path = REPORTS_DIR / "uci_drift_benchmark.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


@st.cache_data
def load_freddie_drift_results() -> dict:
    path = REPORTS_DIR / "freddie_drift_replay.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


@st.cache_data
def load_freddie_strategy_results() -> dict:
    path = REPORTS_DIR / "freddie_strategy_comparison.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


drift_data = load_drift_results()
strategy_data = load_strategy_results()
uci_data = load_uci_results()
freddie_drift_data = load_freddie_drift_results()
freddie_strategy_data = load_freddie_strategy_results()

has_sba = bool(drift_data and strategy_data)
has_uci = bool(uci_data)
has_freddie = bool(freddie_drift_data and freddie_strategy_data)

if not has_sba and not has_uci and not has_freddie:
    st.error(
        "No results found. Run the pipeline first:\n\n"
        "```\npython -m drift.run_pipeline\n```"
    )
    st.stop()

if has_sba:
    periods = drift_data["periods"]
    strats = {s["name"]: s for s in strategy_data["strategies"]}
else:
    periods = []
    strats = {}

uci_periods = uci_data["periods"] if has_uci else []


# ── Plotly theme ───────────────────────────────────────────────────────────

PLOTLY_LAYOUT = dict(
    font=dict(family="Inter, -apple-system, sans-serif", size=13, color="#334155"),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=60, r=24, t=48, b=56),
    title_font=dict(size=15, color="#0f172a", family="Inter, sans-serif"),
    title_x=0.0,
    title_xanchor="left",
    legend=dict(
        font=dict(size=12, color="#475569"),
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="#e2e8f0",
        borderwidth=1,
    ),
    xaxis=dict(
        gridcolor="#f1f5f9",
        linecolor="#cbd5e1",
        linewidth=1,
        tickfont=dict(size=11, color="#64748b"),
        title_font=dict(size=12, color="#475569"),
    ),
    yaxis=dict(
        gridcolor="#f1f5f9",
        linecolor="#cbd5e1",
        linewidth=1,
        tickfont=dict(size=11, color="#64748b"),
        title_font=dict(size=12, color="#475569"),
    ),
    hoverlabel=dict(
        bgcolor="white",
        bordercolor="#e2e8f0",
        font=dict(size=12, color="#0f172a", family="Inter, sans-serif"),
    ),
)


# ── Helpers ─────────────────────────────────────────────────────────────────


def kpi_card(label: str, value: str, delta: str = "", delta_color: str = "neutral", accent: str = ""):
    accent_class = f"accent-{accent}" if accent else ""
    delta_html = ""
    if delta:
        delta_html = f'<div class="kpi-delta {delta_color}">{delta}</div>'
    html = (
        f'<div class="kpi-card {accent_class}">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        f'{delta_html}'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def section_header(title: str, icon: str = "", caption: str = ""):
    icon_html = f'<div class="section-icon">{icon}</div>' if icon else ""
    st.markdown(
        f'<div class="section-header">{icon_html}<h3>{title}</h3></div>',
        unsafe_allow_html=True,
    )
    if caption:
        st.markdown(f'<div class="section-caption">{caption}</div>', unsafe_allow_html=True)


def make_plotly_chart(fig: go.Figure, height: int = 450, title: str = ""):
    fig.update_layout(**PLOTLY_LAYOUT, height=height)
    if title:
        fig.update_layout(title_text=title)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


strat_labels = {
    "never": "Never Retrain",
    "scheduled": "Scheduled (Yearly)",
    "drift_triggered": "Triggered by Drift",
}

strat_colors = {
    "never": CHART_PALETTE["never_retrain"],
    "scheduled": CHART_PALETTE["scheduled"],
    "drift_triggered": CHART_PALETTE["drift_triggered"],
}


def build_heatmap_data(period_list):
    all_feature_names = set()
    for p in period_list:
        for d in p.get("top_drifting", []):
            all_feature_names.add(d["name"])
    feature_names = sorted(all_feature_names)
    if not feature_names:
        return None, None, None
    heatmap_z = []
    for feat in feature_names:
        row = []
        for p in period_list:
            val = 0
            for d in p.get("top_drifting", []):
                if d["name"] == feat:
                    val = d["psi"]
                    break
            row.append(val)
        heatmap_z.append(row)
    return heatmap_z, feature_names, period_list


def render_heatmap(heatmap_z, feature_names, period_list, x_label_fn, x_title, title):
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_z,
        x=[x_label_fn(p) for p in period_list],
        y=[readable_name(f) for f in feature_names],
        colorscale=[
            [0, "#f0f9ff"],
            [0.05, "#e0f2fe"],
            [0.10, "#bae6fd"],
            [0.25, "#fde68a"],
            [0.50, "#fb923c"],
            [1.0, "#dc2626"],
        ],
        hovertemplate="Feature: %{y}<br>Period: %{x}<br>PSI: %{z:.3f}<extra></extra>",
        colorbar=dict(
            title=dict(text="PSI", font=dict(size=12)),
            thickness=14,
            len=0.9,
            tickfont=dict(size=10),
        ),
    ))
    fig.update_layout(
        height=max(300, len(feature_names) * 38),
        margin=dict(l=180, r=20, t=48, b=56),
        xaxis_title=x_title,
    )
    make_plotly_chart(fig, title=title)


def render_alert_timeline(period_list, x_title="Fiscal Year"):
    years = [p["period"] for p in period_list]
    alerts = [p["overall_alert"] for p in period_list]
    fig = go.Figure()
    for alert_level in ["green", "amber", "red"]:
        mask_x = [y for y, a in zip(years, alerts, strict=False) if a == alert_level]
        mask_y = [1] * len(mask_x)
        fig.add_trace(go.Bar(
            x=mask_x, y=mask_y,
            name=alert_level.title(),
            marker_color=alert_color(alert_level),
            marker_line=dict(width=0),
            hovertemplate="%{x}: " + alert_level.title() + "<extra></extra>",
        ))
    fig.update_layout(
        barmode="stack",
        showlegend=True,
        yaxis=dict(visible=False),
        xaxis_title=x_title,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )
    make_plotly_chart(fig, 190, title="Alert Timeline")


def render_strategy_auc(strat_dict, x_title="Fiscal Year"):
    fig = go.Figure()
    for sname in ["never", "scheduled", "drift_triggered"]:
        s = strat_dict[sname]
        per_years = [p["period"] for p in s["periods"]]
        per_aucs = [p["auc"] for p in s["periods"]]
        fig.add_trace(go.Scatter(
            x=per_years,
            y=per_aucs,
            mode="lines+markers",
            name=strat_labels[sname],
            line=dict(color=strat_colors[sname], width=2.5),
            marker=dict(size=6),
            hovertemplate=f"{strat_labels[sname]}<br>%{{x}}<br>AUC: %{{y:.4f}}<extra></extra>",
        ))
    fig.update_layout(
        xaxis_title=x_title,
        yaxis_title="AUC-ROC",
        yaxis_range=[0.5, 1.0],
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    add_event_annotations(fig)
    make_plotly_chart(fig, 440, title="Model AUC by Retraining Strategy")


def render_retrain_events(strat_dict):
    for sname in ["scheduled", "drift_triggered"]:
        s = strat_dict[sname]
        events = s.get("retrain_events", [])
        if not events:
            continue
        promoted = sum(1 for e in events if e["promoted"])
        with st.expander(f"**{strat_labels[sname]}** : {len(events)} attempts, {promoted} promoted", expanded=False):
            event_rows = []
            for e in events:
                status = "Promoted" if e["promoted"] else "Kept champion"
                event_rows.append({
                    "Period": e.get("period", ""),
                    "Training": f"{e['training_years'][0]} to {e['training_years'][-1]}",
                    "Compare": e.get("comparison_year", ""),
                    "Champion AUC": f"{e['champion_auc']:.4f}",
                    "Challenger AUC": f"{e['challenger_auc']:.4f}",
                    "AUC Diff": f"{e['auc_diff']:+.4f}",
                    "95% CI": f"[{e['ci_lo']:+.4f}, {e['ci_hi']:+.4f}]",
                    "Result": status,
                })
            st.dataframe(event_rows, use_container_width=True, hide_index=True)


def render_comparison_table(comparison, label=""):
    comp_rows = []
    for sname in ["never", "scheduled", "drift_triggered"]:
        c = comparison[sname]
        comp_rows.append({
            "Strategy": strat_labels.get(sname, sname),
            "Retrains": c["n_retrains"],
            "Attempts": c["n_attempts"],
            "Mean Mon. AUC": f"{c['mean_monitoring_auc']:.4f}",
            "Mean Final AUC": f"{c['final_mean_auc']:.4f}",
        })
    st.dataframe(comp_rows, use_container_width=True, hide_index=True)


def render_final_eval(strat_dict):
    for sname in ["never", "scheduled", "drift_triggered"]:
        s = strat_dict[sname]
        with st.expander(f"**{strat_labels.get(sname, sname)}**", expanded=sname == "drift_triggered"):
            eval_rows = []
            for p in s.get("final_eval", []):
                eval_rows.append({
                    "Period": p.get("period", ""),
                    "Loans": f"{p['n_samples']:,}",
                    "Default Rate": f"{p['default_rate_pct']:.1f}%",
                    "AUC": f"{p['auc']:.4f}" if p["auc"] else "N/A",
                    "Champion": p["champion"],
                })
            st.dataframe(eval_rows, use_container_width=True, hide_index=True)


# ── Sidebar ────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## Credit Risk Drift Monitor")
    st.markdown("---")

    datasets_available = []
    if has_sba:
        datasets_available.append("SBA 7(a)")
    if has_freddie:
        datasets_available.append("Freddie Mac")
    if has_uci:
        datasets_available.append("UCI Credit Card")

    st.markdown(f"**Datasets loaded:** {len(datasets_available)}")
    for ds in datasets_available:
        st.markdown(f"- {ds}")

    st.markdown("---")
    st.markdown("**Alert Thresholds**")
    st.markdown(
        f"- Green: PSI < {ALERT_GREEN_SCORE_PSI}\n"
        f"- Amber: {ALERT_GREEN_SCORE_PSI} to {ALERT_AMBER_SCORE_PSI}\n"
        f"- Red: PSI > {ALERT_AMBER_SCORE_PSI}"
    )



# ── Tabs ────────────────────────────────────────────────────────────────────

tab_names = ["Overview", "Drift Analysis", "Model Performance", "Strategy Comparison"]
if has_freddie:
    tab_names.append("Freddie Mac")
if has_uci:
    tab_names.append("UCI Benchmark")
tab_names.append("About")

tabs = st.tabs(tab_names)
tab_idx = 0

# ============================================================================
# TAB 1 — Overview
# ============================================================================
with tabs[tab_idx]:
    tab_idx += 1
    if has_sba:
        st.markdown(
            '<div class="hero-header">'
            '<div class="hero-title">SBA 7(a) Loan Portfolio Drift Monitoring</div>'
            '<div class="hero-subtitle">Monitoring FY1996 to FY2018 against FY1991 to 1995 baseline · 1.5M loans · 31 year span</div>'
            "</div>",
            unsafe_allow_html=True,
        )

        n_periods = len(periods)
        n_red = sum(1 for p in periods if p["overall_alert"] == "red")
        n_amber = sum(1 for p in periods if p["overall_alert"] == "amber")
        n_green = sum(1 for p in periods if p["overall_alert"] == "green")
        total_loans = sum(p["n_samples"] for p in periods)
        avg_def_rate = np.mean([p["default_rate_pct"] for p in periods])
        baseline_auc = periods[0]["auc"] if periods else 0
        latest_auc = periods[-1]["auc"] if periods else 0

        c1, c2, c3 = st.columns(3)
        with c1:
            kpi_card("Periods", str(n_periods))
        with c2:
            kpi_card("Red Alerts", str(n_red), f"{n_red}/{n_periods}", "negative", accent="red")
        with c3:
            kpi_card("Loans Scored", f"{total_loans:,}")

        c4, c5, c6 = st.columns(3)
        with c4:
            kpi_card("Avg Default", f"{avg_def_rate:.1f}%", "", "neutral", accent="amber")
        with c5:
            auc_drop = latest_auc - baseline_auc if baseline_auc and latest_auc else 0
            d_color = "positive" if auc_drop >= 0 else "negative"
            kpi_card("Baseline AUC", f"{baseline_auc:.3f}", "", "neutral")
        with c6:
            kpi_card("Latest AUC", f"{latest_auc:.3f}", f"{auc_drop:+.3f}", d_color)

        st.markdown("<br>", unsafe_allow_html=True)

        col_left, col_right = st.columns([2, 1])

        with col_left:
            render_alert_timeline(periods)

        with col_right:
            st.markdown(
                '<div class="summary-card">'
                "<h4 style='margin:0 0 12px 0; font-size:1rem; color:#0f172a;'>Alert Summary</h4>"
                f"<div style='display:flex; gap:8px; flex-wrap:wrap;'>"
                f'<span class="alert-badge alert-green">Green: {n_green}</span>'
                f'<span class="alert-badge alert-amber">Amber: {n_amber}</span>'
                f'<span class="alert-badge alert-red">Red: {n_red}</span>'
                "</div>"
                "<div style='margin-top:16px; font-size:0.85rem; color:#64748b;'>"
                "All monitoring periods show red drift alerts, driven by the 2007 to 2009 "
                "crisis and by SBA starting to record interest rates in FY2009, a field "
                "that was empty for every loan before then."
                "</div>"
                "</div>",
                unsafe_allow_html=True,
            )

        section_header("Default Rate Over Time", "", "Percentage of loans charged off within 5 years, by fiscal year of approval")

        fig_def = go.Figure()
        fig_def.add_trace(go.Bar(
            x=[p["period"] for p in periods],
            y=[p["default_rate_pct"] for p in periods],
            marker_color=BRAND,
            marker_line=dict(width=0),
            hovertemplate="FY%{x}<br>Default rate: %{y:.1f}%<br>Loans: %{customdata:,}<extra></extra>",
            customdata=[p["n_samples"] for p in periods],
        ))
        fig_def.update_layout(
            xaxis_title="Fiscal Year",
            yaxis_title="Default Rate (%)",
        )
        add_event_annotations(fig_def)
        make_plotly_chart(fig_def, 360, title="Default Rate by Fiscal Year")
    else:
        st.info("SBA data not yet processed. Run the pipeline to see Overview results.")

# ============================================================================
# TAB 2 — Drift Analysis
# ============================================================================
with tabs[tab_idx]:
    tab_idx += 1
    if has_sba:
        st.markdown(
            '<div class="hero-header">'
            '<div class="hero-title">Drift Analysis</div>'
            '<div class="hero-subtitle">PSI and statistical tests comparing each monitoring period to the FY1991 to 1995 reference distribution</div>'
            "</div>",
            unsafe_allow_html=True,
        )

        section_header("Score PSI Over Time", "", "Population Stability Index of model score distribution vs reference")

        years = [p["period"] for p in periods]
        score_psis = [p["score_psi"] for p in periods]
        fig_psi = go.Figure()
        fig_psi.add_trace(go.Scatter(
            x=years,
            y=score_psis,
            mode="lines+markers",
            name="Score PSI",
            line=dict(color=BRAND, width=2.5),
            marker=dict(size=7, line=dict(width=1, color="white")),
            hovertemplate="FY%{x}<br>PSI: %{y:.4f}<extra></extra>",
            fill="tozeroy",
            fillcolor="rgba(30,64,175,0.06)",
        ))
        max_psi = max(score_psis) * 1.1
        add_alert_bands(fig_psi, ALERT_GREEN_SCORE_PSI, ALERT_AMBER_SCORE_PSI, max_psi)
        fig_psi.update_layout(
            xaxis_title="Fiscal Year",
            yaxis_title="Population Stability Index",
        )
        add_event_annotations(fig_psi)
        make_plotly_chart(fig_psi, 400, title="Score PSI with Alert Bands")

        section_header("Top Feature Drift by Period", "", "PSI values for the top 5 drifting features, taken from drift replay")

        heatmap_z, feature_names, _ = build_heatmap_data(periods)
        if heatmap_z:
            render_heatmap(
                heatmap_z, feature_names, periods,
                lambda p: f"FY{p['period']}", "Fiscal Year",
                "Feature PSI Heatmap",
            )

        section_header("Baseline Model AUC Over Time", "", "AUC degradation of the original FY1991 to 1995 baseline model")

        aucs = [p["auc"] for p in periods]
        fig_auc_base = go.Figure()
        fig_auc_base.add_trace(go.Scatter(
            x=years,
            y=aucs,
            mode="lines+markers",
            name="Baseline GBM",
            line=dict(color=CHART_PALETTE["red"], width=2.5),
            marker=dict(size=7, line=dict(width=1, color="white")),
            hovertemplate="FY%{x}<br>AUC: %{y:.4f}<extra></extra>",
        ))
        fig_auc_base.add_hline(
            y=0.9440,
            line=dict(color=CHART_PALETTE["reference"], width=1.5, dash="dash"),
            annotation_text="Reference AUC (0.944)",
            annotation_position="top left",
        )
        fig_auc_base.update_layout(
            xaxis_title="Fiscal Year",
            yaxis_title="AUC-ROC",
            yaxis_range=[0.5, 1.0],
        )
        add_event_annotations(fig_auc_base)
        make_plotly_chart(fig_auc_base, 400, title="Baseline Model AUC Degradation")
    else:
        st.info("SBA data not yet processed. Run the pipeline to see Drift Analysis.")

# ============================================================================
# TAB 3 — Model Performance
# ============================================================================
with tabs[tab_idx]:
    tab_idx += 1
    if has_sba:
        st.markdown(
            '<div class="hero-header">'
            '<div class="hero-title">Model Performance by Strategy</div>'
            '<div class="hero-subtitle">AUC over time under three retraining strategies: never retrain, scheduled yearly, and triggered by drift alerts</div>'
            "</div>",
            unsafe_allow_html=True,
        )

        render_strategy_auc(strats, x_title="Fiscal Year")

        section_header("Retrain Events", "", "Champion/challenger comparison with bootstrap confidence intervals")
        render_retrain_events(strats)
    else:
        st.info("SBA data not yet processed. Run the pipeline to see Model Performance.")

# ============================================================================
# TAB 4 — Strategy Comparison
# ============================================================================
with tabs[tab_idx]:
    tab_idx += 1
    if has_sba:
        st.markdown(
            '<div class="hero-header">'
            '<div class="hero-title">Strategy Comparison</div>'
            '<div class="hero-subtitle">Comparing retraining strategies across monitoring and final evaluation periods</div>'
            "</div>",
            unsafe_allow_html=True,
        )

        section_header("Summary Metrics")
        comparison = strategy_data.get("comparison", {})
        render_comparison_table(comparison)

        st.markdown("<br>", unsafe_allow_html=True)

        section_header("Final Evaluation (FY2019 to 2021)", "", "Scored exactly once with each strategy's final champion model, never used to choose anything")
        render_final_eval(strats)

        st.markdown("<br>", unsafe_allow_html=True)

        section_header("AUC Lift from Retraining", "", "Difference in AUC between the scheduled retrained model and the original at each monitoring period")

        never_aucs = {p["period"]: p["auc"] for p in strats["never"]["periods"]}
        sched_aucs = {p["period"]: p["auc"] for p in strats["scheduled"]["periods"]}
        lift_years = sorted(never_aucs.keys())
        lift_vals = []
        for yr in lift_years:
            na = never_aucs.get(yr)
            sa = sched_aucs.get(yr)
            if na is not None and sa is not None:
                lift_vals.append(sa - na)
            else:
                lift_vals.append(0)

        fig_lift = go.Figure()
        fig_lift.add_trace(go.Bar(
            x=lift_years,
            y=lift_vals,
            marker_color=[
                CHART_PALETTE["scheduled"] if v > 0 else CHART_PALETTE["red"]
                for v in lift_vals
            ],
            marker_line=dict(width=0),
            hovertemplate="FY%{x}<br>AUC lift: %{y:+.4f}<extra></extra>",
        ))
        fig_lift.add_hline(y=0, line=dict(color=INK, width=1))
        fig_lift.update_layout(
            xaxis_title="Fiscal Year",
            yaxis_title="AUC Lift (Scheduled vs Never)",
        )
        make_plotly_chart(fig_lift, 350, title="AUC Lift: Scheduled vs Never Retrain")
    else:
        st.info("SBA data not yet processed. Run the pipeline to see Strategy Comparison.")

# ============================================================================
# TAB 5 — Freddie Mac (conditional)
# ============================================================================
if has_freddie:
    with tabs[tab_idx]:
        tab_idx += 1

        fm_periods = freddie_drift_data["periods"]
        fm_strats = {s["name"]: s for s in freddie_strategy_data["strategies"]}
        fm_comparison = freddie_strategy_data.get("comparison", {})

        st.markdown(
            '<div class="hero-header">'
            '<div class="hero-title">Freddie Mac SFLLD Mortgage Drift Monitoring</div>'
            '<div class="hero-subtitle">'
            f"Monitoring {fm_periods[0]['period']} to {fm_periods[-1]['period']} · "
            "24-month outcome window · 90+ DPD default · 1.25M sample loans"
            "</div></div>",
            unsafe_allow_html=True,
        )

        # KPI row
        fm_n_periods = len(fm_periods)
        fm_n_green = sum(1 for p in fm_periods if p["overall_alert"] == "green")
        fm_n_amber = sum(1 for p in fm_periods if p["overall_alert"] == "amber")
        fm_n_red = sum(1 for p in fm_periods if p["overall_alert"] == "red")
        fm_total_loans = sum(p["n_samples"] for p in fm_periods)
        fm_avg_def = np.mean([p["default_rate_pct"] for p in fm_periods])
        fm_base_auc = fm_periods[0]["auc"] if fm_periods else 0
        fm_worst_auc = min(p["auc"] for p in fm_periods if p["auc"] is not None)

        c1, c2, c3 = st.columns(3)
        with c1:
            kpi_card("Periods", str(fm_n_periods))
        with c2:
            kpi_card(
                "Alerts",
                f"{fm_n_green}G / {fm_n_amber}A / {fm_n_red}R",
                f"{fm_n_red} red alerts", "negative" if fm_n_red > fm_n_green else "neutral",
            )
        with c3:
            kpi_card("Loans Scored", f"{fm_total_loans:,}")

        c4, c5, c6 = st.columns(3)
        with c4:
            kpi_card("Avg Default", f"{fm_avg_def:.1f}%", "", "neutral", accent="amber")
        with c5:
            kpi_card("Baseline AUC", f"{fm_base_auc:.4f}", "", "neutral")
        with c6:
            kpi_card("Worst AUC", f"{fm_worst_auc:.4f}", f"{fm_worst_auc - fm_base_auc:+.4f}", "negative", accent="red")

        st.markdown("<br>", unsafe_allow_html=True)

        col_left, col_right = st.columns(2)

        with col_left:
            render_alert_timeline(fm_periods, x_title="Origination Year")

        with col_right:
            section_header("Default Rate Over Time")
            fig_fm_def = go.Figure()
            fig_fm_def.add_trace(go.Bar(
                x=[p["period"] for p in fm_periods],
                y=[p["default_rate_pct"] for p in fm_periods],
                marker_color=BRAND,
                marker_line=dict(width=0),
                hovertemplate="%{x}<br>Default rate: %{y:.1f}%<br>Loans: %{customdata:,}<extra></extra>",
                customdata=[p["n_samples"] for p in fm_periods],
            ))
            fig_fm_def.update_layout(
                xaxis_title="Origination Year",
                yaxis_title="Default Rate (%)",
            )
            make_plotly_chart(fig_fm_def, 230, title="Default Rate by Origination Year")

        section_header("Score PSI Over Time", "", "Population Stability Index of model scores vs reference distribution")

        fm_years = [p["period"] for p in fm_periods]
        fm_score_psis = [p["score_psi"] for p in fm_periods]
        fig_fm_psi = go.Figure()
        fig_fm_psi.add_trace(go.Scatter(
            x=fm_years,
            y=fm_score_psis,
            mode="lines+markers",
            name="Score PSI",
            line=dict(color=BRAND, width=2.5),
            marker=dict(size=7, line=dict(width=1, color="white")),
            hovertemplate="%{x}<br>PSI: %{y:.4f}<extra></extra>",
            fill="tozeroy",
            fillcolor="rgba(30,64,175,0.06)",
        ))
        max_fm_psi = max(fm_score_psis) * 1.1
        add_alert_bands(fig_fm_psi, ALERT_GREEN_SCORE_PSI, ALERT_AMBER_SCORE_PSI, max_fm_psi)
        fig_fm_psi.update_layout(
            xaxis_title="Origination Year",
            yaxis_title="Population Stability Index",
        )
        add_event_annotations(fig_fm_psi)
        make_plotly_chart(fig_fm_psi, 400, title="Score PSI with Alert Bands")

        section_header("Top Feature Drift by Period")
        fm_z, fm_feats, _ = build_heatmap_data(fm_periods)
        if fm_z:
            render_heatmap(
                fm_z, fm_feats, fm_periods,
                lambda p: str(p["period"]), "Origination Year",
                "Feature PSI Heatmap",
            )

        render_strategy_auc(fm_strats, x_title="Origination Year")

        section_header("Strategy Comparison")
        render_comparison_table(fm_comparison)

        st.markdown("<br>", unsafe_allow_html=True)

        section_header("Retrain Events")
        render_retrain_events(fm_strats)

        section_header("Final Evaluation", "", "Scored once with each strategy's final champion model")
        render_final_eval(fm_strats)

# ============================================================================
# TAB 6 — UCI Benchmark (conditional)
# ============================================================================
if has_uci:
    with tabs[tab_idx]:
        tab_idx += 1

        st.markdown(
            '<div class="hero-header">'
            '<div class="hero-title">UCI Credit Card Synthetic Drift Benchmark</div>'
            '<div class="hero-subtitle">'
            "30,000 Taiwan credit card clients · 40% reference split · "
            "20 simulated periods with progressive feature shifts"
            "</div></div>",
            unsafe_allow_html=True,
        )

        n_uci = uci_data["n_reference"] + uci_data["n_pool"]
        base_auc = uci_data["baseline_auc"]
        cv_auc = uci_data["baseline_cv_auc"]
        worst_auc = min(p["auc"] for p in uci_periods if p["auc"] is not None)
        final_auc = uci_periods[-1]["auc"] if uci_periods else 0

        c1, c2, c3 = st.columns(3)
        with c1:
            kpi_card("Total Clients", f"{n_uci:,}")
        with c2:
            kpi_card("Baseline AUC", f"{cv_auc:.4f}", "Cross-validated", "neutral", accent="green")
        with c3:
            kpi_card("Worst AUC", f"{worst_auc:.4f}", f"{worst_auc - base_auc:+.4f}", "negative", accent="red")

        c4, c5, c6 = st.columns(3)
        with c4:
            kpi_card("Recovery AUC", f"{final_auc:.4f}", f"{final_auc - worst_auc:+.4f}", "positive", accent="green")
        with c5:
            n_uci_red = sum(1 for p in uci_periods if p["overall_alert"] == "red")
            kpi_card("Red Alerts", str(n_uci_red), f"{n_uci_red}/{len(uci_periods)}", "negative")
        with c6:
            kpi_card("Drift Phases", "4", "Simulated progression", "neutral")

        st.markdown("<br>", unsafe_allow_html=True)

        section_header("Score PSI Across Drift Phases", "", "Simulated drift progresses through mild, moderate, severe, and recovery phases")

        uci_ids = [p["period"] for p in uci_periods]
        uci_psis = [p["score_psi"] for p in uci_periods]
        uci_phases = [p["phase"] for p in uci_periods]

        phase_colors = {
            "mild": "#16a34a",
            "moderate": "#f59e0b",
            "severe": "#dc2626",
            "recovery": "#0ea5e9",
        }

        fig_uci_psi = go.Figure()
        for phase in ["mild", "moderate", "severe", "recovery"]:
            mask_x = [x for x, ph in zip(uci_ids, uci_phases, strict=False) if ph == phase]
            mask_y = [y for y, ph in zip(uci_psis, uci_phases, strict=False) if ph == phase]
            fig_uci_psi.add_trace(go.Bar(
                x=mask_x,
                y=mask_y,
                name=phase.title(),
                marker_color=phase_colors[phase],
                marker_line=dict(width=0),
                hovertemplate=f"{phase.title()}<br>Period %{{x}}<br>PSI: %{{y:.4f}}<extra></extra>",
            ))

        max_uci_psi = max(uci_psis) * 1.1
        add_alert_bands(fig_uci_psi, ALERT_GREEN_SCORE_PSI, ALERT_AMBER_SCORE_PSI, max_uci_psi)
        fig_uci_psi.update_layout(
            barmode="group",
            xaxis_title="Simulated Period",
            yaxis_title="Population Stability Index",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        make_plotly_chart(fig_uci_psi, 400, title="Score PSI by Drift Phase")

        section_header("Model AUC Degradation and Recovery")

        uci_aucs = [p["auc"] for p in uci_periods]
        fig_uci_auc = go.Figure()
        fig_uci_auc.add_trace(go.Scatter(
            x=uci_ids,
            y=uci_aucs,
            mode="lines+markers",
            name="Baseline GBM",
            line=dict(color=BRAND, width=2.5),
            marker=dict(
                size=9,
                color=[phase_colors.get(ph, BRAND) for ph in uci_phases],
                line=dict(width=1.5, color="white"),
            ),
            hovertemplate="Period %{x}<br>AUC: %{y:.4f}<extra></extra>",
        ))
        fig_uci_auc.add_hline(
            y=base_auc,
            line=dict(color=CHART_PALETTE["reference"], width=1.5, dash="dash"),
            annotation_text=f"Reference AUC ({base_auc:.3f})",
            annotation_position="top left",
        )
        fig_uci_auc.update_layout(
            xaxis_title="Simulated Period",
            yaxis_title="AUC-ROC",
            yaxis_range=[0.4, 0.9],
        )
        fig_uci_auc.add_vrect(x0=0.5, x1=5.5, fillcolor="#16a34a", opacity=0.04,
                              annotation_text="Mild", annotation_position="top left",
                              annotation_font=dict(size=11, color="#16a34a"))
        fig_uci_auc.add_vrect(x0=5.5, x1=10.5, fillcolor="#f59e0b", opacity=0.04,
                              annotation_text="Moderate", annotation_position="top left",
                              annotation_font=dict(size=11, color="#d97706"))
        fig_uci_auc.add_vrect(x0=10.5, x1=15.5, fillcolor="#dc2626", opacity=0.04,
                              annotation_text="Severe", annotation_position="top left",
                              annotation_font=dict(size=11, color="#dc2626"))
        fig_uci_auc.add_vrect(x0=15.5, x1=20.5, fillcolor="#0ea5e9", opacity=0.04,
                              annotation_text="Recovery", annotation_position="top left",
                              annotation_font=dict(size=11, color="#0ea5e9"))
        make_plotly_chart(fig_uci_auc, 400, title="AUC Across Drift Phases")

        section_header("Top Feature Drift by Period")
        uci_z, uci_feats, _ = build_heatmap_data(uci_periods)
        if uci_z:
            render_heatmap(
                uci_z, uci_feats, uci_periods,
                lambda p: f"P{p['period']}", "Simulated Period",
                "Feature PSI Heatmap",
            )

        section_header("Period Detail")
        detail_rows = []
        for p in uci_periods:
            detail_rows.append({
                "Period": p["period"],
                "Phase": p["phase"].title(),
                "Drift Intensity": f"{p['drift_intensity']:.2f}",
                "Samples": p["n_samples"],
                "Default Rate": f"{p['default_rate_pct']:.1f}%",
                "Score PSI": f"{p['score_psi']:.4f}",
                "AUC": f"{p['auc']:.4f}" if p["auc"] else "N/A",
                "Alert": p["overall_alert"].upper(),
            })
        st.dataframe(detail_rows, use_container_width=True, hide_index=True)

# ============================================================================
# TAB — About
# ============================================================================
with tabs[tab_idx]:
    st.markdown(
        '<div class="hero-header">'
        '<div class="hero-title">About This Dashboard</div>'
        '<div class="hero-subtitle">Methodology, design decisions, and dataset documentation</div>'
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown('<div class="about-block">', unsafe_allow_html=True)
    st.markdown("""#### Methodology

We trained a probability of default model on the earliest years of real lending
data, then replayed it through decades of new loans to see how it holds up over
time. The pipeline works in four stages.

First, raw loan records are loaded and default labels are derived using time aware
outcome windows so that only loans with a full observation period count. Second,
numeric and categorical features are engineered and baseline models (Logistic
Regression and Gradient Boosting) are trained with time ordered cross validation.
Third, the baseline is scored against each subsequent year, and drift is measured
with PSI, KS, and chi square tests. Each period gets a green, amber, or red alert
based on how far the data has shifted. Fourth, three retraining strategies are
compared using champion/challenger testing backed by bootstrap confidence intervals.
""")
    st.markdown("</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="about-block">', unsafe_allow_html=True)
        st.markdown("""#### Retraining Strategies

| Strategy | Rule |
|----------|------|
| **Never Retrain** | Keep the original baseline model |
| **Scheduled** | Retrain yearly using a 3 year rolling window |
| **Triggered by Drift** | Retrain only when the drift alert is red |

A challenger model must beat the sitting champion by at least 0.005 AUC on a
comparison year it was not trained on, and that edge is confirmed with 200
bootstrap iterations.
""")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="about-block">', unsafe_allow_html=True)
        st.markdown("""#### Design Principles

The model only sees data that would have been available at each point in time,
with no future information leaking in. Final evaluation years are scored once
and were never used to pick models or tune anything. Seeds, thresholds, and
splits are all fixed so results are fully reproducible. Raw data stays out of
the repository; only code and aggregated outputs are committed.

Alert thresholds for Score PSI:

| Level | PSI Range |
|-------|-----------|
| Green | Below 0.10 |
| Amber | 0.10 to 0.25 |
| Red | Above 0.25 |
""")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="about-block">', unsafe_allow_html=True)
    st.markdown("""#### Datasets

| Dataset | Records | Time Span | Outcome Window | Use |
|---------|---------|-----------|----------------|-----|
| **SBA 7(a) FOIA** | ~1.5M | FY1991 to 2021 | 5 years | Primary: real world drift replay |
| **Freddie Mac SFLLD** | 1.25M | 1999 to 2023 | 24 months | Mortgage drift replay |
| **UCI Credit Card** | 30,000 | Single snapshot | N/A | Benchmark: synthetic drift test |
""")
    st.markdown("</div>", unsafe_allow_html=True)

    if has_freddie:
        st.markdown('<div class="about-block">', unsafe_allow_html=True)
        st.markdown("""#### Freddie Mac Findings

The Freddie Mac dataset covers 25 years of mortgage originations (1999 to 2023),
with 50,000 sample loans per vintage year. A loan counts as defaulted if it goes
90 or more days past due, or if it exits with a credit related zero balance code,
within 24 months of origination.

Default rates reached 6.2% for 2007 vintages during the financial crisis and 4.6%
for 2019 vintages. The drift pattern here is more varied than SBA, which flagged
red on every period. Freddie Mac produced 1 green, 3 amber, and 13 red alerts
across its monitoring years. Retraining lifted the final evaluation AUC from 0.753
to 0.784.
""")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(
        '<div style="text-align:center; padding: 24px 0 8px 0;">'
        '<div style="font-size:1rem; font-weight:600; color:#0f172a; margin-bottom:8px;">'
        "Credit Risk Model Drift Monitoring</div>"
        '<div style="font-size:0.82rem; color:#64748b; margin-bottom:16px;">'
        "Data sources: SBA 7(a) FOIA, Freddie Mac SFLLD, UCI Credit Card</div>"
        "</div>",
        unsafe_allow_html=True,
    )
    footer_col1, footer_col2, footer_col3 = st.columns([1, 1, 1])
    with footer_col2:
        st.markdown(
            '[GitHub](https://github.com/vamshidharsingh) · '
            '[LinkedIn](https://www.linkedin.com/in/vamshidhars1/)'
        )
