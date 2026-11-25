"""Shared chart style — every chart in the dashboard and README uses this module."""

import plotly.graph_objects as go
import plotly.io as pio

from drift.config import (
    BRAND,
    CHART_PALETTE,
    COLORBLIND_SAFE_PALETTE,
    EVENT_ANNOTATIONS,
    FEATURE_READABLE_NAMES,
    INK,
    LINE,
    MUTED,
)

FONT_FAMILY = "Inter, -apple-system, BlinkMacSystemFont, sans-serif"

_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        font=dict(family=FONT_FAMILY, color=INK, size=13),
        title=dict(font=dict(size=18, color=INK), x=0, xanchor="left", pad=dict(l=0)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        colorway=COLORBLIND_SAFE_PALETTE,
        xaxis=dict(
            gridcolor=LINE,
            linecolor=LINE,
            tickfont=dict(size=11, color=MUTED),
            title=dict(font=dict(size=12, color=MUTED)),
        ),
        yaxis=dict(
            gridcolor=LINE,
            linecolor=LINE,
            tickfont=dict(size=11, color=MUTED),
            title=dict(font=dict(size=12, color=MUTED)),
        ),
        legend=dict(
            font=dict(size=11),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor=LINE,
            borderwidth=1,
        ),
        hoverlabel=dict(font=dict(size=12, family=FONT_FAMILY)),
        margin=dict(l=60, r=20, t=60, b=60),
    )
)

pio.templates["drift"] = _TEMPLATE
pio.templates.default = "drift"


def readable_name(col: str) -> str:
    """Return a human-readable name for a feature column."""
    return FEATURE_READABLE_NAMES.get(col, col.replace("_", " ").title())


def format_pct(value: float, decimals: int = 1) -> str:
    return f"{value:.{decimals}f}%"


def format_psi(value: float) -> str:
    return f"{value:.3f}"


def format_auc(value: float) -> str:
    return f"{value:.3f}"


def format_count(value: int) -> str:
    return f"{value:,}"


def format_money(value: float) -> str:
    if abs(value) >= 1e9:
        return f"${value / 1e9:.1f}B"
    if abs(value) >= 1e6:
        return f"${value / 1e6:.1f}M"
    if abs(value) >= 1e3:
        return f"${value / 1e3:.0f}K"
    return f"${value:,.0f}"


def alert_color(level: str) -> str:
    """Return the colour for an alert level."""
    return CHART_PALETTE.get(level, MUTED)


def strategy_color(strategy: str) -> str:
    """Return the colour for a retraining strategy."""
    return CHART_PALETTE.get(strategy, MUTED)


def add_event_annotations(
    fig: go.Figure,
    x_col_is_year: bool = True,
    y_frac: float = 0.95,
) -> go.Figure:
    """Add vertical lines and labels for major real-world events."""
    for year, label in EVENT_ANNOTATIONS.items():
        fig.add_vline(
            x=year,
            line=dict(color=MUTED, width=1, dash="dot"),
            annotation=dict(
                text=label,
                font=dict(size=10, color=MUTED),
                textangle=-90,
                yanchor="top",
                xanchor="left",
                xshift=3,
                bgcolor="rgba(255,255,255,0.85)",
                borderpad=2,
            ),
            annotation_position="top right",
        )
    return fig


def add_alert_bands(
    fig: go.Figure,
    green_threshold: float,
    amber_threshold: float,
    y_max: float | None = None,
) -> go.Figure:
    """Add coloured horizontal bands for alert levels (green, amber, red)."""
    if y_max is None:
        y_max = max(amber_threshold * 2, 0.5)

    fig.add_hrect(
        y0=0, y1=green_threshold,
        fillcolor=CHART_PALETTE["green"], opacity=0.08,
        line_width=0,
        annotation_text="Green", annotation_position="top left",
        annotation=dict(font=dict(size=10, color=CHART_PALETTE["green"])),
    )
    fig.add_hrect(
        y0=green_threshold, y1=amber_threshold,
        fillcolor=CHART_PALETTE["amber"], opacity=0.08,
        line_width=0,
        annotation_text="Amber", annotation_position="top left",
        annotation=dict(font=dict(size=10, color=CHART_PALETTE["amber"])),
    )
    fig.add_hrect(
        y0=amber_threshold, y1=y_max,
        fillcolor=CHART_PALETTE["red"], opacity=0.08,
        line_width=0,
        annotation_text="Red", annotation_position="top left",
        annotation=dict(font=dict(size=10, color=CHART_PALETTE["red"])),
    )
    return fig


def save_chart(fig: go.Figure, path: str, width: int = 1200, height: int = 500) -> None:
    """Save a Plotly figure as a static PNG image."""
    fig.write_image(path, width=width, height=height, scale=2)


def default_rate_by_period(
    periods: list,
    rates: list,
    counts: list,
    dataset_name: str,
    caption: str = "",
) -> go.Figure:
    """Bar chart of default rate per period with loan counts on hover."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=periods,
        y=rates,
        marker_color=BRAND,
        hovertemplate=(
            "Period: %{x}<br>"
            "Default rate: %{y:.1f}%<br>"
            "Loans: %{customdata:,}<extra></extra>"
        ),
        customdata=counts,
    ))
    fig.update_layout(
        title=f"Default Rate by Period — {dataset_name}",
        xaxis_title="Period",
        yaxis_title="Default Rate (%)",
    )
    if caption:
        fig.add_annotation(
            text=caption,
            xref="paper", yref="paper",
            x=0, y=-0.15,
            showarrow=False,
            font=dict(size=11, color=MUTED),
        )
    add_event_annotations(fig)
    return fig
