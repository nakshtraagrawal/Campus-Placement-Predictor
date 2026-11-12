"""Shared UI helpers for every Streamlit page: CSS, colours, cards, metric tiles, cached loaders.

Pages call `setup_page(title, subtitle)` first; it sets the page config, injects the CSS,
draws the sidebar and returns nothing. All numbers come from the pipeline outputs.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # make `src` importable when run via `streamlit run app/Home.py`
    sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.utils import load_json  # noqa: E402

# ---------------------------------------------------------------------------
# Colours (UI spec)
# ---------------------------------------------------------------------------
PRIMARY = "#4F46E5"
PLACED = config.COLOR_PLACED
NOT_PLACED = config.COLOR_NOT_PLACED
AMBER = "#F59E0B"
SLATE = "#64748B"
BORDER = "#E2E8F0"
PALETTE = config.MODEL_PALETTE
FONT = "Inter, -apple-system, Segoe UI, sans-serif"

# Plotly template: plotly_white + Inter, left-aligned titles, no chart junk
_tpl = go.layout.Template(pio.templates["plotly_white"])
_tpl.layout.font = dict(family=FONT, size=13, color="#0F172A")
_tpl.layout.title = dict(x=0, xanchor="left", font=dict(size=15))
_tpl.layout.colorway = PALETTE
_tpl.layout.margin = dict(l=10, r=10, t=40, b=10)
_tpl.layout.hoverlabel = dict(font_family=FONT)
pio.templates["placement"] = _tpl
pio.templates.default = "placement"

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"], .stMarkdown, .stText, button, input, textarea, select {{
    font-family: {FONT};
}}
h1, h2, h3, h4 {{ font-family: {FONT}; font-weight: 600 !important; letter-spacing: -0.02em; }}
#MainMenu, footer, [data-testid="stToolbar"] {{ visibility: hidden; height: 0; }}
[data-testid="stDecoration"] {{ display: none; }}
.block-container {{ padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1400px; }}
.subtitle {{ color: {SLATE}; font-size: 1.02rem; margin-top: -0.6rem; margin-bottom: 1.2rem; }}
.card {{
    background: #FFFFFF; border: 1px solid {BORDER}; border-radius: 12px;
    box-shadow: 0 1px 3px rgba(15,23,42,0.06), 0 1px 2px rgba(15,23,42,0.04);
    padding: 20px 22px; margin-bottom: 14px;
}}
.card h4 {{ margin: 0 0 6px 0; font-size: 1.02rem; }}
.card p {{ margin: 0; color: #334155; font-size: 0.93rem; }}
.tile {{
    background: #FFFFFF; border: 1px solid {BORDER}; border-radius: 12px; padding: 16px 18px;
    box-shadow: 0 1px 3px rgba(15,23,42,0.06); height: 100%;
}}
.tile .label {{ color: {SLATE}; font-size: 0.82rem; text-transform: uppercase; letter-spacing: 0.04em; }}
.tile .value {{ font-size: 1.75rem; font-weight: 700; color: #0F172A; line-height: 1.25; }}
.tile .delta {{ font-size: 0.85rem; color: {SLATE}; }}
.verdict {{ border-radius: 12px; padding: 18px 22px; color: white; margin-bottom: 12px; }}
.verdict .big {{ font-size: 1.45rem; font-weight: 700; }}
.verdict .small {{ font-size: 0.95rem; opacity: 0.95; }}
.step-num {{
    display: inline-block; width: 26px; height: 26px; border-radius: 50%; background: {PRIMARY};
    color: white; text-align: center; line-height: 26px; font-weight: 600; margin-right: 8px; font-size: 0.85rem;
}}
.badge {{
    display: inline-block; background: #EEF2FF; color: {PRIMARY}; border-radius: 999px;
    padding: 3px 10px; font-size: 0.78rem; font-weight: 600;
}}
.action {{ border-left: 4px solid {PLACED}; }}
.insight {{ border-left: 4px solid {PRIMARY}; padding: 12px 16px; }}
[data-testid="stMain"] [data-testid="stPageLink"] a {{
    border: 1px solid {PRIMARY}; border-radius: 8px; justify-content: center; background: #FFFFFF;
}}
[data-testid="stMain"] [data-testid="stPageLink"] a p {{ color: {PRIMARY}; font-weight: 600; }}
[data-testid="stSidebar"] .sidebar-title {{ font-weight: 700; font-size: 1.1rem; }}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
@st.cache_data
def load_metrics() -> dict:
    return load_json(config.METRICS_JSON)


@st.cache_data
def load_meta() -> dict:
    return load_json(config.MODEL_META_PATH)


@st.cache_data
def load_csv(name: str, **kw) -> pd.DataFrame:
    path = config.REPORTS_DIR / name
    return pd.read_csv(path, **kw) if path.exists() else pd.DataFrame()


@st.cache_data
def load_report_json(name: str) -> dict:
    return load_json(config.REPORTS_DIR / name)


@st.cache_data
def load_clean() -> pd.DataFrame:
    return pd.read_csv(config.CLEAN_CSV)


@st.cache_resource
def load_pipeline():
    from src.predict import load_model
    return load_model()


@st.cache_resource
def load_explainer():
    from src.predict import load_background, make_explainer
    pipe, meta = load_pipeline()
    return make_explainer(pipe, load_background(), meta.get("is_tree_model", False))


def outputs_ready() -> bool:
    return config.BEST_MODEL_PATH.exists() and config.METRICS_JSON.exists()


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------
def setup_page(title: str, subtitle: str) -> bool:
    """Page config + CSS + sidebar + header. Returns False if the pipeline hasn't been run."""
    st.set_page_config(page_title=f"{title} · Campus Placement Predictor", layout="wide")
    inject_css()
    sidebar()
    st.title(title)
    st.markdown(f'<div class="subtitle">{subtitle}</div>', unsafe_allow_html=True)
    if not outputs_ready():
        card("Model not found",
             "The trained model and metrics are missing. Run <code>python run_all.py</code> from the "
             "project folder first, then refresh this page.")
        return False
    return True


def sidebar() -> None:
    with st.sidebar:
        st.markdown('<div class="sidebar-title">Campus Placement Predictor</div>', unsafe_allow_html=True)
        st.caption("ML project · explainable placement prediction")
        if outputs_ready():
            meta = load_meta()
            st.markdown(f'<span class="badge">Model: {meta["model_name"]} · ROC-AUC '
                        f'{meta["test_roc_auc"]:.2f}</span>', unsafe_allow_html=True)


def card(title: str, body: str, extra_class: str = "") -> None:
    st.markdown(f'<div class="card {extra_class}"><h4>{title}</h4><p>{body}</p></div>', unsafe_allow_html=True)


def metric_tile(label: str, value: str, delta: str = "") -> None:
    d = f'<div class="delta">{delta}</div>' if delta else ""
    st.markdown(f'<div class="tile"><div class="label">{label}</div><div class="value">{value}</div>{d}</div>',
                unsafe_allow_html=True)


def tiles(items: list[tuple]) -> None:
    """Row of metric tiles: items = [(label, value, delta?), ...]."""
    for col, item in zip(st.columns(len(items)), items):
        with col:
            metric_tile(*item)


def show_figure(name: str, caption: str = "") -> None:
    path = config.FIG_DIR / name
    if path.exists():
        st.image(str(path), caption=caption, width="stretch")
    else:
        st.info(f"Figure {name} not found - run `python run_all.py`.")


def nice_feature(f: str) -> str:
    """'AptitudeScore' -> 'Aptitude score'."""
    import re
    words = re.sub(r"(?<!^)(?=[A-Z])", " ", f).split()
    if f == "CGPA":
        return "CGPA"
    return " ".join([words[0]] + [w.lower() for w in words[1:]])


def fmt_value(f: str, v: float) -> str:
    if f == "Attendance":
        return f"{v:.0f}%"
    if f in ("CGPA", "CommunicationSkills"):
        return f"{v:.1f}"
    return f"{v:.0f}"
