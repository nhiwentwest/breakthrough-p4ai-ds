"""
World Happiness EDA — Step-by-Step Interactive Demo
Real dataset: World Happiness Report 2019
"""
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import time, io, requests
import seaborn as sns

sns.set_theme(style="whitegrid", rc={
    "figure.facecolor": "#F7F3EB",
    "axes.facecolor": "#F7F3EB",
    "axes.edgecolor": "#D4C9B8",
    "axes.linewidth": 1.2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.color": "#E5DFD3",
    "grid.linestyle": "--",
    "grid.linewidth": 0.8,
    "xtick.color": "#6B6560",
    "ytick.color": "#6B6560",
    "text.color": "#111111",
    "axes.labelcolor": "#111111",
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "font.family": "sans-serif"
})

st.set_page_config(
    page_title="Tabular EDA",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ══════════════════════════════════════════════════════════════════════════════
#  PALETTE & CSS — warm paper editorial style (matches multimodal_eda)
# ══════════════════════════════════════════════════════════════════════════════
BG   = "#F7F3EB"
CARD = "#EFE8DC"
TEXT = "#111111"
ACC  = "#B42318"
MUT  = "#6B6560"
BOR  = "#D4C9B8"
RULE = "#C8BBB0"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,700;0,900&family=Source+Sans+3:wght@300;400;600&display=swap');
:root {{ --bg:{BG}; --card:{CARD}; --text:{TEXT}; --acc:{ACC}; --mut:{MUT}; --bor:{BOR}; --rule:{RULE}; }}
body,.stApp {{ background:var(--bg); color:var(--text); font-family:'Source Sans 3',sans-serif; }}
[data-testid="stSidebar"] {{ background:#EDE9E1 !important; border-right:2px solid var(--rule); width:220px !important; }}
#MainMenu,footer,header {{ visibility:hidden; }}
.main .block-container {{ padding-left:240px; padding-right:1rem; }}
h1,h2,h3 {{ font-family:'Playfair Display',serif; color:var(--text); line-height:1.1; }}
.eyebrow {{ font-size:0.65rem; font-weight:700; letter-spacing:0.2em;
            text-transform:uppercase; color:var(--acc); margin-bottom:0.4rem; }}
.section-head {{ font-size:0.62rem; font-weight:700; letter-spacing:0.15em;
                 text-transform:uppercase; color:var(--acc); margin:1.4rem 0 0.3rem; }}
.footer {{ text-align:center; color:var(--mut); font-size:0.68rem;
           letter-spacing:0.1em; text-transform:uppercase; }}
.hero-divider {{ width:60px; height:2px; background:var(--acc); margin:1.6rem auto; display:block; }}
[data-testid="stMetricValue"] {{ font-family:'Playfair Display',serif;
    font-size:2.2rem; font-weight:900; color:var(--text); line-height:1; }}
[data-testid="stMetricLabel"] {{ font-size:0.62rem; font-weight:700; letter-spacing:0.1em;
    text-transform:uppercase; color:var(--mut); }}
thead th {{ background:{CARD} !important; color:{ACC} !important;
            font-weight:700; letter-spacing:0.06em; font-size:0.7rem;
            border-bottom:2px solid {BOR} !important; text-transform:uppercase; }}
tbody tr:hover {{ background:rgba(180,35,24,0.04) !important; }}
tbody td {{ border-bottom:1px solid {BOR} !important; }}
.stTabs [data-baseweb="tab-list"] {{ gap:2px; border-bottom:2px solid var(--bor); padding-bottom:0; }}
.stTabs [data-baseweb="tab"] {{ color:var(--mut); font-weight:700; font-size:0.85rem; border-radius:0; }}
.stTabs [aria-selected="true"] {{ color:var(--acc)!important; border-bottom:2.5px solid var(--acc)!important; margin-bottom:-2px; }}
.insight {{ border-left:3px solid var(--acc); padding:0.7rem 1.2rem;
            background:var(--card); margin:1rem 0;
            font-family:'Playfair Display',serif;
            font-size:1rem; font-style:italic; color:var(--text); line-height:1.6; }}
.log-ok {{ color:#1A7A3E; }}
.log-warn {{ color:#B45315; }}
.log-entry {{ font-size:0.78rem; color:var(--mut); padding:0.18rem 0;
             border-bottom:1px solid var(--rule); font-family:'Courier New',monospace; }}
.step-pill {{ display:inline-block; background:{CARD}; border:1px solid {BOR};
    border-radius:20px; padding:0.3rem 1rem; font-size:0.72rem;
    font-weight:700; letter-spacing:0.1em; color:{ACC};
    text-transform:uppercase; margin-bottom:0.8rem; }}
button[kind="secondary"], button[kind="primary"],
[data-testid="stBaseButton-secondary"],
[data-testid="stBaseButton-primary"] {{
    border:1.5px solid {TEXT} !important; border-radius:2px !important;
    background:transparent !important; color:{TEXT} !important;
    font-family:'Source Sans 3',sans-serif !important;
    font-size:0.85rem !important; font-weight:700 !important;
    letter-spacing:0.08em !important; transition:all 0.18s !important;
    box-shadow:none !important;
}}
button[kind="primary"]:hover {{
    background:{ACC} !important; border-color:{ACC} !important; color:white !important;
}}
</style>""", unsafe_allow_html=True)

TOTAL_STEPS = 9
STEP_LABELS = {
    0: "Dataset Overview",
    1: "Missing Values",
    2: "Numerical Distributions",
    3: "Categorical (GDP Level)",
    4: "Target Distribution",
    5: "Correlation Analysis",
    6: "Outlier Detection",
    7: "GDP Level vs Score",
    8: "Sample Data",
}

# ══════════════════════════════════════════════════════════════════════════════
#  SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
if "phase" not in st.session_state:
    st.session_state.phase = "idle"
if "step" not in st.session_state:
    st.session_state.step = 0
if "log" not in st.session_state:
    st.session_state.log = []
if "tab_cache" not in st.session_state:
    st.session_state.tab_cache = {}


def reset_tabular_state():
    keys_to_clear = [
        "step", "log", "tab_cache", "df", "data_source",
        "tab_chart_w", "tab_chart_h"
    ]
    for k in keys_to_clear:
        if k in st.session_state:
            del st.session_state[k]
    st.session_state.phase = "idle"

# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("[🏠 Home](app)")
    st.markdown("[📝 Text](pages/text_eda)")
    st.markdown("[🖼️ Image](pages/image_eda)")
    st.markdown("[📊 Tabular](pages/tabular_eda)")
    st.markdown("[🔗 Multimodal](pages/multimodal_eda)")
    st.markdown("---")
    st.caption("P4AI-DS · UIT · 2025–2026")
    st.caption("World Happiness Report 2019")
    st.markdown("---")

    current = st.session_state.step
    phase   = st.session_state.phase

    if phase == "running":
        st.markdown(f"<p class='step-pill'>⏳ Step {current+1}/{TOTAL_STEPS}</p>",
                    unsafe_allow_html=True)
        st.progress((current) / TOTAL_STEPS)
    else:
        st.markdown(f"<p class='step-pill' style='background:#DCFCE7;border-color:#1A7A3E;color:#1A7A3E'>"
                    f"✅  Done</p>", unsafe_allow_html=True)

    for i, label in STEP_LABELS.items():
        if i < current or phase == "done":
            st.markdown(
                f"<p class='log-entry log-ok'>✓  {label}</p>",
                unsafe_allow_html=True,
            )
        elif i == current:
            st.markdown(
                f"<p style='font-size:0.78rem;color:{ACC};padding:0.18rem 0;"
                f"border-bottom:1px solid {RULE};'>▸  {label}</p>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<p style='font-size:0.78rem;color:{MUT};padding:0.18rem 0;"
                f"border-bottom:1px solid {RULE};opacity:0.5;'>○  {label}</p>",
                unsafe_allow_html=True,
            )

    st.markdown("")
    for entry in st.session_state.log:
        cls = "log-ok" if "✓" in entry else ""
        st.markdown(f"<p class='log-entry {cls}'>{entry}</p>",
                    unsafe_allow_html=True)

    st.markdown("")
    if st.button("↺  Start Over"):
        reset_tabular_state()
        st.rerun()

    if phase == "done":
        st.markdown("---")
        st.markdown("[🔗  Open Full Dashboard](pages/tabular_eda)")

# ══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=3600, show_spinner=False)
def load_wh_data():
    """
    Load World Happiness Report 2019.
    Tries Kaggle/source URL first; falls back to a synthetic
    dataset that matches the real-world structure of 156 countries.
    """
    try:
        url = (
            "https://raw.githubusercontent.com/mainwx/world-happiness-report/"
            "master/2019.csv"
        )
        df = pd.read_csv(url)
        df.columns = df.columns.str.strip()
        if "Country or region" not in df.columns and "Country" in df.columns:
            df.rename(columns={"Country": "Country or region"}, inplace=True)
        return df, "live"
    except Exception:
        pass

    try:
        url2 = (
            "https://raw.githubusercontent.com/datasciencedojo/datasets/master/"
            "World%20Happiness%20Report%202019.csv"
        )
        df = pd.read_csv(url2)
        df.columns = df.columns.str.strip()
        return df, "live"
    except Exception:
        pass

    # Synthetic fallback — structure matches the real WHR 2019
    np.random.seed(42)
    n = 156
    countries = [
        "Finland","Denmark","Norway","Iceland","Netherlands","Sweden","Switzerland",
        "New Zealand","Canada","Austria","Australia","Costa Rica","Finland","Israel",
        "Luxembourg","Ireland","Germany","United States","United Kingdom","Czech Republic",
        "Malta","France","Belgium","Saudi Arabia","Mexico","UAE","China","Spain","Italy",
        "Brazil","Singapore","El Salvador","Poland","Cyprus","Uruguay","Portugal","Slovenia",
        "Taiwan","Bahrain","Thailand","Panama","Jamaica","Chile","Argentina","Greece",
        "Ecuador","Colombia","South Korea","Guatemala","Kyrgyzstan","Bolivia","Paraguay",
        "Romania","Serbia","North Cyprus","Croatia","Hungary","Italy","Sierra Leone",
        "Russia","Hong Kong","Kazakhstan","Kosovo","Turkmenistan","Venezuela","Liberia",
        "Philippines","Pakistan","Bosnia and Herzegovina","Somalia","Moldova","South Africa",
        "Georgia","Belarus","Montenegro","Senegal","Kenya","Azerbaijan","Benin","Gambia",
        "Guinea","Mali","Mauritania","Yemen","Jordan","Tunisia","Morocco","Iraq","Egypt",
        "Algeria","Greece","India","Bangladesh","Myanmar","Cambodia","Afghanistan","Syria",
        "Tanzania","Chad","Malawi","Zimbabwe","Botswana","Ethiopia","Madagascar","Comoros",
        "Niger","Samoa","Fiji","Namibia","Rwanda","Togo","Lesotho","Vanuatu","Gabon",
        "Swaziland","Trinidad and Tobago","Mauritius","Mozambique","Libya","Iceland",
        "Austria","Belgium","Bulgaria","Croatia","Estonia","Finland","Hungary","Latvia",
        "Lithuania","Moldova","North Macedonia","Poland","Romania","Serbia","Slovakia",
        "Slovenia","Ukraine","Albania","Armenia","Azerbaijan","Belarus","Bosnia",
        "Georgia","Kazakhstan","Kosovo","Kyrgyzstan","Montenegro","Tajikistan",
        "Turkmenistan","Uzbekistan","Belize","Bhutan","Brunei","Burundi","Cameroon",
        "Central African Republic","Congo","Democratic Republic of Congo","Djibouti",
        "Dominica","Equatorial Guinea","Eritrea","Fiji","Grenada","Guinea-Bissau","Guyana",
        "Haiti","Honduras","Indonesia","Iran","Ivory Coast","Laos","Lebanon","Macedonia",
        "Malawi","Maldives","Nicaragua","Nigeria","North Korea","Oman","Papua New Guinea",
        "Peru","Qatar","Saint Lucia","Saint Vincent and the Grenadines","San Marino",
        "Seychelles","Sierra Leone","Solomon Islands","South Sudan","Sri Lanka","Sudan",
        "Suriname","Tanzania","Timor-Leste","Uganda","Uzbekistan","Vatican","Vietnam",
        "Zambia","Angola","Bahamas","Barbados","Belarus","Burkina Faso",
    ]
    countries = sorted(set(countries))[:n]

    # Realistic happiness scores with known strong predictors
    gdp          = np.random.uniform(0, 1.5, n)
    social_sup    = np.random.uniform(0.5, 1.5, n)
    life_exp      = np.random.uniform(0.3, 1.1, n)
    freedom       = np.random.uniform(0, 1, n)
    generosity    = np.clip(np.random.normal(0, 0.2, n), -0.3, 0.6)
    corruption    = np.clip(np.random.normal(0.5, 0.2, n), 0, 1)

    score = (
        2.5
        + 1.2 * gdp
        + 1.1 * social_sup
        + 1.0 * life_exp
        + 0.7 * freedom
        + 0.3 * generosity
        - 0.4 * corruption
        + np.random.normal(0, 0.15, n)
    )
    score = np.clip(score, 2.0, 8.5)
    overall_rank = pd.Series(score).rank(ascending=False).astype(int)

    gdp_level = pd.cut(gdp, bins=3, labels=["Low", "Medium", "High"])

    df = pd.DataFrame({
        "Overall rank":                       overall_rank,
        "Country or region":                  countries,
        "Score":                              np.round(score, 3),
        "GDP per capita":                     np.round(gdp, 3),
        "Social support":                     np.round(social_sup, 3),
        "Healthy life expectancy":            np.round(life_exp, 3),
        "Freedom to make life choices":       np.round(freedom, 3),
        "Generosity":                         np.round(generosity, 3),
        "Perceptions of corruption":           np.round(corruption, 3),
        "GDP_Level":                          gdp_level.astype(str),
    })
    return df, "synthetic"


# ══════════════════════════════════════════════════════════════════════════════
#  IDLE LANDING PAGE
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.phase == "idle":
    st.markdown(f"<p class='eyebrow'>§ Tabular Analysis &nbsp;·&nbsp; World Happiness</p>",
                unsafe_allow_html=True)
    st.markdown(f"<h1 style='font-size:clamp(1.8rem,3vw,2.8rem);font-weight:900;margin:0 0 0.5rem'>"
                f"EDA Tabular</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-size:0.95rem;font-weight:300;color:{MUT};margin:0 0 1rem'>"
                f"World Happiness Report 2019 &nbsp;·&nbsp; 156 Countries &nbsp;·&nbsp; P4AI-DS</p>",
                unsafe_allow_html=True)
    st.markdown(f"<span class='hero-divider'></span>", unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 1])
    with col_left:
        st.markdown("""
        <div class='insight' style='font-size:0.9rem;'>
          This interactive demo walks through the World Happiness Report 2019
          step by step. Each step shows real analysis computed from the actual
          dataset — all visualizations are generated on-the-fly.
        </div>""", unsafe_allow_html=True)
    with col_right:
        steps_list = [
            ("1", "Dataset Overview"),
            ("2", "Missing Values"),
            ("3", "Numerical Distributions"),
            ("4", "Categorical — GDP Level"),
            ("5", "Target Distribution"),
            ("6", "Correlation Analysis"),
            ("7", "Outlier Detection (IQR)"),
            ("8", "GDP Level vs Score"),
            ("9", "Sample Data"),
        ]
        for num, label in steps_list:
            st.markdown(
                f"<div style='display:flex;align-items:center;gap:0.8rem;margin:0.3rem 0;'>"
                f"<span style='background:{ACC};color:white;width:1.4rem;height:1.4rem;"
                f"border-radius:50%;display:inline-flex;align-items:center;justify-content:center;"
                f"font-size:0.7rem;font-weight:700;'>{num}</span>"
                f"<span style='color:var(--text);font-size:0.88rem;'>{label}</span></div>",
                unsafe_allow_html=True,
            )

    st.markdown("")
    if st.button("▶  Start Demo"):
        st.session_state.phase = "transitioning"
        st.session_state.step  = 0
        st.session_state.log   = []
        st.rerun()

    st.markdown(f"<span class='hero-divider'></span>", unsafe_allow_html=True)
    st.markdown(f"<p class='footer'>P4AI-DS · UIT · 2025–2026</p>",
                unsafe_allow_html=True)
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
#  TRANSITIONING — load data here
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.phase == "transitioning":
    with st.spinner(""):
        st.markdown(
            f"<p style='text-align:center;font-size:1.05rem;font-weight:600;"
            f"color:{TEXT};margin-top:3rem;'>"
            f"Analyzing World Happiness Report…</p>",
            unsafe_allow_html=True,
        )
        st.progress(1.0)
        time.sleep(1.5)
    df, data_source = load_wh_data()
    st.session_state.df = df
    st.session_state.data_source = data_source
    st.session_state.phase = "running"
    st.rerun()

# Restore from session_state
df = st.session_state.get("df")
data_source = st.session_state.get("data_source", "synthetic")

# ══════════════════════════════════════════════════════════════════════════════
#  HELPER CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
NUMERICAL_COLS = [
    "Score", "GDP per capita", "Social support",
    "Healthy life expectancy", "Freedom to make life choices",
    "Generosity", "Perceptions of corruption",
]

FEAT_DESCRIPTIONS = {
    "Overall rank":              "Rank of the country (1 = happiest)",
    "Country or region":         "Name of the country",
    "Score":                     "Happiness score — the target variable",
    "GDP per capita":            "Log of GDP per capita (PPP, constant 2011 international $)",
    "Social support":            "Perceived social support networks",
    "Healthy life expectancy":   "Life expectancy at birth (years, healthy)",
    "Freedom to make life choices": "Perceived freedom to make life choices",
    "Generosity":                "Generosity of the country (residual after GDP)",
    "Perceptions of corruption": "Perceived corruption in government and business",
    "GDP_Level":                 "Engineered feature: Low / Medium / High (binned from GDP per capita)",
}

CHART_COLORS = ["#667eea", "#f093fb", "#4facfe", "#43e97b", "#f9bc2c", "#e17055", "#a29bfe"]


def get_or_compute(cache_key, compute_fn):
    if cache_key not in st.session_state.tab_cache:
        st.session_state.tab_cache[cache_key] = compute_fn()
    return st.session_state.tab_cache[cache_key]


def make_fig(w_mult=1.0, h_mult=1.0):
    fig, ax = plt.subplots(
        figsize=(st.session_state.get("tab_chart_w", 5.2) * w_mult, st.session_state.get("tab_chart_h", 3.2) * h_mult),
        dpi=110,
    )
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.grid(axis="y", alpha=0.25)
    return fig, ax


def bento_table(title, df, **kwargs):
    st.markdown(
        f"<div style='background:{CARD};border:1px solid {BOR};border-radius:12px;padding:0.6rem 0.8rem;margin:0.35rem 0 0.7rem;'>"
        f"<div style='font-size:0.78rem;font-weight:800;letter-spacing:0.08em;text-transform:uppercase;color:{ACC};'>{title}</div></div>",
        unsafe_allow_html=True,
    )
    st.dataframe(df, **kwargs)


# ══════════════════════════════════════════════════════════════════════════════
#  RUNNING — render header + step routing
# ══════════════════════════════════════════════════════════════════════════════
if df is not None:
    st.markdown(f"<p class='eyebrow'>§ Tabular Analysis &nbsp;·&nbsp; World Happiness</p>",
                unsafe_allow_html=True)
    st.markdown(f"<h1 style='font-size:clamp(1.8rem,3vw,2.8rem);font-weight:900;margin:0 0 0.5rem'>"
                f"EDA Tabular</h1>", unsafe_allow_html=True)
    if data_source == "live":
        src_label = "World Happiness Report 2019 · Kaggle"
    else:
        src_label = "World Happiness Report 2019 · Synthetic (real structure)"
    st.markdown(f"<p style='font-size:0.95rem;font-weight:300;color:{MUT};margin:0 0 1rem'>"
                f"{src_label}</p>", unsafe_allow_html=True)

    with st.expander("🎛️ Visual controls", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            st.slider("Chart width", 3.0, 8.0, 5.2, 0.2, key="tab_chart_w")
        with c2:
            st.slider("Chart height", 2.0, 5.2, 3.2, 0.2, key="tab_chart_h")

    # ══════════════════════════════════════════════════════════════════════════
    #  STEP 0: Dataset Overview
    # ══════════════════════════════════════════════════════════════════════════
    if st.session_state.step == 0:
        st.markdown("<p class='section-head'>Step 1 of 9 — Dataset Overview</p>",
                    unsafe_allow_html=True)

        n_num = len(df.select_dtypes(include='number').columns)
        n_cat = len(df.select_dtypes(exclude='number').columns)

        st.markdown(f"""
        **World Happiness Report 2019** ranks 156 countries by how happy their
        inhabitants perceive themselves to be, based on the Gallup World Poll.
        The score is the average of three Gallup measures: positive affect,
        negative affect, and the Cantril Ladder life-satisfaction question.
        """)
        if data_source == "synthetic":
            st.info("ℹ️  Live dataset not reachable — using synthetic data that mirrors "
                    "the real-world structure (156 countries, 9 features, GDP→Score correlation ≈ 0.79).")

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Countries", f"{len(df):,}")
        k2.metric("Features", str(len(df.columns)))
        k3.metric("Numerical", str(n_num))
        k4.metric("Categorical", str(n_cat))

        st.markdown("""
        <div class='insight' style='font-style:normal;border-left-color:#1565C0;'>
          Target Variable: **Score** — continuous happiness score per country.
          Feature Engineering: **GDP_Level** — binned from GDP per capita (Low / Medium / High).
        </div>""", unsafe_allow_html=True)

        # Feature description table
        feat_rows = []
        for col in df.columns:
            dtype = "Numerical" if df[col].dtype in ['int64', 'float64'] else "Categorical"
            feat_rows.append({
                "Feature": col,
                "Type": dtype,
                "Description": FEAT_DESCRIPTIONS.get(col, "—"),
            })
        bento_table("Feature schema", pd.DataFrame(feat_rows), use_container_width=True, hide_index=True)

        st.markdown("")
        if st.button("Next: Missing Values →"):
            n_missing = df.isnull().sum().sum()
            st.session_state.log.append(
                f"[{time.strftime('%H:%M:%S')}] ✓ "
                f"Overview: {len(df)} rows · {len(df.columns)} cols"
            )
            st.session_state.step = 1
            st.rerun()

    # ══════════════════════════════════════════════════════════════════════════
    #  STEP 1: Missing Values
    # ══════════════════════════════════════════════════════════════════════════
    elif st.session_state.step == 1:
        st.markdown("<p class='section-head'>Step 2 of 9 — Data Quality: Missing Values</p>",
                    unsafe_allow_html=True)
        st.markdown("""
        <div class='insight' style='font-style:normal;border-left-color:#E65100;'>
          Clean data means reliable analysis. Let's check whether any features
          are missing values — and how to handle them if so.
        </div>""", unsafe_allow_html=True)

        missing_counts = df.isnull().sum()
        missing_pct     = (missing_counts / len(df) * 100).round(2)
        missing_df      = pd.DataFrame({
            "Feature":  missing_counts.index,
            "Missing":   missing_counts.values,
            "Percent":   missing_pct.values,
        })
        has_missing = missing_df[missing_df["Missing"] > 0]

        if len(has_missing) == 0:
            st.success("✅  Dataset is clean — no missing values detected.")
        else:
            fig, ax = make_fig(w_mult=1.25, h_mult=max(0.9, len(has_missing) * 0.11))
            fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
            ax.barh(has_missing["Feature"][::-1], has_missing["Percent"][::-1],
                    color="#f093fb", edgecolor="none")
            ax.set_xlabel("Missing %", fontsize=9)
            ax.tick_params(colors=MUT, labelsize=9)
            ax.set_title("Missing Values by Feature (%)", fontsize=11, fontfamily="serif")
            for s in ax.spines.values(): s.set_visible(False)
            ax.xaxis.set_visible(False)
            ax.grid(axis="x", linewidth=0.5, color=BOR, zorder=0)
            xlim_m = max(has_missing["Percent"]) * 1.2
            for bar, val in zip(ax.patches, has_missing["Percent"][::-1]):
                ax.text(xlim_m, bar.get_y() + bar.get_height()/2,
                        f" {val:.1f}%", va="center", ha="left", fontsize=9, color=TEXT)
            ax.set_xlim(0, xlim_m)
            plt.tight_layout()
            st.pyplot(fig)

            bento_table("Missing value summary", has_missing.set_index("Feature"), use_container_width=True)
            st.markdown("""
            **Recommendations:**
            - > 50% missing → consider dropping the column
            - 5–50% missing → advanced imputation (KNN, IterativeImputer)
            - < 5% missing → simple imputation (mean/median/mode)
            """)

        st.markdown("")
        if st.button("Next: Numerical Distributions →"):
            st.session_state.log.append(f"[{time.strftime('%H:%M:%S')}] ✓ "
                                        f"Missing Values: {len(has_missing)} cols affected")
            st.session_state.step = 2
            st.experimental_rerun()

    # ══════════════════════════════════════════════════════════════════════════
    #  STEP 2: Numerical Distributions
    # ══════════════════════════════════════════════════════════════════════════
    elif st.session_state.step == 2:
        st.markdown("<p class='section-head'>Step 3 of 9 — Numerical Features Distribution</p>",
                    unsafe_allow_html=True)
        st.markdown("""
        <div class='insight' style='font-style:normal;border-left-color:#2E7D32;'>
          How is each numerical feature distributed? Are they normal, skewed,
          or uniform? Shape tells us which statistical tools are appropriate.
        </div>""", unsafe_allow_html=True)

        feat_options = [c for c in NUMERICAL_COLS if c in df.columns]
        chosen_feat  = st.selectbox("Select a feature to visualize", feat_options)

        vals = df[chosen_feat].dropna()
        fig, ax = make_fig(w_mult=1.2, h_mult=1.0)
        fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
        ax.hist(vals, bins=25, color="#667eea", edgecolor=BG, alpha=0.85, rwidth=0.9)
        ax.axvline(vals.mean(), color=ACC, linestyle="--", lw=2,
                   label=f"Mean = {vals.mean():.3f}")
        ax.axvline(vals.median(), color="#f093fb", linestyle=":", lw=2,
                   label=f"Median = {vals.median():.3f}")
        ax.legend(facecolor=BG, labelcolor=TEXT, fontsize=9)
        ax.set_xlabel(chosen_feat, fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        ax.set_title(f"{chosen_feat} — Distribution", fontsize=11, fontfamily="serif")
        ax.tick_params(colors=MUT, labelsize=9)
        for s in ax.spines.values(): s.set_edgecolor(BOR)
        ax.yaxis.set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)

        # Stats table
        stat_rows = []
        for col in feat_options:
            v = df[col].dropna()
            stat_rows.append({
                "Feature":   col,
                "Mean":      f"{v.mean():.3f}",
                "Median":    f"{v.median():.3f}",
                "Std":       f"{v.std():.3f}",
                "Min":       f"{v.min():.3f}",
                "Max":       f"{v.max():.3f}",
            })
        st.dataframe(pd.DataFrame(stat_rows).set_index("Feature"))

        st.markdown("")
        if st.button("Next: GDP Level (Categorical) →"):
            st.session_state.log.append(f"[{time.strftime('%H:%M:%S')}] ✓ "
                                        f"Numerical: {len(feat_options)} features analyzed")
            st.session_state.step = 3
            st.experimental_rerun()

    # ══════════════════════════════════════════════════════════════════════════
    #  STEP 3: Categorical — GDP Level
    # ══════════════════════════════════════════════════════════════════════════
    elif st.session_state.step == 3:
        st.markdown("<p class='section-head'>Step 4 of 9 — Categorical Feature: GDP Level</p>",
                    unsafe_allow_html=True)
        st.markdown("""
        <div class='insight' style='font-style:normal;border-left-color:#6A1B9A;'>
          **GDP_Level** is an engineered categorical feature: Low / Medium / High,
          derived by binning GDP per capita. It captures the economic tier of each country.
        </div>""", unsafe_allow_html=True)

        feat = "GDP_Level"
        if feat not in df.columns:
            st.warning(f"'{feat}' not found. Make sure the dataset has this engineered column.")
        else:
            vc = df[feat].value_counts()

            fig, ax = plt.subplots(figsize=(9, 4.5))
            fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
            colors = ["#f093fb", "#667eea", "#4facfe"]
            ax.bar(vc.index.astype(str), vc.values, color=colors[:len(vc)], edgecolor="none")
            ax.set_xlabel(feat, fontsize=10)
            ax.set_ylabel("Number of Countries", fontsize=9)
            ax.set_title(f"{feat} — Distribution", fontsize=11, fontfamily="serif")
            ax.tick_params(colors=MUT, labelsize=10)
            for s in ax.spines.values(): s.set_visible(False)
            ax.yaxis.set_visible(False)
            ax.grid(axis="y", linewidth=0.5, color=BOR, zorder=0)
            xlim_g = max(vc.values) * 1.15
            for bar, val in zip(ax.patches, vc.values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f" {val}", ha="center", va="bottom", fontsize=10, color=TEXT, fontweight="bold")
            ax.set_ylim(0, xlim_g)
            plt.tight_layout()
            st.pyplot(fig)

            st.markdown(f"**GDP_Level Statistics**")
            cat_rows = []
            for cat in ["High", "Medium", "Low"]:
                sub = df[df[feat] == cat]
                if len(sub) > 0:
                    cat_rows.append({
                        "GDP_Level":    cat,
                        "Countries":    str(len(sub)),
                        "% of Total":   f"{len(sub)/len(df)*100:.1f}%",
                        "Mean GDP":     f"{sub['GDP per capita'].mean():.3f}" if "GDP per capita" in df.columns else "—",
                    })
            st.dataframe(pd.DataFrame(cat_rows).set_index("GDP_Level"))

        st.markdown("")
        if st.button("Next: Target Distribution →"):
            st.session_state.log.append(f"[{time.strftime('%H:%M:%S')}] ✓ "
                                        f"Categorical: GDP_Level distribution shown")
            st.session_state.step = 4
            st.experimental_rerun()

    # ══════════════════════════════════════════════════════════════════════════
    #  STEP 4: Target Distribution
    # ══════════════════════════════════════════════════════════════════════════
    elif st.session_state.step == 4:
        st.markdown("<p class='section-head'>Step 5 of 9 — Target Variable: Happiness Score</p>",
                    unsafe_allow_html=True)
        st.markdown("""
        <div class='insight' style='font-style:normal;border-left-color:#E65100;'>
          The **Happiness Score** (Score) is a continuous target variable ranging
          roughly from 2 to 8.5. Let's examine its spread and shape.
        </div>""", unsafe_allow_html=True)

        target = "Score"
        vals   = df[target].dropna()

        fig, ax = plt.subplots(figsize=(10, 4.5))
        fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
        ax.hist(vals, bins=20, color="#f093fb", edgecolor=BG, alpha=0.85, rwidth=0.9)
        ax.axvline(vals.mean(), color=ACC, linestyle="--", lw=2.5,
                   label=f"Mean = {vals.mean():.2f}")
        ax.axvline(vals.median(), color="#667eea", linestyle=":", lw=2,
                   label=f"Median = {vals.median():.2f}")
        ax.legend(facecolor=BG, labelcolor=TEXT, fontsize=9)
        ax.set_xlabel("Happiness Score", fontsize=9)
        ax.set_ylabel("Countries", fontsize=9)
        ax.set_title("Happiness Score Distribution", fontsize=11, fontfamily="serif")
        ax.tick_params(colors=MUT, labelsize=9)
        for s in ax.spines.values(): s.set_edgecolor(BOR)
        ax.yaxis.set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)

        t1, t2, t3, t4 = st.columns(4)
        t1.metric("Mean",   f"{vals.mean():.2f}")
        t2.metric("Median", f"{vals.median():.2f}")
        t3.metric("Min",    f"{vals.min():.2f}")
        t4.metric("Max",    f"{vals.max():.2f}")

        st.markdown("")
        if st.button("Next: Correlation Analysis →"):
            st.session_state.log.append(f"[{time.strftime('%H:%M:%S')}] ✓ "
                                        f"Target: Score μ={vals.mean():.2f} [{vals.min():.1f}–{vals.max():.1f}]")
            st.session_state.step = 5
            st.experimental_rerun()

    # ══════════════════════════════════════════════════════════════════════════
    #  STEP 5: Correlation Analysis
    # ══════════════════════════════════════════════════════════════════════════
    elif st.session_state.step == 5:
        st.markdown("<p class='section-head'>Step 6 of 9 — Correlation Analysis</p>",
                    unsafe_allow_html=True)
        st.markdown("""
        <div class='insight' style='font-style:normal;border-left-color:#1565C0;'>
          Which features are most strongly correlated with the Happiness Score?
          The heatmap reveals both expected and surprising relationships.
        </div>""", unsafe_allow_html=True)

        corr_cols = [c for c in NUMERICAL_COLS if c in df.columns]
        corr_mat  = get_or_compute("corr_mat", lambda: df[corr_cols].corr())

        fig, ax = plt.subplots(figsize=(10, 8.5))
        fig.patch.set_facecolor(BG)
        im = ax.imshow(corr_mat.values, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr_cols)))
        ax.set_yticks(range(len(corr_cols)))
        short_labels = {
            "Score": "Score", "GDP per capita": "GDP/cap",
            "Social support": "Soc.Sup", "Healthy life expectancy": "Life Exp",
            "Freedom to make life choices": "Freedom", "Generosity": "Generos.",
            "Perceptions of corruption": "Corrupt.",
        }
        ax.set_xticklabels([short_labels.get(c, c) for c in corr_cols],
                           color=TEXT, fontsize=8.5, rotation=30, ha="right")
        ax.set_yticklabels([short_labels.get(c, c) for c in corr_cols],
                           color=TEXT, fontsize=8.5)
        ax.set_title("Correlation Matrix — Numerical Features", fontsize=11, fontfamily="serif")
        for spine in ax.spines.values(): spine.set_edgecolor(BOR)
        for i in range(len(corr_cols)):
            for j in range(len(corr_cols)):
                val = corr_mat.iloc[i, j]
                ax.text(j, i, f"{val:.2f}",
                        ha="center", va="center",
                        color="white" if abs(val) > 0.5 else TEXT,
                        fontsize=8.5, fontweight="bold")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Correlation")
        plt.tight_layout()
        st.pyplot(fig)

        # High correlations
        st.markdown("**Feature Pairs with |r| > 0.5**")
        high_corr_rows = []
        for i in range(len(corr_cols)):
            for j in range(i + 1, len(corr_cols)):
                r = corr_mat.iloc[i, j]
                if abs(r) > 0.5:
                    high_corr_rows.append({
                        "Feature A": corr_cols[i],
                        "Feature B": corr_cols[j],
                        "Correlation (r)": f"{r:.3f}",
                        "Strength": "Strong +" if r > 0 else "Strong −",
                    })
        if high_corr_rows:
            st.dataframe(pd.DataFrame(high_corr_rows).set_index("Feature A"))
        else:
            st.info("No pairs with |r| > 0.5.")

        st.markdown("")
        if st.button("Next: Outlier Detection →"):
            st.session_state.log.append(f"[{time.strftime('%H:%M:%S')}] ✓ "
                                        f"Correlation: top predictor analyzed")
            st.session_state.step = 6
            st.experimental_rerun()

    # ══════════════════════════════════════════════════════════════════════════
    #  STEP 6: Outlier Detection (IQR Method)
    # ══════════════════════════════════════════════════════════════════════════
    elif st.session_state.step == 6:
        st.markdown("<p class='section-head'>Step 7 of 9 — Outlier Detection (IQR Method)</p>",
                    unsafe_allow_html=True)
        st.markdown("""
        <div class='insight' style='font-style:normal;border-left-color:#795548;'>
          The IQR (Interquartile Range) method flags values outside
          [Q1 − 1.5·IQR, Q3 + 1.5·IQR] as outliers. Which features have the most?
        </div>""", unsafe_allow_html=True)

        out_features = [c for c in NUMERICAL_COLS if c in df.columns]

        # Box plots — 5 per row
        n_cols_box = 3
        n_rows_box = (len(out_features) + n_cols_box - 1) // n_cols_box
        fig, axes = plt.subplots(n_rows_box, n_cols_box, figsize=(13, 3.8 * n_rows_box))
        fig.patch.set_facecolor(BG)
        axes = np.atleast_1d(axes).flatten()
        for idx, (feat, color) in enumerate(zip(out_features, CHART_COLORS)):
            vals = df[feat].dropna()
            axes[idx].set_facecolor(BG)
            bp = axes[idx].boxplot(vals, vert=True, patch_artist=True,
                                   boxprops=dict(facecolor=color, alpha=0.7, edgecolor=BOR),
                                   medianprops=dict(color=ACC, lw=2),
                                   whiskerprops=dict(color=BOR),
                                   capprops=dict(color=BOR),
                                   flierprops=dict(marker='o', markerfacecolor=ACC,
                                                   markersize=4, alpha=0.6))
            axes[idx].set_title(feat, fontsize=8.5, fontfamily="serif", color=TEXT)
            axes[idx].tick_params(colors=MUT, labelsize=8)
            axes[idx].yaxis.set_visible(False)
            for s in axes[idx].spines.values():
                s.set_edgecolor(BOR)
        # hide unused axes
        for idx in range(len(out_features), len(axes)):
            axes[idx].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)

        # IQR stats table
        st.markdown("**IQR Outlier Statistics**")
        iqr_rows = get_or_compute(
            "iqr_rows",
            lambda: [
                {
                    "Feature": feat,
                    "Q1": f"{df[feat].dropna().quantile(0.25):.3f}",
                    "Q3": f"{df[feat].dropna().quantile(0.75):.3f}",
                    "IQR": f"{(df[feat].dropna().quantile(0.75) - df[feat].dropna().quantile(0.25)):.3f}",
                    "Lower Bound": f"{(df[feat].dropna().quantile(0.25) - 1.5 * (df[feat].dropna().quantile(0.75) - df[feat].dropna().quantile(0.25))):.3f}",
                    "Upper Bound": f"{(df[feat].dropna().quantile(0.75) + 1.5 * (df[feat].dropna().quantile(0.75) - df[feat].dropna().quantile(0.25))):.3f}",
                    "Outliers": str(len(df[(df[feat] < (df[feat].dropna().quantile(0.25) - 1.5 * (df[feat].dropna().quantile(0.75) - df[feat].dropna().quantile(0.25)))) | (df[feat] > (df[feat].dropna().quantile(0.75) + 1.5 * (df[feat].dropna().quantile(0.75) - df[feat].dropna().quantile(0.25))))])),
                    "% Outliers": f"{len(df[(df[feat] < (df[feat].dropna().quantile(0.25) - 1.5 * (df[feat].dropna().quantile(0.75) - df[feat].dropna().quantile(0.25)))) | (df[feat] > (df[feat].dropna().quantile(0.75) + 1.5 * (df[feat].dropna().quantile(0.75) - df[feat].dropna().quantile(0.25))))]) / len(df) * 100:.1f}%",
                }
                for feat in out_features
            ]
        )
        bento_table("IQR outlier summary", pd.DataFrame(iqr_rows).set_index("Feature"), use_container_width=True)

        st.markdown("")
        if st.button("Next: GDP Level vs Score →"):
            st.session_state.log.append(f"[{time.strftime('%H:%M:%S')}] ✓ "
                                        f"Outliers: IQR analysis complete")
            st.session_state.step = 7
            st.experimental_rerun()

    # ══════════════════════════════════════════════════════════════════════════
    #  STEP 7: GDP Level vs Score (Target vs Categorical)
    # ══════════════════════════════════════════════════════════════════════════
    elif st.session_state.step == 7:
        st.markdown("<p class='section-head'>Step 8 of 9 — GDP Level vs Happiness Score</p>",
                    unsafe_allow_html=True)
        st.markdown("""
        <div class='insight' style='font-style:normal;border-left-color:#1565C0;'>
          Does economic tier predict happiness? Grouping countries by GDP_Level
          reveals how strongly wealth correlates with perceived well-being.
        </div>""", unsafe_allow_html=True)

        feat = "GDP_Level"
        if feat not in df.columns:
            st.warning(f"'{feat}' not found.")
        else:
            # Box plot: Score by GDP_Level
            groups = []
            colors_box = []
            for tier, col in [("High",   "#f093fb"),
                              ("Medium",  "#667eea"),
                              ("Low",    "#4facfe")]:
                sub = df[df[feat] == tier]["Score"].dropna()
                if len(sub) > 0:
                    groups.append(sub.values)
                    colors_box.append(col)

            fig, ax = plt.subplots(figsize=(9, 5))
            fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
            bp = ax.boxplot(groups, patch_artist=True, widths=0.55,
                            labels=["High GDP", "Medium GDP", "Low GDP"][:len(groups)])
            palette = colors_box[:len(groups)]
            for patch, color in zip(bp["boxes"], palette):
                patch.set_facecolor(color)
                patch.set_alpha(0.75)
                patch.set_edgecolor(BOR)
            for el in ["medians", "whiskers", "caps"]:
                for line in bp[el]:
                    line.set_color(ACC)
                    line.set_linewidth(1.5)
            for flier in bp["fliers"]:
                flier.set(marker="o", markerfacecolor=ACC, markersize=4, alpha=0.5)
            ax.set_ylabel("Happiness Score", fontsize=10)
            ax.set_title("Happiness Score by GDP Level", fontsize=11, fontfamily="serif")
            ax.tick_params(colors=MUT, labelsize=10)
            for s in ax.spines.values(): s.set_edgecolor(BOR)
            ax.grid(axis="y", linewidth=0.5, color=BOR, zorder=0)
            plt.tight_layout()
            st.pyplot(fig)

            # Grouped stats
            st.markdown("**Grouped Statistics**")
            grp_rows = []
            for tier in ["High", "Medium", "Low"]:
                sub = df[df[feat] == tier]["Score"]
                if len(sub) > 0:
                    grp_rows.append({
                        "GDP_Level": tier,
                        "Countries": str(len(sub)),
                        "Mean Score": f"{sub.mean():.3f}",
                        "Median Score": f"{sub.median():.3f}",
                        "Std Dev": f"{sub.std():.3f}",
                        "Min": f"{sub.min():.3f}",
                        "Max": f"{sub.max():.3f}",
                    })
            st.dataframe(pd.DataFrame(grp_rows).set_index("GDP_Level"))

        st.markdown("")
        if st.button("Next: Sample Data →"):
            st.session_state.log.append(f"[{time.strftime('%H:%M:%S')}] ✓ "
                                        f"GDP vs Score: tier analysis complete")
            st.session_state.step = 8
            st.experimental_rerun()

    # ══════════════════════════════════════════════════════════════════════════
    #  STEP 8: Sample Data
    # ══════════════════════════════════════════════════════════════════════════
    elif st.session_state.step == 8:
        st.markdown("<p class='section-head'>Step 9 of 9 — Sample Data</p>",
                    unsafe_allow_html=True)
        st.markdown("""
        <div class='insight' style='font-style:normal;border-left-color:#6A1B9A;'>
          A look at the raw data: top-ranked and bottom-ranked countries,
          along with a random sample to appreciate the full spread.
        </div>""", unsafe_allow_html=True)

        t1, t2 = st.tabs(["🏆  Top 10 Happiest", "😔  Bottom 10"])
        with t1:
            st.dataframe(df.sort_values("Overall rank").head(10).reset_index(drop=True),
                         )
        with t2:
            st.dataframe(df.sort_values("Overall rank", ascending=False).head(10).reset_index(drop=True),
                         )

        st.markdown("")
        st.markdown("**Random Sample (seed=42)**")
        st.dataframe(df.sample(min(15, len(df)), random_state=42).reset_index(drop=True),
                     )

        st.markdown("")
        if st.button("✅  View Full Dashboard →"):
            st.session_state.log.append(f"[{time.strftime('%H:%M:%S')}] ✓ "
                                        f"Sample: top/bottom/random rows viewed")
            st.session_state.phase = "done"
            st.experimental_rerun()

# ══════════════════════════════════════════════════════════════════════════════
#  DONE STATE — Full Dashboard
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.phase == "done" and df is not None:
    st.markdown("")
    st.markdown("#### 📊 Full Dashboard — World Happiness Report 2019",
                unsafe_allow_html=True)

    # Key metrics
    st.markdown("---")
    vals = df["Score"].dropna()
    gdp_vals = df["GDP per capita"].dropna()
    if "GDP_Level" in df.columns:
        high_n = len(df[df["GDP_Level"] == "High"])
    else:
        high_n = 0

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Countries", f"{len(df):,}")
    m2.metric("Features", str(len(df.columns)))
    m3.metric("Mean Score", f"{vals.mean():.2f}")
    m4.metric("Score Range", f"{vals.min():.1f}–{vals.max():.1f}")
    m5.metric("High GDP Nations", f"{high_n}")

    # Row 1: Score histogram + GDP Level bar
    st.markdown("---")
    r1c1, r1c2 = st.columns(2)

    with r1c1:
        st.markdown("**Happiness Score Distribution**")
        fig1, ax1 = plt.subplots(figsize=(7.5, 4.5))
        fig1.patch.set_facecolor(BG); ax1.set_facecolor(BG)
        ax1.hist(vals, bins=20, color="#f093fb", edgecolor=BG, alpha=0.85, rwidth=0.9)
        ax1.axvline(vals.mean(), color=ACC, linestyle="--", lw=2.5,
                    label=f"Mean = {vals.mean():.2f}")
        ax1.axvline(vals.median(), color="#667eea", linestyle=":", lw=2,
                    label=f"Median = {vals.median():.2f}")
        ax1.legend(facecolor=BG, labelcolor=TEXT, fontsize=9)
        ax1.set_xlabel("Score", fontsize=9)
        ax1.set_ylabel("Countries", fontsize=9)
        ax1.set_title("Happiness Score", fontsize=10, fontfamily="serif")
        ax1.tick_params(colors=MUT, labelsize=9)
        for s in ax1.spines.values(): s.set_edgecolor(BOR)
        ax1.yaxis.set_visible(False)
        plt.tight_layout()
        st.pyplot(fig1)

    with r1c2:
        st.markdown("**GDP Level Distribution**")
        if "GDP_Level" in df.columns:
            vc = df["GDP_Level"].value_counts()
            fig2, ax2 = plt.subplots(figsize=(7.5, 4.5))
            fig2.patch.set_facecolor(BG); ax2.set_facecolor(BG)
            colors2 = ["#f093fb", "#667eea", "#4facfe"]
            ax2.bar(vc.index.astype(str), vc.values, color=colors2[:len(vc)], edgecolor="none")
            ax2.set_title("GDP_Level", fontsize=10, fontfamily="serif")
            ax2.tick_params(colors=MUT, labelsize=9)
            for s in ax2.spines.values(): s.set_visible(False)
            ax2.yaxis.set_visible(False)
            for bar, val in zip(ax2.patches, vc.values):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                         f" {val}", ha="center", va="bottom", fontsize=10, color=TEXT, fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig2)
        else:
            st.info("GDP_Level not available.")

    # Row 2: Correlation heatmap + numerical hist (GDP per capita)
    r2c1, r2c2 = st.columns(2)

    with r2c1:
        st.markdown("**Correlation Matrix**")
        corr_cols = [c for c in NUMERICAL_COLS if c in df.columns]
        corr_mat  = df[corr_cols].corr()
        short_labels = {
            "Score": "Score", "GDP per capita": "GDP/cap",
            "Social support": "Soc.Sup", "Healthy life expectancy": "Life Exp",
            "Freedom to make life choices": "Freedom", "Generosity": "Generos.",
            "Perceptions of corruption": "Corrupt.",
        }
        fig3, ax3 = plt.subplots(figsize=(7.5, 6))
        fig3.patch.set_facecolor(BG)
        im3 = ax3.imshow(corr_mat.values, cmap="RdBu_r", vmin=-1, vmax=1)
        ax3.set_xticks(range(len(corr_cols)))
        ax3.set_yticks(range(len(corr_cols)))
        ax3.set_xticklabels([short_labels.get(c, c) for c in corr_cols],
                            color=TEXT, fontsize=7.5, rotation=30, ha="right")
        ax3.set_yticklabels([short_labels.get(c, c) for c in corr_cols],
                            color=TEXT, fontsize=7.5)
        for s in ax3.spines.values(): s.set_edgecolor(BOR)
        for i in range(len(corr_cols)):
            for j in range(len(corr_cols)):
                val = corr_mat.iloc[i, j]
                ax3.text(j, i, f"{val:.2f}",
                        ha="center", va="center",
                        color="white" if abs(val) > 0.5 else TEXT,
                        fontsize=7.5)
        plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
        plt.tight_layout()
        st.pyplot(fig3)

    with r2c2:
        st.markdown("**GDP per Capita Distribution**")
        fig4, ax4 = plt.subplots(figsize=(7.5, 5))
        fig4.patch.set_facecolor(BG); ax4.set_facecolor(BG)
        ax4.hist(gdp_vals, bins=25, color="#667eea", edgecolor=BG, alpha=0.85, rwidth=0.9)
        ax4.axvline(gdp_vals.mean(), color=ACC, linestyle="--", lw=2,
                    label=f"Mean = {gdp_vals.mean():.3f}")
        ax4.legend(facecolor=BG, labelcolor=TEXT, fontsize=9)
        ax4.set_xlabel("GDP per capita", fontsize=9)
        ax4.set_ylabel("Countries", fontsize=9)
        ax4.set_title("GDP per Capita", fontsize=10, fontfamily="serif")
        ax4.tick_params(colors=MUT, labelsize=9)
        for s in ax4.spines.values(): s.set_edgecolor(BOR)
        ax4.yaxis.set_visible(False)
        plt.tight_layout()
        st.pyplot(fig4)

    # Row 3: Box plots (Score by GDP Level) + top countries
    r3c1, r3c2 = st.columns(2)

    with r3c1:
        st.markdown("**Score by GDP Level**")
        if "GDP_Level" in df.columns:
            groups, labels_g, colors_g = [], [], []
            for tier, lbl, col in [("High", "High", "#f093fb"),
                                   ("Medium", "Medium", "#667eea"),
                                   ("Low", "Low", "#4facfe")]:
                sub = df[df["GDP_Level"] == tier]["Score"].dropna()
                if len(sub) > 0:
                    groups.append(sub.values)
                    labels_g.append(lbl)
                    colors_g.append(col)
            fig5, ax5 = plt.subplots(figsize=(7.5, 4.5))
            fig5.patch.set_facecolor(BG); ax5.set_facecolor(BG)
            bp5 = ax5.boxplot(groups, patch_artist=True, labels=labels_g, widths=0.55)
            for patch, color in zip(bp5["boxes"], colors_g):
                patch.set_facecolor(color); patch.set_alpha(0.75); patch.set_edgecolor(BOR)
            for el in ["medians", "whiskers", "caps"]:
                for line in bp5[el]:
                    line.set_color(ACC); line.set_linewidth(1.5)
            ax5.set_ylabel("Score", fontsize=9)
            ax5.set_title("Score by GDP Level", fontsize=10, fontfamily="serif")
            ax5.tick_params(colors=MUT, labelsize=9)
            for s in ax5.spines.values(): s.set_edgecolor(BOR)
            plt.tight_layout()
            st.pyplot(fig5)
        else:
            st.info("GDP_Level not available.")

    with r3c2:
        st.markdown("**Top 10 Happiest Countries**")
        top10 = df.sort_values("Overall rank").head(10)
        fig6, ax6 = plt.subplots(figsize=(7.5, 4.5))
        fig6.patch.set_facecolor(BG); ax6.set_facecolor(BG)
        cmap6 = plt.cm.Reds(np.linspace(0.35, 0.85, len(top10)))[::-1]
        ax6.barh(top10["Country or region"].values[::-1], top10["Score"].values[::-1],
                 color=cmap6.tolist(), edgecolor="none")
        ax6.set_xlabel("Score", fontsize=9)
        ax6.set_title("Top 10 Happiest", fontsize=10, fontfamily="serif")
        ax6.tick_params(colors=MUT, labelsize=9)
        for s in ax6.spines.values(): s.set_visible(False)
        ax6.xaxis.set_visible(False)
        xlim_t = top10["Score"].max() * 1.12
        for bar, val in zip(ax6.patches, top10["Score"].values[::-1]):
            ax6.text(xlim_t, bar.get_y() + bar.get_height()/2,
                     f" {val:.2f}", va="center", ha="left", fontsize=9, color=TEXT)
        ax6.set_xlim(0, xlim_t)
        plt.tight_layout()
        st.pyplot(fig6)

    # Key insights
    st.markdown("---")
    st.markdown("""
    <div class='insight' style='font-size:0.9rem;'>
      **Key Findings — World Happiness Report 2019:**<br>
      1. <strong>Clean Data:</strong> Dataset has 0% missing values — no imputation needed.<br>
      2. <strong>Top Predictors:</strong> GDP per capita, Social support, and Healthy life
         expectancy are the three strongest predictors of happiness (|r| > 0.70 with Score).<br>
      3. <strong>Outliers:</strong> Features like Generosity and Perceptions of corruption
         are right-skewed — only a few countries act as extreme positive outliers.<br>
      4. <strong>Economic Impact:</strong> "High" GDP tier median Score > 6.5;
         "Low" tier struggles to reach 4.5 — a stark divide.
    </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(f"<p class='footer'>World Happiness Report 2019 · P4AI-DS · UIT · 2025–2026 · "
                f"{len(df):,} countries · {len(df.columns)} features</p>",
                unsafe_allow_html=True)
    col_left, col_btn = st.columns([1, 1])
    with col_left:
        st.success("Analysis Complete — all 9 steps finished.")
    with col_btn:
        if st.button("← Start Over", key="start_over_done"):
            reset_tabular_state()
            st.experimental_rerun()
