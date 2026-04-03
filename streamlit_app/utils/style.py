"""
Shared styling helpers for the RSITMD EDA Streamlit app.
"""

import streamlit as st

# ── Palette ──────────────────────────────────────────────────────────────────
ACCENT        = "#FF6B35"
ACCENT_LIGHT  = "#FFE0D0"
SECONDARY     = "#1A1A2E"
BG            = "#0F0F1A"
CARD_BG       = "#1A1A2E"
TEXT          = "#EAEAEA"
MUTED         = "#8888AA"
BORDER        = "#2A2A4A"

# ── Page shell ───────────────────────────────────────────────────────────────
def page_config(title: str, emoji: str = "📊"):
    st.set_page_config(
        page_title=title,
        page_icon=emoji,
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    _inject_css()


def _inject_css():
    st.markdown(f"""
    <style>
    /* ── Root vars ── */
    :root {{
        --accent:        {ACCENT};
        --accent-light:  {ACCENT_LIGHT};
        --secondary:    {SECONDARY};
        --bg:           {BG};
        --card-bg:      {CARD_BG};
        --text:         {TEXT};
        --muted:        {MUTED};
        --border:       {BORDER};
    }}

    /* ── Body ── */
    body, .stApp {{ background-color: var(--bg); color: var(--text); }}

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {{
        background: #12122A !important;
        border-right: 1px solid var(--border);
    }}
    [data-testid="stSidebarNav"] a {{
        color: var(--muted) !important;
        font-weight: 600;
    }}
    [data-testid="stSidebarNav"] a:hover,
    [data-testid="stSidebarNav"] a[aria-selected="true"] {{
        color: var(--accent) !important;
        background: rgba(255,107,53,0.08) !important;
    }}

    /* ── Main content padding ── */
    .main .block-container {{
        padding-top: 2rem;
        padding-bottom: 2rem;
    }}

    /* ── Headings ── */
    h1, h2, h3 {{ color: {TEXT}; }}

    /* ── Metric cards ── */
    [data-testid="stMetric"] {{
        background: {CARD_BG};
        border: 1px solid {BORDER};
        border-radius: 12px;
        padding: 1rem 1.25rem;
    }}
    [data-testid="stMetricLabel"] {{ color: {MUTED}; }}
    [data-testid="stMetricValue"]  {{ color: {ACCENT}; font-weight: 700; }}

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 4px;
        background: {SECONDARY};
        border-radius: 10px;
        padding: 4px;
    }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 8px;
        color: {MUTED};
        font-weight: 600;
    }}
    .stTabs [aria-selected="true"] {{
        background: {ACCENT} !important;
        color: white !important;
    }}

    /* ── DataFrames ── */
    .dataframe {{
        background: {CARD_BG} !important;
        color: {TEXT} !important;
        border: none !important;
    }}
    thead th {{ background: {SECONDARY} !important; color: {ACCENT} !important; }}
    tbody tr:hover {{ background: rgba(255,107,53,0.06) !important; }}

    /* ── Expanders ── */
    .streamlit-expanderHeader {{
        background: {CARD_BG};
        border: 1px solid {BORDER};
        border-radius: 8px;
        color: {TEXT};
    }}
    .streamlit-expanderContent {{ background: {BG}; border-radius: 0 0 8px 8px; }}

    /* ── Code blocks ── */
    code {{ color: {ACCENT_LIGHT}; background: {SECONDARY}; padding: 2px 6px; border-radius: 4px; }}

    /* ── Buttons ── */
    .stButton > button {{
        background: {ACCENT};
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 700;
        padding: 0.5rem 1.5rem;
        transition: 0.2s;
    }}
    .stButton > button:hover {{ background: #ff8c5a; transform: translateY(-1px); }}

    /* ── Hide default Streamlit elements ── */
    #MainMenu, footer, header {{ visibility: hidden; }}
    [data-testid="stStatusWidget"] {{ visibility: hidden; }}
    </style>
    """, unsafe_allow_html=True)


# ── Section title ───────────────────────────────────────────────────────────
def section(title: str, emoji: str = ""):
    st.markdown(f"### {emoji} {title}" if emoji else f"### {title}")
    st.markdown("<hr style='border-color:#2A2A4A; margin-top:0; margin-bottom:1rem'/>",
                unsafe_allow_html=True)


# ── Insight card ─────────────────────────────────────────────────────────────
def insight(text: str, icon: str = "💡"):
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, rgba(255,107,53,0.12), rgba(255,107,53,0.04));
        border-left: 3px solid {ACCENT};
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 1rem;
        color: {TEXT};
    ">
    <strong>{icon} Insight:</strong> {text}
    </div>
    """, unsafe_allow_html=True)
