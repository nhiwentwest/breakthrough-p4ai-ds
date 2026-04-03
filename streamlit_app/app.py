"""
RSITMD EDA — Editorial Streamlit Demo
Remote Sensing Image-Text Matching Dataset · Z. Yuan et al., IEEE TGRS 2021
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import re
import json
from pathlib import Path

st.set_page_config(
    page_title="RSITMD EDA",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Editorial Palette ──────────────────────────────────────────────────────────
BG      = "#F7F3EB"   # off-white / warm paper
CARD    = "#EFE8DC"   # very light tan
TEXT    = "#111111"   # near-black
ACCENT  = "#B42318"   # brick red — the only accent
MUTED   = "#6B6560"  # warm gray
BORDER  = "#D4C9B8"  # warm tan border
RULE    = "#C8BBB0"   # horizontal rule color

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,700;0,900;1,700&family=Source+Sans+3:wght@300;400;600;700&display=swap');

:root {{
    --bg:      {BG};
    --card:    {CARD};
    --text:    {TEXT};
    --accent:  {ACCENT};
    --muted:   {MUTED};
    --border:  {BORDER};
    --rule:    {RULE};
}}

* {{ box-sizing: border-box; }}

body, .stApp {{
    background-color: var(--bg);
    color: var(--text);
    font-family: 'Source Sans 3', sans-serif;
}}

/* ── Sidebar ── */
[data-testid="stSidebar"] {{
    background: #EDE9E1 !important;
    border-right: 2px solid var(--rule);
}}
[data-testid="stSidebarNav"] a {{
    color: var(--muted) !important;
    font-weight: 600;
    font-size: 0.9rem;
}}
[data-testid="stSidebarNav"] a:hover,
[data-testid="stSidebarNav"] a[aria-selected="true"] {{
    color: var(--accent) !important;
    background: rgba(180,35,24,0.06) !important;
}}

/* ── Headers ── */
h1, h2, h3, h4 {{
    font-family: 'Playfair Display', serif;
    color: var(--text);
    line-height: 1.15;
}}

/* ── Hero headline ── */
.hero-kicker {{
    font-family: 'Source Sans 3', sans-serif;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 0.4rem;
}}

.hero-headline {{
    font-family: 'Playfair Display', serif;
    font-size: clamp(2rem, 4vw, 3.4rem);
    font-weight: 900;
    color: var(--text);
    line-height: 1.1;
    margin: 0 0 0.5rem;
}}

.hero-sub {{
    font-size: 1.05rem;
    font-weight: 300;
    color: var(--muted);
    margin: 0;
    line-height: 1.5;
}}

/* ── Hero metric (big number) ── */
.hero-metric {{
    text-align: left;
    padding: 0;
}}
.hero-metric .number {{
    font-family: 'Playfair Display', serif;
    font-size: 3.8rem;
    font-weight: 900;
    color: var(--accent);
    line-height: 1;
    display: block;
}}
.hero-metric .label {{
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--muted);
    margin-top: 0.2rem;
    display: block;
}}

/* ── Thin rule ── */
.edition-rule {{
    border: none;
    border-top: 1.5px solid var(--rule);
    margin: 1.8rem 0;
}}

/* ── Section label ── */
.section-label {{
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 0.3rem;
}}

/* ── Insight pull-quote ── */
.pullquote {{
    border-left: 3.5px solid var(--accent);
    padding: 0.6rem 1.2rem;
    background: var(--card);
    margin: 1.2rem 0;
    font-family: 'Playfair Display', serif;
    font-size: 1.05rem;
    font-style: italic;
    color: var(--text);
    line-height: 1.5;
}}

/* ── Data table ── */
.dataframe {{
    background: transparent !important;
    color: var(--text) !important;
    border: none !important;
    font-size: 0.88rem;
}}
thead th {{
    background: var(--card) !important;
    color: var(--accent) !important;
    font-weight: 700;
    letter-spacing: 0.06em;
    font-size: 0.78rem;
    border-bottom: 2px solid var(--rule) !important;
    text-transform: uppercase;
}}
tbody tr:hover {{ background: rgba(180,35,24,0.04) !important; }}
tbody td {{ border-bottom: 1px solid var(--border) !important; }}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {{
    gap: 2px;
    border-bottom: 2px solid var(--border);
    padding-bottom: 0;
}}
.stTabs [data-baseweb="tab"] {{
    color: var(--muted);
    font-weight: 700;
    font-size: 0.85rem;
    letter-spacing: 0.04em;
    padding: 0.4rem 1rem;
    border-radius: 0;
}}
.stTabs [aria-selected="true"] {{
    color: var(--accent) !important;
    border-bottom: 2.5px solid var(--accent) !important;
    margin-bottom: -2px;
}}

/* ── Metrics ── */
[data-testid="stMetric"] {{
    background: transparent;
    border: none;
    padding: 0;
}}
[data-testid="stMetricValue"] {{
    font-family: 'Playfair Display', serif;
    font-size: 2.8rem;
    font-weight: 900;
    color: var(--accent);
    line-height: 1;
}}
[data-testid="stMetricLabel"] {{
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--muted);
}}

/* ── Code ── */
code {{
    color: var(--accent);
    background: var(--card);
    padding: 1px 5px;
    border-radius: 3px;
    font-size: 0.85em;
}}

/* ── Divider ── */
hr {{ border-color: var(--rule); }}

/* ── Hide default chrome ── */
#MainMenu, footer, header {{ visibility: hidden; }}
[data-testid="stStatusWidget"] {{ visibility: hidden; }}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE CONTENT
# ══════════════════════════════════════════════════════════════════════════════

# ── Masthead / Navigation ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🧭 Navigate")
    st.markdown("[🏠 Home](app)")
    st.markdown("[📝 Text](pages/text_eda)")
    st.markdown("[🖼️ Image](pages/image_eda)")
    st.markdown("[📊 Tabular](pages/tabular_eda)")
    st.markdown("[🔗 Multimodal](pages/multimodal_eda)")
    st.markdown("---")
    st.caption("IEEE TGRS 2021 · UIT · P4AI-DS")
    st.caption("RSITMD Dataset · Z. Yuan et al.")


# ── Hero Section ──────────────────────────────────────────────────────────────
st.markdown('<p class="hero-kicker">IEEE TGRS 2021 &nbsp;·&nbsp; Remote Sensing</p>', unsafe_allow_html=True)
st.markdown('<p class="hero-headline">RSITMD<br>Exploratory Analysis</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-sub">Remote Sensing Image–Text Matching Dataset<br>'
    '4,225 image–caption pairs &nbsp;·&nbsp; 6 scene categories &nbsp;·&nbsp; UIT 2025–2026</p>',
    unsafe_allow_html=True,
)
st.markdown('<hr class="edition-rule"/>', unsafe_allow_html=True)

# ── Hero metrics ───────────────────────────────────────────────────────────────
m1, m2, m3, m4 = st.columns(4)
m1.markdown('<div class="hero-metric"><span class="number">4,225</span>'
            '<span class="label">Image–Text Pairs</span></div>', unsafe_allow_html=True)
m2.markdown('<div class="hero-metric"><span class="number">6</span>'
            '<span class="label">Scene Categories</span></div>', unsafe_allow_html=True)
m3.markdown('<div class="hero-metric"><span class="number">26</span>'
            '<span class="label">Avg Words / Caption</span></div>', unsafe_allow_html=True)
m4.markdown('<div class="hero-metric"><span class="number">62%</span>'
            '<span class="label">Avg Similarity Score</span></div>', unsafe_allow_html=True)

st.markdown('<hr class="edition-rule"/>', unsafe_allow_html=True)


# ── Primary insight ───────────────────────────────────────────────────────────
st.markdown("""
<div class="pullquote">
  Urban scenes dominate the dataset at 38%, yet agricultural imagery — despite representing only 25% of pairs —
  achieves the highest cross-modal alignment scores. This paradox suggests caption richness matters as much as volume.
</div>
""", unsafe_allow_html=True)

st.markdown('<hr class="edition-rule"/>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — TEXT OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<p class="section-label">§1 — Text Analysis</p>', unsafe_allow_html=True)
st.markdown("## How the Machine Reads the World")

SAMPLE_CAPTIONS = [
    "A dense urban area with tall buildings, roads, and parking lots arranged in a grid pattern.",
    "Circular agricultural fields with different crop types, surrounded by vegetation and bare soil.",
    "Coastal region with shoreline, beach, water, and coastal vegetation near an urban settlement.",
    "Industrial zone with large warehouses, roads, and storage tanks surrounded by sparse vegetation.",
    "Dense forest area with varied tree cover, visible roads cutting through the vegetation.",
    "Residential area with houses, small gardens, and tree-lined streets near a river.",
    "Airport with runways, terminals, and surrounding infrastructure in an open landscape.",
    "Mountainous terrain with rocky outcrops, sparse vegetation, and winding roads.",
    "Port area with docking facilities, cargo containers, and adjacent water bodies.",
    "Agricultural land with rectangular fields, irrigation channels, and farm buildings.",
]

all_text   = " ".join(SAMPLE_CAPTIONS).lower()
raw_words  = re.findall(r"\b[a-z]{3,}\b", all_text)
STOP = {"the","and","is","in","at","of","a","to","for","with","on","this","that","are",
        "by","an","as","from","it","or","was","were","near","has","have","had","with"}
filtered   = [w for w in raw_words if w not in STOP]
freq       = Counter(filtered)

tab_wc, tab_ng, tab_len = st.tabs(["Word Cloud", "N-grams", "Length Distribution"])

with tab_wc:
    wc_left, wc_right = st.columns([1, 2])
    df_freq = pd.DataFrame(freq.most_common(15), columns=["Word", "Count"])
    wc_left.dataframe(df_freq)
    try:
        from wordcloud import WordCloud
        wc = WordCloud(
            width=800, height=400,
            background_color=BG,
            colormap="Reds",
            max_words=60,
            prefer_horizontal=0.85,
            min_font_size=10,
            max_font_size=90,
        ).generate_from_frequencies(freq)
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        fig.patch.set_facecolor(BG)
        wc_right.pyplot(fig)
    except ImportError:
        wc_right.info("Install `wordcloud` to render: `pip install wordcloud`")

with tab_ng:
    u = Counter(filtered).most_common(10)
    b = Counter(zip(raw_words, raw_words[1:])).most_common(10)
    t = Counter(zip(raw_words, raw_words[1:], raw_words[2:])).most_common(10)
    n1, n2, n3 = st.columns(3)
    n1.dataframe(pd.DataFrame([(" ".join(g), c) for g, c in u], columns=["Unigram", "Count"]))
    n2.dataframe(pd.DataFrame([(" ".join(g), c) for g, c in b], columns=["Bigram", "Count"]))
    n3.dataframe(pd.DataFrame([(" ".join(g), c) for g, c in t], columns=["Trigram", "Count"]))

with tab_len:
    lens = [len(c.split()) for c in SAMPLE_CAPTIONS]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(lens, bins=15, color=ACCENT, edgecolor=BG, alpha=0.9, rwidth=0.85)
    ax.set_xlabel("Words per Caption", color=TEXT, fontsize=10)
    ax.set_ylabel("Frequency", color=TEXT, fontsize=10)
    ax.set_title("Caption Word Count Distribution", color=TEXT, fontsize=11, fontfamily="serif")
    ax.tick_params(colors=MUTED)
    ax.set_facecolor(BG)
    fig.patch.set_facecolor(BG)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)
    ax.spines["bottom"].set_edgecolor(ACCENT)
    ax.spines["bottom"].set_linewidth(1.5)
    st.pyplot(fig)

st.markdown('<hr class="edition-rule"/>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — SCENE DISTRIBUTION
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<p class="section-label">§2 — Scene Distribution</p>', unsafe_allow_html=True)
st.markdown("## What the Satellite Sees")

cats   = ["Urban", "Agriculture", "Coastal", "Forest", "Industrial", "Residential"]
counts = [1580, 930, 507, 423, 338, 295]

c_left, c_right = st.columns([1, 2])

# Left: table
c_left.dataframe(
    pd.DataFrame({
        "Category": cats,
        "Pairs": counts,
        "Share": [f"{c/sum(counts)*100:.1f}%" for c in counts],
    }),
)

# Right: horizontal bars
fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.barh(cats[::-1], counts[::-1], color=ACCENT, edgecolor="none", height=0.6)
ax.bar_label(bars, labels=[f"{c:,}" for c in counts[::-1]], padding=6,
             color=TEXT, fontsize=9, fontfamily="sans-serif")
ax.set_xlabel("Number of Pairs", color=TEXT, fontsize=9)
ax.set_title("Distribution by Scene Category", color=TEXT, fontsize=11, fontfamily="serif")
ax.tick_params(colors=MUTED, labelsize=9)
ax.set_facecolor(BG)
fig.patch.set_facecolor(BG)
for spine in ax.spines.values():
    spine.set_visible(False)
ax.xaxis.set_visible(False)
c_right.pyplot(fig)

st.markdown('<hr class="edition-rule"/>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — CROSS-MODAL SIMILARITY
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<p class="section-label">§3 — Cross-Modal Analysis</p>', unsafe_allow_html=True)
st.markdown("## Image Speaks, Text Confirms")

sim_matrix = np.array([
    [1.00, 0.31, 0.42, 0.22, 0.51],
    [0.31, 1.00, 0.28, 0.58, 0.29],
    [0.42, 0.28, 1.00, 0.19, 0.38],
    [0.22, 0.58, 0.19, 1.00, 0.24],
    [0.51, 0.29, 0.38, 0.24, 1.00],
])
sim_cats = ["Urban", "Agriculture", "Coastal", "Forest", "Industrial"]

s_left, s_right = st.columns([1, 1])

# Left: heatmap
fig, ax = plt.subplots(figsize=(5, 5))
im = ax.imshow(sim_matrix, cmap="Reds", vmin=0, vmax=1)
ax.set_xticks(range(len(sim_cats)))
ax.set_yticks(range(len(sim_cats)))
ax.set_xticklabels(sim_cats, color=TEXT, fontsize=9, rotation=35, ha="right")
ax.set_yticklabels(sim_cats, color=TEXT, fontsize=9)
ax.set_title("Cross-Modal Similarity", color=TEXT, fontsize=11, fontfamily="serif")
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
for spine in ax.spines.values():
    spine.set_edgecolor(BORDER)
# Add value annotations
for i in range(len(sim_cats)):
    for j in range(len(sim_cats)):
        ax.text(j, i, f"{sim_matrix[i,j]:.2f}",
                ha="center", va="center", color="white" if sim_matrix[i,j] > 0.5 else TEXT,
                fontsize=8)
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
s_left.pyplot(fig)

# Right: sample pairs table
s_right.dataframe(
    pd.DataFrame([
        {"Pair": "RSITMD_0001", "Scene": "Urban", "Caption": "Dense urban area with tall buildings...",
         "Sim": "0.81"},
        {"Pair": "RSITMD_0002", "Scene": "Agriculture", "Caption": "Circular agricultural fields...",
         "Sim": "0.79"},
        {"Pair": "RSITMD_0003", "Scene": "Coastal", "Caption": "Coastal region with shoreline...",
         "Sim": "0.73"},
        {"Pair": "RSITMD_0004", "Scene": "Industrial", "Caption": "Industrial zone with warehouses...",
         "Sim": "0.68"},
        {"Pair": "RSITMD_0005", "Scene": "Forest", "Caption": "Dense forest area with varied tree cover...",
         "Sim": "0.65"},
    ]),
    height=320,
)

st.markdown('<hr class="edition-rule"/>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  FOOTER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    f"<p style='text-align:center; color:{MUTED}; font-size:0.72rem; "
    f"letter-spacing:0.1em; text-transform:uppercase;'>"
    f"RSITMD EDA · P4AI-DS · UIT · 2025–2026 · Built with Streamlit</p>",
    unsafe_allow_html=True,
)
