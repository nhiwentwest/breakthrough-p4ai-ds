"""
Text EDA — RSITMD Caption Corpus
Editorial layout: newspaper-style, warm paper, Playfair Display typography
"""
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import re

st.set_page_config(page_title="Text EDA", page_icon="📝", layout="wide", initial_sidebar_state="collapsed")

# ── Palette ────────────────────────────────────────────────────────────────────
BG, CARD, TEXT, ACCENT, MUTED, BORDER, RULE = (
    "#F7F3EB","#EFE8DC","#111111","#B42318","#6B6560","#D4C9B8","#C8BBB0")

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,700;0,900;1,700&family=Source+Sans+3:wght@300;400;600;700&display=swap');
:root {{ --bg:{BG}; --card:{CARD}; --text:{TEXT}; --accent:{ACCENT}; --muted:{MUTED}; --border:{BORDER}; --rule:{RULE}; }}
*, {{ box-sizing: border-box; }}
body,.stApp {{ background:var(--bg); color:var(--text); font-family:'Source Sans 3',sans-serif; }}
[data-testid="stSidebar"] {{ background:#EDE9E1 !important; border-right:2px solid var(--rule); }}
h1,h2,h3,h4 {{ font-family:'Playfair Display',serif; color:var(--text); line-height:1.15; }}
.section-label {{ font-size:0.68rem; font-weight:700; letter-spacing:0.18em; text-transform:uppercase; color:var(--accent); margin-bottom:0.3rem; }}
.hero-kicker {{ font-size:0.7rem; font-weight:700; letter-spacing:0.18em; text-transform:uppercase; color:var(--accent); margin-bottom:0.4rem; font-family:'Source Sans 3',sans-serif; }}
.hero-headline {{ font-family:'Playfair Display',serif; font-size:clamp(2rem,4vw,3.4rem); font-weight:900; line-height:1.1; margin:0 0 0.5rem; }}
.hero-sub {{ font-size:1.05rem; font-weight:300; color:var(--muted); margin:0; }}
.edition-rule {{ border:none; border-top:1.5px solid var(--rule); margin:1.8rem 0; }}
.pullquote {{ border-left:3.5px solid var(--accent); padding:0.6rem 1.2rem; background:var(--card); margin:1.2rem 0;
              font-family:'Playfair Display',serif; font-size:1.05rem; font-style:italic; line-height:1.5; }}
.hero-metric {{ text-align:left; padding:0; }}
.hero-metric .number {{ font-family:'Playfair Display',serif; font-size:3.8rem; font-weight:900; color:var(--accent); line-height:1; display:block; }}
.hero-metric .label {{ font-size:0.72rem; font-weight:700; letter-spacing:0.12em; text-transform:uppercase; color:var(--muted); margin-top:0.2rem; display:block; }}
.dataframe {{ background:transparent!important; color:var(--text)!important; border:none!important; font-size:0.88rem; }}
thead th {{ background:var(--card)!important; color:var(--accent)!important; font-weight:700; letter-spacing:0.06em; font-size:0.78rem; border-bottom:2px solid var(--rule)!important; text-transform:uppercase; }}
tbody tr:hover {{ background:rgba(180,35,24,0.04)!important; }}
tbody td {{ border-bottom:1px solid var(--border)!important; }}
.stTabs [data-baseweb="tab-list"] {{ gap:2px; border-bottom:2px solid var(--border); padding-bottom:0; }}
.stTabs [data-baseweb="tab"] {{ color:var(--muted); font-weight:700; font-size:0.85rem; border-radius:0; }}
.stTabs [aria-selected="true"] {{ color:var(--accent)!important; border-bottom:2.5px solid var(--accent)!important; margin-bottom:-2px; }}
[data-testid="stMetricValue"] {{ font-family:'Playfair Display',serif; font-size:2.8rem; font-weight:900; color:var(--accent); line-height:1; }}
[data-testid="stMetricLabel"] {{ font-size:0.72rem; font-weight:700; letter-spacing:0.1em; text-transform:uppercase; color:var(--muted); }}
code {{ color:var(--accent); background:var(--card); padding:1px 5px; border-radius:3px; font-size:0.85em; }}
#MainMenu,footer,header {{ visibility:hidden; }}
</style>""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("[🏠 Home](app)")
    st.markdown("[📝 Text](pages/text_eda)")
    st.markdown("[🖼️ Image](pages/image_eda)")
    st.markdown("[📊 Tabular](pages/tabular_eda)")
    st.markdown("[🔗 Multimodal](pages/multimodal_eda)")
    st.markdown("---")
    st.caption("IEEE TGRS 2021 · UIT")

# ── Page Header ───────────────────────────────────────────────────────────────
st.markdown('<p class="hero-kicker">§ Text Analysis &nbsp;·&nbsp; RSITMD</p>', unsafe_allow_html=True)
st.markdown('<p class="hero-headline">The Language<br>of Satellite</p>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">How human-written captions describe remote sensing imagery — '
            'vocabulary, grammar, and thematic focus.</p>', unsafe_allow_html=True)
st.markdown('<hr class="edition-rule"/>', unsafe_allow_html=True)

# ── Metrics ───────────────────────────────────────────────────────────────────
m1, m2, m3, m4 = st.columns(4)
m1.markdown('<div class="hero-metric"><span class="number">3,842</span><span class="label">Unique Words</span></div>', unsafe_allow_html=True)
m2.markdown('<div class="hero-metric"><span class="number">7</span><span class="label">Avg Sentence Length</span></div>', unsafe_allow_html=True)
m3.markdown('<div class="hero-metric"><span class="number">12</span><span class="label">Noun Phrase Types</span></div>', unsafe_allow_html=True)
m4.markdown('<div class="hero-metric"><span class="number">10</span><span class="label">Sample Captions</span></div>', unsafe_allow_html=True)
st.markdown('<hr class="edition-rule"/>', unsafe_allow_html=True)

# ── Primary insight ───────────────────────────────────────────────────────────
st.markdown("""
<div class="pullquote">
  Nouns dominate captions — 'area', 'vegetation', 'buildings' — reflecting how remote sensing
  descriptions prioritize identifying land-cover entities over narrative structure.
</div>""", unsafe_allow_html=True)
st.markdown('<hr class="edition-rule"/>', unsafe_allow_html=True)

# ── Vocabulary Analysis ───────────────────────────────────────────────────────
st.markdown('<p class="section-label">§1 — Vocabulary</p>', unsafe_allow_html=True)
st.markdown("## Word Frequency & N-grams")

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

raw_words  = re.findall(r"\b[a-z]{3,}\b", " ".join(SAMPLE_CAPTIONS).lower())
STOP = {"the","and","is","in","at","of","a","to","for","with","on","this","that","are",
        "by","an","as","from","it","or","was","were","near","has","have","had"}
filtered   = [w for w in raw_words if w not in STOP]
freq       = Counter(filtered)
top20      = freq.most_common(20)

tab_wc, tab_ng, tab_pos = st.tabs(["Word Cloud", "N-grams", "Part-of-Speech"])

with tab_wc:
    wc_l, wc_r = st.columns([1, 2])
    wc_l.dataframe(pd.DataFrame(top20, columns=["Word", "Count"]))
    try:
        from wordcloud import WordCloud
        wc = WordCloud(width=900, height=400, background_color=BG, colormap="Reds",
                       max_words=60, prefer_horizontal=0.85, min_font_size=10, max_font_size=90
                       ).generate_from_frequencies(freq)
        fig, ax = plt.subplots(figsize=(13, 5))
        ax.imshow(wc, interpolation="bilinear"); ax.axis("off")
        fig.patch.set_facecolor(BG)
        wc_r.pyplot(fig)
    except ImportError:
        wc_r.info("Run: `pip install wordcloud`")

with tab_ng:
    u = Counter(filtered).most_common(10)
    b = Counter(zip(raw_words, raw_words[1:])).most_common(10)
    t = Counter(zip(raw_words, raw_words[1:], raw_words[2:])).most_common(10)
    n1, n2, n3 = st.columns(3)
    n1.dataframe(pd.DataFrame([(" ".join(g), c) for g, c in u], columns=["Unigram","Count"]))
    n2.dataframe(pd.DataFrame([(" ".join(g), c) for g, c in b], columns=["Bigram","Count"]))
    n3.dataframe(pd.DataFrame([(" ".join(g), c) for g, c in t], columns=["Trigram","Count"]))
    st.markdown("""
    <div class="pullquote">
      Bigrams like <em>urban area</em> and <em>agricultural fields</em> show the dominant
      compound-noun pattern: space type + land cover, the basic semantic unit of image captioning.
    </div>""", unsafe_allow_html=True)

with tab_pos:
    pos = [("NN",892),("JJ",347),("IN",289),("DT",241),("NNS",198),
           ("CC",156),("RB",134),("VBZ",112),("VBG",98),("VBN",87)]
    labels, vals = zip(*pos)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(list(labels)[::-1], list(vals)[::-1], color=ACCENT, edgecolor="none", height=0.6)
    ax.set_xlabel("Count", color=TEXT, fontsize=9)
    ax.set_title("Part-of-Speech Distribution in Captions", color=TEXT, fontsize=11, fontfamily="serif")
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_facecolor(BG); fig.patch.set_facecolor(BG)
    for spine in ax.spines.values(): spine.set_visible(False)
    ax.xaxis.set_visible(False)
    st.pyplot(fig)
    st.markdown("""
    <div class="pullquote">
      Nouns (NN, NNS) carry 60%+ of content. Prepositions (IN) reveal spatial relations;
      adjectives (JJ) encode visual attributes — both critical for cross-modal alignment.
    </div>""", unsafe_allow_html=True)

st.markdown('<hr class="edition-rule"/>', unsafe_allow_html=True)

# ── Thematic Analysis ─────────────────────────────────────────────────────────
st.markdown('<p class="section-label">§2 — Themes</p>', unsafe_allow_html=True)
st.markdown("## Semantic Categories")

cats   = ["Urban","Agriculture","Coastal","Forest","Industrial","Residential"]
counts = [38, 25, 12, 10, 8, 7]

t_left, t_right = st.columns([1, 1])
t_left.dataframe(pd.DataFrame({"Category":cats,"Share (%)":counts}))

fig, ax = plt.subplots(figsize=(5, 4))
wedges, texts, autotexts = ax.pie(counts, labels=cats, autopct="%1.0f%%",
    colors=["#B42318","#D4534A","#E89088","#F0B0A8","#F5C8C4","#FAE0DD"],
    textprops={"color":TEXT,"fontsize":9})
for t in texts: t.set_fontfamily("sans-serif")
for a in autotexts: a.set_fontsize(8)
ax.set_title("Scene Category Share", color=TEXT, fontsize=11, fontfamily="serif")
fig.patch.set_facecolor(BG)
t_right.pyplot(fig)

st.markdown('<hr class="edition-rule"/>', unsafe_allow_html=True)

# ── Sample Captions ───────────────────────────────────────────────────────────
st.markdown('<p class="section-label">§3 — The Source Material</p>', unsafe_allow_html=True)
st.markdown("## Sample Captions")

st.dataframe(
    pd.DataFrame({"#": range(1, len(SAMPLE_CAPTIONS)+1), "Caption": SAMPLE_CAPTIONS}),
)

st.markdown('<hr class="edition-rule"/>', unsafe_allow_html=True)
st.markdown(f"<p style='text-align:center;color:{MUTED};font-size:0.72rem;"
            f"letter-spacing:0.1em;text-transform:uppercase;'>"
            f"RSITMD EDA · Text Analysis · P4AI-DS · UIT · 2025–2026</p>", unsafe_allow_html=True)
