"""
RSITMD EDA — Home
Simple landing page with 4 navigation cards.
"""

import streamlit as st

st.set_page_config(
    page_title="RSITMD EDA",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Palette
BG = "#F7F3EB"
CARD = "#EFE8DC"
TEXT = "#111111"
ACCENT = "#B42318"
MUTED = "#6B6560"
BORDER = "#D4C9B8"

st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=Source+Sans+3:wght@300;400;600;700&display=swap');

:root {{
  --bg: {BG};
  --card: {CARD};
  --text: {TEXT};
  --accent: {ACCENT};
  --muted: {MUTED};
  --border: {BORDER};
}}

body, .stApp {{
  background: var(--bg);
  color: var(--text);
  font-family: 'Source Sans 3', sans-serif;
}}

/* Hide Streamlit left sidebar and toggle */
[data-testid="stSidebar"] {{
  display: none !important;
}}
[data-testid="collapsedControl"] {{
  display: none !important;
}}

/* Hide default chrome */
#MainMenu, footer, header {{
  visibility: hidden;
}}

.hero-kicker {{
  font-size: 0.78rem;
  font-weight: 700;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--accent);
  margin-bottom: 0.5rem;
}}

.hero-title {{
  font-family: 'Playfair Display', serif;
  font-size: clamp(2rem, 4vw, 3.2rem);
  font-weight: 900;
  line-height: 1.1;
  margin: 0;
}}

.hero-sub {{
  color: var(--muted);
  margin-top: 0.6rem;
  margin-bottom: 1.6rem;
  font-size: 1.05rem;
}}

.card {{
  border: 1px solid var(--border);
  border-radius: 14px;
  background: var(--card);
  padding: 1.1rem 1rem;
  min-height: 120px;
  transition: all 0.18s ease;
}}

.card:hover {{
  border-color: var(--accent);
  transform: translateY(-2px);
}}

.card a {{
  text-decoration: none;
  color: inherit;
  display: block;
}}

.card-title {{
  font-family: 'Playfair Display', serif;
  font-size: 1.35rem;
  margin-bottom: 0.35rem;
}}

.card-desc {{
  color: var(--muted);
  font-size: 0.96rem;
  line-height: 1.45;
}}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown('<p class="hero-kicker">RSITMD · P4AI-DS</p>', unsafe_allow_html=True)
st.markdown('<p class="hero-title">BREAKTHROUGH DEMO</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-sub">Main routing page: Assignment 02 (Machine Learning) and Assignment 01 (EDA).</p>',
    unsafe_allow_html=True,
)

st.markdown('<p class="hero-kicker" style="margin-top:0.2rem">Assignment 02 · Machine Learning</p>', unsafe_allow_html=True)
a2c1, a2c2, a2c3 = st.columns(3)
with a2c1:
    st.markdown(
        """
<div class="card">
  <a href="/demo2_tabular" target="_self">
    <div class="card-title">📊 Tabular</div>
    <div class="card-desc">Assignment 02 · Machine Learning<br>Regression Demo</div>
  </a>
</div>
""",
        unsafe_allow_html=True,
    )
with a2c2:
    st.markdown(
        """
<div class="card">
  <a href="/demo2_image" target="_self">
    <div class="card-title">🖼️ Image</div>
    <div class="card-desc">Assignment 02 · Machine Learning<br>Classification Demo</div>
  </a>
</div>
""",
        unsafe_allow_html=True,
    )
with a2c3:
    st.markdown(
        """
<div class="card">
  <a href="/demo2_text" target="_self">
    <div class="card-title">📝 Text</div>
    <div class="card-desc">Assignment 02 · Machine Learning<br>Regression Demo</div>
  </a>
</div>
""",
        unsafe_allow_html=True,
    )

st.markdown('<p class="hero-kicker" style="margin-top:1.2rem">Assignment 01 · Exploratory Data Analysis</p>', unsafe_allow_html=True)
a1r1c1, a1r1c2 = st.columns(2)
with a1r1c1:
    st.markdown(
        """
<div class="card">
  <a href="/tabular_eda" target="_self">
    <div class="card-title">📊 Tabular</div>
    <div class="card-desc">World Happiness EDA</div>
  </a>
</div>
""",
        unsafe_allow_html=True,
    )
with a1r1c2:
    st.markdown(
        """
<div class="card">
  <a href="/image_eda" target="_self">
    <div class="card-title">🖼️ Image</div>
    <div class="card-desc">MNIST EDA</div>
  </a>
</div>
""",
        unsafe_allow_html=True,
    )

a1r2c1, a1r2c2 = st.columns(2)
with a1r2c1:
    st.markdown(
        """
<div class="card">
  <a href="/text_eda" target="_self">
    <div class="card-title">📝 Text</div>
    <div class="card-desc">Twitter Financial News EDA</div>
  </a>
</div>
""",
        unsafe_allow_html=True,
    )
with a1r2c2:
    st.markdown(
        """
<div class="card">
  <a href="/multimodal_eda" target="_self">
    <div class="card-title">🔗 Multimodal</div>
    <div class="card-desc">RSITMD EDA</div>
  </a>
</div>
""",
        unsafe_allow_html=True,
    )
