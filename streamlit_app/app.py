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
st.markdown('<p class="hero-title">Exploratory Data Analysis</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-sub">Chọn một workspace để bắt đầu: Tabular, Image, Text hoặc Multimodal.</p>',
    unsafe_allow_html=True,
)

row1_col1, row1_col2 = st.columns(2)
with row1_col1:
    st.markdown(
        """
<div class="card">
  <a href="/tabular_eda" target="_self">
    <div class="card-title">📊 Tabular</div>
    <div class="card-desc">Khám phá thống kê bảng, phân phối thuộc tính và tương quan dữ liệu.</div>
  </a>
</div>
""",
        unsafe_allow_html=True,
    )
with row1_col2:
    st.markdown(
        """
<div class="card">
  <a href="/image_eda" target="_self">
    <div class="card-title">🖼️ Image</div>
    <div class="card-desc">Phân tích ảnh vệ tinh, đặc trưng thị giác và xu hướng theo scene.</div>
  </a>
</div>
""",
        unsafe_allow_html=True,
    )

row2_col1, row2_col2 = st.columns(2)
with row2_col1:
    st.markdown(
        """
<div class="card">
  <a href="/text_eda" target="_self">
    <div class="card-title">📝 Text</div>
    <div class="card-desc">Khảo sát caption, từ khóa nổi bật, n-gram và độ dài mô tả.</div>
  </a>
</div>
""",
        unsafe_allow_html=True,
    )
with row2_col2:
    st.markdown(
        """
<div class="card">
  <a href="/multimodal_eda" target="_self">
    <div class="card-title">🔗 Multimodal</div>
    <div class="card-desc">Đối chiếu ảnh-văn bản, đánh giá mức liên kết và độ tương thích.</div>
  </a>
</div>
""",
        unsafe_allow_html=True,
    )
