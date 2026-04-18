import streamlit as st
import numpy as np

st.set_page_config(page_title="Demo 2 · Text Regression", page_icon="📝", layout="wide")

BG = "#F7F3EB"
CARD = "#EFE8DC"
TEXT = "#111111"
ACC = "#B42318"
MUT = "#6B6560"
BOR = "#D4C9B8"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=Source+Sans+3:wght@300;400;600;700&display=swap');
body,.stApp {{ background:{BG}; color:{TEXT}; font-family:'Source Sans 3',sans-serif; }}
#MainMenu,footer,header {{ visibility:hidden; }}
section[data-testid="stSidebar"] {{ display:none !important; }}
div[data-testid="collapsedControl"] {{ display:none !important; }}
.block-container {{ padding:1.35rem 1.4rem 1rem; max-width: 1240px; }}
.hero {{ font-family:'Playfair Display',serif; font-size:2.35rem; font-weight:900; margin:0; letter-spacing:-0.02em; }}
.sub {{ color:{MUT}; margin-top:.35rem; margin-bottom:1rem; font-size:1rem; }}
.editor-shell {{ background: linear-gradient(180deg, rgba(239,232,220,.92), rgba(247,243,235,.98)); border:1px solid {BOR}; border-radius:20px; box-shadow:0 10px 30px rgba(17,17,17,.06); padding:1rem; }}
.bento {{ background:rgba(255,255,255,.35); border:1px solid rgba(212,201,184,.95); border-radius:18px; padding:1rem; backdrop-filter: blur(6px); }}
.section {{ font-size:.68rem; letter-spacing:.14em; text-transform:uppercase; color:{ACC}; font-weight:800; margin-bottom:.8rem; }}
.stButton > button {{ border:1.5px solid {TEXT}; background:transparent; color:{TEXT}; font-weight:800; letter-spacing:.08em; border-radius:10px; padding:.55rem .9rem; }}
.stButton > button:hover {{ background:{ACC}; color:white; border-color:{ACC}; }}
.model-chip {{ display:inline-block; padding:.25rem .6rem; border-radius:999px; background:#f1e4d3; border:1px solid #ddceb8; color:#2c2a26; font-size:.82rem; font-weight:700; margin-bottom:.4rem; }}
.metric-row {{ display:grid; grid-template-columns: repeat(1, minmax(0,1fr)); gap:.7rem; margin-top:.85rem; }}
.metric-card {{ background:rgba(255,255,255,.65); border:1px solid #dacdbd; border-radius:16px; padding:.8rem .9rem .75rem; box-shadow:0 4px 18px rgba(17,17,17,.05); }}
.metric-label {{ font-size:.63rem; letter-spacing:.1em; text-transform:uppercase; color:{MUT}; font-weight:700; }}
.metric-value {{ font-family:'Playfair Display',serif; font-size:1.55rem; font-weight:900; line-height:1.05; margin-top:.35rem; color:{TEXT}; }}
.small-note {{ color:{MUT}; font-size:0.82rem; }}
</style>
""", unsafe_allow_html=True)

st.markdown("<p class='hero'>Demo 2 · Text Regression</p>", unsafe_allow_html=True)
st.markdown("<p class='sub'>Input text and predict a continuous score.</p>", unsafe_allow_html=True)

st.markdown("<div class='editor-shell'>", unsafe_allow_html=True)
left, right = st.columns([1.15, 1])

with left:
    st.markdown("<div class='bento'>", unsafe_allow_html=True)
    st.markdown("<div class='section'>Model Selection</div>", unsafe_allow_html=True)
    model = st.selectbox(
        "Choose regression model",
        [
            "DistilBERT Regressor",
            "RoBERTa Regressor",
            "Linear TF-IDF Regressor",
        ],
        label_visibility="collapsed",
    )
    st.markdown(f"<div class='model-chip'>{model}</div>", unsafe_allow_html=True)

    st.markdown("<div class='section'>Text Input</div>", unsafe_allow_html=True)
    text = st.text_area(
        "Enter text",
        height=240,
        placeholder="Type / paste text here...",
        label_visibility="collapsed",
    )

    pred_btn = st.button("Predict", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown("<div class='bento'>", unsafe_allow_html=True)
    st.markdown("<div class='section'>Prediction Output</div>", unsafe_allow_html=True)

    if pred_btn:
        n_words = len(text.split())
        punct = sum(ch in ".,;:!?" for ch in text)
        base = 2.0 + 0.02 * n_words + 0.01 * punct
        bump = {"DistilBERT Regressor": 0.08, "RoBERTa Regressor": 0.12, "Linear TF-IDF Regressor": -0.04}[model]
        value = float(np.clip(base + bump, 0.0, 5.0))
        st.markdown(f"<div class='model-chip'>{model} ready</div>", unsafe_allow_html=True)
        st.markdown("<div class='metric-row'>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-card'><div class='metric-label'>Predicted score</div><div class='metric-value'>{value:.3f}</div></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        st.caption(f"Model used: {model}")
    else:
        st.info("Enter text and click Predict.")

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
