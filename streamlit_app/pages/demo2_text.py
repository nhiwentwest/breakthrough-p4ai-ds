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
.block-container {{ padding-top:1.2rem; }}
.hero {{ font-family:'Playfair Display',serif; font-size:2.2rem; font-weight:900; margin:0; }}
.sub {{ color:{MUT}; margin-top:.25rem; margin-bottom:1rem; }}
.bento {{ background:{CARD}; border:1px solid {BOR}; border-radius:14px; padding:1rem; }}
.section {{ font-size:.68rem; letter-spacing:.12em; text-transform:uppercase; color:{ACC}; font-weight:700; margin-bottom:.6rem; }}
.stButton > button {{ border:1.5px solid {TEXT}; background:transparent; color:{TEXT}; font-weight:700; letter-spacing:.08em; border-radius:4px; }}
.stButton > button:hover {{ background:{ACC}; color:white; border-color:{ACC}; }}
</style>
""", unsafe_allow_html=True)

st.markdown("<p class='hero'>Demo 2 · Text Regression</p>", unsafe_allow_html=True)
st.markdown("<p class='sub'>Input text and predict a continuous score.</p>", unsafe_allow_html=True)

left, right = st.columns([1.2, 1])

with left:
    st.markdown("<div class='bento'>", unsafe_allow_html=True)
    st.markdown("<div class='section'>Model Selection</div>", unsafe_allow_html=True)
    model = st.selectbox("Choose regression model", [
        "DistilBERT Regressor",
        "RoBERTa Regressor",
        "Linear TF-IDF Regressor",
    ])

    st.markdown("<div class='section'>Text Input</div>", unsafe_allow_html=True)
    text = st.text_area(
        "Enter text",
        height=220,
        placeholder="Type / paste financial news text here...",
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
        st.metric("Predicted score", f"{value:.3f}")
        st.caption(f"Model used: {model}")
    else:
        st.info("Enter text and click Predict.")

    st.markdown("</div>", unsafe_allow_html=True)
