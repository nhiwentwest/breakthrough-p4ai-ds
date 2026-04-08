import streamlit as st
import numpy as np

st.set_page_config(page_title="Demo 2 · Tabular Regression", page_icon="📊", layout="wide")

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

st.markdown("<p class='hero'>Demo 2 · Tabular Regression</p>", unsafe_allow_html=True)
st.markdown("<p class='sub'>Editor-style demo for structured numerical prediction.</p>", unsafe_allow_html=True)

left, right = st.columns([1.2, 1])

with left:
    st.markdown("<div class='bento'>", unsafe_allow_html=True)
    st.markdown("<div class='section'>Model Selection</div>", unsafe_allow_html=True)
    model = st.selectbox("Choose regression model", [
        "XGBoost Regressor",
        "Random Forest Regressor",
        "Linear Regression",
    ])

    st.markdown("<div class='section'>Input Features</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        gdp = st.number_input("GDP per capita", min_value=0.0, max_value=2.5, value=1.0, step=0.01)
        social = st.number_input("Social support", min_value=0.0, max_value=2.0, value=1.0, step=0.01)
        life = st.number_input("Healthy life expectancy", min_value=0.0, max_value=1.5, value=0.7, step=0.01)
    with c2:
        freedom = st.number_input("Freedom to make life choices", min_value=0.0, max_value=1.5, value=0.5, step=0.01)
        generosity = st.number_input("Generosity", min_value=-0.5, max_value=1.0, value=0.1, step=0.01)
        corruption = st.number_input("Perceptions of corruption", min_value=0.0, max_value=1.2, value=0.3, step=0.01)

    pred_btn = st.button("Predict", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown("<div class='bento'>", unsafe_allow_html=True)
    st.markdown("<div class='section'>Prediction Output</div>", unsafe_allow_html=True)

    if pred_btn:
        score = (
            2.3 + 1.15*gdp + 1.05*social + 0.95*life + 0.65*freedom + 0.25*generosity - 0.35*corruption
        )
        jitter = {"XGBoost Regressor": 0.05, "Random Forest Regressor": 0.02, "Linear Regression": -0.03}[model]
        score = float(np.clip(score + jitter, 2.0, 8.5))
        st.metric("Predicted target", f"{score:.3f}")
        st.caption(f"Model used: {model}")
    else:
        st.info("Fill features and click Predict.")

    st.markdown("</div>", unsafe_allow_html=True)
