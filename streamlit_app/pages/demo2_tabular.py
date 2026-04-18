import json
from pathlib import Path

import gdown
import joblib
import numpy as np
import streamlit as st

st.set_page_config(page_title="Demo 2 · Tabular Regression", page_icon="📊", layout="wide")

BG = "#F7F3EB"
CARD = "#EFE8DC"
TEXT = "#111111"
ACC = "#B42318"
MUT = "#6B6560"
BOR = "#D4C9B8"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_DIR = PROJECT_ROOT / "streamlit_app" / "checkpoints"
MODEL_DRIVE_IDS = {
    "linear_regression.joblib": "1SiLVaci0rpQjPs3iO9dVjeIcNmrSKW3a",
    "random_forest.joblib": "1eF-Kk7ZMBr67BnSNi1_pjVvkKpuz32hM",
    "gradient_boosting.joblib": "1LAtn7OCcyjXnZJOugWPeTJCNtc-ABEml",
    "scaler.joblib": "1dxAAIgxlVnOYB8ZK2I1eYd-8nnZUwcqX",
    "feature_columns.json": "1ah34c8PDl4_P9v5UTg5tdIfWbTamdBRY",
}

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
.small-note {{ color:{MUT}; font-size:0.82rem; }}
.kpi-grid {{ display:grid; grid-template-columns: repeat(2, minmax(0,1fr)); gap:.65rem; margin:.8rem 0 1rem; }}
.kpi-card {{ background:{CARD}; border:1px solid {BOR}; border-radius:12px; padding:.65rem .8rem; }}
.kpi-lbl {{ font-size:.62rem; letter-spacing:.08em; text-transform:uppercase; color:{MUT}; }}
.kpi-val {{ font-weight:700; font-size:1rem; margin-top:.15rem; }}
</style>
""", unsafe_allow_html=True)

st.markdown("<p class='hero'>Demo 2 · Tabular Regression</p>", unsafe_allow_html=True)
st.markdown("<p class='sub'>Load real tabular models from the provided checkpoint assets.</p>", unsafe_allow_html=True)

MODEL_ASSETS = {
    "Linear Regression": {
        "model": "linear_regression.joblib",
        "scaler": "scaler.joblib",
        "feature_columns": "feature_columns.json",
        "kind": "regression",
    },
    "Random Forest Regressor": {
        "model": "random_forest.joblib",
        "scaler": "scaler.joblib",
        "feature_columns": "feature_columns.json",
        "kind": "regression",
    },
    "Gradient Boosting Regressor": {
        "model": "gradient_boosting.joblib",
        "scaler": "scaler.joblib",
        "feature_columns": "feature_columns.json",
        "kind": "regression",
    },
}

RAW_INPUT_ORDER = [
    "age",
    "bmi",
    "children",
    "sex",
    "smoker",
    "region",
]


def _resolve_asset(filename: str) -> Path:
    candidate = CHECKPOINT_DIR / filename
    if candidate.exists() and candidate.stat().st_size > 0:
        return candidate
    file_id = MODEL_DRIVE_IDS.get(filename)
    if file_id is None:
        raise FileNotFoundError(f"No Drive file ID configured for {filename}")
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, str(candidate), quiet=False)
    if not candidate.exists() or candidate.stat().st_size == 0:
        raise FileNotFoundError(f"Failed to download {filename} from Drive")
    return candidate


@st.cache_resource(show_spinner=True)
def load_tabular_model(model_name: str):
    assets = MODEL_ASSETS[model_name]
    model_path = _resolve_asset(assets["model"])
    scaler_path = _resolve_asset(assets["scaler"])
    feature_path = _resolve_asset(assets["feature_columns"])

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    with open(feature_path, "r", encoding="utf-8") as f:
        feature_columns = json.load(f)

    return model, scaler, feature_columns, str(model_path)


def build_feature_row(age, bmi, children, sex, smoker, region):
    region_map = {
        "northwest": "region_northwest",
        "southeast": "region_southeast",
        "southwest": "region_southwest",
        "northeast": None,
    }
    row = {
        "age": age,
        "bmi": bmi,
        "children": children,
        "sex_male": 1 if sex == "male" else 0,
        "smoker_yes": 1 if smoker == "yes" else 0,
        "region_northwest": 0,
        "region_southeast": 0,
        "region_southwest": 0,
    }
    region_key = region_map.get(region)
    if region_key is not None:
        row[region_key] = 1
    return row


left, right = st.columns([1.2, 1])

with left:
    st.markdown("<div class='bento'>", unsafe_allow_html=True)
    st.markdown("<div class='section'>Model Selection</div>", unsafe_allow_html=True)
    model_name = st.selectbox(
        "Choose model",
        [
            "Linear Regression",
            "Random Forest Regressor",
            "Gradient Boosting Regressor",
        ],
    )

    st.markdown("<div class='section'>Input Features</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input("Age", min_value=0, max_value=120, value=30, step=1)
        bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0, step=0.1)
        children = st.number_input("Children", min_value=0, max_value=10, value=0, step=1)
    with c2:
        sex = st.selectbox("Sex", ["male", "female"])
        smoker = st.selectbox("Smoker", ["yes", "no"])
        region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

    pred_btn = st.button("Predict", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown("<div class='bento'>", unsafe_allow_html=True)
    st.markdown("<div class='section'>Prediction Output</div>", unsafe_allow_html=True)

    if pred_btn:
        model, scaler, feature_columns, _model_path = load_tabular_model(model_name)
        feature_row = build_feature_row(age, bmi, children, sex, smoker, region)

        missing_cols = [c for c in feature_columns if c not in feature_row]
        if missing_cols:
            st.error(f"Missing required feature columns: {missing_cols}")
        else:
            ordered = np.array([[feature_row[c] for c in feature_columns]], dtype=np.float32)
            scaled = scaler.transform(ordered)
            pred = model.predict(scaled)
            pred_value = float(np.ravel(pred)[0])

            st.metric("Predicted target", f"{pred_value:.3f}")
            st.caption(f"Model used: {model_name}")
            st.markdown("### Feature snapshot")
            st.dataframe({"feature": feature_columns, "value": ordered[0].tolist()}, use_container_width=True, hide_index=True)
    else:
        st.info("Fill features and click Predict.")

    st.markdown("</div>", unsafe_allow_html=True)
