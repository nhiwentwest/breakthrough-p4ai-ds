import json
from pathlib import Path

import gdown
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Demo 2 · Tabular Regression", page_icon="📊", layout="wide")

if st.session_state.get("current_page") != "demo2_tabular":
    st.cache_resource.clear()
    import gc; gc.collect()
    try:
        import torch
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    except ImportError:
        pass
    st.session_state["current_page"] = "demo2_tabular"

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
    "insurance.csv": "16hHeuqWKFhrdk-PtyfVVOqDV1y77v9MS",
}

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
.small-note {{ color:{MUT}; font-size:0.82rem; }}
.metric-row {{ display:grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap:.7rem; margin-top:.85rem; }}
.metric-card {{ background:rgba(255,255,255,.65); border:1px solid #dacdbd; border-radius:16px; padding:.8rem .9rem .75rem; box-shadow:0 4px 18px rgba(17,17,17,.05); }}
.metric-label {{ font-size:.63rem; letter-spacing:.1em; text-transform:uppercase; color:{MUT}; font-weight:700; }}
.metric-value {{ font-family:'Source Sans 3',sans-serif; font-size:1.45rem; font-weight:800; line-height:1.05; margin-top:.35rem; color:{TEXT}; letter-spacing:-0.02em; }}
.metric-value.small {{ font-size:1.15rem; font-weight:800; }}
.model-chip {{ display:inline-block; padding:.25rem .6rem; border-radius:999px; background:#f1e4d3; border:1px solid #ddceb8; color:#2c2a26; font-size:.82rem; font-weight:700; margin-bottom:.4rem; }}
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


@st.cache_resource(show_spinner=True, max_entries=3)
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


@st.cache_data(show_spinner=True)
def load_insurance_dataset():
    dataset_path = _resolve_asset("insurance.csv")
    return pd.read_csv(dataset_path)


def prepare_insurance_data(df: pd.DataFrame):
    df_encoded = pd.get_dummies(df, columns=["sex", "smoker", "region"], drop_first=True)
    X = df_encoded.drop(columns=["charges"])
    y = df_encoded["charges"]
    return train_test_split(X, y, test_size=0.2, random_state=42)


def evaluate_model_on_insurance(model, scaler, feature_columns, X_test, y_test):
    X_test_scaled = scaler.transform(X_test)
    preds = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, preds)
    eval_df = X_test.copy().reset_index(drop=True)
    eval_df["actual"] = y_test.reset_index(drop=True)
    eval_df["pred"] = preds
    eval_df["residual"] = eval_df["pred"] - eval_df["actual"]
    return {
        "MSE": float(mse),
        "MAE": float(mean_absolute_error(y_test, preds)),
        "RMSE": float(np.sqrt(mse)),
        "R2": float(r2_score(y_test, preds)),
        "preds": preds,
        "eval_df": eval_df,
    }


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


st.markdown("<div class='editor-shell'>", unsafe_allow_html=True)
left, right = st.columns([1.15, 1])

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
        label_visibility="collapsed",
    )
    st.markdown(f"<div class='model-chip'>{model_name}</div>", unsafe_allow_html=True)

    st.markdown("<div class='section'>Input Features</div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.number_input("Age", min_value=0, max_value=120, value=30, step=1)
    with c2:
        bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0, step=0.1)
    with c3:
        children = st.number_input("Children", min_value=0, max_value=10, value=0, step=1)

    c4, c5, c6 = st.columns(3)
    with c4:
        sex = st.selectbox("Sex", ["male", "female"])
    with c5:
        smoker = st.selectbox("Smoker", ["yes", "no"])
    with c6:
        region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

    pred_btn = st.button("Predict", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown("<div class='bento'>", unsafe_allow_html=True)
    st.markdown("<div class='section'>Prediction Output</div>", unsafe_allow_html=True)

    df = load_insurance_dataset()
    X_train, X_test, y_train, y_test = prepare_insurance_data(df)

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
            score = evaluate_model_on_insurance(model, scaler, feature_columns, X_test, y_test)
            eval_df = score["eval_df"]
            residual_std = float(eval_df["residual"].std())
            outlier_mask = eval_df["residual"].abs() > (2 * residual_std if residual_std > 0 else np.inf)
            outlier_count = int(outlier_mask.sum())
            abs_err_by_smoker = eval_df.assign(smoker_group=np.where(eval_df.get("smoker_yes", 0) == 1, "yes", "no")).groupby("smoker_group")["residual"].apply(lambda s: s.abs().mean())
            abs_err_by_region = {}
            for region_col, region_label in [("region_northwest", "northwest"), ("region_southeast", "southeast"), ("region_southwest", "southwest")]:
                if region_col in eval_df.columns:
                    abs_err_by_region[region_label] = float(eval_df.loc[eval_df[region_col] == 1, "residual"].abs().mean())
            abs_err_by_sex = {
                "female": float(eval_df.loc[eval_df.get("sex_male", 0) == 0, "residual"].abs().mean()),
                "male": float(eval_df.loc[eval_df.get("sex_male", 0) == 1, "residual"].abs().mean()),
            }

            st.markdown(f"<div class='model-chip'>{model_name} ready</div>", unsafe_allow_html=True)
            st.metric("Predicted target", f"{pred_value:.3f}")
            st.markdown("<div class='metric-row'>", unsafe_allow_html=True)
            for label, value, small in [
                ("MSE", score['MSE'], False),
                ("MAE", score['MAE'], False),
                ("RMSE", score['RMSE'], False),
                ("R²", score['R2'], True),
            ]:
                st.markdown(
                    f"<div class='metric-card'><div class='metric-label'>{label}</div><div class='metric-value{' small' if small else ''}'>{value:.4f}</div></div>",
                    unsafe_allow_html=True,
                )
            st.markdown("</div>", unsafe_allow_html=True)

    else:
        st.info("Fill features and click Predict.")

st.markdown("</div>", unsafe_allow_html=True)

# full-width diagnostics section below both panels
if pred_btn and 'score' in locals():
    st.markdown("<div class='bento' style='margin-top:1rem;'>", unsafe_allow_html=True)
    tab_fit, tab_errors, tab_why = st.tabs(["Model fit", "Errors", "Why this prediction"])

    with tab_fit:
        fig_fit, ax = plt.subplots(figsize=(8.5, 4.5))
        ax.scatter(eval_df["actual"], eval_df["pred"], s=18, alpha=0.65, color="#b42318")
        mn = float(min(eval_df["actual"].min(), eval_df["pred"].min()))
        mx = float(max(eval_df["actual"].max(), eval_df["pred"].max()))
        ax.plot([mn, mx], [mn, mx], linestyle="--", color="#111111")
        ax.set_xlabel("Actual charges")
        ax.set_ylabel("Predicted charges")
        ax.set_title("Actual vs Predicted")
        st.pyplot(fig_fit, use_container_width=True)

    with tab_errors:
        c_left, c_right = st.columns(2)
        with c_left:
            fig_res, ax_res = plt.subplots(figsize=(6.2, 4))
            ax_res.scatter(eval_df["pred"], eval_df["residual"], s=18, alpha=0.65, color="#4f46e5")
            ax_res.axhline(0, linestyle="--", color="#111111")
            ax_res.set_xlabel("Predicted charges")
            ax_res.set_ylabel("Residuals (pred - actual)")
            ax_res.set_title("Residuals vs Predicted")
            st.pyplot(fig_res, use_container_width=True)
        with c_right:
            fig_hist, ax_hist = plt.subplots(figsize=(6.2, 4))
            ax_hist.hist(eval_df["residual"], bins=30, color="#d97706", alpha=0.85)
            ax_hist.axvline(0, linestyle="--", color="#111111")
            ax_hist.set_xlabel("Residual")
            ax_hist.set_ylabel("Count")
            ax_hist.set_title("Residual Histogram")
            st.pyplot(fig_hist, use_container_width=True)
        st.markdown(f"**Outliers**: {outlier_count} points with residual > 2σ")

    with tab_why:
        top_features = []
        feature_values = ordered[0]
        if hasattr(model, "coef_"):
            coefs = np.ravel(getattr(model, "coef_"))
            for feat, val, coef in zip(feature_columns, feature_values, coefs):
                top_features.append((feat, float(abs(coef) * abs(val)), float(coef), float(val)))
            top_features.sort(key=lambda x: x[1], reverse=True)
            top_features = top_features[:5]
        elif hasattr(model, "feature_importances_"):
            imps = np.ravel(getattr(model, "feature_importances_"))
            for feat, val, imp in zip(feature_columns, feature_values, imps):
                top_features.append((feat, float(imp), float(imp), float(val)))
            top_features.sort(key=lambda x: x[1], reverse=True)
            top_features = top_features[:5]
        else:
            top_features = [(feat, float(abs(val)), float(val), float(val)) for feat, val in zip(feature_columns, feature_values)][:5]

        st.markdown("**Top feature influences for this input**")
        cards = []
        for feat, score_v, direction, raw_v in top_features:
            sign = "+" if direction >= 0 else "-"
            cards.append((feat, score_v, f"{sign}{abs(direction):.4f}", raw_v))
        for feat, score_v, direction_label, raw_v in cards:
            st.markdown(
                f"<div class='metric-card' style='margin-bottom:.55rem;'><div class='metric-label'>{feat}</div><div class='metric-value' style='font-size:1.05rem;'>{direction_label}</div><div class='small-note'>impact {score_v:.4f} · value {raw_v:.4f}</div></div>",
                unsafe_allow_html=True,
            )
        st.markdown("**Segment view**")
        st.markdown(f"<div class='metric-card' style='margin-bottom:.55rem;'><div class='metric-label'>MAE by sex</div><div class='metric-value' style='font-size:1.1rem;'>male {abs_err_by_sex['male']:.2f} / female {abs_err_by_sex['female']:.2f}</div></div>", unsafe_allow_html=True)
        smoker_yes = float(abs_err_by_smoker.get("yes", np.nan))
        smoker_no = float(abs_err_by_smoker.get("no", np.nan))
        st.markdown(f"<div class='metric-card' style='margin-bottom:.55rem;'><div class='metric-label'>MAE by smoker</div><div class='metric-value' style='font-size:1.1rem;'>yes {smoker_yes:.2f} / no {smoker_no:.2f}</div></div>", unsafe_allow_html=True)
        region_summary = ", ".join([f"{k} {v:.2f}" for k, v in abs_err_by_region.items()]) or "n/a"
        st.markdown(f"<div class='metric-card'><div class='metric-label'>MAE by region</div><div class='metric-value' style='font-size:1.1rem;'>{region_summary}</div></div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
