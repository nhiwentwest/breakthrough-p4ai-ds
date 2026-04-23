from pathlib import Path
import os
import warnings

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
warnings.filterwarnings("ignore", message=r".*Accessing `__path__`.*", category=UserWarning)
warnings.filterwarnings("ignore", message=r".*alias will be removed in future versions.*", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import gdown
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from datasets import load_dataset
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModel, AutoTokenizer

st.set_page_config(page_title="Demo 2 · Text Classification", page_icon="📝", layout="wide")

BG = "#F7F3EB"
CARD = "#EFE8DC"
TEXT = "#111111"
ACC = "#B42318"
MUT = "#6B6560"
BOR = "#D4C9B8"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_DIR = PROJECT_ROOT / "streamlit_app" / "checkpoints"
TEXT_CHECKPOINT_ID = "1IfVsAt5c9cHgaNiM3Q-y6z8XCTwDBsJw"
MODEL_NAME = "bert-base-uncased"
LABELS = {0: "Bearish", 1: "Bullish", 2: "Neutral"}

st.markdown(
    f"""
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
.metric-row {{ display:grid; grid-template-columns: repeat(2, minmax(0,1fr)); gap:.7rem; margin-top:.85rem; }}
.metric-card {{ background:rgba(255,255,255,.65); border:1px solid #dacdbd; border-radius:16px; padding:.8rem .9rem .75rem; box-shadow:0 4px 18px rgba(17,17,17,.05); }}
.metric-label {{ font-size:.63rem; letter-spacing:.1em; text-transform:uppercase; color:{MUT}; font-weight:700; }}
.metric-value {{ font-family:'Playfair Display',serif; font-size:1.55rem; font-weight:900; line-height:1.05; margin-top:.35rem; color:{TEXT}; }}
.small-note {{ color:{MUT}; font-size:0.82rem; }}
.code {{ background:#1C1A16; color:#D4C9BB; border-radius:14px; padding:14px 16px; font-family:'Menlo','Consolas',monospace; font-size:.72rem; line-height:1.7; overflow:auto; }}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown("<p class='hero'>Demo 2 · Text Classification</p>", unsafe_allow_html=True)
st.markdown("<p class='sub'>Single-model demo using BERT fine-tuning on the Twitter financial news sentiment dataset.</p>", unsafe_allow_html=True)


def _download_checkpoint() -> Path:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = CHECKPOINT_DIR / "text_bert_checkpoint.bin"
    if checkpoint_path.exists() and checkpoint_path.stat().st_size > 0:
        return checkpoint_path
    url = f"https://drive.google.com/uc?id={TEXT_CHECKPOINT_ID}"
    gdown.download(url, str(checkpoint_path), quiet=False)
    return checkpoint_path


class TwitterSentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=64):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            str(self.texts[idx]),
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(int(self.labels[idx]), dtype=torch.long),
        }


class BERTMeanPoolingClassifier(nn.Module):
    def __init__(self, model_name: str, num_labels: int = 3):
        super().__init__()
        config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name, config=config)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        last_hidden = out.last_hidden_state
        mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        pooled = (last_hidden * mask).sum(dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)
        pooled = self.dropout(pooled)
        return self.classifier(pooled)


@st.cache_resource(show_spinner=True)
def load_text_model():
    checkpoint_path = _download_checkpoint()
    model = BERTMeanPoolingClassifier(MODEL_NAME, num_labels=3)
    state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict):
        cleaned = {k.replace("module.", ""): v for k, v in state.items()}
        model.load_state_dict(cleaned, strict=False)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    return model, tokenizer, str(checkpoint_path)


@st.cache_data(show_spinner=True)
def load_sample_dataset():
    ds = load_dataset("zeroshot/twitter-financial-news-sentiment")
    split_name = "test" if "test" in ds else ("validation" if "validation" in ds else list(ds.keys())[-1])
    return ds[split_name].to_pandas().copy()


def evaluate_model(model, tokenizer, df: pd.DataFrame, sample_size: int = 64):
    sample = df.sample(n=min(sample_size, len(df)), random_state=42).reset_index(drop=True)
    dataset = TwitterSentimentDataset(sample["text"], sample["label"], tokenizer, max_length=64)
    loader = DataLoader(dataset, batch_size=16)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            logits = model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="weighted", zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    return sample, acc, prec, rec, f1, cm


model, tokenizer, checkpoint_path = load_text_model()
df = load_sample_dataset()

st.markdown("<div class='editor-shell'>", unsafe_allow_html=True)
left, right = st.columns([1.15, 1])

with left:
    st.markdown("<div class='bento'>", unsafe_allow_html=True)
    st.markdown("<div class='section'>Model</div>", unsafe_allow_html=True)
    st.markdown("<div class='model-chip'>BERT fine-tuning</div>", unsafe_allow_html=True)
    st.caption("Checkpoint")
    st.code(checkpoint_path, language="text")

    st.markdown("<div class='section'>Input Text</div>", unsafe_allow_html=True)
    sample_options = df.sample(6, random_state=7).reset_index(drop=True)
    sample_labels = [f"sample {i} · {LABELS[int(row['label'])]}" for i, row in sample_options.iterrows()]
    selected = st.selectbox("Choose sample", sample_labels, label_visibility="collapsed")
    idx = sample_labels.index(selected)
    text = st.text_area("Text", value=str(sample_options.iloc[idx]["text"]), height=220, label_visibility="collapsed")
    pred_btn = st.button("Predict", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown("<div class='bento'>", unsafe_allow_html=True)
    st.markdown("<div class='section'>Evaluation</div>", unsafe_allow_html=True)
    sample, acc, prec, rec, f1, cm = evaluate_model(model, tokenizer, df)

    st.markdown("<div class='metric-row'>", unsafe_allow_html=True)
    for label, value in [("Accuracy", acc), ("Precision", prec), ("Recall", rec), ("F1", f1)]:
        st.markdown(
            f"<div class='metric-card'><div class='metric-label'>{label}</div><div class='metric-value'>{value:.4f}</div></div>",
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    if pred_btn:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        encoded = tokenizer(text, truncation=True, padding="max_length", max_length=64, return_tensors="pt")
        with torch.no_grad():
            logits = model.to(device)(encoded["input_ids"].to(device), encoded["attention_mask"].to(device))
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred_id = int(np.argmax(probs))
        st.markdown(f"<div class='model-chip'>Predicted label: {LABELS[pred_id]}</div>", unsafe_allow_html=True)
        st.write("Class probabilities")
        for i, p in enumerate(probs):
            st.markdown(
                f"<div style='margin-bottom:0.6rem;'><div style='display:flex; justify-content:space-between; font-size:0.85rem; margin-bottom:0.25rem; font-weight:600;'><span>{LABELS[i]}</span><span>{p:.1%}</span></div><div style='width:100%; background:rgba(212,201,184,0.4); border-radius:99px; height:8px; overflow:hidden;'><div style='width:{p*100}%; background:#B42318; height:100%; border-radius:99px;'></div></div></div>",
                unsafe_allow_html=True
            )
    else:
        st.info("Choose a sample or type text, then click Predict.")
        st.caption("The app evaluates the loaded checkpoint on a sample of the Hugging Face test split.")

    st.caption(f"Confusion matrix (sample): {cm.tolist()}")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    "<div class='bento' style='margin-top:1rem'><div class='section'>Dataset</div><div class='small-note'>Loaded with <code>from datasets import load_dataset</code> and <code>load_dataset(\"zeroshot/twitter-financial-news-sentiment\")</code>. The demo uses the test split for evaluation.</div></div>",
    unsafe_allow_html=True,
)

st.markdown("</div>", unsafe_allow_html=True)
