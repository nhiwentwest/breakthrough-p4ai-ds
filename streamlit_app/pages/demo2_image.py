import json
import os
import random
from pathlib import Path

import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from datasets import load_from_disk
from PIL import Image
from torchvision import models, transforms

st.set_page_config(page_title="Demo 2 · Image Classification", page_icon="🖼️", layout="wide")

BG = "#F7F3EB"
CARD = "#EFE8DC"
TEXT = "#111111"
ACC = "#B42318"
MUT = "#6B6560"
BOR = "#D4C9B8"

st.markdown(
    f"""
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
</style>
""",
    unsafe_allow_html=True,
)


# =========================
# Model definition (same family as training script)
# =========================
class InceptionBlock(nn.Module):
    def __init__(self, in_ch, b1=64, b3=64, b5=32, bp=32):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_ch, b1, kernel_size=1, bias=False),
            nn.BatchNorm2d(b1),
            nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_ch, b3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(b3),
            nn.ReLU(inplace=True),
        )
        self.branch5 = nn.Sequential(
            nn.Conv2d(in_ch, b5, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(b5),
            nn.ReLU(inplace=True),
        )
        self.branchp = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_ch, bp, kernel_size=1, bias=False),
            nn.BatchNorm2d(bp),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return torch.cat([
            self.branch1(x),
            self.branch3(x),
            self.branch5(x),
            self.branchp(x),
        ], dim=1)


class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerEncoderBlock(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(self, x):
        h = x
        x, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False)
        x = x + h
        h = x
        x = self.mlp(self.norm2(x))
        x = x + h
        return x


class HybridCNNViT(nn.Module):
    def __init__(self, num_classes=21, embed_dim=256, num_heads=8, depth=4, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        self.cnn_front = nn.Sequential(*list(vgg.features.children())[:10])

        self.inception = InceptionBlock(in_ch=128, b1=64, b3=64, b5=32, bp=32)
        self.patch_embed = nn.Conv2d(192, embed_dim, kernel_size=4, stride=4)
        self.pos_drop = nn.Dropout(dropout)
        self.transformer = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )
        self.pos_embed = None

    def _build_pos_embed_if_needed(self, n_tokens, dim, device):
        if self.pos_embed is None or self.pos_embed.shape[1] != n_tokens or self.pos_embed.shape[2] != dim:
            pe = torch.zeros(1, n_tokens, dim, device=device)
            nn.init.trunc_normal_(pe, std=0.02)
            self.pos_embed = nn.Parameter(pe)

    def forward(self, x):
        x = self.cnn_front(x)
        x = self.inception(x)
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        self._build_pos_embed_if_needed(x.shape[1], x.shape[2], x.device)
        x = self.pos_drop(x + self.pos_embed)
        for blk in self.transformer:
            x = blk(x)
        x = self.norm(x)
        x = x.mean(dim=1)
        return self.head(x)


# =========================
# Paths + model loading
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[2]

CHECKPOINT_CANDIDATES = [
    PROJECT_ROOT / "assign2-ml" / "outputs_hybrid" / "best_hybrid_cnn_vit.pt",
    PROJECT_ROOT / "outputs_hybrid" / "best_hybrid_cnn_vit.pt",
]

MAPPING_CANDIDATES = [
    PROJECT_ROOT / "assign2-ml" / "outputs_hybrid" / "label_mapping.json",
    PROJECT_ROOT / "outputs_hybrid" / "label_mapping.json",
]


def _pick_existing(paths):
    for p in paths:
        if p.exists():
            return p
    return None


@st.cache_resource(show_spinner=False)
def load_model_and_labels():
    ckpt_path = _pick_existing(CHECKPOINT_CANDIDATES)
    map_path = _pick_existing(MAPPING_CANDIDATES)

    if ckpt_path is None:
        raise FileNotFoundError(
            "Checkpoint not found. Expected at assign2-ml/outputs_hybrid/best_hybrid_cnn_vit.pt"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device)

    cfg = ckpt.get("cfg", {})
    label2id = ckpt.get("label2id", None)
    id2label = ckpt.get("id2label", None)

    if (label2id is None or id2label is None) and map_path is not None:
        with open(map_path, "r", encoding="utf-8") as f:
            mp = json.load(f)
        label2id = mp.get("label2id", {})
        id2label = {int(k): v for k, v in mp.get("id2label", {}).items()}

    if id2label is None:
        raise RuntimeError("Cannot load label mapping from checkpoint or label_mapping.json")

    model = HybridCNNViT(
        num_classes=len(id2label),
        embed_dim=cfg.get("embed_dim", 256),
        num_heads=cfg.get("num_heads", 8),
        depth=cfg.get("depth", 4),
        mlp_ratio=cfg.get("mlp_ratio", 4.0),
        dropout=cfg.get("dropout", 0.1),
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    return model, id2label, device, str(ckpt_path)


def preprocess_image(img: Image.Image):
    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return tfm(img.convert("RGB")).unsqueeze(0)


def resolve_kaggle_processed_dataset_path():
    candidates = [
        "/kaggle/input/processed-rice-244",
        "/kaggle/input/processed-rice-244/processed_rice_224",
        "/kaggle/input/processed_rice_224",
        "/kaggle/input/processed_rice_224/processed_rice_224",
        "/kaggle/input/datasets/bocon66/processed-rice-244",
        "/kaggle/input/datasets/bocon66/processed-rice-244/processed_rice_224",
        "processed_rice_224",
    ]

    for p in candidates:
        if os.path.exists(os.path.join(p, "dataset_dict.json")):
            return p

    # recursive scan in kaggle input
    root = "/kaggle/input"
    if os.path.isdir(root):
        for r, _d, files in os.walk(root):
            if "dataset_dict.json" in files:
                return r

    return None


@st.cache_resource(show_spinner=False)
def load_sample_source():
    p = resolve_kaggle_processed_dataset_path()
    if p is None:
        return None, None
    ds = load_from_disk(p)
    return ds, p


def get_random_sample_image():
    ds, ds_path = load_sample_source()
    if ds is None:
        return None, None, None

    split_name = "test" if "test" in ds else ("validation" if "validation" in ds else ("train" if "train" in ds else None))
    if split_name is None or len(ds[split_name]) == 0:
        return None, None, None

    idx = random.randint(0, len(ds[split_name]) - 1)
    ex = ds[split_name][idx]
    img = ex["image"].convert("RGB")
    label = ex.get("label", None)
    meta = {"split": split_name, "index": idx, "dataset_path": ds_path}
    return img, label, meta


@torch.no_grad()
def predict_topk(model, id2label, device, img_tensor, k=5):
    img_tensor = img_tensor.to(device)
    logits = model(img_tensor)
    probs = torch.softmax(logits, dim=1)[0]
    top_vals, top_idx = torch.topk(probs, k=min(k, probs.shape[0]))
    rows = []
    for p, i in zip(top_vals.cpu().numpy(), top_idx.cpu().numpy()):
        rows.append((id2label[int(i)], float(p)))
    return rows


# =========================
# UI
# =========================
st.markdown("<p class='hero'>Demo 2 · Image Classification</p>", unsafe_allow_html=True)
st.markdown(
    "<p class='sub'>Rice disease classifier (Hybrid CNN–ViT). Upload one image to predict class with checkpoint.</p>",
    unsafe_allow_html=True,
)

left, right = st.columns([1.2, 1])

with left:
    st.markdown("<div class='bento'>", unsafe_allow_html=True)
    st.markdown("<div class='section'>Model</div>", unsafe_allow_html=True)

    try:
        model, id2label, device, ckpt_used = load_model_and_labels()
        st.success(f"Loaded checkpoint: `{ckpt_used}`")
        st.markdown(f"<div class='small-note'>Device: <b>{device}</b> · Classes: <b>{len(id2label)}</b></div>", unsafe_allow_html=True)
        model_ready = True
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        st.info("Put checkpoint at `assign2-ml/outputs_hybrid/best_hybrid_cnn_vit.pt` and label_mapping.json in same folder.")
        model_ready = False

    st.markdown("<div class='section'>Image Input</div>", unsafe_allow_html=True)
    mode = st.radio("Input mode", ["Upload image", "Kaggle sample (bocon66/processed-rice-244)"], horizontal=True)

    image = None
    true_label = None

    if mode == "Upload image":
        up = st.file_uploader("Upload rice leaf image", type=["png", "jpg", "jpeg", "webp"])
        if up:
            image = Image.open(up).convert("RGB")
            st.image(image, caption="Input image", use_container_width=True)
    else:
        if st.button("Load random sample", use_container_width=True):
            sample_img, sample_label, sample_meta = get_random_sample_image()
            if sample_img is None:
                st.warning("Cannot find mounted Kaggle dataset. Please attach bocon66/processed-rice-244.")
            else:
                st.session_state["sample_img"] = sample_img
                st.session_state["sample_label"] = sample_label
                st.session_state["sample_meta"] = sample_meta

        if "sample_img" in st.session_state:
            image = st.session_state["sample_img"]
            true_label = st.session_state.get("sample_label")
            meta = st.session_state.get("sample_meta", {})
            st.image(image, caption=f"Sample from {meta.get('split', '?')}[{meta.get('index', '?')}]", use_container_width=True)
            st.caption(f"Dataset path: {meta.get('dataset_path', '?')}")
            if true_label is not None:
                st.caption(f"Ground truth label: {true_label}")

    pred_btn = st.button("Predict", use_container_width=True, disabled=not model_ready)
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown("<div class='bento'>", unsafe_allow_html=True)
    st.markdown("<div class='section'>Prediction Output</div>", unsafe_allow_html=True)

    if pred_btn:
        if image is None:
            st.warning("Please upload an image first.")
        else:
            x = preprocess_image(image)
            topk = predict_topk(model, id2label, device, x, k=5)
            top_label, top_prob = topk[0]

            st.metric("Predicted class", top_label)
            st.progress(float(top_prob))
            st.write(f"Confidence: **{top_prob:.2%}**")

            st.markdown("---")
            st.markdown("**Top-5 predictions**")
            for label, prob in topk:
                st.write(f"- {label}: {prob:.2%}")
    else:
        st.info("Upload image and click Predict.")

    st.markdown("</div>", unsafe_allow_html=True)
