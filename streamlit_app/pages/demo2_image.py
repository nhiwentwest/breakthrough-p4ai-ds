import json
import os
import random
from pathlib import Path

import gdown
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset, DatasetDict, load_from_disk
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
section[data-testid="stSidebar"] {{ display:none !important; }}
div[data-testid="collapsedControl"] {{ display:none !important; }}
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

    def forward(self, x, return_attn=False):
        h = x
        x_norm = self.norm1(x)
        if return_attn:
            x_attn, attn = self.attn(x_norm, x_norm, x_norm, need_weights=True, average_attn_weights=False)
        else:
            x_attn, attn = self.attn(x_norm, x_norm, x_norm, need_weights=False), None
        if isinstance(x_attn, tuple):
            x_attn = x_attn[0]
        x = x_attn + h
        h = x
        x = self.mlp(self.norm2(x))
        x = x + h
        if return_attn:
            return x, attn
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

    def forward(self, x, return_maps=False):
        x = self.cnn_front(x)
        inc = self.inception(x)
        x = self.patch_embed(inc)
        h_t, w_t = x.shape[-2], x.shape[-1]
        x = x.flatten(2).transpose(1, 2)
        self._build_pos_embed_if_needed(x.shape[1], x.shape[2], x.device)
        x = self.pos_drop(x + self.pos_embed)

        last_attn = None
        for i, blk in enumerate(self.transformer):
            if return_maps and i == len(self.transformer) - 1:
                x, last_attn = blk(x, return_attn=True)
            else:
                x = blk(x)

        x = self.norm(x)
        x = x.mean(dim=1)
        logits = self.head(x)

        if return_maps:
            return logits, inc, last_attn, (h_t, w_t)
        return logits


# =========================
# Paths + model loading
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[2]

CHECKPOINT_CANDIDATES = [
    PROJECT_ROOT / "assign2-ml" / "outputs_hybrid" / "best_hybrid_cnn_vit.pt",
    PROJECT_ROOT / "outputs_hybrid" / "best_hybrid_cnn_vit.pt",
    PROJECT_ROOT / "streamlit_app" / "checkpoints" / "best_hybrid_cnn_vit.pt",
]

MAPPING_CANDIDATES = [
    PROJECT_ROOT / "assign2-ml" / "outputs_hybrid" / "label_mapping.json",
    PROJECT_ROOT / "outputs_hybrid" / "label_mapping.json",
    PROJECT_ROOT / "streamlit_app" / "checkpoints" / "label_mapping.json",
]

# Google Drive assets
DRIVE_CHECKPOINT_FILE_ID = "1V5rcx3EsIAUK5-TNr98pIMfkXnTiOkcu"
DRIVE_LABEL_MAP_FILE_ID = "13tGhOSCdiQi2MTwEqR4TPCnvZav1n2EE"
DRIVE_DATASET_FOLDER_URL = "https://drive.google.com/drive/folders/1d5xkpBh-Rzeuj8dN7bNsPATrYsP-38ap?usp=sharing"


def _pick_existing(paths):
    for p in paths:
        if p.exists():
            return p
    return None


def ensure_checkpoint_from_drive():
    target_dir = PROJECT_ROOT / "streamlit_app" / "checkpoints"
    target_dir.mkdir(parents=True, exist_ok=True)
    target_ckpt = target_dir / "best_hybrid_cnn_vit.pt"

    if target_ckpt.exists():
        return target_ckpt

    url = f"https://drive.google.com/uc?id={DRIVE_CHECKPOINT_FILE_ID}"
    gdown.download(url, str(target_ckpt), quiet=False)

    if not target_ckpt.exists() or target_ckpt.stat().st_size == 0:
        raise FileNotFoundError("Downloaded checkpoint is missing or empty.")

    return target_ckpt


def ensure_label_mapping_from_drive():
    target_dir = PROJECT_ROOT / "streamlit_app" / "checkpoints"
    target_dir.mkdir(parents=True, exist_ok=True)
    target_map = target_dir / "label_mapping.json"

    if target_map.exists():
        return target_map

    url = f"https://drive.google.com/uc?id={DRIVE_LABEL_MAP_FILE_ID}"
    gdown.download(url, str(target_map), quiet=False)

    if not target_map.exists() or target_map.stat().st_size == 0:
        raise FileNotFoundError("Downloaded label_mapping.json is missing or empty.")

    return target_map


def load_model_and_labels():
    ckpt_path = _pick_existing(CHECKPOINT_CANDIDATES)
    map_path = _pick_existing(MAPPING_CANDIDATES)

    if ckpt_path is None:
        ckpt_path = ensure_checkpoint_from_drive()

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
        raise RuntimeError(
            "Cannot load label mapping from checkpoint. "
            "Please include 'label2id' and 'id2label' in checkpoint OR provide label_mapping.json."
        )

    model = HybridCNNViT(
        num_classes=len(id2label),
        embed_dim=cfg.get("embed_dim", 256),
        num_heads=cfg.get("num_heads", 8),
        depth=cfg.get("depth", 4),
        mlp_ratio=cfg.get("mlp_ratio", 4.0),
        dropout=cfg.get("dropout", 0.1),
    ).to(device)

    state_dict = ckpt["model_state_dict"]
    if "pos_embed" in state_dict:
        state_dict = {k: v for k, v in state_dict.items() if k != "pos_embed"}

    load_msg = model.load_state_dict(state_dict, strict=False)
    if len(load_msg.unexpected_keys) > 0:
        st.warning(f"Unexpected keys ignored: {load_msg.unexpected_keys}")
    if len(load_msg.missing_keys) > 0:
        st.warning(f"Missing keys initialized by model: {load_msg.missing_keys}")

    model.eval()

    return model, id2label, device, str(ckpt_path)


def preprocess_image(img: Image.Image):
    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return tfm(img.convert("RGB")).unsqueeze(0)


def find_dataset_dict_root(base_dir: str):
    for r, _d, files in os.walk(base_dir):
        if "dataset_dict.json" in files:
            return r
    return None


def ensure_dataset_folder_from_drive():
    target_dir = PROJECT_ROOT / "streamlit_app" / "data_cache" / "processed_rice_244"
    target_dir.mkdir(parents=True, exist_ok=True)

    existing = find_dataset_dict_root(str(target_dir))
    if existing is not None:
        return existing

    gdown.download_folder(url=DRIVE_DATASET_FOLDER_URL, output=str(target_dir), quiet=False, use_cookies=False)

    found = find_dataset_dict_root(str(target_dir))
    if found is None:
        raise FileNotFoundError(
            "Downloaded Drive folder but no dataset_dict.json found. "
            "Please verify the folder contains a HuggingFace load_from_disk dataset."
        )
    return found


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

    # strict mode: if Kaggle mount not found -> must load from provided Drive folder
    return ensure_dataset_folder_from_drive()


@st.cache_resource(show_spinner=False)
def load_sample_source():
    p = resolve_kaggle_processed_dataset_path()
    if p is None:
        return None, None
    ds = load_from_disk(p)

    # normalize return type to DatasetDict-like behavior
    if isinstance(ds, Dataset):
        ds = DatasetDict({"train": ds})

    return ds, p


def _choose_sample_split(ds):
    # preferred split order
    for k in ["test", "validation", "train", "healthy", "pests", "diseases", "nutrition"]:
        if k in ds and len(ds[k]) > 0:
            return k

    # any non-empty split
    for k in ds.keys():
        if len(ds[k]) > 0:
            return k

    return None


def _extract_label(ex):
    if "label" in ex:
        return ex["label"]
    if "labels" in ex:
        return ex["labels"]
    return None


def get_random_sample_image():
    ds, ds_path = load_sample_source()
    if ds is None:
        raise RuntimeError("dataset path not found")

    split_name = _choose_sample_split(ds)
    if split_name is None:
        raise RuntimeError(f"dataset loaded but all splits are empty: {list(ds.keys())}")

    idx = random.randint(0, len(ds[split_name]) - 1)
    ex = ds[split_name][idx]

    if "image" not in ex:
        raise RuntimeError(f"split '{split_name}' does not contain 'image' column. columns={list(ex.keys())}")

    img = ex["image"].convert("RGB")
    label = _extract_label(ex)
    meta = {"split": split_name, "index": idx, "dataset_path": ds_path, "columns": list(ex.keys())}
    return img, label, meta


def _to_uint8(img_pil: Image.Image):
    return np.array(img_pil.convert("RGB"), dtype=np.uint8)


def _norm01(arr):
    arr = arr.astype(np.float32)
    mn, mx = arr.min(), arr.max()
    return (arr - mn) / (mx - mn + 1e-8)


def _apply_heatmap_overlay(base_rgb_uint8, heatmap_01, alpha=0.45):
    heat = _norm01(heatmap_01)
    heat_img = Image.fromarray((heat * 255).astype(np.uint8)).resize((base_rgb_uint8.shape[1], base_rgb_uint8.shape[0]), Image.BILINEAR)
    heat_np = np.array(heat_img, dtype=np.float32) / 255.0

    # simple jet-like RGB without matplotlib dependency
    r = np.clip(1.5 * heat_np - 0.5, 0, 1)
    g = np.clip(1.5 - np.abs(2 * heat_np - 1.0), 0, 1)
    b = np.clip(1.5 * (1 - heat_np) - 0.5, 0, 1)
    color = np.stack([r, g, b], axis=-1)
    color = (color * 255.0).astype(np.uint8)

    overlay = (alpha * color + (1 - alpha) * base_rgb_uint8).astype(np.uint8)
    return overlay


def predict_with_explanations(model, id2label, device, img_pil, k=5):
    x = preprocess_image(img_pil).to(device)

    # hooks for Grad-CAM at inception output
    cache = {}
    def fwd_hook(_m, _i, o):
        cache["act"] = o
    def bwd_hook(_m, _gi, go):
        cache["grad"] = go[0]

    h1 = model.inception.register_forward_hook(fwd_hook)
    h2 = model.inception.register_full_backward_hook(bwd_hook)

    logits, _inc, last_attn, token_hw = model(x, return_maps=True)
    probs = torch.softmax(logits, dim=1)[0]
    top_vals, top_idx = torch.topk(probs, k=min(k, probs.shape[0]))
    pred_idx = int(top_idx[0].item())

    model.zero_grad(set_to_none=True)
    logits[0, pred_idx].backward()

    h1.remove(); h2.remove()

    # Grad-CAM
    act = cache["act"][0]        # (C,H,W)
    grad = cache["grad"][0]      # (C,H,W)
    w = grad.mean(dim=(1, 2), keepdim=True)
    cam = F.relu((w * act).sum(dim=0)).detach().cpu().numpy()
    cam = _norm01(cam)

    # Attention map (last block)
    attn_map = None
    if last_attn is not None:
        # (B, heads, N, N)
        a = last_attn[0].mean(dim=0)   # (N,N)
        a = a.mean(dim=0)              # (N,)
        h_t, w_t = token_hw
        attn_map = a.view(h_t, w_t).detach().cpu().numpy()
        attn_map = _norm01(attn_map)

    base = _to_uint8(img_pil)
    gradcam_overlay = _apply_heatmap_overlay(base, cam, alpha=0.45)
    attention_overlay = _apply_heatmap_overlay(base, attn_map if attn_map is not None else np.zeros((8,8), dtype=np.float32), alpha=0.45)

    rows = [(id2label[int(i)], float(p)) for p, i in zip(top_vals.detach().cpu().numpy(), top_idx.detach().cpu().numpy())]
    return rows, gradcam_overlay, attention_overlay


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

    mode = st.radio("Input mode", ["Upload image", "Drive random sample"], horizontal=True)

    image = None
    true_label = None

    if mode == "Upload image":
        up = st.file_uploader("Upload rice leaf image", type=["png", "jpg", "jpeg", "webp"])
        if up:
            image = Image.open(up).convert("RGB")
            st.image(image, caption="Input image", use_container_width=True)
    else:
        if st.button("Load random sample from Drive", use_container_width=True):
            try:
                sample_img, sample_label, sample_meta = get_random_sample_image()
                st.session_state["sample_img"] = sample_img
                st.session_state["sample_label"] = sample_label
                st.session_state["sample_meta"] = sample_meta
            except Exception as e:
                st.error(f"Drive sample error: {e}")

        if "sample_img" in st.session_state:
            image = st.session_state["sample_img"]
            true_label = st.session_state.get("sample_label")
            meta = st.session_state.get("sample_meta", {})
            st.image(image, caption=f"Drive sample: {meta.get('split', '?')}[{meta.get('index', '?')}]", use_container_width=True)
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
            topk, gradcam_overlay, attention_overlay = predict_with_explanations(model, id2label, device, image, k=5)
            top_label, top_prob = topk[0]

            st.metric("Predicted class", top_label)
            st.progress(float(top_prob))
            st.write(f"Confidence: **{top_prob:.2%}**")

            st.markdown("---")
            st.markdown("**Top-5 predictions**")
            for label, prob in topk:
                st.write(f"- {label}: {prob:.2%}")

            st.markdown("---")
            st.markdown("**Visual Explanations**")
            c1, c2 = st.columns(2)
            with c1:
                st.image(gradcam_overlay, caption="Grad-CAM (CNN focus)", use_container_width=True)
            with c2:
                st.image(attention_overlay, caption="Attention map (Transformer focus)", use_container_width=True)
    else:
        st.info("Upload image and click Predict.")

    st.markdown("</div>", unsafe_allow_html=True)
