import json
import os
import random
import re
import importlib.util
from pathlib import Path

import gdown
import joblib
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
.kpi-grid {{ display:grid; grid-template-columns: repeat(3, minmax(0,1fr)); gap:.65rem; margin:.8rem 0 1rem; }}
.kpi-card {{ background:{CARD}; border:1px solid {BOR}; border-radius:12px; padding:.65rem .8rem; }}
.kpi-lbl {{ font-size:.62rem; letter-spacing:.08em; text-transform:uppercase; color:{MUT}; }}
.kpi-val {{ font-weight:700; font-size:1rem; margin-top:.15rem; }}
.editor-bar {{ display:flex; align-items:center; gap:.35rem; margin-bottom:.55rem; }}
.dot {{ width:.55rem; height:.55rem; border-radius:999px; display:inline-block; }}
.dot-r {{ background:#e57373; }} .dot-y {{ background:#ffca28; }} .dot-g {{ background:#66bb6a; }}
.demo-label {{ color:#111111 !important; font-weight:700; background:#f3eadb; border:1px solid #d7c7b2; padding:.35rem .55rem; border-radius:8px; display:inline-block; margin-top:.35rem; }}
.demo-label-light {{ color:#f7f3eb !important; font-weight:700; background:#2b2b2b; border:1px solid #555; padding:.35rem .55rem; border-radius:8px; display:inline-block; margin-top:.35rem; }}
@media (max-width: 900px) {{ .kpi-grid {{ grid-template-columns: 1fr; }} }}
</style>
""",
    unsafe_allow_html=True,
)


# =========================
# Model definition (same family as training script)
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_source_mblanet_class():
    model_path = PROJECT_ROOT / "assign2-ml" / "image" / "mblanet.py"
    spec = importlib.util.spec_from_file_location("mblanet_source", model_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load model source from {model_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MBLANet


MBLANet = _load_source_mblanet_class()


class ConvBNReLU(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class CNNScratch(nn.Module):
    def __init__(self, num_classes=21, dropout=0.3):
        super().__init__()
        # Match the training script: ResNet18 backbone trained from scratch.
        self.model = models.resnet18(weights=None)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x):
        return self.model(x)


def build_resnet50_classifier(
    num_classes: int,
    dropout: float = 0.3,
    pretrained: bool = True,
    freeze_backbone: bool = False,
):
    weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet50(weights=weights)
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_features, num_classes),
    )
    return model


# =========================
# Paths + model loading
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[2]

CHECKPOINT_CANDIDATES = {
    "MBLANet": [
        PROJECT_ROOT / "assign2-ml" / "outputs_mblanet" / "best_mblanet.pt",
        PROJECT_ROOT / "outputs_mblanet" / "best_mblanet.pt",
        PROJECT_ROOT / "streamlit_app" / "checkpoints" / "best_mblanet.pt",
    ],
    "CNN Scratch": [
        PROJECT_ROOT / "assign2-ml" / "outputs_cnn_scratch" / "best_cnn_scratch.pt",
        PROJECT_ROOT / "outputs_cnn_scratch" / "best_cnn_scratch.pt",
        PROJECT_ROOT / "streamlit_app" / "checkpoints" / "best_cnn_scratch.pt",
    ],
    "Pretrained CNN Frozen": [
        PROJECT_ROOT / "streamlit_app" / "checkpoints" / "resnet50_extractor.pt",
        PROJECT_ROOT / "streamlit_app" / "checkpoints" / "best_resnet50_model.pt",
        PROJECT_ROOT / "streamlit_app" / "checkpoints" / "best_resnet50_model.pth",
    ],
    "Pretrained CNN Fine-tuned": [
        PROJECT_ROOT / "streamlit_app" / "checkpoints" / "best_resnet50_finetuned_model.pt",
        PROJECT_ROOT / "streamlit_app" / "checkpoints" / "best_resnet50_finetuned_model.pth",
    ],
    "SVM + ResNet50": [
        PROJECT_ROOT / "streamlit_app" / "checkpoints" / "svm_model.joblib",
    ],
}

MAPPING_CANDIDATES = {
    "MBLANet": [
        PROJECT_ROOT / "assign2-ml" / "outputs_mblanet" / "label_mapping.json",
        PROJECT_ROOT / "outputs_mblanet" / "label_mapping.json",
        PROJECT_ROOT / "streamlit_app" / "checkpoints" / "label_mapping_mblanet.json",
    ],
    "CNN Scratch": [
        PROJECT_ROOT / "assign2-ml" / "outputs_cnn_scratch" / "label_mapping.json",
        PROJECT_ROOT / "outputs_cnn_scratch" / "label_mapping.json",
        PROJECT_ROOT / "streamlit_app" / "checkpoints" / "label_mapping_cnn_scratch.json",
    ],
    "Pretrained CNN Frozen": [
        PROJECT_ROOT / "streamlit_app" / "checkpoints" / "best_resnet50_model_labels.json",
        PROJECT_ROOT / "streamlit_app" / "checkpoints" / "resnet50_extractor_labels.json",
    ],
    "Pretrained CNN Fine-tuned": [
        PROJECT_ROOT / "streamlit_app" / "checkpoints" / "best_resnet50_finetuned_labels.json",
    ],
    "SVM + ResNet50": [
        PROJECT_ROOT / "streamlit_app" / "checkpoints" / "resnet50_label_mapping.json",
    ],
}

RSITMD_CLASSES = [
    'airport', 'bareland', 'baseballfield', 'beach', 'boat', 'bridge', 'center', 'church', 'commercial',
    'denseresidential', 'desert', 'farmland', 'forest', 'industrial', 'intersection', 'meadow',
    'mediumresidential', 'mountain', 'park', 'parking', 'plane', 'playground', 'pond', 'port',
    'railwaystation', 'resort', 'river', 'school', 'sparseresidential', 'square', 'stadium',
    'storagetanks', 'viaduct'
]

# Google Drive assets
MBLANET_CHECKPOINT_FILE_ID = "1JOHbgznyN358XsnUX4jGYAjQc_wsjKZI"
MBLANET_LABEL_MAP_FILE_ID = "13wXU29DAVfo0MWqHWTHSzRB5c-p3d9Wq"

CNN_SCRATCH_CHECKPOINT_FILE_ID = "1D6eAxGMvARoY3Nrt9nsgRxYX7mBAIKAw"
CNN_SCRATCH_LABEL_MAP_FILE_ID = "13wXU29DAVfo0MWqHWTHSzRB5c-p3d9Wq"
PRETRAINED_CNN_FROZEN_CHECKPOINT_FILE_ID = "1yFW5oxqwUcrQiEFei7_FeJGxanLj3l2b"
PRETRAINED_CNN_FROZEN_LABEL_MAP_FILE_ID = "1-APpqmy6mofO90bfGPeqcNoZCfjLHNAU"
PRETRAINED_CNN_FINETUNED_CHECKPOINT_FILE_ID = "1vIgcLba9ylYT7wNUVeQ7FLRBauvioi8_"
PRETRAINED_CNN_FINETUNED_LABEL_MAP_FILE_ID = "1cOLEUL0kULFGM0b0YuJA35-Oc1ohBenV"
SVM_JOBLIB_FILE_ID = "1IdUgQx5KeCUehWOtBIfeFPhjBXwd_AsY"
SVM_LABEL_MAP_FILE_ID = "1-APpqmy6mofO90bfGPeqcNoZCfjLHNAU"

DRIVE_DATASET_FOLDER_URL = "https://drive.google.com/drive/folders/1vmk07ZO_5hi6yBZQ15N0TfhZ2D9Y9-mv?usp=sharing"
FORCE_DRIVE_REFRESH = False


def ensure_checkpoint_from_drive(model_choice: str):
    target_dir = PROJECT_ROOT / "streamlit_app" / "checkpoints"
    target_dir.mkdir(parents=True, exist_ok=True)

    if model_choice == "MBLANet":
        target_ckpt = target_dir / "best_mblanet.pt"
        file_id = MBLANET_CHECKPOINT_FILE_ID
        if not target_ckpt.exists() or target_ckpt.stat().st_size == 0 or FORCE_DRIVE_REFRESH:
            if FORCE_DRIVE_REFRESH and target_ckpt.exists():
                target_ckpt.unlink()
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, str(target_ckpt), quiet=False)
    elif model_choice == "CNN Scratch":
        target_ckpt = target_dir / "best_cnn_scratch.pt"
        file_id = CNN_SCRATCH_CHECKPOINT_FILE_ID
        if not target_ckpt.exists() or target_ckpt.stat().st_size == 0 or FORCE_DRIVE_REFRESH:
            if FORCE_DRIVE_REFRESH and target_ckpt.exists():
                target_ckpt.unlink()
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, str(target_ckpt), quiet=False)
    elif model_choice == "Pretrained CNN Frozen":
        target_ckpt = target_dir / "resnet50_extractor.pt"
        file_id = PRETRAINED_CNN_FROZEN_CHECKPOINT_FILE_ID
        if not target_ckpt.exists() or target_ckpt.stat().st_size == 0 or FORCE_DRIVE_REFRESH:
            if FORCE_DRIVE_REFRESH and target_ckpt.exists():
                target_ckpt.unlink()
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, str(target_ckpt), quiet=False)
    elif model_choice == "Pretrained CNN Fine-tuned":
        target_ckpt = target_dir / "best_resnet50_finetuned_model.pt"
        file_id = PRETRAINED_CNN_FINETUNED_CHECKPOINT_FILE_ID
        if not target_ckpt.exists() or target_ckpt.stat().st_size == 0 or FORCE_DRIVE_REFRESH:
            if FORCE_DRIVE_REFRESH and target_ckpt.exists():
                target_ckpt.unlink()
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, str(target_ckpt), quiet=False)
    elif model_choice == "SVM + ResNet50":
        target_ckpt = target_dir / "svm_model.joblib"
        file_id = SVM_JOBLIB_FILE_ID
        if not target_ckpt.exists() or target_ckpt.stat().st_size == 0 or FORCE_DRIVE_REFRESH:
            if FORCE_DRIVE_REFRESH and target_ckpt.exists():
                target_ckpt.unlink()
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, str(target_ckpt), quiet=False)
    else:
        raise ValueError(f"Unknown model choice: {model_choice}")

    if not target_ckpt.exists() or target_ckpt.stat().st_size == 0:
        raise FileNotFoundError("Downloaded checkpoint is missing or empty.")

    return target_ckpt


def ensure_label_mapping_from_drive(model_choice: str):
    target_dir = PROJECT_ROOT / "streamlit_app" / "checkpoints"
    target_dir.mkdir(parents=True, exist_ok=True)

    if model_choice == "MBLANet":
        target_map = target_dir / "label_mapping_mblanet.json"
        file_id = MBLANET_LABEL_MAP_FILE_ID
        if file_id is None:
            return target_map
        if not target_map.exists() or target_map.stat().st_size == 0 or FORCE_DRIVE_REFRESH:
            if FORCE_DRIVE_REFRESH and target_map.exists():
                target_map.unlink()
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, str(target_map), quiet=False)
    elif model_choice == "CNN Scratch":
        target_map = target_dir / "label_mapping_cnn_scratch.json"
        file_id = CNN_SCRATCH_LABEL_MAP_FILE_ID
        if not target_map.exists() or target_map.stat().st_size == 0 or FORCE_DRIVE_REFRESH:
            if FORCE_DRIVE_REFRESH and target_map.exists():
                target_map.unlink()
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, str(target_map), quiet=False)
    elif model_choice == "Pretrained CNN Frozen":
        target_map = target_dir / "best_resnet50_model_labels.json"
        file_id = PRETRAINED_CNN_FROZEN_LABEL_MAP_FILE_ID
        if not target_map.exists() or target_map.stat().st_size == 0 or FORCE_DRIVE_REFRESH:
            if FORCE_DRIVE_REFRESH and target_map.exists():
                target_map.unlink()
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, str(target_map), quiet=False)
    elif model_choice == "Pretrained CNN Fine-tuned":
        target_map = target_dir / "best_resnet50_finetuned_labels.json"
        file_id = PRETRAINED_CNN_FINETUNED_LABEL_MAP_FILE_ID
        if not target_map.exists() or target_map.stat().st_size == 0 or FORCE_DRIVE_REFRESH:
            if FORCE_DRIVE_REFRESH and target_map.exists():
                target_map.unlink()
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, str(target_map), quiet=False)
    elif model_choice == "SVM + ResNet50":
        target_map = target_dir / "resnet50_label_mapping.json"
        file_id = SVM_LABEL_MAP_FILE_ID
        if not target_map.exists() or target_map.stat().st_size == 0 or FORCE_DRIVE_REFRESH:
            if FORCE_DRIVE_REFRESH and target_map.exists():
                target_map.unlink()
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, str(target_map), quiet=False)
    else:
        raise ValueError(f"Unknown model choice: {model_choice}")

    if not target_map.exists() or target_map.stat().st_size == 0:
        raise FileNotFoundError("Downloaded label_mapping.json is missing or empty.")

    return target_map


@st.cache_resource(show_spinner=True)
def get_checkpoint_and_mapping(model_choice: str):
    return str(ensure_checkpoint_from_drive(model_choice)), str(ensure_label_mapping_from_drive(model_choice))


@st.cache_resource(show_spinner=True)
def load_model_and_labels(model_choice: str, ckpt_path: str, map_path: str):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_choice == "SVM + ResNet50":
        svm = joblib.load(ckpt_path)
        with open(map_path, "r", encoding="utf-8") as f:
            mp = json.load(f)
        id2label = {int(k): v for k, v in mp.get("id2label", {}).items()}
        if not id2label or not all(isinstance(v, str) for v in id2label.values()):
            id2label = {i: name for i, name in enumerate(RSITMD_CLASSES)}
        return svm, id2label, None, str(ckpt_path)

    ckpt = torch.load(ckpt_path, map_location=device)

    cfg = ckpt.get("cfg", {})
    label2id = ckpt.get("label2id", None)
    id2label = ckpt.get("id2label", None)

    if (label2id is None or id2label is None) and map_path is not None:
        with open(map_path, "r", encoding="utf-8") as f:
            mp = json.load(f)
        label2id = mp.get("label2id", {})
        id2label = {int(k): v for k, v in mp.get("id2label", {}).items()}

    if id2label is None or all(isinstance(v, (int, np.integer)) or (isinstance(v, str) and v.isdigit()) for v in id2label.values()):
        id2label = {i: name for i, name in enumerate(RSITMD_CLASSES)}
        label2id = {name: i for i, name in enumerate(RSITMD_CLASSES)}

    if model_choice == "MBLANet":
        model = MBLANet(
            num_classes=len(id2label),
            pretrained=False,
        ).to(device)
        for module in model.modules():
            if module.__class__.__name__ == "LSAM":
                if not hasattr(module, "raw_attn"):
                    module.raw_attn = None
                if not hasattr(module, "att_map"):
                    module.att_map = None
                if not hasattr(module, "input_stats"):
                    module.input_stats = None
    elif model_choice == "Pretrained CNN Frozen":
        model = build_resnet50_classifier(
            num_classes=len(id2label),
            dropout=cfg.get("dropout", 0.3),
            pretrained=True,
            freeze_backbone=True,
            head_style="sequential",
        ).to(device)
    elif model_choice == "Pretrained CNN Fine-tuned":
        model = build_resnet50_classifier(
            num_classes=len(id2label),
            dropout=cfg.get("dropout", 0.3),
            pretrained=True,
            freeze_backbone=False,
            head_style="linear",
        ).to(device)
    else:
        model = CNNScratch(
            num_classes=len(id2label),
            dropout=cfg.get("dropout", 0.3),
        ).to(device)

    state_dict = ckpt.get("model_state_dict", ckpt)
    if not isinstance(state_dict, dict):
        raise TypeError(f"Unsupported checkpoint format: {type(ckpt)!r}")

    # Some checkpoints may be wrapped with "module." from DataParallel.
    # `CNNScratch`/`MBLANet` keep the real backbone under `self.model`, so
    # their checkpoints may also include a top-level `model.` prefix that must
    # be preserved. ResNet50 classifier checkpoints are expected to use the
    # plain ResNet keys (conv1, bn1, layer1... fc.0/fc if present).
    key_samples = list(state_dict.keys())
    if key_samples and all(k.startswith("module.") for k in key_samples):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    if model_choice in ("Pretrained CNN Frozen", "Pretrained CNN Fine-tuned"):
        if key_samples and all(k.startswith("model.") for k in key_samples):
            state_dict = {k.replace("model.", "", 1): v for k, v in state_dict.items()}

    try:
        load_msg = model.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        if model_choice == "Pretrained CNN Frozen":
            st.error(f"Frozen CNN checkpoint does not match frozen ResNet50 architecture: {e}")
        elif model_choice == "Pretrained CNN Fine-tuned":
            st.error(f"Fine-tuned CNN checkpoint does not match ResNet50 architecture: {e}")
        elif model_choice == "CNN Scratch":
            st.error(f"CNN Scratch checkpoint does not match ResNet18 scratch architecture: {e}")
        else:
            st.error(f"MBLANet checkpoint does not match expected architecture: {e}")
        raise
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


def preprocess_image_tensor(img: Image.Image):
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
    target_dir = PROJECT_ROOT / "streamlit_app" / "data_cache" / "drive_image_demo_v2"
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
    # strict: only use test split for demo random sample
    if "test" in ds and len(ds["test"]) > 0:
        return "test"
    return None


def _extract_label(ex, ds_split=None):
    raw = ex.get("label", ex.get("labels", None))
    if raw is None:
        return None
    # If the dataset split has ClassLabel features, convert int to str
    if ds_split is not None and hasattr(ds_split, "features") and "label" in ds_split.features:
        from datasets import ClassLabel
        feature = ds_split.features["label"]
        if isinstance(feature, ClassLabel) and isinstance(raw, (int, np.integer)):
            return feature.int2str(int(raw))
    return raw


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
    label = _extract_label(ex, ds[split_name])
    meta = {"split": split_name, "index": idx, "dataset_path": ds_path, "columns": list(ex.keys())}
    return img, label, meta


def parse_split_index(user_text: str):
    s = (user_text or "").strip().lower()
    m = re.fullmatch(r"([a-z_]+)\s*\[\s*(\d+)\s*\]", s)
    if not m:
        raise ValueError("Format must be like: test[7]")
    split_name = m.group(1)
    idx = int(m.group(2))
    return split_name, idx


def get_named_sample_image(spec_text: str):
    ds, ds_path = load_sample_source()
    if ds is None:
        raise RuntimeError("dataset path not found")

    split_name, idx = parse_split_index(spec_text)
    if split_name not in ds:
        raise ValueError(f"Split '{split_name}' not found. Available: {list(ds.keys())}")
    if idx < 0 or idx >= len(ds[split_name]):
        raise IndexError(f"Index out of range for split '{split_name}': 0..{len(ds[split_name]) - 1}")

    ex = ds[split_name][idx]
    if "image" not in ex:
        raise RuntimeError(f"split '{split_name}' does not contain 'image' column. columns={list(ex.keys())}")

    img = ex["image"].convert("RGB")
    label = _extract_label(ex, ds[split_name])
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


def _occlusion_sensitivity_heatmap(feature_extractor_fn, predict_proba_fn, img_pil, target_idx, patch_size=32, stride=32):
    base = _to_uint8(img_pil)
    pil = Image.fromarray(base).resize((224, 224), Image.BILINEAR)
    arr = np.array(pil, dtype=np.uint8)
    h, w = arr.shape[:2]
    heat = np.zeros((h, w), dtype=np.float32)
    counts = np.zeros((h, w), dtype=np.float32)

    orig_feat = feature_extractor_fn(preprocess_image(img_pil))
    orig_prob = float(predict_proba_fn(orig_feat)[0, target_idx])

    for y in range(0, h, stride):
        for x in range(0, w, stride):
            x2 = min(x + patch_size, w)
            y2 = min(y + patch_size, h)
            occluded = arr.copy()
            occluded[y:y2, x:x2] = 0
            occl_pil = Image.fromarray(occluded)
            feat = feature_extractor_fn(preprocess_image(occl_pil))
            prob = float(predict_proba_fn(feat)[0, target_idx])
            drop = max(orig_prob - prob, 0.0)
            heat[y:y2, x:x2] += drop
            counts[y:y2, x:x2] += 1

    heat = heat / np.maximum(counts, 1.0)
    return _apply_heatmap_overlay(arr, heat, alpha=0.55)


def predict_with_explanations(model, id2label, device, img_pil, model_choice, k=5):
    if model_choice == "SVM + ResNet50":
        base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        base_model.fc = nn.Identity()
        base_model.eval()

        def extract_feats(img_tensor):
            with torch.no_grad():
                return base_model(img_tensor).cpu().numpy()

        feats = extract_feats(preprocess_image(img_pil))
        probs = model.predict_proba(feats)[0]
        top_idx = np.argsort(probs)[::-1][:min(k, len(probs))]
        top_vals = probs[top_idx]
        rows = [(id2label[int(i)], float(p)) for p, i in zip(top_vals, top_idx)]
        overlay = _occlusion_sensitivity_heatmap(extract_feats, model.predict_proba, img_pil, int(top_idx[0]))
        return rows, overlay, overlay, None

    if model_choice in ("Pretrained CNN Frozen", "CNN Scratch"):
        x = preprocess_image_tensor(img_pil).to(device).clone().detach().requires_grad_(True)

        cache = {}

        def fwd_hook(_m, _i, o):
            cache["act"] = o

        def bwd_hook(_m, _gi, go):
            cache["grad"] = go[0]

        target_layer = model.layer4[-1].conv2 if model_choice == "Pretrained CNN Frozen" else model.model.layer4[-1].conv2
        h1 = target_layer.register_forward_hook(fwd_hook)
        h2 = target_layer.register_full_backward_hook(bwd_hook)

        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        top_vals, top_idx = torch.topk(probs, k=min(k, probs.shape[0]))
        pred_idx = int(top_idx[0].item())

        model.zero_grad(set_to_none=True)
        logits[0, pred_idx].backward()

        h1.remove(); h2.remove()

        sal = x.grad.detach()[0].abs().max(dim=0).values.cpu().numpy()
        sal = _norm01(sal)
        saliency_overlay = _apply_heatmap_overlay(_to_uint8(img_pil), sal, alpha=0.50)

        act = cache["act"][0]
        grad = cache["grad"][0]
        w = grad.mean(dim=(1, 2), keepdim=True)
        cam = F.relu((w * act).sum(dim=0)).detach().cpu().numpy()
        cam = _norm01(cam)
        gradcam_overlay = _apply_heatmap_overlay(_to_uint8(img_pil), cam, alpha=0.45)

        attn_overlay = None
        rows = [(id2label[int(i)], float(p)) for p, i in zip(top_vals.detach().cpu().numpy(), top_idx.detach().cpu().numpy())]
        return rows, saliency_overlay, gradcam_overlay, attn_overlay

    x = preprocess_image(img_pil).to(device).clone().detach().requires_grad_(True)

    # hooks for saliency + Grad-CAM
    cache = {}

    def fwd_hook(_m, _i, o):
        cache["act"] = o

    def bwd_hook(_m, _gi, go):
        cache["grad"] = go[0]

    if model_choice == "MBLANet":
        target_layer = model.backbone.layer4
        h1 = target_layer.register_forward_hook(fwd_hook)
        h2 = target_layer.register_full_backward_hook(bwd_hook)

        lsam_module = model.backbone.layer4[-1].clam.lsam
        def lsam_hook(m, i, o):
            cache["lsam_map"] = o.detach()
        h3 = lsam_module.register_forward_hook(lsam_hook)

        logits = model(x)
    else:
        target_layer = model.layer4[-1].conv2
        h1 = target_layer.register_forward_hook(fwd_hook)
        h2 = target_layer.register_full_backward_hook(bwd_hook)
        h3 = None
        logits = model(x)

    probs = torch.softmax(logits, dim=1)[0]
    top_vals, top_idx = torch.topk(probs, k=min(k, probs.shape[0]))
    pred_idx = int(top_idx[0].item())

    model.zero_grad(set_to_none=True)
    logits[0, pred_idx].backward()

    h1.remove(); h2.remove()
    if h3: h3.remove()

    grad_input = x.grad.detach()[0]
    sal = grad_input.abs().max(dim=0).values.cpu().numpy()
    sal = _norm01(sal)
    saliency_overlay = _apply_heatmap_overlay(_to_uint8(img_pil), sal, alpha=0.50)

    act = cache["act"][0]
    grad = cache["grad"][0]
    w = grad.mean(dim=(1, 2), keepdim=True)
    cam = F.relu((w * act).sum(dim=0)).detach().cpu().numpy()
    cam = _norm01(cam)
    gradcam_overlay = _apply_heatmap_overlay(_to_uint8(img_pil), cam, alpha=0.45)

    attn_overlay = None
    if model_choice == "MBLANet" and "lsam_map" in cache:
        attn_tensor = cache["lsam_map"]
        attn_np = attn_tensor.cpu().numpy()[0, 0]
        attn_np = _norm01(attn_np)
        attn_overlay = _apply_heatmap_overlay(_to_uint8(img_pil), attn_np, alpha=0.60)

    rows = [(id2label[int(i)], float(p)) for p, i in zip(top_vals.detach().cpu().numpy(), top_idx.detach().cpu().numpy())]
    return rows, saliency_overlay, gradcam_overlay, attn_overlay



# =========================
# UI
# =========================
st.markdown("<p class='hero'>Demo 2 · Image Classification</p>", unsafe_allow_html=True)
st.markdown(
    "<p class='sub'>Editor-style inference console with bento layout · MBLANet, CNN Scratch, Pretrained CNN Fine-tuned, and SVM + ResNet50.</p>",
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class='kpi-grid'>
      <div class='kpi-card'><div class='kpi-lbl'>Task</div><div class='kpi-val'>RSITMD Image Classification</div></div>
      <div class='kpi-card'><div class='kpi-lbl'>Input Size</div><div class='kpi-val'>224 × 224 px</div></div>
      <div class='kpi-card'><div class='kpi-lbl'>Explainability</div><div class='kpi-val'>Saliency · Grad-CAM · Attention</div></div>
    </div>
    """,
    unsafe_allow_html=True,
)

left, right = st.columns([1.25, 1])

with left:
    st.markdown("<div class='bento'><div class='editor-bar'><span class='dot dot-r'></span><span class='dot dot-y'></span><span class='dot dot-g'></span></div>", unsafe_allow_html=True)
    st.markdown("<div class='section'>Model & Input Console</div>", unsafe_allow_html=True)

    model_choice = st.selectbox("Choose model", ["MBLANet", "CNN Scratch", "Pretrained CNN Frozen", "Pretrained CNN Fine-tuned", "SVM + ResNet50"], index=0)


    model = None
    id2label = None
    device = None
    model_ready = True
    st.info("Model checkpoint will be loaded only when you press Predict.")
    ckpt_candidates = CHECKPOINT_CANDIDATES.get(model_choice, [])
    existing_ckpt = next((p for p in ckpt_candidates if p.exists() and p.stat().st_size > 0), None)
    if existing_ckpt is not None:
        st.markdown(
            f"<div class='small-note'>Checkpoint ready: <b>{existing_ckpt}</b></div>",
            unsafe_allow_html=True,
        )
    else:
        st.warning("Checkpoint not found locally yet; it will be fetched when you press Predict.")

    st.markdown("<div class='section'>Image Input</div>", unsafe_allow_html=True)

    mode = st.radio("Input mode", ["Upload image", "Drive random sample", "Drive named sample"], horizontal=True)

    image = None
    true_label = None

    if mode == "Upload image":
        up = st.file_uploader("Upload rice leaf image", type=["png", "jpg", "jpeg", "webp"])
        if up:
            image = Image.open(up).convert("RGB")
            st.image(image, caption="Input image", width=240)
    elif mode == "Drive random sample":
        if st.button("Load random sample from Drive", use_container_width=True):
            try:
                with st.spinner("Loading one random sample from Drive cache..."):
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
            st.image(image, caption=f"Drive sample: {meta.get('split', '?')}[{meta.get('index', '?')}]", width=240)
            if true_label is not None:
                if isinstance(true_label, (int, np.integer)) and id2label is not None and int(true_label) in id2label:
                    true_name = id2label[int(true_label)]
                else:
                    true_name = str(true_label)
                st.markdown(f"<div class='demo-label'>Ground truth label: <b>{true_name}</b></div>", unsafe_allow_html=True)
    else:
        spec = st.text_input("Sample key", value="test[0]", help="Format: split[index], e.g., test[7]")
        if st.button("Load named sample", use_container_width=True):
            try:
                with st.spinner("Loading requested sample from Drive cache..."):
                    sample_img, sample_label, sample_meta = get_named_sample_image(spec)
                st.session_state["sample_img"] = sample_img
                st.session_state["sample_label"] = sample_label
                st.session_state["sample_meta"] = sample_meta
            except Exception as e:
                st.error(f"Drive named sample error: {e}")

        if "sample_img" in st.session_state:
            image = st.session_state["sample_img"]
            true_label = st.session_state.get("sample_label")
            meta = st.session_state.get("sample_meta", {})
            st.image(image, caption=f"Drive sample: {meta.get('split', '?')}[{meta.get('index', '?')}]", width=240)
            if true_label is not None:
                if isinstance(true_label, (int, np.integer)) and id2label is not None and int(true_label) in id2label:
                    true_name = id2label[int(true_label)]
                else:
                    true_name = str(true_label)
                st.markdown(f"<div class='demo-label'>Ground truth label: {true_name}</div>", unsafe_allow_html=True)

    pred_btn = st.button("Predict", use_container_width=True, disabled=not model_ready)
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown("<div class='bento'><div class='editor-bar'><span class='dot dot-r'></span><span class='dot dot-y'></span><span class='dot dot-g'></span></div>", unsafe_allow_html=True)
    st.markdown("<div class='section'>Prediction & Explainability</div>", unsafe_allow_html=True)

    if pred_btn:
        if image is None:
            st.warning("Please upload an image first.")
        else:
            if model is None or id2label is None or device is None:
                with st.spinner(f"Loading {model_choice} checkpoint for prediction..."):
                    ckpt_path, map_path = get_checkpoint_and_mapping(model_choice)
                    model, id2label, device, _ckpt_used = load_model_and_labels(model_choice, ckpt_path, map_path)
            with st.spinner("Running inference..."):
                topk, saliency_overlay, gradcam_overlay, attention_overlay = predict_with_explanations(model, id2label, device, image, model_choice=model_choice, k=5)
            top_label, top_prob = topk[0]

            st.metric("Predicted class", top_label)
            st.markdown(f"<div class='demo-label'>Predicted label: {top_label}</div>", unsafe_allow_html=True)
            st.progress(float(top_prob))
            st.markdown(f"<div class='demo-label'>Confidence: {top_prob:.2%}</div>", unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("**Top-5 predictions**")
            for label, prob in topk:
                st.write(f"- {label}: {prob:.2%}")

            if model_choice == "SVM + ResNet50":
                st.markdown("---")
                st.markdown("**Occlusion sensitivity**")
                st.image(saliency_overlay, caption="Occlusion heatmap", use_container_width=True)
            else:
                st.markdown("---")
                st.markdown("**Visual Explanations**")
                if attention_overlay is not None:
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.image(saliency_overlay, caption="Saliency map", use_container_width=True)
                    with c2:
                        st.image(gradcam_overlay, caption="Grad-CAM (CNN focus)", use_container_width=True)
                    with c3:
                        st.image(attention_overlay, caption="Attention map", use_container_width=True)
                else:
                    c1, c2 = st.columns(2)
                    with c1:
                        st.image(saliency_overlay, caption="Saliency map", use_container_width=True)
                    with c2:
                        st.image(gradcam_overlay, caption="Grad-CAM (CNN Scratch)", use_container_width=True)
    else:
        st.info("Upload image and click Predict.")

    st.markdown("</div>", unsafe_allow_html=True)
