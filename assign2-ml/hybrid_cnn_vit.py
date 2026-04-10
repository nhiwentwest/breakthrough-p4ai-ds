"""
Hybrid CNN–ViT (single-file implementation)
===========================================
This script keeps EVERYTHING in one file, but split into clear steps:

STEP 0: Config
STEP 1: Imports + Seed
STEP 2: Dataset
STEP 3: Model (VGG blocks -> Inception -> Transformer)
STEP 4: Class-Balanced Focal Loss
STEP 5: Metrics + Timing
STEP 6: Train / Validate / Test loops
STEP 7: Main

Run (local):
  python hybrid_cnn_vit.py --data_dir processed_rice_224 --epochs 20 --batch_size 32

Run (Kaggle, dataset owner bocon66):
  # Add dataset in notebook: bocon66/processed_rice_224
  python hybrid_cnn_vit.py --kaggle_dataset_ref bocon66/processed_rice_224 --epochs 20 --batch_size 32
"""

# =========================
# STEP 0 — Config
# =========================
import os
import argparse
from dataclasses import dataclass


@dataclass
class CFG:
    # Kaggle dataset identifier and mounted input directory
    # Example Kaggle dataset reference: bocon66/processed_rice_224
    kaggle_dataset_ref: str = "bocon66/processed_rice_224"
    data_dir: str = "processed_rice_224"
    output_dir: str = "outputs_hybrid"
    seed: int = 42
    image_size: int = 224
    num_classes: int = 21

    # training
    epochs: int = 20
    batch_size: int = 32
    num_workers: int = 4
    lr: float = 3e-4
    weight_decay: float = 1e-4

    # cb-focal
    focal_gamma: float = 2.0
    cb_beta: float = 0.9999

    # early stopping
    early_stop_patience: int = 3
    early_stop_min_delta: float = 1e-4

    # visualization
    viz_samples: int = 12

    # hybrid model
    embed_dim: int = 256
    num_heads: int = 8
    depth: int = 4
    mlp_ratio: float = 4.0
    dropout: float = 0.1


# =========================
# STEP 1 — Imports + Seed
# =========================
import json
import os
import random
import time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================
# STEP 2 — Dataset
# =========================
class HFDiskImageDataset(Dataset):
    def __init__(self, hf_split, label2id, tfm=None):
        self.ds = hf_split
        self.label2id = label2id
        self.tfm = tfm

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        ex = self.ds[idx]
        img = ex["image"].convert("RGB")
        label = self.label2id[ex["label"]]
        if self.tfm is not None:
            img = self.tfm(img)
        return img, label


def resolve_data_dir(cfg: CFG) -> str:
    """Resolve processed dataset path robustly (local + Kaggle)."""
    candidates = [
        cfg.data_dir,
        "/kaggle/input/processed-rice-224/processed_rice_224",
        "/kaggle/input/processed_rice_224/processed_rice_224",
        "/kaggle/input/processed-rice-224",
        "/kaggle/input/processed_rice_224",
    ]

    # 1) direct candidates
    for p in candidates:
        if os.path.exists(os.path.join(p, "dataset_dict.json")):
            return p

    # 2) auto-discover recursively under /kaggle/input
    kaggle_root = "/kaggle/input"
    if os.path.isdir(kaggle_root):
        for root, dirs, files in os.walk(kaggle_root):
            if "dataset_dict.json" in files:
                return root

    raise FileNotFoundError(
        "Cannot find processed dataset directory with dataset_dict.json. "
        f"Checked direct candidates: {candidates}. "
        "Attach dataset in Kaggle Input or pass --data_dir explicitly to the folder that contains dataset_dict.json."
    )


def build_dataloaders(cfg: CFG):
    resolved_data_dir = resolve_data_dir(cfg)
    print(f"Using data_dir: {resolved_data_dir}")

    ds = load_from_disk(resolved_data_dir)
    train_split = ds["train"]
    val_split = ds["validation"]
    test_split = ds["test"]

    classes = sorted(list(set(train_split["label"])))
    label2id = {c: i for i, c in enumerate(classes)}
    id2label = {i: c for c, i in label2id.items()}

    train_tfm = transforms.Compose([
        transforms.RandomResizedCrop(cfg.image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.03),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    eval_tfm = transforms.Compose([
        transforms.Resize((cfg.image_size, cfg.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_ds = HFDiskImageDataset(train_split, label2id, tfm=train_tfm)
    val_ds = HFDiskImageDataset(val_split, label2id, tfm=eval_tfm)
    test_ds = HFDiskImageDataset(test_split, label2id, tfm=eval_tfm)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    train_counts = Counter(train_split["label"])
    counts_array = np.array([train_counts[c] for c in classes], dtype=np.int64)

    return train_loader, val_loader, test_loader, label2id, id2label, counts_array


# =========================
# STEP 3 — Model (Hybrid CNN–ViT)
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
        x1 = self.branch1(x)
        x3 = self.branch3(x)
        x5 = self.branch5(x)
        xp = self.branchp(x)
        return torch.cat([x1, x3, x5, xp], dim=1)


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
            x_attn, attn_w = self.attn(x_norm, x_norm, x_norm, need_weights=True, average_attn_weights=False)
        else:
            x_attn, attn_w = self.attn(x_norm, x_norm, x_norm, need_weights=False), None
        if isinstance(x_attn, tuple):
            x_attn = x_attn[0]
        x = x_attn + h
        h = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = x + h
        if return_attn:
            return x, attn_w
        return x


class HybridCNNViT(nn.Module):
    """
    Sequential hybrid:
    VGG16 first 2 blocks -> Inception -> Patch Embedding -> 4x Transformer Encoder -> Classifier
    """

    def __init__(self, num_classes=21, embed_dim=256, num_heads=8, depth=4, mlp_ratio=4.0, dropout=0.1):
        super().__init__()

        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        features = list(vgg.features.children())
        # block1 + block2 (up to second maxpool)
        self.cnn_front = nn.Sequential(*features[:10])  # output ~ (B,128,56,56) for 224 input

        self.inception = InceptionBlock(in_ch=128, b1=64, b3=64, b5=32, bp=32)
        inc_out = 64 + 64 + 32 + 32  # 192

        self.patch_embed = nn.Conv2d(inc_out, embed_dim, kernel_size=4, stride=4)  # (56->14)
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

    def forward(self, x, return_last_attn=False):
        x = self.cnn_front(x)
        x = self.inception(x)
        x = self.patch_embed(x)  # (B, D, H', W')

        b, d, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, N, D), N=h*w

        self._build_pos_embed_if_needed(n_tokens=x.shape[1], dim=x.shape[2], device=x.device)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        last_attn = None
        for i, blk in enumerate(self.transformer):
            if return_last_attn and i == len(self.transformer) - 1:
                x, last_attn = blk(x, return_attn=True)
            else:
                x = blk(x)

        x = self.norm(x)
        x = x.mean(dim=1)  # token pooling
        logits = self.head(x)

        if return_last_attn:
            return logits, last_attn, (h, w)
        return logits


# =========================
# STEP 4 — Class-Balanced Focal Loss
# =========================
class CBFocalLoss(nn.Module):
    def __init__(self, class_counts, beta=0.9999, gamma=2.0):
        super().__init__()
        counts = torch.tensor(class_counts, dtype=torch.float32)
        effective_num = 1.0 - torch.pow(torch.tensor(beta, dtype=torch.float32), counts)
        weights = (1.0 - beta) / (effective_num + 1e-12)
        weights = weights / weights.sum() * len(class_counts)
        self.register_buffer("weights", weights)
        self.gamma = gamma

    def forward(self, logits, targets):
        weights = self.weights.to(device=logits.device, dtype=logits.dtype)
        ce = F.cross_entropy(logits, targets, reduction="none", weight=weights)
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce
        return focal.mean()


# =========================
# STEP 5 — Metrics + Timing
# =========================
@torch.no_grad()
def run_eval(model, loader, device):
    model.eval()
    ys, ps = [], []
    total_loss = 0.0
    n = 0

    ce = nn.CrossEntropyLoss()
    amp_enabled = device.type == "cuda"

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(x)
            loss = ce(logits, y)

        pred = logits.argmax(dim=1)
        ys.extend(y.cpu().numpy().tolist())
        ps.extend(pred.cpu().numpy().tolist())

        bs = y.size(0)
        total_loss += loss.item() * bs
        n += bs

    acc = accuracy_score(ys, ps)
    bacc = balanced_accuracy_score(ys, ps)
    macro_f1 = f1_score(ys, ps, average="macro")

    return {
        "loss": total_loss / max(n, 1),
        "acc": acc,
        "balanced_acc": bacc,
        "macro_f1": macro_f1,
        "y_true": ys,
        "y_pred": ps,
    }


@torch.no_grad()
def measure_inference_speed(model, loader, device, warmup=10):
    model.eval()
    times = []
    amp_enabled = device.type == "cuda"

    it = iter(loader)
    for _ in range(warmup):
        try:
            x, _ = next(it)
        except StopIteration:
            break
        x = x.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            _ = model(x)

    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.time() - t0)

    if not times:
        return {"ms_per_batch": None, "images_per_sec": None}

    ms_per_batch = 1000.0 * float(np.mean(times))
    images_per_sec = float(loader.batch_size / np.mean(times))
    return {"ms_per_batch": ms_per_batch, "images_per_sec": images_per_sec}


# =========================
# STEP 6 — Train / Validate / Test
# =========================
def train_one_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    total_loss = 0.0
    ys, ps = [], []
    n = 0

    amp_enabled = device.type == "cuda"

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(x)
            loss = criterion(logits, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        pred = logits.argmax(dim=1)
        ys.extend(y.detach().cpu().numpy().tolist())
        ps.extend(pred.detach().cpu().numpy().tolist())

        bs = y.size(0)
        total_loss += loss.item() * bs
        n += bs

    return {
        "loss": total_loss / max(n, 1),
        "acc": accuracy_score(ys, ps),
        "balanced_acc": balanced_accuracy_score(ys, ps),
        "macro_f1": f1_score(ys, ps, average="macro"),
    }


def denormalize_image(x):
    """x: tensor (3,H,W) normalized by ImageNet stats -> uint8 image"""
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(3, 1, 1)
    y = x * std + mean
    y = y.clamp(0, 1)
    y = (y * 255.0).byte().permute(1, 2, 0).cpu().numpy()
    return y


def save_overlay(base_img_uint8, heatmap, save_path, alpha=0.45):
    # heatmap expected [H,W] in [0,1]
    cmap = plt.get_cmap("jet")
    color = (cmap(heatmap)[..., :3] * 255.0).astype(np.uint8)
    overlay = (alpha * color + (1 - alpha) * base_img_uint8).astype(np.uint8)

    fig = plt.figure(figsize=(6, 6))
    plt.imshow(overlay)
    plt.axis("off")
    plt.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def generate_visualizations(model, test_loader, device, output_dir, max_samples=12):
    viz_dir = os.path.join(output_dir, "viz")
    os.makedirs(viz_dir, exist_ok=True)

    model.eval()
    done = 0

    for x, y in test_loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        for i in range(x.size(0)):
            if done >= max_samples:
                return

            xi = x[i:i+1].clone().detach().requires_grad_(True)
            yi = y[i:i+1]

            # ---------- Attention map (last transformer block) ----------
            with torch.enable_grad():
                logits_attn, attn_w, token_hw = model(xi, return_last_attn=True)
            pred_idx = int(torch.argmax(logits_attn, dim=1).item())

            attn_map = None
            if attn_w is not None:
                # attn_w: (B, num_heads, N, N)
                attn = attn_w[0].mean(dim=0)  # (N,N)
                attn_map = attn.mean(dim=0)   # (N,)
                h_t, w_t = token_hw
                attn_map = attn_map.view(h_t, w_t).detach().cpu().numpy()
                attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)

            # ---------- Saliency map ----------
            model.zero_grad(set_to_none=True)
            logits_sal = model(xi)
            score = logits_sal[0, pred_idx]
            score.backward(retain_graph=True)
            grad = xi.grad.detach()[0]  # (3,H,W)
            sal = grad.abs().max(dim=0).values
            sal = sal.cpu().numpy()
            sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)

            # ---------- Grad-CAM (on inception output) ----------
            activations = {}
            gradients = {}

            def fwd_hook(_m, _inp, out):
                activations["x"] = out

            def bwd_hook(_m, _gin, gout):
                gradients["x"] = gout[0]

            h1 = model.inception.register_forward_hook(fwd_hook)
            h2 = model.inception.register_full_backward_hook(bwd_hook)

            model.zero_grad(set_to_none=True)
            logits_gc = model(xi)
            score_gc = logits_gc[0, pred_idx]
            score_gc.backward()

            h1.remove()
            h2.remove()

            if "x" in activations and "x" in gradients:
                act = activations["x"][0]      # (C,h,w)
                grd = gradients["x"][0]        # (C,h,w)
                w = grd.mean(dim=(1, 2), keepdim=True)
                cam = (w * act).sum(dim=0)
                cam = F.relu(cam)
                cam = cam.detach().cpu().numpy()
                cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
            else:
                cam = np.zeros((xi.shape[-2], xi.shape[-1]), dtype=np.float32)

            # ---------- Save ----------
            base = denormalize_image(xi[0].detach())

            # resize maps to image size
            H, W = base.shape[:2]
            sal_r = np.array(Image.fromarray((sal * 255).astype(np.uint8)).resize((W, H), resample=Image.BILINEAR), dtype=np.float32) / 255.0
            cam_r = np.array(Image.fromarray((cam * 255).astype(np.uint8)).resize((W, H), resample=Image.BILINEAR), dtype=np.float32) / 255.0

            if attn_map is not None:
                attn_r = np.array(Image.fromarray((attn_map * 255).astype(np.uint8)).resize((W, H), resample=Image.BILINEAR), dtype=np.float32) / 255.0
            else:
                attn_r = np.zeros((H, W), dtype=np.float32)

            save_overlay(base, cam_r, os.path.join(viz_dir, f"sample_{done:03d}_gradcam_pred{pred_idx}.png"))
            save_overlay(base, sal_r, os.path.join(viz_dir, f"sample_{done:03d}_saliency_pred{pred_idx}.png"))
            save_overlay(base, attn_r, os.path.join(viz_dir, f"sample_{done:03d}_attention_pred{pred_idx}.png"))

            done += 1


def run_training(cfg: CFG):
    os.makedirs(cfg.output_dir, exist_ok=True)
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader, test_loader, label2id, id2label, class_counts = build_dataloaders(cfg)

    model = HybridCNNViT(
        num_classes=len(label2id),
        embed_dim=cfg.embed_dim,
        num_heads=cfg.num_heads,
        depth=cfg.depth,
        mlp_ratio=cfg.mlp_ratio,
        dropout=cfg.dropout,
    ).to(device)

    # T4-friendly memory/perf knobs
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    criterion = CBFocalLoss(class_counts=class_counts, beta=cfg.cb_beta, gamma=cfg.focal_gamma).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    scaler = torch.amp.GradScaler(device="cuda", enabled=(device.type == "cuda"))

    best_f1 = -1.0
    best_path = os.path.join(cfg.output_dir, "best_hybrid_cnn_vit.pt")
    epochs_no_improve = 0
    early_stopped = False
    stop_epoch = None

    history = []
    train_start = time.time()

    def safe_torch_save(obj, path):
        tmp_path = path + ".tmp"
        torch.save(obj, tmp_path)
        os.replace(tmp_path, path)

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()

        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_metrics = run_eval(model, val_loader, device)
        scheduler.step()

        epoch_time = time.time() - t0

        row = {
            "epoch": epoch,
            "epoch_time_sec": epoch_time,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["acc"],
            "train_balanced_acc": train_metrics["balanced_acc"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["acc"],
            "val_balanced_acc": val_metrics["balanced_acc"],
            "val_macro_f1": val_metrics["macro_f1"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)

        print(
            f"[Epoch {epoch:02d}/{cfg.epochs}] "
            f"train_f1={train_metrics['macro_f1']:.4f} val_f1={val_metrics['macro_f1']:.4f} "
            f"val_bacc={val_metrics['balanced_acc']:.4f} time={epoch_time:.1f}s"
        )

        improved = val_metrics["macro_f1"] > (best_f1 + cfg.early_stop_min_delta)
        if improved:
            best_f1 = val_metrics["macro_f1"]
            epochs_no_improve = 0
            ckpt_obj = {
                "model_state_dict": model.state_dict(),
                "label2id": label2id,
                "id2label": id2label,
                "cfg": cfg.__dict__,
                "best_val_macro_f1": best_f1,
                "epoch": epoch,
            }
            safe_torch_save(ckpt_obj, best_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg.early_stop_patience:
                early_stopped = True
                stop_epoch = epoch
                print(f"Early stopping triggered at epoch {epoch} (best_val_macro_f1={best_f1:.4f}).")
                break

    total_train_time = time.time() - train_start

    if not os.path.exists(best_path):
        raise RuntimeError(
            f"Best checkpoint was not created at {best_path}. "
            "Training likely failed before first successful validation checkpoint save."
        )

    ckpt = torch.load(best_path, map_location=device)
    if "model_state_dict" not in ckpt:
        raise RuntimeError(f"Checkpoint file is invalid or corrupted: {best_path}")
    model.load_state_dict(ckpt["model_state_dict"])

    test_metrics = run_eval(model, test_loader, device)
    speed = measure_inference_speed(model, test_loader, device)

    cls_report = classification_report(
        test_metrics["y_true"],
        test_metrics["y_pred"],
        target_names=[id2label[i] for i in range(len(id2label))],
        digits=4,
        output_dict=True,
        zero_division=0,
    )

    # Generate Grad-CAM + Saliency + Attention visualizations
    generate_visualizations(
        model=model,
        test_loader=test_loader,
        device=device,
        output_dir=cfg.output_dir,
        max_samples=cfg.viz_samples,
    )

    report = {
        "best_val_macro_f1": best_f1,
        "early_stopping": {
            "patience": cfg.early_stop_patience,
            "min_delta": cfg.early_stop_min_delta,
            "triggered": early_stopped,
            "stop_epoch": stop_epoch,
        },
        "visualizations": {
            "dir": os.path.join(cfg.output_dir, "viz"),
            "samples": cfg.viz_samples,
            "types": ["gradcam", "saliency", "attention"],
        },
        "test": {
            "loss": test_metrics["loss"],
            "acc": test_metrics["acc"],
            "balanced_acc": test_metrics["balanced_acc"],
            "macro_f1": test_metrics["macro_f1"],
        },
        "timing": {
            "total_train_time_sec": total_train_time,
            "avg_epoch_time_sec": float(np.mean([h["epoch_time_sec"] for h in history])) if history else None,
            **speed,
        },
        "classification_report": cls_report,
        "history": history,
    }

    report_path = os.path.join(cfg.output_dir, "hybrid_report.json")
    report_tmp = report_path + ".tmp"
    with open(report_tmp, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    os.replace(report_tmp, report_path)

    mapping_path = os.path.join(cfg.output_dir, "label_mapping.json")
    mapping_tmp = mapping_path + ".tmp"
    with open(mapping_tmp, "w", encoding="utf-8") as f:
        json.dump({"label2id": label2id, "id2label": {str(k): v for k, v in id2label.items()}}, f, indent=2, ensure_ascii=False)
    os.replace(mapping_tmp, mapping_path)

    print("\nDone.")
    print(f"Best checkpoint: {best_path}")
    print(f"Report: {os.path.join(cfg.output_dir, 'hybrid_report.json')}")


# =========================
# STEP 7 — Main
# =========================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--kaggle_dataset_ref", type=str, default="bocon66/processed_rice_224")
    p.add_argument("--data_dir", type=str, default="processed_rice_224")
    p.add_argument("--output_dir", type=str, default="outputs_hybrid")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--focal_gamma", type=float, default=2.0)
    p.add_argument("--cb_beta", type=float, default=0.9999)
    p.add_argument("--embed_dim", type=int, default=256)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--early_stop_patience", type=int, default=3)
    p.add_argument("--early_stop_min_delta", type=float, default=1e-4)
    p.add_argument("--viz_samples", type=int, default=12)
    p.add_argument("--seed", type=int, default=42)

    # Important for Kaggle/Colab/Jupyter where kernel injects extra argv
    args, _ = p.parse_known_args()
    return args


def main():
    args = parse_args()
    cfg = CFG(
        kaggle_dataset_ref=args.kaggle_dataset_ref,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        lr=args.lr,
        weight_decay=args.weight_decay,
        focal_gamma=args.focal_gamma,
        cb_beta=args.cb_beta,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        depth=args.depth,
        dropout=args.dropout,
        early_stop_patience=args.early_stop_patience,
        early_stop_min_delta=args.early_stop_min_delta,
        viz_samples=args.viz_samples,
        seed=args.seed,
    )

    try:
        run_training(cfg)
    except FileNotFoundError as e:
        print("\n[DATA PATH ERROR]", str(e))
        if os.path.isdir("/kaggle/input"):
            print("\nContents of /kaggle/input (recursive, max 200 entries):")
            shown = 0
            for root, dirs, files in os.walk('/kaggle/input'):
                print('-', root)
                shown += 1
                if shown >= 200:
                    print('... truncated ...')
                    break
        raise


if __name__ == "__main__":
    main()
