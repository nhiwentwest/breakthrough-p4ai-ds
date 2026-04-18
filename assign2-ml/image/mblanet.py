"""
MBLANet (Multi-Branch Local Attention Network) based on ResNet50
================================================================
This script implements a paper-aligned version of MBLANet (ResNet50 + CLAM)
for image classification.

Features:
- STEP 0: Config
- STEP 1: Imports + Seed
- STEP 2: Dataset (HF Disk)
- STEP 3: Model (ResNet50 + CCAM + LSAM embedded in blocks)
- STEP 4: Metrics + Timing + GradCAM/Saliency Viz
- STEP 5: Train / Validate / Test loops
- STEP 6: Main
"""

# =========================
# STEP 0 — Config
# =========================
import os
import argparse
from dataclasses import dataclass

@dataclass
class CFG:
    data_dir: str = "processed_rsitmd_256_clean"
    output_dir: str = "outputs_mblanet"
    seed: int = 42
    image_size: int = 224
    num_classes: int = 21

    epochs: int = 100
    batch_size: int = 64
    num_workers: int = 4
    lr: float = 0.01
    weight_decay: float = 1e-4

    early_stop_patience: int = 8
    early_stop_min_delta: float = 1e-4
    viz_samples: int = 12


# =========================
# STEP 1 — Imports + Seed
# =========================
import json
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
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, f1_score, confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import DataLoader, Dataset
from copy import deepcopy
from torchvision import models, transforms
from torchvision.models.resnet import Bottleneck

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
    if os.path.exists(os.path.join(cfg.data_dir, "dataset_dict.json")):
        return cfg.data_dir
    kaggle_path = "/kaggle/input/datasets/phantrntngvyk64cntt/processed-rsitmd-256-clean"
    if os.path.exists(os.path.join(kaggle_path, "dataset_dict.json")):
        return kaggle_path
    raise FileNotFoundError(f"Cannot find dataset_dict.json in {cfg.data_dir} or {kaggle_path}.")

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
# STEP 3 — Model (MBLANet: ResNet50 + CLAM)
# =========================
class CCAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv2x1 = nn.Conv2d(channels, channels, kernel_size=(2, 1), bias=False)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = self.avg_pool(x)                  # (B, C, 1, 1)
        mx = self.max_pool(x)                   # (B, C, 1, 1)
        fused = torch.cat([mx, avg], dim=2)     # (B, C, 2, 1)
        fused = self.conv2x1(fused)             # (B, C, 1, 1)
        fused = self.mlp(fused)                 # (B, C, 1, 1)
        return self.sigmoid(fused)

class LSAM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.eps = 8
        self.local_max = nn.MaxPool2d(kernel_size=self.eps, stride=self.eps)
        self.local_avg = nn.AvgPool2d(kernel_size=self.eps, stride=self.eps)
        self.dilated = nn.Conv2d(2, 1, kernel_size=3, dilation=2, padding=2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h, w = x.shape[-2:]
        k = max(2, min(self.eps, h // 2, w // 2, h, w))
        local_max = torch.nn.functional.max_pool2d(x, kernel_size=k, stride=k)
        local_avg = torch.nn.functional.avg_pool2d(x, kernel_size=k, stride=k)
        max_desc = torch.max(local_max, dim=1, keepdim=True).values
        avg_desc = torch.mean(local_avg, dim=1, keepdim=True)
        fused = torch.cat([max_desc, avg_desc], dim=1)
        fused = self.dilated(fused)
        fused = torch.nn.functional.interpolate(fused, size=x.shape[-2:], mode='nearest')
        attn = self.sigmoid(fused)
        self.raw_attn = attn
        self.att_map = attn
        self.input_stats = {
            "shape": tuple(x.shape),
            "kernel_size": int(k),
        }
        return attn

class CLAM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.ccam = CCAM(channels)
        self.lsam = LSAM(channels)

    def forward(self, x):
        mc = self.ccam(x)
        ms = self.lsam(x)
        return (mc * ms) * x

class CLAMResBlock(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.conv1 = deepcopy(block.conv1)
        self.bn1 = deepcopy(block.bn1)
        self.conv2 = deepcopy(block.conv2)
        self.bn2 = deepcopy(block.bn2)
        self.conv3 = deepcopy(block.conv3)
        self.bn3 = deepcopy(block.bn3)
        self.relu = deepcopy(block.relu)
        self.clam = CLAM(self.conv3.out_channels)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.clam(out)
        out += identity
        out = self.relu(out)
        return out

class CLAMDnSample(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.conv1 = deepcopy(block.conv1)
        self.bn1 = deepcopy(block.bn1)
        self.conv2 = deepcopy(block.conv2)
        self.bn2 = deepcopy(block.bn2)
        self.conv3 = deepcopy(block.conv3)
        self.bn3 = deepcopy(block.bn3)
        self.relu = deepcopy(block.relu)
        self.downsample = deepcopy(block.downsample)
        self.clam = CLAM(self.conv3.out_channels)

    def forward(self, x):
        identity = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.clam(out)
        out += identity
        out = self.relu(out)
        return out

class MBLANet(nn.Module):
    def __init__(self, num_classes=21, pretrained=True):
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.resnet50(weights=weights)
        self._inject_clam(self.backbone.layer1, channels=256)
        self._inject_clam(self.backbone.layer2, channels=512)
        self._inject_clam(self.backbone.layer3, channels=1024)
        self._inject_clam(self.backbone.layer4, channels=2048)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def _inject_clam(self, stage, channels):
        for i in range(len(stage)):
            block = stage[i]
            if block.downsample is not None:
                stage[i] = CLAMDnSample(block)
            else:
                stage[i] = CLAMResBlock(block)

    def forward(self, x):
        return self.backbone(x)

# =========================
# STEP 4 — Metrics + Timing + Viz
# =========================
@torch.no_grad()
def run_eval(model, loader, device):
    model.eval()
    ys, ps = [], []
    total_loss, n = 0.0, 0
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
    return {
        "loss": total_loss / max(n, 1),
        "acc": accuracy_score(ys, ps),
        "balanced_acc": balanced_accuracy_score(ys, ps),
        "macro_f1": f1_score(ys, ps, average="macro"),
        "y_true": ys,
        "y_pred": ps,
    }

def measure_inference_speed(model, loader, device, warmup=10):
    model.eval()
    times = []
    amp_enabled = device.type == "cuda"
    it = iter(loader)
    with torch.no_grad():
        for _ in range(warmup):
            try: x, _ = next(it)
            except StopIteration: break
            x = x.to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                _ = model(x)
        for x, _ in loader:
            x = x.to(device, non_blocking=True)
            if device.type == "cuda": torch.cuda.synchronize()
            t0 = time.time()
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                _ = model(x)
            if device.type == "cuda": torch.cuda.synchronize()
            times.append(time.time() - t0)
    if not times: return {"ms_per_batch": None, "images_per_sec": None}
    ms_per_batch = 1000.0 * float(np.mean(times))
    images_per_sec = float(loader.batch_size / np.mean(times))
    return {"ms_per_batch": ms_per_batch, "images_per_sec": images_per_sec}

def denormalize_image(x):
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(3, 1, 1)
    y = x * std + mean
    y = y.clamp(0, 1)
    return (y * 255.0).byte().permute(1, 2, 0).cpu().numpy()

def generate_visualizations(model, test_loader, device, output_dir, max_samples=12, epoch=None):
    # Debug-time inspection only; keep in-memory and avoid writing PNGs that the demo never reads.
    model.eval()
    done = 0
    for x, y in test_loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        for i in range(x.size(0)):
            if done >= max_samples:
                return
            xi = x[i:i+1].clone().detach().requires_grad_(True)

            model.zero_grad(set_to_none=True)
            logits = model(xi)
            pred_idx = int(torch.argmax(logits, dim=1).item())
            logits[0, pred_idx].backward(retain_graph=True)

            # Saliency map for quick inspection.
            grad = xi.grad.detach()[0]
            sal = grad.abs().max(dim=0).values.cpu().numpy()
            sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)

            # Multi-block raw attention inspection: layer3, layer4.
            for layer_name in ["layer3", "layer4"]:
                layer = getattr(model.backbone, layer_name)
                last_block = layer[-1]
                if hasattr(last_block, "clam") and hasattr(last_block.clam, "lsam"):
                    _ = model(xi)
                    attn = getattr(last_block.clam.lsam, "raw_attn", None)
                    if attn is not None:
                        attn_np = attn.detach().cpu().numpy() if torch.is_tensor(attn) else np.asarray(attn)
                        attn_map = attn_np[0, 0] if attn_np.ndim == 4 else attn_np.squeeze()
                        _ = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)

            # Grad-CAM from layer4.
            activations, gradients = {}, {}
            def fwd_hook(_m, _inp, out): activations["x"] = out
            def bwd_hook(_m, _gin, gout): gradients["x"] = gout[0]
            h1 = model.backbone.layer4.register_forward_hook(fwd_hook)
            h2 = model.backbone.layer4.register_full_backward_hook(bwd_hook)
            _ = model(xi)
            h1.remove(); h2.remove()
            if "x" in activations and "x" in gradients:
                act = activations["x"][0]
                grd = gradients["x"][0]
                w = grd.mean(dim=(1, 2), keepdim=True)
                cam = F.relu((w * act).sum(dim=0)).detach().cpu().numpy()
                cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

            done += 1


# =========================
# STEP 6 — Training Loop
# =========================
def train_one_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    ys, ps = [], []
    total_loss, n = 0.0, 0
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

def run_training(cfg: CFG):
    os.makedirs(cfg.output_dir, exist_ok=True)
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    train_loader, val_loader, test_loader, label2id, id2label, class_counts = build_dataloaders(cfg)
    
    # Initialize implementation of paper
    model = MBLANet(num_classes=len(label2id), pretrained=True).to(device)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    scaler = torch.amp.GradScaler(device="cuda", enabled=(device.type == "cuda"))

    best_f1, epochs_no_improve = -1.0, 0
    best_path = os.path.join(cfg.output_dir, "best_mblanet.pt")
    history = []
    
    def safe_torch_save(obj, path):
        tmp_path = path + ".tmp"
        torch.save(obj, tmp_path)
        os.replace(tmp_path, path)

    start = time.time()
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        tr = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        va = run_eval(model, val_loader, device)
        scheduler.step()
        epoch_time = time.time() - t0
        
        row = {"epoch": epoch, "epoch_time_sec": epoch_time, "lr": optimizer.param_groups[0]["lr"]}
        row.update({f"train_{k}": v for k, v in tr.items()})
        row.update({f"val_{k}": v for k, v in va.items()})
        history.append(row)
        
        print(f"[Epoch {epoch:02d}/{cfg.epochs}] train_f1={tr['macro_f1']:.4f} val_f1={va['macro_f1']:.4f} time={epoch_time:.1f}s")

        if va["macro_f1"] > (best_f1 + cfg.early_stop_min_delta):
            best_f1 = va["macro_f1"]
            epochs_no_improve = 0
            safe_torch_save({"model_state_dict": model.state_dict(), "cfg": cfg.__dict__,"epoch": epoch}, best_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg.early_stop_patience:
                print(f"Early stop at epoch {epoch}")
                break

    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device)["model_state_dict"])
    
    test_metrics = run_eval(model, test_loader, device)
    
    # Render and save Confusion Matrix
    labels = list(range(len(id2label)))
    cm = confusion_matrix(test_metrics["y_true"], test_metrics["y_pred"], labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[id2label[i] for i in labels])
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(ax=ax, cmap="Blues", xticks_rotation="vertical")
    plt.tight_layout()
    fig.savefig(os.path.join(cfg.output_dir, "confusion_matrix.png"), dpi=150)
    plt.close(fig)

    cls_report = classification_report(
        test_metrics["y_true"],
        test_metrics["y_pred"],
        labels=labels,
        target_names=[id2label[i] for i in labels],
        digits=4,
        output_dict=True,
        zero_division=0,
    )
    
    generate_visualizations(model, test_loader, device, cfg.output_dir, max_samples=cfg.viz_samples, epoch=epoch)

    # Plot Learning Curves
    if history:
        ep_range = [h["epoch"] for h in history]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.plot(ep_range, [h.get("train_loss") for h in history], label="Train", marker="o")
        ax1.plot(ep_range, [h.get("val_loss") for h in history], label="Validation", marker="o")
        ax1.set_title("Loss over Epochs")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax1.grid(True)
        ax2.plot(ep_range, [h.get("train_macro_f1") for h in history], label="Train", marker="o")
        ax2.plot(ep_range, [h.get("val_macro_f1") for h in history], label="Validation", marker="o")
        ax2.set_title("Macro F1 over Epochs")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Macro F1")
        ax2.legend()
        ax2.grid(True)
        plt.tight_layout()
        fig.savefig(os.path.join(cfg.output_dir, "learning_curves.png"), dpi=150)
        plt.close(fig)

    report_path = os.path.join(cfg.output_dir, "mblanet_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({"test": test_metrics, "classification_report": cls_report, "history": history}, f, indent=2)

    print(f"Done. Report: {report_path}")

# =========================
# STEP 7 — Main
# =========================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="processed_rsitmd_256_clean")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.01)
    args, _ = p.parse_known_args()
    return args

def main():
    args = parse_args()
    cfg = CFG(data_dir=args.data_dir, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
    run_training(cfg)

if __name__ == "__main__":
    main()
