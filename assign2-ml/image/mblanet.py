"""
MBLANet (Multi-Branch Local Attention Network) based on ResNet50
================================================================
This script implements a practical version of MBLANet (ResNet50 + CLAM)
for image classification. 

Features:
- STEP 0: Config
- STEP 1: Imports + Seed
- STEP 2: Dataset (HF Disk)
- STEP 3: Model (ResNet50 + CCAM + LSAM embedded in blocks)
- STEP 4: Class-Balanced Focal Loss
- STEP 5: Metrics + Timing + GradCAM/Saliency Viz
- STEP 6: Train / Validate / Test loops
- STEP 7: Main
"""

# =========================
# STEP 0 — Config
# =========================
import os
import argparse
from dataclasses import dataclass

@dataclass
class CFG:
    data_dir: str = "processed_rice_224"
    output_dir: str = "outputs_mblanet"
    seed: int = 42
    image_size: int = 224
    num_classes: int = 21

    epochs: int = 20
    batch_size: int = 32
    num_workers: int = 4
    lr: float = 1e-4  # Fine-tuning learning rate (thường nhỏ hơn)
    weight_decay: float = 1e-4

    focal_gamma: float = 2.0
    cb_beta: float = 0.9999

    early_stop_patience: int = 3
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
    if os.path.exists(os.path.join(cfg.data_dir, "dataset_dict.json")):
        return cfg.data_dir
    kaggle_path = "/kaggle/input/processed-rice-224/processed_rice_224"
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
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # Prevent zero division if channels < reduction
        reduced_dim = max(1, channels // reduction)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, reduced_dim, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_dim, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        a = self.mlp(self.avg_pool(x))
        m = self.mlp(self.max_pool(x))
        return self.sigmoid(a + m)

class LSAM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.local_avg = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.local_max = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.conv = nn.Conv2d(2, 1, kernel_size=3, padding=2, dilation=2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        a = self.local_avg(x).mean(dim=1, keepdim=True)
        # max pool returns (values, indices), we take values [0]
        m = self.local_max(x).max(dim=1, keepdim=True)[0]
        s = torch.cat([a, m], dim=1)
        return self.sigmoid(self.conv(s))

class CLAM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.ccam = CCAM(channels)
        self.lsam = LSAM(channels)

    def forward(self, x):
        c_attn = self.ccam(x)
        s_attn = self.lsam(x)
        return x * c_attn * s_attn

class BottleneckWithCLAM(nn.Module):
    def __init__(self, original_block, out_channels):
        super().__init__()
        self.block = original_block
        self.clam = CLAM(out_channels)

    def forward(self, x):
        y = self.block(x)
        y = self.clam(y)
        return y

class MBLANet(nn.Module):
    def __init__(self, num_classes=21, pretrained=True):
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.resnet50(weights=weights)
        
        # MBLANet embeds CLAM into residual blocks at each stage
        self._wrap_stage_with_clam(self.backbone.layer1, out_channels=256)
        self._wrap_stage_with_clam(self.backbone.layer2, out_channels=512)
        self._wrap_stage_with_clam(self.backbone.layer3, out_channels=1024)
        self._wrap_stage_with_clam(self.backbone.layer4, out_channels=2048)
        
        # Replace the fully connected classifier head
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def _wrap_stage_with_clam(self, stage, out_channels):
        for i in range(len(stage)):
            stage[i] = BottleneckWithCLAM(stage[i], out_channels)

    def forward(self, x):
        return self.backbone(x)

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
# STEP 5 — Metrics + Timing + Viz
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

def save_overlay(base_img_uint8, heatmap, save_path, alpha=0.45):
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
            if done >= max_samples: return
            xi = x[i:i+1].clone().detach().requires_grad_(True)
            
            # Saliency map
            model.zero_grad(set_to_none=True)
            logits_sal = model(xi)
            pred_idx = int(torch.argmax(logits_sal, dim=1).item())
            score = logits_sal[0, pred_idx]
            score.backward(retain_graph=True)
            
            grad = xi.grad.detach()[0]
            sal = grad.abs().max(dim=0).values.cpu().numpy()
            sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)

            # Grad-CAM on layer4 and LSAM Attention Map
            activations, gradients = {}, {}
            lsam_map = {}
            def fwd_hook(m, inp, out): activations["x"] = out
            def bwd_hook(m, gin, gout): gradients["x"] = gout[0]
            def lsam_hook(m, inp, out): lsam_map["s_attn"] = out.detach()
            
            h1 = model.backbone.layer4.register_forward_hook(fwd_hook)
            h2 = model.backbone.layer4.register_full_backward_hook(bwd_hook)
            
            # Hook the LSAM module in the very last block of layer4
            last_block = model.backbone.layer4[-1]
            if hasattr(last_block, "clam"):
                h3 = last_block.clam.lsam.register_forward_hook(lsam_hook)
            else:
                h3 = None
            
            model.zero_grad(set_to_none=True)
            logits_gc = model(xi)
            score_gc = logits_gc[0, pred_idx]
            score_gc.backward()
            
            h1.remove()
            h2.remove()
            if h3 is not None: h3.remove()
            
            if "x" in activations and "x" in gradients:
                act = activations["x"][0]
                grd = gradients["x"][0]
                w = grd.mean(dim=(1, 2), keepdim=True)
                cam = F.relu((w * act).sum(dim=0)).detach().cpu().numpy()
                cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
            else:
                cam = np.zeros((xi.shape[-2], xi.shape[-1]), dtype=np.float32)

            if "s_attn" in lsam_map:
                attn_map = lsam_map["s_attn"][0, 0].cpu().numpy()
                attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
            else:
                attn_map = np.zeros((xi.shape[-2], xi.shape[-1]), dtype=np.float32)

            base = denormalize_image(xi[0].detach())
            H, W = base.shape[:2]
            sal_r = np.array(Image.fromarray((sal*255).astype(np.uint8)).resize((W,H), resample=Image.BILINEAR), dtype=np.float32) / 255.0
            cam_r = np.array(Image.fromarray((cam*255).astype(np.uint8)).resize((W,H), resample=Image.BILINEAR), dtype=np.float32) / 255.0
            attn_r = np.array(Image.fromarray((attn_map*255).astype(np.uint8)).resize((W,H), resample=Image.BILINEAR), dtype=np.float32) / 255.0

            save_overlay(base, cam_r, os.path.join(viz_dir, f"sample_{done:03d}_gradcam_pred{pred_idx}.png"))
            save_overlay(base, sal_r, os.path.join(viz_dir, f"sample_{done:03d}_saliency_pred{pred_idx}.png"))
            save_overlay(base, attn_r, os.path.join(viz_dir, f"sample_{done:03d}_attention_pred{pred_idx}.png"))
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

    criterion = CBFocalLoss(class_counts=class_counts, beta=cfg.cb_beta, gamma=cfg.focal_gamma).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
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
    cm = confusion_matrix(test_metrics["y_true"], test_metrics["y_pred"])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[id2label[i] for i in range(len(id2label))])
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(ax=ax, cmap="Blues", xticks_rotation="vertical")
    plt.tight_layout()
    fig.savefig(os.path.join(cfg.output_dir, "confusion_matrix.png"), dpi=150)
    plt.close(fig)

    cls_report = classification_report(test_metrics["y_true"], test_metrics["y_pred"], target_names=[id2label[i] for i in range(len(id2label))], digits=4, output_dict=True, zero_division=0)
    
    generate_visualizations(model, test_loader, device, cfg.output_dir, max_samples=cfg.viz_samples)

    report_path = os.path.join(cfg.output_dir, "mblanet_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({"test": test_metrics, "classification_report": cls_report, "history": history}, f, indent=2)

    print(f"Done. Report: {report_path}")

# =========================
# STEP 7 — Main
# =========================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="processed_rice_224")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4) # Smaller LR for fine-tuning
    args, _ = p.parse_known_args()
    return args

def main():
    args = parse_args()
    cfg = CFG(data_dir=args.data_dir, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
    run_training(cfg)

if __name__ == "__main__":
    main()
