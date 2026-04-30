"""
CNN pre-trained ResNet50 fine-tuning baseline with augmentation (single-file, step-by-step)

This script uses a standard ResNet50 backbone pre-trained on ImageNet,
then fine-tunes all layers end-to-end without any special custom architecture.

Run:
  python cnn_pretrain_resnet50_finetune.py

NOTE: This augmentation clone applies stronger train-time augmentation so you can compare against the non-augmentation version.
"""

# =========================
# STEP 0 — Config
# =========================
import argparse
from dataclasses import dataclass


@dataclass
class CFG:
    data_dir: str = "processed_rsitmd_256_clean"
    output_dir: str = "outputs_cnn_scratch"
    seed: int = 42
    image_size: int = 224

    epochs: int = 50
    batch_size: int = 32
    num_workers: int = 4
    lr: float = 1e-4
    weight_decay: float = 1e-4

    early_stop_patience: int = 8
    early_stop_min_delta: float = 1e-4


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
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, f1_score, confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import matplotlib.pyplot as plt


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
        y = self.label2id[ex["label"]]
        if self.tfm is not None:
            img = self.tfm(img)
        return img, y


def resolve_data_dir(data_dir: str) -> str:
    # 1. Local or explicitly provided path
    if os.path.exists(os.path.join(data_dir, "dataset_dict.json")):
        return data_dir

    # 2. Kaggle fallback path
    kaggle_path = "/kaggle/input/datasets/phantrntngvyk64cntt/processed-rsitmd-256-clean"
    if os.path.exists(os.path.join(kaggle_path, "dataset_dict.json")):
        return kaggle_path

    raise FileNotFoundError(f"Cannot find dataset folder containing dataset_dict.json in {data_dir} or {kaggle_path}")


def build_dataloaders(cfg: CFG):
    data_dir = resolve_data_dir(cfg.data_dir)
    print(f"Using data_dir: {data_dir}")
    ds = load_from_disk(data_dir)

    train_split = ds["train"]
    val_split = ds["validation"]
    test_split = ds["test"]

    classes = sorted(list(set(train_split["label"])))
    label2id = {c: i for i, c in enumerate(classes)}
    id2label = {i: c for c, i in label2id.items()}

    train_tfm = transforms.Compose([
        transforms.RandomResizedCrop(cfg.image_size, scale=(0.65, 0.7)),
        transforms.RandomRotation(10),
        transforms.RandomPerspective(distortion_scale=0.15, p=0.5),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05), shear=5),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.18, contrast=0.18, saturation=0.12, hue=0.04),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    eval_tfm = transforms.Compose([
        transforms.Resize((cfg.image_size, cfg.image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_ds = HFDiskImageDataset(train_split, label2id, train_tfm)
    val_ds = HFDiskImageDataset(val_split, label2id, eval_tfm)
    test_ds = HFDiskImageDataset(test_split, label2id, eval_tfm)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    counts = Counter(train_split["label"])
    class_counts = np.array([counts[c] for c in classes], dtype=np.int64)

    return train_loader, val_loader, test_loader, label2id, id2label, class_counts


# =========================
# STEP 3 — Pretrained ResNet50 Fine-Tuning Model
# =========================
from torchvision import models

class PretrainedResNet50FineTune(nn.Module):
    def __init__(self, num_classes=33, dropout=0.3):
        super().__init__()
        # Pre-trained ResNet50 on ImageNet, fine-tuned end-to-end
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

        # Full-layer fine-tuning: keep every layer trainable
        for param in self.model.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.model(x)


# =========================
# STEP 4 — Metrics
# =========================
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, ps = [], []
    ce = nn.CrossEntropyLoss()
    total_loss, n = 0.0, 0

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


# =========================
# STEP 6 — Train / Validate / Test
# =========================
def run_training(cfg: CFG):
    os.makedirs(cfg.output_dir, exist_ok=True)
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    print(f"Device: {device}")

    train_loader, val_loader, test_loader, label2id, id2label, class_counts = build_dataloaders(cfg)

    model = PretrainedResNet50FineTune(num_classes=len(label2id)).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    scaler = torch.amp.GradScaler(device="cuda", enabled=(device.type == "cuda"))

    best_f1 = -1.0
    no_improve = 0
    best_path = os.path.join(cfg.output_dir, "best_resnet50_finetune.pt")
    history = []

    def safe_torch_save(obj, path):
        tmp = path + ".tmp"
        torch.save(obj, tmp)
        os.replace(tmp, path)

    start = time.time()
    stop_epoch = None
    patience_cap = cfg.early_stop_patience

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        tr = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        va = evaluate(model, val_loader, device)
        scheduler.step()

        row = {
            "epoch": epoch,
            "epoch_time_sec": time.time() - t0,
            "train_loss": tr["loss"],
            "train_acc": tr["acc"],
            "train_balanced_acc": tr["balanced_acc"],
            "train_macro_f1": tr["macro_f1"],
            "val_loss": va["loss"],
            "val_acc": va["acc"],
            "val_balanced_acc": va["balanced_acc"],
            "val_macro_f1": va["macro_f1"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)

        print(f"[Epoch {epoch:02d}/{cfg.epochs}] train_f1={tr['macro_f1']:.4f} val_f1={va['macro_f1']:.4f}")

        if va["macro_f1"] > (best_f1 + cfg.early_stop_min_delta):
            best_f1 = va["macro_f1"]
            no_improve = 0
            safe_torch_save({
                "model_state_dict": model.state_dict(),
                "label2id": label2id,
                "id2label": id2label,
                "cfg": cfg.__dict__,
                "best_val_macro_f1": best_f1,
                "epoch": epoch,
            }, best_path)
        else:
            no_improve += 1
            if no_improve >= patience_cap:
                stop_epoch = epoch
                print(f"Early stopping at epoch {epoch}")
                break

    total_train_time = time.time() - start

    if not os.path.exists(best_path):
        raise RuntimeError("Best checkpoint was not saved.")

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    test = evaluate(model, test_loader, device)
    inference = measure_inference_speed(model, test_loader, device)

    # Render and save Confusion Matrix
    labels = sorted(id2label.keys())
    cm = confusion_matrix(test["y_true"], test["y_pred"], labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[id2label[i] for i in labels])
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(ax=ax, cmap="Blues", xticks_rotation="vertical")
    plt.tight_layout()
    fig.savefig(os.path.join(cfg.output_dir, "confusion_matrix.png"), dpi=150)
    plt.close(fig)

    labels = sorted(id2label.keys())
    cls_report = classification_report(
        test["y_true"], test["y_pred"],
        labels=labels,
        target_names=[id2label[i] for i in labels],
        digits=4, output_dict=True, zero_division=0,
    )

    report = {
        "best_val_macro_f1": best_f1,
        "early_stopping": {
            "patience": cfg.early_stop_patience,
            "min_delta": cfg.early_stop_min_delta,
            "triggered": stop_epoch is not None,
            "stop_epoch": stop_epoch,
        },
        "test": {
            "loss": test["loss"],
            "acc": test["acc"],
            "balanced_acc": test["balanced_acc"],
            "macro_f1": test["macro_f1"],
        },
        "inference": inference,
        "timing": {
            "total_train_time_sec": total_train_time,
            "avg_epoch_time_sec": float(np.mean([h["epoch_time_sec"] for h in history])) if history else None,
        },
        "classification_report": cls_report,
        "history": history,
    }

    print(f"Test Overall Accuracy: {test['acc']:.4f}")
    print(f"Test Balanced Accuracy: {test['balanced_acc']:.4f}")
    print(f"Test Macro F1: {test['macro_f1']:.4f}")
    print(f"Inference Time: {inference['ms_per_batch']:.4f} ms/batch | {inference['images_per_sec']:.4f} images/sec")

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
        ax2.plot(ep_range, [h.get("val_macro_f1") for h in history], label="Val", marker="o")
        ax2.set_title("Macro F1 over Epochs (Train vs Val)")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Macro F1")
        ax2.legend()
        ax2.grid(True)
        plt.tight_layout()
        fig.savefig(os.path.join(cfg.output_dir, "learning_curves.png"), dpi=150)
        plt.close(fig)

    report_path = os.path.join(cfg.output_dir, "resnet50_finetune_report.json")
    tmp = report_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    os.replace(tmp, report_path)

    mapping_path = os.path.join(cfg.output_dir, "label_mapping.json")
    tmp = mapping_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"label2id": label2id, "id2label": {str(k): v for k, v in id2label.items()}}, f, indent=2, ensure_ascii=False)
    os.replace(tmp, mapping_path)

    print("\nDone.")
    print(f"Best checkpoint: {best_path}")
    print(f"Report: {report_path}")
    print("Model: pre-trained ResNet50 fine-tuned end-to-end, no special architecture")


# =========================
# STEP 7 — Main
# =========================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="processed_rsitmd_256_clean")
    p.add_argument("--output_dir", type=str, default="outputs_cnn_scratch")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--early_stop_patience", type=int, default=8)
    p.add_argument("--early_stop_min_delta", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    args, _ = p.parse_known_args()
    return args


def main():
    a = parse_args()
    cfg = CFG(
        data_dir=a.data_dir,
        output_dir=a.output_dir,
        epochs=a.epochs,
        batch_size=a.batch_size,
        num_workers=a.num_workers,
        lr=a.lr,
        weight_decay=a.weight_decay,
        early_stop_patience=a.early_stop_patience,
        early_stop_min_delta=a.early_stop_min_delta,
        seed=a.seed,
    )
    run_training(cfg)


if __name__ == "__main__":
    main()
