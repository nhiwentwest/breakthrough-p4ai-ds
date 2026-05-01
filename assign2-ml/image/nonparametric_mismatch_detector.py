"""
Non-parametric image mismatch detector from scratch.

What it does:
- Reads an image folder structured as root/class_name/*.jpg
- Extracts handcrafted features from each image
- Computes per-class centroids
- Computes kNN label agreement
- Flags potentially mislabeled / out-of-class samples
- Writes a CSV ranked by suspicion score

Run:
  python nonparametric_mismatch_detector.py --data_dir /path/to/dataset --output_csv suspicious_samples.csv
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass
class Config:
    data_dir: str
    output_csv: str
    image_size: int = 128
    k_neighbors: int = 15
    top_percent: float = 5.0
    max_images_per_class: int = 0


def list_images(data_dir: Path) -> List[Tuple[Path, str]]:
    items: List[Tuple[Path, str]] = []
    for class_dir in sorted([p for p in data_dir.iterdir() if p.is_dir()]):
        label = class_dir.name
        for path in sorted(class_dir.rglob("*")):
            if path.suffix.lower() in IMAGE_EXTS:
                items.append((path, label))
    return items


def resize_and_load(path: Path, size: int) -> np.ndarray:
    with Image.open(path) as img:
        img = img.convert("RGB").resize((size, size))
        return np.asarray(img, dtype=np.float32) / 255.0


def channel_histogram(arr: np.ndarray, bins: int = 16) -> np.ndarray:
    feats = []
    for ch in range(3):
        hist, _ = np.histogram(arr[:, :, ch], bins=bins, range=(0.0, 1.0), density=True)
        feats.append(hist.astype(np.float32))
    return np.concatenate(feats)


def grayscale_entropy(gray: np.ndarray, bins: int = 32) -> float:
    hist, _ = np.histogram(gray, bins=bins, range=(0.0, 1.0), density=False)
    p = hist.astype(np.float32)
    p = p / max(p.sum(), 1.0)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum()) if len(p) else 0.0


def edge_density(gray_uint8: np.ndarray) -> float:
    import cv2

    edges = cv2.Canny(gray_uint8, 80, 160)
    return float(edges.mean() / 255.0)


def extract_features(path: Path, image_size: int) -> np.ndarray:
    import cv2

    arr = resize_and_load(path, image_size)
    gray = (arr.mean(axis=2) * 255.0).astype(np.uint8)

    feats: List[float] = []

    # Color distribution
    feats.extend(channel_histogram(arr, bins=16).tolist())
    feats.append(float(arr.mean()))
    feats.append(float(arr.std()))
    feats.extend(arr.mean(axis=(0, 1)).astype(np.float32).tolist())
    feats.extend(arr.std(axis=(0, 1)).astype(np.float32).tolist())

    # Texture / contrast
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    feats.append(lap_var)
    feats.append(edge_density(gray))
    feats.append(grayscale_entropy(gray))
    feats.append(float(gray.std() / 255.0))

    # Simple shape / structure proxies
    h, w = gray.shape
    mid_row = gray[h // 2, :].astype(np.float32) / 255.0
    mid_col = gray[:, w // 2].astype(np.float32) / 255.0
    feats.append(float(mid_row.mean()))
    feats.append(float(mid_col.mean()))
    feats.append(float(np.abs(mid_row - mid_row.mean()).mean()))
    feats.append(float(np.abs(mid_col - mid_col.mean()).mean()))

    return np.asarray(feats, dtype=np.float32)


def build_feature_matrix(items: List[Tuple[Path, str]], image_size: int) -> Tuple[np.ndarray, List[str], List[str]]:
    X = []
    paths = []
    labels = []
    for path, label in items:
        try:
            X.append(extract_features(path, image_size))
            paths.append(str(path))
            labels.append(label)
        except Exception as exc:
            print(f"Skipping {path}: {exc}")
    if not X:
        raise RuntimeError("No valid images found.")
    return np.vstack(X), paths, labels


def compute_scores(X: np.ndarray, labels: List[str], k_neighbors: int) -> List[dict]:
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    classes = sorted(set(labels))
    label_array = np.asarray(labels)
    centroids = {c: Xs[label_array == c].mean(axis=0) for c in classes}

    nbrs = NearestNeighbors(n_neighbors=min(k_neighbors + 1, len(Xs)), metric="cosine")
    nbrs.fit(Xs)
    distances, indices = nbrs.kneighbors(Xs)

    rows = []
    for i, y in enumerate(labels):
        nbr_ids = [j for j in indices[i] if j != i][:k_neighbors]
        same_ratio = float(np.mean([labels[j] == y for j in nbr_ids])) if nbr_ids else 0.0

        own_dist = float(np.linalg.norm(Xs[i] - centroids[y]))
        other_dists = [(c, float(np.linalg.norm(Xs[i] - centroids[c]))) for c in classes if c != y]
        nearest_other_label, nearest_other_dist = min(other_dists, key=lambda t: t[1]) if other_dists else (y, own_dist)

        centroid_gap = nearest_other_dist - own_dist
        mismatch_strength = max(0.0, -centroid_gap)
        score = (1.0 - same_ratio) + mismatch_strength

        if same_ratio < 0.3 and nearest_other_dist < own_dist:
            verdict = f"likely mismatch -> closer to {nearest_other_label}"
        elif same_ratio < 0.5 or centroid_gap < 0.0:
            verdict = "review"
        else:
            verdict = "clear"

        rows.append({
            "path": None,
            "label": y,
            "same_ratio": same_ratio,
            "own_dist": own_dist,
            "nearest_other_label": nearest_other_label,
            "nearest_other_dist": nearest_other_dist,
            "centroid_gap": centroid_gap,
            "score": float(score),
            "verdict": verdict,
        })

    return rows


def add_paths(rows: List[dict], paths: List[str]) -> List[dict]:
    for row, path in zip(rows, paths):
        row["path"] = path
    return rows


def summarize_and_flag(rows: List[dict], top_percent: float) -> Tuple[List[dict], float]:
    scores = np.asarray([r["score"] for r in rows], dtype=np.float32)
    if len(scores) == 0:
        return rows, 0.0
    threshold = float(np.percentile(scores, max(0.0, 100.0 - top_percent)))
    for row in rows:
        row["flagged"] = row["score"] >= threshold
    return rows, threshold


def write_csv(rows: List[dict], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "path",
        "label",
        "same_ratio",
        "own_dist",
        "nearest_other_label",
        "nearest_other_dist",
        "centroid_gap",
        "score",
        "verdict",
        "flagged",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args() -> Config:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Root folder with class subfolders")
    parser.add_argument("--output_csv", type=str, default="suspicious_samples.csv")
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--k_neighbors", type=int, default=15)
    parser.add_argument("--top_percent", type=float, default=5.0)
    parser.add_argument("--max_images_per_class", type=int, default=0, help="0 means no limit")
    args = parser.parse_args()
    return Config(
        data_dir=args.data_dir,
        output_csv=args.output_csv,
        image_size=args.image_size,
        k_neighbors=args.k_neighbors,
        top_percent=args.top_percent,
        max_images_per_class=args.max_images_per_class,
    )


def limit_per_class(items: List[Tuple[Path, str]], max_images_per_class: int) -> List[Tuple[Path, str]]:
    if max_images_per_class <= 0:
        return items
    seen: Dict[str, int] = {}
    limited: List[Tuple[Path, str]] = []
    for path, label in items:
        count = seen.get(label, 0)
        if count >= max_images_per_class:
            continue
        seen[label] = count + 1
        limited.append((path, label))
    return limited


def main() -> None:
    cfg = parse_args()
    data_dir = Path(cfg.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    items = list_images(data_dir)
    items = limit_per_class(items, cfg.max_images_per_class)
    print(f"Found {len(items)} images across {len(set(label for _, label in items))} classes")

    X, paths, labels = build_feature_matrix(items, cfg.image_size)
    rows = compute_scores(X, labels, cfg.k_neighbors)
    rows = add_paths(rows, paths)
    rows, threshold = summarize_and_flag(rows, cfg.top_percent)

    rows = sorted(rows, key=lambda r: r["score"], reverse=True)
    write_csv(rows, Path(cfg.output_csv))

    flagged = sum(1 for r in rows if r["flagged"])
    print(f"Score threshold for top {cfg.top_percent:.1f}%: {threshold:.4f}")
    print(f"Flagged {flagged}/{len(rows)} samples")
    print(f"Saved to {cfg.output_csv}")


if __name__ == "__main__":
    main()
