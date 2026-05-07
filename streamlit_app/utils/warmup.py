"""
Warm-up utility — pre-download all Demo 2 checkpoint files and datasets to disk.

This module separates the slow network download (Google Drive) from the fast
local disk load (torch.load / joblib.load).  Running ``warmup_download_all``
once before the demo ensures every page can load models in 2-4 seconds from
disk instead of waiting minutes for a Drive download.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Optional

import gdown

# ---------------------------------------------------------------------------
# Central registry: filename → Google Drive file ID
# Only files actually needed for the demo flow (Text → Tabular → Image).
# ---------------------------------------------------------------------------
WARMUP_REGISTRY: dict[str, str] = {
    # --- Text (BERT) ---
    "text_bert_checkpoint.bin": "1IfVsAt5c9cHgaNiM3Q-y6z8XCTwDBsJw",
    # --- Text (Traditional ML) ---
    "pipeline_Naive_Bayes.joblib": "1k5Fiy7HlnUNLKhRYvm3akoysTcXa5Ejs",
    "pipeline_Logistic_Regression.joblib": "1Z7bLzVeUmioeQC4m9o2xoaYdnsT6d-cr",
    "best_pipeline.joblib": "1X2UbAGLQUMlsiXxAaHddPMwvOlBaEjXN",
    # --- Tabular (sklearn) ---
    "linear_regression.joblib": "1Q79AlDSSM1UNSMkIqKSO28iMGrgsjpzM",
    "random_forest.joblib": "1tl8yowNqcPzfqIRlGDXJr2fQM3Dm-bCR",
    "gradient_boosting.joblib": "1lWLbpkX6iq6hi6pfroY7Y_9cLIwHXQlH",
    "scaler.joblib": "1G1sl40ilyTF4IPoRbvvPuhKDoXoxpKsJ",
    "feature_columns.joblib": "1FxeRjLNkBIl6pn2KUjV7dccnHYBgsrYt",
    "insurance.csv": "16hHeuqWKFhrdk-PtyfVVOqDV1y77v9MS",
    # --- Image: MBLANet ---
    "best_mblanet.pt": "11YKkizz4FBzY72wrFFSZPgDDm48074Lx",
    "label_mapping_mblanet.json": "13wXU29DAVfo0MWqHWTHSzRB5c-p3d9Wq",
    # --- Image: Pretrained CNN Fine-tuned ---
    "best_resnet50_finetuned_model.pt": "1vIgcLba9ylYT7wNUVeQ7FLRBauvioi8_",
    "label_mapping_finetuned.json": "1cOLEUL0kULFGM0b0YuJA35-Oc1ohBenV",
}

# ---------------------------------------------------------------------------
# Image demo dataset (Google Drive folder)
# ---------------------------------------------------------------------------
IMAGE_DATASET_FOLDER_URL = "https://drive.google.com/drive/folders/1vmk07ZO_5hi6yBZQ15N0TfhZ2D9Y9-mv?usp=sharing"
IMAGE_DATASET_CACHE_DIR = Path(__file__).resolve().parents[1] / "data_cache" / "drive_image_demo_v2"

DEFAULT_CHECKPOINT_DIR = Path(__file__).resolve().parents[1] / "checkpoints"


def warmup_download_all(
    checkpoint_dir: Path | None = None,
    progress_callback: Optional[Callable] = None,
) -> dict[str, str]:
    """Download every registered checkpoint to *checkpoint_dir*.

    Files that already exist (and are non-empty) are skipped.

    Parameters
    ----------
    checkpoint_dir : Path, optional
        Defaults to ``streamlit_app/checkpoints``.
    progress_callback : callable, optional
        Called with ``(fraction: float, text: str)`` after each file.
        Works directly with a Streamlit ``st.progress`` object if you call
        ``progress_callback.progress(fraction, text=text)``.

    Returns
    -------
    dict
        Mapping of filename → status string ("skipped" | "downloaded" | "FAILED").
    """
    if checkpoint_dir is None:
        checkpoint_dir = DEFAULT_CHECKPOINT_DIR
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    total = len(WARMUP_REGISTRY)
    results: dict[str, str] = {}

    for idx, (filename, file_id) in enumerate(WARMUP_REGISTRY.items(), start=1):
        target = checkpoint_dir / filename
        frac = idx / total

        if target.exists() and target.stat().st_size > 0:
            results[filename] = "skipped"
            if progress_callback is not None:
                progress_callback.progress(frac, text=f"✅ {filename} (already on disk)")
            continue

        try:
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, str(target), quiet=True)
            if target.exists() and target.stat().st_size > 0:
                results[filename] = "downloaded"
                if progress_callback is not None:
                    progress_callback.progress(frac, text=f"⬇️ {filename} downloaded")
            else:
                results[filename] = "FAILED"
                if progress_callback is not None:
                    progress_callback.progress(frac, text=f"❌ {filename} failed")
        except Exception as exc:
            results[filename] = f"FAILED: {exc}"
            if progress_callback is not None:
                progress_callback.progress(frac, text=f"❌ {filename}: {exc}")

    # --- Image demo dataset (folder download) ---
    _download_image_dataset(results, progress_callback)

    return results


def _find_dataset_dict_root(base_dir: str) -> str | None:
    """Walk *base_dir* looking for a ``dataset_dict.json`` file."""
    for r, _d, files in os.walk(base_dir):
        if "dataset_dict.json" in files:
            return r
    return None


def _download_image_dataset(
    results: dict[str, str],
    progress_callback: Optional[Callable] = None,
) -> None:
    """Download the image demo dataset folder from Drive if not cached."""
    cache_dir = IMAGE_DATASET_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)

    if _find_dataset_dict_root(str(cache_dir)) is not None:
        results["image_dataset_folder"] = "skipped"
        if progress_callback is not None:
            progress_callback.progress(1.0, text="✅ Image dataset (already on disk)")
        return

    try:
        if progress_callback is not None:
            progress_callback.progress(0.95, text="⬇️ Downloading image dataset folder...")
        gdown.download_folder(
            url=IMAGE_DATASET_FOLDER_URL,
            output=str(cache_dir),
            quiet=True,
            use_cookies=False,
        )
        if _find_dataset_dict_root(str(cache_dir)) is not None:
            results["image_dataset_folder"] = "downloaded"
            if progress_callback is not None:
                progress_callback.progress(1.0, text="⬇️ Image dataset downloaded")
        else:
            results["image_dataset_folder"] = "FAILED"
            if progress_callback is not None:
                progress_callback.progress(1.0, text="❌ Image dataset: no dataset_dict.json found")
    except Exception as exc:
        results["image_dataset_folder"] = f"FAILED: {exc}"
        if progress_callback is not None:
            progress_callback.progress(1.0, text=f"❌ Image dataset: {exc}")


# ---------------------------------------------------------------------------
# Session-state cleanup helpers
# ---------------------------------------------------------------------------
# Keys that each demo page may create in st.session_state.
# When navigating away, these must be deleted so GC can free heavy objects.
DEMO_SESSION_KEYS = {
    "text": [
        "text_demo_model_loaded",
        "text_demo_checkpoint_path",
    ],
    "tabular": [],
    "image": [
        "image_demo_model_loaded",
        "sample_img",
        "sample_label",
        "sample_meta",
    ],
}


def cleanup_other_pages(current_page: str) -> None:
    """Remove session-state refs from pages other than *current_page*."""
    import gc
    import streamlit as st

    for page_name, keys in DEMO_SESSION_KEYS.items():
        if page_name == current_page:
            continue
        for k in keys:
            st.session_state.pop(k, None)

    st.cache_resource.clear()
    gc.collect()

    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
