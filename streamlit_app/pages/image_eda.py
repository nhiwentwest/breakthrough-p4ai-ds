"""MNIST Image EDA demo mapped from mnist_eda.ipynb."""

from pathlib import Path
from urllib.request import urlretrieve

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Image EDA · MNIST", page_icon="🖼️", layout="wide", initial_sidebar_state="collapsed")

st.markdown(
    """
<style>
[data-testid="stSidebar"] { display:none !important; }
[data-testid="stSidebarNav"] { display:none !important; }
#MainMenu, footer, header { visibility:hidden; }
.main .block-container { padding:1rem; }
</style>
""",
    unsafe_allow_html=True,
)

STEP_LABELS = {
    0: "Load Dataset",
    1: "Convert + Reshape",
    2: "Dataset Properties",
    3: "Data Anomaly Check",
    4: "Sample MNIST Images",
    5: "Digit Distribution",
    6: "Pixel Intensity Distribution",
    7: "Example Image per Digit",
    8: "Mean Image per Digit",
    9: "Std Deviation per Digit",
}
TOTAL_STEPS = len(STEP_LABELS)

if "step" not in st.session_state:
    st.session_state.step = 0
if "mnist" not in st.session_state:
    st.session_state.mnist = None


@st.cache_data(show_spinner=False)
def load_mnist_data():
    cache_dir = Path("/tmp/mnist_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    npz_path = cache_dir / "mnist.npz"
    if not npz_path.exists():
        urlretrieve("https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz", npz_path)

    with np.load(npz_path) as f:
        x_train = f["x_train"]
        y_train = f["y_train"]
        x_test = f["x_test"]
        y_test = f["y_test"]

    images_np = x_train.reshape(len(x_train), -1)
    labels_np = y_train
    images_reshaped = x_train

    return {
        "x_train": x_train,
        "y_train": y_train,
        "x_test": x_test,
        "y_test": y_test,
        "images_np": images_np,
        "labels_np": labels_np,
        "images_reshaped": images_reshaped,
    }


if st.session_state.mnist is None:
    with st.spinner("Loading MNIST..."):
        st.session_state.mnist = load_mnist_data()

D = st.session_state.mnist
step = st.session_state.step

st.markdown("## Image EDA · MNIST")
st.caption(f"Step {step+1}/{TOTAL_STEPS}: {STEP_LABELS[step]}")

if step == 0:
    st.write("Load train and test split from MNIST.")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Train images", f"{len(D['x_train']):,}")
    m2.metric("Train labels", f"{len(D['y_train']):,}")
    m3.metric("Test images", f"{len(D['x_test']):,}")
    m4.metric("Test labels", f"{len(D['y_test']):,}")

elif step == 1:
    st.write("Convert to NumPy arrays and reshape train images.")
    info = pd.DataFrame(
        {
            "name": ["images_np", "labels_np", "images_reshaped"],
            "shape": [
                str(D["images_np"].shape),
                str(D["labels_np"].shape),
                str(D["images_reshaped"].shape),
            ],
            "dtype": [
                str(D["images_np"].dtype),
                str(D["labels_np"].dtype),
                str(D["images_reshaped"].dtype),
            ],
        }
    )
    st.dataframe(info, use_container_width=True)

elif step == 2:
    x_flat = D["images_np"]
    props = pd.DataFrame(
        {
            "metric": [
                "Train images",
                "Train labels",
                "Test images",
                "Test labels",
                "Min pixel value",
                "Max pixel value",
                "Missing values",
            ],
            "value": [
                len(D["x_train"]),
                len(D["y_train"]),
                len(D["x_test"]),
                len(D["y_test"]),
                int(x_flat.min()),
                int(x_flat.max()),
                int(np.isnan(x_flat).sum()),
            ],
        }
    )
    st.dataframe(props, use_container_width=True)

elif step == 3:
    sums_per_image = D["images_np"].sum(axis=1)
    blank_images = int((sums_per_image == 0).sum())
    c1, c2 = st.columns(2)
    c1.metric("Blank images", blank_images)
    c2.metric("Non-blank images", len(sums_per_image) - blank_images)

    bins = st.slider("Histogram bins", 20, 120, 60, key="sum_bins")
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.hist(sums_per_image, bins=bins, color="#8a3ffc", alpha=0.85, edgecolor="white")
    ax.set_title("Pixel-sum per image (train)")
    ax.set_xlabel("Sum of 784 pixels")
    ax.set_ylabel("Image count")
    st.pyplot(fig)

elif step == 4:
    start_idx = st.slider("Start index", 0, len(D["x_train"]) - 9, 0, key="sample_start")
    fig, axes = plt.subplots(3, 3, figsize=(7, 7))
    for i, ax in enumerate(axes.flatten()):
        idx = start_idx + i
        ax.imshow(D["images_reshaped"][idx], cmap="gray")
        ax.set_title(f"Label: {D['y_train'][idx]}")
        ax.axis("off")
    fig.suptitle("Sample MNIST Images")
    plt.tight_layout()
    st.pyplot(fig)

elif step == 5:
    split = st.radio("Split", ["train", "test"], horizontal=True, key="dist_split")
    labels = D["y_train"] if split == "train" else D["y_test"]

    counts = pd.Series(labels).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.bar(counts.index.astype(str), counts.values, color="#2563eb")
    ax.set_title(f"{split.title()} Digit Distribution")
    ax.set_xlabel("Digit")
    ax.set_ylabel("Count")
    st.pyplot(fig)

    st.dataframe(pd.DataFrame({"digit": counts.index, "count": counts.values}), use_container_width=True)

elif step == 6:
    split = st.radio("Split", ["train", "test"], horizontal=True, key="pix_split")
    x = D["x_train"] if split == "train" else D["x_test"]
    pixels = x.flatten()

    bins = st.slider("Bins", 20, 100, 50, key="pix_bins")
    log_scale = st.toggle("Log y-scale", value=True, key="pix_log")

    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.hist(pixels, bins=bins, color="#dc2626", alpha=0.85)
    if log_scale:
        ax.set_yscale("log")
    ax.set_title(f"Pixel Intensity Distribution ({split})")
    ax.set_xlabel("Pixel value")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

elif step == 7:
    split = st.radio("Split", ["train", "test"], horizontal=True, key="ex_split")
    x = D["x_train"] if split == "train" else D["x_test"]
    y = D["y_train"] if split == "train" else D["y_test"]

    first_or_random = st.selectbox("Selection mode", ["First occurrence", "Random occurrence"], index=0)
    rng = np.random.default_rng(42)

    fig, axes = plt.subplots(2, 5, figsize=(11, 4.5))
    for digit, ax in enumerate(axes.flatten()):
        idxs = np.where(y == digit)[0]
        idx = idxs[0] if first_or_random == "First occurrence" else int(rng.choice(idxs))
        ax.imshow(x[idx], cmap="gray")
        ax.set_title(f"Digit {digit}")
        ax.axis("off")
    fig.suptitle(f"Example for Each Digit ({split})")
    plt.tight_layout()
    st.pyplot(fig)

elif step == 8:
    split = st.radio("Split", ["train", "test"], horizontal=True, key="mean_split")
    x = D["x_train"] if split == "train" else D["x_test"]
    y = D["y_train"] if split == "train" else D["y_test"]

    cmap = st.selectbox("Colormap", ["inferno", "viridis", "magma", "gray"], index=0)

    fig, axes = plt.subplots(2, 5, figsize=(11, 4.5))
    for digit, ax in enumerate(axes.flatten()):
        mean_image = x[y == digit].mean(axis=0)
        ax.imshow(mean_image, cmap=cmap)
        ax.set_title(f"Mean {digit}")
        ax.axis("off")
    fig.suptitle(f"Mean Image per Digit ({split})")
    plt.tight_layout()
    st.pyplot(fig)

elif step == 9:
    split = st.radio("Split", ["train", "test"], horizontal=True, key="std_split")
    x = D["x_train"] if split == "train" else D["x_test"]
    y = D["y_train"] if split == "train" else D["y_test"]

    cmap = st.selectbox("Colormap", ["magma", "inferno", "plasma", "gray"], index=0)

    fig, axes = plt.subplots(2, 5, figsize=(11, 4.5))
    for digit, ax in enumerate(axes.flatten()):
        std_image = x[y == digit].std(axis=0)
        ax.imshow(std_image, cmap=cmap)
        ax.set_title(f"Std {digit}")
        ax.axis("off")
    fig.suptitle(f"Std Deviation per Digit ({split})")
    plt.tight_layout()
    st.pyplot(fig)


col1, col2 = st.columns(2)
with col1:
    if step == 0:
        if st.button("↺ Start Over"):
            st.session_state.step = 0
            st.session_state.mnist = None
            st.rerun()
    else:
        if st.button("← Previous"):
            st.session_state.step = max(0, step - 1)
            st.rerun()

with col2:
    if step == TOTAL_STEPS - 1:
        if st.button("↺ Start Over"):
            st.session_state.step = 0
            st.session_state.mnist = None
            st.rerun()
    else:
        if st.button("Next →"):
            st.session_state.step = min(TOTAL_STEPS - 1, step + 1)
            st.rerun()
