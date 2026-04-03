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
    st.dataframe(info.style.set_properties(**{'background-color': '#fafafa', 'color': '#333'}), use_container_width=True, hide_index=True)

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
    st.dataframe(props.style.set_properties(**{'background-color': '#f8f9fa', 'color': '#222'}), use_container_width=True, hide_index=True)

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
    split = st.radio("Split", ["train", "test"], horizontal=True, key="sample_split")
    x = D["x_train"] if split == "train" else D["x_test"]
    y = D["y_train"] if split == "train" else D["y_test"]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        only_digit = st.selectbox("Digit filter", ["All"] + [str(i) for i in range(10)], index=0)
    with c2:
        pick_mode = st.selectbox("Pick mode", ["Random", "Sequential"], index=0)
    with c3:
        seed = st.number_input("Random seed", min_value=0, max_value=99999, value=42, step=1)
    with c4:
        grid_cols = st.slider("Grid columns", 2, 20, 8, 1, key="sample_cols")

    if only_digit == "All":
        idx_pool = np.arange(len(x))
    else:
        idx_pool = np.where(y == int(only_digit))[0]

    allow_unlimited = st.checkbox("🔥 Mở khóa giới hạn Max Samples (Cho phép kéo tới mức tối đa)")
    max_show = len(idx_pool) if allow_unlimited else min(len(idx_pool), 400)
    n_show = st.slider("Samples to show", 1, max_show if max_show > 0 else 1, min(64, max_show if max_show > 0 else 1), 1, key="sample_n")

    rng = np.random.default_rng(int(seed))
    if len(idx_pool) > n_show:
        if pick_mode == "Random":
            idx_sel = rng.choice(idx_pool, size=n_show, replace=False)
        else:
            idx_sel = idx_pool[:n_show]
    else:
        idx_sel = idx_pool

    cols = min(int(grid_cols), max(1, len(idx_sel)))
    rows = int(np.ceil(len(idx_sel) / cols))

    if n_show > 500:
        st.warning("⚡ Đang Render bằng Fast Canvas (Raw NumPy Grid) do số lượng quá lớn. (Đã ẩn nhãn để chống Crash/Freeze)")
        img_h, img_w = x[0].shape
        grid = np.zeros((rows * img_h, cols * img_w), dtype=x.dtype)
        for i in range(len(idx_sel)):
            r, c = i // cols, i % cols
            idx = int(idx_sel[i])
            grid[r*img_h:(r+1)*img_h, c*img_w:(c+1)*img_w] = x[idx]
        st.image(grid, width=min(cols * 40, 1200), clamp=True)
    else:
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 1.6))
        axes = np.array(axes).reshape(rows, cols)
        for i, ax in enumerate(axes.flatten()):
            if i < len(idx_sel):
                idx = int(idx_sel[i])
                ax.imshow(x[idx], cmap="gray")
                ax.set_title(str(int(y[idx])), fontsize=8)
                ax.axis("off")
            else:
                ax.axis("off")
        fig.suptitle(f"Sample MNIST Images ({split})")
        plt.tight_layout()
        st.pyplot(fig)

    st.caption(f"Showing {len(idx_sel):,} samples from pool of {len(idx_pool):,}.")

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

    df_counts = pd.DataFrame({"digit": counts.index, "count": counts.values})
    st.dataframe(
        df_counts, 
        use_container_width=True, 
        hide_index=True,
        column_config={
            "digit": st.column_config.TextColumn("Digit 🔢"),
            "count": st.column_config.ProgressColumn("Count 📊", min_value=0, max_value=int(df_counts["count"].max()), format="%d")
        }
    )

elif step == 6:
    split = st.radio("Split", ["train", "test"], horizontal=True, key="pix_split")
    x = D["x_train"] if split == "train" else D["x_test"]
    pixels = x.flatten()

    bins = st.slider("Bins", 20, 150, 50, key="pix_bins")
    log_scale = st.toggle("Log y-scale", value=True, key="pix_log")

    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.hist(pixels, bins=bins, color="#dc2626", alpha=0.85)
    if log_scale:
        ax.set_yscale("log")
    ax.set_title(f"Pixel Intensity Distribution ({split})")
    ax.set_xlabel("Pixel value")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

    digit_stats = []
    for d in range(10):
        dimgs = x[(D["y_train"] if split == "train" else D["y_test"]) == d]
        digit_stats.append(
            {
                "digit": d,
                "mean_pixel": float(dimgs.mean()),
                "std_pixel": float(dimgs.std()),
                "ink_ratio_(>0)": float((dimgs > 0).mean()),
            }
        )
    df_stats = pd.DataFrame(digit_stats)
    st.dataframe(
        df_stats, 
        use_container_width=True, 
        hide_index=True,
        column_config={
            "digit": st.column_config.TextColumn("Digit 🔢"),
            "mean_pixel": st.column_config.ProgressColumn("Mean Pixel 💡", min_value=0, max_value=255, format="%.2f"),
            "std_pixel": st.column_config.NumberColumn("Std Dev 📉", format="%.2f"),
            "ink_ratio_(>0)": st.column_config.ProgressColumn("Ink Ratio 🖋️", min_value=0.0, max_value=1.0, format="%.3f")
        }
    )

elif step == 7:
    split = st.radio("Split", ["train", "test"], horizontal=True, key="ex_split")
    x = D["x_train"] if split == "train" else D["x_test"]
    y = D["y_train"] if split == "train" else D["y_test"]

    first_or_random = st.selectbox("Selection mode", ["First occurrence", "Random occurrence"], index=0)
    rng = np.random.default_rng(int(st.number_input("Seed", min_value=0, max_value=99999, value=7, step=1, key="ex_seed")))

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

    st.markdown("### Top-k sample analytics")
    d = st.slider("Target Digit", 0, 9, 0, key="outlier_digit")
    k = st.slider("Top-k samples", 1, 30, 12, key="outlier_k")
    
    basis_opts = ["To class mean", "To class median", "To nearest other class (Ambiguity)"]
    mode = st.selectbox("Distance basis / Metric", basis_opts, index=0)

    dimgs = x[y == d]
    dflat = dimgs.reshape(len(dimgs), -1)
    
    if mode == "To class mean":
        ref = dimgs.mean(axis=0).reshape(-1)
        dist = np.linalg.norm(dflat - ref, axis=1)
        title_suffix = "farthest from mean"
    elif mode == "To class median":
        ref = np.median(dimgs, axis=0).reshape(-1)
        dist = np.linalg.norm(dflat - ref, axis=1)
        title_suffix = "farthest from median"
    else:
        # Nearest other class: find samples that look most like another digit
        other_means = []
        for other_d in range(10):
            if other_d == d: continue
            other_means.append(x[y == other_d].mean(axis=0).reshape(-1))
        other_means = np.array(other_means)
        
        # For each image of digit 'd', find distance to the CLOSEST mean of a DIFFERENT digit
        # High similarity (low distance) to another class mean = Ambiguous
        dists_to_others = []
        for i in range(len(dflat)):
            dists_to_others.append(np.linalg.norm(other_means - dflat[i], axis=1).min())
        dist = np.array(dists_to_others)
        # We want the SMALLEST distances to other classes (most ambiguous)
        # To reuse the "top_idx" logic (which takes largest), we invert or just change the sort
        dist = -dist # Smallest distance becomes largest value
        title_suffix = "most ambiguous (nearest to other classes)"

    top_idx = np.argsort(dist)[-k:][::-1]

    cols = min(6, k)
    rows = int(np.ceil(k / cols))
    fig2, axes2 = plt.subplots(rows, cols, figsize=(cols * 1.8, rows * 1.8))
    axes2 = np.array(axes2).reshape(rows, cols)
    for i, ax in enumerate(axes2.flatten()):
        if i < len(top_idx):
            idx = int(top_idx[i])
            actual_dist = -dist[idx] if mode == "To nearest other class (Ambiguity)" else dist[idx]
            ax.imshow(dimgs[idx], cmap="gray")
            ax.set_title(f"dist={actual_dist:.1f}", fontsize=8)
            ax.axis("off")
        else:
            ax.axis("off")
    fig2.suptitle(f"Digit {d} · Top-{k} {title_suffix} ({split})")
    plt.tight_layout()
    st.pyplot(fig2)

elif step == 8:
    split = st.radio("Split", ["train", "test"], horizontal=True, key="mean_split")
    x = D["x_train"] if split == "train" else D["x_test"]
    y = D["y_train"] if split == "train" else D["y_test"]

    cmap = st.selectbox("Colormap", ["inferno", "viridis", "magma", "gray"], index=0)

    means = []
    fig, axes = plt.subplots(2, 5, figsize=(11, 4.5))
    for digit, ax in enumerate(axes.flatten()):
        mean_image = x[y == digit].mean(axis=0)
        means.append(mean_image.reshape(-1))
        ax.imshow(mean_image, cmap=cmap)
        ax.set_title(f"Mean {digit}")
        ax.axis("off")
    fig.suptitle(f"Mean Image per Digit ({split})")
    plt.tight_layout()
    st.pyplot(fig)

    means = np.array(means)
    norm = np.linalg.norm(means, axis=1, keepdims=True)
    cosine = (means @ means.T) / np.clip(norm @ norm.T, 1e-8, None)

    fig2, ax2 = plt.subplots(figsize=(6, 5))
    im = ax2.imshow(cosine, cmap="YlGnBu", vmin=0, vmax=1)
    ax2.set_xticks(range(10)); ax2.set_yticks(range(10))
    ax2.set_xticklabels([str(i) for i in range(10)])
    ax2.set_yticklabels([str(i) for i in range(10)])
    ax2.set_title(f"Cosine Similarity between Mean Digits ({split})")
    fig2.colorbar(im, ax=ax2)
    st.pyplot(fig2)

elif step == 9:
    split = st.radio("Split", ["train", "test"], horizontal=True, key="std_split")
    x = D["x_train"] if split == "train" else D["x_test"]
    y = D["y_train"] if split == "train" else D["y_test"]

    cmap = st.selectbox("Colormap", ["magma", "inferno", "plasma", "gray"], index=0)

    fig, axes = plt.subplots(2, 5, figsize=(11, 4.5))
    outlier_rows = []
    for digit, ax in enumerate(axes.flatten()):
        dimgs = x[y == digit]
        std_image = dimgs.std(axis=0)
        mean_image = dimgs.mean(axis=0)
        d_flat = dimgs.reshape(len(dimgs), -1)
        m_flat = mean_image.reshape(-1)
        dist = np.linalg.norm(d_flat - m_flat, axis=1)
        near_idx = int(np.argmin(dist))
        far_idx = int(np.argmax(dist))
        outlier_rows.append({
            "digit": digit,
            "nearest_to_mean_idx": near_idx,
            "farthest_from_mean_idx": far_idx,
            "nearest_dist": float(dist[near_idx]),
            "farthest_dist": float(dist[far_idx]),
        })

        ax.imshow(std_image, cmap=cmap)
        ax.set_title(f"Std {digit}")
        ax.axis("off")
    fig.suptitle(f"Std Deviation per Digit ({split})")
    plt.tight_layout()
    st.pyplot(fig)

    df_outliers = pd.DataFrame(outlier_rows)
    st.dataframe(
        df_outliers, 
        use_container_width=True, 
        hide_index=True,
        column_config={
            "digit": st.column_config.TextColumn("Digit 🔢"),
            "nearest_to_mean_idx": st.column_config.NumberColumn("Nearest Idx"),
            "farthest_from_mean_idx": st.column_config.NumberColumn("Farthest Idx"),
            "nearest_dist": st.column_config.NumberColumn("Min Dist 🎯", format="%.1f"),
            "farthest_dist": st.column_config.ProgressColumn("Max Dist 🚀", min_value=0.0, max_value=float(df_outliers["farthest_dist"].max()), format="%.1f"),
        }
    )


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
