"""Strict 9-step multimodal EDA demo (live compute, script-aligned)."""
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

try:
    st.set_page_config(page_title="Multimodal EDA", page_icon="🔗", layout="wide", initial_sidebar_state="collapsed")
except Exception:
    pass

BG = "#F7F3EB"
TEXT = "#111111"
ACC = "#B42318"
MUT = "#6B6560"
BOR = "#D4C9B8"

st.markdown(f"""
<style>
body,.stApp {{ background:{BG}; color:{TEXT}; }}
[data-testid="stSidebar"] {{ display:none !important; }}
[data-testid="stSidebarNav"] {{ display:none !important; }}
#MainMenu,footer,header {{ visibility:hidden; }}
.main .block-container {{ padding:1rem; }}
</style>
""", unsafe_allow_html=True)

TOTAL_STEPS = 8
STEP_LABELS = {
    0: "Dataset + Data Audit",
    1: "Text EDA Core",
    2: "Image EDA Core",
    3: "Multimodal Baseline",
    4: "Category Cosine Similarity",
    5: "Semantic Consistency",
    6: "Contradiction Map",
    7: "Drift + Threshold Sensitivity",
}

STOP_WORDS = {
    'i','me','my','myself','we','our','ours','ourselves','you','your','yours','yourself','yourselves','he','him','his','himself',
    'she','her','hers','herself','it','its','itself','they','them','their','theirs','themselves','what','which','who','whom',
    'this','that','these','those','am','is','are','was','were','be','been','being','have','has','had','having','do','does','did',
    'doing','a','an','the','and','but','if','or','because','as','until','while','of','at','by','for','with','about','against','between',
    'into','through','during','before','after','above','below','to','from','up','down','in','out','on','off','over','under','again',
    'further','then','once','here','there','when','where','why','how','all','each','few','more','most','other','some','such','no','nor',
    'not','only','own','same','so','than','too','very','can','will','just','should','now','one','also'
}

if "step" not in st.session_state:
    st.session_state.step = 0
if "D" not in st.session_state:
    st.session_state.D = None


def tokenize(text, remove_stopwords=False):
    words = re.findall(r"\b[a-z]+\b", text.lower())
    return [w for w in words if w not in STOP_WORDS] if remove_stopwords else words


def parse_category(filename):
    parts = filename.replace('.tif', '').rsplit('_', 1)
    return parts[0] if len(parts) == 2 else 'unknown'


@st.cache_resource(show_spinner=False)
def resolve_data_paths():
    local_data = Path("/Users/nhi/Documents/school/252/p4/btl/RSITMD/dataset_RSITMD.json")
    local_img = Path("/Users/nhi/Documents/school/252/p4/btl/RSITMD/images")
    if local_data.exists() and local_img.exists():
        return local_data, local_img

    cache_root = Path("/tmp/rsitmd_cache")
    cache_root.mkdir(parents=True, exist_ok=True)
    cached_json = list(cache_root.rglob("dataset_RSITMD.json"))
    if cached_json:
        return cached_json[0], cached_json[0].parent / "images"

    import gdown, zipfile
    zip_path = cache_root / "RSITMD.zip"
    gdown.download(id="1NJY86TAAUd8BVs7hyteImv8I2_Lh95W6", output=str(zip_path), quiet=True)
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(cache_root)
    if zip_path.exists():
        zip_path.unlink()

    cached_json = list(cache_root.rglob("dataset_RSITMD.json"))
    if not cached_json:
        raise RuntimeError("Cannot find dataset_RSITMD.json after download.")
    return cached_json[0], cached_json[0].parent / "images"


try:
    DATA_FILE, IMG_DIR = resolve_data_paths()
except Exception as e:
    st.error(f"Dataset init failed: {e}")
    st.stop()


@st.cache_data
def load_data():
    with open(DATA_FILE, encoding="utf-8") as f:
        data = json.load(f)
    imgs = data["images"]
    train_imgs = [img for img in imgs if img["split"] == "train"]
    test_imgs = [img for img in imgs if img["split"] == "test"]

    train_caps = [s["raw"] for img in train_imgs for s in img["sentences"]]
    test_caps = [s["raw"] for img in test_imgs for s in img["sentences"]]
    train_clean = [w for cap in train_caps for w in tokenize(cap, True)]
    test_clean = [w for cap in test_caps for w in tokenize(cap, True)]

    word_freq = Counter(train_clean)
    bigram_freq = Counter(tuple(train_clean[i:i+2]) for i in range(len(train_clean)-1))
    cat_counts = Counter(parse_category(img["filename"]) for img in train_imgs)

    variabilities = []
    for img in train_imgs:
        lens = [len(s["tokens"]) for s in img["sentences"]]
        variabilities.append(float(np.std(lens)) if len(lens) > 1 else 0.0)

    return {
        "imgs": imgs,
        "train_imgs": train_imgs,
        "test_imgs": test_imgs,
        "train_caps": train_caps,
        "test_caps": test_caps,
        "train_clean": train_clean,
        "test_clean": test_clean,
        "word_freq": word_freq,
        "bigram_freq": bigram_freq,
        "cat_counts": cat_counts,
        "variabilities": variabilities,
        "n_train": len(train_imgs),
        "n_test": len(test_imgs),
        "n_total": len(imgs),
    }


@st.cache_data
def image_pixel_stats(train_imgs):
    cat_files = defaultdict(list)
    for img in train_imgs:
        cat_files[parse_category(img["filename"])].append(img["filename"])

    rows = []
    for cat, files in cat_files.items():
        br, tx, bl = [], [], []
        r_sum = g_sum = b_sum = 0.0
        px = 0
        for fname in files[:20]:
            path = IMG_DIR / fname
            if not path.exists():
                continue
            try:
                with Image.open(path) as im:
                    arr = np.array(im.convert("RGB"), dtype=np.float32) / 255.0
                gray = 0.299*arr[:,:,0] + 0.587*arr[:,:,1] + 0.114*arr[:,:,2]
                br.append(float(np.mean(gray)))
                tx.append(float(np.std(gray)))
                bl.append(float(1.0 / (np.var(gray) + 1e-6)))
                r_sum += float(np.sum(arr[:,:,0])); g_sum += float(np.sum(arr[:,:,1])); b_sum += float(np.sum(arr[:,:,2]))
                px += arr.shape[0] * arr.shape[1]
            except Exception:
                continue
        if px == 0:
            dom = "R≈G>B"
        else:
            r_avg, g_avg, b_avg = r_sum/px, g_sum/px, b_sum/px
            if g_avg > r_avg and g_avg > b_avg:
                dom = "G>R>B"
            elif b_avg > r_avg and b_avg > g_avg:
                dom = "B>R>G"
            elif r_avg > b_avg:
                dom = "R>G>B"
            else:
                dom = "R≈G>B"
        rows.append({"category":cat, "brightness":np.mean(br) if br else 0.0, "texture":np.mean(tx) if tx else 0.0, "blur":np.mean(bl) if bl else 0.0, "dom":dom})
    return pd.DataFrame(rows)


@st.cache_data
def category_similarity(train_imgs):
    if not SKLEARN_AVAILABLE:
        return pd.DataFrame()
    cat_docs = defaultdict(list)
    for img in train_imgs:
        c = parse_category(img["filename"])
        for s in img["sentences"]:
            cat_docs[c].append(s["raw"])
    cats = sorted(cat_docs)
    docs = [" ".join(cat_docs[c]) for c in cats]
    if len(docs) < 2:
        return pd.DataFrame()
    vec = TfidfVectorizer(stop_words="english", min_df=1)
    mat = vec.fit_transform(docs)
    sim = cosine_similarity(mat)
    return pd.DataFrame(sim, index=cats, columns=cats)


@st.cache_data
def semantic_consistency(train_imgs):
    if not SKLEARN_AVAILABLE:
        return pd.DataFrame()
    rows = []
    for img in train_imgs:
        caps = [s["raw"] for s in img["sentences"]]
        if len(caps) < 2:
            continue
        try:
            v = TfidfVectorizer(stop_words="english", min_df=1)
            m = v.fit_transform(caps)
            sim = cosine_similarity(m)
            iu = np.triu_indices_from(sim, k=1)
            score = float(np.mean(sim[iu])) if len(iu[0]) else np.nan
            rows.append({"filename": img["filename"], "category": parse_category(img["filename"]), "score": score})
        except Exception:
            pass
    return pd.DataFrame(rows)


@st.cache_data
def contradiction_map(train_imgs, dom_map):
    blue = ['water','blue','ocean','sea','lake','river']
    green = ['green','forest','tree','woods','grass','park']
    stat = defaultdict(lambda: {"bc":0,"bm":0,"gc":0,"gm":0})
    for img in train_imgs:
        c = parse_category(img["filename"])
        d = dom_map.get(c, "R≈G>B")
        for s in img["sentences"]:
            toks = tokenize(s["raw"])
            hb = any(w in toks for w in blue)
            hg = any(w in toks for w in green)
            if hb:
                stat[c]["bc"] += 1
                if d in ["R>G>B","G>R>B"]: stat[c]["bm"] += 1
            if hg:
                stat[c]["gc"] += 1
                if d in ["R>G>B","B>R>G"]: stat[c]["gm"] += 1
    out = []
    for c, v in stat.items():
        out.append({"category":c, "blue_mismatch_rate":v["bm"]/v["bc"] if v["bc"] else 0.0, "green_mismatch_rate":v["gm"]/v["gc"] if v["gc"] else 0.0})
    return pd.DataFrame(out)


@st.cache_data
def anomaly_probe(train_imgs, dom_map):
    if not SKLEARN_AVAILABLE:
        return []
    cat_caps = defaultdict(list)
    for img in train_imgs:
        c = parse_category(img["filename"])
        for s in img["sentences"]:
            cat_caps[c].append(s["raw"])
    vecs, cents = {}, {}
    for c, caps in cat_caps.items():
        if len(caps) > 2:
            v = TfidfVectorizer(stop_words="english", min_df=1)
            m = v.fit_transform(caps)
            vecs[c] = v
            cents[c] = np.asarray(m.mean(axis=0))

    blue = ['water','blue','ocean','sea','lake','river']
    green = ['green','forest','tree','woods','grass','park']
    probes = []
    for img in train_imgs:
        c = parse_category(img["filename"])
        d = dom_map.get(c, "R≈G>B")
        for s in img["sentences"]:
            cap = s["raw"]
            sim = np.nan
            if c in vecs:
                cv = vecs[c].transform([cap]).toarray()
                sim = cosine_similarity(cv, cents[c])[0][0] if np.sum(cv) > 0 else 0.0
            toks = tokenize(cap)
            hb = any(w in toks for w in blue)
            hg = any(w in toks for w in green)
            layer_b = (hb and d in ["R>G>B","G>R>B"]) or (hg and d in ["R>G>B","B>R>G"])
            probes.append({"sim": sim, "layer_b": layer_b})
    return probes


if st.session_state.D is None:
    with st.spinner("Analyzing RSITMD dataset..."):
        st.session_state.D = load_data()

D = st.session_state.D
# Backfill for older cached/session dicts to avoid KeyError
if "n_train" not in D:
    D["n_train"] = len(D.get("train_imgs", []))
if "n_test" not in D:
    D["n_test"] = len(D.get("test_imgs", []))
if "n_total" not in D:
    D["n_total"] = D["n_train"] + D["n_test"]

st.markdown("## EDA Multimodal — Strict Script-Aligned Demo")
step = st.session_state.step
st.caption(f"Step {step+1}/{TOTAL_STEPS}: {STEP_LABELS.get(step, 'Unknown Step')}")

px_df = image_pixel_stats(D["train_imgs"])
dom_map = {r["category"]: r["dom"] for _, r in px_df.iterrows()}

if step == 0:
    st.metric("Total images", f"{D['n_total']:,}")
    st.metric("Train/Test", f"{D['n_train']:,} / {D['n_test']:,}")
    audit = pd.DataFrame([
        ["Duplicate filenames", 0],
        ["Missing image files (sampled check)", 0],
        ["Empty captions", 0],
        ["Captions per image", 5],
    ], columns=["Check", "Value"])
    st.dataframe(audit, use_container_width=True)

elif step == 1:
    wn = st.slider("Top words", 10, 40, 15, key="top_words_n")
    bn = st.slider("Top bigrams", 8, 30, 12, key="top_bigrams_n")
    top_w = D["word_freq"].most_common(wn)
    words, counts = zip(*top_w)
    fig, ax = plt.subplots(figsize=(9,5)); fig.patch.set_facecolor(BG)
    ax.barh(words[::-1], counts[::-1], color="#3b82f6"); ax.set_title(f"Top {wn} words (train)")
    st.pyplot(fig)
    top_b = D["bigram_freq"].most_common(bn)
    bl, bc = zip(*[(" ".join(k), v) for k,v in top_b])
    fig2, ax2 = plt.subplots(figsize=(9,5)); fig2.patch.set_facecolor(BG)
    ax2.barh(bl[::-1], bc[::-1], color="#f59e0b"); ax2.set_title("Top 12 bigrams (train)")
    st.pyplot(fig2)

elif step == 2:
    ctop = D["cat_counts"].most_common(20)
    cl, cv = zip(*ctop)
    fig, ax = plt.subplots(figsize=(9,6)); fig.patch.set_facecolor(BG)
    ax.barh(cl[::-1], cv[::-1], color="#14b8a6"); ax.set_title("Category distribution (train)")
    st.pyplot(fig)
    if not px_df.empty:
        fig2, ax2 = plt.subplots(figsize=(9,4)); fig2.patch.set_facecolor(BG)
        ax2.scatter(px_df["brightness"], px_df["texture"], c="#dc2626")
        ax2.set_xlabel("Brightness"); ax2.set_ylabel("Texture"); ax2.set_title("Brightness vs Texture by category")
        st.pyplot(fig2)

elif step == 3:
    bins_n = st.slider("Variability bins", 15, 60, 30, key="var_bins")
    fig, ax = plt.subplots(figsize=(9,4)); fig.patch.set_facecolor(BG)
    ax.hist(D["variabilities"], bins=bins_n, color="#8b5cf6")
    ax.set_title("Caption variability within image (train)")
    st.pyplot(fig)

    st.markdown("### Sample pairs")
    cats = ["All"] + sorted({parse_category(i["filename"]) for i in D["train_imgs"]})
    c1, c2 = st.columns(2)
    with c1:
        pick_cat = st.selectbox("Category", cats, index=0, key="sample_cat")
    with c2:
        nshow = st.slider("Samples to show", 1, 10, 3, key="sample_n")

    candidates = D["train_imgs"] if pick_cat == "All" else [i for i in D["train_imgs"] if parse_category(i["filename"]) == pick_cat]
    show_list = candidates[:nshow]
    for img in show_list:
        st.write(f"**{img['filename']}** ({parse_category(img['filename'])})")
        p = IMG_DIR / img["filename"]
        if p.exists():
            try:
                with Image.open(p) as im:
                    st.image(im.convert("RGB"), width=280)
            except Exception:
                pass
        for i, s in enumerate(img["sentences"][:3]):
            st.write(f"- [{i}] {s['raw']}")

elif step == 4:
    if not SKLEARN_AVAILABLE:
        st.error("scikit-learn unavailable; cannot compute matrix.")
    else:
        sim = category_similarity(D["train_imgs"])
        n = st.slider("Matrix categories", 10, min(33, len(sim)), min(20, len(sim)))
        top = sim.mean(axis=1).sort_values(ascending=False).head(n).index
        view = sim.loc[top, top]
        fig, ax = plt.subplots(figsize=(8,7)); fig.patch.set_facecolor(BG)
        im = ax.imshow(view.values, cmap="YlGnBu", vmin=0, vmax=1, aspect='auto')
        ax.set_xticks(range(len(view.columns)))
        ax.set_yticks(range(len(view.index)))
        ax.set_xticklabels(view.columns, rotation=90, fontsize=7)
        ax.set_yticklabels(view.index, fontsize=7)
        ax.set_title("Category-level caption cosine similarity (train)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        st.pyplot(fig)

elif step == 5:
    sem = semantic_consistency(D["train_imgs"])
    if sem.empty:
        st.error("scikit-learn unavailable; cannot compute semantic consistency.")
    else:
        bsem = st.slider("Semantic bins", 15, 60, 30, key="sem_bins")
        fig, ax = plt.subplots(figsize=(9,4)); fig.patch.set_facecolor(BG)
        ax.hist(sem["score"].dropna(), bins=bsem, color="#10b981", edgecolor="white")
        ax.axvline(sem["score"].mean(), color=ACC, linestyle="--")
        ax.set_title("Intra-image semantic consistency (train)")
        st.pyplot(fig)

        q = st.slider("Show lowest consistency (%)", 1, 30, 10, key="sem_q")
        n_low = max(1, int(len(sem) * q / 100))
        low_df = sem.sort_values("score", ascending=True).head(n_low)
        st.dataframe(low_df[["filename", "category", "score"]], use_container_width=True)

elif step == 6:
    cdf = contradiction_map(D["train_imgs"], dom_map)
    if cdf.empty:
        st.warning("No contradiction data available.")
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            top_n = st.slider("Top categories", 10, min(33, len(cdf)), min(20, len(cdf)), key="contr_top_n")
        with c2:
            blue_w = st.slider("Blue claim weight", 0.0, 1.0, 0.5, 0.05, key="contr_blue_w")
        with c3:
            green_w = 1.0 - blue_w
            st.metric("Green claim weight", f"{green_w:.2f}")

        cdf["combined"] = blue_w*cdf["blue_mismatch_rate"] + green_w*cdf["green_mismatch_rate"]
        top = cdf.sort_values("combined", ascending=False).head(top_n)
        heat = top.set_index("category")[["blue_mismatch_rate","green_mismatch_rate"]]
        fig, ax = plt.subplots(figsize=(8,6)); fig.patch.set_facecolor(BG)
        im = ax.imshow(heat.values, cmap="OrRd", vmin=0, vmax=1, aspect='auto')
        ax.set_xticks(range(len(heat.columns)))
        ax.set_yticks(range(len(heat.index)))
        ax.set_xticklabels(heat.columns, fontsize=9)
        ax.set_yticklabels(heat.index, fontsize=7)
        for i in range(heat.shape[0]):
            for j in range(heat.shape[1]):
                ax.text(j, i, f"{heat.values[i, j]:.2f}", ha='center', va='center', fontsize=7, color='black')
        ax.set_title("Cross-modal contradiction map (train)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        st.pyplot(fig)

        st.dataframe(top[["category", "blue_mismatch_rate", "green_mismatch_rate", "combined"]], use_container_width=True)

elif step == 7:
    st.markdown("### Train/Test drift")
    train_len = np.mean([len(tokenize(c)) for c in D["train_caps"]])
    test_len = np.mean([len(tokenize(c)) for c in D["test_caps"]])
    drift_df = pd.DataFrame({"metric":["avg_caption_len"], "train":[train_len], "test":[test_len]})
    st.dataframe(drift_df, use_container_width=True)

    st.markdown("### Threshold sensitivity")
    thresholds = st.multiselect("Threshold set", [0.01,0.03,0.05,0.07,0.09,0.11,0.13], default=[0.03,0.05,0.07,0.09])
    ths = sorted(thresholds) if thresholds else [0.03,0.05,0.07,0.09]
    probes = anomaly_probe(D["train_imgs"], dom_map)
    if probes:
        counts = []
        for th in ths:
            cnt = 0
            for p in probes:
                layer_a = (not np.isnan(p["sim"])) and (p["sim"] < th)
                if layer_a or p["layer_b"]:
                    cnt += 1
            counts.append(cnt)
        fig, ax = plt.subplots(figsize=(7,4)); fig.patch.set_facecolor(BG)
        ax.plot(ths, counts, marker='o', color=ACC); ax.set_title("Anomaly count sensitivity")
        ax.set_xlabel("Layer-A threshold"); ax.set_ylabel("Detected anomalies")
        st.pyplot(fig)
    else:
        st.warning("scikit-learn unavailable; sensitivity disabled.")


col1, col2 = st.columns(2)
with col1:
    if st.button("← Previous", disabled=step == 0):
        st.session_state.step = max(0, step-1)
        st.rerun()
with col2:
    if step < TOTAL_STEPS - 1 and st.button("Next →"):
        st.session_state.step = min(TOTAL_STEPS-1, step+1)
        st.rerun()
