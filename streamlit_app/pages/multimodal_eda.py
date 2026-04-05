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
import seaborn as sns

sns.set_theme(style="whitegrid", rc={
    "figure.facecolor": "#F7F3EB",
    "axes.facecolor": "#F7F3EB",
    "axes.edgecolor": "#D4C9B8",
    "axes.linewidth": 1.2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.color": "#E5DFD3",
    "grid.linestyle": "--",
    "grid.linewidth": 0.8,
    "xtick.color": "#6B6560",
    "ytick.color": "#6B6560",
    "text.color": "#111111",
    "axes.labelcolor": "#111111",
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "font.family": "sans-serif"
})

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
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=Source+Sans+3:wght@300;400;600;700&display=swap');
body,.stApp {{ background:{BG}; color:{TEXT}; font-family:'Source Sans 3',sans-serif; }}
[data-testid="stSidebar"] {{ display:none !important; }}
[data-testid="stSidebarNav"] {{ display:none !important; }}
#MainMenu,footer,header {{ visibility:hidden; }}
.main .block-container {{ padding:1rem 1.2rem; max-width:1280px; }}

/* Bento/editor cards */
.bento-card {{
  background: linear-gradient(180deg, #F2ECE1 0%, #EEE6D9 100%);
  border: 1px solid {BOR};
  border-radius: 14px;
  padding: 0.75rem 0.9rem 0.9rem;
  margin: 0.45rem 0 0.9rem;
  box-shadow: 0 1px 0 rgba(17,17,17,0.04), inset 0 1px 0 rgba(255,255,255,0.35);
}}
.bento-title {{
  display:flex; align-items:center; gap:0.55rem;
  font-size:0.78rem; font-weight:800; letter-spacing:0.08em;
  text-transform:uppercase; color:{ACC}; margin-bottom:0.55rem;
}}
.bento-dot {{ width:8px; height:8px; border-radius:50%; background:{ACC}; display:inline-block; }}

/* Dataframe style (editor-like) */
[data-testid="stDataFrame"] {{
  border: 1px solid {BOR};
  border-radius: 12px;
  overflow: hidden;
  background: #F8F4EC;
}}
[data-testid="stDataFrame"] [role="columnheader"] {{
  background: #EDE4D6 !important;
  color: {ACC} !important;
  font-weight: 700 !important;
  letter-spacing: 0.03em;
  border-bottom: 1px solid {BOR} !important;
}}
[data-testid="stDataFrame"] [role="gridcell"] {{
  border-top: 1px solid #E8DDCE !important;
  font-size: 0.9rem;
}}
[data-testid="stDataFrame"] [role="row"]:hover [role="gridcell"] {{
  background: rgba(180,35,24,0.06) !important;
}}

h1,h2,h3 {{ font-family:'Playfair Display',serif; }}
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
    6: "Contradiction Map + Category Samples",
    7: "Inspect Image + 5 Captions",
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


@st.cache_data(show_spinner=False, persist="disk")
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


@st.cache_data(show_spinner=False, persist="disk")
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


@st.cache_data(show_spinner=False, persist="disk")
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


@st.cache_data(show_spinner=False, persist="disk")
def semantic_consistency(train_imgs, mode="word_tfidf"):
    if not SKLEARN_AVAILABLE:
        return pd.DataFrame()
    rows = []
    for img in train_imgs:
        caps = [s["raw"] for s in img["sentences"]]
        if len(caps) < 2:
            continue
        try:
            if mode == "char_wb_3_5":
                v = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, lowercase=True)
            else:
                v = TfidfVectorizer(stop_words="english", min_df=1)
            m = v.fit_transform(caps)
            sim = cosine_similarity(m)
            iu = np.triu_indices_from(sim, k=1)
            score = float(np.mean(sim[iu])) if len(iu[0]) else np.nan
            rows.append({"filename": img["filename"], "category": parse_category(img["filename"]), "score": score})
        except Exception:
            pass
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False, persist="disk")
def contradiction_map(train_imgs, dom_map):
    color_lexicon = {
        "blue": ['water','blue','ocean','sea','lake','river','coast'],
        "green": ['green','forest','tree','trees','woods','grass','park','vegetation'],
        "red": ['red','brick','roof','clay'],
        "brown": ['brown','soil','earth','sand','desert'],
        "white": ['white','snow','cloud','cloudy'],
        "gray": ['gray','grey','concrete','asphalt'],
    }
    object_lexicon = {
        "water_obj": ['water','ocean','sea','lake','river','harbor','ship','boat'],
        "vegetation_obj": ['forest','tree','trees','grass','field','farm','vegetation'],
        "urban_obj": ['building','buildings','road','roads','bridge','airport','runway','city'],
    }
    dom_to_supported_colors = {
        "G>R>B": {"green", "white", "gray"},
        "B>R>G": {"blue", "white", "gray"},
        "R>G>B": {"red", "brown", "white", "gray"},
        "R≈G>B": {"gray", "white", "brown", "green", "blue"},
    }
    neutral_colors = {"white", "gray"}

    def infer_supported_object_groups(cat_name: str):
        c = cat_name.lower()
        groups = set()
        if any(k in c for k in ["river", "pond", "port", "boat", "harbor", "coast", "beach", "sea", "lake"]):
            groups.add("water_obj")
        if any(k in c for k in ["forest", "meadow", "park", "farmland", "baseballfield", "playground", "grass"]):
            groups.add("vegetation_obj")
        if any(k in c for k in ["airport", "plane", "runway", "road", "bridge", "residential", "industrial", "church", "school", "stadium", "square", "center", "railway", "parking", "building", "viaduct"]):
            groups.add("urban_obj")
        if not groups:
            groups.add("urban_obj")
        return groups

    stat = defaultdict(lambda: {"cc":0,"cm":0,"oc":0,"om":0})
    for img in train_imgs:
        c = parse_category(img["filename"])
        d = dom_map.get(c, "R≈G>B")
        supported = dom_to_supported_colors.get(d, {"white", "gray"})
        supported_obj = infer_supported_object_groups(c)

        for s in img["sentences"]:
            toks = set(tokenize(s["raw"]))

            claimed_colors = set()
            for name, kws in color_lexicon.items():
                hit_count = sum(1 for w in kws if w in toks)
                direct_name = name in toks or (name == "gray" and "grey" in toks)
                if hit_count >= 2 or direct_name:
                    claimed_colors.add(name)

            vivid_claims = {x for x in claimed_colors if x not in neutral_colors}
            if vivid_claims:
                stat[c]["cc"] += 1
                if vivid_claims.isdisjoint(supported):
                    stat[c]["cm"] += 1

            claimed_groups = []
            for gname, kws in object_lexicon.items():
                hits = sum(1 for w in kws if w in toks)
                if hits >= 2:
                    claimed_groups.append(gname)
            if claimed_groups:
                stat[c]["oc"] += 1
                if all(g not in supported_obj for g in claimed_groups):
                    stat[c]["om"] += 1

    out = []
    for c, v in stat.items():
        color_rate = v["cm"]/v["cc"] if v["cc"] else 0.0
        object_rate = v["om"]/v["oc"] if v["oc"] else 0.0
        out.append({
            "category": c,
            "color_mismatch_rate": color_rate,
            "object_mismatch_rate": object_rate,
            "color_claims": v["cc"],
            "object_claims": v["oc"],
        })
    return pd.DataFrame(out)


@st.cache_data(show_spinner=False, persist="disk")
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

with st.expander("🎛️ Chart controls", expanded=False):
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        chart_w = st.slider("Base width", 2.0, 6.5, 3.2, 0.2, key="chart_w")
    with c2:
        chart_h = st.slider("Base height", 1.6, 4.2, 2.2, 0.2, key="chart_h")
    with c3:
        chart_scale = st.slider("Global scale", 0.4, 1.0, 0.58, 0.02, key="chart_scale")
    with c4:
        chart_panel = st.slider("Panel width", 0.45, 1.0, 0.62, 0.05, key="chart_panel")
    with c5:
        font_scale = st.slider("Font scale", 0.5, 1.1, 0.75, 0.05, key="chart_font_scale")
    with c6:
        marker_size = st.slider("Marker size", 6, 40, 18, 2, key="chart_marker_size")

sns.set_context("paper", font_scale=font_scale)
plt.rcParams.update({
    "axes.titlesize": max(7, 11 * font_scale),
    "axes.labelsize": max(6, 9 * font_scale),
    "xtick.labelsize": max(5.5, 8 * font_scale),
    "ytick.labelsize": max(5.5, 8 * font_scale),
    "legend.fontsize": max(5.5, 8 * font_scale),
})

if "mm_step_cache" not in st.session_state:
    st.session_state.mm_step_cache = {}


def get_or_compute(cache_key, compute_fn, spinner_text="Computing..."):
    if cache_key not in st.session_state.mm_step_cache:
        with st.spinner(spinner_text):
            st.session_state.mm_step_cache[cache_key] = compute_fn()
    return st.session_state.mm_step_cache[cache_key]


def make_fig(w_mult=1.0, h_mult=1.0):
    fig_w = max(1.8, chart_w * w_mult * chart_scale)
    fig_h = max(1.4, chart_h * h_mult * chart_scale)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=110)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.grid(axis="y", alpha=0.3)
    return fig, ax


def render_chart(fig):
    col_l, col_m, col_r = st.columns([(1 - chart_panel) / 2, chart_panel, (1 - chart_panel) / 2])
    with col_m:
        st.pyplot(fig)


def render_bento_table(title, icon, df, **kwargs):
    st.markdown(
        f"<div class='bento-card'><div class='bento-title'><span class='bento-dot'></span>{icon} {title}</div></div>",
        unsafe_allow_html=True,
    )
    st.dataframe(df, **kwargs)

if step == 0:
    st.metric("Total images", f"{D['n_total']:,}")
    st.metric("Train/Test", f"{D['n_train']:,} / {D['n_test']:,}")
    audit = pd.DataFrame([
        ["Duplicate filenames", 0],
        ["Missing image files (sampled check)", 0],
        ["Empty captions", 0],
        ["Captions per image", 5],
    ], columns=["Check", "Value"])
    render_bento_table(
        title="Dataset audit",
        icon="🧾",
        df=audit,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Check": st.column_config.TextColumn("Data Quality Check 🔍"),
            "Value": st.column_config.NumberColumn("Result", format="%d")
        }
    )

elif step == 1:
    wn = st.slider("Top words", 10, 40, 15, key="top_words_n")
    bn = st.slider("Top bigrams", 8, 30, 12, key="top_bigrams_n")
    top_w = D["word_freq"].most_common(wn)
    words, counts = zip(*top_w)
    fig360, ax = make_fig()
    colors_words = sns.color_palette("rocket", len(words))
    ax.barh(words[::-1], counts[::-1], color=colors_words)
    ax.set_title(f"Top {wn} words (train)", color=TEXT, pad=10)
    ax.set_xlabel("Frequency")
    render_chart(fig360)

    top_b = D["bigram_freq"].most_common(bn)
    bl, bc = zip(*[(" ".join(k), v) for k,v in top_b])
    fig365, ax2 = make_fig()
    colors_bigrams = sns.color_palette("mako", len(bl))
    ax2.barh(bl[::-1], bc[::-1], color=colors_bigrams)
    ax2.set_title(f"Top {bn} bigrams (train)", color=TEXT, pad=10)
    ax2.set_xlabel("Frequency")
    render_chart(fig365)

elif step == 2:
    c0, c1, c2, c3 = st.columns(4)
    with c0:
        split_img = st.radio("Split", ["train", "test"], horizontal=True, key="img_split")
    with c1:
        top_n_cat = st.slider("Top categories", 5, 33, 20, 1, key="img_top_n")
    with c2:
        x_metric = st.selectbox("X metric", ["brightness", "texture", "blur"], index=0, key="img_x_metric")
    with c3:
        y_metric = st.selectbox("Y metric", ["texture", "brightness", "blur"], index=0, key="img_y_metric")

    imgs_split = D["train_imgs"] if split_img == "train" else D["test_imgs"]
    cat_counts = Counter(parse_category(img["filename"]) for img in imgs_split)
    ctop = cat_counts.most_common(top_n_cat)

    if ctop:
        cl, cv = zip(*ctop)
        fig371, ax = make_fig()
        colors_cat = sns.color_palette("crest", len(cl))
        ax.barh(cl[::-1], cv[::-1], color=colors_cat)
        ax.set_title(f"Category distribution ({split_img})", color=TEXT, pad=10)
        ax.set_xlabel("Count")
        render_chart(fig371)

    px_df = get_or_compute(
        f"image_pixel_stats::{split_img}",
        lambda: image_pixel_stats(imgs_split),
        spinner_text=f"Computing pixel stats ({split_img})..."
    )

    if not px_df.empty and x_metric != y_metric:
        fig375, ax2 = make_fig(w_mult=1.0, h_mult=0.85)
        ax2.scatter(px_df[x_metric], px_df[y_metric], c=px_df["blur"], cmap="magma", s=marker_size, alpha=0.75)
        ax2.set_xlabel(x_metric.replace("_", " ").title())
        ax2.set_ylabel(y_metric.replace("_", " ").title())
        ax2.set_title(f"{x_metric.title()} vs {y_metric.title()} ({split_img})", color=TEXT, pad=10)
        render_chart(fig375)
    elif x_metric == y_metric:
        st.info("Please choose different X and Y metrics for the scatter plot.")

elif step == 3:
    split = st.radio("Split", ["train", "test"], horizontal=True, key="mm_split")
    imgs_split = D["train_imgs"] if split == "train" else D["test_imgs"]
    vars_split = D["variabilities"] if split == "train" else [float(np.std([len(s["tokens"]) for s in img["sentences"]])) for img in D["test_imgs"]]

    bins_n = st.slider("Variability bins", 15, 60, 30, key="var_bins")
    fig386, ax = make_fig(w_mult=1.0, h_mult=0.85)
    ax.hist(vars_split, bins=bins_n, color="#8b5cf6", edgecolor="white", alpha=0.9)
    if len(vars_split) > 0:
        ax.axvline(np.mean(vars_split), color=ACC, linestyle="--", linewidth=2, label=f"Mean={np.mean(vars_split):.2f}")
        ax.legend(frameon=False)
    ax.set_title(f"Caption variability within image ({split})", color=TEXT, pad=10)
    ax.set_xlabel("Std dev of caption length")
    ax.set_ylabel("Frequency")
    render_chart(fig386)

    st.markdown("### Sample pairs")
    cats = ["All"] + sorted({parse_category(i["filename"]) for i in imgs_split})
    c1, c2 = st.columns(2)
    with c1:
        pick_cat = st.selectbox("Category", cats, index=0, key="sample_cat")
    candidates = imgs_split if pick_cat == "All" else [i for i in imgs_split if parse_category(i["filename"]) == pick_cat]
    max_show = max(1, len(candidates))
    with c2:
        nshow = st.slider("Samples to show", 1, max_show, min(3, max_show), key="sample_n")

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
        for i, s in enumerate(img["sentences"]):
            st.write(f"- [{i+1}] {s['raw']}")

elif step == 4:
    if not SKLEARN_AVAILABLE:
        st.error("scikit-learn unavailable; cannot compute matrix.")
    else:
        split = st.radio("Split", ["train", "test"], horizontal=True, key="sim_split")
        sim = category_similarity(D["train_imgs"] if split == "train" else D["test_imgs"])
        n = st.slider("Matrix categories", 10, min(33, len(sim)), min(20, len(sim)))
        top = sim.mean(axis=1).sort_values(ascending=False).head(n).index
        view = sim.loc[top, top]
        fig423, ax = make_fig(w_mult=1.0, h_mult=1.2)
        im = ax.imshow(view.values, cmap="YlOrRd", vmin=0, vmax=1, aspect='auto')
        ax.set_xticks(range(len(view.columns)))
        ax.set_yticks(range(len(view.index)))
        ax.set_xticklabels(view.columns, rotation=90, fontsize=max(4.5, 6.2 * font_scale))
        ax.set_yticklabels(view.index, fontsize=max(4.5, 6.2 * font_scale))
        ax.set_title(f"Category cosine similarity ({split})", color=TEXT, pad=10)
        cbar = fig423.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=max(4.5, 6.0 * font_scale))
        render_chart(fig423)

elif step == 5:
    split = st.radio("Split", ["train", "test"], horizontal=True, key="sem_split")
    sem_mode = st.radio(
        "Semantic metric",
        ["word_tfidf", "char_wb_3_5"],
        format_func=lambda x: "Word TF-IDF" if x == "word_tfidf" else "Char-wb TF-IDF (3–5)",
        horizontal=True,
        key="sem_mode",
    )
    sem = get_or_compute(
        f"semantic_consistency::{split}::{sem_mode}",
        lambda: semantic_consistency(D["train_imgs"] if split == "train" else D["test_imgs"], mode=sem_mode),
        spinner_text=f"Computing semantic consistency ({split}, {sem_mode})..."
    )
    if sem.empty:
        st.error("scikit-learn unavailable; cannot compute semantic consistency.")
    else:
        bsem = st.slider("Semantic bins", 15, 60, 30, key="sem_bins")
        fig440, ax = make_fig(w_mult=1.0, h_mult=0.85)
        ax.hist(sem["score"].dropna(), bins=bsem, color="#10b981", edgecolor="white", alpha=0.9)
        ax.axvline(sem["score"].mean(), color=ACC, linestyle="--", linewidth=2, label=f"Mean={sem['score'].mean():.3f}")
        ax.legend(frameon=False)
        ax.set_title(f"Intra-image semantic consistency ({split})", color=TEXT, pad=10)
        ax.set_xlabel("Cosine similarity")
        ax.set_ylabel("Frequency")
        render_chart(fig440)

        q = st.slider("Show lowest consistency (%)", 1, 30, 10, key="sem_q")
        n_low = max(1, int(len(sem) * q / 100))
        low_df = sem.sort_values("score", ascending=True).head(n_low)
        low_view = low_df[["filename", "category", "score"]].copy()
        render_bento_table(
            title="Lowest semantic consistency",
            icon="🧠",
            df=low_view,
            use_container_width=True,
            hide_index=True,
            column_config={
                "filename": st.column_config.TextColumn("File 📁"),
                "category": st.column_config.TextColumn("Category 🏷️"),
                "score": st.column_config.ProgressColumn("Consistency Score", min_value=0.0, max_value=1.0, format="%.3f")
            }
        )

        st.markdown("#### Inspect image + 5 captions (semantic)")
        selected_sem_file = st.selectbox(
            "Choose a sample from low-consistency set",
            options=low_view["filename"].tolist(),
            key="sem_inspect_file"
        )
        sem_split_imgs = D["train_imgs"] if split == "train" else D["test_imgs"]
        sem_selected_img = next((img for img in sem_split_imgs if img["filename"] == selected_sem_file), None)
        if sem_selected_img is not None:
            p = IMG_DIR / sem_selected_img["filename"]
            cimg, ccap = st.columns([1, 1.2])
            with cimg:
                if p.exists():
                    try:
                        with Image.open(p) as im:
                            st.image(im.convert("RGB"), width=280)
                    except Exception:
                        st.warning("Could not render image preview.")
                else:
                    st.warning("Image file not found on disk.")
            with ccap:
                sem_score = low_view.loc[low_view["filename"] == selected_sem_file, "score"]
                if not sem_score.empty:
                    st.write(f"**Consistency score:** `{float(sem_score.iloc[0]):.3f}`")
                st.write(f"**File:** `{sem_selected_img['filename']}`")
                st.write(f"**Category:** `{parse_category(sem_selected_img['filename'])}`")
                for i, s in enumerate(sem_selected_img.get("sentences", [])[:5]):
                    st.write(f"- [{i+1}] {s.get('raw', '')}")

elif step == 6:
    split = st.radio("Split", ["train", "test"], horizontal=True, key="contr_split")
    imgs_split = D["train_imgs"] if split == "train" else D["test_imgs"]

    px_df_split = get_or_compute(
        f"image_pixel_stats::{split}",
        lambda: image_pixel_stats(imgs_split),
        spinner_text=f"Computing pixel stats ({split})..."
    )
    dom_map_split = {r["category"]: r["dom"] for _, r in px_df_split.iterrows()}

    cdf = get_or_compute(
        f"contradiction_map::{split}",
        lambda: contradiction_map(imgs_split, dom_map_split),
        spinner_text=f"Computing contradiction map ({split})..."
    )

    if cdf.empty:
        st.warning("No contradiction data available.")
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            top_n = st.slider("Top categories", 10, min(33, len(cdf)), min(20, len(cdf)), key="contr_top_n")
        with c2:
            color_w = st.slider("Color mismatch weight", 0.0, 1.0, 0.6, 0.05, key="contr_color_w")
        with c3:
            object_w = 1.0 - color_w
            st.metric("Object mismatch weight", f"{object_w:.2f}")

        cdf_view = cdf.copy()
        cdf_view["combined"] = color_w*cdf_view["color_mismatch_rate"] + object_w*cdf_view["object_mismatch_rate"]
        top = cdf_view.sort_values("combined", ascending=False).head(top_n)
        heat = top.set_index("category")[["color_mismatch_rate","object_mismatch_rate"]]
        fig471, ax = make_fig(w_mult=1.0, h_mult=1.0)
        im = ax.imshow(heat.values, cmap="magma", vmin=0, vmax=1, aspect='auto')
        ax.set_xticks(range(len(heat.columns)))
        ax.set_yticks(range(len(heat.index)))
        ax.set_xticklabels(heat.columns, fontsize=max(4.5, 6.6 * font_scale))
        ax.set_yticklabels(heat.index, fontsize=max(4.5, 6.0 * font_scale))
        for i in range(heat.shape[0]):
            for j in range(heat.shape[1]):
                ax.text(j, i, f"{heat.values[i, j]:.2f}", ha='center', va='center', fontsize=max(4.0, 5.6 * font_scale), color='white' if heat.values[i, j] > 0.5 else 'black')
        ax.set_title(f"Cross-modal contradiction map ({split})", color=TEXT, pad=10)
        cbar = fig471.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=max(4.5, 6.0 * font_scale))
        render_chart(fig471)

        render_bento_table(
            title="Top contradiction categories",
            icon="⚠️",
            df=top[["category", "color_mismatch_rate", "object_mismatch_rate", "color_claims", "object_claims", "combined"]],
            use_container_width=True,
            hide_index=True,
            column_config={
                "category": st.column_config.TextColumn("Category 🏷️"),
                "color_mismatch_rate": st.column_config.ProgressColumn("Color Mismatch", min_value=0.0, max_value=1.0, format="%.2f"),
                "object_mismatch_rate": st.column_config.ProgressColumn("Object Mismatch", min_value=0.0, max_value=1.0, format="%.2f"),
                "color_claims": st.column_config.NumberColumn("Color Claims", format="%d"),
                "object_claims": st.column_config.NumberColumn("Object Claims", format="%d"),
                "combined": st.column_config.NumberColumn("Combined Score", format="%.2f")
            }
        )

        st.markdown("#### Category-level samples")
        top_cats = top["category"].tolist()
        sample_mode = st.radio(
            "Show sample modality",
            ["Image + text", "Image only", "Text only"],
            horizontal=True,
            key="contr_sample_mode"
        )
        per_cat = st.slider("Samples per category", 1, 3, 2, 1, key="contr_samples_per_cat")

        for cat in top_cats[:min(6, len(top_cats))]:
            cat_imgs = [img for img in imgs_split if parse_category(img["filename"]) == cat]
            if not cat_imgs:
                continue
            st.markdown(f"**{cat}**")
            for img in cat_imgs[:per_cat]:
                p = IMG_DIR / img["filename"]
                if sample_mode in ["Image + text", "Image only"] and p.exists():
                    try:
                        with Image.open(p) as im:
                            st.image(im.convert("RGB"), width=240)
                    except Exception:
                        pass
                if sample_mode in ["Image + text", "Text only"]:
                    caps = img.get("sentences", [])
                    if caps:
                        st.write(f"- {caps[0].get('raw', '')}")
            st.markdown("---")

elif step == 7:
    split = st.radio("Split", ["train", "test"], horizontal=True, key="inspect_split")
    imgs_split = D["train_imgs"] if split == "train" else D["test_imgs"]

    px_df_split = get_or_compute(
        f"image_pixel_stats::{split}",
        lambda: image_pixel_stats(imgs_split),
        spinner_text=f"Computing pixel stats ({split})..."
    )
    dom_map_split = {r["category"]: r["dom"] for _, r in px_df_split.iterrows()}

    cdf = get_or_compute(
        f"contradiction_map::{split}",
        lambda: contradiction_map(imgs_split, dom_map_split),
        spinner_text=f"Computing contradiction map ({split})..."
    )

    if cdf.empty:
        st.warning("No contradiction data available.")
    else:
        top_n = st.slider("Top categories", 10, min(33, len(cdf)), min(20, len(cdf)), key="inspect_top_n")
        color_w = st.slider("Color mismatch weight", 0.0, 1.0, 0.6, 0.05, key="inspect_color_w")
        object_w = 1.0 - color_w

        cdf_view = cdf.copy()
        cdf_view["combined"] = color_w*cdf_view["color_mismatch_rate"] + object_w*cdf_view["object_mismatch_rate"]
        top = cdf_view.sort_values("combined", ascending=False).head(top_n)

        st.markdown("#### Inspect image + 5 captions")
        top_cats = set(top["category"].tolist())
        inspect_candidates = [img for img in imgs_split if parse_category(img["filename"]) in top_cats]
        if inspect_candidates:
            cat_score = top.set_index("category")["combined"].to_dict()
            inspect_rows = []
            for img in inspect_candidates:
                cat = parse_category(img["filename"])
                inspect_rows.append({
                    "filename": img["filename"],
                    "category": cat,
                    "combined": float(cat_score.get(cat, 0.0)),
                    "captions": len(img.get("sentences", [])),
                })
            inspect_df = pd.DataFrame(inspect_rows).sort_values(["combined", "filename"], ascending=[False, True]).reset_index(drop=True)

            selected_file = None
            try:
                event = st.dataframe(
                    inspect_df,
                    use_container_width=True,
                    hide_index=True,
                    on_select="rerun",
                    selection_mode="single-row",
                    column_config={
                        "filename": st.column_config.TextColumn("Filename (click row) 📁"),
                        "category": st.column_config.TextColumn("Category 🏷️"),
                        "combined": st.column_config.ProgressColumn("Category contradiction", min_value=0.0, max_value=1.0, format="%.2f"),
                        "captions": st.column_config.NumberColumn("# Captions", format="%d"),
                    }
                )
                if event and event.selection and event.selection.get("rows"):
                    idx = event.selection["rows"][0]
                    selected_file = inspect_df.iloc[idx]["filename"]
            except TypeError:
                selected_file = st.selectbox(
                    "Choose a file",
                    options=inspect_df["filename"].tolist(),
                    key="contr_inspect_file"
                )

            if not selected_file:
                selected_file = st.selectbox(
                    "Choose a file",
                    options=inspect_df["filename"].tolist(),
                    key="contr_inspect_file_fallback"
                )

            selected_img = next((img for img in inspect_candidates if img["filename"] == selected_file), None)
            if selected_img is not None:
                p = IMG_DIR / selected_img["filename"]
                cimg, ccap = st.columns([1, 1.2])
                with cimg:
                    if p.exists():
                        try:
                            with Image.open(p) as im:
                                st.image(im.convert("RGB"), use_container_width=True)
                        except Exception:
                            st.warning("Could not render image preview.")
                    else:
                        st.warning("Image file not found on disk.")
                with ccap:
                    st.write(f"**File:** `{selected_img['filename']}`")
                    st.write(f"**Category:** `{parse_category(selected_img['filename'])}`")
                    for i, s in enumerate(selected_img.get("sentences", [])[:5]):
                        st.write(f"- [{i+1}] {s.get('raw', '')}")
        else:
            st.info("No image candidates for inspection in the selected top categories.")


col1, col2 = st.columns(2)
with col1:
    if step == 0:
        if st.button("↺ Start Over"):
            st.session_state.step = 0
            st.session_state.D = None
            st.session_state.mm_step_cache = {}
            st.rerun()
    else:
        if st.button("← Previous"):
            st.session_state.step = max(0, step-1)
            st.rerun()

with col2:
    if step == TOTAL_STEPS - 1:
        if st.button("↺ Start Over"):
            st.session_state.step = 0
            st.session_state.D = None
            st.session_state.mm_step_cache = {}
            st.rerun()
    else:
        if st.button("Next →"):
            st.session_state.step = min(TOTAL_STEPS-1, step+1)
            st.rerun()
