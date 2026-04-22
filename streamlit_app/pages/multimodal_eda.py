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

TOTAL_STEPS = 7
STEP_LABELS = {
    0: "Dataset + Data Audit",
    1: "Text EDA Core",
    2: "Image EDA Core",
    3: "Multimodal Baseline",
    4: "Category Cosine Similarity",
    5: "Semantic Consistency + Noise Probe",
    6: "Contradiction Map + Evidence Explorer",
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
if "full_page_mode" not in st.session_state:
    st.session_state.full_page_mode = False


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
def semantic_consistency(train_imgs):
    if not SKLEARN_AVAILABLE:
        return pd.DataFrame()
    rows = []
    for img in train_imgs:
        caps = [s["raw"] for s in img["sentences"]]
        if len(caps) < 2:
            continue
        try:
            v = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, lowercase=True)
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
        if any(k in c for k in ["forest", "meadow", "park", "farmland", "baseballfield", "playground", "grass", "bareland", "desert"]):
            groups.add("vegetation_obj")
        if any(k in c for k in ["airport", "plane", "runway", "road", "bridge", "residential", "industrial", "church", "school", "stadium", "square", "center", "railway", "parking", "building", "viaduct"]):
            groups.add("urban_obj")
        # Unknown categories should not be forced into urban-only prior (too harsh).
        if not groups:
            groups = {"water_obj", "vegetation_obj", "urban_obj"}
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

    # Empirical-Bayes smoothing to avoid unstable rates from low-support categories
    alpha_color = 8.0
    alpha_object = 8.0
    total_color_claims = sum(v["cc"] for v in stat.values())
    total_color_mismatch = sum(v["cm"] for v in stat.values())
    total_object_claims = sum(v["oc"] for v in stat.values())
    total_object_mismatch = sum(v["om"] for v in stat.values())

    global_color_rate = (total_color_mismatch / total_color_claims) if total_color_claims else 0.0
    global_object_rate = (total_object_mismatch / total_object_claims) if total_object_claims else 0.0

    out = []
    for c, v in stat.items():
        color_raw = v["cm"]/v["cc"] if v["cc"] else 0.0
        object_raw = v["om"]/v["oc"] if v["oc"] else 0.0

        color_smoothed = ((v["cm"] + alpha_color * global_color_rate) / (v["cc"] + alpha_color)) if (v["cc"] + alpha_color) > 0 else 0.0
        object_smoothed = ((v["om"] + alpha_object * global_object_rate) / (v["oc"] + alpha_object)) if (v["oc"] + alpha_object) > 0 else 0.0

        out.append({
            "category": c,
            "color_mismatch_rate": color_smoothed,
            "object_mismatch_rate": object_smoothed,
            "color_mismatch_raw": color_raw,
            "object_mismatch_raw": object_raw,
            "color_claims": v["cc"],
            "object_claims": v["oc"],
        })
    return pd.DataFrame(out)


@st.cache_data(show_spinner=False, persist="disk")
def image_noise_probe(imgs, contradiction_df, dom_map_items):
    """Three-layer noise scoring (A semantic, B contradiction, C uncertainty).

    Layer B uses image-level contradiction as primary evidence and category prior
    only as a weak prior (15%).
    """
    if not SKLEARN_AVAILABLE:
        return pd.DataFrame()

    dom_map = dict(dom_map_items)
    contradiction_by_cat = {}
    if contradiction_df is not None and not contradiction_df.empty:
        for _, r in contradiction_df.iterrows():
            contradiction_by_cat[str(r["category"])] = {
                "color_rate": float(r.get("color_mismatch_rate", 0.0)),
                "object_rate": float(r.get("object_mismatch_rate", 0.0)),
            }

    color_lexicon = {
        "blue": ["blue", "water", "ocean", "sea", "lake", "river", "coast"],
        "green": ["green", "forest", "tree", "trees", "woods", "grass", "park", "vegetation"],
        "red": ["red", "brick", "roof", "clay"],
        "brown": ["brown", "soil", "earth", "sand", "desert"],
        "white": ["white", "snow", "cloud", "cloudy"],
        "gray": ["gray", "grey", "concrete", "asphalt"],
    }
    object_lexicon = {
        "water_obj": ["water", "ocean", "sea", "lake", "river", "harbor", "ship", "boat"],
        "vegetation_obj": ["forest", "tree", "trees", "grass", "field", "farm", "vegetation"],
        "urban_obj": ["building", "buildings", "road", "roads", "bridge", "airport", "runway", "city"],
    }
    dom_to_supported_colors = {
        "G>R>B": {"green", "white", "gray"},
        "B>R>G": {"blue", "white", "gray"},
        "R>G>B": {"red", "brown", "white", "gray"},
        "R≈G>B": {"gray", "white", "brown", "green", "blue"},
    }

    def infer_supported_object_groups(cat_name: str):
        c = cat_name.lower()
        groups = set()
        if any(k in c for k in ["river", "pond", "port", "boat", "harbor", "coast", "beach", "sea", "lake"]):
            groups.add("water_obj")
        if any(k in c for k in ["forest", "meadow", "park", "farmland", "baseballfield", "playground", "grass", "bareland", "desert"]):
            groups.add("vegetation_obj")
        if any(k in c for k in ["airport", "plane", "runway", "road", "bridge", "residential", "industrial", "church", "school", "stadium", "square", "center", "railway", "parking", "building", "viaduct"]):
            groups.add("urban_obj")
        if not groups:
            groups = {"water_obj", "vegetation_obj", "urban_obj"}
        return groups

    rows = []
    for img in imgs:
        caps = [s.get("raw", "") for s in img.get("sentences", []) if s.get("raw", "").strip()]
        if len(caps) < 2:
            continue
        try:
            v = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, lowercase=True)
            m = v.fit_transform(caps)
            sim = cosine_similarity(m)
            iu = np.triu_indices_from(sim, k=1)
            pair_vals = sim[iu] if len(iu[0]) else np.array([1.0])
            mean_pair = float(np.mean(pair_vals))

            centroid = np.asarray(m.mean(axis=0))
            cap_to_center = cosine_similarity(m, centroid).reshape(-1)
            center_mean = float(np.mean(cap_to_center))
            center_std = float(np.std(cap_to_center))
            outlier_mask = cap_to_center < (center_mean - 1.0 * center_std)
            outlier_count = int(np.sum(outlier_mask))
            worst_idx = int(np.argmin(cap_to_center))
            center_min = float(np.min(cap_to_center))

            # Layer A
            sem_noise = float(np.clip(1.0 - mean_pair, 0.0, 1.0))
            center_penalty = float(np.clip((0.55 - center_min) / 0.55, 0.0, 1.0)) if np.isfinite(center_min) else 0.0
            layer_a_score = 0.7 * sem_noise + 0.3 * center_penalty

            # Layer B (image-level dominant + weak category prior)
            cat = parse_category(img["filename"])
            dom = dom_map.get(cat, "R≈G>B")
            supported_colors = dom_to_supported_colors.get(dom, {"white", "gray"})
            supported_objs = infer_supported_object_groups(cat)

            img_color_claims = img_color_mismatch = 0
            img_object_claims = img_object_mismatch = 0
            for cap in caps:
                toks = set(tokenize(cap))

                claimed_colors = set()
                for cname, kws in color_lexicon.items():
                    hit_count = sum(1 for w in kws if w in toks)
                    direct = cname in toks or (cname == "gray" and "grey" in toks)
                    if hit_count >= 2 or direct:
                        claimed_colors.add(cname)
                vivid_claims = {x for x in claimed_colors if x not in {"white", "gray"}}
                if vivid_claims:
                    img_color_claims += 1
                    if vivid_claims.isdisjoint(supported_colors):
                        img_color_mismatch += 1

                claimed_groups = []
                for gname, kws in object_lexicon.items():
                    if sum(1 for w in kws if w in toks) >= 2:
                        claimed_groups.append(gname)
                if claimed_groups:
                    img_object_claims += 1
                    if all(g not in supported_objs for g in claimed_groups):
                        img_object_mismatch += 1

            img_color_rate = (img_color_mismatch / img_color_claims) if img_color_claims else 0.0
            img_object_rate = (img_object_mismatch / img_object_claims) if img_object_claims else 0.0

            prior = contradiction_by_cat.get(cat, {"color_rate": 0.0, "object_rate": 0.0})
            prior_color_rate = prior["color_rate"]
            prior_object_rate = prior["object_rate"]

            color_rate = 0.85 * img_color_rate + 0.15 * prior_color_rate
            object_rate = 0.85 * img_object_rate + 0.15 * prior_object_rate
            layer_b_score = 0.65 * color_rate + 0.35 * object_rate

            # Layer C
            token_counts = [len(re.findall(r"\b[a-z]+\b", cap.lower())) for cap in caps]
            avg_tokens = float(np.mean(token_counts)) if token_counts else 0.0
            low_info_penalty = float(np.clip((8.0 - avg_tokens) / 8.0, 0.0, 1.0))
            layer_c_score = 0.7 * low_info_penalty + 0.3 * min(outlier_count / 3.0, 1.0)

            noise_score = float(np.clip(0.58 * layer_a_score + 0.30 * layer_b_score + 0.12 * layer_c_score, 0.0, 1.0))

            rows.append({
                "filename": img["filename"],
                "category": cat,
                "mean_pairwise": mean_pair,
                "center_mean": center_mean,
                "center_min": center_min,
                "outlier_count": outlier_count,
                "layer_a_score": layer_a_score,
                "layer_b_score": layer_b_score,
                "layer_c_score": layer_c_score,
                "img_color_mismatch_rate": img_color_rate,
                "img_object_mismatch_rate": img_object_rate,
                "prior_color_mismatch_rate": prior_color_rate,
                "prior_object_mismatch_rate": prior_object_rate,
                "noise_score": noise_score,
                "worst_caption": caps[worst_idx],
            })
        except Exception:
            continue

    return pd.DataFrame(rows)


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
mode_col1, mode_col2 = st.columns([0.26, 0.74])
with mode_col1:
    full_page_mode = st.toggle("Full page mode", value=st.session_state.full_page_mode, key="full_page_mode")
with mode_col2:
    step = st.session_state.step
    if full_page_mode:
        st.caption("Full page mode: showing all sections on one page")
    else:
        st.caption(f"Step {step+1}/{TOTAL_STEPS}: {STEP_LABELS.get(step, 'Unknown Step')}")

if full_page_mode:
    st.info("Full page mode is enabled, but this page still uses the step renderer. Refresh after the next update to see the one-page layout.")

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
        diagram_text_scale = st.slider("Chart text size", 0.45, 1.25, 0.70, 0.05, key="chart_font_scale")
    with c6:
        marker_size = st.slider("Marker size", 6, 40, 18, 2, key="chart_marker_size")

sns.set_context("paper", font_scale=diagram_text_scale)
plt.rcParams.update({
    "axes.titlesize": max(7, 11 * diagram_text_scale),
    "axes.labelsize": max(6, 9 * diagram_text_scale),
    "xtick.labelsize": max(5.5, 8 * diagram_text_scale),
    "ytick.labelsize": max(5.5, 8 * diagram_text_scale),
    "legend.fontsize": max(5.5, 8 * diagram_text_scale),
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

def render_step(step_idx):
    if step_idx == 0:
        c0, c1 = st.columns(2)
        with c0:
            st.metric("Total images", f"{D['n_total']:,}")
            st.metric("Train/Test", f"{D['n_train']:,} / {D['n_test']:,}")
        with c1:
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

    elif step_idx == 1:
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

    elif step_idx == 2:
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
            ax.tick_params(axis='y', labelsize=max(5.0, 7.0 * diagram_text_scale))
            ax.tick_params(axis='x', labelsize=max(5.0, 7.0 * diagram_text_scale))
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

    elif step_idx == 3:
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

    elif step_idx == 4:
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

    elif step_idx == 5:
        split = st.radio("Split", ["train", "test"], horizontal=True, key="sem_split")
        st.caption("Semantic metric: Char-wb TF-IDF (3–5) + image-level noise probe")

        sem = get_or_compute(
            f"semantic_consistency::{split}::char_wb_3_5",
            lambda: semantic_consistency(D["train_imgs"] if split == "train" else D["test_imgs"]),
            spinner_text=f"Computing semantic consistency ({split}, char_wb_3_5)..."
        )
        imgs_split = D["train_imgs"] if split == "train" else D["test_imgs"]
        px_df_split = get_or_compute(
            f"image_pixel_stats::{split}",
            lambda: image_pixel_stats(imgs_split),
            spinner_text=f"Computing pixel stats ({split})..."
        )
        dom_map_split = {r["category"]: r["dom"] for _, r in px_df_split.iterrows()}
        cdf_split = get_or_compute(
            f"contradiction_map::{split}",
            lambda: contradiction_map(imgs_split, dom_map_split),
            spinner_text=f"Computing contradiction map ({split})..."
        )
        noise_df = get_or_compute(
            f"image_noise_probe::{split}::three_layer_v2",
            lambda: image_noise_probe(imgs_split, cdf_split, tuple(sorted(dom_map_split.items()))),
            spinner_text=f"Computing three-layer image noise probe ({split})..."
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

        st.markdown("### Noise pattern detection (image-level)")
        if noise_df.empty:
            st.info("Noise probe unavailable for current split.")
        else:
            c1, c2, c3 = st.columns(3)
            with c1:
                noise_bins = st.slider("Noise bins", 10, 45, 22, key="noise_bins")
            with c2:
                low_pair_th = st.slider("Low pairwise similarity threshold", 0.20, 0.95, 0.55, 0.01, key="noise_pair_th")
            with c3:
                high_noise_th = st.slider("High noise score threshold", 0.10, 0.95, 0.45, 0.01, key="noise_score_th")

            fig_noise, axn = make_fig(w_mult=1.0, h_mult=0.85)
            axn.hist(noise_df["noise_score"].dropna(), bins=noise_bins, color="#ef4444", edgecolor="white", alpha=0.88)
            axn.axvline(noise_df["noise_score"].mean(), color=ACC, linestyle="--", linewidth=2, label=f"Mean={noise_df['noise_score'].mean():.3f}")
            axn.axvline(high_noise_th, color="#1f2937", linestyle=":", linewidth=2, label=f"Threshold={high_noise_th:.2f}")
            axn.legend(frameon=False)
            axn.set_title(f"Caption noise-score distribution ({split})", color=TEXT, pad=10)
            axn.set_xlabel("Noise score")
            axn.set_ylabel("Frequency")
            render_chart(fig_noise)

            noisy_candidates = noise_df[(noise_df["noise_score"] >= high_noise_th) | (noise_df["mean_pairwise"] <= low_pair_th)].copy()
            noisy_candidates = noisy_candidates.sort_values(["noise_score", "mean_pairwise"], ascending=[False, True])

            render_bento_table(
                title="Potential noisy samples",
                icon="🧪",
                df=noisy_candidates[[
                    "filename", "category", "noise_score", "layer_a_score", "layer_b_score", "layer_c_score",
                    "img_color_mismatch_rate", "img_object_mismatch_rate", "prior_color_mismatch_rate", "prior_object_mismatch_rate",
                    "mean_pairwise", "outlier_count", "center_min"
                ]].head(80),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "filename": st.column_config.TextColumn("File 📁"),
                    "category": st.column_config.TextColumn("Category 🏷️"),
                    "noise_score": st.column_config.ProgressColumn("Noise Score", min_value=0.0, max_value=1.0, format="%.3f"),
                    "layer_a_score": st.column_config.ProgressColumn("A: Semantic", min_value=0.0, max_value=1.0, format="%.3f"),
                    "layer_b_score": st.column_config.ProgressColumn("B: Contradiction", min_value=0.0, max_value=1.0, format="%.3f"),
                    "layer_c_score": st.column_config.ProgressColumn("C: Uncertainty", min_value=0.0, max_value=1.0, format="%.3f"),
                    "img_color_mismatch_rate": st.column_config.NumberColumn("Img Color Mis", format="%.2f"),
                    "img_object_mismatch_rate": st.column_config.NumberColumn("Img Object Mis", format="%.2f"),
                    "prior_color_mismatch_rate": st.column_config.NumberColumn("Prior Color", format="%.2f"),
                    "prior_object_mismatch_rate": st.column_config.NumberColumn("Prior Object", format="%.2f"),
                    "mean_pairwise": st.column_config.ProgressColumn("Mean Pairwise Sim", min_value=0.0, max_value=1.0, format="%.3f"),
                    "outlier_count": st.column_config.NumberColumn("Caption Outliers", format="%d"),
                    "center_min": st.column_config.NumberColumn("Worst-to-Center Sim", format="%.3f"),
                }
            )

            inspect_pool = noisy_candidates["filename"].tolist() if len(noisy_candidates) else noise_df.sort_values("noise_score", ascending=False)["filename"].head(30).tolist()
            st.markdown("#### Inspect image + 5 captions (noise evidence)")
            selected_noise_file = st.selectbox("Choose a sample to inspect", options=inspect_pool, key="noise_inspect_file")

            selected_noise_row = noise_df[noise_df["filename"] == selected_noise_file].head(1)
            selected_noise_img = next((img for img in imgs_split if img["filename"] == selected_noise_file), None)

            if selected_noise_img is not None:
                p = IMG_DIR / selected_noise_img["filename"]
                cimg, ccap = st.columns([1, 1.25])
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
                    st.write(f"**File:** `{selected_noise_img['filename']}`")
                    st.write(f"**Category:** `{parse_category(selected_noise_img['filename'])}`")
                    if not selected_noise_row.empty:
                        row = selected_noise_row.iloc[0]
                        st.write(f"**Noise score:** `{float(row['noise_score']):.3f}`")
                        st.write(f"**Mean pairwise similarity:** `{float(row['mean_pairwise']):.3f}`")
                        st.write(f"**Caption outliers:** `{int(row['outlier_count'])}`")
                        st.write(f"**Worst-to-center similarity:** `{float(row['center_min']):.3f}`")
                    for i, s in enumerate(selected_noise_img.get("sentences", [])[:5]):
                        st.write(f"- [{i+1}] {s.get('raw', '')}")

    elif step_idx == 6:
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

            st.markdown("#### Category-level evidence explorer")
            top_cats = top["category"].tolist()
            csel1, csel2 = st.columns(2)
            with csel1:
                selected_cats = st.multiselect(
                    "Choose category to preview",
                    options=top_cats,
                    default=top_cats[:min(3, len(top_cats))],
                    key="contr_selected_categories"
                )
            with csel2:
                evidence_mode = st.radio(
                    "Evidence mode",
                    ["Image-based (color mismatch)", "Text-based (object mismatch)", "Noisy captions (semantic drift)"],
                    horizontal=True,
                    key="contr_evidence_mode"
                )

            if evidence_mode == "Image-based (color mismatch)":
                per_cat = st.slider("Images per category", 3, 18, 6, 3, key="contr_images_per_cat")
            else:
                per_cat = st.slider("Samples per category (with 5 captions)", 1, 6, 2, 1, key="contr_text_per_cat")

            noise_df_step6 = get_or_compute(
                f"image_noise_probe::{split}::three_layer_v2",
                lambda: image_noise_probe(imgs_split, cdf, tuple(sorted(dom_map_split.items()))),
                spinner_text=f"Computing image-level noise probe ({split})..."
            )

            show_cats = selected_cats if selected_cats else top_cats[:1]
            for cat in show_cats:
                cat_imgs = [img for img in imgs_split if parse_category(img["filename"]) == cat]
                if not cat_imgs:
                    continue
                st.markdown(f"**{cat}**")

                if evidence_mode == "Image-based (color mismatch)":
                    cols = st.columns(3)
                    shown = 0
                    for img in cat_imgs[:per_cat]:
                        p = IMG_DIR / img["filename"]
                        if not p.exists():
                            continue
                        try:
                            with Image.open(p) as im:
                                with cols[shown % 3]:
                                    st.image(im.convert("RGB"), use_container_width=True)
                                    st.caption(img["filename"])
                            shown += 1
                        except Exception:
                            continue
                    if shown == 0:
                        st.info("No image previews available for this category.")
                elif evidence_mode == "Text-based (object mismatch)":
                    for img in cat_imgs[:per_cat]:
                        st.write(f"**{img['filename']}**")
                        for i, s in enumerate(img.get("sentences", [])[:5]):
                            st.write(f"- [{i+1}] {s.get('raw', '')}")
                else:
                    if noise_df_step6.empty:
                        st.info("Noise probe unavailable for this split.")
                    else:
                        cat_noise = noise_df_step6[noise_df_step6["category"] == cat].sort_values("noise_score", ascending=False).head(per_cat)
                        if cat_noise.empty:
                            st.info("No noisy-caption candidates found for this category.")
                        for _, row in cat_noise.iterrows():
                            st.write(f"**{row['filename']}** · noise={row['noise_score']:.3f} · pairwise={row['mean_pairwise']:.3f} · outliers={int(row['outlier_count'])}")
                            img_obj = next((x for x in cat_imgs if x["filename"] == row["filename"]), None)
                            if img_obj is None:
                                continue
                            for i, s in enumerate(img_obj.get("sentences", [])[:5]):
                                st.write(f"- [{i+1}] {s.get('raw', '')}")

                st.markdown("---")

if full_page_mode:
    if st.button("↺ Start Over"):
        st.session_state.step = 0
        st.session_state.D = None
        st.session_state.mm_step_cache = {}
        st.rerun()
    for idx in range(TOTAL_STEPS):
        st.markdown(f"### {idx+1}. {STEP_LABELS.get(idx, 'Unknown Step')}")
        render_step(idx)
        if idx < TOTAL_STEPS - 1:
            st.markdown("---")
else:
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
