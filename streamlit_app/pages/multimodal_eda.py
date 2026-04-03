"""
RSITMD Multimodal EDA — Step-by-Step Interactive Demo
Real dataset · Real satellite images · Real statistics
"""
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from PIL import Image
from collections import Counter, defaultdict
from pathlib import Path
import json, re, time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="Multimodal EDA",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ══════════════════════════════════════════════════════════════════════════════
#  PALETTE & CSS
# ══════════════════════════════════════════════════════════════════════════════
BG   = "#F7F3EB"
CARD = "#EFE8DC"
TEXT = "#111111"
ACC  = "#B42318"
MUT  = "#6B6560"
BOR  = "#D4C9B8"
RULE = "#C8BBB0"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,700;0,900&family=Source+Sans+3:wght@300;400;600&display=swap');
:root {{ --bg:{BG}; --card:{CARD}; --text:{TEXT}; --acc:{ACC}; --mut:{MUT}; --bor:{BOR}; --rule:{RULE}; }}
body,.stApp {{ background:var(--bg); color:var(--text); font-family:'Source Sans 3',sans-serif; }}
[data-testid="stSidebar"] {{ background:#EDE9E1 !important; border-right:2px solid var(--rule); width:220px !important; }}
#MainMenu,footer,header {{ visibility:hidden; }}
.main .block-container {{ padding-left:240px; padding-right:1rem; }}
h1,h2,h3 {{ font-family:'Playfair Display',serif; color:var(--text); line-height:1.1; }}
.eyebrow {{ font-size:0.65rem; font-weight:700; letter-spacing:0.2em;
            text-transform:uppercase; color:var(--acc); margin-bottom:0.4rem; }}
.section-head {{ font-size:0.62rem; font-weight:700; letter-spacing:0.15em;
                 text-transform:uppercase; color:var(--acc); margin:1.4rem 0 0.3rem; }}
.footer {{ text-align:center; color:var(--mut); font-size:0.68rem;
           letter-spacing:0.1em; text-transform:uppercase; }}
.hero-divider {{ width:60px; height:2px; background:var(--acc); margin:1.6rem auto; display:block; }}
[data-testid="stMetricValue"] {{ font-family:'Playfair Display',serif;
    font-size:2.2rem; font-weight:900; color:var(--text); line-height:1; }}
[data-testid="stMetricLabel"] {{ font-size:0.62rem; font-weight:700; letter-spacing:0.1em;
    text-transform:uppercase; color:var(--mut); }}
thead th {{ background:{CARD} !important; color:{ACC} !important;
            font-weight:700; letter-spacing:0.06em; font-size:0.7rem;
            border-bottom:2px solid {BOR} !important; text-transform:uppercase; }}
tbody tr:hover {{ background:rgba(180,35,24,0.04) !important; }}
tbody td {{ border-bottom:1px solid {BOR} !important; }}
.stTabs [data-baseweb="tab-list"] {{ gap:2px; border-bottom:2px solid var(--bor); padding-bottom:0; }}
.stTabs [data-baseweb="tab"] {{ color:var(--mut); font-weight:700; font-size:0.85rem; border-radius:0; }}
.stTabs [aria-selected="true"] {{ color:var(--acc)!important; border-bottom:2.5px solid var(--acc)!important; margin-bottom:-2px; }}
.insight {{ border-left:3px solid var(--acc); padding:0.7rem 1.2rem;
            background:var(--card); margin:1rem 0;
            font-family:'Playfair Display',serif;
            font-size:1rem; font-style:italic; color:var(--text); line-height:1.6; }}
.log-ok {{ color:#1A7A3E; }}
.log-warn {{ color:#B45315; }}
.log-entry {{ font-size:0.78rem; color:var(--mut); padding:0.18rem 0;
             border-bottom:1px solid var(--rule); font-family:'Courier New',monospace; }}
.step-pill {{ display:inline-block; background:{CARD}; border:1px solid {BOR};
    border-radius:20px; padding:0.3rem 1rem; font-size:0.72rem;
    font-weight:700; letter-spacing:0.1em; color:{ACC};
    text-transform:uppercase; margin-bottom:0.8rem; }}
button[kind="secondary"], button[kind="primary"],
[data-testid="stBaseButton-secondary"],
[data-testid="stBaseButton-primary"] {{
    border:1.5px solid {TEXT} !important; border-radius:2px !important;
    background:transparent !important; color:{TEXT} !important;
    font-family:'Source Sans 3',sans-serif !important;
    font-size:0.85rem !important; font-weight:700 !important;
    letter-spacing:0.08em !important; transition:all 0.18s !important;
    box-shadow:none !important;
}}
button[kind="primary"]:hover {{
    background:{ACC} !important; border-color:{ACC} !important; color:white !important;
}}
</style>""", unsafe_allow_html=True)

TOTAL_STEPS = 8
STEP_LABELS = {
    0: "Dataset Overview",
    1: "Category Distribution",
    2: "Image Properties",
    3: "Caption Vocabulary",
    4: "Color Words",
    5: "Spatial Relations",
    6: "Caption Variability",
    7: "Noise Detection",
}

# ══════════════════════════════════════════════════════════════════════════════
#  SESSION STATE — must be initialised before sidebar & any step logic runs
# ══════════════════════════════════════════════════════════════════════════════
if "phase" not in st.session_state:
    st.session_state.phase = "idle"
if "step" not in st.session_state:
    st.session_state.step = 0
if "log" not in st.session_state:
    st.session_state.log = []

# D is module-level so it persists across reruns after load_data() is called
D = None
st.session_state.data_loaded = False

# Left panel content lives inside the sidebar for a clean single-column main area
with st.sidebar:
    st.markdown("[🏠 Home](app)")
    st.markdown("[📝 Text](pages/text_eda)")
    st.markdown("[🖼️ Image](pages/image_eda)")
    st.markdown("[📊 Tabular](pages/tabular_eda)")
    st.markdown("[🔗 Multimodal](pages/multimodal_eda)")
    st.markdown("---")
    st.caption("IEEE TGRS 2021 · UIT")
    st.caption("RSITMD Dataset · Z. Yuan et al.")
    st.markdown("---")

    current = st.session_state.step
    phase   = st.session_state.phase

    if phase == "running":
        st.markdown(f"<p class='step-pill'>⏳ Step {current+1}/{TOTAL_STEPS}</p>",
                    unsafe_allow_html=True)
        st.progress((current) / TOTAL_STEPS)
    else:
        st.markdown(f"<p class='step-pill' style='background:#DCFCE7;border-color:#1A7A3E;color:#1A7A3E'>"
                    f"✅  Done</p>", unsafe_allow_html=True)

    for i, label in STEP_LABELS.items():
        if i < current or phase == "done":
            st.markdown(
                f"<p class='log-entry log-ok'>✓  {label}</p>",
                unsafe_allow_html=True,
            )
        elif i == current:
            st.markdown(
                f"<p style='font-size:0.78rem;color:{ACC};padding:0.18rem 0;"
                f"border-bottom:1px solid {RULE};'>▸  {label}</p>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<p style='font-size:0.78rem;color:{MUT};padding:0.18rem 0;"
                f"border-bottom:1px solid {RULE};opacity:0.5;'>○  {label}</p>",
                unsafe_allow_html=True,
            )

    st.markdown("")

    for entry in st.session_state.log:
        cls = "log-ok" if "✓" in entry else ""
        st.markdown(f"<p class='log-entry {cls}'>{entry}</p>",
                    unsafe_allow_html=True)

    st.markdown("")
    if st.button("↺  Start Over"):
        for k in list(st.session_state.keys()):
            if k not in ["phase"]:
                del st.session_state[k]
        st.session_state.phase = "idle"
        st.rerun()

    if phase == "done":
        st.markdown("---")
        st.markdown("[🔗  Open Full Dashboard](pages/multimodal_eda)")

# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def resolve_data_paths():
    local_data = Path("/Users/nhi/Documents/school/252/p4/btl/RSITMD/dataset_RSITMD.json")
    local_img = Path("/Users/nhi/Documents/school/252/p4/btl/RSITMD/images")
    if local_data.exists() and local_img.exists():
        return local_data, local_img
    
    # Check if we already downloaded it on the cloud
    possible_json = list(Path('.').rglob('dataset_RSITMD.json'))
    if possible_json:
        return possible_json[0], possible_json[0].parent / "images"

    # Not found -> Download from Google Drive
    st.warning("📡 Dữ liệu đang được tải từ Google Drive xuống Streamlit Cloud (~916MB)...")
    progress_bar = st.progress(0)
    with st.spinner("Đang kết nối G-Drive... Việc này cần khoảng 1-2 phút..."):
        import gdown
        zip_path = "RSITMD.zip"
        
        # We don't have a reliable progress callback from gdown in streamlit unless we do a custom wrap, 
        # but standard gdown will block until complete.
        progress_bar.progress(20, text="Downloading (Khoảng 900MB)...")
        file_id = "1NJY86TAAUd8BVs7hyteImv8I2_Lh95W6"
        gdown.download(id=file_id, output=zip_path, quiet=True)
        
        progress_bar.progress(85, text="Đang giải nén dữ liệu...")
        import zipfile
        import os
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(".")
        os.remove(zip_path) # Cleanup
        progress_bar.progress(100, text="Hoàn tất!")
        time.sleep(1)
        progress_bar.empty()

    possible_json = list(Path('.').rglob('dataset_RSITMD.json'))
    if possible_json:
        return possible_json[0], possible_json[0].parent / "images"
    else:
        st.error("Lỗi: Không tìm thấy file JSON bên trong file Zip vừa tải!")
        st.stop()

DATA_FILE, IMG_DIR = resolve_data_paths()

STOP_WORDS = {
    'i','me','my','myself','we','our','ours','ourselves','you','your','yours',
    'yourself','yourselves','he','him','his','himself','she','her','hers',
    'herself','it','its','itself','they','them','their','theirs','themselves',
    'what','which','who','whom','this','that','these','those','am','is','are',
    'was','were','be','been','being','have','has','had','having','do','does',
    'did','doing','a','an','the','and','but','if','or','because','as','until',
    'while','of','at','by','for','with','about','against','between','into',
    'through','during','before','after','above','below','to','from','up','down',
    'in','out','on','off','over','under','again','further','then','once','here',
    'there','when','where','why','how','all','each','few','more','most','other',
    'some','such','no','nor','not','only','own','same','so','than','too','very',
    'can','will','just','should','now','one','also','the'
}

COLOR_WORDS = {
    'green','white','red','blue','gray','grey','yellow','dark','black',
    'brown','light','orange','bright','purple','pink','silver','golden'
}

COLOR_HEX = {
    'green': '#4CAF50', 'white': '#E0E0E0', 'red': '#F44336',
    'blue': '#2196F3', 'gray': '#9E9E9E', 'grey': '#9E9E9E',
    'yellow': '#FFEB3B', 'dark': '#212121', 'black': '#111111',
    'brown': '#795548', 'light': '#F5F5F5', 'orange': '#FF9800',
    'bright': '#FFEB3B', 'purple': '#9C27B0', 'pink': '#E91E63',
    'silver': '#BDBDBD', 'golden': '#FFD700'
}

SPATIAL_KEYWORDS = [
    'near', 'surrounded', 'next to', 'around', 'beside',
    'in the middle of', 'in the center of', 'above', 'below',
    'on the edge', 'across', 'along', 'between'
]

CLUSTERS = [
    ("Commercial",        ["center","church","commercial","school","stadium","playground"]),
    ("Natural",           ["bareland","beach","desert","farmland","forest","meadow","mountain","pond","river"]),
    ("Urban Features",    ["baseballfield","boat","industrial","intersection","park","plane","resort","square","storagetanks"]),
]
CLUSTER_COLORS = {
    "Commercial":    "#D84315",
    "Natural":       "#5D4037",
    "Urban Features":"#6A1B9A",
}

SENTENCE_COLORS = [
    ('#FFF8E1', '#E65100'),
    ('#E8F5E9', '#2E7D32'),
    ('#EAF2FF', '#1565C0'),
    ('#F3E5F5', '#6A1B9A'),
    ('#FCE4EC', '#880E4F'),
]

def tokenize(text, remove_stopwords=False):
    words = re.findall(r'\b[a-z]+\b', text.lower())
    if remove_stopwords:
        words = [w for w in words if w not in STOP_WORDS]
    return words

def parse_category(filename):
    parts = filename.replace('.tif', '').rsplit('_', 1)
    return parts[0] if len(parts) == 2 else 'unknown'

# ══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data
def load_data():
    with open(DATA_FILE, encoding="utf-8") as f:
        data = json.load(f)
    
    imgs = data["images"]
    train_imgs = [img for img in imgs if img["split"] == "train"]
    test_imgs  = [img for img in imgs if img["split"] == "test"]

    train_caps = [s["raw"] for img in train_imgs for s in img["sentences"]]
    test_caps  = [s["raw"] for img in test_imgs  for s in img["sentences"]]

    train_clean = [w for cap in train_caps for w in tokenize(cap, True)]
    test_clean  = [w for cap in test_caps  for w in tokenize(cap, True)]

    word_freq   = Counter(train_clean)
    bigrams     = [tuple(train_clean[i:i+2]) for i in range(len(train_clean)-1)]
    bigram_freq = Counter(bigrams)

    train_cats = Counter(parse_category(img["filename"]) for img in train_imgs)

    color_freq = Counter()
    caps_with_color = 0
    for cap in train_caps:
        words = tokenize(cap)
        hit = False
        for w in words:
            if w in COLOR_WORDS:
                color_freq[w] += 1
                hit = True
        if hit:
            caps_with_color += 1

    spatial_counts = Counter()
    caps_with_spatial = 0
    for cap in train_caps:
        cl = cap.lower()
        hit = False
        for kw in SPATIAL_KEYWORDS:
            if kw in cl:
                spatial_counts[kw] += 1
                hit = True
        if hit:
            caps_with_spatial += 1

    cap_lens = [len(tokenize(cap)) for cap in train_caps]

    cat_to_imgs = defaultdict(list)
    for img in train_imgs:
        cat_to_imgs[parse_category(img["filename"])].append(img)

    variabilities = []
    for img in train_imgs:
        lens = [len(s["tokens"]) for s in img["sentences"]]
        std  = np.std(lens) if len(lens) > 1 else 0
        variabilities.append(std)

    per_category = defaultdict(list)
    for img in train_imgs:
        cat = parse_category(img["filename"])
        lens = [len(s["tokens"]) for s in img["sentences"]]
        per_category[cat].append(np.std(lens))

    cat_var = {c: round(np.mean(stds), 2) for c, stds in per_category.items()}

    sample_images = train_imgs[:3]

    vocab_overlap = len(set(train_clean) & set(test_clean)) / max(len(set(test_clean)), 1) * 100

    # --- TF-IDF Centroids for Noise Detection ---
    cat_to_captions = defaultdict(list)
    for img in train_imgs:
        cat = parse_category(img["filename"])
        for s in img["sentences"]:
            cat_to_captions[cat].append(s["raw"])
            
    category_centroids = {}
    category_vectorizers = {}
    for cat, caps in cat_to_captions.items():
        if len(caps) > 2:
            vec = TfidfVectorizer(stop_words='english', min_df=1)
            tfidf_matrix = vec.fit_transform(caps)
            category_centroids[cat] = np.asarray(tfidf_matrix.mean(axis=0))
            category_vectorizers[cat] = vec

    # --- Fast Image Pixel Sampling (Dominant Channel) for Layer B ---
    fast_dom_channels = {}
    import random
    random.seed(42)
    for cat, imgs_list in cat_to_imgs.items():
        sample_imgs = random.sample(imgs_list, min(5, len(imgs_list)))
        r_sum, g_sum, b_sum, pcnt = 0.0, 0.0, 0.0, 0
        for simg in sample_imgs:
            try:
                pil = Image.open(IMG_DIR / simg["filename"])
                if pil.mode in ("RGBA", "LA", "P"):
                    pil = pil.convert("RGB")
                arr = np.array(pil, dtype=np.float32) / 255.0
                r_sum += float(np.sum(arr[:,:,0]))
                g_sum += float(np.sum(arr[:,:,1]))
                b_sum += float(np.sum(arr[:,:,2]))
                pcnt += arr.shape[0] * arr.shape[1]
            except Exception:
                pass
        if pcnt > 0:
            r_avg, g_avg, b_avg = r_sum/pcnt, g_sum/pcnt, b_sum/pcnt
            if g_avg > r_avg and g_avg > b_avg:
                fast_dom_channels[cat] = 'G>R>B'
            elif b_avg > r_avg and b_avg > g_avg:
                fast_dom_channels[cat] = 'B>R>G'
            elif r_avg > b_avg:
                fast_dom_channels[cat] = 'R>G>B'
            else:
                fast_dom_channels[cat] = 'R≈G>B'
        else:
            fast_dom_channels[cat] = 'R≈G>B'

    return {
        "imgs": imgs,
        "train_imgs": train_imgs,
        "test_imgs": test_imgs,
        "train_caps": train_caps,
        "test_caps": test_caps,
        "train_clean": train_clean,
        "word_freq": word_freq,
        "bigram_freq": bigram_freq,
        "train_cats": train_cats,
        "color_freq": color_freq,
        "caps_with_color": caps_with_color,
        "spatial_counts": spatial_counts,
        "caps_with_spatial": caps_with_spatial,
        "cap_lens": cap_lens,
        "cat_to_imgs": dict(cat_to_imgs),
        "variabilities": variabilities,
        "cat_var": cat_var,
        "sample_images": sample_images,
        "vocab_overlap": vocab_overlap,
        "total_caps": len(train_caps),
        "n_train": len(train_imgs),
        "n_test": len(test_imgs),
        "n_cats": len(train_cats),
        "vocab_clean": len(set(train_clean)),
        "avg_cap_len": round(np.mean(cap_lens), 2),
        "std_cap_len": round(np.std(cap_lens), 2),
        "mean_var": round(np.mean(variabilities), 2),
        "max_var": round(np.max(variabilities), 2),
        "category_centroids": category_centroids,
        "category_vectorizers": category_vectorizers,
        "fast_dom_channels": fast_dom_channels,
    }

# ══════════════════════════════════════════════════════════════════════════════
#  IDLE LANDING PAGE  (shown before Start Demo — no data loaded)
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.phase == "idle":
    st.markdown(f"<p class='eyebrow'>§ Multimodal Analysis &nbsp;·&nbsp; RSITMD</p>",
                unsafe_allow_html=True)
    st.markdown(f"<h1 style='font-size:clamp(1.8rem,3vw,2.8rem);font-weight:900;margin:0 0 0.5rem'>"
                f"EDA Multimodal - Breakthrough Group</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-size:0.95rem;font-weight:300;color:{MUT};margin:0 0 1rem'>"
                f"Remote Sensing Image-Text Matching Dataset &nbsp;·&nbsp; Z. Yuan et al., IEEE TGRS 2021</p>",
                unsafe_allow_html=True)
    st.markdown(f"<span class='hero-divider'></span>", unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 1])
    with col_left:
        st.markdown("""
        <div class='insight' style='font-size:0.9rem;'>
          This interactive demo walks through the RSITMD dataset step by step.
          Each step shows real analysis computed from the actual dataset —
          no simulated data, no hardcoded numbers.
        </div>""", unsafe_allow_html=True)
    with col_right:
        steps_list = [
            ("1", "Dataset Overview"),
            ("2", "Category Distribution"),
            ("3", "Image Properties"),
            ("4", "Caption Vocabulary"),
            ("5", "Color Words"),
            ("6", "Spatial Relations"),
            ("7", "Caption Variability"),
            ("8", "Noise Detection"),
        ]
        for num, label in steps_list:
            st.markdown(
                f"<div style='display:flex;align-items:center;gap:0.8rem;margin:0.3rem 0;'>"
                f"<span style='background:{ACC};color:white;width:1.4rem;height:1.4rem;"
                f"border-radius:50%;display:inline-flex;align-items:center;justify-content:center;"
                f"font-size:0.7rem;font-weight:700;'>{num}</span>"
                f"<span style='color:var(--text);font-size:0.88rem;'>{label}</span></div>",
                unsafe_allow_html=True,
            )

    st.markdown("")
    if st.button("▶  Start Demo"):
        st.session_state.phase = "transitioning"
        st.session_state.step  = 0
        st.session_state.log   = []
        st.rerun()

    st.markdown(f"<span class='hero-divider'></span>", unsafe_allow_html=True)
    st.markdown(f"<p class='footer'>IEEE TGRS 2021 · RSITMD · UIT · 2025–2026</p>",
                unsafe_allow_html=True)
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
#  TRANSITIONING — load data here, only when user presses Start Demo
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.phase == "transitioning":
    with st.spinner(""):
        st.markdown(
            f"<p style='text-align:center;font-size:1.05rem;font-weight:600;"
            f"color:{TEXT};margin-top:3rem;'>"
            f"Analyzing RSITMD dataset…</p>",
            unsafe_allow_html=True,
        )
        st.progress(1.0)
        time.sleep(1.5)
    st.session_state.D = load_data()
    st.session_state.phase = "running"
    st.session_state.data_loaded = True
    st.rerun()

# Restore D from session_state on every run (module-level D is reset after rerun)
D = st.session_state.get("D")

# ══════════════════════════════════════════════════════════════════════════════
#  RUNNING — data is loaded; render header + step routing
# ══════════════════════════════════════════════════════════════════════════════
if D is not None:
    st.markdown(f"<p class='eyebrow'>§ Multimodal Analysis &nbsp;·&nbsp; RSITMD</p>",
                unsafe_allow_html=True)
    st.markdown(f"<h1 style='font-size:clamp(1.8rem,3vw,2.8rem);font-weight:900;margin:0 0 0.5rem'>"
                f"EDA Multimodal</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-size:0.95rem;font-weight:300;color:{MUT};margin:0 0 1rem'>"
                f"Remote Sensing Image-Text Matching Dataset &nbsp;·&nbsp; Z. Yuan et al., IEEE TGRS 2021</p>",
                unsafe_allow_html=True)

    # ── STEP 0: Overview ─────────────────────────────────────────────────────
    if st.session_state.step == 0:
        st.markdown("<p class='section-head'>Step 1 of 7 — Dataset Overview</p>",
                    unsafe_allow_html=True)

        top_word   = D["word_freq"].most_common(1)[0]
        top_bigram = D["bigram_freq"].most_common(1)[0]
        top_color  = D["color_freq"].most_common(1)[0]
        top_cat    = D["train_cats"].most_common(1)[0]
        bot_cat    = D["train_cats"].most_common()[::-1][0]
        imbalance  = top_cat[1] / bot_cat[1]
        color_pct  = D["caps_with_color"] / D["total_caps"] * 100

        st.markdown("""
        **RSITMD** (Remote Sensing Image-Text Matching Dataset) was published by
        Yuan et al. in IEEE TGRS 2021. Each of the 4,743 satellite images has
        exactly 5 human-written English captions describing the same aerial scene
        from different perspectives.
        """)

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Images", f"{D['n_train'] + D['n_test']:,}")
        k2.metric("Train / Test", f"{D['n_train']:,} / {D['n_test']:,}")
        k3.metric("Categories", str(D['n_cats']))
        k4.metric("Unique Words", f"{D['vocab_clean']:,}")

        test_caps  = D["test_caps"]
        test_clean = [w for cap in test_caps for w in tokenize(cap, True)]
        raw_words  = [w for cap in D["train_caps"] for w in tokenize(cap)]
        ov = pd.DataFrame({
            "Metric": [
                "Train Images", "Test Images",
                "Train Captions", "Test Captions",
                "Captions per Image",
                "Avg Words / Caption",
                "Std Dev Words",
                "Vocab (raw)", "Vocab (no stopwords)",
                "Vocab Overlap (train→test)",
            ],
            "Value": [
                f"{D['n_train']:,}", f"{D['n_test']:,}",
                f"{D['total_caps']:,}", f"{len(test_caps):,}",
                "5",
                f"{D['avg_cap_len']:.1f}",
                f"{D['std_cap_len']:.1f}",
                f"{len(set(raw_words)):,}",
                f"{D['vocab_clean']:,}",
                f"{D['vocab_overlap']:.1f}%",
            ],
        })
        st.dataframe(ov)

        sample = D["sample_images"][0]
        img_path = IMG_DIR / sample["filename"]
        cat = parse_category(sample["filename"])
        col_img, col_cap = st.columns([1, 2])
        with col_img:
            try:
                img_pil = Image.open(img_path)
                if img_pil.mode in ("RGBA", "LA", "P"):
                    img_pil = img_pil.convert("RGB")
                st.image(img_pil, width=280)
            except Exception as e:
                st.warning(f"Could not load sample image.")
            st.caption(f"{cat}")
            st.caption(f"`{sample['filename']}`")

        with col_cap:
            st.markdown("##### Sample Captions")
            for i, s in enumerate(sample["sentences"][:5]):
                bg, fg = SENTENCE_COLORS[i]
                st.markdown(
                    f"<div style='background:{bg};border-left:3px solid {fg};"
                    f"padding:0.4rem 0.7rem;margin:0.25rem 0;border-radius:0 3px 3px 0;'>"
                    f"<span style='color:{fg};font-size:0.72rem;font-weight:700;'>"
                    f"#{i+1}</span>&nbsp;"
                    f"<span style='color:{TEXT};font-size:0.85rem;'>{s['raw'][:120]}"
                    f"{'…' if len(s['raw']) > 120 else ''}</span></div>",
                    unsafe_allow_html=True,
                )

        st.markdown("")
        if st.button("Next: Category Distribution →"):
            st.session_state.log.append(f"[{time.strftime('%H:%M:%S')}] ✓ Dataset: "
                                        f"{D['n_train']:,} train / {D['n_test']:,} test")
            st.session_state.step = 1
            st.rerun()
    # ── STEP 1: Category Distribution ─────────────────────────────────────────
    elif st.session_state.step == 1:
        st.markdown("<p class='section-head'>Step 2 of 7 — Category Distribution</p>",
                    unsafe_allow_html=True)
        st.markdown("""
        <div class='insight' style='font-style:normal;border-left-color:#1565C0;'>
          How are images distributed across the 33 land-use categories?
          Are all classes equally represented, or is there imbalance?
        </div>""", unsafe_allow_html=True)

        cat_counts = dict(D["train_cats"])
        sorted_cats = D["train_cats"].most_common()

        # ── Controls ──────────────────────────────────────────────────────────
        ctrl_col = st.columns(1)[0]
        with ctrl_col:
            view_mode = st.radio(
                "Show:", ["All Categories", "Top 10", "Top 20"],
                index=0, horizontal=True,
            )

        # ── Build label/value/color lists based on mode ─────────────────────
        all_labels, all_vals, all_colors = [], [], []

        if view_mode == "All Categories":
            for cat, cnt in sorted_cats:
                for cn, cats in CLUSTERS:
                    if cat in cats:
                        all_colors.append(CLUSTER_COLORS[cn])
                        break
                all_labels.append(cat)
                all_vals.append(cnt)
        elif view_mode == "Top 10":
            for cat, cnt in sorted_cats[:10]:
                for cn, cats in CLUSTERS:
                    if cat in cats:
                        all_colors.append(CLUSTER_COLORS[cn])
                        break
                all_labels.append(cat)
                all_vals.append(cnt)
        else:  # Top 20
            for cat, cnt in sorted_cats[:20]:
                for cn, cats in CLUSTERS:
                    if cat in cats:
                        all_colors.append(CLUSTER_COLORS[cn])
                        break
                all_labels.append(cat)
                all_vals.append(cnt)

        n = len(all_labels)

        # Căn chỉnh lại đoạn vẽ biểu đồ ở đây:
        fig_h = max(5, n * 0.24)
        fig, ax = plt.subplots(figsize=(13, fig_h))
        fig.patch.set_facecolor(BG)
        ax.set_facecolor(BG)
        
        bars = ax.barh(range(n), all_vals,
                       color=all_colors, edgecolor="none", height=0.68)
        
        ax.set_yticks(range(n))
        ax.set_yticklabels(all_labels, fontsize=9.5)
        ax.set_xlabel("Number of Images", fontsize=10)
        
        if view_mode == "All Categories":
            ax.set_title("Category Distribution — Training Set (All Categories)",
                         fontsize=13, fontweight="bold", fontfamily="serif")
        else:
            ax.set_title(f"Category Distribution — {view_mode} (Training Set)",
                         fontsize=13, fontweight="bold", fontfamily="serif")
            ax.invert_yaxis()
            
        ax.tick_params(colors=MUT, labelsize=9.5)
        for s in ax.spines.values():
            s.set_visible(False)
            
        ax.xaxis.set_visible(False)
        ax.grid(axis="x", linewidth=0.5, color=BOR, zorder=0)
        
        xlim = max(all_vals) * 1.2
        for bar, val in zip(bars, all_vals):
            ax.text(xlim, bar.get_y() + bar.get_height()/2,
                    f" {val:,}", va="center", ha="left", fontsize=9, color=TEXT, fontweight="bold")
        ax.set_xlim(0, xlim)

        from matplotlib.patches import Patch
        legend_patches = [Patch(facecolor=CLUSTER_COLORS[c], edgecolor="none", label=c)
                          for c in CLUSTER_COLORS]
        ax.legend(handles=legend_patches, loc="lower right", frameon=False, fontsize=9)

        plt.tight_layout()
        st.pyplot(fig)

        # ── Image Gallery Tabs ─────────────────────────────────────────────────
        st.markdown("")
        gal_tabs = st.tabs([
            "🗂️  Browse by Category",
            "⚖️  Largest vs Smallest",
            "🏞️  Category Snapshots",
        ])

        # helper: render N images in a row — silently skip corrupt files
        def render_img_grid(imgs_list, n_cols=3):
            loadable = []
            for img in imgs_list:
                try:
                    pil = Image.open(IMG_DIR / img["filename"])
                    if pil.mode in ("RGBA", "LA", "P"):
                        pil = pil.convert("RGB")
                    loadable.append((pil, parse_category(img["filename"])))
                except Exception:
                    pass  # skip corrupt files silently
            n = len(loadable)
            if n == 0:
                st.info("No images could be loaded for this selection.")
                return
            rows = (n + n_cols - 1) // n_cols
            for r in range(rows):
                cols = st.columns(n_cols)
                for ci in range(n_cols):
                    idx = r * n_cols + ci
                    if idx >= n:
                        break
                    pil, cat = loadable[idx]
                    with cols[ci]:
                        st.image(pil, use_column_width=True)
                        st.caption(f"{cat}")

        # tab 1: browse by category
        with gal_tabs[0]:
            sorted_cats = D["train_cats"].most_common()
            cat_opts = [c for c, _ in sorted_cats]
            chosen = st.selectbox("Choose a category", cat_opts)
            cat_imgs = D["cat_to_imgs"].get(chosen, [])
            import random
            random.seed(42)
            pick = random.sample(cat_imgs, min(6, len(cat_imgs)))
            render_img_grid(pick, n_cols=3)

        # tab 2: largest vs smallest
        with gal_tabs[1]:
            top_img_cat  = sorted_cats[0][0]
            bot_img_cat  = sorted_cats[-1][0]
            top_imgs = D["cat_to_imgs"].get(top_img_cat, [])[:3]
            bot_imgs = D["cat_to_imgs"].get(bot_img_cat, [])[:3]

            sub1, sub2 = st.columns(2)
            with sub1:
                st.markdown(f"**{top_img_cat}** ({sorted_cats[0][1]:,} imgs)")
                for img in top_imgs:
                    try:
                        pil = Image.open(IMG_DIR / img["filename"])
                        if pil.mode in ("RGBA", "LA", "P"):
                            pil = pil.convert("RGB")
                        st.image(pil, use_column_width=True)
                    except Exception:
                        pass
                st.caption(f"{top_img_cat}")
            with sub2:
                st.markdown(f"**{bot_img_cat}** ({sorted_cats[-1][1]} imgs)")
                for img in bot_imgs:
                    try:
                        pil = Image.open(IMG_DIR / img["filename"])
                        if pil.mode in ("RGBA", "LA", "P"):
                            pil = pil.convert("RGB")
                        st.image(pil, use_column_width=True)
                    except Exception:
                        pass
                st.caption(f"{bot_img_cat}")

        # tab 3: snapshots from 4 diverse categories
        with gal_tabs[2]:
            diverse_cats = [sorted_cats[0][0], sorted_cats[5][0],
                            sorted_cats[15][0], sorted_cats[-5][0]]
            col_a, col_b = st.columns(2)
            pairs = list(zip(diverse_cats[:2], [col_a, col_b]))
            for (cat, col) in pairs:
                imgs = D["cat_to_imgs"].get(cat, [])[:3]
                with col:
                    st.markdown(f"**{cat}**")
                    for img in imgs:
                        try:
                            pil = Image.open(IMG_DIR / img["filename"])
                            if pil.mode in ("RGBA", "LA", "P"):
                                pil = pil.convert("RGB")
                            st.image(pil, use_column_width=True)
                        except Exception:
                            pass
                    st.caption(f"{cat}")
            col_c, col_d = st.columns(2)
            for (cat, col) in zip(diverse_cats[2:], [col_c, col_d]):
                imgs = D["cat_to_imgs"].get(cat, [])[:3]
                with col:
                    st.markdown(f"**{cat}**")
                    for img in imgs:
                        try:
                            pil = Image.open(IMG_DIR / img["filename"])
                            if pil.mode in ("RGBA", "LA", "P"):
                                pil = pil.convert("RGB")
                            st.image(pil, use_column_width=True)
                        except Exception:
                            pass
                    st.caption(f"{cat}")

        top_cat   = D["train_cats"].most_common(1)[0]
        bot_cat   = D["train_cats"].most_common()[::-1][0]
        imb_ratio = top_cat[1] / bot_cat[1]
        small_5   = sum(v for c, v in D["train_cats"].most_common()[-5:])

        if view_mode == "All Categories":
            k1, k2, k3 = st.columns(3)
            k1.metric("Categories", str(D["n_cats"]))
            k2.metric("Imbalance Ratio", f"{imb_ratio:.0f}:1")
            k3.metric("5 Rarest Total", f"{small_5} ({small_5/D['n_train']*100:.1f}%)")
        elif view_mode == "Top 10":
            k1, k2 = st.columns(2)
            k1.metric("Showing", "Top 10 Categories")
            k2.metric("% of Training Set", f"{sum(all_vals)/D['n_train']*100:.1f}%")
        else:
            k1, k2 = st.columns(2)
            k1.metric("Showing", "Top 20 Categories")
            k2.metric("% of Training Set", f"{sum(all_vals)/D['n_train']*100:.1f}%")

        st.markdown("")
        if st.button("Next: Image Properties →"):
            st.session_state.log.append(f"[{time.strftime('%H:%M:%S')}] ✓ "
                                        f"Categories: {D['n_cats']} classes, imbalance {imb_ratio:.0f}:1")
            st.session_state.step = 2
            st.rerun()

    # ── STEP 2: Image Properties ──────────────────────────────────────────────
    elif st.session_state.step == 2:
        st.markdown("<p class='section-head'>Step 3 of 7 — Image Properties</p>",
                    unsafe_allow_html=True)
        st.markdown("""
        <div class='insight' style='font-style:normal;border-left-color:#5D4037;'>
          Beyond captions — what do the satellite images themselves look like?
          Brightness, color channels, and texture reveal the visual character of each category.
        </div>""", unsafe_allow_html=True)

        st.info(
            "Computing image statistics from **a sample of 600 images** "
            "(100 per decile of variability) to keep analysis fast. "
            "All statistics are computed on-the-fly from the actual pixel data."
        )

        import random
        random.seed(77)

        # build image sample
        train_imgs = D["train_imgs"]
        n_sample = min(600, len(train_imgs))
        # stratified sample: equal numbers from each category, then random within
        per_cat_n = max(1, n_sample // D["n_cats"])
        sampled = []
        for imgs in D["cat_to_imgs"].values():
            sampled.extend(random.sample(imgs, min(per_cat_n, len(imgs))))
        random.shuffle(sampled)
        sampled = sampled[:n_sample]

        @st.cache_data(ttl=300)
        def compute_img_stats(img_list):
            brightness, r_ratio, g_ratio, b_ratio = [], [], [], []
            entropy_vals = []
            for img in img_list:
                try:
                    pil = Image.open(IMG_DIR / img["filename"])
                    if pil.mode in ("RGBA", "LA", "P"):
                        pil = pil.convert("RGB")
                    arr = np.array(pil).astype(float)
                    gray = 0.299*arr[:,:,0] + 0.587*arr[:,:,1] + 0.114*arr[:,:,2]
                    brightness.append(float(gray.mean()))
                    total = arr.sum(axis=2) + 1e-9
                    r_ratio.append(float((arr[:,:,0] / total).mean()))
                    g_ratio.append(float((arr[:,:,1] / total).mean()))
                    b_ratio.append(float((arr[:,:,2] / total).mean()))
                    edge = np.abs(np.diff(gray, axis=0)).mean() + np.abs(np.diff(gray, axis=1)).mean()
                    entropy_vals.append(float(edge))
                except Exception:
                    pass
            return brightness, r_ratio, g_ratio, b_ratio, entropy_vals

        with st.spinner("Computing image statistics from pixels..."):
            brightness, r_ratio, g_ratio, b_ratio, entropy_vals = compute_img_stats(sampled)

        if not brightness:
            st.warning("Could not load images. Check that the image directory is accessible.")
        else:
            # per-category means for brightness and entropy
            cat_brightness = defaultdict(list)
            cat_entropy    = defaultdict(list)
            for i, img in enumerate(sampled):
                cat = parse_category(img["filename"])
                cat_brightness[cat].append(brightness[i])
                cat_entropy[cat].append(entropy_vals[i])

            avg_brightness = {c: np.mean(v) for c, v in cat_brightness.items()}
            avg_entropy    = {c: np.mean(v) for c, v in cat_entropy.items()}

            tabs = st.tabs([
                "☀️  Brightness by Category",
                "🎨  Color Channels",
                "🔲  Texture / Edge Density",
                "🖼️  Sample Images",
            ])

            with tabs[0]:
                st.markdown("""
                **Mean brightness** (grayscale 0–255) per category.
                Higher = lighter / more open terrain; lower = darker / denser coverage.
                """)
                sort_bright = sorted(avg_brightness.items(), key=lambda x: x[1], reverse=True)
                cats_b, vals_b = zip(*sort_bright)
                cmap_b = plt.cm.YlOrRd(np.linspace(0.35, 0.9, len(cats_b)))
                fig, ax = plt.subplots(figsize=(11, max(5, len(cats_b)*0.28)))
                fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
                ax.barh(list(cats_b)[::-1], list(vals_b)[::-1],
                        color=cmap_b.tolist()[::-1], edgecolor="none")
                ax.set_xlabel("Mean Brightness (0–255)", fontsize=9)
                ax.tick_params(colors=MUT, labelsize=9)
                ax.set_title("Mean Brightness by Category", fontsize=11, fontfamily="serif")
                for s in ax.spines.values(): s.set_visible(False)
                ax.xaxis.set_visible(False)
                ax.grid(axis="x", linewidth=0.5, color=BOR, zorder=0)
                xlim = max(vals_b) * 1.18
                for bar, val in zip(ax.patches, vals_b[::-1]):
                    ax.text(xlim, bar.get_y() + bar.get_height()/2,
                            f" {val:.1f}", va="center", ha="left", fontsize=8.5, color=TEXT)
                ax.set_xlim(0, xlim)
                plt.tight_layout()
                st.pyplot(fig)

            with tabs[1]:
                st.markdown("""
                **Mean R/G/B channel ratio** per category.
                R > G,B → reddish/brown terrain (bare soil, desert). G dominant → vegetation.
                """)
                cat_rgb = defaultdict(lambda: [0.0, 0.0, 0.0])
                cat_cnt = defaultdict(int)
                for i, img in enumerate(sampled):
                    cat = parse_category(img["filename"])
                    cat_rgb[cat][0] += r_ratio[i]
                    cat_rgb[cat][1] += g_ratio[i]
                    cat_rgb[cat][2] += b_ratio[i]
                    cat_cnt[cat] += 1
                for c in cat_rgb:
                    cat_rgb[c][0] /= cat_cnt[c]
                    cat_rgb[c][1] /= cat_cnt[c]
                    cat_rgb[c][2] /= cat_cnt[c]

                sort_rgb = sorted(cat_rgb.items(), key=lambda x: x[1][0]-x[1][1], reverse=True)
                cats_rgb = [c for c, _ in sort_rgb]
                r_vals   = [v[0] for _, v in sort_rgb]
                g_vals   = [v[1] for _, v in sort_rgb]
                b_vals   = [v[2] for _, v in sort_rgb]

                x = np.arange(len(cats_rgb))
                w = 0.25
                fig, ax = plt.subplots(figsize=(13, max(5, len(cats_rgb)*0.28)))
                fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
                ax.barh(x - w,   r_vals, height=w, color="#E57373", label="Red",   edgecolor="none")
                ax.barh(x,        g_vals, height=w, color="#81C784", label="Green", edgecolor="none")
                ax.barh(x + w,   b_vals, height=w, color="#64B5F6", label="Blue",  edgecolor="none")
                ax.set_yticks(x)
                ax.set_yticklabels(cats_rgb, fontsize=9)
                ax.set_xlabel("Mean Channel Ratio", fontsize=9)
                ax.set_title("Mean R/G/B Channel Ratio by Category", fontsize=11, fontfamily="serif")
                ax.legend(loc="lower right", fontsize=9, frameon=False)
                for s in ax.spines.values(): s.set_visible(False)
                ax.xaxis.set_visible(False)
                plt.tight_layout()
                st.pyplot(fig)

                # find most green and most red categories
                most_green = max(cat_rgb.items(), key=lambda x: x[1][1]-x[1][0])
                most_red   = max(cat_rgb.items(), key=lambda x: x[1][0]-x[1][2])
                insight_lines = [
                    f"→ Most greenish category: *{most_green[0]}* "
                    f"(G ratio = {most_green[1][1]:.3f}, likely vegetation or water reflections).",
                    f"→ Most reddish category: *{most_red[0]}* "
                    f"(R ratio = {most_red[1][0]:.3f}, likely bare soil or arid terrain).",
                ]
                st.markdown(
                    "<div class='insight' style='font-style:normal;'>" +
                    "".join(f"<p style='margin:0.3rem 0'>{p}</p>" for p in insight_lines) +
                    "</div>",
                    unsafe_allow_html=True,
                )

            with tabs[2]:
                st.markdown("""
                **Edge density** as a proxy for texture complexity.
                High edge density → many small objects / fine texture (urban, forest).
                Low edge density → smooth / uniform areas (desert, ocean, farmland).
                """)
                sort_ent = sorted(avg_entropy.items(), key=lambda x: x[1], reverse=True)
                cats_e, vals_e = zip(*sort_ent)
                cmap_e = plt.cm.Purples(np.linspace(0.35, 0.9, len(cats_e)))
                fig, ax = plt.subplots(figsize=(11, max(5, len(cats_e)*0.28)))
                fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
                ax.barh(list(cats_e)[::-1], list(vals_e)[::-1],
                        color=cmap_e.tolist()[::-1], edgecolor="none")
                ax.set_xlabel("Mean Edge Density", fontsize=9)
                ax.tick_params(colors=MUT, labelsize=9)
                ax.set_title("Texture Complexity by Category", fontsize=11, fontfamily="serif")
                for s in ax.spines.values(): s.set_visible(False)
                ax.xaxis.set_visible(False)
                ax.grid(axis="x", linewidth=0.5, color=BOR, zorder=0)
                xlim_e = max(vals_e) * 1.18
                for bar, val in zip(ax.patches, vals_e[::-1]):
                    ax.text(xlim_e, bar.get_y() + bar.get_height()/2,
                            f" {val:.2f}", va="center", ha="left", fontsize=8.5, color=TEXT)
                ax.set_xlim(0, xlim_e)
                plt.tight_layout()
                st.pyplot(fig)

                insight_lines = [
                    f"→ Highest texture: *{sort_ent[0][0]}* = {sort_ent[0][1]:.2f} "
                    f"(many edges → fine-grained or complex visual patterns).",
                    f"→ Smoothest: *{sort_ent[-1][0]}* = {sort_ent[-1][1]:.2f} "
                    f"(uniform surface with few edges).",
                    f"→ Range: {sort_ent[0][1]-sort_ent[-1][1]:.2f} — "
                    f"{'significant variation in scene complexity across land-use types.' if sort_ent[0][1]-sort_ent[-1][1] > 5 else 'moderate spread.'}",
                ]
                st.markdown(
                    "<div class='insight' style='font-style:normal;'>" +
                    "".join(f"<p style='margin:0.3rem 0'>{p}</p>" for p in insight_lines) +
                    "</div>",
                    unsafe_allow_html=True,
                )

            with tabs[3]:
                st.markdown("""
                Random sample of satellite images from the dataset, grouped by brightness level.
                """)
                # pick 3 bright, 3 medium, 3 dark from sampled
                bright_imgs = sorted(zip(brightness, sampled), key=lambda x: x[0], reverse=True)
                mid = len(bright_imgs) // 2
                bright3 = [img for _, img in bright_imgs[:3]]
                mid3    = [img for _, img in bright_imgs[mid-1:mid+2]]
                dark3   = [img for _, img in bright_imgs[-3:]]

                for label, grp in [("☀️ Brightest", bright3),
                                   ("🌤️  Medium",    mid3),
                                   ("🌑 Darkest",     dark3)]:
                    st.markdown(f"**{label}**")
                    cols = st.columns(3)
                    for ci, img in enumerate(grp):
                        p = IMG_DIR / img["filename"]
                        try:
                            pil = Image.open(p)
                            if pil.mode in ("RGBA", "LA", "P"):
                                pil = pil.convert("RGB")
                            with cols[ci]:
                                st.image(pil, use_column_width=True)
                                st.caption(f"`{img['filename']}`")
                        except Exception:
                            pass
                    st.markdown("")

        st.markdown("")
        if st.button("Next: Caption Vocabulary →"):
            st.session_state.log.append(f"[{time.strftime('%H:%M:%S')}] ✓ "
                                        f"Image Properties: brightness, RGB, texture computed")
            st.session_state.step = 3
            st.rerun()

    # ── STEP 3: Caption Vocabulary ───────────────────────────────────────────
    elif st.session_state.step == 3:
        st.markdown("<p class='section-head'>Step 4 of 7 — Caption Vocabulary</p>",
                    unsafe_allow_html=True)
        st.markdown("""
        <div class='insight' style='font-style:normal;border-left-color:#2E7D32;'>
          What words and phrases are most common in the captions?
          Does caption length follow a fixed pattern — hinting at a template — or is it varied?
        </div>""", unsafe_allow_html=True)

        top_n = st.slider("Top-N", 5, 30, 12)

        tab_w, tab_bg, tab_len = st.tabs(["Top Words", "Bigrams", "Length Distribution"])

        with tab_w:
            wf = D["word_freq"].most_common(top_n)
            words, counts = zip(*wf)
            fig, ax = plt.subplots(figsize=(10, max(4, len(words)*0.5)))
            fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
            cmap = plt.cm.Blues(np.linspace(0.4, 0.9, len(words)))[::-1]
            ax.barh(list(words)[::-1], list(counts)[::-1],
                    color=cmap.tolist(), edgecolor="none")
            ax.set_xlabel("Frequency", fontsize=9)
            ax.tick_params(colors=MUT, labelsize=10)
            ax.set_title(f"Top {top_n} Content Words", fontsize=11, fontfamily="serif")
            for s in ax.spines.values(): s.set_visible(False)
            ax.xaxis.set_visible(False)
            ax.grid(axis="x", linewidth=0.5, color=BOR, zorder=0)
            xlim_w = max(counts) * 1.18
            for bar, val in zip(ax.patches, counts[::-1]):
                ax.text(xlim_w, bar.get_y() + bar.get_height()/2,
                        f" {val:,}", va="center", ha="left", fontsize=9, color=TEXT)
            ax.set_xlim(0, xlim_w)
            plt.tight_layout()
            st.pyplot(fig)

        with tab_bg:
            top_bg_n = st.slider("Top-N Bigrams", 5, 20, 10, key="bg_n2")
            bg = D["bigram_freq"].most_common(top_bg_n)
            labels = [" ".join(bg_) for bg_, c_ in bg]
            vals   = [c_ for _, c_ in bg]
            fig, ax = plt.subplots(figsize=(10, max(3, len(labels)*0.5)))
            fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
            cmap = plt.cm.Oranges(np.linspace(0.4, 0.9, len(labels)))[::-1]
            ax.barh(labels[::-1], vals[::-1], color=cmap.tolist(), edgecolor="none")
            ax.tick_params(colors=MUT, labelsize=10)
            ax.set_title(f"Top {top_bg_n} Bigrams", fontsize=11, fontfamily="serif")
            for s in ax.spines.values(): s.set_visible(False)
            ax.xaxis.set_visible(False)
            ax.grid(axis="x", linewidth=0.5, color=BOR, zorder=0)
            xlim_bg = max(vals) * 1.18
            for bar, val in zip(ax.patches, vals[::-1]):
                ax.text(xlim_bg, bar.get_y() + bar.get_height()/2,
                        f" {val:,}", va="center", ha="left", fontsize=9, color=TEXT)
            ax.set_xlim(0, xlim_bg)
            plt.tight_layout()
            st.pyplot(fig)

        with tab_len:
            fig, ax = plt.subplots(figsize=(9, 4))
            fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
            ax.hist(D["cap_lens"], bins=28, color=ACC, edgecolor=BG, alpha=0.9, rwidth=0.85)
            ax.axvline(D["avg_cap_len"], color=TEXT, linestyle="--",
                       linewidth=2, label=f"μ = {D['avg_cap_len']:.1f}")
            ax.axvline(D["avg_cap_len"] - D["std_cap_len"], color=MUT,
                       linestyle=":", linewidth=1.5, alpha=0.7)
            ax.axvline(D["avg_cap_len"] + D["std_cap_len"], color=MUT,
                       linestyle=":", linewidth=1.5, alpha=0.7)
            ax.legend(facecolor=BG, labelcolor=TEXT, fontsize=9)
            ax.set_xlabel("Words per Caption", fontsize=9)
            ax.set_ylabel("Frequency", fontsize=9)
            ax.set_title("Caption Length Distribution", fontsize=11,
                         fontfamily="serif", fontweight="bold")
            ax.tick_params(colors=MUT, labelsize=9)
            for s in ax.spines.values(): s.set_edgecolor(BOR)
            ax.yaxis.set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)

        top_word    = D["word_freq"].most_common(1)[0]
        top_bigram  = D["bigram_freq"].most_common(1)[0]
        total_words = sum(D["word_freq"].values())

        st.markdown("")
        if st.button("Next: Caption Vocabulary →"):
            st.session_state.log.append(f"[{time.strftime('%H:%M:%S')}] ✓ "
                                        f"Vocab: {D['vocab_clean']:,} words · Top: '{top_word[0]}'")
            st.session_state.step = 4
            st.rerun()

    # ── STEP 4: Color Words ─────────────────────────────────────────────────
    elif st.session_state.step == 4:
        st.markdown("<p class='section-head'>Step 5 of 7 — Color Words</p>",
                    unsafe_allow_html=True)
        st.markdown("""
        <div class='insight' style='font-style:normal;border-left-color:#FF9800;'>
          Do captions describe visual properties like color?
          If so, which colors appear most frequently — and what does that reveal about aerial imagery?
        </div>""", unsafe_allow_html=True)

        color_pct   = D["caps_with_color"] / D["total_caps"] * 100
        total_color = sum(D["color_freq"].values())
        top_color   = D["color_freq"].most_common(1)[0]

        k1, k2, k3 = st.columns(3)
        k1.metric("Captions with Color", f"{color_pct:.1f}%")
        k2.metric("Total Color Words", f"{total_color:,}")
        k3.metric("Top Color", f"'{top_color[0]}' ({top_color[1]:,})")

        tab_cf, tab_img, tab_ex = st.tabs([
            "📊  Frequency Chart",
            "🖼️  Color → Images",
            "📝  Caption Examples",
        ])

        with tab_cf:
            top_n = st.slider("Top-N colors", 5, 17, 12, key="color_n2")
            cf = D["color_freq"].most_common(top_n)
            colors, counts = zip(*cf)
            fig, ax = plt.subplots(figsize=(10, max(3, len(colors)*0.5)))
            fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
            hex_cols = [COLOR_HEX.get(c, '#9E9E9E') for c in colors]
            ax.barh(list(colors)[::-1], list(counts)[::-1],
                    color=hex_cols[::-1], edgecolor="none")
            ax.set_xlabel("Frequency", fontsize=9)
            ax.tick_params(colors=MUT, labelsize=10)
            ax.set_title(f"Top {len(colors)} Color Words", fontsize=11, fontfamily="serif")
            for s in ax.spines.values(): s.set_visible(False)
            ax.xaxis.set_visible(False)
            ax.grid(axis="x", linewidth=0.5, color=BOR, zorder=0)
            xlim_c = max(counts) * 1.18
            for bar, val in zip(ax.patches, counts[::-1]):
                ax.text(xlim_c, bar.get_y() + bar.get_height()/2,
                        f" {val:,}", va="center", ha="left", fontsize=9, color=TEXT)
            ax.set_xlim(0, xlim_c)
            plt.tight_layout()
            st.pyplot(fig)

            swatches = " &nbsp; ".join([
                f"<span style='background:{COLOR_HEX.get(c,'#999')};color:{'white' if c in ('dark','black','brown','purple') else '#111'};"
                f"padding:0.1rem 0.4rem;border-radius:3px;font-size:0.78rem;'>"
                f"{c} ({v})</span>"
                for c, v in cf[:8]
            ])
            st.markdown(f"<div style='padding:0.5rem 0;'>{swatches}</div>",
                        unsafe_allow_html=True)

        with tab_img:
            st.markdown("##### Pick a color — see the satellite images it describes")
            top_color_name = D["color_freq"].most_common(1)[0][0]
            color_options  = [c for c, _ in D["color_freq"].most_common(15)]
            chosen_color  = st.selectbox(
                "Color word", color_options,
                index=color_options.index(top_color_name),
            )

            # build list of images whose captions contain this color
            color_img_samples = []
            for img in D["train_imgs"]:
                matched_caps = []
                for s in img["sentences"]:
                    words = tokenize(s["raw"])
                    if chosen_color in words:
                        matched_caps.append(s["raw"])
                if matched_caps:
                    color_img_samples.append((img, matched_caps))
                if len(color_img_samples) >= 6:
                    break

            if not color_img_samples:
                st.info(f"No images found with caption containing '{chosen_color}'.")
            else:
                hx = COLOR_HEX.get(chosen_color, '#9E9E9E')
                dark_label = chosen_color in ('dark', 'black', 'brown', 'purple')
                label_color = "white" if dark_label else "#111"

                for idx, (img, caps) in enumerate(color_img_samples):
                    img_path = IMG_DIR / img["filename"]
                    cat = parse_category(img["filename"])
                    row_img, row_cap = st.columns([1, 2])
                    with row_img:
                        try:
                            pil = Image.open(img_path)
                            if pil.mode in ("RGBA", "LA", "P"):
                                pil = pil.convert("RGB")
                            st.image(pil, use_column_width=True)
                        except Exception:
                            pass
                        st.caption(f"{cat}")
                    with row_cap:
                        st.markdown(f"Captions with **'{chosen_color}'**:")
                        for cap_idx, cap in enumerate(caps[:3]):
                            words = cap.split()
                            hl = []
                            for w in words:
                                if w.lower() == chosen_color:
                                    hl.append(
                                        f"<span style='background:{hx};color:{label_color};"
                                        f"padding:0.1rem 0.3rem;border-radius:3px;"
                                        f"font-weight:600;font-size:0.85rem;'>{w}</span>"
                                    )
                                else:
                                    hl.append(f"<span style='color:{TEXT}'>{w}</span>")
                            st.markdown(
                                f"<div style='background:{CARD};padding:0.4rem 0.7rem;"
                                f"margin:0.2rem 0;border-radius:0 4px 4px 0;"
                                f"border-left:3px solid {hx};line-height:1.8;font-size:0.85rem;'>"
                                f"{' '.join(hl)}</div>",
                                unsafe_allow_html=True,
                            )
                    st.markdown("")

        with tab_ex:
            n_ex = st.slider("Show N samples", 5, 20, 10, key="color_ex2")
            sample_caps = []
            for cap in D["train_caps"][:500]:
                words = tokenize(cap)
                colors_in = [w for w in words if w in COLOR_WORDS]
                if colors_in:
                    sample_caps.append((cap, colors_in))
                if len(sample_caps) >= n_ex:
                    break

            for cap, colors_in in sample_caps:
                words = cap.split()
                hl = []
                for w in words:
                    if w.lower() in colors_in:
                        hx = COLOR_HEX.get(w.lower(), '#9E9E9E')
                        dark = w.lower() in ('dark', 'black', 'brown', 'purple')
                        hl.append(
                            f"<span style='background:{hx};color:{'white' if dark else TEXT};"
                            f"padding:0.1rem 0.3rem;border-radius:3px;"
                            f"font-weight:600;font-size:0.88rem;'>{w}</span>"
                        )
                    else:
                        hl.append(f"<span style='color:{TEXT}'>{w}</span>")
                st.markdown(
                    f"<div style='background:{CARD};padding:0.5rem 0.8rem;margin:0.3rem 0;"
                    f"border-radius:4px;line-height:2;font-size:0.88rem;'>"
                    f"{' '.join(hl)}</div>",
                    unsafe_allow_html=True,
                )

        color_pct   = D["caps_with_color"] / D["total_caps"] * 100
        total_color = sum(D["color_freq"].values())
        top_color   = D["color_freq"].most_common(1)[0]

        st.markdown("")
        if st.button("Next: Spatial Relations →"):
            st.session_state.log.append(f"[{time.strftime('%H:%M:%S')}] ✓ "
                                        f"Colors: {color_pct:.1f}% caps · Top: '{top_color[0]}'")
            st.session_state.step = 5
            st.rerun()

    # ── STEP 5: Spatial Relations ────────────────────────────────────────────
    elif st.session_state.step == 5:
        st.markdown("<p class='section-head'>Step 6 of 7 — Spatial Relations</p>",
                    unsafe_allow_html=True)
        st.markdown("""
        <div class='insight' style='font-style:normal;border-left-color:#4488BB;'>
          Do captions describe spatial relationships between objects?
          From a top-down aerial view, which spatial prepositions are most common?
        </div>""", unsafe_allow_html=True)

        sp_pct    = D["caps_with_spatial"] / D["total_caps"] * 100
        top_sp    = D["spatial_counts"].most_common(1)[0]
        total_sp  = sum(D["spatial_counts"].values())

        k1, k2, k3 = st.columns(3)
        k1.metric("Captions with Spatial", f"{sp_pct:.1f}%")
        k2.metric("Total Occurrences", f"{total_sp:,}")
        k3.metric("Top Preposition", f"'{top_sp[0]}' ({top_sp[1]:,})")

        sp = D["spatial_counts"].most_common()
        labels, vals = zip(*sp)
        fig, ax = plt.subplots(figsize=(9, max(3, len(labels)*0.55)))
        fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
        ax.barh(list(labels)[::-1], list(vals)[::-1],
                               color="#4488BB", edgecolor="none")
        ax.tick_params(colors=MUT, labelsize=11)
        ax.set_title("Spatial Preposition Frequency", fontsize=12,
                     fontfamily="serif", fontweight="bold")
        for s in ax.spines.values(): s.set_visible(False)
        ax.xaxis.set_visible(False)
        ax.grid(axis="x", linewidth=0.5, color=BOR, zorder=0)
        xlim_sp = max(vals) * 1.18
        for bar, val in zip(ax.patches, vals[::-1]):
            ax.text(xlim_sp, bar.get_y() + bar.get_height()/2,
                    f" {val:,}", va="center", ha="left", fontsize=9, color=TEXT)
        ax.set_xlim(0, xlim_sp)
        plt.tight_layout()
        st.pyplot(fig)

        n_ex = st.slider("Show N examples", 3, 15, 6, key="sp_ex2")
        sample_caps = []
        for cap in D["train_caps"]:
            cl = cap.lower()
            found = [kw for kw in SPATIAL_KEYWORDS if kw in cl]
            if found:
                sample_caps.append((cap, found[0]))
                if len(sample_caps) >= n_ex:
                    break

        st.markdown("##### Captions with Spatial Relations")
        for cap, kw in sample_caps:
            start = cap.lower().find(kw)
            if start < 0:
                continue
            end = start + len(kw)
            st.markdown(
                f"<div style='background:{CARD};padding:0.5rem 0.8rem;margin:0.3rem 0;"
                f"border-left:3px solid #4488BB;border-radius:0 4px 4px 0;"
                f"font-size:0.88rem;line-height:1.7;'>"
                f"{cap[:start]}"
                f"<mark style='background:#FFF9C4;color:#E65100;"
                f"padding:0.1rem 0.3rem;border-radius:3px;font-weight:700;'>"
                f"{cap[start:end]}</mark>"
                f"{cap[end:]}</div>",
                unsafe_allow_html=True,
            )

        st.markdown("")
        if st.button("Next: Caption Variability →"):
            st.session_state.log.append(f"[{time.strftime('%H:%M:%S')}] ✓ "
                                        f"Spatial: {sp_pct:.1f}% caps · Top: '{top_sp[0]}'")
            st.session_state.step = 6
            st.rerun()

    # ── STEP 6: Caption Variability ─────────────────────────────────────────
    elif st.session_state.step == 6:
        st.markdown("<p class='section-head'>Step 7 of 7 — Caption Variability</p>",
                    unsafe_allow_html=True)
        st.markdown("""
        <div class='insight' style='font-style:normal;border-left-color:#9b59b6;'>
          How much do the 5 captions for each image differ in length?
          Low variance → strict template. High variance → rich paraphrasing diversity.
        </div>""", unsafe_allow_html=True)

        k1, k2 = st.columns(2)
        k1.metric("Mean within-image std", f"{D['mean_var']} words")
        k2.metric("Max within-image std", f"{D['max_var']} words")

        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

        axes[0].hist(D["variabilities"], bins=28, edgecolor=BG,
                        color="#9b59b6", alpha=0.85)
        axes[0].axvline(D["mean_var"], color=ACC, linestyle="--", lw=2,
                        label=f"Mean = {D['mean_var']}")
        axes[0].set_xlabel("Caption-Length Std (within same image)", fontsize=9)
        axes[0].set_ylabel("Frequency", fontsize=9)
        axes[0].set_title("Within-Image Variability", fontsize=11,
                          fontfamily="serif")
        axes[0].legend(facecolor=BG, labelcolor=TEXT, fontsize=9)
        axes[0].set_facecolor(BG)
        for s in axes[0].spines.values(): s.set_edgecolor(BOR)
        axes[0].tick_params(colors=MUT, labelsize=8)
        axes[0].yaxis.set_visible(False)

        cat_var_sorted = sorted(D["cat_var"].items(),
                                key=lambda x: x[1], reverse=True)[:12]
        cats_, means_ = zip(*cat_var_sorted)
        axes[1].barh(list(cats_)[::-1], list(means_)[::-1],
                     color="#3498db", edgecolor="none")
        axes[1].set_xlabel("Mean Caption-Length Std", fontsize=9)
        axes[1].set_title("By Category (Top 12)", fontsize=11, fontfamily="serif")
        axes[1].set_facecolor(BG)
        for s in axes[1].spines.values(): s.set_visible(False)
        axes[1].tick_params(colors=MUT, labelsize=9)
        axes[1].xaxis.set_visible(False)
        axes[1].grid(axis="x", linewidth=0.5, color=BOR, zorder=0)
        xlim_v = max(means_) * 1.18
        for bar, val in zip(axes[1].patches, means_[::-1]):
            axes[1].text(xlim_v, bar.get_y() + bar.get_height()/2,
                         f" {val:.2f}", va="center", ha="left", fontsize=8.5, color=TEXT)
        axes[1].set_xlim(0, xlim_v)

        plt.suptitle("Caption Variability Analysis",
                     fontsize=12, fontweight="bold", fontfamily="serif")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        st.pyplot(fig)

        st.markdown("##### One Image — Five Perspectives")

        # User controls for Step 7
        ctrl1, ctrl2, ctrl3 = st.columns([1, 1, 1])
        with ctrl1:
            gallery_mode = st.selectbox(
                "Gallery mode",
                ["Balanced", "Most variable", "Least variable"],
                index=0,
            )
        with ctrl2:
            n_pick = st.slider("Number of samples", 8, 40, 20, 4)
        with ctrl3:
            only_category = st.selectbox(
                "Filter category",
                ["All"] + sorted(D["train_cats"].keys()),
                index=0,
            )

        import random
        random.seed(99)
        all_train = D["train_imgs"]
        variabilities = D["variabilities"]
        idx_var_pairs = list(enumerate(variabilities))

        if only_category != "All":
            idx_var_pairs = [
                (i, v) for i, v in idx_var_pairs
                if parse_category(all_train[i]["filename"]) == only_category
            ]

        if not idx_var_pairs:
            st.warning("No sample available for this category filter.")
            st.stop()

        sorted_by_var = sorted(idx_var_pairs, key=lambda x: x[1])

        if gallery_mode == "Most variable":
            selected_pairs = sorted_by_var[-n_pick:][::-1]
        elif gallery_mode == "Least variable":
            selected_pairs = sorted_by_var[:n_pick]
        else:
            step = max(1, len(sorted_by_var) // max(1, n_pick))
            selected_pairs = [sorted_by_var[i] for i in range(0, len(sorted_by_var), step)][:n_pick]

        gallery_pool = [all_train[i] for i, _ in selected_pairs]
        random.shuffle(gallery_pool)

        opts = [f"{parse_category(img['filename'])} — {img['filename']}" for img in gallery_pool]
        sel = st.selectbox("Select an image to view its 5 captions", opts, index=0)
        sample = gallery_pool[opts.index(sel)]
        img_path = IMG_DIR / sample["filename"]
        cat = parse_category(sample["filename"])

        col_img, col_caps = st.columns([1, 2])
        with col_img:
            try:
                img_pil = Image.open(img_path)
                if img_pil.mode in ("RGBA", "LA", "P"):
                    img_pil = img_pil.convert("RGB")
                st.image(img_pil, use_column_width=True)
            except Exception:
                pass
            st.caption(f"Category: {cat}")
            lens = [len(s["tokens"]) for s in sample["sentences"]]
            st.caption(f"Lengths: {' · '.join(map(str, lens))}")

        with col_caps:
            for i, s in enumerate(sample["sentences"]):
                bg, fg = SENTENCE_COLORS[i]
                st.markdown(
                    f"<div style='background:{bg};border-left:3px solid {fg};"
                    f"padding:0.4rem 0.7rem;margin:0.25rem 0;border-radius:0 3px 3px 0;'>"
                    f"<span style='color:{fg};font-size:0.72rem;font-weight:700;'>"
                    f"#{i+1}</span>&nbsp;"
                    f"<span style='color:{TEXT};font-size:0.85rem;'>{s['raw']}</span></div>",
                    unsafe_allow_html=True,
                )

        st.markdown("")
        if st.button("Next: Noise Detection →", key="view_noise"):
            st.session_state.log.append(f"[{time.strftime('%H:%M:%S')}] ✓ "
                                        f"Variability: mean std={D['mean_var']} words")
            st.session_state.step = 7
            st.rerun()

    # ── STEP 7: Noise Detection ───────────────────────────────────────────────
    elif st.session_state.step == 7:
        st.markdown("<p class='section-head'>Step 8 of 8 — Noise Detection</p>",
                    unsafe_allow_html=True)
        st.markdown("""
        <div class='insight' style='font-style:normal;border-left-color:#C0392B;'>
          Are there any mislabeled captions? We apply a Multi-Layer Anomaly Detection:
          <b>Local Text Noise</b> (Cosine Similarity via TF-IDF) and <b>Cross-Modal Mismatch</b> 
          (Mismatching described visual grounding against pixel dominant color). 
        </div>""", unsafe_allow_html=True)

        st.info("Note: Dominant Colors (RGB Channel) are dynamically fetched via fast 5-image sampling per Category to ensure real-time performance on this tool.")

        ncol1, ncol2 = st.columns(2)
        with ncol1:
            threshold = st.slider(
                "Cosine Similarity Threshold (Layer A)",
                min_value=0.0,
                max_value=0.15,
                value=0.05,
                step=0.01,
            )
        with ncol2:
            layer_b_enabled = st.toggle("Enable Layer B (Cross-Modal Mismatch)", value=True)
            layer_b_strict = st.selectbox(
                "Layer B strictness",
                ["Strict", "Balanced", "Sensitive"],
                index=1,
                disabled=not layer_b_enabled,
            )

        st.caption("Layer A checks text outliers within category; Layer B checks caption color words vs dominant image channel.")

        with st.spinner("Detecting anomalies..."):
            anomalies = []
            category_vectorizers = D["category_vectorizers"]
            category_centroids = D["category_centroids"]
            fast_dom_channels = D["fast_dom_channels"]

            for img in D["train_imgs"]:
                cat = parse_category(img['filename'])
                filename = img['filename']
                
                # Layer B Setup
                img_dom_channel = fast_dom_channels.get(cat, 'R≈G>B')
                
                caps = [s['raw'] for s in img['sentences']]
                
                for i, cap in enumerate(caps):
                    layer_a_flag = False
                    layer_b_flag = False
                    reasons = []
                    
                    # --- LAYER A: Local Text Noise ---
                    if cat in category_vectorizers:
                        vec = category_vectorizers[cat]
                        centroid = category_centroids[cat]
                        cap_vec = vec.transform([cap]).toarray()
                        if np.sum(cap_vec) > 0:
                            sim = cosine_similarity(cap_vec, centroid.reshape(1, -1))[0][0]
                        else:
                            sim = 0.0
                        
                        if sim < threshold:
                            layer_a_flag = True
                            reasons.append(f"CosineSim={sim:.3f}")
                    
                    # --- LAYER B: Multimodal Mismatch ---
                    cap_lower = cap.lower()
                    water_blue_kws = ['water', 'blue', 'ocean', 'sea', 'lake', 'river']
                    green_forest_kws = ['green', 'forest', 'tree', 'woods', 'grass', 'park']
                    
                    has_blue = any(w in cap_lower.replace(',', ' ').replace('.', ' ').split() for w in water_blue_kws)
                    has_green = any(w in cap_lower.replace(',', ' ').replace('.', ' ').split() for w in green_forest_kws)
                    
                    if layer_b_enabled:
                        if layer_b_strict == "Strict":
                            blue_mismatch_channels = ['R>G>B']
                            green_mismatch_channels = ['R>G>B', 'B>R>G']
                        elif layer_b_strict == "Sensitive":
                            blue_mismatch_channels = ['R>G>B', 'G>R>B', 'B>R>G']
                            green_mismatch_channels = ['R>G>B', 'B>R>G', 'R≈G>B']
                        else:  # Balanced
                            blue_mismatch_channels = ['R>G>B', 'G>R>B']
                            green_mismatch_channels = ['R>G>B', 'B>R>G']

                        if has_blue and img_dom_channel in blue_mismatch_channels:
                            layer_b_flag = True
                            reasons.append(f"LayerB Blue/Water mismatch: Pixels={img_dom_channel}")

                        if has_green and img_dom_channel in green_mismatch_channels:
                            layer_b_flag = True
                            reasons.append(f"LayerB Green/Forest mismatch: Pixels={img_dom_channel}")
                    
                    if layer_a_flag or layer_b_flag:
                        confidence = "HIGH" if (layer_a_flag and layer_b_flag) else "LOW"
                        anomalies.append({
                            'filename': filename,
                            'category': cat,
                            'caption_idx': i + 1,
                            'caption': cap,
                            'confidence': confidence,
                            'reason': "; ".join(reasons)
                        })

        high_conf = sum(1 for a in anomalies if a['confidence'] == 'HIGH')
        low_conf = len(anomalies) - high_conf
        layer_a_only = sum(1 for a in anomalies if ('CosineSim=' in a['reason']) and ('LayerB ' not in a['reason']))
        layer_b_only = sum(1 for a in anomalies if ('LayerB ' in a['reason']) and ('CosineSim=' not in a['reason']))

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Anomalies", len(anomalies))
        c2.metric("High Confidence", high_conf)
        c3.metric("Low Confidence", low_conf)
        c4.metric("Layer A only", layer_a_only)
        c5.metric("Layer B only", layer_b_only)

        if len(anomalies) > 0:
            df_anomalies = pd.DataFrame(anomalies)
            # Make sure HIGH puts first
            df_anomalies = df_anomalies.sort_values(by=['confidence', 'category'], ascending=[True, True])
            st.dataframe(df_anomalies, height=250, use_container_width=True)

            st.markdown("##### Inspect Anomaly")
            opts = [f"{a['filename']} (Cap #{a['caption_idx']}) - {a['confidence']} Conf - Reason: {a['reason'][:30]}..." for _, a in df_anomalies.iterrows()]
            sel = st.selectbox("Choose an anomaly to inspect:", opts)
            idx = opts.index(sel)
            selected_anomaly = df_anomalies.iloc[idx]

            col1, col2 = st.columns([1, 2])
            with col1:
                img_path = IMG_DIR / selected_anomaly["filename"]
                try:
                    pil = Image.open(img_path)
                    if pil.mode in ("RGBA", "LA", "P"):
                        pil = pil.convert("RGB")
                    st.image(pil, use_column_width=True)
                except Exception:
                    pass
                st.caption(f"Category: {selected_anomaly['category']}")
            with col2:
                st.error("🚨 Filtered Caption:")
                st.markdown(f"**{selected_anomaly['caption']}**")
                st.write(f"**Reason:** {selected_anomaly['reason']}")

        st.markdown("")
        if st.button("✅  View Full Dashboard →", key="view_dashboard_final"):
            st.session_state.log.append(f"[{time.strftime('%H:%M:%S')}] ✓ "
                                        f"Noise Det: {len(anomalies)} anomalies found")
            
            # Store anomaly count for dashboard
            st.session_state.anomaly_count = len(anomalies)
            st.session_state.phase = "done"
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
#  DONE STATE
# ══════════════════════════════════════════════════════════════════════════════
    if st.session_state.phase == "done":
        st.markdown("")
        st.markdown("#### 📊 Full Dashboard — RSITMD Multimodal Overview", unsafe_allow_html=True)

        # ── Key Metrics Row ────────────────────────────────────────────────────────
        st.markdown("---")
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Images", f"{D['n_train'] + D['n_test']:,}")
        m2.metric("Categories", str(D['n_cats']))
        m3.metric("Captions", f"{D['total_caps']:,}")
        m4.metric("Vocab", f"{D['vocab_clean']:,}")
        color_pct  = D['caps_with_color']  / D['total_caps'] * 100
        sp_pct     = D['caps_with_spatial'] / D['total_caps'] * 100
        m5.metric("Color Caps", f"{color_pct:.0f}%")
        noisy_count = st.session_state.get('anomaly_count', 0)
        m6.metric("Noisy Captions", str(noisy_count))

        # ── Row 1: Category dist + Caption length dist ─────────────────────────────
        st.markdown("---")
        r1_c1, r1_c2 = st.columns(2)

        with r1_c1:
            st.markdown("**Category Distribution**")
            sorted_cats = D["train_cats"].most_common()
            top_cats = sorted_cats[:20]
            labels_c, vals_c = zip(*top_cats)
            n = len(labels_c)
            fig_h = max(4, n * 0.22)
            fig1, ax1 = plt.subplots(figsize=(7.5, fig_h))
            fig1.patch.set_facecolor(BG)
            ax1.set_facecolor(BG)
            ax1.barh(list(labels_c)[::-1], list(vals_c)[::-1],
                     color="#2980b9", edgecolor="none")
            ax1.set_xlabel("Train Images", fontsize=8)
            ax1.set_title("Top 20 Categories", fontsize=9, fontfamily="serif")
            ax1.tick_params(colors=MUT, labelsize=8)
            ax1.xaxis.set_visible(False)
            for s in ax1.spines.values(): s.set_visible(False)
            for bar, val in zip(ax1.patches, vals_c[::-1]):
                ax1.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                         f" {val}", va="center", ha="left", fontsize=7.5, color=TEXT)
            xlim_c = max(vals_c) * 1.12
            ax1.set_xlim(0, xlim_c)
            plt.tight_layout()
            st.pyplot(fig1)

        with r1_c2:
            st.markdown("**Caption Length Distribution**")
            fig2, ax2 = plt.subplots(figsize=(7.5, 4.5))
            fig2.patch.set_facecolor(BG)
            ax2.set_facecolor(BG)
            ax2.hist(D["cap_lens"], bins=28, color=ACC, edgecolor=BG, alpha=0.9, rwidth=0.85)
            ax2.axvline(D["avg_cap_len"], color=TEXT, linestyle="--", linewidth=2,
                        label=f"μ = {D['avg_cap_len']:.1f}")
            ax2.axvline(D["avg_cap_len"] - D["std_cap_len"], color=MUT,
                        linestyle=":", linewidth=1.5, label=f"±σ = {D['std_cap_len']:.1f}")
            ax2.axvline(D["avg_cap_len"] + D["std_cap_len"], color=MUT, linestyle=":", linewidth=1.5)
            ax2.set_xlabel("Words per Caption", fontsize=9)
            ax2.set_ylabel("Frequency", fontsize=9)
            ax2.set_title("Caption Length Histogram", fontsize=10, fontfamily="serif")
            ax2.legend(facecolor=BG, labelcolor=TEXT, fontsize=8)
            ax2.tick_params(colors=MUT, labelsize=8)
            ax2.yaxis.set_visible(False)
            for s in ax2.spines.values(): s.set_edgecolor(BOR)
            plt.tight_layout()
            st.pyplot(fig2)

        # ── Row 2: Top Words + Variability ─────────────────────────────────────────
        r2_c1, r2_c2 = st.columns(2)

        with r2_c1:
            st.markdown("**Top Words & Bigrams**")
            wf = D["word_freq"].most_common(12)
            w_labels, w_vals = zip(*wf)
            fig3, ax3 = plt.subplots(figsize=(7.5, 3.5))
            fig3.patch.set_facecolor(BG)
            ax3.set_facecolor(BG)
            ax3.barh(list(w_labels)[::-1], list(w_vals)[::-1], color=ACC, edgecolor="none")
            ax3.set_xlabel("Frequency", fontsize=8)
            ax3.set_title("Top 12 Words (no stopwords)", fontsize=9, fontfamily="serif")
            ax3.tick_params(colors=MUT, labelsize=8)
            ax3.xaxis.set_visible(False)
            for s in ax3.spines.values(): s.set_visible(False)
            for bar, val in zip(ax3.patches, w_vals[::-1]):
                ax3.text(bar.get_width() + 5, bar.get_y() + bar.get_height() / 2,
                         f" {val}", va="center", ha="left", fontsize=7.5, color=TEXT)
            xlim_w = max(w_vals) * 1.12
            ax3.set_xlim(0, xlim_w)
            plt.tight_layout()
            st.pyplot(fig3)

        with r2_c2:
            st.markdown("**Caption Variability**")
            fig4, ax4 = plt.subplots(figsize=(7.5, 3.5))
            fig4.patch.set_facecolor(BG)
            ax4.set_facecolor(BG)
            ax4.hist(D["variabilities"], bins=28, edgecolor=BG, color="#9b59b6", alpha=0.85)
            ax4.axvline(D["mean_var"], color=ACC, linestyle="--", lw=2,
                        label=f"Mean = {D['mean_var']:.2f}")
            ax4.set_xlabel("Within-Image Std (words)", fontsize=8)
            ax4.set_ylabel("Frequency", fontsize=8)
            ax4.set_title("Caption Length Variability", fontsize=9, fontfamily="serif")
            ax4.legend(facecolor=BG, labelcolor=TEXT, fontsize=8)
            ax4.tick_params(colors=MUT, labelsize=8)
            ax4.yaxis.set_visible(False)
            for s in ax4.spines.values(): s.set_edgecolor(BOR)
            plt.tight_layout()
            st.pyplot(fig4)

        # ── Row 3: Spatial + Color ────────────────────────────────────────────────
        r3_c1, r3_c2 = st.columns(2)

        with r3_c1:
            st.markdown(f"**Spatial Relations** ({sp_pct:.0f}% of captions)")
            sp_data = D["spatial_counts"].most_common(12)
            sp_labs, sp_vals = zip(*sp_data)
            fig5, ax5 = plt.subplots(figsize=(7.5, 3.5))
            fig5.patch.set_facecolor(BG)
            ax5.set_facecolor(BG)
            ax5.barh(list(sp_labs)[::-1], list(sp_vals)[::-1], color="#27ae60", edgecolor="none")
            ax5.set_xlabel("Count", fontsize=8)
            ax5.set_title("Top 12 Spatial Words", fontsize=9, fontfamily="serif")
            ax5.tick_params(colors=MUT, labelsize=8)
            ax5.xaxis.set_visible(False)
            for s in ax5.spines.values(): s.set_visible(False)
            for bar, val in zip(ax5.patches, sp_vals[::-1]):
                ax5.text(bar.get_width() + 2, bar.get_y() + bar.get_height() / 2,
                         f" {val}", va="center", ha="left", fontsize=7.5, color=TEXT)
            xlim_s = max(sp_vals) * 1.12
            ax5.set_xlim(0, xlim_s)
            plt.tight_layout()
            st.pyplot(fig5)

        with r3_c2:
            st.markdown(f"**Color Words** ({color_pct:.0f}% of captions)")
            cf = D["color_freq"].most_common(12)
            cf_labs, cf_vals = zip(*cf)
            fig6, ax6 = plt.subplots(figsize=(7.5, 3.5))
            fig6.patch.set_facecolor(BG)
            ax6.set_facecolor(BG)
            ax6.barh(list(cf_labs)[::-1], list(cf_vals)[::-1], color="#e67e22", edgecolor="none")
            ax6.set_xlabel("Count", fontsize=8)
            ax6.set_title("Top 12 Color Words", fontsize=9, fontfamily="serif")
            ax6.tick_params(colors=MUT, labelsize=8)
            ax6.xaxis.set_visible(False)
            for s in ax6.spines.values(): s.set_visible(False)
            for bar, val in zip(ax6.patches, cf_vals[::-1]):
                ax6.text(bar.get_width() + 2, bar.get_y() + bar.get_height() / 2,
                         f" {val}", va="center", ha="left", fontsize=7.5, color=TEXT)
            xlim_col = max(cf_vals) * 1.12
            ax6.set_xlim(0, xlim_col)
            plt.tight_layout()
            st.pyplot(fig6)

        # ── Row 4: One sample image + all 5 captions ───────────────────────────────
        st.markdown("---")
        st.markdown("**One Image — Five Perspectives**")
        import random
        random.seed(42)
        all_train = D["train_imgs"]
        rand_img = random.choice(all_train)
        img_path = IMG_DIR / rand_img["filename"]
        cat = parse_category(rand_img["filename"])

        col_img, col_caps = st.columns([1, 2])
        with col_img:
            try:
                img_pil = Image.open(img_path)
                if img_pil.mode in ("RGBA", "LA", "P"):
                    img_pil = img_pil.convert("RGB")
                st.image(img_pil, use_column_width=True)
            except Exception:
                pass
            st.caption(f"Category: {cat}")
            lens = [len(s["tokens"]) for s in rand_img["sentences"]]
            st.caption(f"Lengths: {' · '.join(map(str, lens))}")

        with col_caps:
            for i, s in enumerate(rand_img["sentences"]):
                bg, fg = SENTENCE_COLORS[i]
                st.markdown(
                    f"<div style='background:{bg};border-left:3px solid {fg};"
                    f"padding:0.4rem 0.7rem;margin:0.25rem 0;border-radius:0 3px 3px 0;'>"
                    f"<span style='color:{fg};font-size:0.72rem;font-weight:700;'>"
                    f"#{i+1}</span>&nbsp;"
                    f"<span style='color:{TEXT};font-size:0.85rem;'>{s['raw']}</span></div>",
                    unsafe_allow_html=True,
                )

        # ── Footer + restart ───────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown(
            f"<p class='footer'>IEEE TGRS 2021 · RSITMD · UIT · 2025–2026 · "
            f"Source: {DATA_FILE.name} · {D['n_train'] + D['n_test']:,} images · "
            f"{D['total_caps']:,} captions</p>",
            unsafe_allow_html=True,
        )
        col_left, col_btn = st.columns([1, 1])
        with col_left:
            st.success("Analysis Complete — all 7 steps finished.")
        with col_btn:
            if st.button("← Start Over", key="start_over_done"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
