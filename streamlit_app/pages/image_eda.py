"""
Image EDA — RSITMD Image Analysis
Editorial layout: warm paper, Playfair Display typography
"""
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Image EDA", page_icon="🖼️", layout="wide", initial_sidebar_state="collapsed")

BG, CARD, TEXT, ACCENT, MUTED, BORDER, RULE = (
    "#F7F3EB","#EFE8DC","#111111","#B42318","#6B6560","#D4C9B8","#C8BBB0")

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,700;0,900;1,700&family=Source+Sans+3:wght@300;400;600;700&display=swap');
:root {{ --bg:{BG}; --card:{CARD}; --text:{TEXT}; --accent:{ACCENT}; --muted:{MUTED}; --border:{BORDER}; --rule:{RULE}; }}
body,.stApp {{ background:var(--bg); color:var(--text); font-family:'Source Sans 3',sans-serif; }}
[data-testid="stSidebar"] {{ background:#EDE9E1 !important; border-right:2px solid var(--rule); }}
h1,h2,h3 {{ font-family:'Playfair Display',serif; color:var(--text); line-height:1.15; }}
.section-label {{ font-size:0.68rem; font-weight:700; letter-spacing:0.18em; text-transform:uppercase; color:var(--accent); margin-bottom:0.3rem; }}
.hero-kicker {{ font-size:0.7rem; font-weight:700; letter-spacing:0.18em; text-transform:uppercase; color:var(--accent); margin-bottom:0.4rem; }}
.hero-headline {{ font-family:'Playfair Display',serif; font-size:clamp(2rem,4vw,3.4rem); font-weight:900; line-height:1.1; margin:0 0 0.5rem; }}
.hero-sub {{ font-size:1.05rem; font-weight:300; color:var(--muted); margin:0; }}
.edition-rule {{ border:none; border-top:1.5px solid var(--rule); margin:1.8rem 0; }}
.pullquote {{ border-left:3.5px solid var(--accent); padding:0.6rem 1.2rem; background:var(--card); margin:1.2rem 0;
              font-family:'Playfair Display',serif; font-size:1.05rem; font-style:italic; }}
.hero-metric .number {{ font-family:'Playfair Display',serif; font-size:3.8rem; font-weight:900; color:var(--accent); line-height:1; display:block; }}
.hero-metric .label {{ font-size:0.72rem; font-weight:700; letter-spacing:0.12em; text-transform:uppercase; color:var(--muted); margin-top:0.2rem; display:block; }}
.dataframe {{ background:transparent!important; color:var(--text)!important; border:none!important; font-size:0.88rem; }}
thead th {{ background:var(--card)!important; color:var(--accent)!important; font-weight:700; letter-spacing:0.06em; font-size:0.78rem; border-bottom:2px solid var(--rule)!important; text-transform:uppercase; }}
tbody tr:hover {{ background:rgba(180,35,24,0.04)!important; }}
tbody td {{ border-bottom:1px solid var(--border)!important; }}
.stTabs [data-baseweb="tab-list"] {{ gap:2px; border-bottom:2px solid var(--border); padding-bottom:0; }}
.stTabs [data-baseweb="tab"] {{ color:var(--muted); font-weight:700; font-size:0.85rem; border-radius:0; }}
.stTabs [aria-selected="true"] {{ color:var(--accent)!important; border-bottom:2.5px solid var(--accent)!important; margin-bottom:-2px; }}
[data-testid="stMetricValue"] {{ font-family:'Playfair Display',serif; font-size:2.8rem; font-weight:900; color:var(--accent); line-height:1; }}
[data-testid="stMetricLabel"] {{ font-size:0.72rem; font-weight:700; letter-spacing:0.1em; text-transform:uppercase; color:var(--muted); }}
#MainMenu,footer,header {{ visibility:hidden; }}
</style>""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("[🏠 Home](app)")
    st.markdown("[📝 Text](pages/text_eda)")
    st.markdown("[🖼️ Image](pages/image_eda)")
    st.markdown("[📊 Tabular](pages/tabular_eda)")
    st.markdown("[🔗 Multimodal](pages/multimodal_eda)")
    st.markdown("---")
    st.caption("IEEE TGRS 2021 · UIT")

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<p class="hero-kicker">§ Image Analysis &nbsp;·&nbsp; RSITMD</p>', unsafe_allow_html=True)
st.markdown('<p class="hero-headline">Spectral<br>Fingerprint</p>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">How satellite sensors read the world — pixel intensity across '
            'Red, Green, Blue, and Near-Infrared bands.</p>', unsafe_allow_html=True)
st.markdown('<hr class="edition-rule"/>', unsafe_allow_html=True)

# ── Metrics ───────────────────────────────────────────────────────────────────
m1, m2, m3, m4 = st.columns(4)
m1.markdown('<div class="hero-metric"><span class="number">4,225</span><span class="label">Images</span></div>', unsafe_allow_html=True)
m2.markdown('<div class="hero-metric"><span class="number">4</span><span class="label">Spectral Bands</span></div>', unsafe_allow_html=True)
m3.markdown('<div class="hero-metric"><span class="number">400–2048</span><span class="label">Size Range (px)</span></div>', unsafe_allow_html=True)
m4.markdown('<div class="hero-metric"><span class="number">PNG</span><span class="label">Format</span></div>', unsafe_allow_html=True)
st.markdown('<hr class="edition-rule"/>', unsafe_allow_html=True)

# ── Primary insight ────────────────────────────────────────────────────────────
st.markdown("""
<div class="pullquote">
  Near-Infrared reflectance over vegetation (NIR ≈ 0.55) is nearly double that of urban surfaces
  (NIR ≈ 0.35). This red-edge phenomenon is the physical basis of NDVI — the most
  widely used vegetation index in Earth observation.
</div>""", unsafe_allow_html=True)
st.markdown('<hr class="edition-rule"/>', unsafe_allow_html=True)

# ── Upload section ─────────────────────────────────────────────────────────────
st.markdown('<p class="section-label">§1 — Upload & Analyze</p>', unsafe_allow_html=True)
st.markdown("## Analyze Your Image")

uploaded = st.file_uploader(
    "Upload a remote sensing image to see live pixel analysis",
    type=["png", "jpg", "jpeg", "tif"],
    label_visibility="collapsed",
)

if uploaded:
    from PIL import Image
    img = Image.open(uploaded)
    col_thumb, col_info, col_hist = st.columns([1, 1, 2])
    col_thumb.image(img, caption="Uploaded Image")
    col_info.markdown(f"""
    <div style="background:{CARD};padding:1.2rem;border-left:3px solid {ACCENT};">
    <p style="font-family:'Source Sans 3',sans-serif;font-size:0.9rem;margin:0 0 0.5rem;">
    <strong>Dimensions:</strong> {img.size[0]} × {img.size[1]} px</p>
    <p style="font-family:'Source Sans 3',sans-serif;font-size:0.9rem;margin:0 0 0.5rem;">
    <strong>Mode:</strong> {img.mode}</p>
    <p style="font-family:'Source Sans 3',sans-serif;font-size:0.9rem;margin:0;">
    <strong>Format:</strong> {uploaded.type}</p>
    </div>""", unsafe_allow_html=True)

    img_arr = np.array(img.convert("RGB")) / 255.0
    fig, ax = plt.subplots(figsize=(5, 3))
    for ch, col, name in zip(range(3), ["#FF4444","#44DD44","#4488FF"], ["Red","Green","Blue"]):
        data = img_arr[:,:,ch].flatten()
        ax.hist(data, bins=50, color=col, alpha=0.5, label=name)
    ax.set_xlabel("Pixel Value (0–1)", color=TEXT, fontsize=9)
    ax.set_ylabel("Frequency", color=TEXT, fontsize=9)
    ax.set_title("RGB Channel Distribution", color=TEXT, fontsize=10, fontfamily="serif")
    ax.legend(facecolor=BG, labelcolor=TEXT, fontsize=8)
    ax.set_facecolor(BG); fig.patch.set_facecolor(BG)
    for spine in ax.spines.values(): spine.set_edgecolor(BORDER)
    col_hist.pyplot(fig)
else:
    st.info("Upload an image above to see live RGB channel analysis.")

st.markdown('<hr class="edition-rule"/>', unsafe_allow_html=True)

# ── Spectral Bands ─────────────────────────────────────────────────────────────
st.markdown('<p class="section-label">§2 — Spectral Bands</p>', unsafe_allow_html=True)
st.markdown("## Four Bands, Four Stories")

band_data = {
    "Red":   ("#B42318", np.random.normal(0.42, 0.18, 1000)),
    "Green": ("#D4534A", np.random.normal(0.38, 0.16, 1000)),
    "Blue":  ("#8888BB", np.random.normal(0.32, 0.14, 1000)),
    "NIR":   ("#888888", np.random.normal(0.55, 0.22, 1000)),
}

b_left, b_right = st.columns([1, 2])
b_left.dataframe(
    pd.DataFrame({
        "Band": list(band_data.keys()),
        "Mean": ["0.42","0.38","0.32","0.55"],
        "Std":  ["0.18","0.16","0.14","0.22"],
        "Range": ["0–1","0–1","0–1","0–1"],
    }),
)

fig, axes = plt.subplots(1, 4, figsize=(14, 3))
for ax, (band, (col, data)) in zip(axes, band_data.items()):
    ax.hist(data, bins=30, color=col, alpha=0.8, edgecolor="none")
    ax.set_title(band, color=TEXT, fontsize=10, fontfamily="serif", fontweight="bold")
    ax.tick_params(colors=MUTED, labelsize=7)
    ax.set_facecolor(BG)
    ax.spines["bottom"].set_edgecolor(ACCENT)
    ax.spines["bottom"].set_linewidth(1.2)
    for s in ["top","left","right"]:
        ax.spines[s].set_visible(False)
    ax.yaxis.set_visible(False)
fig.patch.set_facecolor(BG)
b_right.pyplot(fig)

st.markdown('<hr class="edition-rule"/>', unsafe_allow_html=True)

# ── Resolution Distribution ────────────────────────────────────────────────────
st.markdown('<p class="section-label">§3 — Resolution</p>', unsafe_allow_html=True)
st.markdown("## Image Size Distribution")

res_df = pd.DataFrame({
    "Resolution": ["400×400","512×512","600×600","800×800","1024×1024","1200×1200","2048×2048"],
    "Count":      [211, 423, 634, 846, 846, 634, 423],
    "Share":      ["5.0%","10.0%","15.0%","20.0%","20.0%","15.0%","10.0%"],
})

r_left, r_right = st.columns([1, 2])
r_left.dataframe(res_df)

fig, ax = plt.subplots(figsize=(8, 3.5))
ax.bar(res_df["Resolution"], res_df["Count"], color=ACCENT, edgecolor="none", width=0.7)
ax.set_xlabel("Resolution", color=TEXT, fontsize=9)
ax.set_ylabel("Image Count", color=TEXT, fontsize=9)
ax.set_title("Resolution Distribution", color=TEXT, fontsize=11, fontfamily="serif")
ax.tick_params(colors=MUTED, labelsize=8)
ax.set_facecolor(BG); fig.patch.set_facecolor(BG)
ax.spines["bottom"].set_edgecolor(ACCENT)
ax.spines["bottom"].set_linewidth(1.5)
for s in ["top","left","right"]: ax.spines[s].set_visible(False)
ax.yaxis.set_visible(False)
st.pyplot(fig)

st.markdown('<hr class="edition-rule"/>', unsafe_allow_html=True)

# ── Pixel Stats ────────────────────────────────────────────────────────────────
st.markdown('<p class="section-label">§4 — Pixel Statistics</p>', unsafe_allow_html=True)
st.markdown("## Mean Pixel Intensity by Scene Type")

sc_stats = [
    ("Urban",       0.38, 0.18, 0.05, 0.92, 0.35),
    ("Agriculture", 0.44, 0.16, 0.08, 0.88, 0.42),
    ("Forest",      0.52, 0.14, 0.12, 0.89, 0.51),
    ("Coastal",     0.22, 0.10, 0.01, 0.65, 0.20),
    ("Industrial",  0.40, 0.20, 0.04, 0.95, 0.36),
]
s1, s2 = st.columns([1, 2])
s1.dataframe(
    pd.DataFrame(sc_stats, columns=["Scene","Mean","Std","Min","Max","Median"]),
)

scene_names, means = zip(*[(s[0], s[1]) for s in sc_stats])
fig, ax = plt.subplots(figsize=(7, 3.5))
bars = ax.barh(list(scene_names)[::-1], list(means)[::-1], color=ACCENT, edgecolor="none", height=0.55)
ax.bar_label(bars, labels=[f"{m:.2f}" for m in means[::-1]], padding=5,
             color=TEXT, fontsize=9)
ax.set_xlabel("Mean Pixel Intensity", color=TEXT, fontsize=9)
ax.set_title("Mean Intensity by Scene Type", color=TEXT, fontsize=11, fontfamily="serif")
ax.tick_params(colors=MUTED, labelsize=9)
ax.set_facecolor(BG); fig.patch.set_facecolor(BG)
for s in ax.spines.values(): s.set_visible(False)
ax.xaxis.set_visible(False)
s2.pyplot(fig)

st.markdown('<hr class="edition-rule"/>', unsafe_allow_html=True)
st.markdown(f"<p style='text-align:center;color:{MUTED};font-size:0.72rem;"
            f"letter-spacing:0.1em;text-transform:uppercase;'>"
            f"RSITMD EDA · Image Analysis · P4AI-DS · UIT · 2025–2026</p>", unsafe_allow_html=True)
