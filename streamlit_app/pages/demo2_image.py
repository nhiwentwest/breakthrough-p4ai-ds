import streamlit as st
from PIL import Image
import numpy as np

st.set_page_config(page_title="Demo 2 · Image Classification", page_icon="🖼️", layout="wide")

BG = "#F7F3EB"
CARD = "#EFE8DC"
TEXT = "#111111"
ACC = "#B42318"
MUT = "#6B6560"
BOR = "#D4C9B8"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=Source+Sans+3:wght@300;400;600;700&display=swap');
body,.stApp {{ background:{BG}; color:{TEXT}; font-family:'Source Sans 3',sans-serif; }}
#MainMenu,footer,header {{ visibility:hidden; }}
.block-container {{ padding-top:1.2rem; }}
.hero {{ font-family:'Playfair Display',serif; font-size:2.2rem; font-weight:900; margin:0; }}
.sub {{ color:{MUT}; margin-top:.25rem; margin-bottom:1rem; }}
.bento {{ background:{CARD}; border:1px solid {BOR}; border-radius:14px; padding:1rem; }}
.section {{ font-size:.68rem; letter-spacing:.12em; text-transform:uppercase; color:{ACC}; font-weight:700; margin-bottom:.6rem; }}
.stButton > button {{ border:1.5px solid {TEXT}; background:transparent; color:{TEXT}; font-weight:700; letter-spacing:.08em; border-radius:4px; }}
.stButton > button:hover {{ background:{ACC}; color:white; border-color:{ACC}; }}
</style>
""", unsafe_allow_html=True)

st.markdown("<p class='hero'>Demo 2 · Image Classification</p>", unsafe_allow_html=True)
st.markdown("<p class='sub'>Upload an image or choose a sample, then predict class.</p>", unsafe_allow_html=True)

left, right = st.columns([1.25, 1])

sample_paths = {
    "Sample 1": "https://images.unsplash.com/photo-1506744038136-46273834b3fb?w=640",
    "Sample 2": "https://images.unsplash.com/photo-1470770841072-f978cf4d019e?w=640",
    "Sample 3": "https://images.unsplash.com/photo-1469474968028-56623f02e42e?w=640",
}

with left:
    st.markdown("<div class='bento'>", unsafe_allow_html=True)
    st.markdown("<div class='section'>Model Selection</div>", unsafe_allow_html=True)
    model = st.selectbox("Choose classification model", [
        "ResNet50",
        "EfficientNet-B0",
        "ViT-Base",
        "MobileNetV3",
    ])

    st.markdown("<div class='section'>Image Input</div>", unsafe_allow_html=True)
    mode = st.radio("Input mode", ["Upload image", "Use sample image"], horizontal=True)

    image = None
    if mode == "Upload image":
        up = st.file_uploader("Upload image", type=["png", "jpg", "jpeg", "webp"])
        if up:
            image = Image.open(up).convert("RGB")
    else:
        sample = st.selectbox("Choose sample", list(sample_paths.keys()))
        image = Image.open(st.experimental_get_query_params and None) if False else None
        st.image(sample_paths[sample], caption=sample, use_container_width=True)

    pred_btn = st.button("Predict", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown("<div class='bento'>", unsafe_allow_html=True)
    st.markdown("<div class='section'>Prediction Output</div>", unsafe_allow_html=True)

    if pred_btn:
        classes = ["urban", "forest", "water", "agriculture", "residential"]
        probs = np.array([0.18, 0.21, 0.16, 0.24, 0.21], dtype=float)
        shift = {"ResNet50":0, "EfficientNet-B0":1, "ViT-Base":2, "MobileNetV3":3}[model]
        probs = np.roll(probs, shift)
        top_idx = int(np.argmax(probs))
        st.metric("Predicted class", classes[top_idx])
        st.caption(f"Model used: {model}")
        st.progress(float(probs[top_idx]))
        st.write(f"Confidence: {probs[top_idx]:.2%}")
    else:
        st.info("Upload/select image and click Predict.")

    st.markdown("</div>", unsafe_allow_html=True)
