import json
import os
import re
import shutil
import sys
import types
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from datasets import ClassLabel, Dataset, DatasetDict, Features, Image as DatasetsImage, Sequence, Value
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedShuffleSplit


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EDA_DIR = PROJECT_ROOT / "streamlit_app" / "pages"
if str(EDA_DIR) not in sys.path:
    sys.path.append(str(EDA_DIR))

class _DummyState(types.SimpleNamespace):
    def __getitem__(self, key):
        return getattr(self, key)
    def __setitem__(self, key, value):
        setattr(self, key, value)
    def __contains__(self, key):
        return hasattr(self, key)

st = types.ModuleType("streamlit")
st.cache_data = lambda *args, **kwargs: (lambda f: f)
st.cache_resource = lambda *args, **kwargs: (lambda f: f)
st.session_state = _DummyState(D=None, step=0)
st.markdown = lambda *args, **kwargs: None
st.set_page_config = lambda *args, **kwargs: None
st.caption = lambda *args, **kwargs: None
st.error = lambda *args, **kwargs: None
st.stop = lambda *args, **kwargs: (_ for _ in ()).throw(SystemExit)
class _DummyCtx:
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc, tb):
        return False

class _DummyCallable:
    def __call__(self, *args, **kwargs):
        return _DummyCtx()
    def __getattr__(self, name):
        return self

class _DummySlider:
    def __call__(self, label, *args, **kwargs):
        if len(args) >= 2:
            return args[1]
        if len(args) >= 1:
            return args[0]
        return kwargs.get("value", 0)

class _DummyColumns:
    def __call__(self, n):
        return [_DummyCtx() for _ in range(n)]

class _DummyStreamlit(types.ModuleType):
    def __getattr__(self, name):
        if name in {"session_state"}:
            return self.__dict__.setdefault("session_state", _DummyState(D=None, step=0))
        if name in {"cache_data", "cache_resource"}:
            return lambda *args, **kwargs: (lambda f: f)
        if name in {"stop"}:
            return lambda *args, **kwargs: (_ for _ in ()).throw(SystemExit)
        return _DummyCallable()

st = _DummyStreamlit("streamlit")
st.session_state = _DummyState(D=None, step=0)
st.columns = _DummyColumns()
st.slider = _DummySlider()
sys.modules["streamlit"] = st

import multimodal_eda

WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")
COLOR_WORDS = {
    "red", "green", "blue", "yellow", "white", "black", "brown", "gray", "grey",
    "orange", "purple", "pink", "silver", "gold", "beige", "cyan", "magenta",
}
SPATIAL_WORDS = {
    "near", "surrounded", "surrounding", "around", "beside", "besides", "left", "right",
    "above", "below", "front", "behind", "across", "between", "center", "middle", "along",
    "adjacent", "within", "outside", "inside",
}
OBJECT_WORDS = {
    "building", "buildings", "road", "roads", "tree", "trees", "water", "field", "fields",
    "house", "houses", "bridge", "bridges", "parking", "airport", "harbor", "port",
    "stadium", "playground", "farm", "farmland", "river", "lake", "beach", "forest", "vehicle",
    "vehicles", "ship", "ships", "plane", "planes", "car", "cars", "boat", "boats",
}


def normalize_caption(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def caption_tokens(text: str):
    return WORD_RE.findall(normalize_caption(text))


def is_blank_image(image: Image.Image) -> bool:
    arr = np.asarray(image)
    return arr.size == 0 or np.sum(arr) == 0


def extract_label(filename: str) -> str:
    parts = filename.replace(".tif", "").rsplit("_", 1)
    return parts[0] if len(parts) == 2 else "unknown"


def summarize_captions(sentences):
    cleaned = [normalize_caption(s) for s in sentences if normalize_caption(s)]
    lengths = [len(caption_tokens(s)) for s in cleaned]
    return {
        "sentences": cleaned,
        "caption_count": len(cleaned),
        "avg_caption_len": float(np.mean(lengths)) if lengths else 0.0,
        "caption_len_std": float(np.std(lengths)) if lengths else 0.0,
        "has_color": any(any(tok in COLOR_WORDS for tok in caption_tokens(s)) for s in cleaned),
        "has_spatial": any(any(tok in SPATIAL_WORDS for tok in caption_tokens(s)) for s in cleaned),
        "has_object": any(any(tok in OBJECT_WORDS for tok in caption_tokens(s)) for s in cleaned),
    }


def compute_image_statistics(image: Image.Image):
    arr = np.asarray(image).astype(np.float32) / 255.0
    brightness = float(arr.mean())
    texture = float(arr.std())
    channel_means = arr.mean(axis=(0, 1))
    dominant_channel = int(np.argmax(channel_means))
    sorted_channels = np.sort(channel_means)[::-1]
    channel_gap = float(sorted_channels[0] - sorted_channels[-1])
    return {
        "brightness": brightness,
        "texture": texture,
        "dominant_channel": dominant_channel,
        "channel_gap": channel_gap,
    }


def loo_caption_centroid_similarities(caps):
    n = len(caps)
    if n < 2:
        return [np.nan] * n
    out = []
    for i in range(n):
        others = [caps[j] for j in range(n) if j != i]
        vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=1, lowercase=True)
        mat_o = vec.fit_transform(others)
        centroid = np.asarray(mat_o.mean(axis=0))
        cap_vec = vec.transform([caps[i]])
        out.append(float(cosine_similarity(cap_vec, centroid)[0, 0]))
    return out


def compute_layer_scores(caps, label, pixel_stats=None):
    vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=1, lowercase=True)
    mat = vec.fit_transform(caps)
    sim = cosine_similarity(mat)
    upper = np.triu_indices_from(sim, k=1)
    pair_vals = sim[upper]
    mean_pair = float(np.mean(pair_vals)) if len(pair_vals) else 0.0
    center = np.asarray(mat.mean(axis=0))
    center_sims = cosine_similarity(mat, center).reshape(-1)
    center_min = float(np.min(center_sims)) if len(center_sims) else 0.0
    q1 = float(np.percentile(center_sims, 25)) if len(center_sims) else 0.0
    q3 = float(np.percentile(center_sims, 75)) if len(center_sims) else 0.0
    iqr = q3 - q1
    outlier_count = int(np.sum(center_sims < (q1 - 1.5 * iqr))) if iqr > 0 else 0

    sem_noise = float(np.clip(1.0 - mean_pair, 0.0, 1.0))
    center_penalty = float(np.clip((0.55 - center_min) / 0.55, 0.0, 1.0))
    layer_a = 0.7 * sem_noise + 0.3 * center_penalty

    color_lexicon = {
        'blue': ['blue', 'water', 'ocean', 'sea', 'lake', 'river', 'coast'],
        'green': ['green', 'forest', 'tree', 'trees', 'woods', 'grass', 'park', 'vegetation'],
        'red': ['red', 'brick', 'roof', 'clay'],
        'brown': ['brown', 'soil', 'earth', 'sand', 'desert'],
        'white': ['white', 'snow', 'cloud', 'cloudy'],
        'gray': ['gray', 'grey', 'concrete', 'asphalt'],
    }
    object_lexicon = {
        'water_obj': ['water', 'ocean', 'sea', 'lake', 'river', 'harbor', 'ship', 'boat'],
        'vegetation_obj': ['forest', 'tree', 'trees', 'grass', 'field', 'farm', 'vegetation'],
        'urban_obj': ['building', 'buildings', 'road', 'roads', 'bridge', 'airport', 'runway', 'city'],
    }
    dom_to_supported_colors = {
        'G>R>B': {'green', 'white', 'gray'},
        'B>R>G': {'blue', 'white', 'gray'},
        'R>G>B': {'red', 'brown', 'white', 'gray'},
        'R≈G>B': {'gray', 'white', 'brown', 'green', 'blue'},
    }

    dom_ch = 'R≈G>B'
    supported_colors = dom_to_supported_colors.get(dom_ch, {'white', 'gray'})
    cat_lower = label.lower()
    supported_obj_groups = set()
    if any(k in cat_lower for k in ['river', 'pond', 'port', 'boat', 'harbor', 'coast', 'beach', 'sea', 'lake']):
        supported_obj_groups.add('water_obj')
    if any(k in cat_lower for k in ['forest', 'meadow', 'park', 'farmland', 'baseballfield', 'playground', 'grass', 'bareland', 'desert']):
        supported_obj_groups.add('vegetation_obj')
    if any(k in cat_lower for k in ['airport', 'plane', 'runway', 'road', 'bridge', 'residential', 'industrial', 'church', 'school', 'stadium', 'square', 'center', 'railway', 'parking', 'building', 'viaduct']):
        supported_obj_groups.add('urban_obj')
    if not supported_obj_groups:
        supported_obj_groups = {'water_obj', 'vegetation_obj', 'urban_obj'}

    img_color_claims = img_color_mismatch = 0
    img_object_claims = img_object_mismatch = 0
    for cap in caps:
        toks = set(caption_tokens(cap))
        claimed_colors = set()
        for cname, kws in color_lexicon.items():
            hit_count = sum(1 for w in kws if w in toks)
            direct = cname in toks or (cname == 'gray' and 'grey' in toks)
            if hit_count >= 2 or direct:
                claimed_colors.add(cname)
        vivid_claims = {x for x in claimed_colors if x not in {'white', 'gray'}}
        if vivid_claims:
            img_color_claims += 1
            if vivid_claims.isdisjoint(supported_colors):
                img_color_mismatch += 1

        claimed_groups = []
        for gname, kws in object_lexicon.items():
            hit_count = sum(1 for w in kws if w in toks)
            if hit_count >= 2:
                claimed_groups.append(gname)
        if claimed_groups:
            img_object_claims += 1
            if all(g not in supported_obj_groups for g in claimed_groups):
                img_object_mismatch += 1

    img_color_rate = (img_color_mismatch / img_color_claims) if img_color_claims else 0.0
    img_object_rate = (img_object_mismatch / img_object_claims) if img_object_claims else 0.0
    prior_color_rate = 0.0
    prior_object_rate = 0.0
    color_rate = 0.85 * img_color_rate + 0.15 * prior_color_rate
    object_rate = 0.85 * img_object_rate + 0.15 * prior_object_rate
    layer_b = 0.65 * color_rate + 0.35 * object_rate

    token_counts = [len(caption_tokens(cap)) for cap in caps]
    avg_tokens = float(np.mean(token_counts)) if token_counts else 0.0
    low_info_penalty = float(np.clip((8.0 - avg_tokens) / 8.0, 0.0, 1.0))
    uncertainty_score = 0.7 * low_info_penalty + 0.3 * min(outlier_count / 3.0, 1.0)

    if pixel_stats is not None:
        _ = pixel_stats

    noise_score = float(np.clip(0.58 * layer_a + 0.30 * layer_b + 0.12 * uncertainty_score, 0.0, 1.0))
    if noise_score >= 0.62 and layer_a >= 0.45 and (layer_b >= 0.35 or outlier_count >= 2):
        confidence = 'HIGH'
    elif noise_score >= 0.50 and (layer_a >= 0.36 or layer_b >= 0.25 or outlier_count >= 1):
        confidence = 'MEDIUM'
    else:
        confidence = 'LOW'

    return {
        'layer_a': float(layer_a),
        'layer_b': float(layer_b),
        'layer_c': float(uncertainty_score),
        'noise_score': noise_score,
        'confidence': confidence,
    }


def create_rsitmd_dataset(json_path: str, img_dir: str, out_dir: str, target_size=(256, 256), val_size=0.15, seed=42):
    print(f"Loading JSON from: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    images_info = data.get('images', [])
    print(f"Total images found in JSON: {len(images_info)}")

    train_imgs = [img for img in images_info if img.get('split', 'train') == 'train']
    test_imgs = [img for img in images_info if img.get('split', 'train') == 'test']
    train_dom_map = {r['category']: r['dom'] for _, r in multimodal_eda.image_pixel_stats(train_imgs).iterrows()}
    test_dom_map = {r['category']: r['dom'] for _, r in multimodal_eda.image_pixel_stats(test_imgs).iterrows()}
    train_noise = multimodal_eda.image_noise_probe(train_imgs, multimodal_eda.contradiction_map(train_imgs, train_dom_map), tuple(sorted(train_dom_map.items())))
    test_noise = multimodal_eda.image_noise_probe(test_imgs, multimodal_eda.contradiction_map(test_imgs, test_dom_map), tuple(sorted(test_dom_map.items())))

    def _high_noise_mask(df):
        if df is None or df.empty:
            return np.array([], dtype=bool)
        if 'confidence' in df.columns:
            return df['confidence'].eq('HIGH')
        return (
            (df.get('noise_score', 0) >= 0.62)
            & (df.get('layer_a_score', 0) >= 0.45)
            & ((df.get('layer_b_score', 0) >= 0.35) | (df.get('outlier_count', 0) >= 2))
        )

    high_noise_files = set(train_noise.loc[_high_noise_mask(train_noise), 'filename'].tolist()) | set(test_noise.loc[_high_noise_mask(test_noise), 'filename'].tolist())
    print(f"Precomputed HIGH-noise candidates: {len(high_noise_files)}")

    parsed_data = {'train': [], 'test': []}
    dropped = Counter()
    noise_rows = []

    for img in images_info:
        filename = img['filename']
        img_path = os.path.join(img_dir, filename)
        if not os.path.exists(img_path):
            dropped['missing'] += 1
            continue

        label = extract_label(filename)
        raw_sentences = [s.get('raw', '') for s in img.get('sentences', [])]
        sentences_info = summarize_captions(raw_sentences)
        if sentences_info['caption_count'] == 0:
            dropped['empty_captions'] += 1
            continue

        split = img.get('split', 'train')

        try:
            with Image.open(img_path) as raw_img:
                raw_img = raw_img.convert('RGB')
                if is_blank_image(raw_img):
                    dropped['blank_image'] += 1
                    continue
                image_stats = compute_image_statistics(raw_img)
        except Exception:
            dropped['corrupt'] += 1
            continue

        caps = sentences_info['sentences'][:5]
        layers = compute_layer_scores(caps, label, image_stats)
        noise_score = layers['noise_score']

        if filename in high_noise_files:
            dropped['noise_high'] += 1
            noise_rows.append({
                'filename': filename,
                'split': split,
                'label': label,
                'noise_score': noise_score,
                'layer_a': layers['layer_a'],
                'layer_b': layers['layer_b'],
                'layer_c': layers['layer_c'],
                'confidence': 'HIGH',
            })
            continue

        record = {
            'image': img_path,
            'label': label,
            'filename': filename,
            **sentences_info,
            **image_stats,
            **layers,
        }
        parsed_data.setdefault(split, []).append(record)

    print(f"Dropped records summary: {dict(dropped)}")
    print(f"Removed HIGH-noise samples: {dropped.get('noise_high', 0)}")
    print(f"Parsed initially: Train={len(parsed_data['train'])}, Test={len(parsed_data['test'])}")

    all_labels = sorted(list(set(r['label'] for r in parsed_data['train'] + parsed_data['test'])))
    print(f"Found {len(all_labels)} unique classes.")

    train_records = parsed_data['train']
    train_labels = [r['label'] for r in train_records]

    if val_size > 0 and len(set(train_labels)) > 1:
        print("Performing Stratified Validation Split...")
        sss = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
        train_idx, val_idx = next(sss.split(np.zeros(len(train_labels)), train_labels))
        final_train = [train_records[i] for i in train_idx]
        final_val = [train_records[i] for i in val_idx]
    else:
        final_train = train_records
        final_val = []

    final_test = parsed_data['test']
    print(f"Final split sizes: Train={len(final_train)}, Val={len(final_val)}, Test={len(final_test)}")

    def gen(records):
        for r in records:
            try:
                with Image.open(r['image']) as img:
                    img = img.convert('RGB')
                    if is_blank_image(img):
                        continue
                    img = img.resize(target_size, Image.Resampling.LANCZOS)
                    yield {
                        'image': img,
                        'label': r['label'],
                        'filename': r['filename'],
                        'sentences': r['sentences'],
                    }
            except Exception as e:
                print(f"Error reading/resizing {r['image']}: {e}")

    features = Features({
        'image': DatasetsImage(),
        'label': ClassLabel(names=all_labels),
        'filename': Value('string'),
        'sentences': Sequence(Value('string')),
    })

    print("Creating HuggingFace Datasets (resizing images and preserving captions)...")
    ds_train = Dataset.from_generator(lambda: gen(final_train), features=features)
    ds_val = Dataset.from_generator(lambda: gen(final_val), features=features)
    ds_test = Dataset.from_generator(lambda: gen(final_test), features=features)

    dataset_dict = DatasetDict({'train': ds_train, 'validation': ds_val, 'test': ds_test})
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    print(f"Saving to disk at {out_dir}...")
    dataset_dict.save_to_disk(out_dir)
    print("Done!")
    return dataset_dict


if __name__ == '__main__':
    base_raw_dir = '/tmp/RSITMD_unzipped'
    json_path = os.path.join(base_raw_dir, 'dataset_RSITMD.json')
    img_dir = os.path.join(base_raw_dir, 'images')
    out_dir = str(PROJECT_ROOT / 'assign2-ml' / 'data' / 'processed_rsitmd_256_clean')
    os.makedirs(os.path.dirname(out_dir), exist_ok=True)
    create_rsitmd_dataset(json_path=json_path, img_dir=img_dir, out_dir=out_dir, target_size=(256, 256), val_size=0.15, seed=42)
