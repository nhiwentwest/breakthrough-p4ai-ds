#!/usr/bin/env python3
"""
EDA for RSITMD Dataset - Multimodal (Image + Text)
==================================================
Exploratory Data Analysis following the format from AI Learning Hub
Reference: Z. Yuan et al., IEEE TGRS 2021

Analysis includes:
1. Text Analysis (caption-level statistics)
2. Image Analysis (category distribution)
3. Multimodal Analysis (image-text relationships)
"""

import json
import re
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

try:
    from nltk.util import ngrams
except ImportError as e:
    sys.stderr.write("[LỖI] Thiếu nltk. Cài: pip install nltk\n")
    raise SystemExit(1) from e

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError as e:
    sys.stderr.write(
        "[LỖI] Thiếu scikit-learn — cần TF-IDF và cosine_similarity.\n"
        "      Cài: pip install scikit-learn\n"
    )
    raise SystemExit(1) from e

try:
    from PIL import Image
except ImportError as e:
    sys.stderr.write("[LỖI] Thiếu Pillow. Cài: pip install Pillow\n")
    raise SystemExit(1) from e

# Visualization settings
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'seaborn-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 11

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_CANDIDATES = [
    PROJECT_ROOT / "RSITMD" / "dataset_RSITMD.json",
    Path("/Users/nhi/Documents/school/252/p4/btl/RSITMD/dataset_RSITMD.json"),
]
DATA_FILE = next((p for p in DATA_CANDIDATES if p.exists()), DATA_CANDIDATES[0])
IMG_DIR = DATA_FILE.parent / "images"
OUTPUT_DIR = PROJECT_ROOT / "report" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Stopwords
STOP_WORDS = set([
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're",
    "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he',
    'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's",
    'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
    'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are',
    'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do',
    'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because',
    'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against',
    'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
    'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
    'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all',
    'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
    'only', 'own', 'same', 'so', 'than', 'too', 'very', 'can', 'will', 'just',
    'should', 'now', 'one', 'also'
])


def load_data():
    """Load dataset from JSON file."""
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    images = data['images']
    train_imgs = [img for img in images if img['split'] == 'train']
    test_imgs = [img for img in images if img['split'] == 'test']

    return images, train_imgs, test_imgs


def run_data_audit(images, train_imgs, test_imgs):
    """Basic data integrity audit for EDA step 0."""
    print(f"\n{'='*60}")
    print("DATA AUDIT — INTEGRITY & SPLIT CHECK")
    print(f"{'='*60}")

    total_caps = sum(len(img.get('sentences', [])) for img in images)
    unique_files = len({img.get('filename') for img in images})
    duplicate_filenames = len(images) - unique_files

    missing_img_files = 0
    bad_caption_rows = 0
    for img in images:
        fname = img.get('filename', '')
        if not (IMG_DIR / fname).exists():
            missing_img_files += 1
        sents = img.get('sentences', [])
        if len(sents) == 0:
            bad_caption_rows += 1

    print(f"  Total images: {len(images)}")
    print(f"  Total captions: {total_caps}")
    print(f"  Train/Test images: {len(train_imgs)}/{len(test_imgs)}")
    print(f"  Duplicate filenames: {duplicate_filenames}")
    print(f"  Missing image files: {missing_img_files}")
    print(f"  Rows with empty captions: {bad_caption_rows}")


def clean_output_dir():
    """Remove stale figures that are no longer part of the core EDA set."""
    obsolete = [
        'txt_01_category_distribution_train.png',
        'txt_01_category_distribution_test.png',
        'txt_04_vocabulary_richness_train.png',
        'txt_04_vocabulary_richness_test.png',
        'txt_07_bigrams_train.png',
        'txt_07_bigrams_test.png',
        'ia_02_image_visual_analysis.png',
        'ia_02b_texture_top10.png',
        'ia_03_rgb_channel_dominance.png',
    ]
    removed = 0
    for name in obsolete:
        p = OUTPUT_DIR / name
        if p.exists():
            p.unlink()
            removed += 1
    if removed:
        print(f"  Cleaned {removed} stale figure(s) from output directory")


def parse_filename(filename):
    """Extract category from filename."""
    parts = filename.replace('.tif', '').rsplit('_', 1)
    if len(parts) == 2:
        return parts[0]
    return 'unknown'


def tokenize(text, remove_stopwords=False):
    """Tokenize text into words."""
    text = text.lower()
    words = re.findall(r'\b[a-z]+\b', text)
    if remove_stopwords:
        words = [w for w in words if w not in STOP_WORDS]
    return words


# ─────────────────────────────────────────────
# 1. TEXT ANALYSIS
# ─────────────────────────────────────────────
def analyze_text(imgs, split='train'):
    """Analyze text/caption statistics for a split."""
    print(f"\n{'='*60}")
    print(f"TEXT ANALYSIS — {split.upper()} SET")
    print(f"{'='*60}")

    all_captions = []
    for img in imgs:
        for sent in img['sentences']:
            all_captions.append(sent['raw'])

    print(f"  Total Captions: {len(all_captions)}")
    print(f"  Total Images: {len(imgs)}")
    print(f"  Captions/Image: {len(all_captions)/len(imgs):.1f}")

    # Tokenize
    raw_tokens = [tokenize(cap) for cap in all_captions]
    raw_words = [w for tokens in raw_tokens for w in tokens]
    clean_tokens = [tokenize(cap, remove_stopwords=True) for cap in all_captions]
    clean_words = [w for tokens in clean_tokens for w in tokens]

    # Word count stats
    word_counts = [len(tokens) for tokens in raw_tokens]
    char_counts = [len(cap) for cap in all_captions]

    print(f"\n  Total Words: {len(raw_words):,}")
    print(f"  Vocabulary (raw): {len(set(raw_words)):,}")
    print(f"  Vocabulary (clean): {len(set(clean_words)):,}")
    print(f"  Avg Words/Caption: {np.mean(word_counts):.1f}")
    print(f"  Median: {np.median(word_counts):.1f}, Std: {np.std(word_counts):.1f}")
    print(f"  Min: {min(word_counts)}, Max: {max(word_counts)}")

    # Word frequency
    word_freq = Counter(clean_words)
    top_words = word_freq.most_common(50)
    print(f"\n  Top 15 words (no stopwords):")
    for w, c in top_words[:15]:
        print(f"    {w}: {c}")

    # Top words WITH stopwords
    word_freq_raw = Counter(raw_words)
    print(f"\n  Top 15 words (with stopwords):")
    for w, c in word_freq_raw.most_common(15):
        print(f"    {w}: {c}")

    # Category keywords
    cat_keyword = {}
    for img in imgs:
        cat = parse_filename(img['filename'])
        cap_text = ' '.join([s['raw'] for s in img['sentences']])
        words = tokenize(cap_text, remove_stopwords=True)
        if cat not in cat_keyword:
            cat_keyword[cat] = []
        cat_keyword[cat].extend(words)

    for cat in cat_keyword:
        cat_keyword[cat] = Counter(cat_keyword[cat]).most_common(20)

    # Bigrams
    all_clean = [w for tokens in clean_tokens for w in tokens]
    bigrams = list(ngrams(all_clean, 2))
    bigram_freq = Counter(bigrams).most_common(30)
    print(f"\n  Top 15 bigrams:")
    for bg, c in bigram_freq[:15]:
        print(f"    {' '.join(bg)}: {c}")

    # Category distribution
    cats = [parse_filename(img['filename']) for img in imgs]
    cat_counts = Counter(cats)

    return {
        'captions': all_captions,
        'num_imgs': len(imgs),
        'word_counts': word_counts,
        'char_counts': char_counts,
        'raw_words': raw_words,
        'clean_words': clean_words,
        'word_freq': word_freq,
        'word_freq_raw': word_freq_raw,
        'top_words': top_words,
        'cat_keyword': cat_keyword,
        'top_bigrams': bigram_freq,
        'categories': cats,
        'cat_counts': cat_counts,
    }


def viz_text(text_stats, split='train'):
    """Generate text analysis visualizations."""
    print(f"\n  Generating text visualizations...")

    # 1. Caption Length Distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(text_stats['word_counts'], bins=30, edgecolor='white', alpha=0.8, color='#3498db')
    ax.axvline(np.mean(text_stats['word_counts']), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(text_stats["word_counts"]):.1f}')
    ax.axvline(np.median(text_stats['word_counts']), color='green', linestyle='--', linewidth=2,
               label=f'Median: {np.median(text_stats["word_counts"]):.1f}')
    ax.set_xlabel('Word Count per Caption')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Caption Length Distribution ({split.upper()})')
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'txt_02_length_distribution_{split}.png', dpi=150)
    plt.close()

    # 3. Character Count Distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(text_stats['char_counts'], bins=30, edgecolor='white', alpha=0.8, color='#e74c3c')
    ax.axvline(np.mean(text_stats['char_counts']), color='blue', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(text_stats["char_counts"]):.1f}')
    ax.set_xlabel('Character Count per Caption')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Character Count Distribution ({split.upper()})')
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'txt_03_char_distribution_{split}.png', dpi=150)
    plt.close()

    # 3b. Word vs Character count correlation
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(text_stats['word_counts'], text_stats['char_counts'], s=10, alpha=0.35, color='#8e44ad')
    corr_wc = np.corrcoef(text_stats['word_counts'], text_stats['char_counts'])[0, 1]
    ax.set_xlabel('Word Count per Caption')
    ax.set_ylabel('Character Count per Caption')
    ax.set_title(f'Words vs Characters Correlation ({split.upper()}) | r={corr_wc:.3f}')
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'txt_03b_words_vs_chars_{split}.png', dpi=180)
    plt.close()

    # 4. Top 10 Words (no stopwords)
    fig, ax = plt.subplots(figsize=(8, 5))
    top10_words = text_stats['top_words'][:10]
    words = [w for w, c in top10_words]
    counts = [c for w, c in top10_words]
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(words)))[::-1]
    bars = ax.barh(words[::-1], counts[::-1], color=colors, edgecolor='none', height=0.60)
    for bar, val in zip(bars, counts[::-1]):
        ax.text(val + 30, bar.get_y() + bar.get_height()/2,
                f'{val:,}', va='center', fontsize=8, color='#333')
    ax.set_xlabel('Frequency', fontsize=10)
    ax.set_title(
        f'Top 10 Most Frequent Words ({split.upper()}, no stopwords)',
        fontsize=11, fontweight='bold',
    )
    ax.tick_params(axis='y', labelsize=9)
    ax.set_xlim(0, max(counts) * 1.18)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.xaxis.set_visible(False)
    ax.grid(axis='x', color='#EEEEEE', linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'ta_02_word_frequency_{split}.png', dpi=180, bbox_inches='tight', facecolor='#FAFAFA')
    plt.close()

    # 5. Category Keywords Heatmap (Top 15 categories)
    fig, ax = plt.subplots(figsize=(14, 9))
    top_cats = [cat for cat, _ in text_stats['cat_counts'].most_common(15)]
    keyword_matrix = []
    all_keywords = set()
    for cat in top_cats:
        for w, c in text_stats['cat_keyword'].get(cat, [])[:10]:
            all_keywords.add(w)
    all_keywords = list(all_keywords)[:30]

    for cat in top_cats:
        kw = dict(text_stats['cat_keyword'].get(cat, []))
        row = [kw.get(w, 0) for w in all_keywords]
        keyword_matrix.append(row)

    keyword_matrix = np.array(keyword_matrix)
    sns.heatmap(keyword_matrix, xticklabels=all_keywords, yticklabels=top_cats,
                cmap='YlOrRd', ax=ax)
    ax.set_title(f'Category Keywords Heatmap ({split.upper()})')
    ax.set_xlabel('Keywords')
    ax.set_ylabel('Category')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'txt_06_category_keywords_{split}.png', dpi=150)
    plt.close()

    # 6. Top-10 Bigrams (clean standalone chart)
    fig, ax = plt.subplots(figsize=(9, 5))
    top10_bgs = text_stats['top_bigrams'][:10]
    labels = [' '.join(bg) for bg, _ in top10_bgs]
    vals  = [c for _, c in top10_bgs]
    colors = plt.cm.Oranges(np.linspace(0.4, 0.95, len(labels)))
    bars = ax.barh(labels[::-1], vals[::-1], color=colors[::-1], height=0.65, zorder=3)
    for bar, val in zip(bars, vals[::-1]):
        ax.text(val + 15, bar.get_y() + bar.get_height()/2,
                f"{val:,}", va='center', fontsize=8.5, color='#333333')
    ax.set_xlabel("Frequency", fontsize=10)
    ax.set_title(
        f"Top 10 Bigrams ({split.upper()}, stopwords removed)",
        fontsize=11, fontweight="bold", color="#2C3E50", pad=8,
    )
    ax.grid(axis='x', color='#EEEEEE', linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.xaxis.set_visible(False)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'ta_04_bigram_frequency_{split}.png', dpi=180,
                bbox_inches='tight', facecolor='#FAFAFA')
    plt.close()

    print(f"  Generated text visualizations (core set)")


# ─────────────────────────────────────────────
# 2B. IMAGE PIXEL ANALYSIS (visual features from raw TIF files)
# ─────────────────────────────────────────────
def compute_image_level_stats(imgs, max_images_per_cat=30):
    """Compute image-level quality stats: size, aspect, brightness, texture, blur, RGB means."""
    cat_to_files = defaultdict(list)
    for img in imgs:
        cat_to_files[parse_filename(img['filename'])].append(img['filename'])

    rows = []
    for cat, filenames in cat_to_files.items():
        for fname in filenames[:max_images_per_cat]:
            path = IMG_DIR / fname
            if not path.is_file():
                raise FileNotFoundError(f"[LỖI] Thiếu file ảnh (bắt buộc): {path}")
            with Image.open(path) as pil:
                arr = np.array(pil.convert('RGB'), dtype=np.float32) / 255.0
            h, w = arr.shape[:2]
            gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
            lap = (
                -4.0 * gray
                + np.roll(gray, 1, axis=0)
                + np.roll(gray, -1, axis=0)
                + np.roll(gray, 1, axis=1)
                + np.roll(gray, -1, axis=1)
            )
            blur_score = float(np.var(lap))

            rows.append({
                'filename': fname,
                'category': cat,
                'width': int(w),
                'height': int(h),
                'aspect_ratio': float(w / h) if h > 0 else 0.0,
                'brightness': float(np.mean(arr)),
                'texture': float(np.std(arr)),
                'blur_score': blur_score,
                'r_mean': float(np.mean(arr[:, :, 0])),
                'g_mean': float(np.mean(arr[:, :, 1])),
                'b_mean': float(np.mean(arr[:, :, 2])),
            })

    return pd.DataFrame(rows)


def compute_global_rgb_channel_means(imgs):
    """
    Full pass over all images in `imgs`: only per-channel RGB means (no Laplacian / blur).
    Builds global percentiles (e.g. P05) on the full split population — not the per-category
    subsample used in compute_image_level_stats.
    Returns (DataFrame with filename + r_mean,g_mean,b_mean, dict p05 with keys r,g,b).
    """
    rows = []
    for img in imgs:
        fname = img.get('filename', '')
        path = IMG_DIR / fname
        if not path.is_file():
            raise FileNotFoundError(f"[LỖI] Thiếu file ảnh (bắt buộc cho P05): {path}")
        with Image.open(path) as pil:
            arr = np.array(pil.convert('RGB'), dtype=np.float32) / 255.0
        rows.append({
            'filename': fname,
            'r_mean': float(np.mean(arr[:, :, 0])),
            'g_mean': float(np.mean(arr[:, :, 1])),
            'b_mean': float(np.mean(arr[:, :, 2])),
        })

    df = pd.DataFrame(rows)
    if df.empty and imgs:
        raise RuntimeError("[LỖI] compute_global_rgb_channel_means: có ảnh trong JSON nhưng không đọc được RGB.")
    if df.empty:
        p05 = {'r': np.nan, 'g': np.nan, 'b': np.nan}
    else:
        p05 = {
            'r': float(np.percentile(df['r_mean'], 5)),
            'g': float(np.percentile(df['g_mean'], 5)),
            'b': float(np.percentile(df['b_mean'], 5)),
        }
    return df, p05


def loo_caption_centroid_similarities(caps):
    """
    Leave-one-out TF-IDF: for each caption, fit vectorizer on the other captions,
    centroid = mean TF-IDF of those, cosine similarity of held-out caption to centroid.
    """
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
        sim = float(cosine_similarity(cap_vec, centroid)[0, 0])
        out.append(sim)
    return out


def compute_category_pixel_stats(imgs):
    """Compute per-category brightness, texture, and dominant RGB channel from real TIF images."""
    from collections import defaultdict

    brightness = {}
    texture = {}
    dom_channel = {}
    _cat_imgs = defaultdict(list)

    for img in imgs:
        cat = parse_filename(img['filename'])
        _cat_imgs[cat].append(img['filename'])

    for cat, filenames in _cat_imgs.items():
        brights, textures = [], []
        r_sum, g_sum, b_sum = 0.0, 0.0, 0.0
        pixel_count = 0

        for fname in filenames[:20]:  # sample up to 20 images per category
            path = IMG_DIR / fname
            if not path.is_file():
                raise FileNotFoundError(f"[LỖI] Thiếu file ảnh: {path}")
            with Image.open(path) as img_pil:
                arr = np.array(img_pil.convert('RGB'), dtype=np.float32) / 255.0

            brights.append(float(np.mean(arr)))
            textures.append(float(np.std(arr)))

            r_sum += float(np.sum(arr[:, :, 0]))
            g_sum += float(np.sum(arr[:, :, 1]))
            b_sum += float(np.sum(arr[:, :, 2]))
            pixel_count += arr.shape[0] * arr.shape[1]

        if not brights:
            raise RuntimeError(f"[LỖI] Category '{cat}': không đọc được ảnh nào (đường dẫn / định dạng).")

        brightness[cat] = round(np.mean(brights), 3)
        texture[cat] = round(np.mean(textures), 3)

        if pixel_count > 0:
            r_avg = r_sum / pixel_count
            g_avg = g_sum / pixel_count
            b_avg = b_sum / pixel_count
            if g_avg > r_avg and g_avg > b_avg:
                dom_channel[cat] = 'G>R>B'
            elif b_avg > r_avg and b_avg > g_avg:
                dom_channel[cat] = 'B>R>G'
            elif r_avg > b_avg:
                dom_channel[cat] = 'R>G>B'
            else:
                dom_channel[cat] = 'R≈G>B'
        else:
            dom_channel[cat] = 'R≈G>B'

    return brightness, texture, dom_channel


def analyze_image_pixel(imgs, split='train'):
    """Analyze pixel-level visual statistics per image category."""
    print(f"\n{'='*60}")
    print(f"IMAGE PIXEL ANALYSIS — {split.upper()} SET")
    print(f"{'='*60}")

    cats = [parse_filename(img['filename']) for img in imgs]
    cat_counts = Counter(cats)
    brightness, texture, dom_channel = compute_category_pixel_stats(imgs)
    image_df = compute_image_level_stats(imgs)
    global_rgb_df, p05_rgb = compute_global_rgb_channel_means(imgs)

    blur_by_cat = {}
    if not image_df.empty:
        blur_by_cat = image_df.groupby('category')['blur_score'].mean().to_dict()

    rows = []
    for cat, count in sorted(cat_counts.items(), key=lambda x: -x[1]):
        rows.append({
            'Category': cat,
            'Images': count,
            'Brightness': round(brightness.get(cat, 0.50), 3),
            'Texture': round(texture.get(cat, 0.12), 3),
            'Blur': round(float(blur_by_cat.get(cat, 0.0)), 4),
            'Dom. Channel': dom_channel.get(cat, 'R≈G>B'),
        })

    vis_df = pd.DataFrame(rows)
    print(f"\nBrightness range: {vis_df['Brightness'].min():.3f} – {vis_df['Brightness'].max():.3f}")
    print(f"Texture range   : {vis_df['Texture'].min():.3f} – {vis_df['Texture'].max():.3f}")
    if not image_df.empty:
        print(f"Blur range      : {image_df['blur_score'].min():.4f} – {image_df['blur_score'].max():.4f}")
        print(f"Image size      : {image_df['width'].min()}x{image_df['height'].min()} to "
              f"{image_df['width'].max()}x{image_df['height'].max()}")
    print("\nDominant RGB channel distribution:")
    for ch, cnt in vis_df['Dom. Channel'].value_counts().items():
        print(f"  {ch}: {cnt} categories")

    print(f"\nPer-Category Visual Statistics ({split.upper()})")
    print(vis_df.sort_values('Brightness', ascending=False).to_string(index=False))

    if not global_rgb_df.empty:
        print(f"\nGlobal RGB P05 (full split, all images with readable TIF — for multimodal noise rules):")
        print(f"  P05(R)={p05_rgb['r']:.4f}  P05(G)={p05_rgb['g']:.4f}  P05(B)={p05_rgb['b']:.4f}  (n={len(global_rgb_df)} images)")

    return {
        'vis_df': vis_df,
        'brightness': brightness,
        'texture': texture,
        'dom_channel': dom_channel,
        'image_df': image_df,
        'global_rgb_df': global_rgb_df,
        'p05_rgb': p05_rgb,
    }


def viz_image_pixel(pixel_stats, split='train'):
    """Generate image pixel-level visualizations (top 10 per metric)."""
    print(f"\n  Generating image pixel visualizations...")
    vis_df = pixel_stats['vis_df']
    BG = '#F7F3EB'

    # ── Brightness: top 10 categories ──
    df_bright = vis_df.sort_values('Brightness', ascending=False).head(10).sort_values('Brightness', ascending=True)
    fig_b, ax_b = plt.subplots(figsize=(9, 5))
    fig_b.patch.set_facecolor(BG)
    ax_b.set_facecolor(BG)
    colors_b = plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(df_bright)))
    bars_b = ax_b.barh(df_bright['Category'], df_bright['Brightness'],
                        color=colors_b, edgecolor='none', height=0.60)
    for bar, val in zip(bars_b, df_bright['Brightness']):
        ax_b.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                  f'{val:.3f}', va='center', fontsize=8.5, color='#333')
    ax_b.axvline(vis_df['Brightness'].mean(), color='#B42318', linestyle='--', lw=1.5,
                 label=f"Dataset mean = {vis_df['Brightness'].mean():.3f}")
    ax_b.set_xlabel('Mean Pixel Brightness (0–1)', fontsize=10)
    ax_b.set_title(
        f'Top 10 Brightest Categories ({split.upper()}, n={len(vis_df)} cats)',
        fontsize=11, fontweight='bold',
    )
    ax_b.tick_params(axis='y', labelsize=9)
    ax_b.set_xlim(0, df_bright['Brightness'].max() * 1.18)
    ax_b.legend(fontsize=9, loc='lower right')
    for s in ax_b.spines.values():
        s.set_visible(False)
    ax_b.xaxis.set_visible(False)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'ia_02_image_visual_analysis_{split}.png', dpi=180, bbox_inches='tight')
    plt.close()

    # ── Texture: top 10 categories ──
    df_tex = vis_df.sort_values('Texture', ascending=False).head(10).sort_values('Texture', ascending=True)
    fig_t, ax_t = plt.subplots(figsize=(9, 5))
    fig_t.patch.set_facecolor(BG)
    ax_t.set_facecolor(BG)
    colors_t = plt.cm.Purples(np.linspace(0.3, 0.85, len(df_tex)))
    bars_t = ax_t.barh(df_tex['Category'], df_tex['Texture'],
                        color=colors_t, edgecolor='none', height=0.60)
    for bar, val in zip(bars_t, df_tex['Texture']):
        ax_t.text(val + 0.003, bar.get_y() + bar.get_height()/2,
                  f'{val:.3f}', va='center', fontsize=8.5, color='#333')
    ax_t.axvline(vis_df['Texture'].mean(), color='#6B4C8E', linestyle='--', lw=1.5,
                 label=f"Dataset mean = {vis_df['Texture'].mean():.3f}")
    ax_t.set_xlabel('Texture (Pixel Std Dev)', fontsize=10)
    ax_t.set_title(
        f'Top 10 Highest-Texture Categories ({split.upper()}, n={len(vis_df)} cats)',
        fontsize=11, fontweight='bold',
    )
    ax_t.tick_params(axis='y', labelsize=9)
    ax_t.set_xlim(0, df_tex['Texture'].max() * 1.22)
    ax_t.legend(fontsize=9, loc='lower right')
    for s in ax_t.spines.values():
        s.set_visible(False)
    ax_t.xaxis.set_visible(False)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'ia_02b_texture_top10_{split}.png', dpi=180, bbox_inches='tight')
    plt.close()

    # ── RGB channel dominance: already compact (4 bars) ──
    fig2, ax2 = plt.subplots(figsize=(9, 4))
    fig2.patch.set_facecolor(BG)
    CH_COLORS = {'G>R>B': '#4CAF50', 'B>R>G': '#2196F3',
                 'R≈G>B': '#FF9800', 'R>G>B': '#795548'}
    ch_counts = vis_df['Dom. Channel'].value_counts()
    bars = ax2.bar(ch_counts.index, ch_counts.values,
                   color=[CH_COLORS.get(c, '#9E9E9E') for c in ch_counts.index],
                   edgecolor='none', width=0.55)
    for bar, val in zip(bars, ch_counts.values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 str(val), ha='center', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Dominant RGB Channel Pattern', fontsize=10)
    ax2.set_ylabel('Number of Categories', fontsize=10)
    ax2.set_title(
        f'RGB Channel Dominance ({split.upper()}, n={len(vis_df)} categories)',
        fontsize=11, fontweight='bold',
    )
    ax2.set_facecolor(BG)
    ax2.tick_params(axis='x', labelsize=10)
    for s in ax2.spines.values():
        s.set_visible(False)
    ax2.yaxis.set_visible(False)
    ax2.grid(axis='y', color='#D4C9B8', linewidth=0.5, zorder=0)
    leg_patches = [plt.Rectangle((0,0), 1, 1, facecolor=c)
                   for c in CH_COLORS.values()]
    ax2.legend(leg_patches, list(CH_COLORS.keys()),
               loc='upper right', frameon=False, fontsize=9)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'ia_03_rgb_channel_dominance_{split}.png', dpi=180, bbox_inches='tight')
    plt.close()

    # ── Image geometry + blur distributions (image-level) ──
    image_df = pixel_stats.get('image_df', pd.DataFrame())
    if image_df.empty:
        raise RuntimeError(
            "[LỖI] ia_04/ia_05 cần image_df không rỗng — kiểm tra ảnh TIF và compute_image_level_stats."
        )

    fig3, axs = plt.subplots(2, 2, figsize=(12, 9))
    fig3.patch.set_facecolor(BG)

    axs[0, 0].hist(image_df['width'], bins=20, color='#5DA5DA', edgecolor='white')
    axs[0, 0].set_title('Width Distribution')

    axs[0, 1].hist(image_df['height'], bins=20, color='#60BD68', edgecolor='white')
    axs[0, 1].set_title('Height Distribution')

    axs[1, 0].hist(image_df['aspect_ratio'], bins=20, color='#F17CB0', edgecolor='white')
    axs[1, 0].set_title('Aspect Ratio Distribution')

    axs[1, 1].hist(image_df['blur_score'], bins=20, color='#B2912F', edgecolor='white')
    axs[1, 1].set_title('Blur Score Distribution (Laplacian Variance)')

    for a in axs.ravel():
        a.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'ia_04_geometry_blur_distribution_{split}.png', dpi=180, bbox_inches='tight')
    plt.close()

    rgb_corr = image_df[['r_mean', 'g_mean', 'b_mean']].corr()
    fig4, ax4 = plt.subplots(figsize=(5.5, 4.5))
    fig4.patch.set_facecolor(BG)
    sns.heatmap(rgb_corr, annot=True, fmt='.2f', cmap='RdYlBu_r', vmin=-1, vmax=1, ax=ax4)
    ax4.set_title('RGB Mean Channel Correlation')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'ia_05_rgb_channel_corr_{split}.png', dpi=180, bbox_inches='tight')
    plt.close()

    print(
        f"  Generated ia_02_image_visual_analysis_{split}.png, ia_02b_texture_top10_{split}.png, "
        f"ia_03_rgb_channel_dominance_{split}.png, ia_04_geometry_blur_distribution_{split}.png, "
        f"ia_05_rgb_channel_corr_{split}.png"
    )


def analyze_image(imgs, split='train'):
    """Analyze image metadata from filenames."""
    print(f"\n{'='*60}")
    print(f"IMAGE ANALYSIS — {split.upper()} SET")
    print(f"{'='*60}")

    cats = [parse_filename(img['filename']) for img in imgs]
    cat_counts = Counter(cats)

    print(f"  Total Images: {len(imgs)}")
    print(f"  Categories: {len(cat_counts)}")
    print(f"  Avg Images/Category: {len(imgs)/len(cat_counts):.1f}")
    print(f"\n  Top 10 categories:")
    for cat, count in cat_counts.most_common(10):
        print(f"    {cat}: {count}")

    return {'num_imgs': len(imgs), 'cat_counts': cat_counts, 'categories': cats}


def viz_image(image_stats, split='train'):
    """Generate image analysis visualizations."""
    print(f"\n  Generating image visualizations...")

    cat_df = pd.DataFrame(image_stats['cat_counts'].most_common(),
                          columns=['Category', 'Count'])
    cat_sorted = cat_df.sort_values('Count', ascending=False).reset_index(drop=True)

    BG = '#FAFAFA'

    # Split 33 categories into 4 groups for readability
    groups = []
    sizes  = [9, 8, 8, 8]
    start  = 0
    for sz in sizes:
        groups.append(cat_sorted.iloc[start:start + sz].reset_index(drop=True))
        start += sz

    fig, axes = plt.subplots(2, 2, figsize=(14, 13))
    fig.patch.set_facecolor(BG)
    for ax, gdf, grp_idx in zip(axes.flat, groups, range(1, 5)):
        ax.set_facecolor(BG)
        gdf_s = gdf.sort_values('Count', ascending=True)
        n = len(gdf_s)
        cmap = plt.cm.viridis(np.linspace(0.2, 0.9, n))
        bars = ax.barh(range(n), gdf_s['Count'], color=cmap, edgecolor='none', height=0.60)
        ax.set_yticks(range(n))
        ax.set_yticklabels(gdf_s['Category'], fontsize=8)
        for bar, val in zip(bars, gdf_s['Count']):
            ax.text(val + 1, bar.get_y() + bar.get_height() / 2,
                    f'{val}', va='center', fontsize=8, color='#333')
        ax.set_xlim(0, cat_sorted['Count'].max() * 1.15)
        ax.tick_params(axis='x', labelsize=7)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.xaxis.set_visible(False)
        ax.grid(axis='x', color='#EEEEEE', linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
        ax.set_title(f'Group {grp_idx}', fontsize=9, fontweight='bold', color='#555', pad=4)

    fig.text(0.5, 0.01, f'Image Category Distribution — {split.upper()} | {len(cat_sorted)} categories total',
             ha='center', fontsize=11, fontweight='bold')
    
    def get_cnt(i):
        return cat_sorted.iloc[i]["Count"] if i < len(cat_sorted) else 0

    fig.text(0.5, -0.01,
             f'Group 1: {get_cnt(0)}-{get_cnt(8)} imgs | '
             f'Group 2: {get_cnt(9)}-{get_cnt(16)} imgs | '
             f'Group 3: {get_cnt(17)}-{get_cnt(24)} imgs | '
             f'Group 4: {get_cnt(25)}-{get_cnt(32)} imgs',
             ha='center', fontsize=8, color='#666', style='italic')
    plt.tight_layout(rect=[0, 0.025, 1, 1], pad=1.2)
    plt.savefig(OUTPUT_DIR / f'ia_01_category_distribution_{split}.png',
                dpi=180, bbox_inches='tight', facecolor=BG)
    plt.close()

    print(f"  Generated ia_01_category_distribution_{split}.png")


def main():
    print("\n" + "="*60)
    print("RSITMD MULTIMODAL EDA")
    print("="*60)
    print(f"Dataset: Remote Sensing Image-Text Matching Dataset")
    print(f"Reference: Z. Yuan et al., IEEE TGRS 2021")
    print(f"Output: {OUTPUT_DIR}")
    print("="*60)

    clean_output_dir()

    images, train_imgs, test_imgs = load_data()
    print(f"\nLoaded {len(images)} total images: {len(train_imgs)} train, {len(test_imgs)} test")

    # 0. Data audit
    run_data_audit(images, train_imgs, test_imgs)

    # Section B1 — Text EDA (analyze + visualize)
    train_text = analyze_text(train_imgs, 'train')
    test_text = analyze_text(test_imgs, 'test')
    viz_text(train_text, 'train')
    viz_text(test_text, 'test')

    # Section B2 — Image EDA (metadata + pixel quality, analyze + visualize)
    train_img = analyze_image(train_imgs, 'train')
    test_img = analyze_image(test_imgs, 'test')
    train_pixel = analyze_image_pixel(train_imgs, 'train')
    test_pixel  = analyze_image_pixel(test_imgs,  'test')
    viz_image(train_img, 'train')
    viz_image(test_img, 'test')
    viz_image_pixel(train_pixel, 'train')
    viz_image_pixel(test_pixel,  'test')

    # Section C — Multimodal EDA (analyze + visualize)
    train_mm = analyze_multimodal(train_imgs, 'train', cat_keyword_dict=train_text['cat_keyword'], pixel_stats=train_pixel)
    test_mm = analyze_multimodal(test_imgs, 'test', cat_keyword_dict=test_text['cat_keyword'], pixel_stats=test_pixel)
    viz_multimodal(train_mm, 'train')
    viz_multimodal(test_mm, 'test')

    # Section D — Drift + Sensitivity + Summary export
    viz_train_test_drift(train_text, test_text, train_pixel, test_pixel)
    viz_noise_threshold_sensitivity(train_mm)
    generate_summary_csv(train_text, test_text, train_img, test_img)
    print_executive_summary(train_text, test_text, train_pixel, test_pixel, train_mm, test_mm)

    print("\n" + "="*60)
    print("EDA COMPLETE!")
    print("="*60)
    print(f"\nAll figures saved to: {OUTPUT_DIR}")
    for f in sorted(OUTPUT_DIR.glob("*.png")):
        print(f"  - {f.name}")


def analyze_multimodal(imgs, split='train', cat_keyword_dict=None, pixel_stats=None, iqr_multiplier=2.0):
    print(f"\n{'='*60}")
    print(f"MULTIMODAL ANALYSIS — {split.upper()} SET")
    print(f"{'='*60}")

    # Caption variability + semantic consistency within same image
    variabilities = []
    semantic_consistency = []
    caption_lengths_per_img = []
    all_captions_flat = []
    for img in imgs:
        caps_raw = [s['raw'] for s in img['sentences']]
        lens = [len(s['tokens']) for s in img['sentences']]

        sem_sim = np.nan
        if len(caps_raw) >= 2:
            vec_local = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=1, lowercase=True)
            mat_local = vec_local.fit_transform(caps_raw)
            sim_local = cosine_similarity(mat_local)
            upper_idx = np.triu_indices_from(sim_local, k=1)
            sem_sim = float(np.mean(sim_local[upper_idx])) if len(upper_idx[0]) > 0 else np.nan

        caption_lengths_per_img.append({
            'filename': img['filename'],
            'category': parse_filename(img['filename']),
            'lens': lens,
            'mean': np.mean(lens),
            'std': np.std(lens) if len(lens) > 1 else 0,
            'semantic_consistency': sem_sim,
            'captions': caps_raw,
        })
        variabilities.append(np.std(lens) if len(lens) > 1 else 0)
        if not np.isnan(sem_sim):
            semantic_consistency.append(sem_sim)
        for s in img['sentences']:
            all_captions_flat.append(s['raw'])

    print(f"  Total Image-Caption Pairs: {len(imgs)} images × 5 captions = {len(all_captions_flat)}")
    print(f"  Caption length std (within image) — mean: {np.mean(variabilities):.2f}, max: {np.max(variabilities):.2f}")

    # Sample pairs
    print(f"\n  Sample pairs:")
    for sample in caption_lengths_per_img[:3]:
        print(f"    {sample['filename']} ({sample['category']})")
        for i, cap in enumerate(sample['captions'][:3]):
            print(f"      [{i}] {cap[:80]}")

    # Visual word coverage
    raw_tokens = [tokenize(cap) for cap in all_captions_flat]
    all_raw_words = [w for tokens in raw_tokens for w in tokens]
    wf = Counter(all_raw_words)

    COLOR_WORDS = {'green', 'white', 'red', 'blue', 'gray', 'grey', 'yellow', 'dark',
                   'black', 'brown', 'light', 'orange', 'bright', 'purple', 'colored', 'pink'}
    COUNT_WORDS = {'many', 'several', 'two', 'three', 'four', 'five', 'some', 'few',
                   'numerous', 'lots', 'many', 'several', 'multiple', 'a', 'one'}
    SPATIAL_WORDS = {'near', 'surrounded', 'around', 'next', 'beside', 'middle', 'center',
                     'adjacent', 'above', 'below', 'in', 'on', 'with', 'and'}
    SIZE_WORDS = {'large', 'small', 'big', 'long', 'short', 'wide', 'narrow', 'huge', 'tiny'}

    color_count = sum(1 for w in all_raw_words if w in COLOR_WORDS)
    count_count = sum(1 for w in all_raw_words if w in COUNT_WORDS)
    spatial_count = sum(1 for w in all_raw_words if w in SPATIAL_WORDS)
    size_count = sum(1 for w in all_raw_words if w in SIZE_WORDS)

    total_caps = len(all_captions_flat)
    caps_with_color = sum(1 for cap in all_captions_flat if any(w in tokenize(cap) for w in COLOR_WORDS))
    caps_with_spatial = sum(1 for cap in all_captions_flat if any(w in tokenize(cap) for w in SPATIAL_WORDS))

    print(f"\n  Visual word coverage:")
    print(f"    Color words: {color_count:,} ({color_count/len(all_raw_words)*100:.1f}% of words)")
    print(f"    Count words: {count_count:,}")
    print(f"    Spatial words: {spatial_count:,}")
    print(f"    Size words: {size_count:,}")
    print(f"    Caps with color: {caps_with_color}/{total_caps} ({caps_with_color/total_caps*100:.1f}%)")
    print(f"    Caps with spatial: {caps_with_spatial}/{total_caps} ({caps_with_spatial/total_caps*100:.1f}%)")

    # Top color words
    color_word_freq = {w: c for w, c in wf.items() if w in COLOR_WORDS}
    top_colors = sorted(color_word_freq.items(), key=lambda x: -x[1])[:15]
    print(f"\n  Top color words:")
    total_color = sum(color_word_freq.values())
    for w, c in top_colors:
        print(f"    {w}: {c} ({c/total_color*100:.1f}%)")

    # Top objects (no stopwords)
    clean_tokens = [tokenize(cap, remove_stopwords=True) for cap in all_captions_flat]
    clean_words = [w for tokens in clean_tokens for w in tokens]
    object_freq = Counter(clean_words)
    print(f"\n  Top objects (no stopwords):")
    for w, c in object_freq.most_common(15):
        print(f"    {w}: {c}")

    # Spatial relationships
    SPATIAL_PREPS = {'near': 0, 'surrounded': 0, 'around': 0, 'next to': 0, 'beside': 0,
                     'in the middle of': 0, 'in the center of': 0, 'adjacent to': 0,
                     'above': 0, 'below': 0}
    for cap in all_captions_flat:
        cl = cap.lower()
        for prep in SPATIAL_PREPS:
            if prep in cl:
                SPATIAL_PREPS[prep] += 1

    # Category-level contradiction map: multi-color + object token mismatch (heuristic, model-free)
    contradiction_rows = []
    if pixel_stats and 'dom_channel' in pixel_stats:
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

        # dominant-channel to likely supported colors (coarse prior)
        dom_to_supported_colors = {
            'G>R>B': {'green'},
            'B>R>G': {'blue'},
            'R>G>B': {'red', 'brown'},
            'R≈G>B': {'gray', 'white', 'brown'},
        }

        cap_stats = defaultdict(lambda: {
            'n': 0,
            'color_claims': 0,
            'color_mismatch': 0,
            'object_claims': 0,
            'object_mismatch': 0,
        })

        for img in imgs:
            cat = parse_filename(img['filename'])
            img_dom_channel = pixel_stats['dom_channel'].get(cat, 'R≈G>B')
            supported_colors = dom_to_supported_colors.get(img_dom_channel, {'gray'})
            cat_lower = cat.lower()

            for s in img['sentences']:
                toks = set(tokenize(s['raw']))
                cap_stats[cat]['n'] += 1

                # Color claims
                claimed_colors = set()
                for cname, kws in color_lexicon.items():
                    if any(w in toks for w in kws):
                        claimed_colors.add(cname)
                if claimed_colors:
                    cap_stats[cat]['color_claims'] += 1
                    if claimed_colors.isdisjoint(supported_colors):
                        cap_stats[cat]['color_mismatch'] += 1

                # Object claims (mismatch if token group is absent from category name)
                claimed_obj_groups = []
                for group, kws in object_lexicon.items():
                    if any(w in toks for w in kws):
                        claimed_obj_groups.append((group, kws))
                if claimed_obj_groups:
                    cap_stats[cat]['object_claims'] += 1
                    if not any(any(kw in cat_lower for kw in kws) for _, kws in claimed_obj_groups):
                        cap_stats[cat]['object_mismatch'] += 1

        for cat, d in cap_stats.items():
            color_rate = (d['color_mismatch'] / d['color_claims']) if d['color_claims'] > 0 else 0.0
            object_rate = (d['object_mismatch'] / d['object_claims']) if d['object_claims'] > 0 else 0.0
            combined = 0.6 * color_rate + 0.4 * object_rate
            contradiction_rows.append({
                'category': cat,
                'color_mismatch_rate': color_rate,
                'object_mismatch_rate': object_rate,
                'combined_mismatch_rate': combined,
                'color_claims': d['color_claims'],
                'object_claims': d['object_claims'],
                'total_caps': d['n'],
            })
    print(f"\n  Spatial relationships:")
    for prep, cnt in sorted(SPATIAL_PREPS.items(), key=lambda x: -x[1]):
        if cnt > 0:
            print(f"    {prep}: {cnt}")

    # ===== CATEGORY-LEVEL TEXT SIMILARITY (for multimodal interpretation) =====
    category_similarity_df = pd.DataFrame()
    cat_to_captions_for_sim = defaultdict(list)
    for img in imgs:
        cat = parse_filename(img['filename'])
        for s in img['sentences']:
            cat_to_captions_for_sim[cat].append(s['raw'])

    cat_names = sorted(cat_to_captions_for_sim.keys())
    cat_docs = [' '.join(cat_to_captions_for_sim[c]) for c in cat_names]
    if len(cat_docs) >= 2:
        vec_cat = TfidfVectorizer(stop_words='english', min_df=1)
        cat_mat = vec_cat.fit_transform(cat_docs)
        sim_mat = cosine_similarity(cat_mat)
        category_similarity_df = pd.DataFrame(sim_mat, index=cat_names, columns=cat_names)

    # ===== THREE-LAYER IMAGE-LEVEL NOISE DETECTION =====
    # Layer A: intra-caption semantic inconsistency
    # Layer B: cross-modal contradiction evidence
    # Layer C: uncertainty calibration (low-information captions)
    anomalies = []
    anomaly_probe = []
    image_noise_rows = []

    all_loo_scores = []
    for img in imgs:
        caps = [s['raw'] for s in img['sentences']]
        scores = loo_caption_centroid_similarities(caps) if len(caps) >= 2 else [np.nan] * len(caps)
        all_loo_scores.extend(scores)

    loo_arr = np.array(all_loo_scores, dtype=float)
    loo_arr = loo_arr[~np.isnan(loo_arr)]
    layer_a_q1 = layer_a_q3 = layer_a_iqr = layer_a_cutoff = np.nan
    if len(loo_arr) >= 4:
        layer_a_q1 = float(np.percentile(loo_arr, 25))
        layer_a_q3 = float(np.percentile(loo_arr, 75))
        layer_a_iqr = layer_a_q3 - layer_a_q1
        if layer_a_iqr > 0:
            layer_a_cutoff = layer_a_q1 - iqr_multiplier * layer_a_iqr

    print("\n  Three-layer noise scoring (A: semantic, B: contradiction, C: uncertainty)")
    if not np.isnan(layer_a_cutoff):
        print(f"    Layer-A fence: Q1={layer_a_q1:.4f}, Q3={layer_a_q3:.4f}, IQR={layer_a_iqr:.4f}, cutoff={layer_a_cutoff:.4f}")

    contradiction_by_cat = {}
    contradiction_df = pd.DataFrame(contradiction_rows)
    if not contradiction_df.empty:
        for _, r in contradiction_df.iterrows():
            contradiction_by_cat[r['category']] = {
                'color_rate': float(r.get('color_mismatch_rate', 0.0)),
                'object_rate': float(r.get('object_mismatch_rate', 0.0)),
            }

    for img in imgs:
        filename = img['filename']
        cat = parse_filename(filename)
        caps = [s['raw'] for s in img['sentences']]
        if len(caps) < 2:
            continue

        vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=1, lowercase=True)
        mat = vec.fit_transform(caps)
        sim = cosine_similarity(mat)

        iu = np.triu_indices_from(sim, k=1)
        pair_vals = sim[iu]
        mean_pair = float(np.mean(pair_vals)) if len(pair_vals) else np.nan

        center = np.asarray(mat.mean(axis=0))
        center_sims = cosine_similarity(mat, center).reshape(-1)
        center_min = float(np.min(center_sims)) if len(center_sims) else np.nan

        q1_img = float(np.percentile(center_sims, 25))
        q3_img = float(np.percentile(center_sims, 75))
        iqr_img = q3_img - q1_img
        low_cut_img = q1_img - 1.5 * iqr_img
        outlier_count = int(np.sum(center_sims < low_cut_img)) if iqr_img > 0 else 0

        # Layer A
        sem_noise = float(np.clip(1.0 - mean_pair, 0.0, 1.0))
        center_penalty = float(np.clip((0.55 - center_min) / 0.55, 0.0, 1.0)) if np.isfinite(center_min) else 0.0
        layer_a_score = 0.7 * sem_noise + 0.3 * center_penalty

        # Layer B (per-image contradiction + weak category prior)
        color_lexicon_img = {
            'blue': ['blue', 'water', 'ocean', 'sea', 'lake', 'river', 'coast'],
            'green': ['green', 'forest', 'tree', 'trees', 'woods', 'grass', 'park', 'vegetation'],
            'red': ['red', 'brick', 'roof', 'clay'],
            'brown': ['brown', 'soil', 'earth', 'sand', 'desert'],
            'white': ['white', 'snow', 'cloud', 'cloudy'],
            'gray': ['gray', 'grey', 'concrete', 'asphalt'],
        }
        object_lexicon_img = {
            'water_obj': ['water', 'ocean', 'sea', 'lake', 'river', 'harbor', 'ship', 'boat'],
            'vegetation_obj': ['forest', 'tree', 'trees', 'grass', 'field', 'farm', 'vegetation'],
            'urban_obj': ['building', 'buildings', 'road', 'roads', 'bridge', 'airport', 'runway', 'city'],
        }
        dom_to_supported_colors_img = {
            'G>R>B': {'green', 'white', 'gray'},
            'B>R>G': {'blue', 'white', 'gray'},
            'R>G>B': {'red', 'brown', 'white', 'gray'},
            'R≈G>B': {'gray', 'white', 'brown', 'green', 'blue'},
        }

        dom_ch = pixel_stats['dom_channel'].get(cat, 'R≈G>B') if pixel_stats and 'dom_channel' in pixel_stats else 'R≈G>B'
        supported_colors_img = dom_to_supported_colors_img.get(dom_ch, {'white', 'gray'})

        cat_lower = cat.lower()
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
            toks = set(tokenize(cap))

            claimed_colors = set()
            for cname, kws in color_lexicon_img.items():
                hit_count = sum(1 for w in kws if w in toks)
                direct = cname in toks or (cname == 'gray' and 'grey' in toks)
                if hit_count >= 2 or direct:
                    claimed_colors.add(cname)
            vivid_claims = {x for x in claimed_colors if x not in {'white', 'gray'}}
            if vivid_claims:
                img_color_claims += 1
                if vivid_claims.isdisjoint(supported_colors_img):
                    img_color_mismatch += 1

            claimed_groups = []
            for gname, kws in object_lexicon_img.items():
                hit_count = sum(1 for w in kws if w in toks)
                if hit_count >= 2:
                    claimed_groups.append(gname)
            if claimed_groups:
                img_object_claims += 1
                if all(g not in supported_obj_groups for g in claimed_groups):
                    img_object_mismatch += 1

        img_color_rate = (img_color_mismatch / img_color_claims) if img_color_claims else 0.0
        img_object_rate = (img_object_mismatch / img_object_claims) if img_object_claims else 0.0

        prior_meta = contradiction_by_cat.get(cat, {'color_rate': 0.0, 'object_rate': 0.0})
        prior_color_rate = prior_meta['color_rate']
        prior_object_rate = prior_meta['object_rate']

        # weak prior: image-level evidence dominates
        color_rate = 0.85 * img_color_rate + 0.15 * prior_color_rate
        object_rate = 0.85 * img_object_rate + 0.15 * prior_object_rate
        layer_b_score = 0.65 * color_rate + 0.35 * object_rate

        # Layer C (uncertainty)
        token_counts = [len(re.findall(r"\b[a-z]+\b", cap.lower())) for cap in caps]
        avg_tokens = float(np.mean(token_counts)) if token_counts else 0.0
        low_info_penalty = float(np.clip((8.0 - avg_tokens) / 8.0, 0.0, 1.0))
        uncertainty_score = 0.7 * low_info_penalty + 0.3 * min(outlier_count / 3.0, 1.0)

        noise_score = float(np.clip(
            0.58 * layer_a_score + 0.30 * layer_b_score + 0.12 * uncertainty_score,
            0.0,
            1.0,
        ))

        if noise_score >= 0.62 and layer_a_score >= 0.45 and (layer_b_score >= 0.35 or outlier_count >= 2):
            confidence = 'HIGH'
        elif noise_score >= 0.50 and (layer_a_score >= 0.36 or layer_b_score >= 0.25 or outlier_count >= 1):
            confidence = 'MEDIUM'
        else:
            confidence = 'LOW'

        if confidence == 'HIGH' and layer_b_score >= 0.40:
            noise_type = 'cross_modal_mismatch'
        elif confidence in {'HIGH', 'MEDIUM'} and layer_a_score >= 0.42:
            noise_type = 'semantic_drift'
        elif confidence in {'HIGH', 'MEDIUM'} and uncertainty_score >= 0.45:
            noise_type = 'linguistic_low_info'
        else:
            noise_type = 'clean_or_borderline'

        image_noise_rows.append({
            'filename': filename,
            'category': cat,
            'confidence': confidence,
            'noise_type': noise_type,
            'noise_score': noise_score,
            'layer_a_score': layer_a_score,
            'layer_b_score': layer_b_score,
            'layer_c_score': uncertainty_score,
            'mean_pairwise': mean_pair,
            'center_min': center_min,
            'outlier_count': outlier_count,
            'avg_caption_tokens': avg_tokens,
            'color_mismatch_rate': color_rate,
            'object_mismatch_rate': object_rate,
            'img_color_mismatch_rate': img_color_rate,
            'img_object_mismatch_rate': img_object_rate,
            'prior_color_mismatch_rate': prior_color_rate,
            'prior_object_mismatch_rate': prior_object_rate,
            'captions': caps,
        })

        anomaly_probe.append({
            'filename': filename,
            'category': cat,
            'loo_sim': center_min,
            'layer_a': bool((not np.isnan(layer_a_cutoff)) and (center_min < layer_a_cutoff)),
            'layer_b': bool(layer_b_score >= 0.35),
            'layer_c': bool(uncertainty_score >= 0.45),
        })

    image_noise_df = pd.DataFrame(image_noise_rows)
    if not image_noise_df.empty:
        image_noise_df = image_noise_df.sort_values(['noise_score', 'layer_a_score', 'layer_b_score'], ascending=[False, False, False])

        export_df = image_noise_df.copy()
        export_df['captions'] = export_df['captions'].apply(lambda c: ' ||| '.join(c))
        export_df.to_csv(OUTPUT_DIR / f'noisy_samples_strict_{split}.csv', index=False)

        anomalies = []
        for _, r in image_noise_df.iterrows():
            if r['confidence'] == 'LOW':
                continue
            anomalies.append({
                'filename': r['filename'],
                'category': r['category'],
                'caption_idx': '-',
                'caption': ' ||| '.join(r['captions']),
                'confidence': r['confidence'],
                'reason': (
                    f"type={r['noise_type']}; noise={r['noise_score']:.3f}; A={r['layer_a_score']:.3f}; "
                    f"B={r['layer_b_score']:.3f}; C={r['layer_c_score']:.3f}; outliers={int(r['outlier_count'])}"
                ),
            })

        n_high = int((image_noise_df['confidence'] == 'HIGH').sum())
        n_med = int((image_noise_df['confidence'] == 'MEDIUM').sum())
        n_low = int((image_noise_df['confidence'] == 'LOW').sum())
        print(f"\n  Noise ranking complete: {len(image_noise_df)} images")
        print(f"    Confidence counts => HIGH={n_high}, MEDIUM={n_med}, LOW={n_low}")
        if 'noise_type' in image_noise_df.columns:
            print("    Noise type counts:")
            for k, v in image_noise_df['noise_type'].value_counts().to_dict().items():
                print(f"      - {k}: {v}")
        print(f"    Exported: noisy_samples_strict_{split}.csv (all rows)")
        print("    Highest-noise samples (first 10 rows):")
        for _, r in image_noise_df.head(10).iterrows():
            print(f"      - {r['filename']} | {r['category']} | {r['confidence']} | {r['noise_type']} | score={r['noise_score']:.3f}")

    return {
        'variabilities': variabilities,
        'semantic_consistency': semantic_consistency,
        'caption_lengths': caption_lengths_per_img,
        'color_word_freq': top_colors,
        'object_freq': object_freq,
        'spatial_preps': SPATIAL_PREPS,
        'caps_with_color': caps_with_color,
        'caps_with_spatial': caps_with_spatial,
        'total_caps': total_caps,
        'category_similarity_df': category_similarity_df,
        'contradiction_df': contradiction_df,
        'image_noise_df': image_noise_df,
        'anomalies': anomalies,
        'anomaly_probe': anomaly_probe,
        'layer_a_q1': layer_a_q1,
        'layer_a_q3': layer_a_q3,
        'layer_a_iqr': layer_a_iqr,
        'layer_a_cutoff': layer_a_cutoff,
        'iqr_multiplier': iqr_multiplier,
    }


def viz_multimodal(mm_stats, split='train'):
    """Generate multimodal visualizations including real TIF images."""
    print(f"\n  Generating multimodal visualizations...")

    # ── 1. Caption Variability ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    BG = '#FAFAFA'
    fig.patch.set_facecolor(BG)

    ax1 = axes[0]
    ax1.set_facecolor(BG)
    ax1.hist(mm_stats['variabilities'], bins=30, edgecolor='white', alpha=0.8, color='#9b59b6')
    ax1.axvline(np.mean(mm_stats['variabilities']), color='red', linestyle='--', lw=2,
                label=f'Mean: {np.mean(mm_stats["variabilities"]):.2f}')
    ax1.set_xlabel('Caption Length Std (within same image)', fontsize=9)
    ax1.set_ylabel('Frequency', fontsize=9)
    ax1.set_title('Caption Length Variability', fontsize=10, fontweight='bold')
    ax1.legend(fontsize=8)

    ax2 = axes[1]
    ax2.set_facecolor(BG)
    cat_var = defaultdict(list)
    for item in mm_stats['caption_lengths']:
        cat_var[item['category']].append(item['std'])
    cat_means = [(cat, np.mean(stds)) for cat, stds in cat_var.items()]
    cat_means.sort(key=lambda x: x[1], reverse=True)
    cats = [c for c, m in cat_means[:15]]
    means = [m for c, m in cat_means[:15]]
    ax2.barh(cats[::-1], means[::-1], color='#3498db', edgecolor='none', height=0.55)
    ax2.set_xlabel('Mean Caption Length Std', fontsize=9)
    ax2.set_ylabel('')
    ax2.set_title('Caption Variability by Category', fontsize=10, fontweight='bold')
    ax2.tick_params(axis='y', labelsize=7)
    ax2.tick_params(axis='x', labelsize=7)
    for s in ax2.spines.values():
        s.set_visible(False)
    ax2.grid(axis='x', color='#EEEEEE', linewidth=0.5, zorder=0)
    ax2.set_axisbelow(True)

    plt.tight_layout(pad=1)
    plt.savefig(OUTPUT_DIR / f'mm_01_caption_variability_{split}.png', dpi=180,
                bbox_inches='tight', facecolor=BG)
    plt.close()

    # ── 2. Sample Pairs (real TIF images + captions) ──
    samples = mm_stats['caption_lengths'][:3]
    n_samples = len(samples)
    if n_samples == 0:
        raise RuntimeError("[LỖI] Không có caption_lengths — không vẽ được mm_02.")

    # Load thumbnails for all 3 images (bắt buộc đọc được file)
    thumbs = []
    for sample in samples:
        img_path = IMG_DIR / sample['filename']
        if not img_path.is_file():
            raise FileNotFoundError(f"[LỖI] Thiếu ảnh cho mm_02 sample: {img_path}")
        with Image.open(img_path) as img:
            img.thumbnail((320, 320), Image.LANCZOS)
            thumbs.append(np.array(img.convert('RGB')))

    fig_h = 4.0 + n_samples * 1.05
    fig_w = 13
    fig, all_axes = plt.subplots(n_samples, 2,
                                  figsize=(fig_w, fig_h),
                                  gridspec_kw={'width_ratios': [1.0, 2.2], 'wspace': 0.3})
    fig.patch.set_facecolor(BG)

    if n_samples == 1:
        all_axes = all_axes.reshape(1, -1)

    for row_idx, (sample, ax_row) in enumerate(zip(samples, all_axes)):
        ax_img, ax_txt = ax_row

        # ── Left: satellite image ──
        ax_img.set_facecolor('#1a1a1a')
        ax_img.axis('off')
        thumb = thumbs[row_idx]
        ax_img.imshow(thumb)
        ax_img.set_title(f"[{chr(65+row_idx)}] {sample['filename']}\n({sample['category']})",
                         fontsize=9, fontweight='bold', color='white',
                         pad=4, backgroundcolor='#1a1a1a')

        # ── Right: 5 captions ──
        CAP_COLORS = ['#B42318', '#2980b9', '#27ae60', '#e67e22', '#8e44ad']
        ax_txt.set_facecolor(BG)
        ax_txt.set_xlim(0, 1)
        ax_txt.set_ylim(0, 1)
        ax_txt.axis('off')

        ax_txt.text(0.5, 0.97, f"Caption set [{chr(65+row_idx)}]",
                    transform=ax_txt.transAxes,
                    fontsize=9, fontweight='bold',
                    ha='center', va='top', color='#2C3E50')

        for ci, cap in enumerate(sample['captions'][:5]):
            y = 0.84 - ci * 0.16
            badge = f"[{ci+1}]"
            ax_txt.text(0.01, y, badge,
                        transform=ax_txt.transAxes,
                        fontsize=8, fontweight='bold',
                        color=CAP_COLORS[ci % len(CAP_COLORS)],
                        va='top')
            ax_txt.text(0.065, y, cap[:110] + ('…' if len(cap) > 110 else ''),
                        transform=ax_txt.transAxes,
                        fontsize=8, color='#333', va='top', wrap=True)

        ax_txt.text(0.5, 0.01, f"category: {sample['category']}",
                    transform=ax_txt.transAxes,
                    fontsize=7, style='italic', color='#888',
                    ha='center', va='bottom')

    plt.suptitle(f'RSITMD Sample Image-Caption Pairs ({split.upper()})',
                 fontsize=13, fontweight='bold', y=1.01, color='#2C3E50')
    plt.tight_layout(pad=0.8, rect=[0, 0, 1, 0.99])
    plt.savefig(OUTPUT_DIR / f'mm_02_sample_pairs_{split}.png',
                dpi=180, bbox_inches='tight', facecolor=BG)
    plt.close()

    # ── 3. Category-level text cosine similarity matrix ──
    sim_df = mm_stats.get('category_similarity_df', pd.DataFrame())
    if not sim_df.empty:
        plot_df = sim_df
        if sim_df.shape[0] > 20:
            top_cats = sim_df.mean(axis=1).sort_values(ascending=False).head(20).index
            plot_df = sim_df.loc[top_cats, top_cats]

        fig_s, ax_s = plt.subplots(figsize=(10, 8))
        fig_s.patch.set_facecolor(BG)
        sns.heatmap(plot_df, cmap='YlGnBu', vmin=0, vmax=1, ax=ax_s)
        ax_s.set_title(f'Category-level Caption Cosine Similarity ({split.upper()})')
        ax_s.set_xlabel('Category')
        ax_s.set_ylabel('Category')
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f'mm_03_category_cosine_similarity_{split}.png', dpi=180, bbox_inches='tight')
        plt.close()

    # ── 4. Intra-image semantic consistency distribution ──
    sem = mm_stats.get('semantic_consistency', [])
    if sem:
        fig_sem, ax_sem = plt.subplots(figsize=(8, 4.8))
        fig_sem.patch.set_facecolor(BG)
        ax_sem.hist(sem, bins=30, color='#16a085', alpha=0.85, edgecolor='white')
        ax_sem.axvline(np.mean(sem), color='#B42318', linestyle='--', linewidth=2, label=f"Mean={np.mean(sem):.3f}")
        ax_sem.set_title(f'Intra-image Caption Semantic Consistency ({split.upper()})')
        ax_sem.set_xlabel('Mean pairwise cosine similarity (5 captions/image)')
        ax_sem.set_ylabel('Image count')
        ax_sem.legend()
        ax_sem.grid(alpha=0.2)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f'mm_05_semantic_consistency_{split}.png', dpi=180, bbox_inches='tight')
        plt.close()

    # ── 5. Cross-modal contradiction map ──
    contradiction_df = mm_stats.get('contradiction_df', pd.DataFrame())
    if not contradiction_df.empty:
        top_cd = contradiction_df.copy()
        if 'combined_mismatch_rate' not in top_cd.columns:
            top_cd['combined_mismatch_rate'] = 0.0
        top_cd = top_cd.sort_values('combined_mismatch_rate', ascending=False).head(20)

        heat_cols = [c for c in ['color_mismatch_rate', 'object_mismatch_rate', 'combined_mismatch_rate'] if c in top_cd.columns]
        heat_df = top_cd.set_index('category')[heat_cols]

        fig_c, ax_c = plt.subplots(figsize=(9.0, 7.0))
        fig_c.patch.set_facecolor(BG)
        sns.heatmap(heat_df, annot=True, fmt='.2f', cmap='OrRd', vmin=0, vmax=1, ax=ax_c)
        ax_c.set_title(f'Cross-modal Contradiction Map ({split.upper()})')
        ax_c.set_xlabel('Mismatch type')
        ax_c.set_ylabel('Category')
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f'mm_06_contradiction_map_{split}.png', dpi=180, bbox_inches='tight')
        plt.close()

    print(f"  Generated mm_01_caption_variability_{split}.png, mm_02_sample_pairs_{split}.png, "
          f"mm_03_category_cosine_similarity_{split}.png, mm_05_semantic_consistency_{split}.png, "
          f"mm_06_contradiction_map_{split}.png")


def generate_summary_csv(train_text, test_text, train_img, test_img):
    """Save concise train/test summary statistics CSV."""

    def pack(text_stats, img_stats):
        return {
            'Total Images': img_stats['num_imgs'],
            'Total Captions': len(text_stats['captions']),
            'Captions per Image': round(len(text_stats['captions']) / max(img_stats['num_imgs'], 1), 2),
            'Categories': len(img_stats['cat_counts']),
            'Total Words': len(text_stats['raw_words']),
            'Vocabulary (raw)': len(set(text_stats['raw_words'])),
            'Vocabulary (clean)': len(set(text_stats['clean_words'])),
            'Avg Words/Caption': round(np.mean(text_stats['word_counts']), 2),
            'Median Words/Caption': round(np.median(text_stats['word_counts']), 2),
            'Std Words/Caption': round(np.std(text_stats['word_counts']), 2),
            'Min Words/Caption': int(min(text_stats['word_counts'])),
            'Max Words/Caption': int(max(text_stats['word_counts'])),
        }

    train_pack = pack(train_text, train_img)
    test_pack = pack(test_text, test_img)

    rows = []
    for metric in train_pack.keys():
        train_val = train_pack[metric]
        test_val = test_pack[metric]
        if isinstance(train_val, (int, float)) and isinstance(test_val, (int, float)) and train_val != 0:
            drift_pct = round((test_val - train_val) / train_val * 100, 2)
        else:
            drift_pct = np.nan
        rows.append({
            'Metric': metric,
            'Train': train_val,
            'Test': test_val,
            'Test_vs_Train_%': drift_pct,
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / 'summary_stats_train_test.csv', index=False)


def viz_train_test_drift(train_text, test_text, train_pixel, test_pixel):
    """Visualize key train-vs-test drift for text and image quality metrics."""
    train_img_df = train_pixel.get('image_df', pd.DataFrame())
    test_img_df = test_pixel.get('image_df', pd.DataFrame())
    if train_img_df.empty or test_img_df.empty:
        raise RuntimeError(
            "[LỖI] drift_01 cần image_df đầy đủ train/test — kiểm tra ảnh TIF và compute_image_level_stats."
        )

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.patch.set_facecolor('#FAFAFA')

    axs[0].hist(train_text['word_counts'], bins=30, alpha=0.55, label='Train', color='#1f77b4')
    axs[0].hist(test_text['word_counts'], bins=30, alpha=0.55, label='Test', color='#ff7f0e')
    axs[0].set_title('Caption Length Drift')
    axs[0].set_xlabel('Words per caption')
    axs[0].legend()

    axs[1].hist(train_img_df['brightness'], bins=25, alpha=0.55, label='Train', color='#2ca02c')
    axs[1].hist(test_img_df['brightness'], bins=25, alpha=0.55, label='Test', color='#d62728')
    axs[1].set_title('Brightness Drift')
    axs[1].set_xlabel('Mean brightness')
    axs[1].legend()

    axs[2].hist(train_img_df['blur_score'], bins=25, alpha=0.55, label='Train', color='#9467bd')
    axs[2].hist(test_img_df['blur_score'], bins=25, alpha=0.55, label='Test', color='#8c564b')
    axs[2].set_title('Blur Score Drift')
    axs[2].set_xlabel('Laplacian variance')
    axs[2].legend()

    for ax in axs:
        ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'drift_01_train_test_core_metrics.png', dpi=180, bbox_inches='tight')
    plt.close()


def viz_noise_threshold_sensitivity(train_mm, iqr_multipliers=(1.0, 1.25, 1.5, 1.75, 2.0)):
    """Plot anomaly count vs Tukey IQR multiplier for Layer A (uses cached probes + Q1/IQR from train_mm)."""
    probes = train_mm.get('anomaly_probe', [])
    if not probes:
        raise RuntimeError("[LỖI] mm_04_threshold_sensitivity cần anomaly_probe từ analyze_multimodal.")

    q1 = train_mm.get('layer_a_q1', np.nan)
    q3 = train_mm.get('layer_a_q3', np.nan)
    iqr = train_mm.get('layer_a_iqr', np.nan)
    if np.isnan(q1) or np.isnan(iqr) or iqr <= 0:
        raise RuntimeError(
            "[LỖI] mm_04 cần layer_a_q1 / layer_a_iqr hợp lệ (IQR > 0). "
            "Có thể toàn bộ điểm LOO giống nhau hoặc không đủ caption."
        )

    counts = []
    for k in iqr_multipliers:
        cutoff = q1 - k * iqr
        cnt = 0
        for p in probes:
            loo = p.get('loo_sim', np.nan)
            layer_b = bool(p.get('layer_b', False))
            layer_a = (not np.isnan(loo)) and (loo < cutoff)
            if layer_a or layer_b:
                cnt += 1
        counts.append(cnt)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(list(iqr_multipliers), counts, marker='o', color='#B42318', linewidth=2)
    for x, y in zip(iqr_multipliers, counts):
        ax.text(x, y, f' {y}', va='bottom', fontsize=8)
    ax.set_title('Anomaly Count Sensitivity vs Layer-A IQR Multiplier (Tukey fence)')
    ax.set_xlabel('k in cutoff = Q1 − k × IQR')
    ax.set_ylabel('Detected anomalies (Layer A or B)')
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'mm_04_threshold_sensitivity_train.png', dpi=180, bbox_inches='tight')
    plt.close()


def print_executive_summary(train_text, test_text, train_pixel, test_pixel, train_mm, test_mm):
    """Print concise findings and implications after EDA run."""
    t_mean = np.mean(train_text['word_counts'])
    v_mean = np.mean(train_mm['variabilities'])
    an_train = len(train_mm.get('anomalies', []))
    an_test = len(test_mm.get('anomalies', []))

    train_img_df = train_pixel.get('image_df', pd.DataFrame())
    test_img_df = test_pixel.get('image_df', pd.DataFrame())
    b_train = float(train_img_df['brightness'].mean()) if not train_img_df.empty else np.nan
    b_test = float(test_img_df['brightness'].mean()) if not test_img_df.empty else np.nan

    print(f"\n{'='*60}")
    print("EXECUTIVE SUMMARY — KEY FINDINGS")
    print(f"{'='*60}")
    print(f"1) Avg caption length (train): {t_mean:.2f} words")
    print(f"2) Caption variability mean (train): {v_mean:.2f} std words")
    print(f"3) Brightness mean train/test: {b_train:.3f} / {b_test:.3f}")
    print(f"4) Noise flags train/test: {an_train} / {an_test}")
    print("5) Core drift and threshold sensitivity figures generated for model risk review")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    main()
