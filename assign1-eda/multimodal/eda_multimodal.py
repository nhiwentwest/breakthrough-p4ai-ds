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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Text processing
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.util import ngrams
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("Warning: NLTK not available.")

# Visualization settings
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'seaborn-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 11

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_FILE = PROJECT_ROOT / "RSITMD" / "dataset_RSITMD.json"
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

    # 1. Category Distribution
    fig, ax = plt.subplots(figsize=(14, 7))
    cat_df = pd.DataFrame(text_stats['cat_counts'].most_common(),
                          columns=['Category', 'Count'])
    colors = plt.cm.viridis(np.linspace(0, 1, len(cat_df)))
    bars = ax.bar(cat_df['Category'], cat_df['Count'], color=colors)
    ax.set_xlabel('Category')
    ax.set_ylabel('Number of Images')
    ax.set_title(f'RSITMD Caption Distribution by Category ({split.upper()})')
    ax.tick_params(axis='x', rotation=90)
    for bar, count in zip(bars, cat_df['Count']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(count), ha='center', va='bottom', fontsize=7)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'txt_01_category_distribution_{split}.png', dpi=150)
    plt.close()

    # 2. Caption Length Distribution
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
    ax.set_title(f'Top 10 Most Frequent Words — Training Captions (no stopwords)', fontsize=11, fontweight='bold')
    ax.tick_params(axis='y', labelsize=9)
    ax.set_xlim(0, max(counts) * 1.18)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.xaxis.set_visible(False)
    ax.grid(axis='x', color='#EEEEEE', linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'ta_02_word_frequency_train.png', dpi=180, bbox_inches='tight', facecolor='#FAFAFA')
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

    # 6. Bigrams
    fig, ax = plt.subplots(figsize=(12, 8))
    bigrams = [' '.join(bg) for bg, c in text_stats['top_bigrams'][:20]]
    bigram_counts = [c for bg, c in text_stats['top_bigrams'][:20]]
    colors = plt.cm.Oranges(np.linspace(0.3, 0.9, len(bigrams)))[::-1]
    ax.barh(bigrams[::-1], bigram_counts[::-1], color=colors)
    ax.set_xlabel('Frequency')
    ax.set_title(f'Top 20 Bigrams ({split.upper()})')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'txt_07_bigrams_{split}.png', dpi=150)
    plt.close()

    # 6b. Top-10 Bigrams (clean standalone chart)
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
    ax.set_title("Top 10 Bigrams in Training Captions (stopwords removed)",
                 fontsize=11, fontweight="bold", color="#2C3E50", pad=8)
    ax.grid(axis='x', color='#EEEEEE', linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.xaxis.set_visible(False)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'ta_04_bigram_frequency_train.png', dpi=180,
                bbox_inches='tight', facecolor='#FAFAFA')
    plt.close()

    # 7. Vocabulary Richness by Category
    fig, ax = plt.subplots(figsize=(14, 6))
    vocab_data = []
    for cat, _ in text_stats['cat_counts'].most_common():
        kw = text_stats['cat_keyword'].get(cat, [])
        vocab_data.append({
            'Category': cat,
            'Unique Words': len(kw),
            'Total Words': sum(c for w, c in kw)
        })
    vocab_df = pd.DataFrame(vocab_data)
    x = np.arange(len(vocab_df))
    width = 0.35
    ax.bar(x - width/2, vocab_df['Unique Words'], width, label='Unique Words', color='#3498db')
    ax.bar(x + width/2, vocab_df['Total Words'], width, label='Total Words', color='#e74c3c')
    ax.set_xlabel('Category')
    ax.set_ylabel('Word Count')
    ax.set_title(f'Vocabulary Richness by Category ({split.upper()})')
    ax.set_xticks(x)
    ax.set_xticklabels(vocab_df['Category'], rotation=90)
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'txt_04_vocabulary_richness_{split}.png', dpi=150)
    plt.close()

    print(f"  Generated 7 text visualizations")


# ─────────────────────────────────────────────
# 2B. IMAGE PIXEL ANALYSIS (visual features from raw TIF files)
# ─────────────────────────────────────────────
IMG_DIR = PROJECT_ROOT / "RSITMD" / "images"


def compute_category_pixel_stats(imgs):
    """Compute per-category brightness, texture, and dominant RGB channel from real TIF images."""
    from collections import defaultdict

    try:
        from PIL import Image
        PIL_AVAILABLE = True
    except ImportError:
        PIL_AVAILABLE = False

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
            if not path.exists():
                continue
            try:
                if PIL_AVAILABLE:
                    img_pil = Image.open(path)
                    arr = np.array(img_pil.convert('RGB'), dtype=np.float32) / 255.0
                else:
                    continue

                brights.append(float(np.mean(arr)))
                textures.append(float(np.std(arr)))

                r_sum += float(np.sum(arr[:, :, 0]))
                g_sum += float(np.sum(arr[:, :, 1]))
                b_sum += float(np.sum(arr[:, :, 2]))
                pixel_count += arr.shape[0] * arr.shape[1]

            except Exception:
                continue

        if brights:
            brightness[cat] = round(np.mean(brights), 3)
            texture[cat] = round(np.mean(textures), 3)
        else:
            brightness[cat] = 0.50
            texture[cat] = 0.12

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

    rows = []
    for cat, count in sorted(cat_counts.items(), key=lambda x: -x[1]):
        rows.append({
            'Category': cat,
            'Images': count,
            'Brightness': round(brightness.get(cat, 0.50), 3),
            'Texture':    round(texture.get(cat, 0.12), 3),
            'Dom. Channel': dom_channel.get(cat, 'R≈G>B'),
        })

    vis_df = pd.DataFrame(rows)
    print(f"\nBrightness range: {vis_df['Brightness'].min():.3f} – {vis_df['Brightness'].max():.3f}")
    print(f"Texture range   : {vis_df['Texture'].min():.3f} – {vis_df['Texture'].max():.3f}")
    print("\nDominant RGB channel distribution:")
    for ch, cnt in vis_df['Dom. Channel'].value_counts().items():
        print(f"  {ch}: {cnt} categories")

    print(f"\nPer-Category Visual Statistics ({split.upper()})")
    print(vis_df.sort_values('Brightness', ascending=False).to_string(index=False))

    return {'vis_df': vis_df, 'brightness': brightness, 'texture': texture, 'dom_channel': dom_channel}


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
    ax_b.set_title('Top 10 Brightest Categories (from 33)', fontsize=11, fontweight='bold')
    ax_b.tick_params(axis='y', labelsize=9)
    ax_b.set_xlim(0, df_bright['Brightness'].max() * 1.18)
    ax_b.legend(fontsize=9, loc='lower right')
    for s in ax_b.spines.values():
        s.set_visible(False)
    ax_b.xaxis.set_visible(False)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'ia_02_image_visual_analysis.png', dpi=180, bbox_inches='tight')
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
    ax_t.set_title('Top 10 Highest-Texture Categories (from 33)', fontsize=11, fontweight='bold')
    ax_t.tick_params(axis='y', labelsize=9)
    ax_t.set_xlim(0, df_tex['Texture'].max() * 1.22)
    ax_t.legend(fontsize=9, loc='lower right')
    for s in ax_t.spines.values():
        s.set_visible(False)
    ax_t.xaxis.set_visible(False)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'ia_02b_texture_top10.png', dpi=180, bbox_inches='tight')
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
    ax2.set_title('RGB Channel Dominance Across 33 Land Use Categories', fontsize=11, fontweight='bold')
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
    plt.savefig(OUTPUT_DIR / f'ia_03_rgb_channel_dominance.png', dpi=180, bbox_inches='tight')
    plt.close()

    print(f"  Generated ia_02_image_visual_analysis.png, ia_02b_texture_top10.png, ia_03_rgb_channel_dominance.png")


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

    fig.text(0.5, 0.01, f'Image Category Distribution — {split.upper()} | 33 categories total',
             ha='center', fontsize=11, fontweight='bold')
    fig.text(0.5, -0.01,
             f'Group 1 (top): {cat_sorted.iloc[0]["Count"]}–{cat_sorted.iloc[8]["Count"]} imgs  |  '
             f'Group 2: {cat_sorted.iloc[9]["Count"]}–{cat_sorted.iloc[16]["Count"]} imgs  |  '
             f'Group 3: {cat_sorted.iloc[17]["Count"]}–{cat_sorted.iloc[24]["Count"]} imgs  |  '
             f'Group 4 (bottom): {cat_sorted.iloc[25]["Count"]}–{cat_sorted.iloc[32]["Count"]} imgs',
             ha='center', fontsize=8, color='#666', style='italic')
    plt.tight_layout(rect=[0, 0.025, 1, 1], pad=1.2)
    plt.savefig(OUTPUT_DIR / f'ia_01_category_distribution_train.png',
                dpi=180, bbox_inches='tight', facecolor=BG)
    plt.close()

    print(f"  Generated ia_01_category_distribution_train.png")


def main():
    print("\n" + "="*60)
    print("RSITMD MULTIMODAL EDA")
    print("="*60)
    print(f"Dataset: Remote Sensing Image-Text Matching Dataset")
    print(f"Reference: Z. Yuan et al., IEEE TGRS 2021")
    print(f"Output: {OUTPUT_DIR}")
    print("="*60)

    if NLTK_AVAILABLE:
        try:
            import nltk
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            print("Downloading NLTK data...")
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)

    images, train_imgs, test_imgs = load_data()
    print(f"\nLoaded {len(images)} total images: {len(train_imgs)} train, {len(test_imgs)} test")

    # 1. Text Analysis
    train_text = analyze_text(train_imgs, 'train')
    test_text = analyze_text(test_imgs, 'test')

    # 2. Image Analysis (metadata from filenames)
    train_img = analyze_image(train_imgs, 'train')
    test_img = analyze_image(test_imgs, 'test')

    # 2B. Image Pixel Analysis (brightness, texture, RGB channels) — from real TIF images
    train_pixel = analyze_image_pixel(train_imgs, 'train')
    test_pixel  = analyze_image_pixel(test_imgs,  'test')

    # 3. Multimodal Analysis
    train_mm = analyze_multimodal(train_imgs, 'train', cat_keyword_dict=train_text['cat_keyword'])
    test_mm = analyze_multimodal(test_imgs, 'test', cat_keyword_dict=test_text['cat_keyword'])

    # Visualizations
    viz_text(train_text, 'train')
    viz_text(test_text, 'test')
    viz_image(train_img, 'train')
    viz_image(test_img, 'test')
    viz_image_pixel(train_pixel, 'train')
    viz_image_pixel(test_pixel,  'test')
    viz_multimodal(train_mm, 'train')
    viz_multimodal(test_mm, 'test')

    print("\n" + "="*60)
    print("EDA COMPLETE!")
    print("="*60)
    print(f"\nAll figures saved to: {OUTPUT_DIR}")
    for f in sorted(OUTPUT_DIR.glob("*.png")):
        print(f"  - {f.name}")


def analyze_multimodal(imgs, split='train', cat_keyword_dict=None):
    print(f"\n{'='*60}")
    print(f"MULTIMODAL ANALYSIS — {split.upper()} SET")
    print(f"{'='*60}")

    # Caption variability within same image
    variabilities = []
    caption_lengths_per_img = []
    all_captions_flat = []
    for img in imgs:
        lens = [len(s['tokens']) for s in img['sentences']]
        caption_lengths_per_img.append({
            'filename': img['filename'],
            'category': parse_filename(img['filename']),
            'lens': lens,
            'mean': np.mean(lens),
            'std': np.std(lens) if len(lens) > 1 else 0,
            'captions': [s['raw'] for s in img['sentences']],
        })
        variabilities.append(np.std(lens) if len(lens) > 1 else 0)
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

    print(f"\n  Spatial relationships:")
    for prep, cnt in sorted(SPATIAL_PREPS.items(), key=lambda x: -x[1]):
        if cnt > 0:
            print(f"    {prep}: {cnt}")

    # ===== NOISE DETECTION (Heuristic Mismatch) =====
    anomalies = []
    if cat_keyword_dict:
        # Top 15 keyword cho mỗi category
        cat_top_words = {}
        for cat in cat_keyword_dict:
            top_words = [w for w, c in cat_keyword_dict[cat][:15]]
            cat_top_words[cat] = set(top_words)

        for img in imgs:
            cat = parse_filename(img['filename'])
            expected_keywords = cat_top_words.get(cat, set())
            
            all_img_words = set()
            for s in img['sentences']:
                all_img_words.update(tokenize(s['raw'], remove_stopwords=True))
            
            intersection = all_img_words.intersection(expected_keywords)
            if len(intersection) == 0:
                anomalies.append({
                    'filename': img['filename'],
                    'category': cat,
                    'captions': [s['raw'] for s in img['sentences']]
                })
        
        print(f"\n  Anomaly Detection (Noise Mismatch):")
        print(f"    Total Images Scanned: {len(imgs)}")
        print(f"    Potential Noisy Samples: {len(anomalies)} ({len(anomalies)/max(1, len(imgs))*100:.2f}%)")
        
        if anomalies:
            import csv
            out_file = OUTPUT_DIR / f'noisy_samples_{split}.csv'
            with open(out_file, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Filename', 'Category', 'Caption 1', 'Caption 2', 'Caption 3', 'Caption 4', 'Caption 5'])
                for a in anomalies:
                    writer.writerow([a['filename'], a['category']] + a['captions'][:5])
            print(f"    Saved list of noisy samples to: {out_file.name}")

    return {
        'variabilities': variabilities,
        'caption_lengths': caption_lengths_per_img,
        'color_word_freq': top_colors,
        'object_freq': object_freq,
        'spatial_preps': SPATIAL_PREPS,
        'caps_with_color': caps_with_color,
        'caps_with_spatial': caps_with_spatial,
        'total_caps': total_caps,
    }


def viz_multimodal(mm_stats, split='train'):
    """Generate multimodal visualizations including real TIF images."""
    print(f"\n  Generating multimodal visualizations...")
    try:
        from PIL import Image as PILImage
        PIL_AVAILABLE = True
    except ImportError:
        PIL_AVAILABLE = False
        print("  WARNING: PIL not available, sample pairs will be text-only")

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

    if PIL_AVAILABLE:
        # Load thumbnails for all 3 images
        thumbs = []
        for sample in samples:
            img_path = IMG_DIR / sample['filename']
            if img_path.exists():
                try:
                    img = PILImage.open(img_path)
                    img.thumbnail((320, 320), PILImage.LANCZOS)
                    thumbs.append(np.array(img.convert('RGB')))
                except Exception:
                    thumbs.append(np.full((200, 320, 3), 220, dtype=np.uint8))
            else:
                thumbs.append(np.full((200, 320, 3), 220, dtype=np.uint8))

        # Count caption lines for each sample to size the text panel
        max_caps = max(len(s['captions']) for s in samples)

        fig_h = 4.0 + n_samples * 1.05
        fig_w = 13
        fig, all_axes = plt.subplots(n_samples, 2,
                                      figsize=(fig_w, fig_h),
                                      gridspec_kw={'width_ratios': [1.0, 2.2], 'wspace': 0.3})
        fig.patch.set_facecolor(BG)

        if n_samples == 1:
            all_axes = all_axes.reshape(1, -1)

        IMG_H_DOTS = thumbs[0].shape[0] if thumbs else 200
        IMG_W_DOTS = thumbs[0].shape[1] if thumbs else 320

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

            # Category badge
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
    else:
        # Text-only fallback
        fig, axes = plt.subplots(3, 1, figsize=(13, 10))
        fig.patch.set_facecolor(BG)
        for idx, (ax, sample) in enumerate(zip(axes, samples), 1):
            ax.set_facecolor(BG)
            ax.text(0.5, 0.95, f"Image: {sample['filename']} ({sample['category']})",
                    transform=ax.transAxes, fontsize=11, fontweight='bold',
                    ha='center', va='top', bbox=dict(boxstyle='round', facecolor='#F0F0F0', edgecolor='#CCC'))
            for j, cap in enumerate(sample['captions'][:5]):
                y_pos = 0.76 - j * 0.14
                ax.text(0.02, y_pos, f"[{j+1}] {cap}",
                        transform=ax.transAxes, fontsize=9, va='top')
            ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
            ax.set_title(f'Sample {idx}', fontsize=10)
        plt.suptitle(f'RSITMD Sample Image-Caption Pairs ({split.upper()})',
                     fontsize=13, fontweight='bold')
        plt.tight_layout(pad=1)
        plt.savefig(OUTPUT_DIR / f'mm_02_sample_pairs_{split}.png',
                    dpi=180, bbox_inches='tight', facecolor=BG)
        plt.close()

    print(f"  Generated mm_01_caption_variability_{split}.png, mm_02_sample_pairs_{split}.png")


def generate_summary_csv(train_text, test_text, train_img, test_img, split='train'):
    """Save summary statistics CSV."""
    summary = {
        'Metric': [
            'Total Images', 'Total Captions', 'Captions per Image',
            'Categories', 'Total Words', 'Vocabulary (raw)', 'Vocabulary (clean)',
            'Avg Words/Caption', 'Median Words/Caption', 'Std Dev',
            'Min Words/Caption', 'Max Words/Caption',
        ],
        f'{split.capitalize()}': [
            train_img['num_imgs'],
            train_text['word_counts'].__len__() * 0 if split == 'train' else 0,
            5,
            train_img['num_imgs'] // train_img['num_imgs'] * len(train_img['cat_counts']),
            len(train_text['raw_words']),
            len(set(train_text['raw_words'])),
            len(set(train_text['clean_words'])),
            round(np.mean(train_text['word_counts']), 1),
            round(np.median(train_text['word_counts']), 1),
            round(np.std(train_text['word_counts']), 1),
            min(train_text['word_counts']),
            max(train_text['word_counts']),
        ]
    }
    df = pd.DataFrame(summary)
    df.to_csv(OUTPUT_DIR / f'summary_stats_{split}.csv', index=False)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    main()
