#!/usr/bin/env python3
"""
Strict standalone noisy-sample detector for RSITMD captions.

Design goals:
- Reuse ideas from `eda_multimodal.py` (semantic + contradiction map)
- Stricter ranking so top rows are high-confidence mismatches
- Export transparent evidence columns for manual review
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


STOP_WORDS = {
    'i','me','my','myself','we','our','ours','ourselves','you','your','yours','yourself','yourselves','he','him','his','himself',
    'she','her','hers','herself','it','its','itself','they','them','their','theirs','themselves','what','which','who','whom',
    'this','that','these','those','am','is','are','was','were','be','been','being','have','has','had','having','do','does','did',
    'doing','a','an','the','and','but','if','or','because','as','until','while','of','at','by','for','with','about','against','between',
    'into','through','during','before','after','above','below','to','from','up','down','in','out','on','off','over','under','again',
    'further','then','once','here','there','when','where','why','how','all','each','few','more','most','other','some','such','no','nor',
    'not','only','own','same','so','than','too','very','can','will','just','should','now','one','also'
}

COLOR_LEXICON = {
    'blue': ['water', 'blue', 'ocean', 'sea', 'lake', 'river', 'coast'],
    'green': ['green', 'forest', 'tree', 'trees', 'woods', 'grass', 'park', 'vegetation'],
    'red': ['red', 'brick', 'roof', 'clay'],
    'brown': ['brown', 'soil', 'earth', 'sand', 'desert'],
    'white': ['white', 'snow', 'cloud', 'cloudy'],
    'gray': ['gray', 'grey', 'concrete', 'asphalt'],
}

OBJECT_LEXICON = {
    'water_obj': ['water', 'ocean', 'sea', 'lake', 'river', 'harbor', 'ship', 'boat'],
    'vegetation_obj': ['forest', 'tree', 'trees', 'grass', 'field', 'farm', 'vegetation'],
    'urban_obj': ['building', 'buildings', 'road', 'roads', 'bridge', 'airport', 'runway', 'city'],
}

DOM_TO_SUPPORTED_COLORS = {
    'G>R>B': {'green', 'white', 'gray'},
    'B>R>G': {'blue', 'white', 'gray'},
    'R>G>B': {'red', 'brown', 'white', 'gray'},
    'R≈G>B': {'gray', 'white', 'brown', 'green', 'blue'},
}


def parse_category(filename: str) -> str:
    parts = filename.replace('.tif', '').rsplit('_', 1)
    return parts[0] if len(parts) == 2 else 'unknown'


def tokenize(text: str, remove_stopwords: bool = False) -> list[str]:
    words = re.findall(r"\b[a-z]+\b", text.lower())
    if remove_stopwords:
        return [w for w in words if w not in STOP_WORDS]
    return words


def resolve_data_paths() -> tuple[Path, Path]:
    candidates = [
        Path('/Users/nhi/Documents/school/252/p4/btl/RSITMD/dataset_RSITMD.json'),
        Path(__file__).resolve().parents[2] / 'RSITMD' / 'dataset_RSITMD.json',
        Path(__file__).resolve().parents[3] / 'RSITMD' / 'dataset_RSITMD.json',
    ]
    for c in candidates:
        if c.exists():
            return c, c.parent / 'images'
    raise FileNotFoundError('Cannot find dataset_RSITMD.json from known locations.')


def load_split(split: str) -> tuple[list[dict], Path]:
    data_file, img_dir = resolve_data_paths()
    with open(data_file, encoding='utf-8') as f:
        data = json.load(f)
    imgs = [x for x in data['images'] if x.get('split') == split]
    return imgs, img_dir


def category_dom_channels(imgs: list[dict], img_dir: Path, per_cat_limit: int = 25) -> dict[str, str]:
    cat_files: dict[str, list[str]] = defaultdict(list)
    for img in imgs:
        cat_files[parse_category(img['filename'])].append(img['filename'])

    dom_map = {}
    for cat, files in cat_files.items():
        r_sum = g_sum = b_sum = 0.0
        px = 0
        for fname in files[:per_cat_limit]:
            path = img_dir / fname
            if not path.exists():
                continue
            try:
                with Image.open(path) as im:
                    arr = np.array(im.convert('RGB'), dtype=np.float32) / 255.0
                r_sum += float(np.sum(arr[:, :, 0]))
                g_sum += float(np.sum(arr[:, :, 1]))
                b_sum += float(np.sum(arr[:, :, 2]))
                px += arr.shape[0] * arr.shape[1]
            except Exception:
                continue

        if px == 0:
            dom_map[cat] = 'R≈G>B'
            continue

        r_avg, g_avg, b_avg = r_sum / px, g_sum / px, b_sum / px
        if g_avg > r_avg and g_avg > b_avg:
            dom_map[cat] = 'G>R>B'
        elif b_avg > r_avg and b_avg > g_avg:
            dom_map[cat] = 'B>R>G'
        elif r_avg > b_avg:
            dom_map[cat] = 'R>G>B'
        else:
            dom_map[cat] = 'R≈G>B'
    return dom_map


def infer_supported_object_groups(cat_name: str) -> set[str]:
    c = cat_name.lower()
    groups = set()
    if any(k in c for k in ['river', 'pond', 'port', 'boat', 'harbor', 'coast', 'beach', 'sea', 'lake']):
        groups.add('water_obj')
    if any(k in c for k in ['forest', 'meadow', 'park', 'farmland', 'baseballfield', 'playground', 'grass', 'bareland', 'desert']):
        groups.add('vegetation_obj')
    if any(k in c for k in ['airport', 'plane', 'runway', 'road', 'bridge', 'residential', 'industrial', 'church', 'school', 'stadium', 'square', 'center', 'railway', 'parking', 'building', 'viaduct']):
        groups.add('urban_obj')
    if not groups:
        groups = {'water_obj', 'vegetation_obj', 'urban_obj'}
    return groups


def semantic_metrics(captions: list[str]) -> tuple[float, float, int, float]:
    if len(captions) < 2:
        return np.nan, np.nan, 0, np.nan

    vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=1, lowercase=True)
    mat = vec.fit_transform(captions)
    sim = cosine_similarity(mat)

    iu = np.triu_indices_from(sim, k=1)
    pair_vals = sim[iu]
    mean_pair = float(np.mean(pair_vals)) if len(pair_vals) else np.nan
    std_pair = float(np.std(pair_vals)) if len(pair_vals) else np.nan

    center = np.asarray(mat.mean(axis=0))
    center_sims = cosine_similarity(mat, center).reshape(-1)
    center_min = float(np.min(center_sims)) if len(center_sims) else np.nan

    q1 = float(np.percentile(center_sims, 25))
    q3 = float(np.percentile(center_sims, 75))
    iqr = q3 - q1
    low_cut = q1 - 1.5 * iqr
    outlier_count = int(np.sum(center_sims < low_cut)) if iqr > 0 else 0

    return mean_pair, std_pair, outlier_count, center_min


def contradiction_metrics(captions: list[str], category: str, dom_ch: str) -> tuple[float, float, int, int]:
    supported_colors = DOM_TO_SUPPORTED_COLORS.get(dom_ch, {'white', 'gray'})
    supported_objs = infer_supported_object_groups(category)

    color_claims = color_mismatch = 0
    object_claims = object_mismatch = 0

    for cap in captions:
        toks = set(tokenize(cap))

        claimed_colors = set()
        for name, kws in COLOR_LEXICON.items():
            hits = sum(1 for w in kws if w in toks)
            direct = name in toks or (name == 'gray' and 'grey' in toks)
            if hits >= 2 or direct:
                claimed_colors.add(name)

        vivid_claims = {x for x in claimed_colors if x not in {'white', 'gray'}}
        if vivid_claims:
            color_claims += 1
            if vivid_claims.isdisjoint(supported_colors):
                color_mismatch += 1

        claimed_groups = []
        for gname, kws in OBJECT_LEXICON.items():
            hits = sum(1 for w in kws if w in toks)
            if hits >= 2:
                claimed_groups.append(gname)
        if claimed_groups:
            object_claims += 1
            if all(g not in supported_objs for g in claimed_groups):
                object_mismatch += 1

    color_rate = (color_mismatch / color_claims) if color_claims else 0.0
    object_rate = (object_mismatch / object_claims) if object_claims else 0.0
    return color_rate, object_rate, color_claims, object_claims


def build_noise_table(imgs: list[dict], img_dir: Path, sem_weight: float, contr_weight: float) -> pd.DataFrame:
    dom_map = category_dom_channels(imgs, img_dir)
    rows = []
    for img in imgs:
        fname = img['filename']
        cat = parse_category(fname)
        captions = [s.get('raw', '') for s in img.get('sentences', [])][:5]
        if len(captions) < 2:
            continue

        mean_pair, std_pair, outlier_count, center_min = semantic_metrics(captions)
        color_rate, object_rate, color_claims, object_claims = contradiction_metrics(captions, cat, dom_map.get(cat, 'R≈G>B'))

        sem_noise = np.clip((1.0 - mean_pair), 0.0, 1.0)
        contr_noise = 0.65 * color_rate + 0.35 * object_rate
        outlier_boost = min(outlier_count / 3.0, 1.0)
        center_penalty = np.clip((0.55 - center_min) / 0.55, 0.0, 1.0) if np.isfinite(center_min) else 0.0

        # strict: semantic dominates; contradiction boosts but can't dominate alone
        final_score = (
            sem_weight * (0.7 * sem_noise + 0.3 * center_penalty)
            + contr_weight * contr_noise
            + 0.10 * outlier_boost
        )

        rows.append({
            'filename': fname,
            'category': cat,
            'dom_channel': dom_map.get(cat, 'R≈G>B'),
            'mean_pairwise': mean_pair,
            'std_pairwise': std_pair,
            'center_min': center_min,
            'outlier_count': outlier_count,
            'color_mismatch_rate': color_rate,
            'object_mismatch_rate': object_rate,
            'color_claims': color_claims,
            'object_claims': object_claims,
            'sem_noise': sem_noise,
            'contr_noise': contr_noise,
            'noise_score': float(np.clip(final_score, 0.0, 1.0)),
            'captions': ' ||| '.join(captions),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = df.sort_values(['noise_score', 'mean_pairwise', 'center_min'], ascending=[False, True, True]).reset_index(drop=True)

    # strict confidence labels
    high_mask = (
        (df['noise_score'] >= 0.62)
        & (df['mean_pairwise'] <= 0.56)
        & (df['center_min'] <= 0.45)
        & ((df['color_mismatch_rate'] >= 0.5) | (df['object_mismatch_rate'] >= 0.5) | (df['outlier_count'] >= 2))
    )
    med_mask = (
        (df['noise_score'] >= 0.50)
        & (df['mean_pairwise'] <= 0.62)
        & ((df['color_mismatch_rate'] >= 0.3) | (df['object_mismatch_rate'] >= 0.3) | (df['outlier_count'] >= 1))
    )
    df['confidence'] = np.where(high_mask, 'HIGH', np.where(med_mask, 'MEDIUM', 'LOW'))
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description='Strict noisy-sample detector for RSITMD captions')
    ap.add_argument('--split', choices=['train', 'test'], default='train')
    ap.add_argument('--topk', type=int, default=120)
    ap.add_argument('--sem-weight', type=float, default=0.65)
    ap.add_argument('--contr-weight', type=float, default=0.25)
    ap.add_argument('--out-dir', type=str, default='')
    args = ap.parse_args()

    imgs, img_dir = load_split(args.split)
    out_dir = Path(args.out_dir) if args.out_dir else (Path(__file__).resolve().parents[1] / 'report' / 'figures')
    out_dir.mkdir(parents=True, exist_ok=True)

    df = build_noise_table(imgs, img_dir, sem_weight=args.sem_weight, contr_weight=args.contr_weight)
    if df.empty:
        raise RuntimeError('No rows computed. Check dataset files and image availability.')

    out_csv = out_dir / f'noisy_samples_strict_{args.split}.csv'
    df.head(args.topk).to_csv(out_csv, index=False, quoting=csv.QUOTE_MINIMAL)

    n_high = int((df['confidence'] == 'HIGH').sum())
    n_med = int((df['confidence'] == 'MEDIUM').sum())
    n_low = int((df['confidence'] == 'LOW').sum())

    print(f'[OK] Split: {args.split}')
    print(f'[OK] Total images scored: {len(df)}')
    print(f'[OK] Confidence counts => HIGH={n_high}, MEDIUM={n_med}, LOW={n_low}')
    print(f'[OK] Exported top {min(args.topk, len(df))} rows: {out_csv}')

    show_cols = [
        'filename', 'category', 'confidence', 'noise_score',
        'mean_pairwise', 'center_min', 'outlier_count',
        'color_mismatch_rate', 'object_mismatch_rate'
    ]
    print('\nTop 15 preview:')
    print(df[show_cols].head(15).to_string(index=False))


if __name__ == '__main__':
    main()
