"""Text EDA demo aligned to eda_textbook.ipynb pipeline."""

from collections import Counter
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

try:
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

try:
    st.set_page_config(page_title="Text EDA", page_icon="📝", layout="wide", initial_sidebar_state="collapsed")
except Exception:
    pass

st.markdown(
    """
<style>
[data-testid="stSidebar"] { display:none !important; }
[data-testid="stSidebarNav"] { display:none !important; }
#MainMenu, footer, header { visibility:hidden; }
.main .block-container { padding:1rem; }
</style>
""",
    unsafe_allow_html=True,
)

DATA_URL = "https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment/raw/main/sent_train.csv"
STEP_LABELS = {
    0: "Dataset Overview",
    1: "Missing + Duplicate Check",
    2: "Category Distribution",
    3: "Word Count Distribution",
    4: "Character Length Distribution",
    5: "Top Word Frequency + Tickers",
    6: "TF-IDF Top Terms by Category",
    7: "Bigram TF-IDF by Category",
    8: "Category Similarity Matrix",
    9: "OOV Rate",
    10: "Text Statistics",
}
TOTAL_STEPS = len(STEP_LABELS)

STOP_WORDS = set([
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "you're",
    "you've", "you'll", "you'd", "your", "yours", "yourself", "yourselves", "he",
    "him", "his", "himself", "she", "she's", "her", "hers", "herself", "it", "it's",
    "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which",
    "who", "whom", "this", "that", "that'll", "these", "those", "am", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "having", "do",
    "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because",
    "as", "until", "while", "of", "at", "by", "for", "with", "about", "against",
    "between", "into", "through", "during", "before", "after", "above", "below",
    "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again",
    "further", "then", "once", "here", "there", "when", "where", "why", "how", "all",
    "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not",
    "only", "own", "same", "so", "than", "too", "very", "can", "will", "just",
    "should", "now", "one", "also",
])


def remove_urls(text: str) -> str:
    return re.sub(r"https?://\S+|www\.\S+", "", str(text))


def handle_company(text: str) -> str:
    return re.sub(r"\$\w+", "<code>", str(text))


def clean_text(text: str) -> str:
    text = re.sub(r"<code>", "", str(text))
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def take_company(text: str):
    return re.findall(r"\$[A-Za-z]+", str(text))


@st.cache_data(show_spinner=False)
def load_and_prepare():
    raw_df = pd.read_csv(DATA_URL)
    base_df = raw_df.copy()

    df = raw_df.copy()
    df["tickers"] = df["text"].apply(take_company)
    df["text"] = df["text"].apply(remove_urls)
    df["text"] = df["text"].apply(handle_company)
    df["text"] = df["text"].astype(str).str.lower()
    df["text"] = df["text"].apply(clean_text)
    df["text"] = df["text"].apply(lambda x: " ".join([w for w in x.split() if w not in STOP_WORDS]))

    duplicate_count = int(df.astype(str).duplicated().sum())
    if duplicate_count > 0:
        df = df[~df.astype(str).duplicated()].reset_index(drop=True)

    missing_counts = df.isnull().sum()
    missing_pct = (missing_counts / len(df) * 100.0)
    missing_df = pd.DataFrame(
        {
            "column": missing_counts[missing_counts > 0].index,
            "count": missing_counts[missing_counts > 0].values,
            "percentage": missing_pct[missing_counts > 0].values,
        }
    ).sort_values("percentage", ascending=False)
    if len(missing_df) > 0:
        df = df.dropna().reset_index(drop=True)

    label_counts = df["label"].value_counts().sort_index()
    category_table = label_counts.to_frame(name="Frequency")
    category_table["Ratio (%)"] = (label_counts / label_counts.sum()) * 100.0

    df["word_count"] = df["text"].fillna("").apply(lambda x: len(x.split()))
    df["char_count"] = df["text"].fillna("").str.len()

    all_words = " ".join(df["text"].fillna("")).split()
    word_freq = Counter(all_words)
    ticker_freq = Counter([t for row in df["tickers"] for t in row])

    return {
        "raw_df": raw_df,
        "base_df": base_df,
        "df": df,
        "duplicate_count": duplicate_count,
        "missing_df": missing_df,
        "category_table": category_table,
        "word_freq": word_freq,
        "ticker_freq": ticker_freq,
    }


@st.cache_data(show_spinner=False)
def compute_tfidf_terms(df: pd.DataFrame, max_features: int, ngram_min: int, ngram_max: int, top_k: int):
    if not SKLEARN_AVAILABLE:
        return {}
    out = {}
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words=list(STOP_WORDS),
        ngram_range=(ngram_min, ngram_max),
    )
    tfidf_matrix = vectorizer.fit_transform(df["text"])
    features = vectorizer.get_feature_names_out()
    for label in sorted(df["label"].unique()):
        idx = df[df["label"] == label].index
        cat_tfidf = tfidf_matrix[idx]
        mean_scores = np.asarray(cat_tfidf.mean(axis=0)).flatten()
        top_idx = mean_scores.argsort()[-top_k:][::-1]
        out[int(label)] = pd.DataFrame(
            {"term": [features[i] for i in top_idx], "score": [float(mean_scores[i]) for i in top_idx]}
        )
    return out


@st.cache_data(show_spinner=False)
def compute_bigram_terms(df: pd.DataFrame, max_features: int, top_k: int):
    if not SKLEARN_AVAILABLE:
        return {}
    out = {}
    vectorizer = TfidfVectorizer(ngram_range=(2, 2), max_features=max_features, stop_words=list(STOP_WORDS))
    for label in sorted(df["label"].unique()):
        cat_df = df[df["label"] == label]
        text_blob = " ".join(cat_df["text"].values)
        if not text_blob.strip():
            out[int(label)] = pd.DataFrame(columns=["bigram", "score"])
            continue
        matrix = vectorizer.fit_transform([text_blob])
        features = vectorizer.get_feature_names_out()
        scores = matrix.toarray()[0]
        top_idx = scores.argsort()[-top_k:][::-1]
        out[int(label)] = pd.DataFrame(
            {"bigram": [features[i] for i in top_idx], "score": [float(scores[i]) for i in top_idx]}
        )
    return out


@st.cache_data(show_spinner=False)
def compute_similarity(df: pd.DataFrame, max_features: int):
    if not SKLEARN_AVAILABLE:
        return pd.DataFrame()
    labels = sorted(df["label"].unique())
    label_names = [str(x) for x in labels]
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words=list(STOP_WORDS))
    tfidf_matrix = vectorizer.fit_transform(df["text"])
    n_labels = len(labels)
    sim_matrix = np.zeros((n_labels, n_labels))
    for i, label_i in enumerate(labels):
        idx_i = df[df["label"] == label_i].index
        vec_i = tfidf_matrix[idx_i]
        for j, label_j in enumerate(labels):
            idx_j = df[df["label"] == label_j].index
            vec_j = tfidf_matrix[idx_j]
            pairwise = cosine_similarity(vec_i, vec_j)
            sim_matrix[i, j] = float(np.mean(pairwise))
    return pd.DataFrame(sim_matrix, index=label_names, columns=label_names)


@st.cache_data(show_spinner=False)
def compute_oov(df: pd.DataFrame, max_features: int, ngram_min: int, ngram_max: int):
    if not SKLEARN_AVAILABLE:
        return pd.DataFrame()
    vectorizer_50 = CountVectorizer(ngram_range=(ngram_min, ngram_max), max_features=max_features)
    vectorizer_50.fit(df["text"])
    kept_vocab = set(vectorizer_50.get_feature_names_out())

    rows = []
    for label in sorted(df["label"].unique()):
        cat_texts = df[df["label"] == label]["text"]
        cat_vectorizer = CountVectorizer(ngram_range=(ngram_min, ngram_max))
        try:
            cat_matrix = cat_vectorizer.fit_transform(cat_texts)
            cat_features = cat_vectorizer.get_feature_names_out()
            total_occurrences = float(cat_matrix.sum())
            kept_idx = [i for i, feat in enumerate(cat_features) if feat in kept_vocab]
            kept_occurrences = float(cat_matrix[:, kept_idx].sum()) if kept_idx else 0.0
        except ValueError:
            total_occurrences = 0.0
            kept_occurrences = 0.0
        retention = (kept_occurrences / total_occurrences * 100.0) if total_occurrences > 0 else 0.0
        rows.append(
            {
                "Category": int(label),
                "Total Bigram Freq": int(total_occurrences),
                "Kept Freq": int(kept_occurrences),
                "Retention Rate (%)": round(retention, 2),
                "OOV Rate (%)": round(100.0 - retention, 2),
            }
        )
    return pd.DataFrame(rows)


if "text_step" not in st.session_state:
    st.session_state.text_step = 0

with st.spinner("Loading and analyzing text dataset..."):
    D = load_and_prepare()

df = D["df"]
step = st.session_state.text_step

st.markdown("## Text EDA — Twitter Financial News Sentiment")
st.caption(f"Step {step + 1}/{TOTAL_STEPS}: {STEP_LABELS[step]}")

if step == 0:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Training samples", f"{len(df):,}")
    m2.metric("Number of columns", f"{len(df.columns):,}")
    m3.metric("Unique labels", f"{df['label'].nunique():,}")
    m4.metric("Unique words", f"{len(D['word_freq']):,}")
    st.markdown("### Dataset Samples")
    c1, c2 = st.columns(2)
    with c1:
        full_sample = st.checkbox("Show full sample", value=False, key="text_full_sample")
    with c2:
        sample_mode = st.selectbox("Sample mode", ["Sequential", "Random"], index=0, key="text_sample_mode")

    if full_sample:
        show_n = len(df)
    else:
        max_preview = min(len(df), 500)
        show_n = st.slider("Rows to show", 1, max_preview if max_preview > 0 else 1, min(50, max_preview if max_preview > 0 else 1), 1)

    if sample_mode == "Random" and show_n < len(df):
        seed = st.number_input("Random seed", min_value=0, max_value=99999, value=42, step=1, key="text_sample_seed")
        view_df = df.sample(n=show_n, random_state=int(seed))
    else:
        view_df = df.head(show_n)
    st.caption(f"Showing {len(view_df):,}/{len(df):,} rows")
    st.dataframe(view_df, use_container_width=True)

elif step == 1:
    st.write("Duplicate and missing-value check after preprocessing.")
    c1, c2 = st.columns(2)
    c1.metric("Duplicates detected", f"{D['duplicate_count']:,}")
    c2.metric("Rows after cleanup", f"{len(df):,}")
    if len(D["missing_df"]) > 0:
        st.dataframe(D["missing_df"], use_container_width=True)
    else:
        st.success("No missing values found.")

elif step == 2:
    left, right = st.columns([1.1, 1.2])
    left.dataframe(D["category_table"], use_container_width=True)
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    ax.pie(
        D["category_table"]["Ratio (%)"],
        labels=D["category_table"].index.astype(str),
        autopct="%1.1f%%",
        startangle=90,
        colors=["#60a5fa", "#34d399", "#f97316"],
        radius=0.9,
    )
    ax.set_title("Category Distribution", fontsize=11)
    right.pyplot(fig)

elif step == 3:
    bin_size = st.slider("Word-count bin size", 1, 10, 5)
    max_wc = int(df["word_count"].max())
    bins = np.arange(0, max_wc + bin_size, bin_size)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(df["word_count"], bins=bins, color="#2563eb", edgecolor="white")
    ax.set_title("Word Count Distribution")
    ax.set_xlabel("Words per text")
    ax.set_ylabel("Samples")
    st.pyplot(fig)
    st.dataframe(df["word_count"].describe().to_frame("value"), use_container_width=True)

elif step == 4:
    bin_size = st.slider("Character-count bin size", 5, 40, 20)
    max_cc = int(df["char_count"].max())
    bins = np.arange(0, max_cc + bin_size, bin_size)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(df["char_count"], bins=bins, color="#0ea5e9", edgecolor="white")
    ax.set_title("Character Length Distribution")
    ax.set_xlabel("Characters per text")
    ax.set_ylabel("Samples")
    st.pyplot(fig)
    st.dataframe(df["char_count"].describe().to_frame("value"), use_container_width=True)

elif step == 5:
    top_n = st.slider("Top words", 10, 50, 20)
    top_words = D["word_freq"].most_common(top_n)
    wdf = pd.DataFrame(top_words, columns=["Word", "Frequency"])
    left, right = st.columns(2)
    left.dataframe(wdf, use_container_width=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(wdf["Word"][::-1], wdf["Frequency"][::-1], color="#ef4444")
    ax.set_title(f"Top {top_n} Word Frequency")
    right.pyplot(fig)

    st.markdown("### Top Mentioned Tickers")
    top_tickers = D["ticker_freq"].most_common(5)
    tdf = pd.DataFrame(top_tickers, columns=["Ticker", "Frequency"])
    st.dataframe(tdf, use_container_width=True)
    if len(tdf) > 0:
        fig2, ax2 = plt.subplots(figsize=(7, 4))
        ax2.bar(tdf["Ticker"], tdf["Frequency"], color="#10b981")
        ax2.set_title("Top 5 Most Mentioned Tickers")
        st.pyplot(fig2)

elif step == 6:
    if not SKLEARN_AVAILABLE:
        st.error("scikit-learn is required for TF-IDF views.")
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            label = st.selectbox("Label", sorted(df["label"].unique()), index=0)
        with c2:
            max_feat = st.slider("Max features", 20, 500, 100, 10, key="tfidf_maxfeat")
        with c3:
            top_k = st.slider("Top terms", 5, 30, 20, 1, key="tfidf_topk")
        ngram_max = st.radio("N-gram range", [1, 2, 3], index=1, horizontal=True, key="tfidf_ngram_max")
        tfidf_map = compute_tfidf_terms(df, max_feat, 1, int(ngram_max), top_k)
        table = tfidf_map[int(label)]
        st.dataframe(table, use_container_width=True)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(table["term"][::-1], table["score"][::-1], color="#f59e0b")
        ax.set_title(f"Top {len(table)} TF-IDF Terms (Label {label})")
        st.pyplot(fig)

elif step == 7:
    if not SKLEARN_AVAILABLE:
        st.error("scikit-learn is required for bigram TF-IDF views.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            label = st.selectbox("Label", sorted(df["label"].unique()), index=0, key="bigram_label")
        with c2:
            max_feat = st.slider("Max bigram features", 20, 500, 120, 10, key="bigram_maxfeat")
        with c3:
            top_preset = st.selectbox("Top preset", [10, 15, 20, 30], index=1, key="bigram_top_preset")
        with c4:
            custom_top = st.toggle("Custom top-k", value=False, key="bigram_custom_top")
        if custom_top:
            top_k = st.slider("Top bigrams", 5, 30, int(top_preset), 1, key="bigram_topk")
        else:
            top_k = int(top_preset)
        bigram_map = compute_bigram_terms(df, max_feat, top_k)
        table = bigram_map[int(label)]
        st.dataframe(table, use_container_width=True)
        if len(table) > 0:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.barh(table["bigram"][::-1], table["score"][::-1], color="#8b5cf6")
            ax.set_title(f"Top {len(table)} Bigrams by TF-IDF (Label {label})")
            st.pyplot(fig)

elif step == 8:
    if not SKLEARN_AVAILABLE:
        st.error("scikit-learn is required for similarity matrix.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            max_feat = st.slider("Max TF-IDF features", 100, 10000, 5000, 100, key="sim_maxfeat")
        with c2:
            decimals = st.slider("Table decimals", 2, 6, 4, 1, key="sim_decimals")
        sim = compute_similarity(df, max_feat)
        st.dataframe(sim.round(decimals), use_container_width=True)
        fig, ax = plt.subplots(figsize=(6, 5))
        cmap = st.selectbox("Colormap", ["YlGnBu", "viridis", "magma", "coolwarm"], index=0, key="sim_cmap")
        im = ax.imshow(sim.values, cmap=cmap, vmin=0, vmax=1)
        ax.set_xticks(range(len(sim.columns)))
        ax.set_yticks(range(len(sim.index)))
        ax.set_xticklabels(sim.columns)
        ax.set_yticklabels(sim.index)
        ax.set_title("Category Similarity Matrix")
        fig.colorbar(im, ax=ax)
        st.pyplot(fig)

elif step == 9:
    if not SKLEARN_AVAILABLE:
        st.error("scikit-learn is required for OOV analysis.")
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            max_feat = st.slider("Reference vocab size", 100, 10000, 5000, 100, key="oov_maxfeat")
        with c2:
            ngram_max = st.selectbox("N-gram max", [1, 2, 3], index=1, key="oov_ngrammax")
        with c3:
            selected = st.multiselect(
                "Categories",
                options=sorted(df["label"].unique()),
                default=sorted(df["label"].unique()),
                key="oov_categories",
            )
        oov = compute_oov(df, max_feat, 1, int(ngram_max))
        if selected:
            oov = oov[oov["Category"].isin(selected)]
        st.dataframe(oov, use_container_width=True)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(oov["Category"].astype(str), oov["OOV Rate (%)"], color="#dc2626")
        ax.set_title("OOV Rate by Category")
        ax.set_xlabel("Category")
        ax.set_ylabel("OOV Rate (%)")
        st.pyplot(fig)

elif step == 10:
    raw_word_counts = D["base_df"]["text"].apply(lambda x: len(str(x).split()))
    percentiles = st.multiselect(
        "Percentiles to display",
        options=[10, 25, 50, 75, 90, 95, 99],
        default=[25, 50, 75, 90, 95, 99],
        key="stats_percentiles",
    )
    stats = pd.DataFrame(
        {
            "metric": ["Max words", "Min words", "Mean words", "Std words"],
            "value": [
                int(raw_word_counts.max()),
                int(raw_word_counts.min()),
                float(raw_word_counts.mean()),
                float(raw_word_counts.std()),
            ],
        }
    )
    st.dataframe(stats, use_container_width=True)
    if percentiles:
        pct_rows = pd.DataFrame(
            {
                "percentile": percentiles,
                "word_count": [float(raw_word_counts.quantile(p / 100.0)) for p in percentiles],
            }
        )
        st.dataframe(pct_rows, use_container_width=True)
    st.markdown(
        """
- Dataset is imbalanced (neutral label dominates).
- Texts are short, headline-like financial snippets.
- Vocabulary is finance-specific with many ticker mentions.
- Raw data quality is clean (very low/no missing values).
"""
    )

col1, col2 = st.columns(2)
with col1:
    if step == 0:
        if st.button("↺ Start Over"):
            st.session_state.text_step = 0
            st.rerun()
    else:
        if st.button("← Previous"):
            st.session_state.text_step = max(0, step - 1)
            st.rerun()

with col2:
    if step == TOTAL_STEPS - 1:
        if st.button("↺ Start Over"):
            st.session_state.text_step = 0
            st.rerun()
    else:
        if st.button("Next →"):
            st.session_state.text_step = min(TOTAL_STEPS - 1, step + 1)
            st.rerun()
