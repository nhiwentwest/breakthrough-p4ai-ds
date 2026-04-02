import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from collections import Counter
import plotly.graph_objects as go
import re
import numpy as np
url = "https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment/raw/main/sent_train.csv"
df = pd.read_csv(url)
def remove_urls(text):
    return re.sub(r'https?://\S+|www\.\S+', '', text)
def handle_company(text):
    return re.sub(r'\$\w+', '<code>', text)
def clean_text(text):
    text = re.sub(r'<code>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text
# def nlp_using(text):
#     doc = nlp(text)
#     new_text = text
#     for ent in reversed(doc.ents) :
#         if ent.label_ in ["ORG", "PERSON", "GPE", "PRODUCT"]:
#             new_text = new_text[:ent.start_char] + "<ENTITY>" + new_text[ent.end_char:]
    
#     return new_text.lower()
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
def take_company(text):
    return re.findall(r'\$[A-Za-z]+', str(text))
df['text'] = df['text'].apply(remove_urls)
df['tickers'] = df['text'].apply(take_company)
df['text'] = df['text'].apply(handle_company)
df['text'] = df['text'].apply(lambda x: x.lower())
df['text'] = df['text'].apply(clean_text)
# df['text'] = df['text'].apply(nlp_using)
df['text'] = df['text'].apply(
    lambda x: ' '.join([word for word in x.split() if word not in STOP_WORDS])
)

print(df)

if df is not None:
    #══════════════════════════════════════════════════════════════════════════════
    # DATA OVERVIEW
    #══════════════════════════════════════════════════════════════════════════════
    print("\n--- DATA OVERVIEW ---")
    print("Training samples:", f"{len(df):,}")
    print("Number of columns:", f"{len(df.columns)}")
    print(f"Shape: {df.shape}\n")
    
    #══════════════════════════════════════════════════════════════════════════════
    # CATEGORY DISTRIBUTION
    #══════════════════════════════════════════════════════════════════════════════
    print("\n--- CATEGORY STASTISTIC --- ")
    counts = df['label'].value_counts().sort_index()
    table = counts.to_frame(name='Frequency')
    table['Ratio (%)'] = (counts / counts.sum()) * 100
    print(table)
    print("\n--- CATEGORY DISTRIBUTION ---")
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.pie(
    table['Ratio (%)'],
    labels=table.index,
    autopct='%1.1f%%',
    startangle=90,
    colors=['skyblue', 'lightgreen', 'coral']
    )

    ax.set_title("CATEGORY DISTRIBUTION")

    plt.tight_layout()
    plt.show()
    
    #══════════════════════════════════════════════════════════════════════════════
    # WORD COUNT DISTRIBUTION
    #══════════════════════════════════════════════════════════════════════════════
    print("\n--- WORD COUNT DISTRIBUTION ---")
    df['word_count'] = df['text'].apply(lambda x: len(x.split()))
    bins = np.arange(0, df['word_count'].max() + 5, 5)
    hist, bin_edges = np.histogram(df['word_count'], bins=bins)
    bin_labels = [f"{int(bin_edges[i])}-{int(bin_edges[i+1])}" for i in range(len(hist))]
    fig = go.Figure(data=[go.Bar(
    x=bin_labels,
    y=hist,
    text=hist,
    textposition='outside'
    )])

    fig.update_layout(
    title='Word Count Distribution (sent_train)',
    xaxis_title='Word Count Ranges',
    yaxis_title='Number of Samples',
    width=900,
    height=600,
    showlegend=False,
    xaxis={'tickangle': 45}
    )

    fig.show()

    # Print statistics
    print(f"Mean: {df['word_count'].mean():.2f}")
    print(f"Median: {df['word_count'].median():.2f}")
    print(f"Min: {df['word_count'].min()}")
    print(f"Max: {df['word_count'].max()}")

    print("\nPercentiles:")
    for p in [25, 50, 75, 90, 95, 99]:
        print(f"  {p}th: {df['word_count'].quantile(p/100):.0f}")
    #══════════════════════════════════════════════════════════════════════════════
    # TEXT LENGTH
    #══════════════════════════════════════════════════════════════════════════════
    print("\n--- TEXT LENGTH DISTRIBUTION ---")
    df['char_count'] = df['text'].str.len()
    bins = np.arange(0, df['char_count'].max() + 20, 20)

    hist, bin_edges = np.histogram(df['char_count'], bins=bins)

    # Format labels
    bin_labels = [f"{int(bin_edges[i])}-{int(bin_edges[i+1])}" for i in range(len(hist))]

    # Create histogram
    fig = go.Figure(data=[go.Bar(
    x=bin_labels,
    y=hist,
    text=hist,
    textposition='outside'
    )])

    fig.update_layout(
    title='Character Count Distribution (sent_train)',
    xaxis_title='Character Count Ranges',
    yaxis_title='Number of Samples',
    width=900,
    height=600,
    showlegend=False,
    xaxis={'tickangle': 45}
    )

    fig.show()

    # Statistics
    print(f"Mean: {df['char_count'].mean():.2f}")
    print(f"Median: {df['char_count'].median():.2f}")
    print(f"Min: {df['char_count'].min()}")
    print(f"Max: {df['char_count'].max()}")

    # Average word length
    avg_word_length = df['char_count'].sum() / df['text'].str.split().str.len().sum()
    print(f"Average word length: {avg_word_length:.2f} characters")

    #══════════════════════════════════════════════════════════════════════════════
    # WORD FREQUENCY
    #══════════════════════════════════════════════════════════════════════════════
    all_text = ' '.join(df['text'])
    words = all_text.split()
    word_freq = Counter(words)
    top_words = word_freq.most_common(20)
    word_df = pd.DataFrame(top_words, columns=['Word', 'Frequency'])
    fig = go.Figure(data=[go.Bar(
    x=word_df['Word'],
    y=word_df['Frequency'],
    text=word_df['Frequency'],
    textposition='outside'
    )])

    fig.update_layout(
    title="Top 20 Word Frequency Distribution",
    xaxis_title="Words",
    yaxis_title="Frequency",
    width=900,
    height=600
    )

    fig.show()

    #══════════════════════════════════════════════════════════════════════════════
    # TICKERS DISTRIBUTION
    #══════════════════════════════════════════════════════════════════════════════
    all_tickers = [t for sublist in df['tickers'] for t in sublist]
    ticker_freq = Counter(all_tickers)
    top_tickers = ticker_freq.most_common(5)
    tickers_x = [t[0] for t in top_tickers]
    tickers_y = [t[1] for t in top_tickers]

    print(tickers_x)
    print(tickers_y)
    fig = go.Figure(data=[go.Bar(
    x=tickers_x,
    y=tickers_y,
    textposition='outside'
    )])

    fig.update_layout(
    title="Top 5 Most Mentioned Tickers",
    xaxis_title="Ticker",
    yaxis_title="Frequency",
    width=800,
    height=600
    )

    fig.show()
    
    #══════════════════════════════════════════════════════════════════════════════
    # TF - IDF
    #══════════════════════════════════════════════════════════════════════════════
    from sklearn.feature_extraction.text import TfidfVectorizer
    label_name = {0: "0", 1: "1", 2: "2"}
    vectorizer = TfidfVectorizer(max_features=5000, stop_words=list(STOP_WORDS))
    tfidf_matrix = vectorizer.fit_transform(df['text'])
    feature_names = vectorizer.get_feature_names_out()

    for label_to_plot in [0,1, 2]:
        cat_indices = df[df['label'] == label_to_plot].index
        cat_tfidf = tfidf_matrix[cat_indices]
        mean_scores = np.array(cat_tfidf.mean(axis=0)).flatten()

        top_indices = mean_scores.argsort()[-20:][::-1]
        top_words = [feature_names[i] for i in top_indices]
        top_scores = [mean_scores[i] for i in top_indices]

        fig = go.Figure(data=[go.Bar(
            x=top_scores,
            y=top_words,
            orientation='h',
            marker_color='#FF4B4B' if label_to_plot == 0 else ('#00CC96' if label_to_plot == 1 else '#636EFA'),
            text=[f'{score:.4f}' for score in top_scores],
            textposition='outside')])

        fig.update_layout(
            title=f'Top 20 TF-IDF Words in {label_name[label_to_plot]}',
            xaxis_title='Average TF-IDF Score',
            yaxis_title='Words',
            width=800,
            height=600,
            yaxis={'autorange': 'reversed'})

        fig.show()
    
    #══════════════════════════════════════════════════════════════════════════════
    # BIGRAM
    #══════════════════════════════════════════════════════════════════════════════

    label_name = {0: "0", 1: "1", 2: "2"}
    vectorizer = TfidfVectorizer(
            ngram_range=(2, 2), 
            max_features=50,
            stop_words=list(STOP_WORDS)
        )

    for label_to_plot in [0,1, 2]:
        cat_df = df[df['label'] == label_to_plot]

        combined_text = ' '.join(cat_df['text'].values)


       
        tfidf_matrix = vectorizer.fit_transform([combined_text])

        feature_names = vectorizer.get_feature_names_out()
        tfidf_scores = tfidf_matrix.toarray()[0]

        top_indices = tfidf_scores.argsort()[-15:][::-1]
        top_bigrams = [feature_names[i] for i in top_indices]
        top_scores = [tfidf_scores[i] for i in top_indices]

        bar_color = '#FF4B4B' if label_to_plot == 0 else ('#00CC96' if label_to_plot == 1 else '#636EFA')

        fig = go.Figure(data=[go.Bar(
            x=top_scores,
            y=top_bigrams,
            orientation='h',
            marker_color=bar_color,
            text=[f'{score:.4f}' for score in top_scores],
            textposition='outside')])

        fig.update_layout(
            title=f'Top 15 Bigrams in {label_name[label_to_plot]} (TF-IDF)',
            xaxis_title='TF-IDF Score',
            yaxis_title='Bigrams',
            width=900,
            height=600,
            yaxis={'autorange': 'reversed'})

        fig.show()
    #══════════════════════════════════════════════════════════════════════════════
    # CATEGORY SIMILARITY MATRIX
    #══════════════════════════════════════════════════════════════════════════════

    from sklearn.metrics.pairwise import cosine_similarity
    label_names = {0: '0', 1: '1', 2: '2'}
    categories = [0, 1, 2]
    category_labels = [label_names[cat] for cat in categories]

    vectorizer = TfidfVectorizer(max_features=5000, stop_words=list(STOP_WORDS))
    tfidf_matrix = vectorizer.fit_transform(df['text'])

    n_cats = len(categories)
    similarity_matrix = np.zeros((n_cats, n_cats))

    for i, cat1 in enumerate(categories):
        cat1_indices = df[df['label'] == cat1].index
        cat1_vectors = tfidf_matrix[cat1_indices]
    
    for j, cat2 in enumerate(categories):
        cat2_indices = df[df['label'] == cat2].index
        cat2_vectors = tfidf_matrix[cat2_indices]

        pairwise_sim = cosine_similarity(cat1_vectors, cat2_vectors)

        similarity_matrix[i, j] = np.mean(pairwise_sim)
    fig = go.Figure(data=go.Heatmap(
        z=similarity_matrix,
        x=category_labels,
        y=category_labels,
        colorscale='Teal', 
        text=similarity_matrix,
        texttemplate='%{text:.4f}',
        textfont={"size": 14},
        colorbar=dict(title="Similarity")))

    fig.update_layout(
        title='Category Similarity Matrix (Cosine Similarity)',
        xaxis_title='Category',
        yaxis_title='Category',
        width=800,
        height=600,
        yaxis={'autorange': 'reversed'})

    fig.show()




