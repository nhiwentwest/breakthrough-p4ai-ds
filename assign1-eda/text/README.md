# NLP EDA — Twitter Financial News Sentiment

Phân tích dữ liệu văn bản (Text Data EDA) trên bộ dữ liệu **Twitter Financial News Sentiment**.

## Dataset

> Hugging Face · Published by zeroshot

Source data: [huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment](https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment)

- **Samples:** 9,543 training samples
- **Raw Features:** 2 (`text`, `label`)
- **Target:** `label` (Sentiment classification: 0 = Negative, 1 = Positive, 2 = Neutral)
- **Engineered Features:** `tickers` (extracted cashtags), `word_count`, `char_count`

## Scripts

### `eda_textbook.ipynb` / `eda_textbook.py`

10 bước phân tích chuyên sâu cho dữ liệu ngôn ngữ tự nhiên (NLP):

1. **Dataset Overview** — shape, column counts, and general overview.
2. **Missing Values Analysis** — missing value detection (100% clean dataset).
3. **Category Distribution** — frequency table and pie chart of sentiment labels (Highly imbalanced).
4. **Wordcount Distribution** — histograms of word counts per sample + percentiles.
5. **Text Length Distribution** — character count histograms + average word length.
6. **Word Frequency** — bar chart of the top 20 most frequent words across the corpus.
7. **Tickers Distribution** — regex extraction and top 5 most mentioned stock tickers (e.g., $SPY, $TSLA).
8. **TF-IDF Top Terms by Category** — extracting the top 20 discriminative words per sentiment label.
9. **N-gram Analysis (Bigrams)** — top 15 two-word phrases (collocations) per category using TF-IDF.
10. **Category Similarity Matrix** — Cosine similarity heatmap between sentiment categories to identify potential model confusion.

## Dataset Characteristics

- **Imbalanced Dataset:** The categories are highly skewed. The Neutral class (Label 2) dominates the dataset at approximately 64.7%, while Positive (Label 1) accounts for 20.2% and Negative (Label 0) makes up only 15.1%.
- **Article Length:** The texts are extremely short, averaging only about 8-12 words per entry (with a maximum of 32 words). This reflects a "micro-blogging" or rapid-fire news headline format, typical of Twitter or StockTwits.
- **Vocabulary:** The vocabulary is highly specialized for finance and social media. It heavily features "Cashtags" (stock tickers preceded by `$`), financial shorthand, and web URLs.
- **Quality:** The raw dataset is 100% complete with zero missing values. However, it features informal grammar, abbreviations, and sentence fragments rather than professionally structured paragraphs.

## Data Processing Notes

- **Raw Statistics:** Initial metrics (like total word counts and sentence lengths) were calculated using the raw, uncleaned text to understand the true physical shape of the data.
- **Text Cleaning:** Removed URLs (`https://...`), masked `$TICKER` symbols, removed non-alphabetical characters, and converted text to lowercase.
- **Stop Words Removed:** In addition to standard English stop words, a custom list of financial/social media noise words was removed (e.g., *https, co, rt, stock, market, shares, company, said, just*).
- **Why Mixed Approach:** Raw statistics help set exact parameters for deep learning models (like `max_length`), while vocabulary analysis on aggressively cleaned text ensures TF-IDF and N-gram models focus on true sentiment-driving words.

## Usage

```bash
pip install -r requirements.txt
python eda_textbook.py