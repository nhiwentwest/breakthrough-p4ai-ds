import os
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
import re

# Force UTF-8 on Windows Console to support printing Emojis explicitly
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass
# Feature Extraction
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Dimensionality Reduction
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_selection import SelectKBest, chi2

# Classifiers
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

# Evaluation
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

# Visualization
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ============================================================================
# Load Data and Preprocessing
# ============================================================================
base_url = 'https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment/raw/main/'

train_df = pd.read_csv(base_url + 'sent_train.csv')
val_df = pd.read_csv(base_url + 'sent_valid.csv')

print(f'✓ Train: {len(train_df):,} samples')
print(f'✓ Val: {len(val_df):,} samples')
print(f'✓ Categories: {sorted(train_df["label"].unique().tolist())}')

def remove_urls(text):
    return re.sub(r'https?://\S+|www\.\S+', '', text)
def handle_company(text):
    return re.sub(r'\$\w+', '<code>', text)
def clean_text(text):
    text = re.sub(r'<code>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text
STOP_WORDS = set([
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're",
    "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he',
    'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's",
    'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
    'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are',
    'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do',
    'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because',
    'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 
    'between', 'into', 'through', 'during', 'before', 'after', 
    'to', 'from', 'in', 'out', 'on', 'off', 'again',
    'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all',
    'each', 'few', 'more', 'most', 'other', 'some', 'such', 
    'only', 'own', 'same', 'so', 'than', 'too', 'very', 'can', 'will', 'just',
    'should', 'now', 'one', 'also'
])
def take_company(text):
    return re.findall(r'\$[A-Za-z]+', str(text))

def preprocess_data(input_df):
    df = input_df.copy()
    df['text'] = df['text'].apply(remove_urls)
    df['tickers'] = df['text'].apply(take_company)
    df['text'] = df['text'].apply(handle_company)
    df['text'] = df['text'].apply(lambda x: x.lower())
    df['text'] = df['text'].apply(clean_text)
    # df['text'] = df['text'].apply(nlp_using)
    df['text'] = df['text'].apply(
        lambda x: ' '.join([word for word in x.split() if word not in STOP_WORDS])
    )
    duplicate_count = df.astype(str).duplicated().sum()
    duplicate_pct = (duplicate_count / len(df)) * 100
    if duplicate_count > 0: 
        print(f"⚠️ Detect {duplicate_count} duplicates ({duplicate_pct:.2f}% dataset).")
        df = df[~df.astype(str).duplicated()].reset_index(drop=True)
        print(f"✅ Done Cleaning! The remaining dataset size is: {df.shape}")
    else:
        print("✅ Dataset is clean. No duplicate noise found.\n")
    missing_counts = df.isnull().sum()
    missing_pct = (df.isnull().sum() / len(df) * 100)

    missing_df = pd.DataFrame({
        'column': missing_counts[missing_counts > 0].index,
        'count': missing_counts[missing_counts > 0].values,
        'percentage': missing_pct[missing_counts > 0].values
    }).sort_values('percentage', ascending=False)
    if len(missing_df) > 0:
        print("Missing Values Summary:")
        print(missing_df)

        fig = go.Figure(data=[go.Bar(
            x=missing_df['column'],
            y=missing_df['percentage'],
            marker_color='#f093fb',
            text=[f"{p:.2f}%" for p in missing_df['percentage']],
            textposition='outside'
        )])

        fig.update_layout(
            title='Missing Values by Feature (%)',
            xaxis_title='Features',
            yaxis_title='Missing Percentage',
            width=800,
            height=400
        )
        print("Removing Mising Values")
        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)
        print(f"✅ Done Cleaning! The remaining dataset size is: {df.shape}")
        fig.show()
    else:
        print("✅ Dataset is clean. No missing values reported.")
    return df

from sklearn.model_selection import train_test_split

print("\n--- Splitting Train Data into 80% Train, 20% Test ---")
train_df, test_df = train_test_split(
    train_df, 
    test_size=0.2, 
    random_state=42, 
    stratify=train_df['label']
)

print(f"New Train Set: {len(train_df)}")
print(f"New Test Set: {len(test_df)}")

copy_df = train_df.copy()

train_df = preprocess_data(train_df)
test_df = preprocess_data(test_df)
val_df = preprocess_data(val_df)

df = train_df



# ============================================================================
# Define Pipeline Configurations
# ============================================================================

EXTRACTORS = {
    'tfidf': {
        'name': 'TF-IDF',
        'class': TfidfVectorizer,
        'configs': [
            {'max_features': 10000, 'ngram_range': (1, 2), 'min_df': 2, 'max_df': 0.8},
            {'max_features': 5000, 'ngram_range': (1, 2), 'min_df': 2, 'max_df': 0.8},
        ]
    }
}

REDUCERS = {
    'none': {
        'name': 'None',
        'class': None,
        'configs': [{}]
    },
    'svd': {
        'name': 'TruncatedSVD',
        'class': TruncatedSVD,
        'configs': [
            {'n_components': 4000, 'random_state': 42}
            
        ]
    }
}

CLASSIFIERS = {
    'naive_bayes': {
        'name': 'Naive Bayes',
        'class': MultinomialNB,
        'configs': [
            {'alpha': 1.0}
        ]
    },
    'logistic': {
        'name': 'Logistic Regression',
        'class': LogisticRegression,
        'configs': [
            {'C': 10.0, 'max_iter': 1000, 'random_state': 42}
        ]
    }
}

#================================================
# Train Pipeline
#================================================

def format_model_name(name, cfg): 
    return f"{name} (" + ", ".join(f"{k}={v}" for k,v in cfg.items()) + ")" if cfg else name

def train_pipeline(X_train, y_train, X_test, y_test,
                  extractor_name, extractor_config,
                  reducer_name, reducer_config,
                  classifier_name, classifier_config):
    """Train a single pipeline and return results"""
    
    results = {
        'extractor': format_model_name(EXTRACTORS[extractor_name]['name'], extractor_config),
        'reducer': format_model_name(REDUCERS[reducer_name]['name'], reducer_config),
        'classifier': format_model_name(CLASSIFIERS[classifier_name]['name'], classifier_config)
    }
    
    # Feature Extraction
    start_time = time.time()
    vectorizer = EXTRACTORS[extractor_name]['class'](**extractor_config)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    results['extraction_time'] = time.time() - start_time
    
    # Dimensionality Reduction
    if reducer_name != 'none':
        start_time = time.time()
        

        reducer = REDUCERS[reducer_name]['class'](**reducer_config)
        X_train_vec = reducer.fit_transform(X_train_vec)
        X_test_vec = reducer.transform(X_test_vec)
        
        results['reduction_time'] = time.time() - start_time
    else:
        results['reduction_time'] = 0
    
    # Classification
    start_time = time.time()
    
    # Naive Bayes needs dense, non-negative
    if classifier_name == 'naive_bayes':
        if hasattr(X_train_vec, 'toarray'):
            X_train_vec = X_train_vec.toarray()
            X_test_vec = X_test_vec.toarray()
        X_train_vec = np.abs(X_train_vec)
        X_test_vec = np.abs(X_test_vec)
    
    classifier = CLASSIFIERS[classifier_name]['class'](**classifier_config)
    classifier.fit(X_train_vec, y_train)
    results['train_time'] = time.time() - start_time
    
    # Prediction
    start_time = time.time()
    y_pred = classifier.predict(X_test_vec)
    results['inference_time'] = (time.time() - start_time) / len(y_test) * 1000  # ms/sample
    
    # Metrics
    results['accuracy'] = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)
    results['precision'] = precision
    results['recall'] = recall
    results['f1'] = f1
    
    return results

#================================================
# STEP 5: Run All Pipelines
#================================================

def run_all_pipelines(X_train, y_train, X_test, y_test, limit=None):
    """Run all pipeline combinations"""
    print("\n🚀 Training all pipeline combinations...\n")
    
    all_results = []
    count = 0
    
    for ext_name, ext_info in EXTRACTORS.items():
        for ext_config in ext_info['configs']:
            for red_name, red_info in REDUCERS.items():
                for red_config in red_info['configs']:
                    for clf_name, clf_info in CLASSIFIERS.items():
                        for clf_config in clf_info['configs']:
                            if limit and count >= limit:
                                print(f"\n⚠️  Reached limit of {limit} pipelines")
                                return all_results
                            
                            n1 = format_model_name(EXTRACTORS[ext_name]['name'], ext_config)
                            n2 = format_model_name(REDUCERS[red_name]['name'], red_config)
                            n3 = format_model_name(CLASSIFIERS[clf_name]['name'], clf_config)
                            print(f"[{count+1}] {n1} → {n2} → {n3}", end=' ... ')
                            
                            try:
                                results = train_pipeline(
                                    X_train, y_train, X_test, y_test,
                                    ext_name, ext_config,
                                    red_name, red_config,
                                    clf_name, clf_config
                                )
                                all_results.append(results)
                                print(f"✅ Acc: {results['accuracy']*100:.2f}%")
                            except Exception as e:
                                print(f"❌ Error: {e}")
                            
                            count += 1
    
    print(f"\n✅ Completed {count} pipelines!\n")
    return all_results


#================================================
# STEP 6: Generate Comparison Report
#================================================

def generate_comparison_report(results):
    """Generate HTML comparison report"""
    print("📊 Generating comparison report...\n")
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Sort by accuracy
    df = df.sort_values('accuracy', ascending=False)
    
    # Find best models
    best_accuracy = df.iloc[0]
    fastest_train = df.loc[df['train_time'].idxmin()]
    fastest_infer = df.loc[df['inference_time'].idxmin()]
    
    print("="*70)
    print("🏆 BEST PERFORMERS")
    print("="*70)
    print(f"\n📈 Best Accuracy: {best_accuracy['accuracy']*100:.2f}%")
    print(f"\n⚡ Fastest Training: {fastest_train['train_time']:.3f}s")
    print(f"\n💨 Fastest Inference: {fastest_infer['inference_time']:.3f} ms/sample")
    
    # Top 10 table
    print(f"\n\n{'='*70}")
    print("📊 TOP PIPELINES BY ACCURACY")
    print("="*70)
    print(f"{'Number':<6} {'Classifier':<20} {'Accuracy':<10}")
    print("-"*70)
    for idx, row in df.head(10).iterrows():
        print(f"{idx+1:<6}  "
              f"{row['classifier']:<20} {row['accuracy']*100:>6.2f}%")
    
    # Save to CSV
    df.to_csv('pipeline_comparison_results.csv', index=False)
    print(f"\n✅ Results saved to: pipeline_comparison_results.csv\n")
    
    return df


#================================================
# MAIN EXECUTION
#================================================

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🔬 FINANCIAL NEWS PIPELINE COMPARISON")
    print("="*70 + "\n")
    
    X_train = train_df['text']
    y_train = train_df['label']
    X_test = test_df['text']
    y_test = test_df['label']
    
    # Run pipelines (all combinations)
    results = run_all_pipelines(X_train, y_train, X_test, y_test)
    
    # Generate report
    if results:
        df_results = generate_comparison_report(results)
    
    print("="*70)
    print("✅ PIPELINE COMPARISON COMPLETE!")
    print("="*70 + "\n")