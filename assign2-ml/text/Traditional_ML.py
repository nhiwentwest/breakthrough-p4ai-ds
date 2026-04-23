import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from collections import Counter
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
import numpy as np
import time 



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
#TF-IDF
# ============================================================================

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),  # Unigrams + bigrams
    min_df=2,
    max_df=0.8
)

start = time.time()
X_train = vectorizer.fit_transform(train_df['text'])
X_val = vectorizer.transform(val_df['text'])
elapsed = time.time() - start

print('\n')
print(f'✓ Vocabulary: {len(vectorizer.get_feature_names_out()):,} features')
print(f'✓ Train shape: {X_train.shape}')
print(f'✓ Test shape: {X_val.shape}')
print(f'✓ Extraction time: {elapsed:.2f}s\n')




# ============================================================================
# CLASSIFIERS
# ============================================================================
def train_classifier(name, model, X_train, y_train, X_test, y_test):
    """Train and evaluate a classifier"""
    from sklearn.metrics import precision_recall_fscore_support
    
    print("="*70)
    print(f"{name.upper()}")
    print("="*70)
    
    # Train
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start
    
    # Predict
    start = time.time()
    y_pred = model.predict(X_test)
    inference_time = time.time() - start
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"⏱️  Training: {train_time:.4f}s")
    print(f"⏱️  Inference: {inference_time:.4f}s")
    print(f"📊 Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"📊 Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"📊 Recall: {recall:.4f} ({recall*100:.2f}%)")
    print(f"📊 F1-Score: {f1:.4f} ({f1*100:.2f}%)")
    print()
    
    return {
        'name': name,
        'model': model,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'train_time': train_time,
        'inference_time': inference_time,
        'inference_speed': len(y_test) / inference_time if inference_time > 0 else 0,
        'y_pred': y_pred,
        'confusion_matrix': cm
    }

# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_confusion_matrix_plotly(cm, labels, title):
    """Create Plotly confusion matrix heatmap"""
    # Convert numpy arrays to lists to bypass Plotly's base64 "bdata" encoding, 
    # which vanilla Plotly.js may fail to parse directly from the HTML injected JSON.
    if hasattr(cm, 'tolist'):
        cm = cm.tolist()
        
    # Convert labels to strings so they are treated as categorical instead of numerical axes
    labels_str = [str(L) for L in labels]
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=labels_str,
        y=labels_str,
        colorscale='Blues',
        showscale=True,
        text=cm,
        texttemplate='%{text}',
        textfont={"size": 14},
        hovertemplate='True: %{y}<br>Pred: %{x}<br>Count: %{z}<extra></extra>'
    ))
    
    fig.update_layout(
        title=f'Confusion Matrix - {title}',
        xaxis_title='Predicted',
        yaxis_title='True Label',
        yaxis_autorange='reversed',
        width=600,
        height=500,
        font=dict(size=12)
    )
    
    return fig


def create_comparison_chart(results):
    """Create comparison bar chart"""
    names = [r['name'] for r in results]
    accuracies = [r['accuracy']*100 for r in results]
    train_times = [r['train_time'] for r in results]
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Accuracy Comparison', 'Training Time Comparison'),
        specs=[[{"type": "bar"}, {"type": "bar"}]]
    )
    
    # Accuracy
    fig.add_trace(
        go.Bar(x=names, y=accuracies, name='Accuracy (%)',
               marker_color='rgb(102, 126, 234)',
               text=[f'{a:.2f}%' for a in accuracies],
               textposition='outside'),
        row=1, col=1
    )
    
    # Training time
    fig.add_trace(
        go.Bar(x=names, y=train_times, name='Time (s)',
               marker_color='rgb(245, 135, 108)',
               text=[f'{t:.4f}s' for t in train_times],
               textposition='outside'),
        row=1, col=2
    )
    
    fig.update_layout(
        title_text="Performance Comparison",
        showlegend=False,
        height=400,
        font=dict(size=12)
    )
    
    fig.update_yaxes(title_text="Accuracy (%)", row=1, col=1)
    fig.update_yaxes(title_text="Time (seconds)", row=1, col=2)
    
    return fig

# ============================================================================
# HTML REPORT
# ============================================================================

def generate_html_report(results, y_test):
    """Generate HTML report with interactive plots"""
    labels = sorted(y_test.unique())
    
    # Use raw string to avoid # interpretation
    css_style = """
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1400px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1 { color: #333; text-align: center; margin-bottom: 10px; }
        h2 { color: #667eea; margin-top: 30px; }
        .summary { background: #f8f9fa; padding: 20px; border-radius: 8px; margin: 20px 0; }
        .summary table { width: 100%; border-collapse: collapse; font-size: 14px; }
        .summary th { background: #667eea; color: white; padding: 12px; text-align: center; font-weight: 600; }
        .summary td { padding: 10px; text-align: center; border-bottom: 1px solid #e9ecef; }
        .summary td:first-child { text-align: left; font-weight: 600; }
        .summary tr:hover { background: #f8f9fa; }
        .best { background: #d4edda !important; font-weight: bold; color: #155724; }
        .chart-container { margin: 30px 0; background: #f8f9fa; padding: 20px; border-radius: 8px; }
        .cm-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(600px, 1fr)); gap: 20px; margin-top: 20px; }
    """
    
    html_parts = ["""
<!DOCTYPE html>
<html>
<head>
    <title>BBC News Classification Results</title>
    <script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
    <style>
""" + css_style + """
    </style>
</head>
<body>
    <div class="container">
        <h1>BBC News Text Classification Results</h1>
        <p style="text-align: center; color: #666;">TF-IDF + Traditional Machine Learning</p>
        
        <div class="summary">
            <h2>Performance Comparison</h2>
            <table>
                <tr>
                    <th>Method</th>
                    <th>Accuracy</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>F1-Score</th>
                    <th>Train Time</th>
                    <th>Inference Speed</th>
                </tr>
"""]
    
    # Find best values
    best_acc = max(r['accuracy'] for r in results)
    best_prec = max(r['precision'] for r in results)
    best_rec = max(r['recall'] for r in results)
    best_f1 = max(r['f1_score'] for r in results)
    best_train = min(r['train_time'] for r in results)
    best_speed = max(r['inference_speed'] for r in results)
    
    for r in results:
        # Highlight best values
        acc_class = ' class="best"' if r['accuracy'] == best_acc else ''
        prec_class = ' class="best"' if r['precision'] == best_prec else ''
        rec_class = ' class="best"' if r['recall'] == best_rec else ''
        f1_class = ' class="best"' if r['f1_score'] == best_f1 else ''
        train_class = ' class="best"' if r['train_time'] == best_train else ''
        speed_class = ' class="best"' if r['inference_speed'] == best_speed else ''
        
        html_parts.append(f"""
<tr>
    <td><strong>{r['name']}</strong></td>
    <td{acc_class}>{r['accuracy']*100:.2f}% ↑</td>
    <td{prec_class}>{r['precision']*100:.2f}% ↑</td>
    <td{rec_class}>{r['recall']*100:.2f}% ↑</td>
    <td{f1_class}>{r['f1_score']*100:.2f}% ↑</td>
    <td{train_class}>{r['train_time']:.4f}s ↓</td>
    <td{speed_class}>{r['inference_speed']:.4f} samples/s ↑</td>
</tr>
""")
    
    html_parts.append("""
            </table>
        </div>
        
        <div class="chart-container">
            <h2>Performance Comparison</h2>
            <div id="comparison-chart"></div>
        </div>
        
        <div class="chart-container">
            <h2>Confusion Matrices</h2>
            <div class="cm-grid">
""")
    
    # Confusion matrices
    for i, r in enumerate(results):
        html_parts.append(f'                <div id="cm-{i}"></div>\n')
    
    html_parts.append("""
            </div>
        </div>
    </div>
    
    <script>
""")
    
    # Comparison chart
    comp_fig = create_comparison_chart(results)
    html_parts.append(f"        var compData = {comp_fig.to_json()};\n")
    html_parts.append("        Plotly.newPlot('comparison-chart', compData.data, compData.layout);\n\n")
    
    # Confusion matrices
    for i, r in enumerate(results):
        cm_fig = plot_confusion_matrix_plotly(r['confusion_matrix'], labels, r['name'])
        html_parts.append(f"        var cmData{i} = {cm_fig.to_json()};\n")
        html_parts.append(f"        Plotly.newPlot('cm-{i}', cmData{i}.data, cmData{i}.layout);\n\n")
    
    html_parts.append("""
    </script>
</body>
</html>
""")
    
    return ''.join(html_parts)

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run specified classifiers and generate report"""
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, confusion_matrix
    
    # Expose metrics globally for train_classifier function to use
    global accuracy_score, confusion_matrix
    import sklearn.metrics
    accuracy_score = sklearn.metrics.accuracy_score
    confusion_matrix = sklearn.metrics.confusion_matrix
    
    print()
    print("="*70)
    print("🚀 TEXT CLASSIFICATION (Naive Bayes & Logistic Regression)")
    print("="*70)
    print()

    # Get labels from global dataframes
    y_train = train_df['label']
    y_val = val_df['label']

    classifiers = [
        ('Naive Bayes', MultinomialNB()),
        ('Logistic Regression', LogisticRegression(max_iter=1000, random_state=42))
    ]

    results = []
    for name, model in classifiers:
        # Note: Using X_val and y_val for evaluation
        result = train_classifier(name, model, X_train, y_train, X_val, y_val)
        results.append(result)
        print()

    # Generate HTML report
    print("="*70)
    print("📄 GENERATING HTML REPORT")
    print("="*70)
    html_report = generate_html_report(results, y_val)

    with open('classification_results.html', 'w', encoding='utf-8') as f:
        f.write(html_report)

    print("✓ Report saved: classification_results.html")
    print()

    print("="*70)
    print("✅ ALL COMPLETE!")
    print("="*70)
    print()

if __name__ == '__main__':
    main()
