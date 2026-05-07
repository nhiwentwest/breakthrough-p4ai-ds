import os
import sys
import time
import joblib
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Force UTF-8 on Windows
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

# Nhập các class và hàm cần thiết từ BERT-finetuning.py
# (Giả định rằng file BERT-finetuning.py nằm trong cùng thư mục)
import importlib.util
spec = importlib.util.spec_from_file_location("bert_module", "BERT-finetuning.py")
bert_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bert_module)

CONFIG = bert_module.CONFIG
TwitterSentimentDataset = bert_module.TwitterSentimentDataset
BERTMeanPoolingClassifier = bert_module.BERTMeanPoolingClassifier
evaluate = bert_module.evaluate
load_twitter_data = bert_module.load_twitter_data
from transformers import AutoTokenizer

def evaluate_joblib_model(pipeline_path, test_df):
    if not os.path.exists(pipeline_path):
        print(f"❌ Không tìm thấy file {pipeline_path}")
        return None
        
    print(f"⚙️  Loading {pipeline_path}...")
    pipeline = joblib.load(pipeline_path)
    
    start_time = time.time()
    y_pred = pipeline.predict(test_df['text'])
    inference_time = (time.time() - start_time) / len(test_df) * 1000 # ms/sample
    
    y_true = test_df['label']
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    
    size_mb = os.path.getsize(pipeline_path) / (1024 * 1024)
    
    return {
        'name': os.path.basename(pipeline_path),
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'inference_time_ms': inference_time,
        'size_mb': size_mb
    }

def evaluate_bert_model(model_path, test_df):
    if not os.path.exists(model_path):
        print(f"❌ Không tìm thấy file {model_path}")
        return None
        
    print(f"⚙️  Loading {model_path}...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])
    test_dataset = TwitterSentimentDataset(test_df['text'], test_df['label'], tokenizer, CONFIG['max_length'])
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'] * 2)
    
    model = BERTMeanPoolingClassifier(CONFIG['model_name'], CONFIG['num_labels']).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # Sử dụng evaluate từ bert_module
    import torch.nn as nn
    loss_fn = nn.CrossEntropyLoss()
    _, test_acc, test_prec, test_rec, test_f1, test_inf_ms, _, _ = evaluate(model, test_loader, device, loss_fn)
    
    size_mb = os.path.getsize(model_path) / (1024 * 1024)
    
    return {
        'name': os.path.basename(model_path),
        'accuracy': test_acc,
        'precision': test_prec,
        'recall': test_rec,
        'f1': test_f1,
        'inference_time_ms': test_inf_ms,
        'size_mb': size_mb
    }

def main():
    print("="*70)
    print("⚖️ SO SÁNH TRADITIONAL ML (JOBLIB) VÀ BERT FINETUNING")
    print("="*70)
    
    # 1. Load data
    _, _, test_df, _ = load_twitter_data()
    print(f"\n📊 Đã tải tập Test với {len(test_df)} samples\n")
    
    results = []
    
    # 2. Đánh giá các models joblib
    joblib_models = [
        'pipeline_Naive_Bayes.joblib',
        'pipeline_Logistic_Regression.joblib',
        'best_pipeline.joblib'
    ]
    
    for j_model in joblib_models:
        if os.path.exists(j_model):
            res = evaluate_joblib_model(j_model, test_df)
            if res:
                results.append(res)
                
    # 3. Đánh giá BERT model
    bert_model_path = 'best_bert_mean.pt'
    if os.path.exists(bert_model_path):
        res = evaluate_bert_model(bert_model_path, test_df)
        if res:
            results.append(res)
            
    # 4. Hiển thị bảng so sánh
    if not results:
        print("❌ Không có mô hình nào để so sánh.")
        return
        
    print("\n\n" + "="*80)
    print(f"{'Mô hình':<35} | {'Acc (%)':<8} | {'F1 (%)':<8} | {'Tốc độ (ms/mẫu)':<15} | {'Dung lượng (MB)':<10}")
    print("-" * 80)
    
    # Sắp xếp theo F1 giảm dần
    results.sort(key=lambda x: x['f1'], reverse=True)
    
    for r in results:
        print(f"{r['name']:<35} | {r['accuracy']*100:>7.2f}% | {r['f1']*100:>7.2f}% | {r['inference_time_ms']:>15.2f} | {r['size_mb']:>10.1f}")
    print("="*80)

if __name__ == '__main__':
    main()
