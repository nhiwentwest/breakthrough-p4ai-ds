import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime
import time
import os
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer, AutoModel, AutoConfig,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW


# ============================================================================
# 1. Configuration
# ============================================================================
CONFIG = {
    'model_name': 'bert-base-uncased',
    'max_length': 64, # Tweets thường ngắn, 64 là đủ và giúp train nhanh hơn
    'batch_size': 32,
    'learning_rate': 2e-5,
    'num_epochs': 3,
    'num_labels': 3,  # 0: Bearish, 1: Bullish, 2: Neutral
    'seed': 42
}

torch.manual_seed(CONFIG['seed'])
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(CONFIG['seed'])

# ============================================================================
# 2. Data Loading & Dataset Class
# ============================================================================

def load_twitter_data(data_dir='data'):
    """Load Twitter Financial News dataset"""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    label_map = {'Bearish': 0, 'Bullish': 1, 'Neutral': 2}
    
    # Check if data exists locally
    if not (data_dir / 'train.csv').exists():
        print("📥 Dataset not found locally, downloading...")
        base_url = 'https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment/raw/main/'
        
        # Download and split
        full_train_df = pd.read_csv(base_url + 'sent_train.csv')
        train_df, test_df = train_test_split(full_train_df, test_size=0.2, random_state=CONFIG.get('seed', 42))
        val_df = pd.read_csv(base_url + 'sent_valid.csv')
        
        # Save locally
        train_df.to_csv(data_dir / 'train.csv', index=False)
        val_df.to_csv(data_dir / 'val.csv', index=False)
        test_df.to_csv(data_dir / 'test.csv', index=False)
        
        print(f"✓ Downloaded and saved Twitter Financial News dataset")
        print(f"  Train: {len(train_df)} samples")
        print(f"  Val: {len(val_df)} samples")
        print(f"  Test: {len(test_df)} samples")
        print(f"  Classes: {list(label_map.keys())}")
        
        return train_df, val_df, test_df, label_map
    
    # Load preprocessed data
    train_df = pd.read_csv(data_dir / 'train.csv')
    val_df = pd.read_csv(data_dir / 'val.csv')
    test_df = pd.read_csv(data_dir / 'test.csv')
    
    print(f"✓ Loaded Twitter Financial News dataset from local storage")
    print(f"  Train: {len(train_df)} samples")
    print(f"  Val: {len(val_df)} samples")
    print(f"  Test: {len(test_df)} samples")
    print(f"  Classes: {list(label_map.keys())}")
    
    return train_df, val_df, test_df, label_map
    
class TwitterSentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts.values if isinstance(texts, pd.Series) else texts
        self.labels = labels.values if isinstance(labels, pd.Series) else labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }

# ============================================================================
# 4. Model Architecture: BERT + Mean Pooling
# ============================================================================
class BERTMeanPoolingClassifier(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name, config=config)
        
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        last_hidden = outputs.last_hidden_state 
        
        # MEAN POOLING
        attention_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        sum_hidden = torch.sum(last_hidden * attention_mask_expanded, dim=1)
        sum_mask = torch.clamp(attention_mask_expanded.sum(dim=1), min=1e-9)
        pooled_output = sum_hidden / sum_mask
        
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return logits
# ============================================================================
# 5. Evaluation Function
# ============================================================================
def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    start_time = time.time()
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            logits = model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    inference_time = time.time() - start_time
    num_samples = len(all_labels)
    inf_time_ms_per_sample = (inference_time / num_samples) * 1000 if num_samples > 0 else 0
            
    acc = accuracy_score(all_labels, all_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)
    
    return acc, prec, rec, f1, inf_time_ms_per_sample
# ============================================================================
# 6. Main Training Loop
# ============================================================================
def main():
    print("="*70)
    print("🚀 FINETUNING BERT-BASE (MEAN POOLING) - TWITTER FINANCIAL NEWS")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using Device: {device}\n")
    
    # 1. Tải Dữ liệu (Sử dụng hàm mới của bạn)
    train_df, val_df, test_df, label_map = load_twitter_data()
    
    # 2. Tokenizer & DataLoaders
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])
    
    train_dataset = TwitterSentimentDataset(train_df['text'], train_df['label'], tokenizer, CONFIG['max_length'])
    val_dataset = TwitterSentimentDataset(val_df['text'], val_df['label'], tokenizer, CONFIG['max_length'])
    test_dataset = TwitterSentimentDataset(test_df['text'], test_df['label'], tokenizer, CONFIG['max_length'])
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'] * 2)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'] * 2) # Dành cho cuối cùng
    
    # 3. Model, Optimizer, Scheduler
    print("\n⚙️  Loading pre-trained BERT Model...")
    model = BERTMeanPoolingClassifier(CONFIG['model_name'], CONFIG['num_labels']).to(device)
    
    optimizer = AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=0.01)
    total_steps = len(train_loader) * CONFIG['num_epochs']
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)
    loss_fn = nn.CrossEntropyLoss()
    
    # 4. Training
    print("\n🏃‍♂️ Beginning Training Loop...")
    best_val_acc = 0
    train_start_time = time.time()
    
    for epoch in range(CONFIG['num_epochs']):
        model.train()
        total_train_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['num_epochs']}")
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = loss_fn(logits, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            total_train_loss += loss.item()
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        avg_train_loss = total_train_loss / len(train_loader)
        val_acc, val_prec, val_rec, val_f1, _ = evaluate(model, val_loader, device)
        
        print(f"📊 Epoch {epoch+1} Summary: Train Loss: {avg_train_loss:.4f} | Val Acc: {val_acc*100:.2f}% | Val F1: {val_f1*100:.2f}%")
        
        # Lưu lại mô hình tốt nhất
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_bert_mean.pt')

    total_train_time = time.time() - train_start_time

    # 5. Final Testing
    print("\n" + "="*70)
    print("🧪 EVALUATING ON TEST SET (UNSEEN DATA)")
    print("="*70)
    
    # Load lại weights tốt nhất từ validation
    model.load_state_dict(torch.load('best_bert_mean.pt'))
    test_acc, test_prec, test_rec, test_f1, test_inf_ms = evaluate(model, test_loader, device)
    
    total_params = sum(p.numel() for p in model.parameters())
    model_size_mb = os.path.getsize('best_bert_mean.pt') / (1024 * 1024) if os.path.exists('best_bert_mean.pt') else 0
    
    def format_params(num):
        if num > 1e6:
            return f"{num/1e6:.1f}M"
        return str(num)
        
    print(f"Model       : {CONFIG['model_name'].upper()}")
    print(f"Pooling     : Mean")
    print(f"Params      : {format_params(total_params)}")
    print(f"Accuracy    : {test_acc*100:.2f}%")
    print(f"Precision   : {test_prec*100:.2f}%")
    print(f"Recall      : {test_rec*100:.2f}%")
    print(f"F1-Score    : {test_f1*100:.2f}%")
    print(f"Train Time* : {total_train_time:.1f}s")
    print(f"Inference** : ~{test_inf_ms:.2f}ms")
    print(f"Size        : {model_size_mb:.2f}MB")
    print("="*70)

if __name__ == '__main__':
    main()