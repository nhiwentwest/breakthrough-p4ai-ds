import json
import os
import shutil
from pathlib import Path
from collections import defaultdict

from PIL import Image
import numpy as np
from datasets import Dataset, DatasetDict, Features, Image as DatasetsImage, ClassLabel, Sequence, Value
from sklearn.model_selection import StratifiedShuffleSplit


def create_rsitmd_dataset(json_path: str, img_dir: str, out_dir: str, target_size=(256, 256), val_size=0.15, seed=42):
    print(f"Loading JSON from: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    images_info = data.get("images", [])
    print(f"Total images found in JSON: {len(images_info)}")
    
    # 1. Parse Data
    parsed_data = {"train": [], "test": []}
    
    for img in images_info:
        filename = img["filename"]
        # Thư mục gốc có thể chứa ảnh, cần resolve absolute path
        img_path = os.path.join(img_dir, filename)
        
        if not os.path.exists(img_path):
            print(f"Warning: Image missing {img_path}")
            continue
            
        # Trích xuất label từ filename (e.g. airport_123.tif -> airport)
        parts = filename.replace('.tif', '').rsplit('_', 1)
        label = parts[0] if len(parts) == 2 else 'unknown'
        
        # Lấy sentences để hỗ trợ Multimodal sau này
        sentences = [s.get("raw", "") for s in img.get("sentences", [])]
        
        split = img.get("split", "train")
        record = {
            "image": img_path,  # Store path temporarily
            "label": label,
            "filename": filename,
            "sentences": sentences
        }
        
        if split in parsed_data:
            parsed_data[split].append(record)
        else:
            parsed_data["train"].append(record)
            
    print(f"Parsed initially: Train={len(parsed_data['train'])}, Test={len(parsed_data['test'])}")
    
    # Lấy danh sách classes
    all_labels = sorted(list(set(r["label"] for r in parsed_data["train"] + parsed_data["test"])))
    print(f"Found {len(all_labels)} unique classes.")
    
    # 2. Stratified Split for Validation
    train_records = parsed_data["train"]
    train_labels = [r["label"] for r in train_records]
    
    if val_size > 0:
        print("Performing Stratified Validation Split...")
        sss = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
        train_idx, val_idx = next(sss.split(np.zeros(len(train_labels)), train_labels))
        
        final_train = [train_records[i] for i in train_idx]
        final_val = [train_records[i] for i in val_idx]
    else:
        final_train = train_records
        final_val = []
        
    final_test = parsed_data["test"]
    print(f"Final split sizes: Train={len(final_train)}, Val={len(final_val)}, Test={len(final_test)}")
    
    # 3. Build Generator for Resizing and Yielding Dicts
    def gen(records):
        for r in records:
            # Resize image
            try:
                with Image.open(r["image"]) as img:
                    img = img.convert("RGB")
                    img = img.resize(target_size, Image.Resampling.LANCZOS)
                    yield {
                        "image": img,
                        "label": r["label"],
                        "filename": r["filename"],
                        "sentences": r["sentences"]
                    }
            except Exception as e:
                print(f"Error reading/resizing {r['image']}: {e}")
                
    # 4. Create HuggingFace Dataset
    features = Features({
        "image": DatasetsImage(),
        "label": ClassLabel(names=all_labels),
        "filename": Value("string"),
        "sentences": Sequence(Value("string"))
    })
    
    print("Creating HuggingFace Datasets (this will resize images and might take a minute)...")
    ds_train = Dataset.from_generator(lambda: gen(final_train), features=features)
    ds_val = Dataset.from_generator(lambda: gen(final_val), features=features)
    ds_test = Dataset.from_generator(lambda: gen(final_test), features=features)
    
    dataset_dict = DatasetDict({
        "train": ds_train,
        "validation": ds_val,
        "test": ds_test
    })
    
    # 5. Save to disk
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
        
    print(f"Saving to disk at {out_dir}...")
    dataset_dict.save_to_disk(out_dir)
    print("Done!")
    
    return dataset_dict


if __name__ == "__main__":
    # Parameters
    base_raw_dir = "/tmp/RSITMD_unzipped"
    json_path = os.path.join(base_raw_dir, "dataset_RSITMD.json")
    img_dir = os.path.join(base_raw_dir, "images")
    out_dir = "/Users/nhi/Documents/school/252/breakthrough-p4ai-ds/assign2-ml/data/processed_rsitmd_256"
    
    # Create target directory parent if not exists
    os.makedirs(os.path.dirname(out_dir), exist_ok=True)
    
    # Run
    create_rsitmd_dataset(
        json_path=json_path,
        img_dir=img_dir,
        out_dir=out_dir,
        target_size=(256, 256),
        val_size=0.15,
        seed=42
    )
