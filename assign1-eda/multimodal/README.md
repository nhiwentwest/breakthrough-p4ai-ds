# Multimodal EDA — RSITMD

Phân tích dữ liệu đa phương thức (image + text) trên bộ dữ liệu **RSITMD** (Remote Sensing Image-Text Matching Dataset).

## Dataset

> Z. Yuan et al., *"RSITMD & RSICD: Updated Benchmarks and State-of-the-Art Method for Remote Sensing Image Text Retrieval"*, IEEE TGRS, 2021.

- **Train:** 5,000+ ảnh vệ tinh × 5 captions = ~25,000 cặp image-caption
- **Test:** 2,000+ ảnh
- **21 categories** (airport, beach, bridge, ...)

## Scripts

### `eda_multimodal.py`

Phân tích 3 góc nhìn:

1. **Text Analysis** — phân bố caption, độ dài, từ vựng, bigrams, từ khóa theo category
2. **Image Analysis** — phân bố lớp, số lượng ảnh mỗi category
3. **Multimodal Analysis** — caption variability, sample pairs, visual word coverage

## Figures

Output được lưu tại `../report/figures/`:
- `ia_01_*` — Image analysis
- `ta_02_*` — Word frequency
- `ta_04_*` — Bigram frequency
- `mm_01_*` — Caption variability
- `mm_02_*` — Sample pairs

## Usage

```bash
pip install -r requirements.txt
python eda_multimodal.py
```

## Requirements

- Python 3.8+
- numpy, pandas, matplotlib, seaborn
- nltk (tự động tải stopwords khi chạy)
