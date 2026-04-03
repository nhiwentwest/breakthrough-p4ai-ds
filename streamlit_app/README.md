# RSITMD EDA Streamlit Demo

Interactive Exploratory Data Analysis dashboard for the **RSITMD (Remote Sensing Image-Text Matching Dataset)** — IEEE TGRS 2021.

> Uses **Python 3.11 + Streamlit 1.40** in a local venv (`.venv/`).

## Quick Start

```bash
cd streamlit_app
.venv/bin/pip install -r requirements.txt
.venv/bin/streamlit run app.py
```

Then open **http://localhost:8501**.

## Project Structure

```
streamlit_app/
├── .venv/                 # Python 3.11 virtual environment
├── app.py                 # Home — editorial-style overview
├── requirements.txt
├── README.md
├── utils/
│   └── style.py
└── pages/
    ├── text_eda.py        # 📝 Word cloud, n-grams, POS, caption length
    ├── image_eda.py        # 🖼️ Spectral bands, upload + pixel analysis
    ├── tabular_eda.py      # 📊 Distributions, correlation, scatter plots
    └── multimodal_eda.py   # 🔗 Cross-modal similarity, pair tables
```

## Editorial Theme

- **Palette**: warm off-white (`#F7F3EB`), brick red accent (`#B42318`)
- **Typography**: Playfair Display (headlines) + Source Sans 3 (body)
- **Layout**: newspaper hierarchy — hero metrics → pull quote → charts → data
- No cards, no dark mode, no glassmorphism

## Dataset

Place `RSITMD/dataset_RSITMD.json` next to this folder. The app falls back to sample data if the file is not found.