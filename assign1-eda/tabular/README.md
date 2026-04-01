# Tabular EDA — World Happiness Report 2019

Phân tích dữ liệu dạng bảng (tabular data) trên bộ dữ liệu **World Happiness Report 2019** (Gallup World Poll).

## Dataset

> Gallup World Poll · Published by the Sustainable Development Solutions Network

- **Countries:** 156
- **Features:** 9 (7 numerical + 2 categorical)
- **Target:** Score — Happiness score (0–10 scale)
- **Engineered feature:** GDP_Level (Low / Medium / High, binned from GDP per capita)

## Scripts

### `eda_tabular.py`

9 bước phân tích:

1. **Dataset Overview** — shape, feature types, descriptions, GDP_Level engineering
2. **Missing Values** — bar chart + imputation recommendations
3. **Numerical Distributions** — histograms cho 7 features, stats table
4. **Categorical (GDP Level)** — bar chart + grouped statistics
5. **Target Distribution** — Score histogram với μ và median lines
6. **Correlation Analysis** — heatmap RdBu_r + pairs |r| > 0.5
7. **Outlier Detection (IQR)** — box plots + IQR statistics table
8. **GDP Level vs Score** — grouped box plot + grouped statistics
9. **Sample Data** — top 10, bottom 10, random sample

## Figures

Output được lưu tại `../report/figures/`:

- `ta_happiness_score_distribution.png` — Step 5
- `ta_gdp_level_distribution.png` — Step 4
- `ta_correlation_matrix.png` — Step 6
- `ta_gdp_level_vs_score.png` — Step 8
- `ta_missing_values.png` — Step 2
- `ta_numerical_distributions.png` — Step 3
- `ta_outlier_detection.png` — Step 7
- `ta_top10_happiest.png` — Step 9

## Usage

```bash
pip install -r requirements.txt
python eda_tabular.py
```

## Requirements

- Python 3.8+
- pandas, numpy, matplotlib, seaborn, plotly
