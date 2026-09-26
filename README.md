# Customer Personality Segmentation Pipeline & Web App

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end unsupervised machine learning solution for enterprise customer personality analysis and segmentation. This project spans raw multi-touchpoint data ingestion, programmatic data sanitization, behavioral feature derivation, spatial $K$-Means clustering ($K=6$), PCA 2D visualization, unified pipeline serialization, and real-time interactive inference via a lightweight Streamlit web application.

---

## 🏗️ Project Architecture

```text
customer-segmentation-pipeline/
├── data/
│   └── customer_segmentation.csv       # Raw multi-touchpoint dataset
├── models/
│   ├── pipeline.pkl                    # Unified StandardScaler + KMeans artifact
│   └── pca_clusters.png                # 2D PCA cluster spatial visualization
├── src/
│   ├── __init__.py
│   ├── data_loader.py                  # Ingestion, missing value removal, date parsing
│   ├── features.py                     # Derived features & feature subset selection
│   └── train.py                        # Pipeline training, Elbow evaluation, PCA plot, serialization
├── tests/
│   ├── __init__.py
│   ├── test_data_loader.py             # Data loader unit tests
│   ├── test_features.py                # Feature engineering unit tests
│   ├── test_train.py                   # Model training and export tests
│   └── test_app.py                     # Streamlit inference and persona tests
├── docs/                               # Design specs and implementation plans
├── app.py                              # Streamlit real-time customer segmentation UI
├── requirements.txt                    # Project dependencies
├── .gitignore                          # Standard ignore rules
└── PROJECT.md                          # Comprehensive technical report
```

---

## 🚀 Quickstart & Local Environment Setup

### 1. Clone the Repository

```bash
git clone git@github.com:patoliyank1-lab/customer-segmentation-pipeline.git
cd customer-segmentation-pipeline
```

### 2. Create and Activate Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 📊 Model Training & Evaluation

To ingest `data/customer_segmentation.csv`, compute high-order behavioral features, fit the unified `StandardScaler + KMeans(K=6)` pipeline, generate the 2D PCA scatter plot, and serialize the model:

```bash
python -m src.train
```

* Output artifacts:
  - `models/pipeline.pkl` (Serialized unified inference pipeline)
  - `models/pca_clusters.png` (2D PCA cluster distribution plot)

---

## 🎯 Launch the Streamlit Web Application

To run the interactive customer segmentation web interface:

```bash
streamlit run app.py
```

Once running, navigate to `http://localhost:8501` in your browser. Enter customer demographic and spending metrics to obtain their predicted cluster assignment and operational persona.

---

## 🧪 Running Automated Tests

Run the full test suite with `pytest`:

```bash
pytest -v
```

---

## 👥 Operational Personas

| Cluster ID | Segment Title | Behavioral & Marketing Profile |
| :---: | :--- | :--- |
| **0** | **Mature Baseline Demographic** | Moderate income and spend; balanced activity between physical store and digital channels. |
| **1** | **High-Spending Omnichannel Shoppers** | High-income cohort with heavy product spend and frequent omnichannel transactions. |
| **2** | **Digital Frequent Buyers** | Tech-savvy cohort with strong digital website engagement and high web transaction volume. |
| **3** | **Premium High-Spenders** | Concentrated luxury spenders (wines and meats); highest average spend and lowest price sensitivity. |
| **4** | **Budget Web Visitors** | Price-sensitive demographic with lower income, minimal spend, and the highest monthly web browsing frequency. |
| **5** | **Young High-Value Drivers** | Affluent younger demographic demonstrating high earnings and strong adoption of premium services. |
