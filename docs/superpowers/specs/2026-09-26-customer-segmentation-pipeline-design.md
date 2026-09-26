# Design Specification: Customer Segmentation ML Pipeline & Streamlit Application

**Date:** 2026-09-26  
**Status:** Approved  
**Related Reference:** `PROJECT.md`

---

## 1. Overview & Objectives

The goal of this project is to implement an enterprise customer segmentation platform using unsupervised machine learning. It covers the full lifecycle from raw data ingestion and feature engineering to K-Means spatial clustering, PCA dimensionality reduction, unified pipeline serialization, and interactive Streamlit web deployment.

### Key Goals:
1. **Modular Codebase:** Clean separation of concerns across data loading, feature engineering, model training, and web serving.
2. **Unified Pipeline Architecture:** Encapsulate `StandardScaler` and `KMeans(n_clusters=6)` into a single `sklearn.pipeline.Pipeline` artifact (`models/pipeline.pkl`) to guarantee atomic transformations and eliminate feature order mismatch risks.
3. **Interactive Inference:** Real-time web application (`app.py`) allowing non-technical stakeholders to input customer attributes and obtain segment assignments and behavioral personas.
4. **Automated Testing:** Test suite verifying schema validation, transformation logic, and inference consistency.

---

## 2. Architecture & File Structure

```text
/home/picpu-11/project/other/
├── data/
│   ├── .gitkeep
│   └── customer_segmentation.csv       # Ingested dataset (placed in data/)
├── models/
│   ├── .gitkeep
│   ├── pipeline.pkl                    # Unified StandardScaler + KMeans model
│   └── pca_clusters.png                # 2D PCA visual representation
├── src/
│   ├── __init__.py
│   ├── data_loader.py                  # Ingestion, missing value handling, date parsing
│   ├── features.py                     # Derived features & feature subset selection
│   └── train.py                        # Pipeline training, Elbow evaluation, PCA plot, serialization
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py                # Unit & integration tests for features and inference
├── app.py                              # Streamlit web UI
├── requirements.txt                    # Project dependencies
├── .gitignore                          # Exclude .venv, data/*.csv, models/*.pkl, __pycache__
└── PROJECT.md                          # Comprehensive technical report
```

---

## 3. Detailed Component Specifications

### 3.1 Dependencies (`requirements.txt`)
- `pandas>=2.0.0`
- `numpy>=1.24.0`
- `scikit-learn>=1.3.0`
- `joblib>=1.3.0`
- `streamlit>=1.28.0`
- `matplotlib>=3.7.0`
- `seaborn>=0.12.0`
- `pytest>=7.4.0`

### 3.2 Data Ingestion & Sanitization (`src/data_loader.py`)
- **Input:** `data/customer_segmentation.csv` (2,240 rows, 29 raw columns).
- **Functions:**
  - `load_raw_data(filepath: str) -> pd.DataFrame`: Reads CSV with proper separator parsing.
  - `clean_data(df: pd.DataFrame) -> pd.DataFrame`:
    - Drops the 24 null rows located in `Income`.
    - Converts `Dt_Customer` into `pd.to_datetime(..., dayfirst=True)`.
    - Returns sanitized DataFrame (2,216 records).

### 3.3 Feature Engineering (`src/features.py`)
- **Derived Features:**
  - `Age`: Calculated dynamically using reference year (defaulting to current year or customer enrollment year, avoiding hardcoded static drift).
  - `Total_Children = Kidhome + Teenhome`.
  - `Total_Spending = MntWines + MntFruits + MntMeatProducts + MntFishProducts + MntGoldProds`.
  - `Customer_Since = (pd.Timestamp.now() - Dt_Customer).dt.days`.
  - `Accepted_Any = (AcceptedCmp1 + AcceptedCmp2 + AcceptedCmp3 + AcceptedCmp4 + AcceptedCmp5 + Response > 0).astype(int)`.
- **Feature Selection for Clustering:**
  - Standardized feature subset (7 columns):
    `['Age', 'Income', 'Total_Spending', 'NumWebPurchases', 'NumStorePurchases', 'NumWebVisitsMonth', 'Recency']`.
  - Function: `get_clustering_features(df: pd.DataFrame) -> pd.DataFrame`.

### 3.4 Model Training & Evaluation (`src/train.py`)
- **Steps:**
  1. Load and clean data via `src/data_loader.py`.
  2. Engineer features via `src/features.py`.
  3. Extract 7-feature matrix $X$.
  4. Perform Elbow Method evaluation for $K \in [2, 10]$ to verify cluster inertia.
  5. Fit unified `Pipeline`:
     ```python
     pipeline = Pipeline([
         ('scaler', StandardScaler()),
         ('kmeans', KMeans(n_clusters=6, random_state=42, n_init=10))
     ])
     pipeline.fit(X)
     ```
  6. Perform PCA reduction (`n_components=2`) on scaled features to generate a 2D cluster visualization scatter plot saved to `models/pca_clusters.png`.
  7. Persist pipeline object to `models/pipeline.pkl` using `joblib.dump()`.

### 3.5 Interactive Inference Application (`app.py`)
- **UI Elements:**
  - Header & description of the customer segmentation tool.
  - Sidebar or form controls for the 7 feature inputs:
    - Age (18 - 100, default 35)
    - Income (0 - 200,000, default 50,000)
    - Total Spending (0 - 5,000, default 1,000)
    - Number of Web Purchases (0 - 100, default 10)
    - Number of Store Purchases (0 - 100, default 10)
    - Number of Web Visits Per Month (0 - 50, default 3)
    - Recency in Days (0 - 365, default 30)
  - Action button: "Predict Segment".
  - Output display:
    - Predicted Cluster ID (0 - 5).
    - Operational Persona description:
      - Cluster 0: Mature Baseline Demographic (Balanced channel activity)
      - Cluster 1: High Spending Omnichannel Shoppers
      - Cluster 2: Digital Frequent Buyers
      - Cluster 3: Premium High-Spenders (Concentrated high spend)
      - Cluster 4: Budget Web Visitors (Price-sensitive, high web visits)
      - Cluster 5: Young High-Value Drivers (High earnings, premium targets)

### 3.6 Automated Testing (`tests/test_pipeline.py`)
- Test data cleaning handles missing values correctly.
- Test feature engineering generates all required derived features and handles boundary conditions.
- Test pipeline serialization and prediction returns valid cluster labels for sample vectors.

---

## 4. Verification & Success Criteria

1. Environment setup succeeds: `requirements.txt` installs cleanly into `.venv`.
2. Tests pass: `pytest tests/` runs and passes all unit tests.
3. Model training executes end-to-end: running `python -m src.train` generates `models/pipeline.pkl` and `models/pca_clusters.png`.
4. Streamlit application launches: `streamlit run app.py` serves the UI and performs predictions accurately.
