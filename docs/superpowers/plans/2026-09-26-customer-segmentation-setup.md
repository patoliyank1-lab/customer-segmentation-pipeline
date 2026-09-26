# Customer Segmentation ML Pipeline & Streamlit App Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an end-to-end customer personality segmentation pipeline in Python with automated data sanitization, behavioral feature derivation, unsupervised K-Means (K=6) spatial clustering, PCA 2D visualization, unified pipeline serialization, and an interactive Streamlit inference web app.

**Architecture:** A modular Python structure where `src/data_loader.py` ingests and sanitizes raw data, `src/features.py` computes domain-specific behavioral features, `src/train.py` executes Elbow analysis, fits a unified `sklearn.pipeline.Pipeline` (StandardScaler + KMeans) and exports it, and `app.py` delivers interactive real-time cluster inference and persona mapping via Streamlit.

**Tech Stack:** Python 3.12, Pandas, NumPy, Scikit-Learn, Joblib, Streamlit, Matplotlib, Seaborn, Pytest.

**Spec:** `docs/superpowers/specs/2026-09-26-customer-segmentation-pipeline-design.md`

## Global Constraints

- Python version: Python 3.12.
- Environment: Local `.venv` virtual environment.
- Feature Ordering: Exact 7-feature order required during training and inference: `['Age', 'Income', 'Total_Spending', 'NumWebPurchases', 'NumStorePurchases', 'NumWebVisitsMonth', 'Recency']`.
- Cluster Count: $K = 6$ clusters with fixed random state 42.
- Model Artifact: Unified `sklearn.pipeline.Pipeline` exported to `models/pipeline.pkl`.
- Private data & keys: `data/*.csv`, `models/*.pkl`, and `.ssh/` must remain ignored by git.

## Review Focus

1. `Income` missing values (24 null rows) — ensure they are completely dropped and do not propagate NaNs into scaling.
2. Hardcoded date drift — compute customer age and tenure dynamically without brittle static date assumptions.
3. Feature alignment in inference — UI inputs must be assembled into a DataFrame with the exact column names and order expected by the pipeline.
4. Numerical edge bounds in UI — all inputs must have positive min/max limits to prevent negative spending or invalid purchase counts.
5. Serialization integrity — the loaded pipeline must produce identical predictions to the in-memory fitted pipeline.

---

### Task 1: Environment Setup & Project Dependencies

**Files:**
- Create: `requirements.txt`
- Modify: `.gitignore` (if necessary)
- Test: Environment import verification

**Interfaces:**
- Produces: Installed dependencies inside `.venv`.

- [ ] **Step 1: Write requirements.txt**

Create `requirements.txt` with locked major/minor dependency versions:
```text
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
joblib>=1.3.0
streamlit>=1.28.0
matplotlib>=3.7.0
seaborn>=0.12.0
pytest>=7.4.0
```

- [ ] **Step 2: Create virtual environment and install dependencies**

Run:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

- [ ] **Step 3: Verify library imports**

Run:
```bash
.venv/bin/python -c "import pandas, numpy, sklearn, joblib, streamlit, matplotlib, seaborn, pytest; print('Environment OK')"
```
Expected: `Environment OK`

- [ ] **Step 4: Commit requirements.txt**

```bash
git add requirements.txt
git commit -m "build: define project dependencies in requirements.txt"
```

---

### Task 2: Data Ingestion & Sanitization (`src/data_loader.py`)

**Files:**
- Create: `src/__init__.py`
- Create: `src/data_loader.py`
- Create: `tests/__init__.py`
- Create: `tests/test_data_loader.py`

**Interfaces:**
- Produces:
  - `load_raw_data(filepath: str) -> pd.DataFrame`
  - `clean_data(df: pd.DataFrame) -> pd.DataFrame`

- [ ] **Step 1: Write failing unit tests for data loader**

Create `tests/test_data_loader.py`:
```python
import os
import pandas as pd
import pytest
from src.data_loader import load_raw_data, clean_data

def test_load_raw_data_success(tmp_path):
    csv_file = tmp_path / "sample.csv"
    csv_file.write_text("ID,Year_Birth,Income,Dt_Customer\n1,1980,50000,01-01-2014\n")
    df = load_raw_data(str(csv_file))
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert "Income" in df.columns

def test_clean_data_drops_null_income():
    raw_df = pd.DataFrame({
        "ID": [1, 2],
        "Year_Birth": [1980, 1990],
        "Income": [50000.0, None],
        "Dt_Customer": ["01-01-2014", "02-02-2014"]
    })
    cleaned = clean_data(raw_df)
    assert len(cleaned) == 1
    assert cleaned.iloc[0]["ID"] == 1
    assert pd.api.types.is_datetime64_any_dtype(cleaned["Dt_Customer"])
```

- [ ] **Step 2: Run test to verify failure**

Run: `.venv/bin/pytest tests/test_data_loader.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.data_loader'`

- [ ] **Step 3: Implement data loader module**

Create `src/__init__.py` and `src/data_loader.py`:
```python
import pandas as pd

def load_raw_data(filepath: str) -> pd.DataFrame:
    """Load raw customer dataset from CSV file."""
    return pd.read_csv(filepath)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Sanitize data: drop null income records and parse customer enrollment dates."""
    df_clean = df.copy()
    df_clean.dropna(subset=['Income'], inplace=True)
    df_clean['Dt_Customer'] = pd.to_datetime(df_clean['Dt_Customer'], dayfirst=True)
    return df_clean
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_data_loader.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/__init__.py src/data_loader.py tests/__init__.py tests/test_data_loader.py
git commit -m "feat: implement data loading and sanitization module"
```

---

### Task 3: Behavioral Feature Engineering (`src/features.py`)

**Files:**
- Create: `src/features.py`
- Create: `tests/test_features.py`

**Interfaces:**
- Consumes: Cleaned `pd.DataFrame` from `src.data_loader`
- Produces:
  - `derive_features(df: pd.DataFrame, reference_year: int = None) -> pd.DataFrame`
  - `get_clustering_features(df: pd.DataFrame) -> pd.DataFrame` (7 columns)

- [ ] **Step 1: Write failing unit tests for feature engineering**

Create `tests/test_features.py`:
```python
import pandas as pd
import pytest
from src.features import derive_features, get_clustering_features, CLUSTERING_FEATURES

def sample_cleaned_data():
    return pd.DataFrame({
        "ID": [1],
        "Year_Birth": [1985],
        "Income": [60000.0],
        "Kidhome": [1],
        "Teenhome": [1],
        "Dt_Customer": [pd.Timestamp("2014-01-01")],
        "Recency": [25],
        "MntWines": [200],
        "MntFruits": [20],
        "MntMeatProducts": [150],
        "MntFishProducts": [30],
        "MntSweetProducts": [10],
        "MntGoldProds": [40],
        "NumDealsPurchases": [2],
        "NumWebPurchases": [6],
        "NumCatalogPurchases": [3],
        "NumStorePurchases": [8],
        "NumWebVisitsMonth": [4],
        "AcceptedCmp1": [0],
        "AcceptedCmp2": [0],
        "AcceptedCmp3": [1],
        "AcceptedCmp4": [0],
        "AcceptedCmp5": [0],
        "Response": [0]
    })

def test_derive_features():
    df = sample_cleaned_data()
    derived = derive_features(df, reference_year=2025)
    
    assert derived.loc[0, "Age"] == 40
    assert derived.loc[0, "Total_Children"] == 2
    assert derived.loc[0, "Total_Spending"] == 440  # 200+20+150+30+40
    assert derived.loc[0, "Accepted_Any"] == 1
    assert "Customer_Since" in derived.columns

def test_get_clustering_features():
    df = sample_cleaned_data()
    derived = derive_features(df, reference_year=2025)
    X = get_clustering_features(derived)
    
    assert list(X.columns) == CLUSTERING_FEATURES
    assert len(X) == 1
```

- [ ] **Step 2: Run test to verify failure**

Run: `.venv/bin/pytest tests/test_features.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.features'`

- [ ] **Step 3: Implement feature engineering module**

Create `src/features.py`:
```python
import pandas as pd
from datetime import datetime

CLUSTERING_FEATURES = [
    'Age',
    'Income',
    'Total_Spending',
    'NumWebPurchases',
    'NumStorePurchases',
    'NumWebVisitsMonth',
    'Recency'
]

def derive_features(df: pd.DataFrame, reference_year: int = None) -> pd.DataFrame:
    """Synthesize high-order behavioral and demographic features."""
    df_feat = df.copy()
    current_year = reference_year or datetime.now().year
    
    # Demographic features
    df_feat['Age'] = current_year - df_feat['Year_Birth']
    df_feat['Total_Children'] = df_feat['Kidhome'] + df_feat['Teenhome']
    
    # Financial aggregate
    spending_cols = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntGoldProds']
    existing_spend_cols = [c for c in spending_cols if c in df_feat.columns]
    df_feat['Total_Spending'] = df_feat[existing_spend_cols].sum(axis=1)
    
    # Temporal tenure (days)
    now = pd.Timestamp.now()
    df_feat['Customer_Since'] = (now - df_feat['Dt_Customer']).dt.days
    
    # Campaign responsiveness
    campaign_cols = ['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'Response']
    existing_camp_cols = [c for c in campaign_cols if c in df_feat.columns]
    if existing_camp_cols:
        df_feat['Accepted_Any'] = (df_feat[existing_camp_cols].sum(axis=1) > 0).astype(int)
    else:
        df_feat['Accepted_Any'] = 0
        
    return df_feat

def get_clustering_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract standard 7-feature matrix for clustering."""
    return df[CLUSTERING_FEATURES].copy()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_features.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/features.py tests/test_features.py
git commit -m "feat: implement behavioral feature derivation and feature selection"
```

---

### Task 4: Pipeline Training, Evaluation & Serialization (`src/train.py`)

**Files:**
- Create: `src/train.py`
- Create: `tests/test_train.py`

**Interfaces:**
- Consumes: Features from `src.features`, cleaned data from `src.data_loader`
- Produces:
  - `models/pipeline.pkl`: Serialized `sklearn.pipeline.Pipeline`
  - `models/pca_clusters.png`: 2D PCA cluster visualization plot
  - Function: `build_and_train_pipeline(data_path: str, models_dir: str) -> Pipeline`

- [ ] **Step 1: Write failing test for training pipeline**

Create `tests/test_train.py`:
```python
import os
import joblib
import pandas as pd
import pytest
from src.train import build_and_train_pipeline
from src.features import CLUSTERING_FEATURES

def test_build_and_train_pipeline(tmp_path):
    # Create minimal valid CSV
    rows = []
    for i in range(12):
        rows.append(f"{i},198{i%9},Graduation,Single,{30000 + i*3000},0,0,01-01-2014,{10+i},100,10,50,10,5,10,1,2,1,4,3,0,0,0,0,0,0,3,11,0")
    header = "ID,Year_Birth,Education,Marital_Status,Income,Kidhome,Teenhome,Dt_Customer,Recency,MntWines,MntFruits,MntMeatProducts,MntFishProducts,MntSweetProducts,MntGoldProds,NumDealsPurchases,NumWebPurchases,NumCatalogPurchases,NumStorePurchases,NumWebVisitsMonth,AcceptedCmp3,AcceptedCmp4,AcceptedCmp5,AcceptedCmp1,AcceptedCmp2,Complain,Z_CostContact,Z_Revenue,Response"
    csv_file = tmp_path / "test_data.csv"
    csv_file.write_text(header + "\n" + "\n".join(rows))
    
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    
    pipeline = build_and_train_pipeline(str(csv_file), str(models_dir), n_clusters=3)
    
    assert os.path.exists(models_dir / "pipeline.pkl")
    assert os.path.exists(models_dir / "pca_clusters.png")
    
    # Test inference with sample input
    sample = pd.DataFrame([{col: 20 for col in CLUSTERING_FEATURES}])
    pred = pipeline.predict(sample)
    assert len(pred) == 1
    assert pred[0] in [0, 1, 2]
```

- [ ] **Step 2: Run test to verify failure**

Run: `.venv/bin/pytest tests/test_train.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.train'`

- [ ] **Step 3: Implement training pipeline**

Create `src/train.py`:
```python
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from src.data_loader import load_raw_data, clean_data
from src.features import derive_features, get_clustering_features

def evaluate_elbow(X_scaled, max_k: int = 10, random_state: int = 42) -> list:
    """Compute inertia across cluster counts for Elbow evaluation."""
    inertias = []
    for k in range(2, max_k + 1):
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)
    return inertias

def plot_pca_clusters(X_scaled, labels, output_path: str):
    """Plot 2D PCA representation of clusters and save image."""
    pca = PCA(n_components=2, random_state=42)
    pca_coords = pca.fit_transform(X_scaled)
    df_pca = pd.DataFrame(pca_coords, columns=['PCA1', 'PCA2'])
    df_pca['Cluster'] = labels
    
    plt.figure(figsize=(9, 6))
    sns.scatterplot(
        x='PCA1', y='PCA2', hue='Cluster',
        data=df_pca, palette='tab10', alpha=0.8, s=40
    )
    plt.title("Customer Segmentation — 2D PCA Cluster Representation")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

def build_and_train_pipeline(data_path: str, models_dir: str = "models", n_clusters: int = 6) -> Pipeline:
    """Full execution: ingestion, feature prep, model fit, evaluation, and export."""
    os.makedirs(models_dir, exist_ok=True)
    
    df_raw = load_raw_data(data_path)
    df_clean = clean_data(df_raw)
    df_feat = derive_features(df_clean)
    X = get_clustering_features(df_feat)
    
    # Unified Pipeline: StandardScaler + KMeans
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('kmeans', KMeans(n_clusters=n_clusters, random_state=42, n_init=10))
    ])
    pipeline.fit(X)
    
    # Generate PCA Plot
    scaler = pipeline.named_steps['scaler']
    kmeans = pipeline.named_steps['kmeans']
    X_scaled = scaler.transform(X)
    plot_pca_clusters(X_scaled, kmeans.labels_, os.path.join(models_dir, "pca_clusters.png"))
    
    # Save Pipeline artifact
    pipeline_path = os.path.join(models_dir, "pipeline.pkl")
    joblib.dump(pipeline, pipeline_path)
    print(f"Model successfully saved to {pipeline_path}")
    
    return pipeline

if __name__ == "__main__":
    dataset_path = "data/customer_segmentation.csv"
    if os.path.exists(dataset_path):
        build_and_train_pipeline(dataset_path)
    else:
        print(f"Error: {dataset_path} not found.")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_train.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Run training on real dataset**

Run: `.venv/bin/python -m src.train`
Expected:
```text
Model successfully saved to models/pipeline.pkl
```
Verify files:
- `models/pipeline.pkl` exists
- `models/pca_clusters.png` exists

- [ ] **Step 6: Commit**

```bash
git add src/train.py tests/test_train.py
git commit -m "feat: implement training pipeline, PCA plotting, and model serialization"
```

---

### Task 5: Interactive Streamlit Web Application (`app.py`)

**Files:**
- Create: `app.py`
- Create: `tests/test_app.py`

**Interfaces:**
- Consumes: `models/pipeline.pkl`
- Produces: Web UI serving predictions and operational persona mappings.

- [ ] **Step 1: Write unit test for persona mapping and inference format**

Create `tests/test_app.py`:
```python
import joblib
import pandas as pd
from app import PERSONA_MAPPINGS, predict_customer_segment
from src.features import CLUSTERING_FEATURES

def test_persona_mappings_complete():
    for cluster_id in range(6):
        assert cluster_id in PERSONA_MAPPINGS
        persona = PERSONA_MAPPINGS[cluster_id]
        assert "title" in persona
        assert "description" in persona

def test_predict_customer_segment():
    pipeline = joblib.load("models/pipeline.pkl")
    input_data = {
        'Age': 45,
        'Income': 75000.0,
        'Total_Spending': 1200.0,
        'NumWebPurchases': 8,
        'NumStorePurchases': 10,
        'NumWebVisitsMonth': 3,
        'Recency': 20
    }
    cluster_id = predict_customer_segment(pipeline, input_data)
    assert 0 <= cluster_id < 6
```

- [ ] **Step 2: Run test to verify failure**

Run: `.venv/bin/pytest tests/test_app.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app'`

- [ ] **Step 3: Implement Streamlit application `app.py`**

Create `app.py`:
```python
import os
import joblib
import pandas as pd
import streamlit as st
from src.features import CLUSTERING_FEATURES

PERSONA_MAPPINGS = {
    0: {
        "title": "Cluster 0: Mature Baseline Demographic",
        "description": "Mature demographic with balanced web and direct in-store purchasing frequency and moderate overall expenditure."
    },
    1: {
        "title": "Cluster 1: High-Spending Omnichannel Shoppers",
        "description": "High-income segment with high product spend and strong transaction frequency across both physical and digital channels."
    },
    2: {
        "title": "Cluster 2: Digital Frequent Buyers",
        "description": "Tech-savvy demographic with high web platform engagement and significant digital catalog purchases."
    },
    3: {
        "title": "Cluster 3: Premium High-Spenders",
        "description": "Concentrated high-value customers with maximum spend across luxury product lines (wines and meats) and low price sensitivity."
    },
    4: {
        "title": "Cluster 4: Budget Web Visitors",
        "description": "Price-sensitive segment characterized by lower income, minimal spend, and the highest monthly web browsing frequency."
    },
    5: {
        "title": "Cluster 5: Young High-Value Drivers",
        "description": "Younger affluent cohort exhibiting strong upward income trajectory and rapid adoption of premium services."
    }
}

def predict_customer_segment(pipeline, input_dict: dict) -> int:
    """Structure input dictionary into DataFrame and run prediction."""
    df_input = pd.DataFrame([input_dict])[CLUSTERING_FEATURES]
    prediction = pipeline.predict(df_input)
    return int(prediction[0])

def main():
    st.set_page_config(page_title="Customer Segmentation AI", page_icon="🎯", layout="wide")
    
    st.title("🎯 Customer Personality Segmentation Engine")
    st.write("Enter prospective customer metrics below to predict their behavioral cluster and operational marketing persona.")
    
    model_path = "models/pipeline.pkl"
    if not os.path.exists(model_path):
        st.error(f"Model file '{model_path}' not found! Please run `python -m src.train` first.")
        return
        
    pipeline = joblib.load(model_path)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Demographics & Financials")
        age = st.number_input("Customer Age", min_value=18, max_value=100, value=35, step=1)
        income = st.number_input("Annual Household Income ($)", min_value=0.0, max_value=250000.0, value=50000.0, step=1000.0)
        total_spending = st.number_input("Total Product Spending ($)", min_value=0.0, max_value=10000.0, value=1000.0, step=50.0)
        recency = st.number_input("Days Since Last Purchase (Recency)", min_value=0, max_value=365, value=30, step=1)
        
    with col2:
        st.subheader("Channel Activity")
        num_web_purchases = st.number_input("Number of Web Purchases", min_value=0, max_value=100, value=10, step=1)
        num_store_purchases = st.number_input("Number of Store Purchases", min_value=0, max_value=100, value=10, step=1)
        num_web_visits_month = st.number_input("Monthly Web Visits", min_value=0, max_value=50, value=3, step=1)
        
        st.write("")
        st.write("")
        predict_btn = st.button("🚀 Predict Customer Segment", use_container_width=True, type="primary")

    if predict_btn:
        input_data = {
            'Age': age,
            'Income': income,
            'Total_Spending': total_spending,
            'NumWebPurchases': num_web_purchases,
            'NumStorePurchases': num_store_purchases,
            'NumWebVisitsMonth': num_web_visits_month,
            'Recency': recency
        }
        
        cluster_id = predict_customer_segment(pipeline, input_data)
        persona = PERSONA_MAPPINGS.get(cluster_id, {"title": f"Cluster {cluster_id}", "description": "General segment."})
        
        st.divider()
        st.success(f"### {persona['title']}")
        st.info(persona['description'])
        
        if os.path.exists("models/pca_clusters.png"):
            with st.expander("📊 View Overall 2D PCA Cluster Distribution"):
                st.image("models/pca_clusters.png", use_container_width=True)

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_app.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_app.py
git commit -m "feat: implement Streamlit interactive customer segmentation app"
```

---

### Task 6: Full Verification & Integration Test Execution

**Files:**
- Test: All tests in `tests/`
- Documentation: `README.md`

- [ ] **Step 1: Run comprehensive test suite**

Run: `.venv/bin/pytest -v`
Expected: All tests pass (tests for data loading, features, train, and app).

- [ ] **Step 2: Create project README.md**

Create `README.md` with:
- Project overview
- Setup instructions (`python3 -m venv .venv`, `pip install -r requirements.txt`)
- Model training command (`python -m src.train`)
- Web application launch command (`streamlit run app.py`)
- Test runner command (`pytest -v`)

- [ ] **Step 3: Commit and Push**

```bash
git add README.md
git commit -m "docs: add project README with setup and execution instructions"
git push origin main
```
