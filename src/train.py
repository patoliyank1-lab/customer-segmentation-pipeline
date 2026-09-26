import os
import joblib
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib
matplotlib.use("Agg")
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
