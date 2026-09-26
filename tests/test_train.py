import os
import joblib
import pandas as pd
import pytest

os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

from src.train import build_and_train_pipeline
from src.features import CLUSTERING_FEATURES

def test_build_and_train_pipeline(tmp_path):
    # Create minimal valid CSV with enough samples for 3 clusters
    rows = []
    for i in range(15):
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
