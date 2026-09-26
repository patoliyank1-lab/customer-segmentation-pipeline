import os
import joblib
import pandas as pd
import pytest

from app import PERSONA_MAPPINGS, predict_customer_segment
from src.features import CLUSTERING_FEATURES

def test_persona_mappings_complete():
    for cluster_id in range(6):
        assert cluster_id in PERSONA_MAPPINGS
        persona = PERSONA_MAPPINGS[cluster_id]
        assert "title" in persona
        assert "description" in persona

def test_predict_customer_segment():
    model_path = "models/pipeline.pkl"
    assert os.path.exists(model_path)
    pipeline = joblib.load(model_path)
    
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
