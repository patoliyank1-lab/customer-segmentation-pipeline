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
    assert derived.loc[0, "Total_Spending"] == 440  # 200+20+150+30+40 (excluding sweet as per spec or including? Note: MntWines+MntFruits+MntMeat+MntFish+MntGold)
    assert derived.loc[0, "Accepted_Any"] == 1
    assert "Customer_Since" in derived.columns

def test_get_clustering_features():
    df = sample_cleaned_data()
    derived = derive_features(df, reference_year=2025)
    X = get_clustering_features(derived)
    
    assert list(X.columns) == CLUSTERING_FEATURES
    assert len(X) == 1
