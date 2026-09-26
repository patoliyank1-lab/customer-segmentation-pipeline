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
    
    # Financial aggregate (spec defines sum of wine, fruit, meat, fish, gold)
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
