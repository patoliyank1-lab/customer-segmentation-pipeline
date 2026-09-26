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
