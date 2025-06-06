import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

model = joblib.load("fraud_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🚨 Fraud Detection in Last-Mile Delivery")

uploaded_file = st.file_uploader("Upload delivery data CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    df['pickup_time'] = pd.to_datetime(df['pickup_time'])
    df['delivery_time'] = pd.to_datetime(df['delivery_time'])
    df['dropzone_time'] = pd.to_datetime(df['dropzone_time'])

    df['delivery_duration_min'] = (df['delivery_time'] - df['pickup_time']).dt.total_seconds() / 60
    df['dropzone_lag_min'] = (df['delivery_time'] - df['dropzone_time']).dt.total_seconds() / 60
    df['delivery_speed_kmph'] = df['distance_km'] / (df['delivery_duration_min'] / 60)

    features = ['delivery_duration_min', 'dropzone_lag_min', 'delivery_speed_kmph', 'distance_km']
    X = df[features].replace([np.inf, -np.inf], np.nan).fillna(method='bfill')
    X_scaled = scaler.transform(X)

    predictions = model.predict(X_scaled)
    df['predicted_fraud'] = predictions

    st.write("### 📊 Fraud Prediction Results")
    st.dataframe(df[['order_id', 'rider_id', 'predicted_fraud']])

    st.write("### 📈 Fraud Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x='predicted_fraud', data=df, ax=ax)
    ax.set_xticklabels(['Not Fraud', 'Fraud'])
    st.pyplot(fig)

    st.success(f"🚨 Detected {df['predicted_fraud'].sum()} suspected frauds out of {len(df)} orders.")
