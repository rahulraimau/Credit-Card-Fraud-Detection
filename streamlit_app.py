
import streamlit as st, pandas as pd, joblib, numpy as np, os, json
st.set_page_config(page_title='Fraud Demo')
st.title('Credit Card Fraud Detection - Demo')

ART = '/content/artifacts'
scaler = joblib.load(os.path.join(ART,'scaler.pkl'))
xgb_m = joblib.load(os.path.join(ART,'model_xgb.pkl'))
autoencoder = None
# Load autoencoder in the new Keras format
if os.path.exists(os.path.join(ART,'autoencoder.keras')):
    from tensorflow.keras.models import load_model
    autoencoder = load_model(os.path.join(ART,'autoencoder.keras'))

st.write('Upload CSV with columns Time, V1..V28, Amount (Class optional)')
uploaded = st.file_uploader('Upload CSV', type=['csv'])
if uploaded:
    df = pd.read_csv(uploaded)
    df_proc = df.copy()
    if 'Class' in df_proc.columns:
        df_proc = df_proc.drop(columns=['Class'])
    df_proc[['Time','Amount']] = scaler.transform(df_proc[['Time','Amount']])
    probs = xgb_m.predict_proba(df_proc)[:,1]
    df['Fraud_Prob'] = probs
    df['Prediction'] = (probs>=0.5).astype(int)
    st.dataframe(df.head(50))
