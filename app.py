import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Fraud Detection AI", page_icon="🚨", layout="wide")

@st.cache_resource(show_spinner="Loading AI Model...")
def load_artifacts():
    try:
        model = joblib.load('fraud_model.pkl')
        scaler = joblib.load('scaler.pkl')
        feature_cols = joblib.load('features.pkl')
        return model, scaler, feature_cols
    except FileNotFoundError:
        st.error("⚠️ AI Model files not found! Please ensure fraud_model.pkl, scaler.pkl, and features.pkl are uploaded.")
        st.stop()

model, scaler, feature_cols = load_artifacts()

st.title("Credit Card Fraud Detection System 🚨")
st.markdown("""
Welcome to the AI-powered fraud detection portal. 
Upload a CSV file containing new transaction logs to scan them for fraudulent activity.
""")

uploaded_file = st.file_uploader("Upload Transactions CSV for Prediction", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    data = data.dropna(how='all') 
    
    st.write("### Uploaded Data Preview:")
    st.dataframe(data.head())
    
    if 'Class' in data.columns:
        X_infer = data.drop('Class', axis=1)
    else:
        X_infer = data.copy()
        
    if st.button("Scan for Fraud"):
        with st.spinner('Analyzing transactions...'):
            
            X_infer = X_infer.fillna(0)
            
            if 'Time' in X_infer.columns and 'Amount' in X_infer.columns:
                X_infer[['Time', 'Amount']] = scaler.transform(X_infer[['Time', 'Amount']])
                
            for col in feature_cols:
                if col not in X_infer.columns:
                    X_infer[col] = 0  
            X_final = X_infer[feature_cols] 
                
            predictions = model.predict(X_final)
            prediction_probs = model.predict_proba(X_final)[:, 1]
            
            data['Fraud_Prediction'] = predictions
            data['Fraud_Probability (%)'] = (prediction_probs * 100).round(2)
            frauds = data[data['Fraud_Prediction'] == 1.0]
            
            st.divider()
            
            if len(frauds) > 0:
                st.error(f"⚠️ {len(frauds)} fraudulent transactions detected!")
                st.dataframe(frauds.style.highlight_max(axis=0, color='red', subset=['Fraud_Probability (%)']))
            else:
                st.success("✅ No fraudulent transactions detected in this batch. All clear!")
                
            csv = data.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Complete Report",
                data=csv,
                file_name="fraud_detection_report.csv",
                mime="text/csv"
            )