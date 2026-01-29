import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px
import os
from pycaret.classification import load_model, predict_model


st.set_page_config(
    page_title="Bank Churn Predictor",
    page_icon="🏦",
    layout="wide"
)


st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    div[data-testid="stMetricValue"] { font-size: 1.8rem; color: #004a99; }
    .stAlert { border-radius: 10px; }
    [data-testid="stSidebar"] { border-right: 1px solid #e0e0e0; }
    .reportview-container .main .block-container { padding-top: 2rem; }
    </style>
    """, unsafe_allow_html=True)


@st.cache_resource
def load_assets():
    # Loading model and preprocessing artifacts
    model = load_model("models/catboost_classifier_model")
    scaler = joblib.load("models/scaler.pkl")
    le_gender = joblib.load("models/label_encoder_gender.pkl")
    le_country = joblib.load("models/label_encoder_country.pkl")
    return model, scaler, le_gender, le_country

try:
    model, scaler, le_gender, le_country = load_assets()
except Exception as e:
    st.error(f"Model Artifacts Not Found: {e}")
    st.stop()


def transform_data(df):
    temp_df = df.copy()
    

    if 'gender' in temp_df.columns:
        temp_df['gender'] = temp_df['gender'].astype(str).str.title()
        gender_map = {label: i for i, label in enumerate(le_gender.classes_)}
        temp_df['gender'] = temp_df['gender'].map(gender_map).fillna(-1)

    if 'country' in temp_df.columns:
        temp_df['country'] = temp_df['country'].astype(str).str.title()
        country_map = {label: i for i, label in enumerate(le_country.classes_)}
        temp_df['country'] = temp_df['country'].map(country_map).fillna(-1)

    temp_df['balance_salary_ratio'] = temp_df['balance'] / (temp_df['estimated_salary'] + 1)
    temp_df['tenure_age_ratio'] = temp_df['tenure'] / (temp_df['age'] + 1)
    temp_df['credit_score_age_ratio'] = temp_df['credit_score'] / (temp_df['age'] + 1)
    temp_df['balance_per_product'] = temp_df['balance'] / (temp_df['products_number'] + 1)
    temp_df['is_senior'] = temp_df['age'].apply(lambda x: 1 if x >= 60 else 0)
    
    
    cols_to_scale = [
        'credit_score', 'age', 'tenure', 'balance', 'products_number', 
        'estimated_salary', 'balance_salary_ratio', 'tenure_age_ratio', 
        'credit_score_age_ratio', 'balance_per_product'
    ]
    existing_cols = [c for c in cols_to_scale if c in temp_df.columns]
    temp_df[existing_cols] = scaler.transform(temp_df[existing_cols])
    
    return temp_df

#SIDEBAR: ---
with st.sidebar:
    st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
    st.image("https://cdn-icons-png.flaticon.com/512/2830/2830284.png", width=80)
    st.title('Bank Customer Churn Predictor')
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.divider()
    app_mode = st.radio("Navigation", ["Single Prediction", "Batch Analysis", "Data Insights"])
    st.divider()
    
    #User-defined threshold for Risk Sensitivity
    risk_threshold = st.slider("Churn Sensitivity Threshold", 0.05, 0.95, 0.5, 0.05,
                              help=(
        "Adjust the threshold to control how sensitive the model is to predicting churn. "
        "Lower values increase the chance of flagging customers at risk (higher recall) "
        "but may produce more false positives. "
        "Recommended: 0.5 to balance risk detection and accuracy."
    ))
    
    st.divider()
    st.caption("v1.2.0 | Engine: CatBoost")
    st.caption("Built by Rajeev Kumar")

#MODE 1: SINGLE PREDICTION:---
if app_mode == "Single Prediction":
    st.title("📊 Individual Risk Assessment")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        with st.container(border=True):
            st.subheader("Customer Input Parameters")
            c_a, c_b = st.columns(2)
            with c_a:
                country = st.selectbox("Geography", ["France", "Spain", "Germany"])
                gender = st.radio("Gender", ["Male", "Female"], horizontal=True)
                age = st.number_input("Age", 18, 100, 35)
                tenure = st.slider("Tenure (Years)", 0, 10, 5)
            with c_b:
                credit_score = st.slider("Credit Score", 300, 850, 650)
                balance = st.number_input("Balance ($)", 0.0, 300000.0, 10000.0)
                products = st.number_input("Number of Products", 1, 4, 1)
                salary = st.number_input("Estimated Salary ($)", 0.0, 300000.0, 50000.0)
            
            st.divider()
            c_c, c_d = st.columns(2)
            active = c_c.toggle("Is Active Member?", value=True)
            has_card = c_d.toggle("Has Credit Card?", value=True)

    with col2:
        st.subheader("Analysis Result")
        if st.button("Run Prediction", use_container_width=True):
            input_df = pd.DataFrame([{
                'credit_score': credit_score, 'country': country, 'gender': gender, 'age': age,
                'tenure': tenure, 'balance': balance, 'products_number': products,
                'credit_card': 1 if has_card else 0, 'active_member': 1 if active else 0,
                'estimated_salary': salary
            }])
            
            # Predict
            processed = transform_data(input_df)
            prediction = predict_model(model, data=processed)
            
            # Calculate Probability
            prob = prediction['prediction_score'].iloc[0] if prediction['prediction_label'].iloc[0] == 1 else (1 - prediction['prediction_score'].iloc[0])
            is_churn = 1 if prob >= risk_threshold else 0
            
            # Visual Feedback
            if is_churn == 1:
                st.error(f"### ALERT: High Risk ({prob:.1%})")
                st.write("Customer meets threshold for churn intervention.")
            else:
                st.success(f"### Low Risk ({prob:.1%})")
                st.write("Customer is likely to remain with the bank.")

            # Gauge Chart
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = prob * 100,
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#ef4444" if is_churn == 1 else "#22c55e"},
                    'steps': [{'range': [0, risk_threshold*100], 'color': "#e8f5e9"}, 
                              {'range': [risk_threshold*100, 100], 'color': "#ffebee"}]
                }
            ))
            fig.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)

#MODE 2: BATCH ANALYSIS:---
elif app_mode == "Batch Analysis":
    st.title("📂 Bulk Processing Engine")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file:
        raw_df = pd.read_csv(uploaded_file)
        if st.button("Apply Batch Intelligence", use_container_width=True):
            with st.spinner("Processing records..."):
                try:
                    processed_batch = transform_data(raw_df)
                    results = predict_model(model, data=processed_batch)
                    
                    # Apply Threshold Correction
                    results['churn_probability'] = results.apply(lambda r: r['prediction_score'] if r['prediction_label'] == 1 else (1 - r['prediction_score']), axis=1)
                    results['final_prediction'] = (results['churn_probability'] >= risk_threshold).astype(int)
                    
                    st.divider()
                    churn_count = results['final_prediction'].sum()
                    rate = (churn_count / len(results)) * 100
                    
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Processed", len(results))
                    m2.metric("Flagged Churners", churn_count)
                    m3.metric("Churn Rate", f"{rate:.1f}%")
                    
                    st.subheader("Analysis Preview")
                    st.dataframe(results[['churn_probability', 'final_prediction']].join(raw_df).sort_values(by='churn_probability', ascending=False).head(10))
                    
                    csv = results.to_csv(index=False).encode('utf-8')
                    st.download_button("📥 Export Analysis Report", data=csv, file_name="Churn_Predicted.csv")
                except Exception as e:
                    st.error(f"Processing Error: {e}")

#MODE 3: MODEL INSIGHTS ---
elif app_mode == "Data Insights":
    st.title("🧠 Dataset Insights")
    
    # Feature Importance (Simplified for View)
    feat_data = pd.DataFrame({
        'Feature': ['Age', 'NumProducts', 'IsActive', 'Balance', 'Geography', 'CreditScore', 'Salary'],
        'Weight': [45, 25, 15, 8, 4, 2, 1]
    }).sort_values('Weight')
    
    fig = px.bar(feat_data, x='Weight', y='Feature', orientation='h', color='Weight', 
                 title="Key Factors Influencing Churn", color_continuous_scale='Blues')
    st.plotly_chart(fig, use_container_width=True)
    
    st.info(f"""
    **Threshold Strategy:**
    A sensitivity of **{risk_threshold}** is currently active. 
    This means the model prioritizes **Recall**, ensuring that the bank minimizes missing potential churners 
    at the cost of a slightly higher false-alarm rate.
    """)

st.divider()
st.caption("Bank Customer Churn Prediction & Analysis Project (2026) | Developed by Rajeev Kumar")
