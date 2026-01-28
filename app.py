import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="TrueSource | Churn Analytics",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a polished look
# Change this part in your app.py
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #007bff;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True) # Corrected parameter name

# -------------------------------
# Load Model
# -------------------------------
@st.cache_resource
def load_model():
    # Using cache_resource for the model object
    return joblib.load("models/logistic_regression_churn.pkl")

try:
    model = load_model()
except Exception as e:
    st.error("Model file not found. Please ensure the path is correct.")

# -------------------------------
# Sidebar & About Me
# -------------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=100)
    st.title("Project TrueSource")
    st.info("Advancing data integrity and predictive transparency.")
    
    st.divider()
    
    st.subheader("👨‍💻 About Me")
    st.markdown("""
    **Lead Developer** Specializing in Machine Learning Operations (MLOps) and FinTech solutions. 
    
    * **Project:** [TrueSource](https://github.com)
    * **Focus:** Predictive Analytics & Data Truth
    * **Tech:** Python, PyCaret, Streamlit, Scikit-Learn
    """)
    
    st.divider()
    st.caption("© 2026 TrueSource Analytics v2.1")

# -------------------------------
# Main UI Logic
# -------------------------------
st.title("🏦 Customer Churn Intelligence Portal")
st.markdown("Extracting actionable insights from customer behavior data.")

tabs = st.tabs(["🔍 Prediction Tool", "📊 Analytics Dashboard", "📄 Documentation"])

with tabs[0]:
    # Organizing inputs into columns for a "Dashboard" feel
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Customer Demographics & Financials")
        c1, c2 = st.columns(2)
        with c1:
            credit_score = st.slider("Credit Score", 300, 850, 650)
            country = st.selectbox("Geography", ["France", "Spain", "Germany"])
            gender = st.radio("Gender", ["Male", "Female"], horizontal=True)
            age = st.number_input("Age", 18, 100, 35)
        with c2:
            balance = st.number_input("Account Balance ($)", 0.0, 300000.0, 50000.0)
            estimated_salary = st.number_input("Estimated Salary ($)", 0.0, 250000.0, 50000.0)
            tenure = st.slider("Tenure (Years)", 0, 10, 5)
            products_number = st.selectbox("Number of Products", [1, 2, 3, 4])

        st.subheader("Engagement Metrics")
        e1, e2 = st.columns(2)
        with e1:
            credit_card = st.toggle("Has Credit Card", value=True)
        with e2:
            active_member = st.toggle("Is Active Member", value=False)

    with col2:
        st.subheader("Prediction Results")
        
        # Build Input Data
        input_df = pd.DataFrame([{
            "CreditScore": credit_score,
            "Geography": country,
            "Gender": gender,
            "Age": age,
            "Tenure": tenure,
            "Balance": balance,
            "NumOfProducts": products_number,
            "HasCrCard": 1 if credit_card else 0,
            "IsActiveMember": 1 if active_member else 0,
            "EstimatedSalary": estimated_salary
        }])

        if st.button("Run Risk Analysis"):
            # Probability calculation
            prob = model.predict_proba(input_df)[0][1]
            risk_level = "HIGH" if prob >= 0.35 else "LOW"
            color = "red" if risk_level == "HIGH" else "green"

            # Gauge Chart for Risk
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = prob * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': f"Churn Risk: {risk_level}", 'font': {'color': color}},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': color},
                    'steps': [
                        {'range': [0, 35], 'color': "lightgreen"},
                        {'range': [35, 100], 'color': "pink"}],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': 35}
                }
            ))
            fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig, use_container_width=True)

            if risk_level == "HIGH":
                st.warning(f"**Action Required:** Customer has a {prob:.1%} probability of leaving.")
            else:
                st.success(f"**Healthy Status:** Customer retention likely ({prob:.1%}).")

with tabs[1]:
    st.subheader("Dataset Context")
    st.info("In a production environment, this tab would show global churn trends and model drift metrics.")
    # Example metric row
    m1, m2, m3 = st.columns(3)
    m1.metric("Model Recall", "84%", "+2%")
    m2.metric("Avg. Churn Risk", "22%", "-1.5%")
    m3.metric("Data Freshness", "Live", "Stable")

with tabs[2]:
    st.markdown("""
    ### Technical Implementation
    - **Model:** Logistic Regression (Optimized via PyCaret)
    - **Threshold:** 0.35 (Prioritizing Recall to capture potential churners)
    - **Preprocessing:** Categorical encoding and Feature Scaling handled by Pipeline.
    
    ### About TrueSource
    This application is a module of the **TrueSource** ecosystem, focused on creating verifiable and transparent AI decision-making tools for the banking sector.
    """)