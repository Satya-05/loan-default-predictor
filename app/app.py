import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.datasets import fetch_openml

# Page config
st.set_page_config(
    page_title="Loan Default Predictor",
    page_icon="🏦",
    layout="wide"
)

# Base directory path fix for deployment
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Title
st.title("🏦 Loan Default Predictor")
st.markdown("Predict whether a loan applicant is likely to default using Machine Learning")
st.markdown("---")

# Load model
@st.cache_resource
def load_model():
    model = joblib.load(os.path.join(BASE_DIR, 'models', 'xgb_model.pkl'))
    return model

# Load and preprocess data
@st.cache_data
def load_data():
    credit = fetch_openml(name='credit-g', version=1, as_frame=True)
    df = credit.data.copy()
    
    categorical_cols = ['checking_status', 'credit_history', 'purpose',
                        'savings_status', 'employment', 'personal_status',
                        'other_parties', 'property_magnitude', 'other_payment_plans',
                        'housing', 'job', 'own_telephone', 'foreign_worker']
    
    numerical_cols = ['duration', 'credit_amount', 'installment_commitment',
                      'residence_since', 'age', 'existing_credits', 'num_dependents']
    
    le = LabelEncoder()
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col].astype(str))
    
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    return df, categorical_cols, numerical_cols, scaler

model = load_model()
df, categorical_cols, numerical_cols, scaler = load_data()

# Sidebar inputs
st.sidebar.header("📋 Applicant Information")

duration = st.sidebar.slider("Loan Duration (months)", 6, 72, 24)
credit_amount = st.sidebar.number_input("Credit Amount (DM)", 500, 20000, 5000)
age = st.sidebar.slider("Age", 18, 75, 35)
installment_commitment = st.sidebar.slider("Installment Commitment (%)", 1, 4, 2)
residence_since = st.sidebar.slider("Residence Since (years)", 1, 4, 2)
existing_credits = st.sidebar.slider("Existing Credits", 1, 4, 1)
num_dependents = st.sidebar.slider("Number of Dependents", 1, 2, 1)

checking_status = st.sidebar.selectbox("Checking Account Status",
                                        ['no checking', 'less than 0 DM',
                                         '0 to 200 DM', 'greater than 200 DM'])
credit_history = st.sidebar.selectbox("Credit History",
                                       ['no credits', 'all paid', 'existing paid',
                                        'delayed previously', 'critical/other existing credit'])
purpose = st.sidebar.selectbox("Loan Purpose",
                                ['new car', 'used car', 'furniture/equipment',
                                 'radio/tv', 'domestic appliance', 'repairs',
                                 'education', 'vacation', 'retraining', 'business', 'other'])
savings_status = st.sidebar.selectbox("Savings Account Status",
                                       ['no known savings', 'less than 100 DM',
                                        '100 to 500 DM', '500 to 1000 DM', 'greater than 1000 DM'])
employment = st.sidebar.selectbox("Employment Since",
                                   ['unemployed', 'less than 1 year', '1 to 4 years',
                                    '4 to 7 years', 'greater than 7 years'])

# Predict button
if st.sidebar.button("🔍 Predict", use_container_width=True):
    
    # Prepare input
    le = LabelEncoder()
    
    input_data = pd.DataFrame({
        'checking_status': [checking_status],
        'duration': [duration],
        'credit_history': [credit_history],
        'purpose': [purpose],
        'credit_amount': [credit_amount],
        'savings_status': [savings_status],
        'employment': [employment],
        'installment_commitment': [installment_commitment],
        'personal_status': ['male single'],
        'other_parties': ['none'],
        'residence_since': [residence_since],
        'property_magnitude': ['real estate'],
        'age': [age],
        'other_payment_plans': ['none'],
        'housing': ['own'],
        'existing_credits': [existing_credits],
        'job': ['skilled'],
        'num_dependents': [num_dependents],
        'own_telephone': ['yes'],
        'foreign_worker': ['yes']
    })

    # Encode categorical
    for col in categorical_cols:
        input_data[col] = le.fit_transform(input_data[col].astype(str))

    # Scale numerical
    input_data[numerical_cols] = scaler.transform(input_data[numerical_cols])

    # Predict
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    # Show results
    col1, col2, col3 = st.columns(3)

    with col1:
        if prediction == 1:
            st.error("### ❌ High Risk — Likely to Default")
        else:
            st.success("### ✅ Low Risk — Unlikely to Default")

    with col2:
        st.metric("Default Probability", f"{probability[1]*100:.1f}%")

    with col3:
        st.metric("Safe Loan Probability", f"{probability[0]*100:.1f}%")

    st.markdown("---")

    # SHAP explanation
    st.subheader("🔍 Why this prediction?")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_data)

    fig, ax = plt.subplots(figsize=(10, 4))
    shap.summary_plot(shap_values, input_data, plot_type="bar", show=False)
    st.pyplot(fig)

else:
    # Default view
    st.subheader("📊 Model Performance")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Best Model", "XGBoost")
    with col2:
        st.metric("Accuracy", "82.86%")
    with col3:
        st.metric("ROC-AUC Score", "0.9025")

    st.markdown("---")
    st.subheader("ℹ️ How to use")
    st.markdown("""
    1. Fill in the applicant details in the **left sidebar**
    2. Click the **Predict button**
    3. See the prediction with probability scores
    4. View **SHAP explanation** to understand why the model made that prediction
    """)

    st.subheader("📈 Top Features Driving Predictions")
    shap_img_path = os.path.join(BASE_DIR, 'data', 'shap_summary.png')
    if os.path.exists(shap_img_path):
        st.image(shap_img_path)