import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. Load the saved model and scaler
# (Make sure you save these using joblib.dump() in your training script)
model = joblib.load('churn_model.pkl')
scaler = joblib.load('scaler.pkl')
features = joblib.load('feature_names.pkl')

st.set_page_config(page_title="Customer Churn Predictor", layout="centered")

st.title("📊 Customer Retention Tool")
st.markdown("Enter customer details below to predict the probability of them leaving the service.")

# 2. Create Input Fields
st.sidebar.header("Customer Details")

def user_input_features():
    contract = st.sidebar.selectbox("Contract Type", ("Month-to-month", "One year", "Two year"))
    monthly_charges = st.sidebar.slider("Monthly Charges ($)", 18.0, 120.0, 70.0)
    total_charges = st.sidebar.number_input("Total Charges ($)", min_value=0.0, value=500.0)
    tenure = st.sidebar.slider("Tenure (Months)", 0, 72, 12)
    tech_support = st.sidebar.selectbox("Has Tech Support?", ("Yes", "No", "No internet service"))
    
    # Create a dictionary matching the training features
    data = {
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'tenure': tenure,
        'Contract_One year': 1 if contract == "One year" else 0,
        'Contract_Two year': 1 if contract == "Two year" else 0,
        'TechSupport_Yes': 1 if tech_support == "Yes" else 0
        # Add other dummy variables here to match your model's X_train columns
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# 3. Prediction Logic
if st.button('Predict Churn'):
    # Scale inputs
    scaled_input = scaler.transform(input_df)
    
    # Get Prediction
    prediction = model.predict(scaled_input)
    prediction_proba = model.predict_proba(scaled_input)

    # 4. Display Results
    if prediction[0] == 1:
        st.error(f"⚠️ High Risk: This customer is likely to churn.")
    else:
        st.success(f"✅ Low Risk: This customer is likely to stay.")
        
    st.write(f"**Churn Probability:** {prediction_proba[0][1]:.2%}")
