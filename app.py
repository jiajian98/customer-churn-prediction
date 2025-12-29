import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. Load the saved components
model = joblib.load('churn_model.pkl')
scaler = joblib.load('scaler.pkl')
features = joblib.load('feature_names.pkl')

st.set_page_config(page_title="Customer Churn Predictor", layout="centered")
st.title("📊 Customer Retention Tool")

# 2. Input Fields
st.sidebar.header("Customer Details")
contract = st.sidebar.selectbox("Contract Type", ("Month-to-month", "One year", "Two year"))
monthly_charges = st.sidebar.slider("Monthly Charges ($)", 18.0, 120.0, 70.0)
total_charges = st.sidebar.number_input("Total Charges ($)", min_value=0.0, value=500.0)
tenure = st.sidebar.slider("Tenure (Months)", 0, 72, 12)
tech_support = st.sidebar.selectbox("Has Tech Support?", ("Yes", "No", "No internet service"))

# 3. FEATURE ALIGNMENT (The most important part)
# Create a dataframe with all zeros matching the training columns
input_df = pd.DataFrame(np.zeros((1, len(features))), columns=features)

# Map user inputs to the correct columns
input_df['MonthlyCharges'] = monthly_charges
input_df['TotalCharges'] = total_charges
input_df['tenure'] = tenure

# Handle categorical 'One-Hot' columns (match exactly with your training dummy names)
if contract == "One year": input_df['Contract_One year'] = 1
if contract == "Two year": input_df['Contract_Two year'] = 1
if tech_support == "Yes": input_df['TechSupport_Yes'] = 1

# 4. Prediction Logic
if st.button('Predict Churn'):
    # Scale and Predict
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)
    prob = model.predict_proba(scaled_input)[0][1]

    if prediction[0] == 1:
        st.error(f"⚠️ High Risk: {prob:.2%} probability of churn.")
    else:
        st.success(f"✅ Low Risk: {prob:.2%} probability of churn.")
