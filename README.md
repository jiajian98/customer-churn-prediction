# 📊 Customer Churn Prediction & Deployment

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Live-red)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange)

## 🎯 Project Overview
This project predicts the likelihood of a customer leaving a service provider (Churn). This tool helps businesses identify high-risk customers early to offer retention incentives.

## 🎯 Project Objective
The objective of this project is to build a predictive system that identifies customers who are likely to cancel their subscription. By predicting churn before it happens, businesses can take proactive measures (such as targeted discounts or improved support) to retain valuable users.

### [Live Web App Link (https://customer-churn-prediction-bikdbu4g2tfzanfkfmf4it.streamlit.app/)]

## 🛠️ The Tech Stack
* **Analysis:** Pandas, Seaborn, Matplotlib
* **Machine Learning:** Scikit-Learn (Random Forest Classifier)
* **Imbalance Handling:** SMOTE (Synthetic Minority Over-sampling Technique)
* **Deployment:** Streamlit & Joblib

## 🚀 Key Features
1.  **Data Cleaning Pipeline:** Handled missing values in `TotalCharges` and performed feature encoding.
2.  **Feature Engineering:** Created a `TotalServices` metric to measure "product stickiness."
3.  **Class Imbalance Management:** Used SMOTE to ensure the model accurately identifies the minority churn class.
4.  **Interactive Dashboard:** A live web interface where users can adjust customer parameters and see real-time churn probabilities.

## 🚀 Key Technical Features
* **Imbalance Handling:** Utilized SMOTE (Synthetic Minority Over-sampling Technique) to ensure the model learns effectively from the minority 'Churn' class.
* **Machine Learning Pipeline:** Built and tuned a Random Forest Classifier, optimizing for Recall to ensure as many potential churners as possible are caught.
* **Interactive Deployment:** Developed a user-facing web app using Streamlit that allows real-time predictions based on user-inputted customer data.

## 📈 Model Performance
* **ROC-AUC Score:** 0.82
* **Key Insight:** Month-to-month contracts and high monthly charges were the strongest predictors of churn.

## 📂 Project Structure
* `app.ipynb`: Full exploratory data analysis and model training.
* `app.py`: The production-ready Streamlit application.
* `churn_model.pkl`: The serialized Random Forest model.
* `requirements.txt`: Necessary libraries for deployment.

## 🔧 How to Run Locally
1. Clone the repo: `git clone [https://github.com/jiajian98/customer-churn-prediction]]`
2. Install requirements: `pip install -r requirements.txt`
3. Run the app: `streamlit run app.py`
