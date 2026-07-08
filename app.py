import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
import shap
import matplotlib.pyplot as plt

st.title("Предсказание оттока клиентов")

st.sidebar.title("Введите данные клиента")

contract = st.sidebar.radio("Выберите контракт рабочего:", ["Month-to-month", "One year", "Two year"])
internet_service = st.sidebar.radio("Выберите тип интернет-сервиса:", ["DSL", "Fiber optic", "No"])
monthly_charges = st.sidebar.number_input("Введите ежемесячные расходы:", min_value=0.0, step=0.1, format="%.2f")
total_charges = st.sidebar.number_input("Введите общие расходы:", min_value=0.0, step=0.05, format="%.6f")
online_security = st.sidebar.radio("Выберите наличие онлайн-безопасности:", ["Yes", "No", "No internet service"])
tech_support = st.sidebar.radio("Выберите наличие технической поддержки:", ["Yes", "No", "No internet service"])
payment_method = st.sidebar.radio("Выберите способ оплаты:", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
tenure = st.sidebar.number_input("Введите срок работы (в месяцах):", min_value=0, step=1)
dependents = st.sidebar.radio("Выберите наличие зависимости от работодателя:", ["Yes", "No"])
senior_citizen = st.sidebar.radio("Выберите, является ли рабочий пожилым:", ["Yes", "No"])
partner = st.sidebar.radio("Выберите наличие партнера:", ["Yes", "No"])

if st.button("Предсказать отток"):
    input_data = pd.DataFrame({
    "gender": ["Male"],
    "SeniorCitizen": [senior_citizen],
    "Partner": [partner],
    "Dependents": [dependents],
    "tenure": [tenure],
    "PhoneService": ["Yes"],
    "MultipleLines": ["No"],
    "InternetService": [internet_service],
    "OnlineSecurity": [online_security],
    "OnlineBackup": ["No"],
    "DeviceProtection": ["No"],
    "TechSupport": [tech_support],
    "StreamingTV": ["No"],
    "StreamingMovies": ["No"],
    "Contract": [contract],
    "PaperlessBilling": ["Yes"],
    "PaymentMethod": [payment_method],
    "MonthlyCharges": [monthly_charges],
    "TotalCharges": [total_charges],
    "TenureGroup": ["2-4y"],
    "MonthlyToTotalRatio": np.where(total_charges == 0, 
                                   monthly_charges / (total_charges + 1), 
                                   monthly_charges / total_charges),
    "TotalServices": [5],
    "IsLoyal": [(tenure > 24) & (contract != "Month-to-month")]
    })

    model = CatBoostClassifier().load_model("churn_model.cbm")

    pred_proba = model.predict_proba(input_data)[:, 1][0]
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_data)
    list_most = np.abs(shap_values).mean(axis=0)
    
    plt.style.use("seaborn-v0_8-poster")
    fig = plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, input_data, plot_type="bar", show=False)
    
    if pred_proba >= 0.7:
        st.error("Вероятность оттока высокая")

    elif pred_proba < 0.7 and pred_proba >= 0.4:
        st.warning("Вероятность оттока средняя")

    else:
        st.success("Вероятность оттока низкая")

    st.info(f"Вероятность оттока: {pred_proba:.2f}, наиболее важный признак: {input_data.columns[np.argmax(list_most)]}")
    st.pyplot(fig)