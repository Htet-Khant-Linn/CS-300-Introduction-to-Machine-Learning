import streamlit as st
import pickle
import pandas as pd

st.header("Loan Prediction Apps")

with open('loan_prediction.pkl','rb') as f:
    model=pickle.load(f)

with st.sidebar:
    st.image("parami_logo.jfif", caption="ML project")
    st.text("Student Name: John")
    st.text("Student ID: PUI....")
    st.text("Class Name: Introduction to ML")
    

col1,col2=st.columns(2)

with col1:
    no_of_dependents = st.number_input(
        "Number of Dependents", min_value=0, value=5
    )

    education = st.selectbox(
        "Education", [" Graduate", " Not Graduate"]
    )

    self_employed = st.selectbox(
        "Self Employed", [" No", " Yes"]
    )

    income_annum = st.number_input(
        "Annual Income", min_value=0, value=5400000
    )

    loan_term = st.number_input(
        "Loan Term", min_value=1, value=20
    )


with col2:
    cibil_score = st.number_input(
    "CIBIL Score", min_value=300, max_value=900, value=423
    )

    residential_assets_value = st.number_input(
        "Residential Assets Value", min_value=0, value=6500000
    )

    commercial_assets_value = st.number_input(
        "Commercial Assets Value", min_value=0, value=10000000
    )

    luxury_assets_value = st.number_input(
        "Luxury Assets Value", min_value=0, value=15700000
    )

    bank_asset_value = st.number_input(
        "Bank Asset Value", min_value=0, value=7300000
    )


if st.button("Predict Loan Amount"):

    input_data = pd.DataFrame([{
        "no_of_dependents": no_of_dependents,
        "education": education,
        "self_employed": self_employed,
        "income_annum": income_annum,
        "loan_term": loan_term,
        "cibil_score": cibil_score,
        "residential_assets_value": residential_assets_value,
        "commercial_assets_value": commercial_assets_value,
        "luxury_assets_value": luxury_assets_value,
        "bank_asset_value": bank_asset_value
    }])

    prediction = model.predict(input_data)[0]

    st.success(f"Predicted Loan Amount: {prediction:,.2f}")