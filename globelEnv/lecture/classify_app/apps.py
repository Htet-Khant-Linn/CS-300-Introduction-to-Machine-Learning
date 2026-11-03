import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title='Loan application')

df=pd.read_csv('loan_approval_dataset.csv')
df.columns=df.columns.str.strip()


st.header("Loan Prediction Application")


def load_model():
    with open('loan_classify.pkl','rb') as f:
        return pickle.load(f)


with st.form("loan_form"):

    col1,col2=st.columns(2)
    with col1:
        no_of_dependents = st.number_input(
            "Number of Dependents", min_value=0, max_value=10, value=0
        )

        education = st.selectbox(
            "Education Level",
            df['education'].dropna().unique().tolist()
        )

        self_employed = st.selectbox(
            "Self Employed",
            df['self_employed'].dropna().unique().tolist()
        )

        income_annum = st.number_input(
            "Annual Income", min_value=0, value=0
        )

        loan_amount = st.number_input(
            "Loan Amount", min_value=0, value=0
        )

        loan_term = st.number_input(
            "Loan Term (in months)", min_value=0, value=0
        )

    with col2:
        cibil_score = st.number_input(
            "CIBIL Score", min_value=0, max_value=900, value=0
        )

        residential_assets_value = st.number_input(
            "Residential Assets Value", min_value=0, value=0
        )

        commercial_assets_value = st.number_input(
            "Commercial Assets Value", min_value=0, value=0
        )

        luxury_assets_value = st.number_input(
            "Luxury Assets Value", min_value=0, value=0
        )

        bank_asset_value = st.number_input(
            "Bank Asset Value", min_value=0, value=0
        )

    predict_button=st.form_submit_button("Predict")

if predict_button:
    input_data=pd.DataFrame({
            'no_of_dependents':[no_of_dependents],
            'education':[education],
            'self_employed':[self_employed],
            'income_annum':[income_annum],
            'loan_amount':[loan_amount],
            'loan_term':[loan_term],
            'cibil_score':[cibil_score],
            'residential_assets_value':[residential_assets_value],
            'commercial_assets_value':[commercial_assets_value],
            'luxury_assets_value':[luxury_assets_value],
            'bank_asset_value':[bank_asset_value]
        })
    model=load_model()
    result=model.predict(input_data)[0]
    if result==1:
        st.write("Your loan application is approved.")
    else:
        st.write("Sorry, Please try again, your application is not approved right now.s")


with st.expander("Dataset"):
    st.table(df)