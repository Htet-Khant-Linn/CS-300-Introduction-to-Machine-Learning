import streamlit as st
import pickle
import pandas as pd
import numpy as np
import kagglehub
from kagglehub import KaggleDatasetAdapter

file_path = "loan_approval_dataset.csv"

df = kagglehub.load_dataset(
  KaggleDatasetAdapter.PANDAS,
  "architsharma01/loan-approval-prediction-dataset",
  file_path,
)

df.columns = df.columns.str.strip()


def load_model():
    with open('loan_prediction_v2.pkl', 'rb') as f:
        return pickle.load(f)

st.header('Loan Prediction Apps')


with st.sidebar:
    st.image("paramilogo.jpg")
    st.text("Student Name: John")
    st.text("Student ID: Parami University")
    st.text("Class of 2027")

col1, col2 = st.columns(2)


with col1:
    min_v=int(df['no_of_dependents'].min())
    max_v=int(df['no_of_dependents'].max())
    var1=st.number_input(
        "Enter number of dependents: ",
        min_value=min_v,
        max_value=max_v,
        value=4
        )

    var2=st.selectbox(
        "Enter your education: ",
        df['education'].unique()
        )

    var3=st.selectbox(
        "Enter your employment status: ",
        df['self_employed'].unique()
    )

    var4 = st.slider(
        "Enter your annual income: ",
        min_value=df['income_annum'].min(),
        max_value=df['income_annum'].max(),
        step=100
    )

    var5 = st.slider(
            "Enter your loan term: ",
            min_value=df['loan_term'].min(),
            max_value=df['loan_term'].max(),
            step=2
        )

with col2:
    var6 = st.slider(
        "Enter your cibil_score: ",
        min_value=df['cibil_score'].min(),
        max_value=df['cibil_score'].max(),
        step=100)

    var7 = st.slider(
            "Enter your residential_assets_value: ",
            min_value=df['residential_assets_value'].min(),
            max_value=df['residential_assets_value'].max(),
            step=100)

    var8 = st.slider(
                "Enter your commercial_assets_value: ",
                min_value=df['commercial_assets_value'].min(),
                max_value=df['commercial_assets_value'].max(),
                step=100)

    var9 = st.slider(
                "Enter your luxury_assets_value: ",
                min_value=df['luxury_assets_value'].min(),
                max_value=df['luxury_assets_value'].max(),
                step=100)

    var10 = st.slider(
                "Enter your bank_asset_value: ",
                min_value=df['bank_asset_value'].min(),
                max_value=df['bank_asset_value'].max(),
                step=100)
    

predict_button = st.button("Predict Loan Amount")

if predict_button:
    input_data = pd.DataFrame(
        [
            {
                "no_of_dependents": var1,
                "education": var2,
                "self_employed": var3,
                "income_annum": var4,
                "loan_term": var5,
                "cibil_score": var6,
                "residential_assets_value": var7,
                "commercial_assets_value": var8,
                "luxury_assets_value": var9,
                "bank_asset_value": var10
            }
        ]
    )

    st.write(input_data)

    model = load_model()
    result=model.predict(input_data)
    st.write(f"Loan Amount: {np.round(result, 2)}")

