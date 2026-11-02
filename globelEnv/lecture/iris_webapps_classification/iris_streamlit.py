import streamlit as st
import pickle
import os

st.title("My First AI apps")

def load_model():
    with open("iris_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

logo_path = "images/parami.jpg"
if os.path.exists(logo_path):
    st.sidebar.image(logo_path, width=150)

st.sidebar.markdown("**Student Name:** David")
st.sidebar.markdown("**Student ID:** PU100000")


s_l = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, value=5.1, step=0.1)
s_w = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, value=3.5, step=0.1)
p_l = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, value=1.4, step=0.1)
p_w = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, value=0.2, step=0.1)


flower_images = {
    'Setosa': 'images/setosa.jpg',
    'Versicolor': 'images/versicolor.jpg',
    'Virginica': 'images/virginica.jpg'
}

if st.button("Predict"):
    arr=[[s_l,s_w,p_l,p_w]]
    model=load_model()
    result=model.predict(arr)
    labels_names=['Setosa','Versicolor','Virginica']
    flower_name=labels_names[result[0]]
    st.success(f"The predicted flower is **{flower_name}**")
    st.image(flower_images[flower_name], caption=f"{flower_name}",width=400)
