import streamlit as st
import pandas as pd
import joblib
from model import load_data, train_model, predict
from utils import preprocess_data
import os

st.set_page_config(page_title="House Price Prediction", layout="centered")
st.title("🏡 House Price Prediction App")
st.markdown("""
A modern, professional app to predict house prices using machine learning. Enter house features below to get a price estimate.
""")

# Load or train model
data_path = os.path.join(os.path.dirname(__file__), "..", "House Price Prediction Dataset.csv")
if not os.path.exists("app/model.joblib"):
    X, y = load_data(data_path)
    model, mse = train_model(X, y)
else:
    model = joblib.load("app/model.joblib")
    X, y = load_data(data_path)

# Feature input UI
def user_input_features(X):
    inputs = {}
    for col in X.columns:
        if X[col].dtype == 'object':
            options = list(X[col].unique())
            inputs[col] = st.selectbox(col, options)
        else:
            min_val = float(X[col].min())
            max_val = float(X[col].max())
            mean_val = float(X[col].mean())
            inputs[col] = st.slider(col, min_val, max_val, mean_val)
    return inputs

st.header("Enter House Features")
inputs = user_input_features(X)

if st.button("Predict Price"):
    # Preprocess single row
    input_df = pd.DataFrame([inputs])
    input_df, _ = preprocess_data(pd.concat([input_df, X.iloc[:1]], ignore_index=True))
    features = input_df.iloc[0].values
    price = predict(model, features)
    st.success(f"Estimated House Price: ${price:,.2f}")
