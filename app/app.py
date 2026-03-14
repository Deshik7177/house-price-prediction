import streamlit as st
import pandas as pd
import joblib
from model import load_data, train_model, predict
from utils import preprocess_data
import os


st.set_page_config(page_title="House Price Prediction", layout="wide", page_icon="🏡")

# --- Custom CSS for a professional look ---
st.markdown("""
<style>
body, .main {
    background: linear-gradient(120deg, #f5f7fa 0%, #c3cfe2 100%) !important;
}
.hero {
    background: linear-gradient(90deg, #4F8BF9 0%, #235390 100%);
    color: white;
    border-radius: 18px;
    padding: 2.5em 2em 1.5em 2em;
    margin-bottom: 2em;
    box-shadow: 0 4px 24px rgba(79,139,249,0.08);
    text-align: center;
}
.feature-box {
    background: #fff;
    border-radius: 12px;
    box-shadow: 0 2px 8px rgba(79,139,249,0.07);
    padding: 1.2em 1em;
    margin-bottom: 1.2em;
    text-align: center;
}
.stButton>button {
    background: linear-gradient(90deg, #4F8BF9 0%, #235390 100%);
    color: white;
    font-weight: bold;
    border-radius: 8px;
    padding: 0.5em 2em;
    font-size: 1.1em;
    border: none;
    transition: 0.2s;
}
.stButton>button:hover {
    background: #235390;
    color: #fff;
}
.stSuccess {
    background-color: #e6f4ea;
    color: #1a7f37;
    border-radius: 8px;
    font-size: 1.2em;
}
.footer {
    color: #888;
    text-align: center;
    font-size: 0.95em;
    margin-top: 2em;
}
</style>
""", unsafe_allow_html=True)

# --- Hero Section ---
st.markdown("""
<div class="hero">
    <h1 style="margin-bottom:0.2em;font-size:2.6em;">🏡 House Price Prediction</h1>
    <div style="font-size:1.25em;">Predict house prices instantly with a modern, interactive app powered by machine learning.</div>
</div>
""", unsafe_allow_html=True)

# --- Feature Highlights ---
f1, f2, f3 = st.columns(3)
with f1:
    st.markdown('<div class="feature-box"><b>⚡ Fast & Accurate</b><br>Powered by Random Forest</div>', unsafe_allow_html=True)
with f2:
    st.markdown('<div class="feature-box"><b>🎨 Modern UI</b><br>Clean, responsive design</div>', unsafe_allow_html=True)
with f3:
    st.markdown('<div class="feature-box"><b>🔒 Private</b><br>Your data stays on your device</div>', unsafe_allow_html=True)

with st.sidebar:
    st.image("https://images.unsplash.com/photo-1506744038136-46273834b3fb?auto=format&fit=crop&w=400&q=80", use_column_width=True)
    st.markdown("""
    ## About
    This app predicts house prices based on features you provide. Powered by Random Forest and Streamlit.
    
    **Instructions:**
    - Adjust the features below
    - Click **Predict Price**
    - See your result instantly!
    <br><br>
    <span style='color:#4F8BF9;font-weight:bold;'>Developed March 2026</span>
    """, unsafe_allow_html=True)

# Load or train model
data_path = os.path.join(os.path.dirname(__file__), "..", "House Price Prediction Dataset.csv")
if not os.path.exists("app/model.joblib"):
    X, y = load_data(data_path)
    model, mse = train_model(X, y)
else:
    model = joblib.load("app/model.joblib")
    X, y = load_data(data_path)



# --- Modern UI for feature input ---
st.markdown("<h2 style='margin-top:2em;'>Enter House Features</h2>", unsafe_allow_html=True)
cols = st.columns(3)
inputs = {}
for idx, col in enumerate(X.columns):
    with cols[idx % 3]:
        if X[col].dtype == 'object':
            options = list(X[col].unique())
            inputs[col] = st.selectbox(f"{col}", options, help=f"Select the {col.lower()}.")
        else:
            min_val = float(X[col].min())
            max_val = float(X[col].max())
            mean_val = float(X[col].mean())
            step = 1.0 if max_val - min_val > 10 else 0.01
            inputs[col] = st.slider(f"{col}", min_val, max_val, mean_val, step=step, help=f"Set the {col.lower()}.")



if st.button("🔮 Predict Price", use_container_width=True):
    # Preprocess single row
    input_df = pd.DataFrame([inputs])
    input_df, _ = preprocess_data(pd.concat([input_df, X.iloc[:1]], ignore_index=True))
    features = input_df.iloc[0].values
    price = predict(model, features)
    st.success(f"Estimated House Price: ${price:,.2f}")

# --- Footer ---
st.markdown("""
<div class="footer">
    &copy; 2026 House Price Prediction &mdash; Built with ❤️ using Streamlit
</div>
""", unsafe_allow_html=True)
