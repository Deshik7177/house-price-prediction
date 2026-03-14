from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder

def preprocess_data(df):
    df = df.dropna(subset=['Price'])
    y = df['Price']
    X = df.drop(['Price'], axis=1)
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = X[col].fillna('Unknown').astype(str)  # Ensure all are strings
        X[col] = LabelEncoder().fit_transform(X[col])
    for col in X.select_dtypes(include=['float64', 'int64']).columns:
        X[col] = X[col].fillna(X[col].median())
    return X, y

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.joblib')
model = joblib.load(MODEL_PATH)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    df = pd.DataFrame([data])
    X_full = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'House Price Prediction Dataset.csv'))
    df['Price'] = 0
    df, _ = preprocess_data(pd.concat([df, X_full.iloc[:1]], ignore_index=True))
    df = df.drop(columns=['Price'], errors='ignore')
    df = df[model.feature_names_in_]
    pred = model.predict(df)[0]
    return jsonify({'prediction': round(pred, 2)})

if __name__ == '__main__':
    app.run(debug=True)
