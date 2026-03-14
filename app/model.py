import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib
from .utils import preprocess_data

# Load and preprocess data
def load_data(path):
    df = pd.read_csv(path)
    X, y = preprocess_data(df)
    return X, y

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    joblib.dump(model, 'app/model.joblib')
    return model, mse

def predict(model, features):
    return model.predict([features])[0]
