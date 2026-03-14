import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import os

def preprocess_data(df):
    df = df.dropna(subset=['Price'])
    y = df['Price']
    X = df.drop(['Price'], axis=1)
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = X[col].fillna('Unknown')
        X[col] = LabelEncoder().fit_transform(X[col])
    for col in X.select_dtypes(include=['float64', 'int64']).columns:
        X[col] = X[col].fillna(X[col].median())
    return X, y

def main():
    df = pd.read_csv('../House Price Prediction Dataset.csv')
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, 'model.joblib')
    print('Model trained and saved to webapp/model.joblib')

if __name__ == '__main__':
    main()
