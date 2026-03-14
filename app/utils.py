import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data(df):
    # Example: drop rows with missing target, fill missing values, encode categoricals
    df = df.dropna(subset=['SalePrice'])
    y = df['SalePrice']
    X = df.drop(['SalePrice'], axis=1)
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = X[col].fillna('Unknown')
        X[col] = LabelEncoder().fit_transform(X[col])
    for col in X.select_dtypes(include=['float64', 'int64']).columns:
        X[col] = X[col].fillna(X[col].median())
    return X, y
