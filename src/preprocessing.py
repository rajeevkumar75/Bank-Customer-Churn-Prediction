#use for preprocessing data before training:-

import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(df):
    df = df.drop(columns=['customer_id'])

    df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})
    df = pd.get_dummies(df, columns=['country'], drop_first=True)

    X = df.drop('churn', axis=1)
    y = df['churn']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler
