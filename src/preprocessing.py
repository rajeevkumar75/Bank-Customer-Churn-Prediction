# src/preprocessing.py

import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_data(file_path: str):
    df = pd.read_csv(file_path)
    return df


def drop_unnecessary_columns(df: pd.DataFrame):
    drop_cols = [
        'customer_id',
        'credit_card',
        'tenure',
        'estimated_salary',
        'products_number'
    ]
    df = df.drop(columns=drop_cols, errors='ignore')
    return df


def encode_features(df: pd.DataFrame):
    df = df.copy()

    le_country = LabelEncoder()
    le_gender = LabelEncoder()

    df['country'] = le_country.fit_transform(df['country'])
    df['gender'] = le_gender.fit_transform(df['gender'])

    return df


def create_age_group(df: pd.DataFrame):
    df = df.copy()

    def age_group(age):
        if age <= 30:
            return 'Young'
        elif age <= 50:
            return 'Middle'
        else:
            return 'Senior'

    df['age_group'] = df['age'].apply(age_group)
    df['age_group'] = df['age_group'].map({
        'Young': 0,
        'Middle': 1,
        'Senior': 2
    })

    return df


def scale_numeric(df: pd.DataFrame, numeric_cols=None):
    if numeric_cols is None:
        numeric_cols = ['credit_score', 'balance', 'age_group']

    df = df.copy()

    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    return df, scaler


def save_processed_data(df: pd.DataFrame, file_name: str):
    output_dir = "data/processed"
    os.makedirs(output_dir, exist_ok=True)

    file_path = os.path.join(output_dir, file_name)
    df.to_csv(file_path, index=False)

    print(f"✅ Processed data saved at: {file_path}")


def preprocess_pipeline(file_path: str, output_file: str = "bank_churn_processed.csv"):
    df = load_data(file_path)
    df = drop_unnecessary_columns(df)
    df = encode_features(df)
    df = create_age_group(df)
    df, scaler = scale_numeric(df)

    save_processed_data(df, output_file)

    return df, scaler
