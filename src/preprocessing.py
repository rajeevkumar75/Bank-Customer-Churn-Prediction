import os
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_data(file_path: str):
    return pd.read_csv(file_path)

def add_feature_engineering(df: pd.DataFrame):
    """
    Creates new features to help CatBoost identify churn patterns.
    """
    df = df.copy()
    
    #Balance to Salary Ratio: High ratio might indicate high-value but risky customers
    df['balance_salary_ratio'] = df['balance'] / (df['estimated_salary'] + 1)
    
    #Tenure age ratio: How much of their life have they been with the bank?
    df['tenure_age_ratio'] = df['tenure'] / (df['age'] + 1)
    
    #Credit Score per Age: Financial stability relative to life stage
    df['credit_score_age_ratio'] = df['credit_score'] / (df['age'] + 1)
    
    #Wealth Accumulation: Balance divided by products (Capital per product)
    df['balance_per_product'] = df['balance'] / (df['products_number'] + 1)
    
    #Is Senior: Categorizing age can help capture retirement-related churn
    df['is_senior'] = df['age'].apply(lambda x: 1 if x >= 60 else 0)
    
    return df

def drop_unnecessary_columns(df: pd.DataFrame):
    drop_cols = ['customer_id', 'RowNumber']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    return df

def preprocess_pipeline(file_path: str, output_file: str = "bank_churn_processed.csv"):
    df = load_data(file_path)
    
    df = add_feature_engineering(df)
    
    df = drop_unnecessary_columns(df)
    
    #Encoding
    os.makedirs("models", exist_ok=True)
    for col in ['gender', 'country']:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            joblib.dump(le, f"models/label_encoder_{col}.pkl")
    
    #Scaling
    numeric_cols = [
        'credit_score', 'age', 'tenure', 'balance', 'products_number', 
        'estimated_salary', 'balance_salary_ratio', 'tenure_age_ratio', 
        'credit_score_age_ratio', 'balance_per_product'
    ]
    
    existing_numeric = [col for col in numeric_cols if col in df.columns]
    scaler = StandardScaler()
    df[existing_numeric] = scaler.fit_transform(df[existing_numeric])
    joblib.dump(scaler, "models/scaler.pkl")
    
    #Saving
    output_dir = "data/processed"
    os.makedirs(output_dir, exist_ok=True)
    processed_path = os.path.join(output_dir, output_file)
    df.to_csv(processed_path, index=False)
    
    print(f" Feature Engineering & Preprocessing Complete.")
    return df