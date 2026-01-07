#feature engineering:-

import numpy as np

def add_features(df):

    # 1️ Financial strength & risk
    df['balance_salary_ratio'] = df['balance'] / (df['estimated_salary'] + 1)
    df['low_credit_score'] = (df['credit_score'] < 600).astype(int)
    df['high_balance'] = (df['balance'] > 100000).astype(int)

    # 2️ Customer engagement & behavior
    df['products_per_tenure'] = df['products_number'] / (df['tenure'] + 1)
    df['inactive_high_balance'] = (
        (df['active_member'] == 0) & (df['balance'] > 50000)
    ).astype(int)

    # 3️ Tenure-based loyalty
    df['new_customer'] = (df['tenure'] <= 2).astype(int)
    df['long_term_customer'] = (df['tenure'] >= 7).astype(int)

    # 4️ Interaction features
    df['credit_age_interaction'] = df['credit_score'] * df['age']
    df['balance_product_interaction'] = df['balance'] * df['products_number']

    # 5️ Customer value segmentation
    df['customer_value_score'] = (
        0.4 * df['balance_salary_ratio'] +
        0.3 * df['products_number'] +
        0.3 * df['active_member']
    )

    return df


