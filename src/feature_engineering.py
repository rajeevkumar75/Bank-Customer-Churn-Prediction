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




