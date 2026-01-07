#feature engineering:-

def add_features(df):
    df['balance_salary_ratio'] = df['balance'] / (df['estimated_salary'] + 1)
    df['products_per_tenure'] = df['products_number'] / (df['tenure'] + 1)

    df['high_value_customer'] = ((df['balance'] > 100000) &
                                  (df['estimated_salary'] > 100000)).astype(int)
    return df

