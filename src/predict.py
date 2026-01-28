# src/predict.py

import pandas as pd
from pycaret.classification import load_model, predict_model
from preprocessing import preprocess_data


def predict_churn(input_data: pd.DataFrame, model_name: str):
    """
    Predict churn for new customers
    """

    model = load_model()

    data = preprocess_data(input_data)

    predictions = predict_model(model, data=data)

    return predictions
