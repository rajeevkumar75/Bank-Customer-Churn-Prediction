import pandas as pd
from pycaret.classification import load_model, predict_model
import os

def run_prediction_pipeline(input_data_path, output_file="predictions.csv"):
    # 1. Load the trained model
    model_path = "models/catboost_classifier_model"
    
    if not os.path.exists(f"{model_path}.pkl"):
        raise FileNotFoundError(f"Model not found at {model_path}.pkl. Did training finish?")

    print(f"📦 Loading model from {model_path}...")
    model = load_model(model_path)
    
    # 2. Load the data to predict on
    if isinstance(input_data_path, str):
        data = pd.read_csv(input_data_path)
    else:
        data = input_data_path

    # 3. Generate Predictions
    print("🔮 Generating predictions...")
    predictions = predict_model(model, data=data)
    output_dir = "data/output"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, output_file)
    predictions.to_csv(save_path, index=False)

    return save_path