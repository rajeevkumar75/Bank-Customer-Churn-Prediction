import pandas as pd
from pycaret.classification import load_model, predict_model
import os

def run_prediction_pipeline(input_data_path, output_file="predictions.csv"):
    """
    Loads the saved CatBoost model and runs inference on the processed data.
    """
    # 1. Path to the model saved by train_model.py
    # PyCaret saves it as 'models/catboost_classifier_model.pkl' 
    # but load_model expects the name without the .pkl extension
    model_path = "models/catboost_classifier_model"
    
    if not os.path.exists(f"{model_path}.pkl"):
        raise FileNotFoundError(f"Model not found at {model_path}.pkl. Did training finish?")

    print(f"📦 Loading model from {model_path}...")
    model = load_model(model_path)
    
    # 2. Load the data to predict on
    # Since main.py passes the processed DataFrame or path
    if isinstance(input_data_path, str):
        data = pd.read_csv(input_data_path)
    else:
        data = input_data_path

    # 3. Generate Predictions
    # This adds 'prediction_label' (0 or 1) and 'prediction_score' (probability)
    print("🔮 Generating predictions...")
    predictions = predict_model(model, data=data)

    # 4. Save the results
    output_dir = "data/output"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, output_file)
    predictions.to_csv(save_path, index=False)

    return save_path