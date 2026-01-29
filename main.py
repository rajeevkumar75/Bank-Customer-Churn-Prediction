from src.preprocessing import preprocess_pipeline
from src.train_model import run_training_pipeline
from src.predict import run_prediction_pipeline
import os


def run_pipeline(
    raw_data_path="data/bank data.csv",
    processed_file_name="bank_churn_processed.csv"
):
    print("Starting TrueSource: Bank Churn Pipeline...")

    if not os.path.exists(raw_data_path):
        print(f"File not found: {raw_data_path}")
        return

    # 1. Preprocessing
    print("🔄 Preprocessing data...")
    processed_path = preprocess_pipeline(
        raw_data_path,
        processed_file_name
    )

    # 2. Training
    print("Training CatBoost model...")
    results = run_training_pipeline(processed_path)

    # 3. Prediction
    print("Running prediction pipeline...")
    prediction_path = run_prediction_pipeline(
        input_data_path=processed_path
    )

    # 4. Summary
    print("\nPipeline completed successfully")
    print("\nCatBoost Metrics:")
    print(results["catboost"])
    print(f"\nPredictions saved at: {prediction_path}")


if __name__ == "__main__":
    run_pipeline()
