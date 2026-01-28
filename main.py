# main.py

from src.preprocessing import preprocess_pipeline
from src.train_model import run_training_pipeline
import os
import joblib


def run_pipeline(
    raw_data_path="data/bank data.csv",
    processed_file_name="bank_churn_processed.csv"
):
    print("🚀 Starting Bank Churn Pipeline...")

    #Preprocessing
    print("🔄 Running preprocessing pipeline...")
    df_processed, scaler = preprocess_pipeline(
        file_path=raw_data_path,
        output_file=processed_file_name
    )

    #Train models
    print("🤖 Training models (Logistic Regression & LightGBM)...")
    results = run_training_pipeline(df_processed)

    #Save scaler
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/scaler.pkl")

    # Show summary
    print("\n✅ Training completed successfully.")
    print("\n📊 Cross-validation results:")
    print(results)

    print("\n📁 Artifacts saved:")
    print("- Processed data → data/processed/")
    print("- Models → models/")
    print("- Scaler → models/scaler.pkl")


if __name__ == "__main__":
    run_pipeline()
