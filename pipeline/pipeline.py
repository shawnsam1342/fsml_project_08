import os
import gdown

from src.preprocess import preprocess_pipeline, save_processed_data
from src.train import train_and_select_best_model


# 🔹 Dataset URL
DATA_URL = "https://drive.google.com/uc?id=11cacj82VwVw9yRVIkk3m83dcT5YqBf0L"


# 🔹 Download dataset if not present
def download_data():
    os.makedirs("data/raw", exist_ok=True)

    file_path = "data/raw/train_FD001.txt"

    if not os.path.exists(file_path):
        print("Downloading dataset from Google Drive...")
        gdown.download(DATA_URL, file_path, quiet=False)
        print("Download complete!")
    else:
        print("Dataset already exists. Skipping download.")


def run_pipeline():
    # ✅ Step 0: Ensure dataset exists
    download_data()

    print("Step 1: Preprocessing...")
    train_df, val_df, test_df = preprocess_pipeline("data/raw/train_FD001.txt")
    save_processed_data(train_df, val_df, test_df)

    print("Step 2: Training...")
    best_name, _, results, rul_results = train_and_select_best_model()

    print(f"\nBest model: {best_name}")
    print(f"Validation F1: {results[best_name]['validation']['f1']:.4f}")
    print(f"Test F1: {results[best_name]['test']['f1']:.4f}")

    print("\nRUL Model Performance:")
    print(f"Validation MAE: {rul_results['val_mae']:.4f}")
    print(f"Test MAE: {rul_results['test_mae']:.4f}")
    print(f"Test RMSE: {rul_results['test_rmse']:.4f}")


if __name__ == "__main__":
    run_pipeline()
