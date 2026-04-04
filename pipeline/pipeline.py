
from src.preprocess import preprocess_pipeline, save_processed_data
from src.train import train_and_select_best_model


def run_pipeline():
    print("Step 1: Preprocessing...")
    train_df, val_df, test_df = preprocess_pipeline("data/raw/train_FD001.txt")
    save_processed_data(train_df, val_df, test_df)

    print("Step 2: Training...")
    best_name, _, results = train_and_select_best_model()

    print(f"\nBest model: {best_name}")
    print(f"Validation F1: {results[best_name]['validation']['f1']:.4f}")
    print(f"Test F1: {results[best_name]['test']['f1']:.4f}")


if __name__ == "__main__":
    run_pipeline()
