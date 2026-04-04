from pathlib import Path
import pandas as pd
import json
from src.utils import MODELS_DIR, load_pickle

MODEL_PATH = MODELS_DIR / "model_v1.pkl"
THRESHOLD_PATH = Path("logs/threshold.json")


class InferencePipeline:

    def __init__(self):
        self.model = load_pickle(MODEL_PATH)

        # ✅ Load trained threshold
        with open(THRESHOLD_PATH, "r") as f:
            self.threshold = json.load(f)["threshold"]

    def _prepare_input(self, data):
        if isinstance(data, dict):
            X = pd.DataFrame([data])
        else:
            X = data.copy()

        if "label" in X.columns:
            X = X.drop(columns=["label"])

        return X

    def predict(self, data):
        X = self._prepare_input(data)

        #Use probability instead of default predict()
        prob = self.model.predict_proba(X)[0, 1]
        prob = float(prob)

        pred = int(prob >= self.threshold)

        result = {
            "prediction": pred,
            "prediction_label": "near_failure" if pred == 1 else "healthy",
            "failure_probability": round(prob, 4),
        }

        #Updated confidence logic (aligned with threshold)
        result["confidence"] = (
            "high" if prob > (self.threshold + 0.2) else
            "medium" if prob >= self.threshold else
            "low"
        )

        return result

if __name__ == "__main__":
    pipeline = InferencePipeline()

    test_path = Path("data/processed/test.csv")
    df = pd.read_csv(test_path)

    # mix of healthy + failure samples (better demo)
    """
    samples = pd.concat([
        df[df["label"] == 0].head(3),
        df[df["label"] == 1].head(2)
    ])
    """
    samples = pd.concat([
        df[df["label"] == 0].sample(5, random_state=42),
        df[df["label"] == 1].sample(5, random_state=42)
    ])

    print("\n--- Sample Predictions ---\n")

    for i, row in samples.iterrows():
        row_df = row.to_frame().T
        result = pipeline.predict(row_df)
        print(f"Sample {i}: {result}")