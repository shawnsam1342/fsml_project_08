
from pathlib import Path
import pandas as pd
from src.utils import MODELS_DIR, load_pickle

MODEL_PATH = MODELS_DIR / "model_v1.pkl"


class InferencePipeline:

    def __init__(self):
        self.model = load_pickle(MODEL_PATH)

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

        pred = int(self.model.predict(X)[0])

        result = {
            "prediction": pred,
            "prediction_label": "near_failure" if pred == 1 else "healthy",
        }

        if hasattr(self.model, "predict_proba"):
            prob = self.model.predict_proba(X)[0, 1]
            prob = float(prob)

            result["failure_probability"] = round(prob, 4)

            # ✅ ADD THIS LINE HERE
            result["confidence"] = (
                "high" if prob > 0.7 else
                "medium" if prob > 0.4 else
                "low"
            )

        return result
