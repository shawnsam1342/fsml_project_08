from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data_loader import load_processed_splits, split_features_target
from src.evaluate import evaluate_classifier, save_evaluation_report
from src.features import SklearnFeatureEngineer
from src.utils import logger, MODELS_DIR, LOGS_DIR, save_json, save_pickle


BEST_MODEL_PATH = MODELS_DIR / "model_v1.pkl"
METRICS_PATH = LOGS_DIR / "model_metrics.json"
REPORT_PATH = LOGS_DIR / "evaluation_report.txt"
FEATURE_NOTE_PATH = LOGS_DIR / "feature_engineering_notes.json"


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = [col for col in X.columns if col not in numeric_features]

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),   # missing value handling
        ("scaler", StandardScaler()),                    # scaling
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),  # encoding
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )


def build_models(X_train: pd.DataFrame) -> dict[str, Pipeline]:
    preprocessor = build_preprocessor(X_train)

    common_steps = [
        ("feature_engineering", SklearnFeatureEngineer()),
        ("preprocessing", preprocessor),
    ]

    models = {
        "logistic_regression": Pipeline(common_steps + [
            ("model", LogisticRegression(
                max_iter=1500,
                class_weight="balanced",
                random_state=42
            )),
        ]),
        "random_forest": Pipeline(common_steps + [
            ("model", RandomForestClassifier(
                n_estimators=300,
                max_depth=12,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )),
        ]),
        "xgboost": Pipeline(common_steps + [
            ("model", XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.1,
                eval_metric="logloss",
                random_state=42,
                n_jobs=-1,
                verbosity=0,
            )),
        ]),
    }
    return models


def get_feature_documentation() -> dict[str, str]:
    return {
        "sensor_11_12_gap": (
            "Difference between sensor_11 and sensor_12. "
            "This captures divergence between two related sensor responses; "
            "growing mismatch can indicate abnormal operating behavior."
        ),
        "sensor_20_21_ratio": (
            "Ratio of sensor_20 to sensor_21. "
            "Relative change is often more informative than absolute values "
            "when engines operate under varying ranges."
        ),
        "sensor_15_squared": (
            "Squared value of sensor_15 to amplify extreme readings. "
            "This helps the model respond more strongly to non-linear stress patterns."
        ),
    }


def train_and_select_best_model() -> tuple[str, Pipeline, dict[str, Any]]:
    train_df, val_df, test_df = load_processed_splits()

    X_train, y_train = split_features_target(train_df)
    X_val, y_val = split_features_target(val_df)
    X_test, y_test = split_features_target(test_df)

    models = build_models(X_train)

    all_results: dict[str, Any] = {}
    best_name = ""
    best_model: Pipeline | None = None
    best_val_recall = -1.0
    best_val_f1 = -1.0

    for name, pipeline in models.items():
        print(f"\nTraining model: {name}")
        logger.info(f"Training model: {name}")
        pipeline.fit(X_train, y_train)

        val_metrics = evaluate_classifier(pipeline, X_val, y_val)
        test_metrics = evaluate_classifier(pipeline, X_test, y_test)
        logger.info(f"{name} Validation Recall: {val_metrics['recall']}")
        logger.info(f"{name} Validation F1: {val_metrics['f1']}")
        logger.info(f"{name} Test Recall: {test_metrics['recall']}")
        logger.info(f"{name} Test F1: {test_metrics['f1']}")
        logger.info(f"{name} Validation Accuracy: {val_metrics['accuracy']}")
        logger.info(f"{name} Test Accuracy: {test_metrics['accuracy']}")

        all_results[name] = {
            "validation": val_metrics,
            "test": test_metrics,
        }

        print(f"Validation Recall: {val_metrics['recall']:.4f}")
        print(f"Validation F1-score: {val_metrics['f1']:.4f}")
        print(f"Validation Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"Test Recall: {test_metrics['recall']:.4f}")
        print(f"Test F1-score: {test_metrics['f1']:.4f}")
        print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")

        if (val_metrics["recall"] > best_val_recall) or (
            val_metrics["recall"] == best_val_recall and val_metrics["f1"] > best_val_f1
        ):
            best_val_recall = val_metrics["recall"]
            best_val_f1 = val_metrics["f1"]
            best_name = name
            best_model = pipeline

    if best_model is None:
        raise RuntimeError("No model was trained successfully.")

    save_pickle(best_model, BEST_MODEL_PATH)
    save_json(all_results, METRICS_PATH)
    save_json(get_feature_documentation(), FEATURE_NOTE_PATH)
    save_evaluation_report(all_results, REPORT_PATH)

    return best_name, best_model, all_results


if __name__ == "__main__":
    best_name, _, results = train_and_select_best_model()
    print(f"\nBest model: {best_name}")
    logger.info(f"Best model selected: {best_name}")
    print(f"Validation Recall: {results[best_name]['validation']['recall']:.4f}")
    print(f"Validation F1: {results[best_name]['validation']['f1']:.4f}")
    print(f"Validation Accuracy: {results[best_name]['validation']['accuracy']:.4f}")
    print(f"Test Recall: {results[best_name]['test']['recall']:.4f}")
    print(f"Test F1: {results[best_name]['test']['f1']:.4f}")
    print(f"Test Accuracy: {results[best_name]['test']['accuracy']:.4f}")
    print(f"Saved best model to: {BEST_MODEL_PATH}")