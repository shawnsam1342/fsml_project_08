from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, recall_score, mean_squared_error, mean_squared_error
from xgboost import XGBClassifier, XGBRegressor

from src.data_loader import load_processed_splits, split_features_target, split_features_target_regression
from src.features import SklearnFeatureEngineer
from src.preprocess import build_preprocessor
from src.utils import logger, MODELS_DIR, LOGS_DIR, save_json, save_pickle
from src.evaluate import evaluate_classifier, save_evaluation_report

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


BEST_MODEL_PATH = MODELS_DIR / "model_v1.pkl"
METRICS_PATH = LOGS_DIR / "model_metrics.json"
REPORT_PATH = LOGS_DIR / "evaluation_report.txt"
FEATURE_NOTE_PATH = LOGS_DIR / "feature_engineering_notes.json"
THRESHOLD_PATH = LOGS_DIR / "threshold.json"
RUL_MODEL_PATH = MODELS_DIR / "rul_model.pkl"


#NEW: threshold optimization
def find_best_threshold(model, X_val, y_val):
    probs = model.predict_proba(X_val)[:, 1]

    best_threshold = 0.5
    best_precision = 0

    for t in np.linspace(0.1, 0.9, 50):
        preds = (probs >= t).astype(int)
        recall = recall_score(y_val, preds)
        precision = precision_score(y_val, preds)

        if recall >= 0.90 and precision > best_precision:
            best_precision = precision
            best_threshold = t

    return best_threshold


#MODIFIED: added scale_pos_weight
def build_models(X_train: pd.DataFrame, scale_pos_weight: float) -> dict[str, Pipeline]:
    preprocessor = build_preprocessor(X_train)

    common_steps = [
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
                scale_pos_weight=scale_pos_weight,
                random_state=42,
                n_jobs=-1,
                verbosity=0,
            )),
        ]),
    }
    return models

def get_feature_documentation() -> dict[str, str]:
    return {
        "sensor_11_12_gap": "Difference between sensor_11 and sensor_12.",
        "sensor_20_21_ratio": "Ratio of sensor_20 to sensor_21.",
        "sensor_15_squared": "Squared value of sensor_15.",
    }


def train_and_select_best_model() -> tuple[str, Pipeline, dict[str, Any]]:
    train_df, val_df, test_df = load_processed_splits()

    X_train, y_train = split_features_target(train_df)

    # imbalance handling
    pos = sum(y_train == 1)
    neg = sum(y_train == 0)
    scale_pos_weight = neg / pos

    X_val, y_val = split_features_target(val_df)
    X_test, y_test = split_features_target(test_df)

    # pass weight into models
    models = build_models(X_train, scale_pos_weight)

    all_results: dict[str, Any] = {}
    best_name = ""
    best_model: Pipeline | None = None
    best_val_f1 = -1.0

    for name, pipeline in models.items():
        print(f"\nTraining model: {name}")
        logger.info(f"Training model: {name}")
        pipeline.fit(X_train, y_train)

        val_metrics = evaluate_classifier(pipeline, X_val, y_val)
        test_metrics = evaluate_classifier(pipeline, X_test, y_test)

        all_results[name] = {
            "validation": val_metrics,
            "test": test_metrics,
        }

        print(f"Validation Recall: {val_metrics['recall']:.4f}")
        print(f"Validation F1-score: {val_metrics['f1']:.4f}")
        print(f"Validation Precision: {val_metrics['precision']:.4f}")
        print(f"Test Recall: {test_metrics['recall']:.4f}")
        print(f"Test F1-score: {test_metrics['f1']:.4f}")
        print(f"Test Precision: {test_metrics['precision']:.4f}")

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_name = name
            best_model = pipeline

    if best_model is None:
        raise RuntimeError("No model was trained successfully.")

    #NEW: threshold tuning
    best_threshold = find_best_threshold(best_model, X_val, y_val)
    print(f"Optimal threshold: {best_threshold:.3f}")
    logger.info(f"Optimal threshold: {best_threshold:.3f}")

    save_json({"threshold": float(best_threshold)}, THRESHOLD_PATH)

    save_pickle(best_model, BEST_MODEL_PATH)
    save_json(all_results, METRICS_PATH)
    save_json(get_feature_documentation(), FEATURE_NOTE_PATH)
    save_evaluation_report(all_results, REPORT_PATH)

    #Train RUL model inside main pipeline
    print("\n--- Training RUL Model ---")
    rul_model, rul_results = train_rul_model()

    return best_name, best_model, all_results, rul_results

def train_rul_model():
    train_df, val_df, test_df = load_processed_splits()

    X_train, y_train = split_features_target_regression(train_df)
    X_val, y_val = split_features_target_regression(val_df)
    X_test, y_test = split_features_target_regression(test_df)

    preprocessor = build_preprocessor(X_train)

    models = {
        "random_forest": Pipeline([
            ("preprocessing", preprocessor),
            ("model", RandomForestRegressor(
                n_estimators=200,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )),
        ]),
        "xgboost": Pipeline([
            ("preprocessing", preprocessor),
            ("model", XGBRegressor(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1
            )),
        ])
    }

    best_model = None
    best_mae = float("inf")

    for name, model in models.items():
        print(f"\nTraining RUL model: {name}")
        model.fit(X_train, y_train)

        preds = model.predict(X_val)
        mae = np.mean(np.abs(y_val - preds))

        print(f"Validation MAE: {mae:.4f}")

        if mae < best_mae:
            best_mae = mae
            best_model = model

    # Test evaluation
    test_preds = best_model.predict(X_test)
    test_mae = np.mean(np.abs(y_test - test_preds))
    rmse = np.sqrt(np.mean((y_test - test_preds) ** 2))

    print(f"\nTest MAE: {test_mae:.4f}")
    print(f"Test RMSE: {rmse:.4f}")

    save_pickle(best_model, RUL_MODEL_PATH)

    #RETURN METRICS
    rul_results = {
        "val_mae": float(best_mae),
        "test_mae": float(test_mae),
        "test_rmse": float(rmse),
    }

    return best_model, rul_results

if __name__ == "__main__":
    best_name, _, results, rul_results = train_and_select_best_model()

    print(f"\nBest model: {best_name}")
    print(f"Validation Recall: {results[best_name]['validation']['recall']:.4f}")
    print(f"Validation F1: {results[best_name]['validation']['f1']:.4f}")
    print(f"Test Recall: {results[best_name]['test']['recall']:.4f}")
    print(f"Test F1: {results[best_name]['test']['f1']:.4f}")
    print(f"Test Precision: {results[best_name]['test']['precision']:.4f}")
    print(f"Saved best model to: {BEST_MODEL_PATH}")
    print("\nRUL Model Performance:")
    print(f"Validation MAE: {rul_results['val_mae']:.4f}")
    print(f"Test MAE: {rul_results['test_mae']:.4f}")
    print(f"Test RMSE: {rul_results['test_rmse']:.4f}")