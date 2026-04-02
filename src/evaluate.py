from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix, classification_report
from pathlib import Path
from typing import Any

def evaluate_classifier(model, X, y, threshold=0.5):
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[:, 1]
        y_pred = (probs >= threshold).astype(int)
    else:
        y_pred = model.predict(X)

    from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix, classification_report

    return {
        "accuracy": float(accuracy_score(y, y_pred)),
        "f1": float(f1_score(y, y_pred, zero_division=0)),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
        "classification_report": classification_report(y, y_pred, zero_division=0),
    }

def save_evaluation_report(all_results: dict, report_path: str | Path):
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, "w") as f:
        for model_name, split_results in all_results.items():
            f.write(f"\nModel: {model_name}\n")
            for split, metrics in split_results.items():
                f.write(f"\n{split.upper()}:\n")
                f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
                f.write(f"F1: {metrics['f1']:.4f}\n")
                f.write(f"Confusion Matrix: {metrics['confusion_matrix']}\n")