from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)


MODEL_PATH = Path("models/model.pkl")
THRESHOLD_PATH = Path("models/threshold.txt")
DATA_PATH = Path("data/emails.csv")


def load_threshold(default: float = 0.5) -> float:
    if THRESHOLD_PATH.exists():
        try:
            return float(THRESHOLD_PATH.read_text().strip())
        except Exception:
            return default
    return default


def main():
    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(DATA_PATH).dropna(subset=["text", "label"]).copy()
    X = df["text"].astype(str)
    y = (df["label"].str.lower() == "spam").astype(int)

    # Predict probabilities if available
    try:
        proba = model.predict_proba(X)[:, 1]
    except Exception:
        dec = model.decision_function(X)
        proba = (dec - dec.min()) / (dec.max() - dec.min() + 1e-9)

    threshold = load_threshold()
    y_pred = (proba >= threshold).astype(int)

    print("=== Rapport de classification ===")
    print(classification_report(y, y_pred, target_names=["ham", "spam"]))

    cm = confusion_matrix(y, y_pred)
    print("=== Matrice de confusion ===")
    print(cm)
    print(f"Faux positifs (ham -> spam) : {cm[0,1]}")

    metrics = {
        "accuracy": float(accuracy_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "f1": float(f1_score(y, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, proba)),
        "average_precision": float(average_precision_score(y, proba)),
        "threshold": float(threshold),
    }
    print("=== Metrics ===")
    print(metrics)


if __name__ == "__main__":
    main()
