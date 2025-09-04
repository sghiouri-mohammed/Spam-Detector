import os
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score

MODEL_PATH = "models/model.pkl"
DATA_PATH = "data/emails.csv"


def test_model_exists():
    assert os.path.exists(MODEL_PATH), "model.pkl manquant — as‑tu exécuté l'entraînement ?"


def test_model_accuracy_minimum():
    df = pd.read_csv(DATA_PATH)
    X = df["text"].astype(str)
    y = (df["label"].str.lower() == "spam").astype(int)

    model = joblib.load(MODEL_PATH)  # pipeline (tfidf + clf)
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    assert acc >= 0.80, f"Accuracy trop faible : {acc:.3f}"