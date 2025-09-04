import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
from src.utils import build_baseline_pipeline

DATA = Path("data/emails.csv")
MODEL = Path("models/model.pkl")
MODEL.parent.mkdir(parents=True, exist_ok=True)

if not DATA.exists():
    raise FileNotFoundError(f"Dataset introuvable : {DATA}")

df = pd.read_csv(DATA).dropna(subset=["text", "label"])
X = df["text"].astype(str)
y = (df["label"].str.lower() == "spam").astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipe = build_baseline_pipeline()
pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)
metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred, zero_division=0),
    "recall": recall_score(y_test, y_pred, zero_division=0),
    "f1": f1_score(y_test, y_pred, zero_division=0)
}

print("📊 Résultats :", metrics)

joblib.dump(pipe, MODEL)
print(f"✅ Modèle sauvegardé → {MODEL}")
