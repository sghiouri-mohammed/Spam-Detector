from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path
import joblib


class EmailRequest(BaseModel):
    text: str


MODEL_PATH = Path("models/model.pkl")
THRESHOLD_PATH = Path("models/threshold.txt")
model = joblib.load(MODEL_PATH)


def load_threshold(default: float = 0.5) -> float:
    try:
        if THRESHOLD_PATH.exists():
            return float(THRESHOLD_PATH.read_text().strip())
    except Exception:
        pass
    return default


app = FastAPI(title="Spam Detector API", version="1.0")


@app.get("/")
def read_root():
    return {"message": "Bienvenue sur l'API Spam Detector"}


@app.post("/predict")
def predict_email(request: EmailRequest):
    text = request.text
    if not text or not text.strip():
        return {"error": "Le texte est vide"}

    # score probabiliste si possible, sinon decision_function normalisée
    proba = None
    try:
        proba = float(model.predict_proba([text])[0, 1])
    except Exception:
        try:
            decision = model.decision_function([text])
            dmin, dmax = float(decision.min()), float(decision.max())
            proba = float((decision - dmin) / (dmax - dmin + 1e-9))[0]
        except Exception:
            proba = None

    threshold = load_threshold()

    if proba is not None:
        label_bin = 1 if proba >= threshold else 0
    else:
        label_bin = int(model.predict([text])[0])

    label = "SPAM" if label_bin == 1 else "HAM"
    result = {"label": label, "threshold": float(threshold)}
    if proba is not None:
        result["proba_spam"] = float(proba)
    return result


