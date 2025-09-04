from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from pathlib import Path

# Définir le modèle de requête
class EmailRequest(BaseModel):
    text: str

# Charger le modèle
MODEL_PATH = Path("models/model.pkl")
model = joblib.load(MODEL_PATH)

# Créer l'API
app = FastAPI(title="Spam Detector API", version="1.0")

@app.get("/")
def read_root():
    return {"message": "Bienvenue sur l'API Spam Detector"}

@app.post("/predict")
def predict_email(request: EmailRequest):
    text = request.text
    if not text.strip():
        return {"error": "Le texte est vide"}
    
    # Prédiction
    pred = model.predict([text])[0]
    proba = None
    try:
        proba = model.predict_proba([text])[0,1]
    except Exception:
        proba = None
    
    label = "SPAM" if pred==1 else "HAM"
    result = {"label": label}
    if proba is not None:
        result["proba_spam"] = float(proba)
    return result
