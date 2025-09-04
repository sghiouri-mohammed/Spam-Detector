import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix, classification_report

MODEL_PATH = "models/model.pkl"
DATA_PATH = "data/emails.csv"

model = joblib.load(MODEL_PATH)
df = pd.read_csv(DATA_PATH)
X = df["text"].astype(str)
y = (df["label"].str.lower() == "spam").astype(int)

y_pred = model.predict(X)

print("=== Rapport de classification ===")
print(classification_report(y, y_pred))

cm = confusion_matrix(y, y_pred)
print("=== Matrice de confusion ===")
print(cm)
print(f"Faux positifs (ham -> spam) : {cm[0,1]}")
