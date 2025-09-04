import streamlit as st
import joblib
import pandas as pd
from pathlib import Path
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import textwrap

st.set_page_config(page_title="Spam Detector", layout="centered")
st.title("📧 Spam Detector — Prototype")

MODEL_PATH = Path("models/model.pkl")
THRESHOLD_PATH = Path("models/threshold.txt")

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error("⚠️ models/model.pkl introuvable — exécute `python src/train.py`")
        return None
    return joblib.load(MODEL_PATH)

model = load_model()

@st.cache_resource
def load_threshold(default: float = 0.5) -> float:
    try:
        if THRESHOLD_PATH.exists():
            return float(THRESHOLD_PATH.read_text().strip())
    except Exception:
        pass
    return default

threshold = load_threshold()

# Zone de saisie utilisateur
st.subheader("Tester un email")
user_txt = st.text_area("✍️ Collez un email (texte brut)", height=150)

if st.button("🔍 Prédire"):
    if model is None:
        st.warning("Modèle non chargé")
    elif user_txt.strip():
        proba = None
        try:
            proba = model.predict_proba([user_txt])[0, 1]
        except Exception:
            # fallback: decision_function -> min-max scaling
            try:
                dec = model.decision_function([user_txt])
                dmin, dmax = float(dec.min()), float(dec.max())
                proba = (dec - dmin) / (dmax - dmin + 1e-9)
                proba = float(proba[0])
            except Exception:
                proba = None

        if proba is not None:
            pred_bin = 1 if proba >= threshold else 0
            label = "🚨 SPAM" if pred_bin == 1 else "✅ HAM (non-spam)"
        else:
            # ultimate fallback
            pred_bin = int(model.predict([user_txt])[0])
            label = "🚨 SPAM" if pred_bin == 1 else "✅ HAM (non-spam)"

        st.subheader(f"Résultat : {label}")
        if proba is not None:
            st.caption(f"Score (spam) : {proba:.3f} — seuil: {threshold:.3f}")
    else:
        st.warning("Veuillez saisir un texte.")

st.divider()

# Upload CSV
st.subheader("Tester un fichier CSV")
uploaded = st.file_uploader("Uploader un CSV (colonnes : text,label)", type=["csv"])
if uploaded is not None:
    df = pd.read_csv(uploaded)
    if "text" not in df.columns:
        st.error("⚠️ Le fichier doit contenir une colonne `text`")
    else:
        df = df.dropna(subset=["text"])
        texts = df["text"].astype(str)
        # Probabilités si possible, sinon score normalisé
        try:
            probs = model.predict_proba(texts)[:, 1]
        except Exception:
            try:
                dec = model.decision_function(texts)
                dmin, dmax = float(dec.min()), float(dec.max())
                probs = (dec - dmin) / (dmax - dmin + 1e-9)
            except Exception:
                probs = None

        if probs is not None:
            preds = (probs >= threshold).astype(int)
            df["score_spam"] = probs
        else:
            preds = model.predict(texts)

        df["prediction"] = ["spam" if int(p) == 1 else "ham" for p in preds]
        st.write("Aperçu des prédictions :")
        st.dataframe(df.head(10))

        st.subheader("Visualisations du fichier")
        # 1) Répartition des classes vérités si dispo
        col1, col2 = st.columns(2)
        with col1:
            if "label" in df.columns:
                true_counts = df["label"].str.lower().map({"spam": "spam", "ham": "ham"}).value_counts()
                fig1, ax1 = plt.subplots(figsize=(4, 4))
                ax1.pie(true_counts.values, labels=true_counts.index, autopct='%1.1f%%', startangle=90,
                        colors=["#ff6b6b", "#4dabf7"], wedgeprops={'linewidth': 1, 'edgecolor': 'white'})
                ax1.axis('equal')
                ax1.set_title("Répartition labels (vérité)")
                st.pyplot(fig1)
            else:
                st.info("Colonne `label` absente — impossible d'afficher la répartition vraie.")
        with col2:
            pred_counts = pd.Series(df["prediction"]).value_counts()
            fig2, ax2 = plt.subplots(figsize=(4, 4))
            ax2.bar(pred_counts.index, pred_counts.values, color=["#ff6b6b" if x == "spam" else "#4dabf7" for x in pred_counts.index])
            ax2.set_title("Répartition des prédictions")
            ax2.set_ylabel("Nombre")
            st.pyplot(fig2)

        # 2) Distribution des scores (si dispo)
        if "score_spam" in df.columns:
            st.caption("Distribution des scores")
            fig3, ax3 = plt.subplots(figsize=(6, 3))
            sns.histplot(df["score_spam"], bins=30, kde=True, color="#845ef7", ax=ax3)
            ax3.axvline(threshold, color="#fa5252", linestyle="--", label=f"seuil={threshold:.2f}")
            ax3.set_xlabel("score_spam")
            ax3.legend()
            st.pyplot(fig3)

        # 3) Metrics + matrice de confusion si labels dispo
        if "label" in df.columns:
            y_true = (df["label"].str.lower() == "spam").astype(int).values
            y_pred = (df["prediction"] == "spam").astype(int).values

            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy", f"{acc:.3f}")
            c2.metric("Precision", f"{prec:.3f}")
            c3.metric("Recall", f"{rec:.3f}")
            c4.metric("F1", f"{f1:.3f}")

            cm = confusion_matrix(y_true, y_pred)
            fig4, ax4 = plt.subplots(figsize=(4.5, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", cbar=False, ax=ax4,
                        xticklabels=["ham", "spam"], yticklabels=["ham", "spam"])
            ax4.set_xlabel("Prédit")
            ax4.set_ylabel("Réel")
            ax4.set_title("Matrice de confusion")
            st.pyplot(fig4)

            # Rapport détaillé
            report = classification_report(y_true, y_pred, target_names=["ham", "spam"], output_dict=True)
            rep_df = pd.DataFrame(report).T
            st.dataframe(rep_df.style.format({"precision": "{:.3f}", "recall": "{:.3f}", "f1-score": "{:.3f}", "support": "{:.0f}"}))

st.caption(f"Seuil de décision courant: {threshold:.3f}")

st.divider()
