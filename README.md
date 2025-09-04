# 📧 Spam Detector

Détection de spams à partir de texte d’email via un pipeline ML scikit‑learn (TF‑IDF + modèles linéaires/SVM/NB) avec optimisation par validation croisée, sélection de seuil optimale et application Streamlit pour l’inférence.

Le projet fournit:
- un notebook d’exploration et d’expérimentation `spam_emails_detection.ipynb`
- un package d’entraînement/évaluation scriptable dans `src/`
- une app `Streamlit` dans `app/` qui charge le modèle et applique un seuil appris
- des données d’exemple dans `data/` (suivies par DVC pour la source `emails.csv`)

---

## 🧭 Structure du projet

```
Quality and Strategy/
  app/
    streamlit_app.py           # UI pour l’inférence: texte unique et CSV + visualisations
  data/
    emails.csv                 # Dataset text,label (ham/spam) — versionné via DVC
  models/
    model.pkl                  # Pipeline sklearn (TF‑IDF + meilleur modèle)
    threshold.txt              # Seuil de décision optimisé (validation)
  src/
    prepare_data.py            # Chargement + splits stratifiés (train/val/test)
    train.py                   # GridSearchCV multi‑modèles + sélection de seuil + sauvegarde
    evaluate.py                # Rapport sur un CSV avec seuil (métriques + CM)
  spam_emails_detection.ipynb  # Notebook complet (EDA, entrainement, comparaisons, courbes)
  tests/
    test_model.py              # Tests basiques (ex: chargement modèle)
  requirements.txt             # Dépendances
  dvc.yaml / data/*.dvc        # Traçabilité des données (optionnel)
```

---

## 🧪 Données (`data/`)

- Fichier principal: `data/emails.csv` avec les colonnes `text` et `label` (valeurs `ham` ou `spam`).
- Dans le code, on binarise la cible: `label_bin = (label.lower() == "spam").astype(int)`, donc 1=spam, 0=ham.
- Les splits sont stratifiés pour conserver la proportion de spams dans train/val/test.

---

## 🔧 Préparation et splits (`src/prepare_data.py`)

Rôles clés:
- Chargement robuste du CSV (`load_dataset`) et nettoyage minimal (drop `NaN`, cast `text` en `str`).
- Création de la cible binaire `label_bin`.
- Splits stratifiés et reproductibles avec `RANDOM_STATE=42`:
  - 80% train/val, 20% test
  - puis 75% train / 25% val sur le bloc train/val (≈ 60/20/20 global)

API principale:
- `load_dataset(path: Path) -> pd.DataFrame`
- `stratified_splits(df) -> (X_train, X_val, X_test, y_train, y_val, y_test)`

---

## 🧠 Entraînement et optimisation (`src/train.py`)

Objectifs:
- Construire un pipeline `TfidfVectorizer` + estimateur.
- Évaluer plusieurs familles de modèles via `GridSearchCV` (score AP / Average Precision):
  - `LogisticRegression` (class_weight="balanced")
  - `LinearSVC` calibré via `CalibratedClassifierCV`
  - `MultinomialNB`
  - `SVC` RBF (class_weight="balanced")
- Sélectionner le meilleur pipeline sur la moyenne de l’AP en CV.
- Choisir un seuil de décision optimale sur la validation (max F1 sur courbe Precision‑Recall).
- Évaluer sur test avec ce seuil, puis sauver `models/model.pkl` et `models/threshold.txt`.

Points importants:
- Gestion du déséquilibre: `class_weight="balanced"` pour LR/SVM; calibration probabiliste pour LinearSVC.
- Grilles hyperparamètres compactes mais efficaces (min_df, ngram_range, C/alpha/solver…).
- Normalisation de score fallback: si `predict_proba` indisponible, on utilise `decision_function` + min‑max.

Exécution:
```bash
python -m src.train
```
Sorties:
- `models/model.pkl`: pipeline sklearn entraîné
- `models/threshold.txt`: seuil (float) choisi via la validation
- Impression des métriques test (accuracy, precision, recall, F1, ROC AUC, AP)

---

## 📊 Évaluation (`src/evaluate.py`)

Fonctions:
- Charge `models/model.pkl` et le `threshold.txt`.
- Lit un CSV (par défaut `data/emails.csv`).
- Calcule un score par email, applique le seuil, affiche:
  - classification report (precision/recall/F1/support)
  - matrice de confusion + faux positifs
  - métriques agrégées (accuracy/precision/recall/F1/ROC AUC/AP + seuil)

Exécution:
```bash
python -m src.evaluate
```

---

## 📓 Notebook d’expérimentation (`spam_emails_detection.ipynb`)

Contenu principal:
- EDA rapide: répartition ham/spam, pie chart, parts de classes.
- Baseline: TF‑IDF + Logistic Regression (référence).
- Comparaison multi‑modèles (LR, LinearSVC, SVM RBF, MultinomialNB, RandomForest) + courbes ROC/PR.
- Pipeline GridSearch multi‑modèles (scoring AP) + sélection de seuil optimal sur validation.
- Évaluation test détaillée: report, matrice de confusion, ROC/PR, seuil utilisé.
- Analyse d’erreurs (faux positifs / faux négatifs marquants) pour guider l’itération.
- Sauvegarde du meilleur pipeline et du seuil (synchronisé avec `src/train.py`).

Le notebook sert de référence lisible et d’espace d’itération, alors que `src/` est prêt pour l’industrialisation.

---

## 🖥️ Application Streamlit (`app/streamlit_app.py`)

Fonctionnalités:
- Chargement du modèle `models/model.pkl` (mise en cache `@st.cache_resource`).
- Chargement du seuil `models/threshold.txt` (fallback 0.5 si absent).
- Inference sur un texte saisi: score spam via `predict_proba` si dispo, sinon `decision_function` min‑max.
- Application du seuil appris pour décider SPAM/HAM (évite le biais d’un 0.5 fixe).
- Upload CSV `text[,label]` et prédictions batch:
  - colonnes ajoutées: `score_spam` (si dispo), `prediction` (spam/ham)
  - visualisations: 
    - répartition des labels vrais (camembert) si `label` présent
    - répartition des prédictions (bar chart)
    - distribution des scores + ligne de seuil
    - métriques agrégées (Accuracy, Precision, Recall, F1) si `label` présent
    - matrice de confusion (heatmap) + `classification_report` détaillé
- Affichage du seuil courant au bas de la page.

L’app illustre l’usage pratique du pipeline et du seuil optimisé sur validation.

---

## 🧩 Logique de bout‑en‑bout

1) Préparation des données
- Lecture CSV, nettoyage, `label_bin`.
- Splits stratifiés reproductibles: train/val/test.

2) Représentation & Modélisation
- Texte → `TfidfVectorizer` (min_df, n‑grammes) → modèle (LR/LinearSVC calibré/NB/SVM RBF).
- Scoring par Average Precision (AP) en CV pour coller au déséquilibre et à la qualité PR.

3) Sélection de seuil
- Sur la validation, on choisit le seuil qui maximise la F1 via la courbe Precision‑Recall.
- Ce seuil est réutilisé à l’inférence (Streamlit + scripts), évitant l’hypothèse 0.5.

4) Évaluation & Persistance
- Métriques complètes sur test, ROC/PR, matrice de confusion.
- Sauvegarde `model.pkl` + `threshold.txt`.

5) Déploiement local
- App Streamlit charge le pipeline et le seuil, propose prédiction unitaire et par CSV, avec visualisations.

---

## 🛠️ Installation & Exécution

1) Environnement
```bash
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

2) Entraînement
```bash
python -m src.train
```
Résultats: `models/model.pkl` et `models/threshold.txt` + métriques test.

3) Évaluation rapide sur tout le CSV
```bash
python -m src.evaluate
```

4) Lancer l’application Streamlit
```bash
streamlit run app/streamlit_app.py
```

---

## ✅ Tests

- `tests/test_model.py`: tests simples de chargement et d’existence des outputs. Étendre pour:
  - valider l’API d’inférence (probas/decision)
  - vérifier la présence de `threshold.txt` et la compatibilité avec le modèle
  - contrôler que l’entraînement produit des métriques minimales (e.g., F1 > 0.9 sur ce dataset)

---

## 🔍 Choix techniques et justifications

- **AP (Average Precision) en CV**: mesure adaptée au déséquilibre; optimise la zone sous PR.
- **class_weight / calibration / seuil**: trio pour mieux capturer les spams avec coût asymétrique.
- **Seuil appris ≠ 0.5**: en pratique, 0.5 n’est pas optimal; on choisit le seuil max F1 sur validation et on l’applique partout.
- **Fallback probas**: `decision_function` + min‑max pour les modèles non probabilistes.
- **Pipeline scikit‑learn**: reproductible, sérialisable (`joblib`), compatible GridSearch.

---

## 🚧 Améliorations possibles

- Enrichir la pré‑proc (lowercase, strip headers/signatures, suppression de tokens très fréquents, normalisation d’URLs et de nombres, stopwords multilingues, stemming/lemmatisation optionnels).
- Essayer des modèles linéaires régularisés (SGDClassifier log/hinge), ou des encoders plus riches (HashingVectorizer, char n‑grams étendus).
- Calibration isotonic/platt globale post‑sélection.
- Courbes ROC/PR intégrées dans l’app (upload avec labels) + sélection de seuil interactive.
- Intégration DVC end‑to‑end (pipelines) et CI (tests + lint + métriques minimales).

---

## 🧾 Licence & Auteurs

Usage pédagogique. Contributions bienvenues pour tester d’autres modèles, enrichir l’app, ou améliorer la robustesse production.
