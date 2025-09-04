from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC, SVC

from src.prepare_data import load_dataset, stratified_splits


RANDOM_STATE = 42
MODEL_PATH = Path("models/model.pkl")
THRESHOLD_PATH = Path("models/threshold.txt")
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)


def build_candidates():
    return {
        "LogReg": (
            LogisticRegression(max_iter=400, class_weight="balanced", random_state=RANDOM_STATE),
            {
                "tfidf__min_df": [2, 3],
                "tfidf__ngram_range": [(1, 1), (1, 2)],
                "clf__C": [0.5, 1.0, 2.0],
                "clf__penalty": ["l2"],
                "clf__solver": ["liblinear", "lbfgs"],
            },
        ),
        "LinearSVC": (
            CalibratedClassifierCV(
                base_estimator=LinearSVC(class_weight="balanced", random_state=RANDOM_STATE), cv=3
            ),
            {
                "tfidf__min_df": [2, 3],
                "tfidf__ngram_range": [(1, 1), (1, 2)],
            },
        ),
        "MultinomialNB": (
            MultinomialNB(),
            {
                "tfidf__min_df": [2, 3],
                "tfidf__ngram_range": [(1, 1), (1, 2)],
                "clf__alpha": [0.5, 1.0, 2.0],
            },
        ),
        "SVM_rbf": (
            SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=RANDOM_STATE),
            {
                "tfidf__min_df": [2],
                "tfidf__ngram_range": [(1, 1), (1, 2)],
                "clf__C": [0.5, 1.0],
                "clf__gamma": ["scale"],
            },
        ),
    }


def select_threshold(y_true, probas) -> float:
    prec, rec, thr = precision_recall_curve(y_true, probas)
    f1_vals = (2 * prec * rec) / (prec + rec + 1e-9)
    idx = int(np.nanargmax(f1_vals))
    return 0.5 if idx >= len(thr) else float(thr[max(0, idx - 1)])


def evaluate_predictions(y_true, y_pred, y_proba) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "average_precision": float(average_precision_score(y_true, y_proba)),
    }


def main() -> Tuple[Pipeline, float, Dict[str, float]]:
    df = load_dataset()
    X_train, X_val, X_test, y_train, y_val, y_test = stratified_splits(df)

    candidates = build_candidates()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    best_pipe = None
    best_ap = -np.inf
    for name, (estimator, param_grid) in candidates.items():
        pipe = Pipeline([
            ("tfidf", TfidfVectorizer()),
            ("clf", estimator),
        ])
        grid = GridSearchCV(
            estimator=pipe,
            param_grid=param_grid,
            scoring="average_precision",
            cv=cv,
            n_jobs=-1,
            verbose=0,
            refit=True,
        )
        grid.fit(X_train, y_train)
        if grid.best_score_ > best_ap:
            best_ap = grid.best_score_
            best_pipe = grid.best_estimator_

    # Select threshold on validation
    try:
        proba_val = best_pipe.predict_proba(X_val)[:, 1]
    except Exception:
        dec = best_pipe.decision_function(X_val)
        proba_val = (dec - dec.min()) / (dec.max() - dec.min() + 1e-9)
    threshold = select_threshold(y_val, proba_val)

    # Evaluate on test
    try:
        proba_test = best_pipe.predict_proba(X_test)[:, 1]
    except Exception:
        dec_t = best_pipe.decision_function(X_test)
        proba_test = (dec_t - dec_t.min()) / (dec_t.max() - dec_t.min() + 1e-9)
    y_pred_test = (proba_test >= threshold).astype(int)
    test_metrics = evaluate_predictions(y_test, y_pred_test, proba_test)

    # Persist
    joblib.dump(best_pipe, MODEL_PATH)
    THRESHOLD_PATH.write_text(str(threshold))

    print("✅ Meilleur pipeline sauvegardé →", MODEL_PATH)
    print("✅ Seuil sauvegardé →", THRESHOLD_PATH, "=", threshold)
    print("📊 Test metrics:", test_metrics)

    return best_pipe, threshold, test_metrics


if __name__ == "__main__":
    main()
