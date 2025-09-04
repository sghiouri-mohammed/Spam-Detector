from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42
DATA_PATH = Path("data/emails.csv")


def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset introuvable: {path}")
    df = pd.read_csv(path).dropna(subset=["text", "label"]).copy()
    df["label_bin"] = (df["label"].str.lower() == "spam").astype(int)
    return df


def stratified_splits(
    df: pd.DataFrame,
    test_size: float = 0.2,
    val_size_in_train: float = 0.25,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    X_all = df["text"].astype(str)
    y_all = df["label_bin"].astype(int)

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X_all, y_all, test_size=test_size, random_state=RANDOM_STATE, stratify=y_all
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_size_in_train, random_state=RANDOM_STATE, stratify=y_trainval
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":
    df = load_dataset()
    X_train, X_val, X_test, y_train, y_val, y_test = stratified_splits(df)
    print("Répartition des classes (total):")
    print(df["label"].value_counts())
    print("Tailles:", len(X_train), len(X_val), len(X_test))


