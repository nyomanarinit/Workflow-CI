import argparse
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import mlflow
import mlflow.sklearn

# =========================
# KONFIGURASI
# =========================
DATA_PATH = "churn_preprocessed.csv"
EXPERIMENT_NAME = "Customer Churn Modelling"

mlflow.set_tracking_uri("file:./mlruns")


# =========================
# LOAD DATA
# =========================
def load_data(path: str) -> pd.DataFrame:
    if not path:
        raise ValueError("Path dataset tidak boleh kosong")

    df = pd.read_csv(path)
    return df


# =========================
# PREPROCESSING
# =========================
def prepare_data(df: pd.DataFrame):

    X = df.drop("Exited", axis=1)
    y = df["Exited"]

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )


# =========================
# MAIN PIPELINE
# =========================
def main(args):
    mlflow.set_experiment(EXPERIMENT_NAME)
    mlflow.sklearn.autolog(log_models=True)

    with mlflow.start_run():
        df = load_data(DATA_PATH)
        X_train, X_test, y_train, y_test = prepare_data(df)

        pipeline = Pipeline(steps=[
            ("scaler", MinMaxScaler()),
            ("model", RandomForestClassifier(
                n_estimators=args.n_estimators,
                max_depth=args.max_depth,
                min_samples_split=args.min_samples_split,
                min_samples_leaf=args.min_samples_leaf,
                max_features=args.max_features,
                bootstrap=args.bootstrap,
                random_state=42,
                n_jobs=-1
            ))
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        mlflow.log_metrics({
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred)
        })


# =========================
# CLI ARGUMENT
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--n_estimators", type=int, default=500)
    parser.add_argument("--max_depth", type=int, default=15)
    parser.add_argument("--min_samples_split", type=int, default=5)
    parser.add_argument("--min_samples_leaf", type=int, default=4)
    parser.add_argument("--max_features", type=float, default=0.5)
    parser.add_argument("--bootstrap", action="store_true")

    args = parser.parse_args()
    main(args)
