import pandas as pd
import argparse
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

DATA_PATH = "churn_preprocessed.csv"
EXPERIMENT_NAME = "Customer Churn Prediction"

def main(args):
    mlflow.set_experiment(EXPERIMENT_NAME)
    mlflow.sklearn.autolog()

    df = pd.read_csv(DATA_PATH)

    X = df.drop("Exited", axis=1)
    y = df["Exited"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    pipeline = Pipeline([
        ("scaler", MinMaxScaler()),
        ("model", RandomForestClassifier(
            n_estimators=args.n_estimators,
            min_samples_split=args.min_samples_split,
            min_samples_leaf=args.min_samples_leaf,
            max_features=args.max_features,
            max_depth=args.max_depth,
            bootstrap=args.bootstrap,
            random_state=42
        ))
    ])

    with mlflow.start_run():
        pipeline.fit(X_train, y_train)
        acc = pipeline.score(X_test, y_test)
        mlflow.log_metric("accuracy", acc)

        mlflow.sklearn.log_model(
            pipeline,
            artifact_path="model"
        )

        print(f"✅ Training selesai | Accuracy: {acc}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=500)
    parser.add_argument("--min_samples_split", type=int, default=5)
    parser.add_argument("--min_samples_leaf", type=int, default=4)
    parser.add_argument("--max_features", type=float, default=0.5)
    parser.add_argument("--max_depth", type=int, default=15)
    parser.add_argument("--bootstrap", type=bool, default=True)
    args = parser.parse_args()

    main(args)
