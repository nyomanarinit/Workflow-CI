import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn

DATA_PATH = "churn_preprocessed.csv"
EXPERIMENT_NAME = "Customer Churn Modelling"

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment(EXPERIMENT_NAME)
mlflow.sklearn.autolog(log_models=False)


def load_data(path):
    return pd.read_csv(path)


def preprocess(df):

    X = df.drop("Exited", axis=1)
    y = df["Exited"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


def train(X_train, X_test, y_train, y_test, args):
    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        max_features=args.max_features,
        bootstrap=args.bootstrap,
        random_state=42
    )

    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    return model, acc


def main(args):
    df = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test = preprocess(df)

    model, acc = train(X_train, X_test, y_train, y_test, args)

    mlflow.log_metric("accuracy", acc)

    # 🔑 WAJIB untuk Docker
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model"
    )

    print(f"Akurasi: {acc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--n_estimators", type=int, default=300)
    parser.add_argument("--max_depth", type=int, default=20)
    parser.add_argument("--min_samples_split", type=int, default=5)
    parser.add_argument("--min_samples_leaf", type=int, default=2)
    parser.add_argument("--max_features", type=float, default=0.7)
    parser.add_argument("--bootstrap", type=bool, default=True)

    args = parser.parse_args()
    main(args)
