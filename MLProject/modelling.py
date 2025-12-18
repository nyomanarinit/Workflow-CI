import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
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
def load_data(path):
    if not path:
        raise ValueError("Path dataset kosong")

    df = pd.read_csv(path)
    print("Dataset berhasil dimuat")
    return df


# =========================
# PREPROCESSING
# =========================
def preprocess_data(df):
    drop_cols = ["RowNumber", "CustomerId", "Surname"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    X = df.drop("Exited", axis=1)
    y = df["Exited"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print("Preprocessing selesai")
    return X_train, X_test, y_train, y_test


# =========================
# TRAINING MODEL
# =========================
def train_model(X_train, X_test, y_train, y_test, args):
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
    accuracy = model.score(X_test, y_test)

    return model, accuracy


# =========================
# MAIN PIPELINE
# =========================
def main(args):
    mlflow.set_experiment(EXPERIMENT_NAME)
    mlflow.sklearn.autolog()

    with mlflow.start_run():
        df = load_data(DATA_PATH)
        X_train, X_test, y_train, y_test = preprocess_data(df)

        model, acc = train_model(
            X_train, X_test,
            y_train, y_test,
            args
        )

        mlflow.log_metric("accuracy", acc)
        print(f"Akurasi model: {acc:.4f}")


# =========================
# ARGUMENT CLI
# =========================
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
