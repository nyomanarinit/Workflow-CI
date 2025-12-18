import pandas as pd
import argparse
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier


DATASET_FILE = "churn_preprocessing.csv"
EXPERIMENT_NAME = "Customer Churn Prediction"


def read_dataset(path):
    print("📥 Membaca dataset churn...")
    data = pd.read_csv(path)
    return data


def prepare_features(dataframe):
    print("🧹 Menyiapkan fitur dan target...")

    target = dataframe["Exited"]
    features = dataframe.drop(columns=["Exited"])

    # Encoding variabel kategorikal (aman & umum)
    features_encoded = pd.get_dummies(features, drop_first=True)

    return features_encoded, target


def split_scale_data(X, y):
    print("🔄 Split dan scaling data...")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test


def initialize_model(args):
    print("⚙️ Inisialisasi Random Forest...")

    return RandomForestClassifier(
        n_estimators=args.n_estimators,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        max_features=args.max_features,
        max_depth=args.max_depth,
        bootstrap=args.bootstrap,
        random_state=42
    )


def train_log_model(model, X_train, y_train, X_test, y_test):
    with mlflow.start_run():
        print("🚀 Training dimulai...")

        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)

        mlflow.log_metric("accuracy", accuracy)

        # WAJIB untuk Advance & Docker
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model"
        )

        print(f"✅ Training selesai | Accuracy: {accuracy}")


def main(args):
    mlflow.set_experiment(EXPERIMENT_NAME)
    mlflow.sklearn.autolog(log_input_examples=True)

    df = read_dataset(DATASET_FILE)
    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test = split_scale_data(X, y)

    model = initialize_model(args)
    train_log_model(model, X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Customer Churn Model Training")

    parser.add_argument("--n_estimators", type=int, default=500)
    parser.add_argument("--min_samples_split", type=int, default=5)
    parser.add_argument("--min_samples_leaf", type=int, default=4)
    parser.add_argument("--max_features", type=float, default=0.5)
    parser.add_argument("--max_depth", type=int, default=15)
    parser.add_argument("--bootstrap", type=bool, default=True)

    args = parser.parse_args()
    main(args)
