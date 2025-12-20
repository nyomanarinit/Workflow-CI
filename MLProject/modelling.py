"""
====================================================
Customer Churn Model Training
Algoritma  : Random Forest Classifier
Framework  : Scikit-learn + MLflow
Tujuan    : Training, logging, dan persiapan Docker
====================================================
"""

import os
import pandas as pd
import joblib
import argparse
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def train_churn_model(args):
    """
    Fungsi utama untuk:
    1. Load data hasil preprocessing
    2. Training model Random Forest
    3. Logging ke MLflow
    4. Menyimpan artefak model & scaler
    """

    # =================================================
    # 1. Konfigurasi dasar & path data (DINAMIS)
    # =================================================
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "churn_preprocessed.csv")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset tidak ditemukan di: {data_path}")

    mlflow.set_experiment("Customer Churn Prediction")

    print("🚀 Memulai proses training model churn...")

    # =================================================
    # 2. Load Dataset
    # =================================================
    df = pd.read_csv(data_path)

    # =================================================
    # 3. Feature Selection (berdasarkan studi kasus churn)
    # =================================================
    feature_columns = [
        "CreditScore",
        "Age",
        "Tenure",
        "Balance",
        "IsActiveMember",
        "EstimatedSalary"
    ]

    X = df[feature_columns]
    y = df["Exited"]

    # =================================================
    # 4. Split Data
    # =================================================
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    # =================================================
    # 5. Scaling (penting untuk konsistensi inference)
    # =================================================
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # =================================================
    # 6. Aktifkan MLflow Autolog
    # =================================================
    mlflow.sklearn.autolog()

    # =================================================
    # 7. Training Model
    # =================================================
    with mlflow.start_run() as run:
        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            min_samples_split=args.min_samples_split,
            min_samples_leaf=args.min_samples_leaf,
            max_features=args.max_features,
            max_depth=args.max_depth,
            bootstrap=args.bootstrap,
            random_state=42
        )

        model.fit(X_train_scaled, y_train)

        # =================================================
        # 8. Evaluasi Model
        # =================================================
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)

        mlflow.log_metric("accuracy", accuracy)

        print(f"✅ Training selesai | Accuracy: {accuracy:.4f}")

        # =================================================
        # 9. Simpan artefak lokal (untuk inference & CI)
        # =================================================
        model_path = os.path.join(base_dir, "model.pkl")
        scaler_path = os.path.join(base_dir, "scaler.pkl")

        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)

        # =================================================
        # 10. Log artefak & model ke MLflow
        # =================================================
        mlflow.log_artifact(model_path)
        mlflow.log_artifact(scaler_path)

        # WAJIB: agar bisa dipakai mlflow models build-docker
        mlflow.sklearn.log_model(model, artifact_path="model")

        # =================================================
        # 11. Simpan run_id (dipakai di GitHub Actions)
        # =================================================
        run_id = run.info.run_id
        with open(os.path.join(base_dir, "run_id.txt"), "w") as f:
            f.write(run_id)

        print(f"🆔 Run ID tersimpan: {run_id}")
        print("📦 model.pkl & scaler.pkl berhasil disimpan")


# =====================================================
# Entry Point
# =====================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training Random Forest Churn Model")

    parser.add_argument("--n_estimators", type=int, default=500)
    parser.add_argument("--min_samples_split", type=int, default=5)
    parser.add_argument("--min_samples_leaf", type=int, default=4)
    parser.add_argument("--max_features", type=float, default=0.5)
    parser.add_argument("--max_depth", type=int, default=15)
    parser.add_argument("--bootstrap", type=bool, default=True)

    args = parser.parse_args()

    train_churn_model(args)
