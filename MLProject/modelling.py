import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn
import os

# =============================
# CONFIG
# =============================
DATA_PATH = "churn_preprocessed.csv"
EXPERIMENT_NAME = "Customer Churn Prediction"

# Tracking lokal (WAJIB untuk GitHub Actions)
mlflow.set_tracking_uri("file:./mlruns")

def run_model(args):
    print("🚀 Training dimulai...")

    mlflow.set_experiment(EXPERIMENT_NAME)
    mlflow.sklearn.autolog(log_input_examples=True)

    # =============================
    # LOAD DATA
    # =============================
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset tidak ditemukan: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    print("✅ Dataset berhasil dimuat")

    X = df.drop("Exited", axis=1)
    y = df["Exited"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # =============================
    # TRAIN MODEL
    # =============================
    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        max_features=args.max_features,
        max_depth=args.max_depth,
        bootstrap=args.bootstrap,
        random_state=42
    )

    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    print(f"🎯 Akurasi: {acc}")

    # =============================
    # AMBIL RUN AKTIF (INI KUNCI!)
    # =============================
    run = mlflow.active_run()
    run_id = run.info.run_id

    print(f"📁 Run ID: {run_id}")

    # Simpan run_id untuk GitHub Actions
    with open("run_id.txt", "w") as f:
        f.write(run_id)

    print("💾 run_id.txt berhasil dibuat")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=500)
    parser.add_argument("--min_samples_split", type=int, default=5)
    parser.add_argument("--min_samples_leaf", type=int, default=4)
    parser.add_argument("--max_features", type=float, default=0.5)
    parser.add_argument("--max_depth", type=int, default=15)
    parser.add_argument("--bootstrap", type=bool, default=True)
    args = parser.parse_args()

    run_model(args)
