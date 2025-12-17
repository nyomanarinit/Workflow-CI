import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn

# =============================
# CONFIG
# =============================
DATA_PATH = "churn_preprocessed.csv"
EXPERIMENT_NAME = "Customer Churn Prediction"

# Tracking lokal untuk workflow CI
mlflow.set_tracking_uri("file:./mlruns")


def run_model(args):

    mlflow.set_experiment(EXPERIMENT_NAME)
    mlflow.sklearn.autolog(log_input_examples=True)

    print("🚀 Training dimulai...")

    # =============================
    # LOAD DATA
    # =============================
    try:
        df = pd.read_csv(DATA_PATH)
        print("✅ Dataset berhasil dimuat")
    except FileNotFoundError:
        print(f"❌ Dataset tidak ditemukan di {DATA_PATH}")
        return

    X = df.drop("Exited", axis=1)
    y = df["Exited"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print("📌 Data berhasil dibagi train-test")

    # =============================
    # SCALING
    # =============================
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("📌 Scaling selesai")

    # =============================
    # MLflow Run
    # =============================
    with mlflow.start_run() as run:
        print("🤖 Melatih model...")

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
        score = model.score(X_test_scaled, y_test)

        print(f"🎯 Akurasi: {score}")
        print(f"📁 Run ID: {run.info.run_id}")
        print(f"📦 Artifacts: {run.info.artifact_uri}")

        # Save Run ID untuk CI/CD
        with open("run_id.txt", "w") as f:
            f.write(run.info.run_id)

        print("💾 Run ID disimpan ke run_id.txt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--n_estimators", type=int, default=200)
    parser.add_argument("--min_samples_split", type=int, default=2)
    parser.add_argument("--min_samples_leaf", type=int, default=1)
    parser.add_argument("--max_features", type=float, default=0.8)
    parser.add_argument("--max_depth", type=int, default=10)
    parser.add_argument("--bootstrap", type=bool, default=True)

    args = parser.parse_args()

    run_model(args)
