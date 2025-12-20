import pandas as pd
import argparse
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn

DATA_PATH = "churn_preprocessed.csv"
EXPERIMENT_NAME = "Customer Churn Prediction"

def main(args):
    print("🚀 Training dimulai...")

    mlflow.set_experiment(EXPERIMENT_NAME)
    mlflow.sklearn.autolog()  # WAJIB sesuai requirement

    # Load data
    df = pd.read_csv(DATA_PATH)
    X = df.drop("Exited", axis=1)
    y = df["Exited"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Scaling
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

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

        acc = model.score(X_test_scaled, y_test)
        mlflow.log_metric("accuracy", acc)

        # =========================
        # SIMPAN MODEL & SCALER
        # =========================
        joblib.dump(model, "model.pkl")
        joblib.dump(scaler, "scaler.pkl")

        # log ke MLflow juga (artefak)
        mlflow.log_artifact("model.pkl")
        mlflow.log_artifact("scaler.pkl")

        # WAJIB untuk CI / Docker
        mlflow.sklearn.log_model(model, artifact_path="model")

        # Simpan run_id
        run_id = run.info.run_id
        with open("run_id.txt", "w") as f:
            f.write(run_id)

        print(f"✅ Training selesai | Accuracy: {acc}")
        print(f"🆔 Run ID: {run_id}")
        print("📦 model.pkl & scaler.pkl berhasil disimpan")

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
