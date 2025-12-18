import os
import argparse
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def run_model(args):
    print("🚀 Training dimulai...")

    # Ambil run_id dari MLflow Project (PENTING)
    run_id = os.environ.get("MLFLOW_RUN_ID")
    print(f"📁 Run ID: {run_id}")

    # Simpan run_id untuk GitHub Actions
    with open("run_id.txt", "w") as f:
        f.write(run_id)

    # Load dataset
    df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    print("✅ Dataset berhasil dimuat")

    X = df.drop("Churn", axis=1)
    y = df["Churn"].apply(lambda x: 1 if x == "Yes" else 0)

    X_train, X_test, y_train, y_test = train_test_split(
        X.select_dtypes(include="number"),
        y,
        test_size=0.2,
        random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

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

    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"🎯 Akurasi: {acc}")

    # LOG ke MLflow TANPA start_run()
    mlflow.log_params(vars(args))
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(model, "model")

    print("✅ Model & metrics berhasil dilog")

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
