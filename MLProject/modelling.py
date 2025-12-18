import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

DATA_PATH = "churn_preprocessed.csv"
EXPERIMENT_NAME = "Customer Churn Prediction"

def run_model():
    print("🚀 Training dimulai")

    mlflow.set_experiment(EXPERIMENT_NAME)

    df = pd.read_csv(DATA_PATH)
    X = df.drop("Exited", axis=1)
    y = df["Exited"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        random_state=42
    )

    with mlflow.start_run():
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        mlflow.log_param("n_estimators", 200)
        mlflow.log_param("max_depth", 15)
        mlflow.log_metric("accuracy", acc)

        # WAJIB untuk build-docker
        mlflow.sklearn.log_model(model, artifact_path="model")

        print(f"✅ Accuracy: {acc}")

if __name__ == "__main__":
    run_model()
