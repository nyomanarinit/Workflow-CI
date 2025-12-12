import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import os
import numpy as np
import warnings
import sys

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Dataset path
    file_path = (
        sys.argv[3]
        if len(sys.argv) > 3
        else os.path.join(os.path.dirname(os.path.abspath(__file__)), "churn_preprocessed.csv")
    )

    data = pd.read_csv(file_path)

    # Fitur & target
    X = data.drop("Exited", axis=1)
    y = data["Exited"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=42, test_size=0.2
    )

    input_example = X_train.iloc[0:5]

    # Hyperparameters
    n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else 20

    with mlflow.start_run():
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth
        )

        model.fit(X_train, y_train)
        acc = model.score(X_test, y_test)

        # Log Model
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=input_example,
        )

        # Log Metrics
        mlflow.log_metric("accuracy", acc)

        print("Training Selesai")
        print("Akurasi:", acc)
