import yaml
import pandas as pd
import pickle
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import mlflow
import mlflow.sklearn

def load_params(path: str = "params.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)
    
def main():
    params = load_params()
    C = params["C"]
    max_iter = params["max_iter"]
    test_size = params["test_size"]
    random_state = params["random_state"]

    processed_dir = Path("data/processed")
    X_train = pd.read_csv(processed_dir / "X_train.csv")
    X_test = pd.read_csv(processed_dir / "X_test.csv")
    y_train = pd.read_csv(processed_dir / "y_train.csv").values.ravel()
    y_test = pd.read_csv(processed_dir / "y_test.csv").values.ravel()

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("iris-classification")

    with mlflow.start_run():
        clf = LogisticRegression(
            C=C,
            max_iter=max_iter,
        )

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        mlflow.log_param("test_size", test_size)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("C", C)
        mlflow.log_param("max_iter", max_iter)
        mlflow.log_metric("accuracy", acc)

        model_path = Path("model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(clf, f)

        mlflow.log_artifact(str(model_path))
        mlflow.sklearn.log_model(clf, artifact_path="sklearn_model")

        print(f"Accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()

