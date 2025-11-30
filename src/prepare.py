import pandas as pd
from sklearn.model_selection import train_test_split
import yaml
from pathlib import Path

def load_params(path: str = "params.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)
    
def main():
    params = load_params()
    test_size = params["test_size"]
    random_state = params["random_state"]

    raw_path = Path("data/raw/iris.csv")
    df = pd.read_csv(raw_path)

    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)

    X_train.to_csv(processed_dir / "X_train.csv", index=False)
    X_test.to_csv(processed_dir / "X_test.csv", index=False)
    y_train.to_csv(processed_dir / "y_train.csv", index=False)
    y_test.to_csv(processed_dir / "y_test.csv", index=False)

if __name__ == "__main__":
    main()

