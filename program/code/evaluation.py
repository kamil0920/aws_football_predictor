import json
import tarfile
import pandas as pd

from pathlib import Path
from xgboost import XGBClassifier
from sklearn.metrics import f1_score


def evaluate(model_path, test_path, output_path):
    X_test = pd.read_csv(Path(test_path) / "test.csv")
    y_test = X_test[X_test.columns[-1]]
    X_test.drop(X_test.columns[-1], axis=1, inplace=True)

    # Let's now extract the model package so we can load
    # it in memory.
    with tarfile.open(Path(model_path) / "model.tar.gz") as tar:
        tar.extractall(path=Path(model_path))

    extracted_dir = Path(model_path) / "001"
    model_filepath = extracted_dir / "saved_model.bst"

    model = XGBClassifier()
    model.load_model(str(model_filepath))

    predictions = model.predict(X_test)
    f1 = f1_score(y_test, predictions)
    print(f"Test f1 score: {f1}")

    evaluation_report = {
        "metrics": {
            "f1": {
                "value": f1
            },
        },
    }

    Path(output_path).mkdir(parents=True, exist_ok=True)
    with open(Path(output_path) / "evaluation.json", "w") as f:
        f.write(json.dumps(evaluation_report))


if __name__ == "__main__":
    evaluate(
        model_path="/opt/ml/processing/model/",
        test_path="/opt/ml/processing/test/",
        output_path="/opt/ml/processing/evaluation/"
    )