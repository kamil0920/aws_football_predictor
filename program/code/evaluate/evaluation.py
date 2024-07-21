import json
import tarfile
import pandas as pd

from pathlib import Path
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, precision_score, recall_score


def evaluate(model_path, test_path, output_path):
    X_test = pd.read_csv(Path(test_path) / "test.csv")
    y_test = X_test['result_match']
    X_test.drop(labels=['result_match'], axis=1, inplace=True)

    with tarfile.open(Path(model_path) / "model.tar.gz") as tar:
        tar.extractall(path=Path(model_path))

    extracted_dir = Path(model_path)
    model_filepath = extracted_dir / "saved_model.xgb"

    model = XGBClassifier()
    model.load_model(str(model_filepath))

    predictions = model.predict(X_test)

    f1 = f1_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)

    print(f"Test f1 score: {f1}")
    print(f"Test precision score: {precision}")
    print(f"Test recall score: {recall}")

    evaluation_report = {
        "metrics": {
            "f1": {
                "value": f1
            },
            "precision": {
                "value": precision
            },
            "recall": {
                "value": recall
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
