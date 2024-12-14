import json
import os
import shutil
import tarfile
import tempfile
from pathlib import Path

import pytest
from dotenv import load_dotenv

from evaluation import evaluate
from aws_football_predictor.program.code.containers.training.train import train
from aws_football_predictor.program.code.containers.preprocessor.preprocessor import preprocess

load_dotenv()


@pytest.fixture(scope="function", autouse=False)
def directory():
    directory = tempfile.mkdtemp()
    input_directory = Path(directory) / "input"
    input_directory.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(os.environ['DATA_FILEPATH_X']), input_directory / "df.csv")
    shutil.copy2(str(os.environ['DATA_FILEPATH_Y']), input_directory / "y.csv")

    directory = Path(directory)

    os.environ["TEST"] = 'True'

    preprocess(base_directory=directory)

    params = {
        'colsample_bytree': 0.8448004658642061,
        'gamma': 19.050680112277224,
        'learning_rate': 0.10552574516231328,
        'max_depth': 9,
        'min_child_weight': 0.49876571387552704,
        'reg_alpha': 5.065713059538013,
        'reg_lambda': 9.082892225273682,
        'subsample': 0.8953781772368965
    }

    train(
        model_directory=directory / "model",
        train_path=directory / "train",
        pipeline_path=directory / "model",
        hyperparameters=params,
        validation_path=directory / "validation"
    )

    # After training a model, we need to prepare a package just like
    # SageMaker would. This package is what the evaluation script is
    # expecting as an input.
    with tarfile.open(directory / "model.tar.gz", "w:gz") as tar:
        tar.add(directory / "model")

    evaluate(
        model_path=directory / "model",
        input_path=directory / "test",
        output_path=directory / "evaluation",
    )

    yield directory / "evaluation"

    shutil.rmtree(directory)


def test_evaluate_generates_evaluation_report(directory):
    output = os.listdir(directory)
    assert "evaluation.json" in output


def test_evaluation_report_contains_accuracy(directory):
    with open(directory / "evaluation.json", 'r') as file:
        report = json.load(file)

    assert "metrics" in report
    assert "precision" in report["metrics"]
