from pathlib import Path

import pandas as pd
import os
import shutil
import tarfile
import pytest
import tempfile
import joblib
from preprocessor import preprocess

DATA_FILEPATH_0 = Path(r"C:\Users\kamil\Documents\football_project\football_predictor\data\new_features\df.csv")
DATA_FILEPATH_1 = Path(r"C:\Users\kamil\Documents\football_project\football_predictor\data\new_features\y.csv")


@pytest.fixture(scope="function", autouse=False)
def directory():
    directory = tempfile.mkdtemp()
    input_directory = Path(directory) / "input"
    input_directory.mkdir(parents=True, exist_ok=True)
    shutil.copy2(DATA_FILEPATH_0, input_directory / "df.csv")
    shutil.copy2(DATA_FILEPATH_1, input_directory / "y.csv")

    directory = Path(directory)
    preprocess(base_directory=directory)

    yield directory

    shutil.rmtree(directory)


def test_preprocess_generates_data_splits(directory):
    output_directories = os.listdir(directory)

    assert "train" in output_directories
    assert "validation" in output_directories
    assert "test" in output_directories


def test_preprocess_generates_baselines(directory):
    output_directories = os.listdir(directory)

    assert "train-baseline" in output_directories
    assert "test-baseline" in output_directories


def test_preprocess_creates_one_models(directory):
    model_path = directory / "model"
    tar = tarfile.open(model_path / "model.tar.gz", "r:gz")

    assert "features.joblib" in tar.getnames()

def test_train_baseline_includes_header(directory):
    baseline = pd.read_csv(directory / "train-baseline" / "train-baseline.csv")
    assert "result_match" in baseline.columns


def test_test_baseline_does_not_include_header(directory):
    baseline = pd.read_csv(directory / "test-baseline" / "test-baseline.csv")
    assert baseline.columns[0] != "result_match"
