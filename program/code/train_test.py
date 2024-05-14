import os
import shutil
import tarfile
from pathlib import Path

import pytest
import tempfile
import joblib

from dotenv import load_dotenv

from preprocessor import preprocess
from train import train

load_dotenv()

@pytest.fixture(scope="function", autouse=False)
def directory():
    directory = tempfile.mkdtemp()
    input_directory = Path(directory) / "input"
    input_directory.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(os.environ['DATA_FILEPATH_X']), input_directory / "df.csv")
    shutil.copy2(str(os.environ['DATA_FILEPATH_Y']), input_directory / "y.csv")

    directory = Path(directory)
    preprocess(base_directory=directory)
    train(
        model_directory=directory / "model",
        train_path=directory / "train",
        validation_path=directory / "validation",
        early_stopping_rounds=50,
        hyperparameters={}
    )

    yield directory

    shutil.rmtree(directory)


def test_train_saves_a_folder_with_model_assets(directory):
    output = os.listdir(directory / "model")
    assert "saved_model.json" in output

    assets = os.listdir(directory / "model")
    assert "saved_model.json" in assets
