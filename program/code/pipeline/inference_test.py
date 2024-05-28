import json
import os
import shutil
import tarfile
from pathlib import Path

import pytest
import tempfile

from pythonProject.program.code.preprocessor import preprocess
from pythonProject.program.code.train import train
from pythonProject.program.code.pipeline.inference import model_fn, input_fn, predict_fn, output_fn
from dotenv import load_dotenv

load_dotenv()

DATA_FILEPATH_X = os.environ['DATA_FILEPATH_X']
DATA_FILEPATH_Y = os.environ['DATA_FILEPATH_Y']


@pytest.fixture(scope="function", autouse=False)
def directory():
    directory = tempfile.mkdtemp()
    input_directory = Path(directory) / "input"
    input_directory.mkdir(parents=True, exist_ok=True)
    shutil.copy2(DATA_FILEPATH_X, input_directory / "X.csv")
    shutil.copy2(DATA_FILEPATH_Y, input_directory / "y.csv")

    directory = Path(directory)

    preprocess(base_directory=directory)

    train(
        model_directory=directory / "model",
        train_path=directory / "train",
        validation_path=directory / "validation",
        early_stopping_rounds=50,
        hyperparameters={},
        pipeline_path=directory / "model"
    )

    # After training a model, we need to prepare a package just like
    # SageMaker would. This package is what the evaluation script is
    # expecting as an input.
    # Ensure the model.tar.gz is created correctly
    model_tar_path = directory / "model" / "model.tar.gz"
    with tarfile.open(model_tar_path, "w:gz") as tar:
        tar.add(directory / "model", arcname=".")

    os.environ["MODEL_PATH"] = str(directory / "model")

    yield directory

    shutil.rmtree(directory)


@pytest.fixture(scope="function", autouse=False)
def payload():
    return json.dumps({
        "player_rating_home_player_1": 89,
        "player_rating_home_player_2": 79,
        "player_rating_home_player_3": 59,
        "player_rating_home_player_4": 69,
        "player_rating_home_player_5": 69,
        "player_rating_home_player_6": 79,
        "player_rating_home_player_7": 69,
        "player_rating_home_player_8": 79,
        "player_rating_home_player_9": 69,
        "player_rating_home_player_10": 89,
        "player_rating_home_player_11": 89,
        "player_rating_away_player_1": 79,
        "player_rating_away_player_2": 79,
        "player_rating_away_player_3": 79,
        "player_rating_away_player_4": 79,
        "player_rating_away_player_5": 79,
        "player_rating_away_player_6": 79,
        "player_rating_away_player_7": 80,
        "player_rating_away_player_8": 80,
        "player_rating_away_player_9": 71,
        "player_rating_away_player_10": 83,
        "player_rating_away_player_11": 80,
        "ewm_home_team_goals": 5.54,
        "ewm_away_team_goals": 0.61,
        "ewm_home_team_goals_conceded": 0.26,
        "ewm_away_team_goals_conceded": 4.76,
        "points_home": 30,
        "points_away": 15,
        "home_weighted_wins": 5.377149515625,
        "away_weighted_wins": 2.5561203576634663,
        "avg_home_team_rating": 84.18,
        "avg_away_team_rating": 70.91,
        "home_streak_wins": 11.75,
        "away_streak_wins": 5.58,
        "ewm_shoton_home": 3.55,
        "ewm_shoton_away": 1.805,
        "ewm_possession_home": 53.639,
        "ewm_possession_away": 20.03,
        "avg_home_rating_attack": 71.33,
        "avg_away_rating_attack": 78.83,
        "avg_away_rating_defence": 79.0,
        "avg_home_rating_defence": 71.0,
        "average_rating_home": 89.18181818181819,
        "average_rating_away": 78.9090909090909,
        "num_top_players_home": 0,
        "num_top_players_away": 0,
        "ewm_home_team_goals_conceded_x_ewm_shoton_home": 4.473,
        "attacking_strength_home": 80.233606557377048,
        "attacking_strength_away": 31.40637450199203,
        "attacking_strength_diff": -2.172767944614982
    })


def test_handler_response_contains_prediction_and_confidence(directory, payload):
    model = model_fn(directory / "model")
    input_data = input_fn(payload, "application/json")
    prediction = predict_fn(input_data, model)
    response, _ = output_fn(prediction, "application/json")

    assert "prediction" in response
    assert "confidence" in response


def test_handler_response_includes_content_type(directory, payload):
    model = model_fn(directory / "model")
    input_data = input_fn(payload, "application/json")
    prediction = predict_fn(input_data, model)
    _, content_type = output_fn(prediction, "application/json")

    assert content_type == "application/json"


def test_handler_response_prediction_is_categorical(directory, payload):
    model = model_fn(directory / "model")
    input_data = input_fn(payload, "application/json")
    prediction = predict_fn(input_data, model)
    response, _ = output_fn(prediction, "application/json")

    assert response["prediction"] in ["home_not_win", "home_win"]


def test_handler_deals_with_an_invalid_payload(directory):
    model = model_fn(directory / "model")
    with pytest.raises(json.JSONDecodeError):
        input_fn("invalid payload", "application/json")