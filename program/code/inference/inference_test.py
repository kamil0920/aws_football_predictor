import json
import os
import shutil
import tarfile
from pathlib import Path

import pytest
import tempfile

from pythonProject.program.code.preprocessor.preprocessor import preprocess
from pythonProject.program.code.containers.training.train import train
from pythonProject.program.code.inference.inference import model_fn, input_fn, predict_fn, output_fn
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
        pipeline_path=directory / "model",
        experiment=None
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
    return json.dumps([{
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
    },
        {
            "player_rating_home_player_1": 80,
            "player_rating_home_player_2": 70,
            "player_rating_home_player_3": 50,
            "player_rating_home_player_4": 60,
            "player_rating_home_player_5": 60,
            "player_rating_home_player_6": 70,
            "player_rating_home_player_7": 60,
            "player_rating_home_player_8": 70,
            "player_rating_home_player_9": 60,
            "player_rating_home_player_10": 80,
            "player_rating_home_player_11": 80,
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
        }]
    )


@pytest.fixture(scope="function", autouse=False)
def payload_csv():
    return """
86,86,86,86,86,86,83,83,83,88,74,76,76,76,76,76,76,79,68,70,81,79,1.72,2.24,1.7,0.76,49,80,3.2707983786677315,1.3128173828363914,84.27,75.73,22.92,40.63,7.048,5.128,47.796,52.362,82.83,75.5,76.0,86.0,84.27272727272727,75.72727272727273,7,0,11.9816,31.614503816793892,24.044585987261147,7.569917829532745
79,79,79,79,79,79,81,86,88,85,82,76,76,76,76,76,76,74,75,72,76,75,1.52,0.82,0.92,1.92,68,17,4.144031675155845,3.887158709814261,81.45,75.27,50.07,3.55,7.592,4.772,43.5,52.919,83.5,74.67,76.0,79.0,81.45454545454545,75.27272727272727,3,0,6.98464,34.50413223140496,43.41279069767442,-8.908658466269458
77,77,77,77,77,77,76,75,75,79,77,80,80,80,80,80,80,78,80,86,73,88,2.15,2.24,1.63,1.67,46,65,2.6942883775913384,1.008610065655301,76.73,80.45,42.34,27.71,5.379,7.424,50.032,45.678,76.5,80.83,80.0,77.0,76.72727272727273,80.45454545454545,0,2,8.767769999999999,25.081967213114755,25.74203821656051,-0.6600710034457542
"""


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

    response = json.loads(response)

    assert response[0]["prediction"] == "home_win"
    assert response[0]["confidence"] is not None


def test_output_response_predictions_values(directory, payload_csv):
    model = model_fn(directory / "model")
    input_data = input_fn(payload_csv, "text/csv")
    prediction = predict_fn(input_data, model)
    response, _ = output_fn(prediction, "text/csv")

    assert response[0][0] == 'home_win'
    assert response[1][0] == 'home_win'
    assert response[2][0] == 'home_not_win'


def test_handler_deals_with_an_invalid_payload(directory):
    model = model_fn(directory / "model")
    with pytest.raises(json.JSONDecodeError):
        input_fn("invalid payload", "application/json")
