import json
import os
import shutil
import tarfile
import tempfile
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

import pytest

from preprocessing_component import input_fn, predict_fn, output_fn, model_fn
from pythonProject.program.code import preprocessor

load_dotenv()


@pytest.fixture(scope="function", autouse=False)
def directory():
    directory = tempfile.mkdtemp()
    input_directory = Path(directory) / "input"
    input_directory.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(os.environ['DATA_FILEPATH_X']), input_directory / "df.csv")
    shutil.copy2(str(os.environ['DATA_FILEPATH_Y']), input_directory / "y.csv")

    directory = Path(directory)

    preprocessor.preprocess(base_directory=directory)

    with tarfile.open(directory / "model" / "model.tar.gz") as tar:
        tar.extractall(path=directory / "model")

    yield directory / "model"

    shutil.rmtree(directory)


def test_input_csv_drops_target_column_if_present():
    input_data = """
    77,77,77,77,77,77,76,75,75,79,77,80,80,80,80,80,80,78,80,86,73,88,2.15,2.24,1.63,1.67,46,65,2.6942883775913384,1.008610065655301,76.73,80.45,42.34,27.71,5.379,7.424,50.032,45.678,76.5,80.83,80.0,77.0,76.72727272727273,80.45454545454545,2.1244075829383884,2.0757062146892653,1
    """

    df = input_fn(input_data, "text/csv")
    assert len(df.columns) == 46 and "result_match" not in df.columns


def test_input_json_drops_target_column_if_present():
    input_data = json.dumps({
        "player_rating_home_player_1": 79, "player_rating_home_player_2": 79,
        "player_rating_home_player_3": 79, "player_rating_home_player_4": 79,
        "player_rating_home_player_5": 79, "player_rating_home_player_6": 79,
        "player_rating_home_player_7": 81, "player_rating_home_player_8": 86,
        "player_rating_home_player_9": 88, "player_rating_home_player_10": 85,
        "player_rating_home_player_11": 82, "player_rating_away_player_1": 76,
        "player_rating_away_player_2": 76, "player_rating_away_player_3": 76,
        "player_rating_away_player_4": 76, "player_rating_away_player_5": 76,
        "player_rating_away_player_6": 76, "player_rating_away_player_7": 74,
        "player_rating_away_player_8": 75, "player_rating_away_player_9": 72,
        "player_rating_away_player_10": 76, "player_rating_away_player_11": 75,
        "ewm_home_team_goals": 1.52, "ewm_away_team_goals": 0.82,
        "ewm_home_team_goals_conceded": 0.92, "ewm_away_team_goals_conceded": 1.92,
        "points_home": 68, "points_away": 17,
        "home_weighted_wins": 4.144031675155845, "away_weighted_wins": 3.887158709814261,
        "avg_home_team_rating": 81.45, "avg_away_team_rating": 75.27,
        "home_streak_wins": 50.07, "away_streak_wins": 3.55,
        "ewm_shoton_home": 7.592, "ewm_shoton_away": 4.772,
        "ewm_possession_home": 43.5, "ewm_possession_away": 52.919,
        "avg_home_rating_attack": 83.5, "avg_away_rating_attack": 74.67,
        "avg_away_rating_defence": 76.0, "avg_home_rating_defence": 79.0,
        "average_rating_home": 81.45454545454545, "average_rating_away": 75.27272727272727,
        "defensive_weakness_home": 1.1294642857142858, "defensive_weakness_away": 2.5507246376811596,
        "result_match": 1
    })

    df = input_fn(input_data, "application/json")
    assert len(df.columns) == 46 and "result_match" not in df.columns


def test_input_csv_works_without_target_column():
    input_data = """
        77,77,77,77,77,77,76,75,75,79,77,80,80,80,80,80,80,78,80,86,73,88,2.15,2.24,1.63,1.67,46,65,2.6942883775913384,1.008610065655301,76.73,80.45,42.34,27.71,5.379,7.424,50.032,45.678,76.5,80.83,80.0,77.0,76.72727272727273,80.45454545454545,2.1244075829383884,2.0757062146892653
        """

    df = input_fn(input_data, "text/csv")
    assert len(df.columns) == 46


def test_input_json_works_without_target_column():
    input_data = json.dumps({
        "player_rating_home_player_1": 79, "player_rating_home_player_2": 79,
        "player_rating_home_player_3": 79, "player_rating_home_player_4": 79,
        "player_rating_home_player_5": 79, "player_rating_home_player_6": 79,
        "player_rating_home_player_7": 81, "player_rating_home_player_8": 86,
        "player_rating_home_player_9": 88, "player_rating_home_player_10": 85,
        "player_rating_home_player_11": 82, "player_rating_away_player_1": 76,
        "player_rating_away_player_2": 76, "player_rating_away_player_3": 76,
        "player_rating_away_player_4": 76, "player_rating_away_player_5": 76,
        "player_rating_away_player_6": 76, "player_rating_away_player_7": 74,
        "player_rating_away_player_8": 75, "player_rating_away_player_9": 72,
        "player_rating_away_player_10": 76, "player_rating_away_player_11": 75,
        "ewm_home_team_goals": 1.52, "ewm_away_team_goals": 0.82,
        "ewm_home_team_goals_conceded": 0.92, "ewm_away_team_goals_conceded": 1.92,
        "points_home": 68, "points_away": 17,
        "home_weighted_wins": 4.144031675155845, "away_weighted_wins": 3.887158709814261,
        "avg_home_team_rating": 81.45, "avg_away_team_rating": 75.27,
        "home_streak_wins": 50.07, "away_streak_wins": 3.55,
        "ewm_shoton_home": 7.592, "ewm_shoton_away": 4.772,
        "ewm_possession_home": 43.5, "ewm_possession_away": 52.919,
        "avg_home_rating_attack": 83.5, "avg_away_rating_attack": 74.67,
        "avg_away_rating_defence": 76.0, "avg_home_rating_defence": 79.0,
        "average_rating_home": 81.45454545454545, "average_rating_away": 75.27272727272727,
        "defensive_weakness_home": 1.1294642857142858, "defensive_weakness_away": 2.5507246376811596
    })

    df = input_fn(input_data, "application/json")
    assert len(df.columns) == 46


def test_output_csv_raises_exception_if_prediction_is_none():
    with pytest.raises(Exception):
        output_fn(None, "text/csv")


def test_output_json_raises_exception_if_prediction_is_none():
    with pytest.raises(Exception):
        output_fn(None, "application/json")


def test_output_csv_returns_prediction():
    prediction = np.array([
        [77, 77, 77, 77, 77, 77, 76, 75, 75, 79, 77, 80, 80, 80, 80, 80, 80, 78, 80, 86, 73, 88, 2.15, 2.24, 1.63, 1.67, 46, 65, 2.6942883775913384, 1.008610065655301, 76.73, 80.45, 42.34, 27.71,
         5.379, 7.424, 50.032, 45.678, 76.5, 80.83, 80.0, 77.0, 76.72727272727273, 80.45454545454545, 2.1244075829383884, 2.0757062146892653, 1],
        [79, 79, 79, 79, 79, 79, 81, 86, 88, 85, 82, 76, 76, 76, 76, 76, 76, 74, 75, 72, 76, 75, 1.52, 0.82, 0.92, 1.92, 68, 17, 4.144031675155845, 3.887158709814261, 81.45, 75.27, 50.07, 3.55, 7.592,
         4.772, 43.5, 52.919, 83.5, 74.67, 76.0, 79.0, 81.45454545454545, 75.27272727272727, 1.1294642857142858, 2.5507246376811596, 2]
    ])

    response = output_fn(prediction, "text/csv")

    assert response == (prediction, "text/csv")


def test_output_json_returns_tensorflow_ready_input():
    prediction = np.array([
        [77, 77, 77, 77, 77, 77, 76, 75, 75, 79, 77, 80, 80, 80, 80, 80, 80, 78, 80, 86, 73, 88, 2.15, 2.24, 1.63, 1.67, 46, 65, 2.6942883775913384, 1.008610065655301, 76.73, 80.45, 42.34, 27.71,
         5.379, 7.424, 50.032, 45.678, 76.5, 80.83, 80.0, 77.0, 76.72727272727273, 80.45454545454545, 2.1244075829383884, 2.0757062146892653, 1],
        [79, 79, 79, 79, 79, 79, 81, 86, 88, 85, 82, 76, 76, 76, 76, 76, 76, 74, 75, 72, 76, 75, 1.52, 0.82, 0.92, 1.92, 68, 17, 4.144031675155845, 3.887158709814261, 81.45, 75.27, 50.07, 3.55, 7.592,
         4.772, 43.5, 52.919, 83.5, 74.67, 76.0, 79.0, 81.45454545454545, 75.27272727272727, 1.1294642857142858, 2.5507246376811596, 2]
    ])

    response = output_fn(prediction, "application/json")

    assert response[0] == {
        "instances": [
            [77, 77, 77, 77, 77, 77, 76, 75, 75, 79, 77, 80, 80, 80, 80, 80, 80, 78, 80, 86, 73, 88, 2.15, 2.24, 1.63, 1.67, 46, 65, 2.6942883775913384, 1.008610065655301, 76.73, 80.45, 42.34, 27.71,
             5.379, 7.424, 50.032, 45.678, 76.5, 80.83, 80.0, 77.0, 76.72727272727273, 80.45454545454545, 2.1244075829383884, 2.0757062146892653, 1],
            [79, 79, 79, 79, 79, 79, 81, 86, 88, 85, 82, 76, 76, 76, 76, 76, 76, 74, 75, 72, 76, 75, 1.52, 0.82, 0.92, 1.92, 68, 17, 4.144031675155845, 3.887158709814261, 81.45, 75.27, 50.07, 3.55,
             7.592,
             4.772, 43.5, 52.919, 83.5, 74.67, 76.0, 79.0, 81.45454545454545, 75.27272727272727, 1.1294642857142858, 2.5507246376811596, 2]
        ]
    }

    assert response[1] == "application/json"


def test_predict_transforms_data(directory):
    input_data = """
            77,77,77,,77,77,,75,75,79,77,80,80,80,80,80,80,78,80,86,73,,2.15,2.24,1.63,1.67,46,65,2.6942883775913384,,76.73,80.45,42.34,27.71,5.379,7.424,50.032,45.678,76.5,80.83,80.0,77.0,,80.45454545454545,2.1244075829383884,2.0757062146892653
            """

    model = model_fn(str(directory))
    df = input_fn(input_data, "text/csv")
    response = predict_fn(df, model)
    assert type(response) is np.ndarray


def test_predict_returns_none_if_invalid_input(directory):
    input_data = """
            RUTI,77,77,77,77,77,76,75,75,79,77,80,80,80,80,80,80,78,80,86,73,88,2.15,2.24,1.63,1.67,46,65,2.6942883775913384,1.008610065655301,76.73,80.45,42.34,27.71,5.379,7.424,50.032,45.678,76.5,80.83,80.0,77.0,76.72727272727273,80.45454545454545,2.1244075829383884,2.0757062146892653
            """

    model = model_fn(str(directory))
    df = input_fn(input_data, "text/csv")
    assert predict_fn(df, model) is None
