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
        89.0,79.0,59.0,69.0,69.0,79.0,69.0,79.0,69.0,89.0,89.0,79.0,79.0,79.0,79.0,79.0,79.0,80.0,80.0,71.0,83.0,80.0,2.54,0.61,0.26,4.76,28.0,15.0,1.37715,2.55612,84.18,80.91,11.75,14.58,3.55,1.805,53.639,20.03,71.33,78.83,79.0,71.0,71.181818,78.909091,0.0,0.0,4.473,29.233607,31.406375,-2.172768
        """

    df = input_fn(input_data, "text/csv")
    assert len(df.columns) == 50 and "result_match" not in df.columns


def test_input_json_drops_target_column_if_present():
    input_data = json.dumps(
        {"player_rating_home_player_1": 89, "player_rating_home_player_2": 79, "player_rating_home_player_3": 59, "player_rating_home_player_4": 69, "player_rating_home_player_5": 69,
         "player_rating_home_player_6": 79, "player_rating_home_player_7": 69, "player_rating_home_player_8": 79, "player_rating_home_player_9": 69, "player_rating_home_player_10": 89,
         "player_rating_home_player_11": 89, "player_rating_away_player_1": 79, "player_rating_away_player_2": 79, "player_rating_away_player_3": 79, "player_rating_away_player_4": 79,
         "player_rating_away_player_5": 79, "player_rating_away_player_6": 79, "player_rating_away_player_7": 80, "player_rating_away_player_8": 80, "player_rating_away_player_9": 71,
         "player_rating_away_player_10": 83, "player_rating_away_player_11": 80, "ewm_home_team_goals": 2.54, "ewm_away_team_goals": 0.61, "ewm_home_team_goals_conceded": 0.26,
         "ewm_away_team_goals_conceded": 4.76, "points_home": 28, "points_away": 15, "home_weighted_wins": 1.377149515625, "away_weighted_wins": 2.5561203576634663, "avg_home_team_rating": 84.18,
         "avg_away_team_rating": 80.91, "home_streak_wins": 11.75, "away_streak_wins": 14.58, "ewm_shoton_home": 3.55, "ewm_shoton_away": 1.805, "ewm_possession_home": 53.639,
         "ewm_possession_away": 20.03, "avg_home_rating_attack": 71.33, "avg_away_rating_attack": 78.83, "avg_away_rating_defence": 79.0, "avg_home_rating_defence": 71.0,
         "average_rating_home": 71.18181818181819, "average_rating_away": 78.9090909090909, "num_top_players_home": 0, "num_top_players_away": 0,
         "ewm_home_team_goals_conceded_x_ewm_shoton_home": 4.473, "attacking_strength_home": 29.233606557377048, "attacking_strength_away": 31.40637450199203,
         "attacking_strength_diff": -2.172767944614982
         })

    df = input_fn(input_data, "application/json")
    assert len(df.columns) == 50 and "result_match" not in df.columns


def test_input_csv_works_without_target_column():
    input_data = """
        89.0,79.0,59.0,69.0,69.0,79.0,69.0,79.0,69.0,89.0,89.0,79.0,79.0,79.0,79.0,79.0,79.0,80.0,80.0,71.0,83.0,80.0,2.54,0.61,0.26,4.76,28.0,15.0,1.37715,2.55612,84.18,80.91,11.75,14.58,3.55,1.805,53.639,20.03,71.33,78.83,79.0,71.0,71.181818,78.909091,0.0,0.0,4.473,29.233607,31.406375,-2.172768
        """

    df = input_fn(input_data, "text/csv")
    assert len(df.columns) == 50


def test_input_json_works_without_target_column():
    input_data = json.dumps(
        {"player_rating_home_player_1": 89, "player_rating_home_player_2": 79, "player_rating_home_player_3": 59, "player_rating_home_player_4": 69, "player_rating_home_player_5": 69,
         "player_rating_home_player_6": 79, "player_rating_home_player_7": 69, "player_rating_home_player_8": 79, "player_rating_home_player_9": 69, "player_rating_home_player_10": 89,
         "player_rating_home_player_11": 89, "player_rating_away_player_1": 79, "player_rating_away_player_2": 79, "player_rating_away_player_3": 79, "player_rating_away_player_4": 79,
         "player_rating_away_player_5": 79, "player_rating_away_player_6": 79, "player_rating_away_player_7": 80, "player_rating_away_player_8": 80, "player_rating_away_player_9": 71,
         "player_rating_away_player_10": 83, "player_rating_away_player_11": 80, "ewm_home_team_goals": 2.54, "ewm_away_team_goals": 0.61, "ewm_home_team_goals_conceded": 0.26,
         "ewm_away_team_goals_conceded": 4.76, "points_home": 28, "points_away": 15, "home_weighted_wins": 1.377149515625, "away_weighted_wins": 2.5561203576634663, "avg_home_team_rating": 84.18,
         "avg_away_team_rating": 80.91, "home_streak_wins": 11.75, "away_streak_wins": 14.58, "ewm_shoton_home": 3.55, "ewm_shoton_away": 1.805, "ewm_possession_home": 53.639,
         "ewm_possession_away": 20.03, "avg_home_rating_attack": 71.33, "avg_away_rating_attack": 78.83, "avg_away_rating_defence": 79.0, "avg_home_rating_defence": 71.0,
         "average_rating_home": 71.18181818181819, "average_rating_away": 78.9090909090909, "num_top_players_home": 0, "num_top_players_away": 0,
         "ewm_home_team_goals_conceded_x_ewm_shoton_home": 4.473, "attacking_strength_home": 29.233606557377048, "attacking_strength_away": 31.40637450199203,
         "attacking_strength_diff": -2.172767944614982
         })

    df = input_fn(input_data, "application/json")



    assert len(df.columns) == 50


def test_output_csv_raises_exception_if_prediction_is_none():
    with pytest.raises(Exception):
        output_fn(None, "text/csv")


def test_output_json_raises_exception_if_prediction_is_none():
    with pytest.raises(Exception):
        output_fn(None, "application/json")


def test_output_returns_xgboost_ready_input():
    prediction = np.array([
        [71, 71, 71, 71, 71, 71, 72, 70, 70, 71, 74, 79, 79, 79, 79, 79, 79, 80, 80, 71, 83, 80, 1.54, 1.61, 1.26, 0.76, 20, 28, 1.377149515625, 2.5561203576634663, 71.18, 78.91, 11.75, 14.58, 3.55,
         1.805, 53.639, 20.03, 71.33, 78.83, 79.0, 71.0, 71.18181818181819, 78.9090909090909, 0, 0, 4.473, 29.233606557377048, 31.40637450199203, -2.172767944614982, 0],
        [71, 90, 71, 71, 86, 71, 72, 92, 82, 71, 74, 79, 79, 79, 79, 79, 79, 80, 80, 71, 83, 80, 1.54, 1.61, 1.26, 0.76, 20, 28, 2.377149515625, 4.5561203576634663, 71.18, 78.91, 11.75, 23.58, 3.55,
         1.805, 45.639, 20.03, 71.33, 78.83, 79.0, 99.0, 71.18181818181819, 78.9090909090909, 0, 0, 4.473, 35.233606557377048, 25.40637450199203, -2.172767944614982, 1]
    ])

    response = output_fn(prediction, "text/csv")

    assert response == ('71.0,71.0,71.0,71.0,71.0,71.0,72.0,70.0,70.0,71.0,74.0,79.0,79.0,79.0,79.0,79.0,79.0,80.0,80.0,71.0,83.0,80.0,1.54,1.61,1.26,0.76,20.0,28.0,1.377149515625,'
                        '2.5561203576634663,71.18,78.91,11.75,14.58,3.55,1.805,53.639,20.03,71.33,78.83,79.0,71.0,71.18181818181819,78.9090909090909,0.0,0.0,4.473,'
                        '29.233606557377048,31.40637450199203,-2.172767944614982,0.0\n71.0,90.0,71.0,71.0,86.0,71.0,72.0,92.0,82.0,71.0,74.0,79.0,79.0,79.0,79.0,79.0,79.0,80.0,'
                        '80.0,71.0,83.0,80.0,1.54,1.61,1.26,0.76,20.0,28.0,2.377149515625,4.556120357663466,71.18,78.91,11.75,23.58,3.55,1.805,45.639,20.03,71.33,78.83,79.0,'
                        '99.0,71.18181818181819,78.9090909090909,0.0,0.0,4.473,35.23360655737705,25.40637450199203,-2.172767944614982,1.0\n')


def test_predict_transforms_data(directory):
    input_data = """
            71, 71, 71, 71, 71, 71, 72, 70, 70, 71, 74, 79, 79, 79, 79, 79, 79, 80, 80, 71, 83, 80, 1.54, 1.61, 1.26, 0.76, 20, 28, 1.377149515625, 2.5561203576634663, 71.18, 78.91, 11.75, 14.58, 3.55, 1.805, 53.639, 20.03, 71.33, 78.83, 79.0, 71.0, 71.18181818181819, 78.9090909090909, 0, 0, 4.473, 29.233606557377048, 31.40637450199203, -2.172767944614982
            """

    model = model_fn(str(directory))
    df = input_fn(input_data, "text/csv")
    response = predict_fn(df, model)
    assert type(response) is np.ndarray


def test_predict_returns_none_if_invalid_input(directory):
    input_data = """
            RUTI, 71, 71, 71, 71, 71, 72, 70, 70, 71, 74, 79, 79, 79, 79, 79, 79, 80, 80, 71, 83, 80, 1.54, 1.61, 1.26, 0.76, 20, 28, 1.377149515625, 2.5561203576634663, 71.18, 78.91, 11.75, 14.58, 3.55, 1.805, 53.639, 20.03, 71.33, 78.83, 79.0, 71.0, 71.18181818181819, 78.9090909090909, 0, 0, 4.473, 29.233606557377048, 31.40637450199203, -2.172767944614982
            """

    model = model_fn(str(directory))
    df = input_fn(input_data, "text/csv")
    assert predict_fn(df, model) is None
