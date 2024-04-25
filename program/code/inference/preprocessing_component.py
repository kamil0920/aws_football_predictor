import os
import numpy as np
import pandas as pd
import json
import joblib

from io import StringIO

try:
    from sagemaker_containers.beta.framework import encoders, worker
except ImportError:
    # We don't have access to the `worker` instance when testing locally.
    # We'll set it to None so we can change the way functions create a response.
    worker = None

TARGET_COLUMN = "result_match"

FEATURE_COLUMNS = ['player_rating_home_player_1', 'player_rating_home_player_2', 'player_rating_home_player_3', 'player_rating_home_player_4', 'player_rating_home_player_5',
                   'player_rating_home_player_6', 'player_rating_home_player_7', 'player_rating_home_player_8', 'player_rating_home_player_9', 'player_rating_home_player_10',
                   'player_rating_home_player_11', 'player_rating_away_player_1', 'player_rating_away_player_2', 'player_rating_away_player_3', 'player_rating_away_player_4',
                   'player_rating_away_player_5', 'player_rating_away_player_6', 'player_rating_away_player_7', 'player_rating_away_player_8', 'player_rating_away_player_9',
                   'player_rating_away_player_10', 'player_rating_away_player_11', 'ewm_home_team_goals', 'ewm_away_team_goals', 'ewm_home_team_goals_conceded', 'ewm_away_team_goals_conceded',
                   'points_home', 'points_away', 'home_weighted_wins', 'away_weighted_wins', 'avg_home_team_rating', 'avg_away_team_rating', 'home_streak_wins', 'away_streak_wins', 'ewm_shoton_home',
                   'ewm_shoton_away', 'ewm_possession_home', 'ewm_possession_away', 'avg_home_rating_attack', 'avg_away_rating_attack', 'avg_away_rating_defence', 'avg_home_rating_defence',
                   'average_rating_home', 'average_rating_away', 'num_top_players_home', 'num_top_players_away', 'ewm_home_team_goals_conceded_x_ewm_shoton_home', 'attacking_strength_home',
                   'attacking_strength_away', 'attacking_strength_diff']


def input_fn(input_data, content_type):
    """
    Parses the input payload and creates a Pandas DataFrame.

    This function will check whether the target column is present in the
    input data, and will remove it.
    """

    if content_type == "text/csv":
        df = pd.read_csv(StringIO(input_data), header=None, skipinitialspace=True)

        if len(df.columns) == len(FEATURE_COLUMNS) + 1:
            df = df.drop(df.columns[0], axis=1)

        df.columns = FEATURE_COLUMNS
        return df

    if content_type == "application/json":
        df = pd.DataFrame([json.loads(input_data)])

        if "result_match" in df.columns:
            df = df.drop("result_match", axis=1)

        return df

    else:
        raise ValueError(f"{content_type} is not supported.!")


def output_fn(prediction, accept):
    """
    Ensures output is in CSV format for XGBoost compatibility.
    """
    if prediction is None:
        raise Exception("There was an error transforming the input data")

    if isinstance(prediction, np.ndarray):
        buffer = StringIO()
        np.savetxt(buffer, prediction, delimiter=",", fmt='%s')
        csv_payload = buffer.getvalue()
        return worker.Response(csv_payload, mimetype='text/csv') if worker else csv_payload

    raise Exception(f"Unsupported accept type: {accept}")

def predict_fn(input_data, model):
    """
    Preprocess the input using the transformer.
    """

    try:
        response = model.transform(input_data)
        return response
    except ValueError as e:
        print("Error transforming the input data", e)
        return None


def model_fn(model_dir):
    """
    Deserializes the model that will be used in this container.
    """

    return joblib.load(os.path.join(model_dir, "features.joblib"))
