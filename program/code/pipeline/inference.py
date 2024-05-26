import os
import json
from io import StringIO

import requests
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import xgboost as xgb

try:
    from sagemaker_containers.beta.framework import encoders, worker
except ImportError:
    # We don't have access to the `worker` instance when testing locally.
    # We'll set it to None so we can change the way functions create
    # a response.
    worker = None

FEATURE_COLUMNS = ['player_rating_home_player_1', 'player_rating_home_player_2', 'player_rating_home_player_3',
                   'player_rating_home_player_4', 'player_rating_home_player_5',
                   'player_rating_home_player_6', 'player_rating_home_player_7', 'player_rating_home_player_8',
                   'player_rating_home_player_9', 'player_rating_home_player_10',
                   'player_rating_home_player_11', 'player_rating_away_player_1', 'player_rating_away_player_2',
                   'player_rating_away_player_3', 'player_rating_away_player_4',
                   'player_rating_away_player_5', 'player_rating_away_player_6', 'player_rating_away_player_7',
                   'player_rating_away_player_8', 'player_rating_away_player_9',
                   'player_rating_away_player_10', 'player_rating_away_player_11', 'ewm_home_team_goals',
                   'ewm_away_team_goals', 'ewm_home_team_goals_conceded', 'ewm_away_team_goals_conceded',
                   'points_home', 'points_away', 'home_weighted_wins', 'away_weighted_wins', 'avg_home_team_rating',
                   'avg_away_team_rating', 'home_streak_wins', 'away_streak_wins', 'ewm_shoton_home',
                   'ewm_shoton_away', 'ewm_possession_home', 'ewm_possession_away', 'avg_home_rating_attack',
                   'avg_away_rating_attack', 'avg_away_rating_defence', 'avg_home_rating_defence',
                   'average_rating_home', 'average_rating_away', 'num_top_players_home', 'num_top_players_away',
                   'ewm_home_team_goals_conceded_x_ewm_shoton_home', 'attacking_strength_home',
                   'attacking_strength_away', 'attacking_strength_diff']


def model_fn(model_dir):
    model_file = os.path.join(model_dir, "saved_model.xgb")
    model = xgb.XGBClassifier()
    model.load_model(model_file)
    return model


def parse_confidence(lst, func):
    return [(x[0], func(x[1])) for x in lst]


def input_fn(request_body, request_content_type):
    df = None

    if request_content_type == "text/csv":
        df = pd.read_csv(StringIO(request_body), header=None, skipinitialspace=True)

        if len(df.columns) == len(FEATURE_COLUMNS) + 1:
            df = df.drop(df.columns[-1], axis=1)

        df.columns = FEATURE_COLUMNS

        return df

    if request_content_type == "application/json":
        df = pd.DataFrame(json.loads(request_body))

        if "result_match" in df.columns:
            df = df.drop("result_match", axis=1)

        return df

    features_model_path = Path("/opt/ml/model")
    features_pipeline = joblib.load(features_model_path / "features.joblib")
    transformed_data = features_pipeline.transform(df)

    return transformed_data


def predict_fn(input_data, model):
    predictions = model.predict_proba(input_data)
    return predictions


def output_fn(prediction, response_content_type):
    if response_content_type == "text/csv":
        prediction = parse_confidence(prediction, lambda x: x.item())
        return (
            worker.Response(encoders.encode(prediction, response_content_type), mimetype=response_content_type)
            if worker
            else (prediction, response_content_type)
        )

    if response_content_type == "application/json":
        response = []
        for p, c in prediction:
            response.append({"prediction": p, "confidence": c.item()})

        if len(response) == 1:
            response = response[0]

        return (
            worker.Response(json.dumps(response), mimetype=response_content_type)
            if worker
            else (response, response_content_type)
        )

    raise Exception(f"{response_content_type} accept type is not supported.")
