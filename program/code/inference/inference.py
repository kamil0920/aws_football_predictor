import os
import json
from io import StringIO

import requests
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

import sklearn
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
    model_file = os.path.join(Path(model_dir), "saved_model.xgb")

    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Model file not found: {model_file}")

    model = xgb.XGBClassifier()
    print("XGBClassifier initialized.")

    try:
        model.load_model(model_file)
        print("Model loaded successfully.")
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")

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

    if request_content_type == "application/json":
        df = pd.json_normalize(json.loads(request_body))

        if "result_match" in df.columns:
            df = df.drop("result_match", axis=1)

    features_pipeline = _get_feature_pipeline()

    try:
        transformed_data = features_pipeline.transform(df)
        print("Data transformation successful.")
    except Exception as e:
        raise RuntimeError(f"Failed to transform data: {e}")

    return transformed_data


def _get_feature_pipeline():
    model_path = os.getenv("MODEL_PATH", "/opt/ml/model")
    features_pipeline_path = Path(model_path) / "features.joblib"
    try:
        features_pipeline = joblib.load(features_pipeline_path)
        print("Features pipeline loaded successfully.")
    except Exception as e:
        raise RuntimeError(f"Failed to load features pipeline: {e}")
    return features_pipeline


def predict_fn(input_data, model):
    print(f"Input data type: {type(input_data)}")
    if hasattr(input_data, 'shape'):
        print(f"Input data shape: {input_data.shape}")

    try:
        predictions = model.predict_proba(input_data)
    except Exception as e:
        raise RuntimeError(f"Failed to make predictions: {e}")

    print(f"Predictions:: {predictions}")

    return predictions


def output_fn(prediction, response_content_type):
    classes = _get_classes()

    if response_content_type == "text/csv":
        print("Processing CSV response")
        predictions = [(classes[np.argmax(x)], x[np.argmax(x)]) for x in prediction]

        result = worker.Response(encoders.encode(predictions, response_content_type), mimetype=response_content_type) if worker else (predictions, response_content_type)
        return result

    if response_content_type == "application/json":
        print("Processing JSON response")
        predictions = [{'prediction': classes[np.argmax(x)], 'confidence': float(x[np.argmax(x)])} for x in prediction]
        result = json.dumps(predictions)

        return (
            worker.Response(result, mimetype=response_content_type)
            if worker
            else (result, response_content_type)
        )

    raise Exception(f"{response_content_type} accept type is not supported.")


def _get_classes():
    model_path = os.getenv("MODEL_PATH", "/opt/ml/model")
    target_pipeline_path = Path(model_path) / "target.joblib"
    if not target_pipeline_path.exists():
        raise FileNotFoundError(f"Target pipeline file not found: {target_pipeline_path}")
    try:
        target_pipeline = joblib.load(target_pipeline_path)
        print("Target pipeline loaded successfully.")
    except Exception as e:
        raise RuntimeError(f"Failed to load target pipeline: {e}")
    classes = target_pipeline.named_transformers_["result_match"].categories_[0]
    return classes