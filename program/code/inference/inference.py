import json
import os
from io import StringIO
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
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


def input_fn(request_body, request_content_type):
    df = None

    if request_content_type == "text/csv":
        df = pd.read_csv(StringIO(request_body), skipinitialspace=True, header=None)

        ground_truth_labels = None
        if len(df.columns) == len(FEATURE_COLUMNS) + 1:
            ground_truth_labels = df.iloc[:, 0].to_numpy(copy=True)
            df = df.drop(df.columns[-1], axis=1)

        df.columns = FEATURE_COLUMNS

    elif request_content_type == "application/json":
        df = pd.json_normalize(json.loads(request_body))

        ground_truth_labels = None
        if "result_match" in df.columns:
            ground_truth_labels = df["result_match"].copy()
            df = df.drop("result_match", axis=1)

    converted_df = _convert_columns_to_numeric(df)
    features_pipeline = _get_feature_pipeline()
    transformed_data = features_pipeline.transform(converted_df)

    return {"features": transformed_data, "ground_truth": ground_truth_labels}


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
    features = input_data["features"]
    ground_truth = input_data["ground_truth"]

    preds = model.predict_proba(features)

    return {"predictions": preds, "ground_truth": ground_truth}


def output_fn(prediction, response_content_type):
    classes = _get_classes()
    predictions = prediction["predictions"]
    ground_truth_labels = prediction["ground_truth"]

    prediction_data = _prepare_prediction_data(predictions, classes, ground_truth_labels)

    if response_content_type == "text/csv":
        print("Processing CSV response")
        result_rows = _prepare_csv_response(prediction_data)
        result = worker.Response(encoders.encode(result_rows, response_content_type), mimetype=response_content_type) if worker else (result_rows, response_content_type)
        return result

    elif response_content_type == "application/json":
        print("Processing JSON response")
        result = _prepare_json_response(prediction_data)
        return worker.Response(result, mimetype=response_content_type) if worker else (result, response_content_type)

    elif response_content_type == "application/jsonlines":
        print("Processing JSON Lines response")
        result = _prepare_jsonlines_response(prediction_data)
        return worker.Response(result, mimetype=response_content_type) if worker else (result, response_content_type)

    raise Exception(f"{response_content_type} accept type is not supported.")


def _prepare_prediction_data(predictions, classes, ground_truth_labels):
    prediction_data = []

    has_ground_truth = ground_truth_labels is not None

    if has_ground_truth:
        for x, gt in zip(predictions, ground_truth_labels):
            predicted_label = classes[np.argmax(x)]
            probability = float(x[1])
            numerical_label = 1 if predicted_label == 'home_win' else 0
            numerical_ground_truth_label = 1 if gt == 'home_win' else 0

            prediction_data.append({
                "ground_truth": gt,
                "numerical_ground_truth_label": numerical_ground_truth_label,
                "predicted_label": predicted_label,
                "predicted_numerical_label": numerical_label,
                "probability": probability
            })
    else:
        for x in predictions:
            predicted_label = classes[np.argmax(x)]
            probability = float(x[1])
            numerical_label = 1 if predicted_label == 'home_win' else 0

            prediction_data.append({
                "predicted_label": predicted_label,
                "predicted_numerical_label": numerical_label,
                "probability": probability
            })

    return prediction_data


def _prepare_csv_response(prediction_data):
    if not prediction_data:
        return []

    has_ground_truth = "ground_truth" in prediction_data[0]

    if not has_ground_truth:
        # header = ('predicted_label', 'predicted_numerical_label', 'probability')
        rows = [
            (pd["predicted_label"], pd["predicted_numerical_label"], pd["probability"])
            for pd in prediction_data
        ]
    else:
        # header = ('ground_truth_labels', 'numerical_ground_truth_labels',
        #           'predicted_label', 'predicted_numerical_label', 'probability')
        rows = [
            (pd["ground_truth"], pd["numerical_ground_truth_label"], pd["predicted_label"], pd["predicted_numerical_label"], pd["probability"])
            for pd in prediction_data
        ]

    return rows


def _prepare_json_response(prediction_data):
    json_list = []
    for pd in prediction_data:
        item = {
            'prediction': pd["predicted_label"],
            'confidence': pd["probability"],
            'numerical_label': pd["predicted_numerical_label"]
        }
        if "ground_truth" in pd:
            item['ground_truth'] = pd["ground_truth"]
        json_list.append(item)

    return json.dumps(json_list)


def _prepare_jsonlines_response(prediction_data):
    lines = []
    for pd in prediction_data:
        line_dict = {
            'prediction': pd["predicted_label"],
            'confidence': pd["probability"],
            'numerical_label': pd["predicted_numerical_label"]
        }
        if "ground_truth" in pd:
            line_dict['ground_truth'] = pd["ground_truth"]
        lines.append(json.dumps(line_dict))
    return "\n".join(lines)


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


def _convert_columns_to_numeric(df):
    for column in df.columns:
        df[column] = pd.to_numeric(df[column], errors='coerce')
    return df
