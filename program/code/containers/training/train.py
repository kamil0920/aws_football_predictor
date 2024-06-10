import json
import os
import argparse
import tarfile

import numpy as np
import pandas as pd
from comet_ml import Experiment

from pathlib import Path
from sklearn.metrics import f1_score, log_loss

from xgboost import XGBClassifier


def evaluate_model(X_train, X_val, y_train, y_val, early_stopping_rounds, hyperparameters):
    model = XGBClassifier(**hyperparameters, random_state=42, scale_pos_weight=2, early_stopping_rounds=early_stopping_rounds, eval_metric='logloss')
    model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], verbose=1)

    y_pred = model.predict(X_val)
    f1 = f1_score(y_val, y_pred)

    return f1, model, y_pred


def train(model_directory, train_path, validation_path, hyperparameters, pipeline_path, experiment, early_stopping_rounds=25):
    train_files = [file for file in Path(train_path).glob("*.csv")]
    validation_files = [file for file in Path(validation_path).glob("*.csv")]

    if len(train_files) == 0 or len(validation_files) == 0:
        raise ValueError("The are no train or validation files")

    train_data = [pd.read_csv(file, header=None) for file in train_files]
    X_train = pd.concat(train_data)
    y_train = X_train[X_train.columns[-1]]
    X_train.drop(X_train.columns[-1], axis=1, inplace=True)

    validation_data = [pd.read_csv(file, header=None) for file in validation_files]
    X_validation = pd.concat(validation_data)
    y_validation = X_validation[X_validation.columns[-1]]
    X_validation.drop(X_validation.columns[-1], axis=1, inplace=True)

    f1, model, predictions = evaluate_model(X_train, X_validation, y_train, y_validation, early_stopping_rounds, hyperparameters)

    print("F1 score: {:.2f}".format(f1))

    model_directory = Path(model_directory)
    model_directory.mkdir(parents=True, exist_ok=True)

    # Save the model in JSON format
    model_filepath = model_directory / "saved_model.xgb"
    model.save_model(str(model_filepath))
    print('Saving model to {}'.format(str(model_filepath)))

    with tarfile.open(Path(pipeline_path) / "model.tar.gz", "r:gz") as tar:
        tar.extractall(model_directory)

    if experiment:
        experiment.log_parameters(
            {
                "eta": hyperparameters['eta'],
                "max_depth": hyperparameters['max_depth'],
                "subsample": hyperparameters['subsample'],
                "colsample_bytree": hyperparameters['colsample_bytree'],
                "min_child_weight": hyperparameters['min_child_weight'],
                "reg_lambda": hyperparameters['reg_lambda'],
                "reg_alpha": hyperparameters['reg_alpha'],
            }
        )
        experiment.log_dataset_hash(X_train)
        experiment.log_confusion_matrix(
            y_validation.astype(int), predictions.astype(int)
        )
        experiment.log_model("football", model_filepath.as_posix())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--early_stopping_rounds", type=int, default=25)

    parser.add_argument("--eta", type=float, default=0.10552574516231328)
    parser.add_argument("--max_depth", type=int, default=9)
    parser.add_argument("--subsample", type=float, default=0.8953781772368965)
    parser.add_argument("--colsample_bytree", type=float, default=0.8448004658642061)
    parser.add_argument("--lambda_", type=float, default=9.082892225273682)
    parser.add_argument("--alpha", type=float, default=5.065713059538013)
    parser.add_argument("--min_child_weight", type=float, default=0.49876571387552704)

    args, _ = parser.parse_known_args()

    params = {
        "eta": args.eta,
        "max_depth": args.max_depth,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "min_child_weight": args.min_child_weight,
        "reg_lambda": args.lambda_,
        "reg_alpha": args.alpha,
    }

    comet_api_key = os.environ.get("COMET_API_KEY", None)
    comet_project_name = os.environ.get("COMET_PROJECT_NAME", None)

    experiment = (
        Experiment(
            project_name=comet_project_name,
            api_key=comet_api_key,
            auto_metric_logging=True,
            auto_param_logging=True,
            log_code=True,
        )
        if comet_api_key and comet_project_name
        else None
    )

    training_env = json.loads(os.environ.get("SM_TRAINING_ENV", {}))
    job_name = training_env.get("job_name", None) if training_env else None

    if job_name and experiment:
        experiment.set_name(job_name)

    train(
        model_directory=os.environ["SM_MODEL_DIR"],

        train_path=os.environ["SM_CHANNEL_TRAIN"],
        validation_path=os.environ["SM_CHANNEL_VALIDATION"],
        pipeline_path=os.environ["SM_CHANNEL_PIPELINE"],
        experiment=experiment,
        early_stopping_rounds=args.early_stopping_rounds,
        hyperparameters=params
    )
