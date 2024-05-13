import os
import argparse

import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.metrics import f1_score, log_loss

from xgboost import XGBClassifier


def evaluate_model(X_train, X_val, y_train, y_val, early_stopping_rounds, hyperparameters):
    model = XGBClassifier(**hyperparameters, random_state=42, scale_pos_weight=2, early_stopping_rounds=early_stopping_rounds, eval_metric='logloss')
    model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], verbose=1)

    y_pred = model.predict(X_val)
    f1 = f1_score(y_val, y_pred)

    return f1, model


def train(model_directory, train_path, validation_path, hyperparameters, early_stopping_rounds=25):
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

    f1, model = evaluate_model(X_train, X_validation, y_train, y_validation, early_stopping_rounds, hyperparameters)

    print("F1 score: {:.2f}".format(f1))

    model_directory = Path(model_directory)
    model_directory.mkdir(parents=True, exist_ok=True)

    # Save the model in JSON format
    model_filepath = model_directory / "saved_model.json"
    model.save_model(str(model_filepath))


if __name__ == "__main__":
    # Any hyperparameters provided by the training job are passed to
    # the entry point as script arguments.
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

    train(
        # This is the location where we need to save our model. SageMaker will
        # create a model.tar.gz file with anything inside this directory when
        # the training script finishes.
        model_directory=os.environ["SM_MODEL_DIR"],

        # SageMaker creates one channel for each one of the inputs to the
        # Training Step.
        train_path=os.environ["SM_CHANNEL_TRAIN"],
        validation_path=os.environ["SM_CHANNEL_VALIDATION"],
        early_stopping_rounds=args.early_stopping_rounds,
        hyperparameters=params

    )
