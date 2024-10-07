import argparse
import os
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.sparse import csr_matrix
from sklearn.metrics import f1_score


def get_hook_config(train_dmatrix, validation_dmatrix, hyperparameters):
    is_test = os.getenv('TEST')
    if is_test is None:
        from smdebug.xgboost import Hook

        hook = Hook.create_from_json_file()
        hook.train_data = train_dmatrix
        hook.validation_data = validation_dmatrix
        hook.hyperparameters = hyperparameters
        print(f'hook: {hook}')
        return [hook]
    else:
        return None


def evaluate_model(X_train, X_val, y_train, y_val, early_stopping_rounds, hyperparameters):
    X_train_sparse = csr_matrix(X_train)
    X_val_sparse = csr_matrix(X_val)

    y_train = np.ravel(y_train)
    y_val = np.ravel(y_val)

    dtrain = xgb.DMatrix(X_train_sparse, label=y_train)
    dval = xgb.DMatrix(X_val_sparse, label=y_val)

    hook = get_hook_config(dtrain, dval, hyperparameters)

    model = xgb.train(
        params=hyperparameters,
        dtrain=dtrain,
        num_boost_round=120,
        evals=[(dtrain, 'train'), (dval, 'eval')],
        early_stopping_rounds=early_stopping_rounds,
        callbacks=hook
    )

    y_pred = model.predict(dval)
    y_pred = (y_pred > 0.5).astype(int)

    f1 = f1_score(y_val, y_pred)
    return f1, model


def train(model_directory, train_path, validation_path, hyperparameters, pipeline_path, early_stopping_rounds=25):
    train_files = [file for file in Path(train_path).glob("*.csv")]
    validation_files = [file for file in Path(validation_path).glob("*.csv")]

    if len(train_files) == 0 or len(validation_files) == 0:
        raise ValueError("The are no train or validation files")

    train_data = [pd.read_csv(file) for file in train_files]
    X_train = pd.concat(train_data)
    y_train = X_train.pop('result_match')

    validation_data = [pd.read_csv(file) for file in validation_files]
    X_validation = pd.concat(validation_data)
    y_validation = X_validation.pop('result_match')

    f1, model = evaluate_model(X_train, X_validation, y_train, y_validation, early_stopping_rounds, hyperparameters)

    print("F1 score: {:.2f}".format(f1))

    model_directory = Path(model_directory)
    model_directory.mkdir(parents=True, exist_ok=True)

    model_filepath = model_directory / "saved_model.xgb"
    model.save_model(str(model_filepath))
    print('Saving model to {}'.format(str(model_filepath)))

    print(f'pipeline path: {pipeline_path}')

    with tarfile.open(Path(pipeline_path) / "model.tar.gz", "r:gz") as tar:
        tar.extractall(model_directory)


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
    parser.add_argument("--scale_pos_weight", type=float, default=2.0)
    parser.add_argument("--objective", type=str, default='binary:logistic')

    args, _ = parser.parse_known_args()

    params = {
        "eta": args.eta,
        "max_depth": args.max_depth,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "min_child_weight": args.min_child_weight,
        "reg_lambda": args.lambda_,
        "reg_alpha": args.alpha,
        "scale_pos_weight": args.scale_pos_weight,
        "objective": args.objective,
    }

    print(f'xgb version: {xgb.__version__}')

    train(
        # This is the location where we need to save our model. SageMaker will
        # create a model.tar.gz file with anything inside this directory when
        # the training script finishes.
        model_directory=os.environ["SM_MODEL_DIR"],

        # SageMaker creates one channel for each one of the inputs to the
        # Training Step.
        train_path=os.environ["SM_CHANNEL_TRAIN"],
        validation_path=os.environ["SM_CHANNEL_VALIDATION"],
        pipeline_path=os.environ["SM_CHANNEL_PIPELINE"],
        early_stopping_rounds=args.early_stopping_rounds,
        hyperparameters=params
    )
