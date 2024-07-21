import json
import os
import argparse
import tarfile

import numpy as np
import pandas as pd
from comet_ml import Experiment

from pathlib import Path
from sklearn.metrics import f1_score, log_loss, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedKFold

from xgboost import XGBClassifier
from xgboost import cv
from xgboost import DMatrix


# def evaluate_model(X_train, X_val, y_train, y_val, early_stopping_rounds, hyperparameters):
#     model = XGBClassifier(**hyperparameters, random_state=42, early_stopping_rounds=early_stopping_rounds, eval_metric=['logloss', 'auc'], objective='binary:logistic')
#     model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], verbose=1)
#
#     y_pred = model.predict(X_val)
#     f1 = f1_score(y_val, y_pred)
#
#     return f1, model, y_pred

def evaluate_model(X, y, early_stopping_rounds, hyperparameters, n_splits=3):
    feature_names = X.columns.tolist()

    auc_scores = []
    logloss_scores = []
    best_iterations = []

    data_dmatrix = DMatrix(data=X, label=y, feature_names=feature_names)

    xgb_cv = cv(dtrain=data_dmatrix, params=hyperparameters, nfold=n_splits,
                num_boost_round=50, early_stopping_rounds=early_stopping_rounds,
                metrics=['logloss', 'auc'], as_pandas=True, stratified=True, seed=123)

    test_auc_mean = xgb_cv['test-auc-mean'].max()
    best_iteration = xgb_cv['test-auc-mean'].idxmax()

    auc_scores.append(test_auc_mean)
    logloss_scores.append(xgb_cv['test-logloss-mean'].min())
    best_iterations.append(best_iteration)

    mean_auc = np.mean(auc_scores)
    std_auc = np.std(auc_scores)
    mean_logloss = np.mean(logloss_scores)
    std_logloss = np.std(logloss_scores)
    best_iteration = np.mean(best_iterations)

    print(f"Mean Test AUC: {mean_auc:.2f} (±{std_auc:.2f})")
    print(f"Mean Test Log Loss: {mean_logloss:.2f} (±{std_logloss:.2f})")
    print(f"Average Best Iteration: {best_iteration:.2f}")

    final_model = XGBClassifier(**hyperparameters, random_state=42, n_estimators=int(best_iteration))
    final_model.fit(X, y)

    return mean_auc, std_auc, mean_logloss, std_logloss, final_model


def train(model_directory, train_path, hyperparameters, pipeline_path, experiment, early_stopping_rounds=25):
    train_files = [file for file in Path(train_path).glob("*.csv")]

    train_data = [pd.read_csv(file) for file in train_files]
    X_train = pd.concat(train_data)
    y_train = X_train['result_match']
    X_train.drop(labels=['result_match'], axis=1, inplace=True)

    mean_auc, std_auc, mean_logloss, std_logloss, model = evaluate_model(X_train, y_train, early_stopping_rounds, hyperparameters, 4)

    model_directory = Path(model_directory)
    model_directory.mkdir(parents=True, exist_ok=True)

    model_filepath = model_directory / "saved_model.xgb"
    model.save_model(str(model_filepath))
    print('Saving model to {}'.format(str(model_filepath)))

    with tarfile.open(Path(pipeline_path) / "model.tar.gz", "r:gz") as tar:
        tar.extractall(model_directory)

    process_experiment(X_train, experiment, hyperparameters, model, model_filepath, y_train)


def process_experiment(X_train, experiment, hyperparameters, model, model_filepath, y_train):
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
                "objective": hyperparameters['objective'],
            }
        )
        experiment.log_dataset_hash(X_train)

        # Use the model's predictions to calculate the confusion matrix and F1 score
        y_pred = model.predict(X_train)
        f1 = f1_score(y_train.astype(int), y_pred.astype(int))
        conf_matrix = confusion_matrix(y_train.astype(int), y_pred.astype(int))

        experiment.log_confusion_matrix(matrix=conf_matrix, labels=["home_not_win", "home_win"])
        experiment.log_model("football", model_filepath.as_posix())
        experiment.log_metrics({'f1_score': f1})
        experiment.log_dataframe_profile(X_train, name="football_train")


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
        pipeline_path=os.environ["SM_CHANNEL_PIPELINE"],
        experiment=experiment,
        early_stopping_rounds=args.early_stopping_rounds,
        hyperparameters=params
    )
