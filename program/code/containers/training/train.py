import json
import os
import argparse
import tarfile

import numpy as np
import pandas as pd
from comet_ml import Experiment

from pathlib import Path
from sklearn.metrics import f1_score, log_loss, confusion_matrix, roc_auc_score

from xgboost import XGBClassifier, cv, DMatrix
import lightgbm as lgb


class ModelEvaluator:
    @staticmethod
    def evaluate_model_lgb(X, y, hyperparameters, n_splits=3):
        feature_names = X.columns.tolist()
        lgb_data = lgb.Dataset(data=X, label=y, feature_name=feature_names)

        cv_results = lgb.cv(params=hyperparameters,
                            train_set=lgb_data,
                            num_boost_round=10000,
                            nfold=n_splits,
                            metrics=['auc', 'logloss'],
                            stratified=True,
                            shuffle=True,
                            seed=123,
                            return_cvbooster=True)

        best_iteration = np.argmax(cv_results['valid auc-mean'])
        mean_auc = cv_results['valid auc-mean'][best_iteration]
        std_auc = np.std(cv_results['valid auc-stdv'])
        best_iteration += 1

        print(f"LGB Mean Test AUC: {mean_auc:.2f} (±{std_auc :.2f})")
        print(f"LGB Average Best Iteration: {best_iteration:.2f}")

        final_model = lgb.LGBMClassifier(**hyperparameters, random_state=42, n_estimators=best_iteration)
        final_model.fit(X, y)

        return mean_auc, std_auc, final_model

    @staticmethod
    def evaluate_model_xgb(X, y, early_stopping_rounds, hyperparameters, n_splits=3):
        feature_names = X.columns.tolist()
        data_dmatrix = DMatrix(data=X, label=y, feature_names=feature_names)

        xgb_cv = cv(dtrain=data_dmatrix, params=hyperparameters, nfold=n_splits,
                    num_boost_round=10000, early_stopping_rounds=early_stopping_rounds,
                    metrics=['logloss', 'auc'], as_pandas=True, stratified=True, seed=123)

        best_iteration = xgb_cv['test-auc-mean'].idxmax()
        mean_auc = xgb_cv['test-auc-mean'][best_iteration]
        mean_logloss = xgb_cv['test-logloss-mean'][best_iteration]

        print(f"XGB Mean Test AUC: {mean_auc:.2f} (±{np.std(xgb_cv['test-auc-mean']):.2f})")
        print(f"XGB Mean Test Log Loss: {mean_logloss:.2f} (±{np.std(xgb_cv['test-logloss-mean']):.2f})")
        print(f"XGB Average Best Iteration: {best_iteration + 1:.2f}")

        final_model = XGBClassifier(**hyperparameters, random_state=42, n_estimators=best_iteration + 1)
        final_model.fit(X, y)

        return mean_auc, np.std(xgb_cv['test-auc-mean']), final_model


def train(model_directory, train_path, hyperparameters, pipeline_path, experiment, early_stopping_rounds=25):
    X_train = pd.read_csv(Path(train_path) / 'train.csv')
    y_train = X_train.pop('result_match')

    mean_auc_xgb, std_auc_xgb, model_xgb = ModelEvaluator.evaluate_model_xgb(X_train, y_train, early_stopping_rounds, hyperparameters, 4)
    mean_auc_lgb, std_auc_lgb, model_lgb = ModelEvaluator.evaluate_model_lgb(X_train, y_train, hyperparameters, 4)

    if mean_auc_lgb > mean_auc_xgb:
        best_model = model_lgb
        best_model_name = "saved_model_lgb.txt"
        print("LightGBM model is better.")
    else:
        best_model = model_xgb
        best_model_name = "saved_model_xgb.xgb"
        print("XGBoost model is better.")

    model_directory = Path(model_directory)
    model_directory.mkdir(parents=True, exist_ok=True)

    model_filepath = model_directory / best_model_name
    if best_model_name.endswith("xgb.xgb"):
        best_model.save_model(str(model_filepath))
    else:
        best_model.booster_.save_model(str(model_filepath))

    with tarfile.open(Path(pipeline_path) / "model.tar.gz", "r:gz") as tar:
        tar.extractall(model_directory)

    process_experiment(X_train, experiment, hyperparameters, best_model, model_filepath, y_train)
    print('Training finished.')


def process_experiment(X_train, experiment, hyperparameters, model, model_filepath, y_train):
    if experiment:
        experiment.log_parameters({
            "eta": hyperparameters['eta'],
            "max_depth": hyperparameters['max_depth'],
            "subsample": hyperparameters['subsample'],
            "colsample_bytree": hyperparameters['colsample_bytree'],
            "min_child_weight": hyperparameters['min_child_weight'],
            "reg_lambda": hyperparameters['reg_lambda'],
            "reg_alpha": hyperparameters['reg_alpha'],
            "objective": hyperparameters['objective'],
        })
        experiment.log_dataset_hash(X_train)

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

    parser.add_argument("--eta", type=float, default=0.105)
    parser.add_argument("--max_depth", type=int, default=9)
    parser.add_argument("--subsample", type=float, default=0.895)
    parser.add_argument("--colsample_bytree", type=float, default=0.844)
    parser.add_argument("--lambda_", type=float, default=9.082)
    parser.add_argument("--alpha", type=float, default=5.065)
    parser.add_argument("--min_child_weight", type=float, default=0.498)
    parser.add_argument("--scale_pos_weight", type=float, default=2.0)
    parser.add_argument("--objective", type=str, default='binary:logistic')

    args, _ = parser.parse_known_args()

    hyperparameters = {
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

    training_env = json.loads(os.environ.get("SM_TRAINING_ENV", "{}"))
    job_name = training_env.get("job_name", None) if training_env else None

    if job_name and experiment:
        experiment.set_name(job_name)

    if 'train' in job_name:
        hyperparameters = {}

    print(f'job_name: {job_name}')
    print(f'hyperparameters: {hyperparameters}')

    train(
        model_directory=os.environ["SM_MODEL_DIR"],
        train_path=os.environ["SM_CHANNEL_TRAIN"],
        pipeline_path=os.environ["SM_CHANNEL_PIPELINE"],
        experiment=experiment,
        early_stopping_rounds=args.early_stopping_rounds,
        hyperparameters=hyperparameters
    )
