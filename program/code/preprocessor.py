import os
import tarfile
import tempfile
import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder


def preprocess(base_directory):
    """
    This function loads the supplied data, splits it and transforms it.
    """

    df = _read_data_from_input_csv_files(base_directory)

    numeric_transformer = make_pipeline(
        SimpleImputer(strategy="median"),
    )

    features_transformer = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, make_column_selector(dtype_exclude="object")),
        ]
    )

    df_train, df_validation, df_test = _split_data(df)

    _save_baselines(base_directory, df_train, df_test)

    y_train = np.array(df_train.result_match.values).reshape(-1, 1)
    y_validation = np.array(df_validation.result_match.values).reshape(-1, 1)
    y_test = np.array(df_test.result_match.values).reshape(-1, 1)

    df_train = df_train.drop("result_match", axis=1)
    df_validation = df_validation.drop("result_match", axis=1)
    df_test = df_test.drop("result_match", axis=1)

    X_train = features_transformer.fit_transform(df_train)
    X_validation = features_transformer.transform(df_validation)
    X_test = features_transformer.transform(df_test)

    _save_splits(base_directory, X_train, y_train, X_validation, y_validation, X_test, y_test)
    _save_model(base_directory, features_transformer)


def _read_data_from_input_csv_files(base_directory):
    """
    This function reads every CSV file available and concatenates
    them into a single dataframe.
    """

    input_directory = Path(base_directory) / "input"
    files = [file for file in input_directory.glob("*.csv")]

    if len(files) == 0:
        raise ValueError(f"The are no CSV files in {str(input_directory)}/")

    raw_data = [pd.read_csv(file) for file in files]
    df = pd.concat(raw_data, axis=1)

    # Shuffle the data
    return df.sample(frac=1, random_state=42)


def _split_data(df):
    """
    Splits the data into three sets: train, validation and test.
    """

    df_train, temp = train_test_split(df, test_size=0.2, random_state=42)
    df_validation, df_test = train_test_split(temp, test_size=0.5)

    return df_train, df_validation, df_test


def _save_baselines(base_directory, df_train, df_test):
    """
    During the data and quality monitoring steps, we will need baselines
    to compute constraints and statistics. This function saves the
    untransformed data to disk so we can use them as baselines later.
    """

    for split, data in [("train", df_train), ("test", df_test)]:
        baseline_path = Path(base_directory) / f"{split}-baseline"
        baseline_path.mkdir(parents=True, exist_ok=True)

        df = data.copy().dropna()

        # We want to save the header only for the train baseline
        # but not for the test baseline. We'll use the test baseline
        # to generate predictions later, and we can't have a header line
        # because the model won't be able to make a prediction for it.
        header = split == "train"
        df.to_csv(baseline_path / f"{split}-baseline.csv", header=header, index=False)


def _save_splits(base_directory, X_train, y_train, X_validation, y_validation, X_test, y_test):
    """
    This function concatenates the transformed features and the target variable, and
    saves each one of the split sets to disk.
    """

    train = np.concatenate((X_train, y_train), axis=1)
    validation = np.concatenate((X_validation, y_validation), axis=1)
    test = np.concatenate((X_test, y_test), axis=1)

    train_path = Path(base_directory) / "train"
    validation_path = Path(base_directory) / "validation"
    test_path = Path(base_directory) / "test"

    train_path.mkdir(parents=True, exist_ok=True)
    validation_path.mkdir(parents=True, exist_ok=True)
    test_path.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(train).to_csv(train_path / "train.csv", header=False, index=False)
    pd.DataFrame(validation).to_csv(validation_path / "validation.csv", header=False, index=False)
    pd.DataFrame(test).to_csv(test_path / "test.csv", header=False, index=False)


def _save_model(base_directory, features_transformer):
    """
    This function creates a model.tar.gz file that contains the two transformation
    pipelines we built to transform the data.
    """

    with tempfile.TemporaryDirectory() as directory:
        joblib.dump(features_transformer, os.path.join(directory, "features.joblib"))

        model_path = Path(base_directory) / "model"
        model_path.mkdir(parents=True, exist_ok=True)

        with tarfile.open(f"{str(model_path / 'model.tar.gz')}", "w:gz") as tar:
            tar.add(os.path.join(directory, "features.joblib"), arcname="features.joblib")


if __name__ == "__main__":
    preprocess(base_directory="/opt/ml/processing")