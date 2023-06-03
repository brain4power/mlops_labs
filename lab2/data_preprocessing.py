import os
from datetime import datetime

import numpy as np
import pandas as pd
from utils import prepare_logger
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Project
from config import TEST_DIR, LOG_LEVEL, TEST_PATH, TRAIN_DIR, TRAIN_PATH, DATA_FILE_NAME

logger = prepare_logger(LOG_LEVEL, "data_preprocessing")


def separate_target_data(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    y = df[column_name]
    df.drop(columns=[column_name], inplace=True)
    return y


def to_numpy(df: pd.DataFrame) -> np.ndarray:
    y = df.values
    y = np.expand_dims(y, axis=1)
    return y


def split_features(df: pd.DataFrame) -> tuple:
    column_names = df.columns.to_list()
    cat_columns = []
    num_columns = []

    for column_name in column_names:
        if (df[column_name].dtypes == "int64") or (df[column_name].dtypes == "float64"):
            num_columns += [column_name]
        else:
            cat_columns += [column_name]
    return num_columns, cat_columns


def save_data(features, targets, path) -> None:
    data = np.concatenate((features, targets), axis=1)
    np.save(path, data)


def preprocess_data() -> None:
    x_train = pd.read_csv(TRAIN_PATH)
    x_test = pd.read_csv(TEST_PATH)

    y_train = to_numpy(separate_target_data(x_train, "price"))
    y_test = to_numpy(separate_target_data(x_test, "price"))

    num_cols, cat_cols = split_features(x_train)

    preprocessors = make_column_transformer(
        (StandardScaler(), num_cols), (OneHotEncoder(drop="if_binary", handle_unknown="ignore"), cat_cols)
    )

    x_train = preprocessors.fit_transform(x_train)
    x_test = preprocessors.transform(x_test)

    save_data(x_train, y_train, os.path.join(TRAIN_DIR, DATA_FILE_NAME))
    save_data(x_test, y_test, os.path.join(TEST_DIR, DATA_FILE_NAME))


if __name__ == "__main__":
    start_time = datetime.now()
    logger.info(f"Data preprocessing started at {start_time}")
    preprocess_data()
