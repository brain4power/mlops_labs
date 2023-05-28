import logging
from datetime import datetime

import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import (LOG_LEVEL, X_TEST_PATH, X_TEST_SCALED_PATH, X_TRAIN_PATH,
                    X_TRAIN_SCALED_PATH)

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)


def split_features(df) -> dict:
    column_names = df.columns.to_list()
    cat_columns = []
    num_columns = []

    for column_name in column_names:
        if (df[column_name].dtypes == "int64") or (df[column_name].dtypes == "float64"):
            num_columns += [column_name]
        else:
            cat_columns += [column_name]
    return {"num_columns": num_columns, "cat_columns": cat_columns}


def preprocess_data() -> None:
    x_train = pd.read_csv(X_TRAIN_PATH)
    x_test = pd.read_csv(X_TEST_PATH)

    train_num_columns = split_features(x_train)["num_columns"]
    train_cat_columns = split_features(x_train)["cat_columns"]
    test_num_columns = split_features(x_test)["num_columns"]
    test_cat_columns = split_features(x_test)["cat_columns"]

    sc = StandardScaler()
    x_train_sc = sc.fit_transform(x_train[train_num_columns])
    x_test_sc = sc.transform(x_test[test_num_columns])

    ohe = OneHotEncoder(drop="if_binary", handle_unknown="ignore", sparse_output=False)
    x_train_ohe = ohe.fit_transform(x_train[train_cat_columns])
    x_test_ohe = ohe.transform(x_test[test_cat_columns])

    x_train_scaled = pd.concat(
        [pd.DataFrame(x_train_sc), pd.DataFrame(x_train_ohe)], axis=1
    )
    x_test_scaled = pd.concat(
        [pd.DataFrame(x_test_sc), pd.DataFrame(x_test_ohe)], axis=1
    )

    x_train_scaled.to_csv(X_TRAIN_SCALED_PATH, index=False)
    x_test_scaled.to_csv(X_TEST_SCALED_PATH, index=False)


if __name__ == "__main__":
    start_time = datetime.now()
    logger.info(f"Data preprocessing started at {start_time}")
    preprocess_data()
