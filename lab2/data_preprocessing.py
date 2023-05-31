import logging
from datetime import datetime

import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import (LOG_LEVEL, X_TEST_PATH, X_TEST_SCALED_PATH, X_TRAIN_PATH,
                    X_TRAIN_SCALED_PATH)

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)


def split_features(df) -> tuple:
    column_names = df.columns.to_list()
    cat_columns = []
    num_columns = []

    for column_name in column_names:
        if (df[column_name].dtypes == "int64") or (df[column_name].dtypes == "float64"):
            num_columns += [column_name]
        else:
            cat_columns += [column_name]
    return num_columns, cat_columns


def preprocess_data() -> None:
    x_train = pd.read_csv(X_TRAIN_PATH)
    x_test = pd.read_csv(X_TEST_PATH)

    num_cols, cat_cols = split_features(x_train)

    preprocessors = make_column_transformer(
        (StandardScaler(), num_cols),
        (OneHotEncoder(drop="if_binary", handle_unknown="ignore"), cat_cols)
    )
   
    x_train_scaled = preprocessors.fit_transform(x_train)
    x_test_scaled = preprocessors.transform(x_test)
    
    pd.DataFrame(x_train_scaled).to_csv(X_TRAIN_SCALED_PATH, index=False)
    pd.DataFrame(x_test_scaled).to_csv(X_TEST_SCALED_PATH, index=False


if __name__ == "__main__":
    start_time = datetime.now()
    logger.info(f"Data preprocessing started at {start_time}")
    preprocess_data()
