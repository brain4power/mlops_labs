import logging
import os
from datetime import datetime

import pandas as pd
from sklearn.model_selection import train_test_split

from config import (DATA_PATH, LOG_LEVEL, TEST_DIR, TRAIN_DIR, X_TEST_PATH,
                    X_TRAIN_PATH, Y_TEST_PATH, Y_TRAIN_PATH)

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)


def create_dataset() -> None:
    df = pd.read_csv(DATA_PATH)

    x = df.drop(columns=["price"])
    y = df["price"]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=26
    )

    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)

    x_train.to_csv(X_TRAIN_PATH, index=False)
    y_train.to_csv(Y_TRAIN_PATH, index=False)
    x_test.to_csv(X_TEST_PATH, index=False)
    y_test.to_csv(Y_TEST_PATH, index=False)


if __name__ == "__main__":
    start_time = datetime.now()
    logger.info(f"Dataset creation started at {start_time}")
    create_dataset()
