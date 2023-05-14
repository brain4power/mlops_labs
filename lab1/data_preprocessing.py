import logging
from datetime import datetime

import numpy as np
from sklearn.preprocessing import StandardScaler

# Project
from config import (
    LOG_LEVEL,
    X_TEST_PATH,
    X_TRAIN_PATH,
    X_TEST_SCALED_PATH,
    X_TRAIN_SCALED_PATH,
)

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)


def preprocess_data() -> None:
    x_train = np.load(X_TRAIN_PATH)
    x_test = np.load(X_TEST_PATH)

    sc = StandardScaler()
    x_train_scaled = sc.fit_transform(x_train)
    x_test_scaled = sc.transform(x_test)

    np.save(X_TRAIN_SCALED_PATH, x_train_scaled)
    np.save(X_TEST_SCALED_PATH, x_test_scaled)


if __name__ == "__main__":
    start_time = datetime.now()
    logger.info(f"Data preprocessing started at {start_time}")
    preprocess_data()
