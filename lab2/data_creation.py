import os
from datetime import datetime

import pandas as pd
from sklearn.model_selection import train_test_split

from config import (DATA_URI, LOG_LEVEL, TEST_DIR, TRAIN_DIR, TEST_PATH,
                    TRAIN_PATH, TEST_SIZE, RANDOM_STATE)

from utils import prepare_logger

logger = prepare_logger(LOG_LEVEL)

def create_dataset(test_size=TEST_SIZE, random_state=RANDOM_STATE) -> None:
    df = pd.read_csv(DATA_URI)

    data_train, data_test = train_test_split(
        df, test_size=test_size, random_state=random_state)

    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)

    data_train.to_csv(TRAIN_PATH, index=False)
    data_test.to_csv(TEST_PATH, index=False)

if __name__ == "__main__":
    start_time = datetime.now()
    logger.info(f"Dataset creation started at {start_time}")
    create_dataset()
