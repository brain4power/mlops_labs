import os
import pickle
import logging
from datetime import datetime

import numpy as np
from sklearn.linear_model import LinearRegression

# Project
from config import LOG_LEVEL, MODEL_DIR, MODEL_PATH, Y_TRAIN_PATH, X_TRAIN_SCALED_PATH

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)


def prepare_model() -> None:
    x_train = np.load(X_TRAIN_SCALED_PATH)
    y_train = np.load(Y_TRAIN_PATH)

    model = LinearRegression(fit_intercept=True, n_jobs=-1)
    model.fit(x_train, y_train)

    os.makedirs(MODEL_DIR, exist_ok=True)

    with open(MODEL_PATH, "wb") as file:
        pickle.dump(model, file)


if __name__ == "__main__":
    start_time = datetime.now()
    logger.info(f"Model preparation started at {start_time}")
    prepare_model()
