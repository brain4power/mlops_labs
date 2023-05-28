import logging
import os
import pickle
from datetime import datetime

import pandas as pd
from xgboost import XGBRegressor

from config import (LOG_LEVEL, MODEL_DIR, MODEL_PATH, X_TRAIN_SCALED_PATH,
                    Y_TRAIN_PATH)

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)


def prepare_model() -> None:
    x_train = pd.read_csv(X_TRAIN_SCALED_PATH)
    y_train = pd.read_csv(Y_TRAIN_PATH)

    model = XGBRegressor(
        max_depth=8,
        learning_rate=0.02,
        n_estimators=990,
        min_child_weight=4,
        gamma=0.95,
        subsample=0.95,
        colsample_bytree=0.97,
        reg_alpha=0.74,
        reg_lambda=0.78,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)

    os.makedirs(MODEL_DIR, exist_ok=True)

    with open(MODEL_PATH, "wb") as file:
        pickle.dump(model, file)


if __name__ == "__main__":
    start_time = datetime.now()
    logger.info(f"Model preparation started at {start_time}")
    prepare_model()
