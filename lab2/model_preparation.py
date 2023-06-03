import os
import pickle
from datetime import datetime

from utils import read_ml_data, prepare_logger
from xgboost import XGBRegressor

# Project
from config import TEST_DIR, LOG_LEVEL, MODEL_DIR, MODEL_PATH, DATA_FILE_NAME

logger = prepare_logger(LOG_LEVEL, "model_preparation")


def prepare_model() -> None:
    x_train, y_train = read_ml_data(os.path.join(TEST_DIR, DATA_FILE_NAME))

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
