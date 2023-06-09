import os
import pickle
from datetime import datetime

from utils import read_ml_data, prepare_logger
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error as mse

# Project
from config import TEST_DIR, LOG_LEVEL, MODEL_PATH, DATA_FILE_NAME

logger = prepare_logger(LOG_LEVEL, "model_testing")


def test_model() -> dict:
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)

    x_test, y_test = read_ml_data(os.path.join(TEST_DIR, DATA_FILE_NAME))
    y_pred = model.predict(x_test)

    result = {
        "MSE": mse(y_test, y_pred),
        "RMSE": mse(y_test, y_pred, squared=False),
        "R2": r2_score(y_test, y_pred),
    }
    return result


if __name__ == "__main__":
    start_time = datetime.now()
    logger.info(f"Model testing started at {start_time}")
    test_results = test_model()
    logger.info(
        f"""
    Test results:
    MSE: {test_results['MSE']:1f}
    RMSE: {test_results['RMSE']:1f}
    R2: {test_results['R2']:4f}
    """
    )
