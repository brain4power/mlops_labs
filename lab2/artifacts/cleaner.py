import logging
import shutil
from datetime import datetime

from config import LOG_LEVEL, MODEL_DIR, TEST_DIR, TRAIN_DIR

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)


def clean() -> None:
    shutil.rmtree(TRAIN_DIR)
    shutil.rmtree(TEST_DIR)
    shutil.rmtree(MODEL_DIR)


if __name__ == "__main__":
    start_time = datetime.now()
    logger.info(f"Cleaning started at {start_time}")
    clean()
    finish_time = datetime.now()
    logger.info(f"Cleaning finished at {finish_time}")
