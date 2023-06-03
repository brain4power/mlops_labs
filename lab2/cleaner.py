import shutil
import logging
import contextlib
from datetime import datetime

# Project
from config import TEST_DIR, LOG_LEVEL, MODEL_DIR, TRAIN_DIR

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)


def clean() -> None:
    for path_to_clean in (TRAIN_DIR, TEST_DIR, MODEL_DIR):
        with contextlib.suppress(Exception):
            shutil.rmtree(path_to_clean)


if __name__ == "__main__":
    start_time = datetime.now()
    logger.info(f"Cleaning started at {start_time}")
    clean()
    finish_time = datetime.now()
    logger.info(f"Cleaning finished at {finish_time}")
