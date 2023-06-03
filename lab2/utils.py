import os
import logging

import numpy as np


def read_ml_data(path) -> tuple:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Can't find file at {path}")

    data = np.load(path)

    if data.size == 0:
        return np.ndarray(), np.ndarray()

    x = data[:, :-1]
    y = data[:, -1]
    return x, y


def prepare_logger(level) -> logging.Logger:
    logging.basicConfig()
    logger = logging.getLogger(__name__)
    logger.setLevel(level)
    return logger
