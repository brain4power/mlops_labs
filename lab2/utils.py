import os
import logging

import numpy as np


def read_ml_data(path) -> tuple:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Can't find file at {path}")

    data = np.load(path)

    if data.size == 0:
        raise ValueError(f"Data loaded from {path} has 0 size.")

    x = data[:, :-1]
    y = data[:, -1]
    return x, y


def prepare_logger(level, name) -> logging.Logger:
    logging.basicConfig()
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger
