import datetime
import logging
import numpy as np
import os

def read_ml_data(path)->tuple:
    if not os.path.isfile(path):
       raise FileNotFoundError(path)

    with open(path, "rb") as f:
        data = np.load(path)

    if data.size== 0:
       return np.ndarray(), np.ndarray()

    x = data[:,:-1]
    y = data[:,-1]
    return x, y

def prepare_logger(level)->logging.Logger:
    logging.basicConfig()
    logger = logging.getLogger(__name__)
    logger.setLevel(level)
    return logger

