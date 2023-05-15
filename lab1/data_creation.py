import os
import logging
from datetime import datetime

import numpy as np
from sklearn.model_selection import train_test_split

# Project
from config import (
    TEST_DIR,
    LOG_LEVEL,
    TRAIN_DIR,
    X_TEST_PATH,
    Y_TEST_PATH,
    X_TRAIN_PATH,
    Y_TRAIN_PATH,
)

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)


def true_fun(
    input_array: np.array, multiplier: float = np.pi, addend: float = 0.0, math_function: np.ufunc = np.sin
) -> np.array:
    multiplier_array = np.atleast_1d(multiplier)
    new_array = np.sum([ai * np.power(input_array, i + 1) for i, ai in enumerate(multiplier_array)], axis=0)
    return math_function(new_array + addend)


def noises(shape: tuple, noise_power: float) -> np.array:
    return np.random.randn(*shape) * noise_power


def dataset(
    multiplier: float,
    addend: float,
    math_function: np.ufunc = np.sin,
    n_points: int = 250,
    max_value: float = 1.0,
    noise_power: float = 0.0,
    is_random: bool = True,
    seed: int = 1234,
) -> np.array:
    np.random.seed(seed)
    if is_random:
        x_array = np.sort(np.random.rand(n_points)) * max_value
    else:
        x_array = np.linspace(0, max_value, n_points)
    y_array = true_fun(x_array, multiplier, addend, math_function)
    y_array = y_array.reshape(-1, n_points).T
    new_y_array = y_array + noises(y_array.shape, noise_power)
    return np.atleast_2d(x_array).T, new_y_array


def create_dataset() -> None:
    (
        x,
        y,
    ) = dataset(
        multiplier=-200, addend=200, math_function=np.abs, n_points=500, max_value=111.2, noise_power=0.1, seed=555
    )

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)

    np.save(X_TRAIN_PATH, x_train)
    np.save(Y_TRAIN_PATH, y_train)
    np.save(X_TEST_PATH, x_test)
    np.save(Y_TEST_PATH, y_test)


if __name__ == "__main__":
    start_time = datetime.now()
    logger.info(f"Dataset creation started at {start_time}")
    create_dataset()
