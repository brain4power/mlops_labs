import logging
import os

LOG_LEVEL = logging.INFO

DATA_PATH = (
    "https://raw.githubusercontent.com/tidyverse/ggplot2/main/data-raw/diamonds.csv"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR = os.path.join(BASE_DIR, "test")
MODEL_DIR = os.path.join(BASE_DIR, "models")

X_TRAIN_PATH = os.path.join(TRAIN_DIR, "x_train.csv")
X_TEST_PATH = os.path.join(TEST_DIR, "x_test.csv")
X_TRAIN_SCALED_PATH = os.path.join(TRAIN_DIR, "x_train_scaled.csv")
X_TEST_SCALED_PATH = os.path.join(TEST_DIR, "x_test_scaled.csv")
Y_TRAIN_PATH = os.path.join(TRAIN_DIR, "y_train.csv")
Y_TEST_PATH = os.path.join(TEST_DIR, "y_test.csv")

MODEL_PATH = os.path.join(MODEL_DIR, "xgb.pkl")
