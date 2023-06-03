import os
import logging

LOG_LEVEL = logging.INFO

DATA_URI = "https://raw.githubusercontent.com/tidyverse/ggplot2/main/data-raw/diamonds.csv"

TEST_SIZE = 0.3
RANDOM_STATE = 26

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR = os.path.join(BASE_DIR, "test")
MODEL_DIR = os.path.join(BASE_DIR, "models")

TRAIN_PATH = os.path.join(TRAIN_DIR, "train.csv")
TEST_PATH = os.path.join(TEST_DIR, "test.csv")

DATA_FILE_NAME = "data.npy"

MODEL_PATH = os.path.join(MODEL_DIR, "xgb.pkl")
