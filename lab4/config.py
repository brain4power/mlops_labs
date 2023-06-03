import os
import logging

LOG_LEVEL = logging.INFO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
DATA_PATH = os.path.join(DATA_DIR, "train.csv")

TRAIN_AGE_PATH = os.path.join(DATA_DIR, "train_age.csv")
TRAIN_AGE_OHE_PATH = os.path.join(DATA_DIR, "train_age_ohe.csv")
