import os
import logging

LOG_LEVEL = logging.INFO

BASE_DIR = "./"

TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR = os.path.join(BASE_DIR, "test")
MODEL_DIR = os.path.join(BASE_DIR, "models")

X_TRAIN_PATH = os.path.join(TRAIN_DIR, "x_train.npy")
X_TEST_PATH = os.path.join(TEST_DIR, "x_test.npy")
X_TRAIN_SCALED_PATH = os.path.join(TRAIN_DIR, "x_train_scaled.npy")
X_TEST_SCALED_PATH = os.path.join(TEST_DIR, "x_test_scaled.npy")
Y_TRAIN_PATH = os.path.join(TRAIN_DIR, "y_train.npy")
Y_TEST_PATH = os.path.join(TEST_DIR, "y_test.npy")

MODEL_PATH = os.path.join(MODEL_DIR, "lin_reg.pkl")
