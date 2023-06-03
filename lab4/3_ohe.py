import logging
from datetime import datetime

import pandas as pd
import titanic as tt

# Project
from config import LOG_LEVEL, TRAIN_AGE_PATH

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)

start_time = datetime.now()
logger.info(f"train_age.csv dataset readed at {start_time}")
train_age_df = pd.read_csv(TRAIN_AGE_PATH)

start_time = datetime.now()
logger.info(f"OHE applied to cat_columns at {start_time}")
train_age_ohe_df = tt.ohe(train_age_df)

start_time = datetime.now()
logger.info(f"New train_age_ohe.csv dataset written at {start_time}")
tt.export_csv(train_age_ohe_df, "train_age_ohe")
