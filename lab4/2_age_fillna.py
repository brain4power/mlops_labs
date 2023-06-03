import logging
from datetime import datetime

import titanic as tt

# Project
from config import LOG_LEVEL

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)

start_time = datetime.now()
logger.info(f"Dataset readed at {start_time}")
train_df = tt.read_data()

start_time = datetime.now()
logger.info(f"Age column filled NaN by mean at {start_time}")
train_age_df = tt.age_fillna(train_df)

start_time = datetime.now()
logger.info(f"age_class column added at {start_time}")
train_age_df = tt.add_age_classes(train_age_df)

start_time = datetime.now()
logger.info(f"New train_age.csv dataset written at {start_time}")
tt.export_csv(train_age_df, "train_age")
