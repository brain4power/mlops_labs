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
logger.info(f"Cabin column changed at {start_time}")
train_df = tt.cabin_fillna(train_df)

start_time = datetime.now()
logger.info(f"Changed dataset rewritten at {start_time}")
tt.export_csv(train_df, "train")
