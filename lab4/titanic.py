import logging
from datetime import datetime

import pandas as pd

# Project
from config import DATA_DIR, DATA_PATH, LOG_LEVEL

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)


def read_data() -> pd.DataFrame:
    train_df = pd.read_csv(DATA_PATH)
    return train_df


def export_csv(df: pd.DataFrame, csv_name: str) -> None:
    df.to_csv(f"{DATA_DIR}/{csv_name}.csv", index=False)


def cabin_fillna(df: pd.DataFrame) -> pd.DataFrame:
    df["Cabin"] = df["Cabin"].fillna(value="Not_indicated")
    return df


def age_fillna(df: pd.DataFrame) -> pd.DataFrame:
    mean_age = df["Age"].mean()
    df["Age"] = df["Age"].fillna(value=mean_age)
    return df


def _age_classes(age: float) -> str:
    if age <= 10:
        return "child"
    elif age <= 18:
        return "teenager"
    elif age < 30:
        return "young adult"
    elif age < 50:
        return "middle-aged"
    elif age < 70:
        return "senior"
    elif age >= 70:
        return "old"


def add_age_classes(df: pd.DataFrame) -> pd.DataFrame:
    df["age_class"] = df["Age"].apply(_age_classes)
    return df


def ohe(df: pd.DataFrame) -> pd.DataFrame:
    cat_columns = ["Sex", "Embarked", "age_class"]
    ohe_df = pd.get_dummies(df[cat_columns])
    result_df = pd.concat([df, ohe_df], axis=1)
    result_df = result_df.drop(labels=cat_columns, axis=1)
    return result_df


if __name__ == "__main__":
    # Read dataset
    start_time = datetime.now()
    logger.info(f"Dataset readed at {start_time}")
    train_df = read_data()

    # Cabin
    start_time = datetime.now()
    logger.info(f"Cabin column changed at {start_time}")
    train_df = cabin_fillna(train_df)

    start_time = datetime.now()
    logger.info(f"Changed dataset rewritten at {start_time}")
    export_csv(train_df, "train")

    # Age
    start_time = datetime.now()
    logger.info(f"Age column filled NaN by mean at {start_time}")
    train_age_df = age_fillna(train_df)

    start_time = datetime.now()
    logger.info(f"age_class column added at {start_time}")
    train_age_df = add_age_classes(train_age_df)

    start_time = datetime.now()
    logger.info(f"New train_age.csv dataset written at {start_time}")
    export_csv(train_age_df, "train_age")

    # OHE
    start_time = datetime.now()
    logger.info(f"OHE applied to cat_columns at {start_time}")
    train_age_ohe_df = ohe(train_age_df)

    start_time = datetime.now()
    logger.info(f"New train_age_ohe.csv dataset written at {start_time}")
    export_csv(train_age_ohe_df, "train_age_ohe")
