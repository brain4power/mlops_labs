import os
import logging
import argparse

import pandas as pd

# Project
from config import DATA_DIR, LOG_LEVEL

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)


class Pipeline:
    def __init__(self, *args, **kwargs):
        # load df from file
        source_file_name = kwargs.get("source_file_name")  # type: str
        if not source_file_name.endswith(".csv"):
            source_file_name = f"{source_file_name}.csv"
        df_path = os.path.join(DATA_DIR, source_file_name)
        if not os.path.isfile(df_path):
            raise FileNotFoundError(f"Can't find file at {df_path}")
        self.df = pd.read_csv(df_path)

        # load actions
        actions = kwargs.get("actions")
        for action in actions:
            if not isinstance(action, AbstractAction):
                raise TypeError(f"Action {action} must be instance of AbstractAction subclass")
        self._actions = actions

        # prepare result path
        rewrite_dataset = kwargs.get("rewrite_dataset")
        if rewrite_dataset:
            self.result_file_path = df_path
        else:
            result_name = kwargs.get("result_name")
            if not result_name:
                raise ValueError(
                    f"Params value error. "
                    f"Required name of result dataset cose --rewrite arg not provided. "
                    f"Use --result_name arg to specify it."
                )
            if not result_name.endswith(".csv"):
                result_name = f"{result_name}.csv"
            self.result_file_path = os.path.join(DATA_DIR, result_name)

    def _save_result(self):
        self.df.to_csv(self.result_file_path, index=False)

    def run(self):
        for action in self._actions:
            action.apply(self.df)
        self._save_result()


class AbstractAction:
    def __init__(self, *args, **kwargs):
        column_names = kwargs.get("column_names")
        if not column_names:
            raise ValueError(f"required valid column_names")
        self.column_names = column_names


class AgeToCategories(AbstractAction):
    __identity__ = "age_to_categories"

    def __init__(self, *args, **kwargs):
        super(AgeToCategories, self).__init__(*args, **kwargs)
        self.column_name = self.column_names[0]

    @staticmethod
    def _get_age_class_name_by_value(age: int) -> str:
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
        else:
            return "old"

    def apply(self, df: pd.DataFrame):
        df["age_class"] = df[self.column_name].apply(self._get_age_class_name_by_value)


class FillNAValues(AbstractAction):
    __identity__ = "fill_na_values"

    def __init__(self, *args, **kwargs):
        super(FillNAValues, self).__init__(*args, **kwargs)
        self.column_name = self.column_names[0]
        # parse fill_na action
        fill_na_action = kwargs.get("fill_na_action", "")
        if not fill_na_action:
            raise ValueError(f"required valid fill_na_action")

        fill_na_action_map = {
            "by_text": self._by_text,
            "mean": self._mean_action,
        }
        try:
            self._fill_action = fill_na_action_map[fill_na_action]
        except KeyError as e:
            raise ValueError(f"Invalid fill_na_action. Possible values: {fill_na_action_map.keys()}") from e
        # save kwargs for by_text fill_na_text extraction
        self.init_kwargs = kwargs

    def _mean_action(self, df, *args, **kwargs):
        return df[self.column_name].mean()

    def _by_text(self, *args, **kwargs):
        fill_na_text = self.init_kwargs.get("fill_na_text")
        if not fill_na_text:
            raise ValueError(f"required valid fill_na_text")
        return fill_na_text

    def apply(self, df: pd.DataFrame):
        df[self.column_name] = df[self.column_name].fillna(value=self._fill_action(df))


class OneHotEncoding(AbstractAction):
    __identity__ = "one_hot_encoding"

    def __init__(self, *args, **kwargs):
        super(OneHotEncoding, self).__init__(*args, **kwargs)
        column_names = kwargs.get("column_names")
        if not column_names:
            raise ValueError(f"required valid column_names")
        self.column_names = column_names

    def apply(self, df: pd.DataFrame):
        ohe_df = pd.get_dummies(df[self.column_names])

        result_df = pd.concat([df, ohe_df], axis=1)
        df[result_df.columns] = result_df
        for column_name in ohe_df.columns:
            df[column_name] = df[column_name].replace({True: 1, False: 0})
        df.drop(labels=self.column_names, axis=1, inplace=True)


action_map = {action.__identity__: action for action in (AgeToCategories, FillNAValues, OneHotEncoding)}

if __name__ == "__main__":
    # parse args
    parser = argparse.ArgumentParser(description="Process some params.")
    parser.add_argument(
        "--source_file_name",
        action="store",
        dest="source_file_name",
        help="Source dataset file name. Will search at config.DATA_DIR",
        type=str,
    )
    parser.add_argument(
        "--actions",
        action="store",
        dest="actions",
        help="Transform actions",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--column_names",
        action="store",
        dest="column_names",
        help="Column names",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--fill_na_action",
        action="store",
        dest="fill_na_action",
        help="Action for filling nan",
        type=str,
    )
    parser.add_argument(
        "--fill_na_text",
        action="store",
        dest="fill_na_text",
        help="Text for filling nan",
        type=str,
    )
    parser.add_argument(
        "--rewrite",
        action="store_true",
        dest="rewrite_dataset",
        help="Rewrite dataset or create a copy",
    )
    parser.add_argument(
        "--result_name",
        action="store",
        dest="result_name",
        help="Result dataset name, if need to create copy after transformations",
        type=str,
    )
    parser_args = parser.parse_args()
    actions_names = parser_args.actions.split(",")
    action_kwargs = {
        "column_names": parser_args.column_names.split(","),
        "fill_na_action": parser_args.fill_na_action,
        "fill_na_text": parser_args.fill_na_text,
    }
    logger.info(f"action_kwargs: {action_kwargs}")
    pipeline_actions = list()
    for actions_name in actions_names:
        try:
            pipeline_actions.append(action_map[actions_name](**action_kwargs))
        except KeyError as e:
            raise ValueError(f"Invalid action identity: {actions_name}") from e
    pipeline = Pipeline(
        source_file_name=parser_args.source_file_name,
        actions=pipeline_actions,
        rewrite_dataset=parser_args.rewrite_dataset,
        result_name=parser_args.result_name,
    )
    pipeline.run()
