"""Data loaders."""

import os
from typing import Optional

import pandas as pd
from datasets import ClassLabel, Dataset, DatasetDict

from tools.constants import (
    TEST,
    TF_ANSWERS,
    TF_DESCRIPTION,
    TF_DIFFICULTY,
    TF_LABEL,
    TF_QUESTION_ID,
    TF_TEXT,
    TRAIN,
    VALIDATION,
)
from tools.metrics import get_difficulty_labels
from tools.utils import set_seed


class QDET:
    """Class to load QDET datasets."""

    def __init__(
        self,
        name: str,
        num_classes: int,
        regression: bool,
        small_dev: bool,
        seed: Optional[int] = 42,
    ):
        """Initialize QDET dataset.

        Parameters
        ----------
        name : str
            Name of the dataset.
        num_classes : int
            Number of classes.
        regression : bool
            Whether regression or classification.
        small_dev : bool
            Whether to use small dev set.
        seed : Optional[int], optional
            Random seed, by default 42
        """
        self.name = name
        self.num_classes = num_classes
        self.regression = regression
        self.small_dev = small_dev
        self.seed = seed

    def preprocess_datasets(self) -> DatasetDict:
        """Preprocess datasets.

        Returns
        -------
        DatasetDict
            Dict of dataset splits.
        """
        df_train_original = pd.read_csv(
            os.path.join(
                "../data/processed", f"tf_{self.name}_text_difficulty_train.csv"
            )
        )
        df_test_original = pd.read_csv(
            os.path.join(
                "../data/processed", f"tf_{self.name}_text_difficulty_test.csv"
            )
        )
        dev_prefix = "small_" if self.small_dev else ""
        df_dev_original = pd.read_csv(
            os.path.join(
                "../data/processed",
                f"tf_{self.name}_text_difficulty_{dev_prefix}dev.csv",
            )
        )

        # load answers to integrate the stem
        df_answers = pd.read_csv(
            os.path.join("../data/processed", f"tf_{self.name}_answers_texts.csv")
        )
        answers_dict = dict()
        for q_id, text in df_answers[[TF_QUESTION_ID, TF_DESCRIPTION]].values:
            if q_id not in answers_dict.keys():
                answers_dict[q_id] = ""
            answers_dict[q_id] = f"{answers_dict[q_id]} [SEP] {text}"
        df_answers = pd.DataFrame(
            answers_dict.items(), columns=[TF_QUESTION_ID, TF_ANSWERS]
        )
        df_train_original = pd.merge(
            df_answers,
            df_train_original,
            right_on=TF_QUESTION_ID,
            left_on=TF_QUESTION_ID,
        )
        df_train_original[TF_DESCRIPTION] = (
            df_train_original[TF_DESCRIPTION] + df_train_original[TF_ANSWERS]
        )
        df_test_original = pd.merge(
            df_answers,
            df_test_original,
            right_on=TF_QUESTION_ID,
            left_on=TF_QUESTION_ID,
        )
        df_test_original[TF_DESCRIPTION] = (
            df_test_original[TF_DESCRIPTION] + df_test_original[TF_ANSWERS]
        )
        df_dev_original = pd.merge(
            df_answers,
            df_dev_original,
            right_on=TF_QUESTION_ID,
            left_on=TF_QUESTION_ID,
        )
        df_dev_original[TF_DESCRIPTION] = (
            df_dev_original[TF_DESCRIPTION] + df_dev_original[TF_ANSWERS]
        )

        df_train_original = df_train_original.rename(
            columns={TF_DESCRIPTION: TF_TEXT, TF_DIFFICULTY: TF_LABEL}
        )
        df_test_original = df_test_original.rename(
            columns={TF_DESCRIPTION: TF_TEXT, TF_DIFFICULTY: TF_LABEL}
        )
        df_dev_original = df_dev_original.rename(
            columns={TF_DESCRIPTION: TF_TEXT, TF_DIFFICULTY: TF_LABEL}
        )

        if self.regression:
            df_train_original = df_train_original.astype({TF_LABEL: float})
            df_test_original = df_test_original.astype({TF_LABEL: float})
            df_dev_original = df_dev_original.astype({TF_LABEL: float})

        dataset = DatasetDict(
            {
                TRAIN: Dataset.from_pandas(
                    df_train_original[[TF_QUESTION_ID, TF_TEXT, TF_LABEL]]
                ),
                TEST: Dataset.from_pandas(
                    df_test_original[[TF_QUESTION_ID, TF_TEXT, TF_LABEL]]
                ),
                VALIDATION: Dataset.from_pandas(
                    df_dev_original[[TF_QUESTION_ID, TF_TEXT, TF_LABEL]]
                ),
            }
        )
        if not self.regression:
            dataset = dataset.cast_column(
                TF_LABEL,
                ClassLabel(
                    num_classes=self.num_classes,
                    names=get_difficulty_labels(self.name),
                ),
            )
        # dataset = dataset.remove_columns(["__index_level_0__"])

        return dataset

    def load_all(self) -> tuple:
        """Load all datasets and transform.

        Returns
        -------
        tuple
            Tuple of datasets and transform
        """
        if self.seed is not None:
            set_seed(self.seed)
        dataset = self.preprocess_datasets()
        return dataset
