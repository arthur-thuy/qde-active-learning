"""RACE++ data manager.

Adapted from Github repo qdet_utils/data_manager/_race_data_manager.py
"""

import json
import os
from os import listdir
from typing import Dict

import numpy as np
import pandas as pd

from tools.constants import (
    CONTEXT,
    CONTEXT_ID,
    CORRECT_ANSWER,
    DEV,
    DF_COLS,
    DIFFICULTY,
    OPTION_0,
    OPTION_1,
    OPTION_2,
    OPTION_3,
    OPTIONS,
    Q_ID,
    QUESTION,
    SPLIT,
    TEST,
    TRAIN,
)

from ._data_manager import DataManager


class RaceDatamanager(DataManager):
    """Class for RACE data management."""

    ANSWERS = "answers"
    OPTIONS = "options"
    QUESTIONS = "questions"
    ARTICLE = "article"
    ID = "id"

    HIGH = "high"
    MIDDLE = "middle"
    COLLEGE = "college"

    LEVEL_TO_INT_DIFFICULTY_MAP = {MIDDLE: 0, HIGH: 1, COLLEGE: 2}

    def get_racepp_dataset(
        self,
        race_data_dir: str,
        race_c_data_dir: str,
        output_data_dir: str,
        save_dataset: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """Return RACE++ dataset.

        Parameters
        ----------
        race_data_dir : str
            Folder of RACE dataset
        race_c_data_dir : str
            Folder of RACE-c dataset
        output_data_dir : str
            Folder to store the output dataset
        save_dataset : bool, optional
            Whether to save dataset to file, by default True

        Returns
        -------
        Dict[str, pd.DataFrame]
            Dict of dataset splits
        """
        dataset = dict()
        for split in [TRAIN, DEV, TEST]:
            df_race = self.get_raw_race_df(data_dir=race_data_dir, split=split)
            df_race_c = self.get_raw_race_c_df(data_dir=race_c_data_dir, split=split)
            df = pd.concat([df_race, df_race_c])
            if save_dataset:
                df.to_csv(
                    os.path.join(output_data_dir, f"race_pp_{split}.csv"), index=False
                )
            dataset[split] = df.copy()
        return dataset

    def get_subsampled_racepp_dataset(
        self,
        data_dir: str,
        training_size: int,
        output_data_dir: str,
        random_state: int = None,
        balanced_sampling: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """Return subsampled RACE++ dataset.

        Parameters
        ----------
        data_dir : str
            Folder to read data
        training_size : int
            Number of samples in subsampled training set
        output_data_dir : str
            Folder to store the output dataset
        random_state : int, optional
            Random seed for sampling, by default None
        balanced_sampling : bool, optional
            Whether to have balanced classes, by default True

        Returns
        -------
        Dict[str, pd.DataFrame]
            Dict of subsampled dataset splits
        """
        df_train = pd.read_csv(os.path.join(data_dir, f"race_pp_{TRAIN}.csv"))
        df_dev = pd.read_csv(os.path.join(data_dir, f"race_pp_{DEV}.csv"))
        df_test = pd.read_csv(os.path.join(data_dir, f"race_pp_{TEST}.csv"))
        subsampled_dataset = dict()

        # NOTE: balanced sampling to have exactly `training_size` samples
        NUM_CLASSES = 3
        if balanced_sampling:
            # find observations per class (need exactly init_size samples in total)
            obs_per_class = {
                i: training_size // NUM_CLASSES for i in range(NUM_CLASSES)
            }
            if training_size % NUM_CLASSES != 0:
                extra_idx = np.random.choice(
                    np.arange(NUM_CLASSES),
                    size=(training_size % NUM_CLASSES),
                    replace=False,
                )
                for i in extra_idx:
                    obs_per_class[i] += 1

            # sample indices
            train_list = []
            for i in range(NUM_CLASSES):
                df_train_sampled = df_train[df_train[DIFFICULTY] == i].sample(
                    obs_per_class[i], random_state=random_state
                )
                train_list.append(df_train_sampled)
            df_train = pd.concat(train_list)
        else:
            # sample observations randomly
            df_train = df_train.sample(training_size, random_state=random_state)

        # # NOTE: balanced sampling to have approximately `training_size` samples
        # if balanced_sampling:
        #     # get total of `training_size` samples
        #     samples_per_class = training_size // 3
        #     df_train = pd.concat(
        #         [
        #             df_train[df_train[DIFFICULTY] == 0].sample(
        #                 samples_per_class, random_state=random_state
        #             ),
        #             df_train[df_train[DIFFICULTY] == 1].sample(
        #                 samples_per_class, random_state=random_state
        #             ),
        #             df_train[df_train[DIFFICULTY] == 2].sample(
        #                 samples_per_class, random_state=random_state
        #             ),
        #         ]
        #     )
        # else:
        #     df_train = df_train.sample(training_size, random_state=random_state)

        df_train.to_csv(
            os.path.join(output_data_dir, f"race_pp_{training_size}_{TRAIN}.csv"),
            index=False,
        )
        df_dev.to_csv(
            os.path.join(output_data_dir, f"race_pp_{training_size}_{DEV}.csv"),
            index=False,
        )
        df_test.to_csv(
            os.path.join(output_data_dir, f"race_pp_{training_size}_{TEST}.csv"),
            index=False,
        )
        subsampled_dataset[TRAIN] = df_train.copy()
        subsampled_dataset[DEV] = df_dev.copy()
        subsampled_dataset[TEST] = df_test.copy()
        return subsampled_dataset

    def get_race_dataset(
        self, data_dir: str, out_data_dir: str, save_dataset: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """Return RACE dataset.

        Parameters
        ----------
        data_dir : str
            Folder to read data
        out_data_dir : str
            Folder to store the output dataset
        save_dataset : bool, optional
            Whether to save dataset to file, by default True

        Returns
        -------
        Dict[str, pd.DataFrame]
            Dict of dataset splits
        """
        dataset = dict()
        for split in [TRAIN, DEV, TEST]:
            df = self.get_raw_race_df(data_dir=data_dir, split=split)
            if save_dataset:
                df.to_csv(os.path.join(out_data_dir, f"race_{split}.csv"), index=False)
            dataset[split] = df.copy()
        return dataset

    def get_race_c_dataset(
        self, data_dir: str, out_data_dir: str, save_dataset: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """Return RACE-c dataset.

        Parameters
        ----------
        data_dir : str
            Folder to read data
        out_data_dir : str
            Folder to store the output dataset
        save_dataset : bool, optional
            Whether to save dataset to file, by default True

        Returns
        -------
        Dict[str, pd.DataFrame]
            Dict of dataset splits
        """
        dataset = dict()
        for split in [TRAIN, DEV, TEST]:
            df = self.get_raw_race_c_df(data_dir=data_dir, split=split)
            if save_dataset:
                df.to_csv(
                    os.path.join(out_data_dir, f"race_c_{split}.csv"), index=False
                )
            dataset[split] = df.copy()
        return dataset

    def _append_new_reading_passage_to_df(
        self,
        df: pd.DataFrame,
        reading_passage_data,
        split: str,
        level: str,
    ) -> pd.DataFrame:
        """Append new reading passage to DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to append new reading passage
        reading_passage_data : _type_
            Reading passage to add
        split : str
            Split of the dataset
        level : str
            Level of the passage

        Returns
        -------
        pd.DataFrame
            DataFrame with new reading passage
        """
        answers = reading_passage_data[self.ANSWERS]
        options = reading_passage_data[self.OPTIONS]
        questions = reading_passage_data[self.QUESTIONS]
        article = reading_passage_data[self.ARTICLE]
        context_id = reading_passage_data[self.ID]

        for idx in range(len(questions)):
            # this is just to check that there are no anomalies
            assert ord("A") <= ord(answers[idx]) <= ord("Z")
            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        [
                            {
                                CORRECT_ANSWER: ord(answers[idx]) - ord("A"),
                                OPTIONS: options[idx],
                                OPTION_0: options[idx][0],
                                OPTION_1: options[idx][1],
                                OPTION_2: options[idx][2],
                                OPTION_3: options[idx][3],
                                QUESTION: questions[idx],
                                CONTEXT: article,
                                CONTEXT_ID: context_id[:-4],
                                Q_ID: "%s_q%d" % (context_id[:-4], idx),
                                SPLIT: split,
                                DIFFICULTY: self.LEVEL_TO_INT_DIFFICULTY_MAP[level],
                            }
                        ]
                    ),
                ]
            )
        return df

    def get_raw_race_df(self, data_dir: str, split: str) -> pd.DataFrame:
        """Get RACE dataset from /raw folder.

        Parameters
        ----------
        data_dir : str
            Folder to read data
        split : str
            Split of the dataset

        Returns
        -------
        pd.DataFrame
            RACE dataset from /raw folder
        """
        df = pd.DataFrame(columns=DF_COLS)
        for level in [self.HIGH, self.MIDDLE]:
            for filename in listdir(os.path.join(data_dir, split, level)):
                with open(os.path.join(data_dir, split, level, filename), "r") as f:
                    reading_passage_data = json.load(f)
                df = self._append_new_reading_passage_to_df(
                    df, reading_passage_data, split, level
                )
        assert set(df.columns) == set(DF_COLS)
        return df

    def get_raw_race_c_df(self, data_dir: str, split: str) -> pd.DataFrame:
        """Get RACE-c dataset from /raw folder.

        Parameters
        ----------
        data_dir : str
            Folder to read data
        split : str
            Split of the dataset

        Returns
        -------
        pd.DataFrame
            RACE-c dataset from /raw folder
        """
        df = pd.DataFrame(columns=DF_COLS)
        for filename in listdir(os.path.join(data_dir, split)):
            with open(os.path.join(data_dir, split, filename), "r") as f:
                reading_passage_data = json.load(f)
            df = self._append_new_reading_passage_to_df(
                df, reading_passage_data, split, level=self.COLLEGE
            )
        assert set(df.columns) == set(DF_COLS)
        return df
