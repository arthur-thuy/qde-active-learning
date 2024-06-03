"""Abstract data manager.

Adapted from Github repo qdet_utils/data_manager/_data_manager.py
"""

# standard library imports
import os
from typing import Dict

# related third party imports
import pandas as pd

# local application/library specific imports
from tools.constants import (
    CONTEXT,
    DEV,
    DF_COLS,
    DIFFICULTY,
    Q_ID,
    QUESTION,
    TEST,
    TF_ANS_ID,
    TF_CORRECT,
    TF_DESCRIPTION,
    TF_QUESTION_ID,
    TRAIN,
)


class DataManager:
    """Abstract data manager."""

    def __init__(self):
        """No constructor."""
        pass

    TF_AS_TYPE_DICT = {
        TF_CORRECT: bool,
        TF_DESCRIPTION: str,
        TF_ANS_ID: str,
        TF_QUESTION_ID: str,
    }
    TF_DF_COLS_ANS_DF = [TF_CORRECT, TF_DESCRIPTION, TF_ANS_ID, TF_QUESTION_ID]
    TF_DF_COLS_TEXT_DIFFICULTY_DF = [TF_DESCRIPTION, TF_QUESTION_ID, DIFFICULTY]

    def convert_to_transformers_format_and_store_dataset(
        self,
        dataset: Dict[str, pd.DataFrame],
        data_dir: str,
        dataset_name: str,
        skip_answers_texts: bool,
    ) -> None:
        """Convert dataset to transformers format and store it.

        Parameters
        ----------
        dataset : Dict[str, pd.DataFrame]
            Dataset to convert.
        data_dir : str
            Directory to store the dataset.
        dataset_name : str
            Name of the dataset.
        skip_answers_texts : bool
            Whether to skip answers texts.
        """
        answer_texts_df = pd.DataFrame(columns=self.TF_DF_COLS_ANS_DF)

        text_difficulty_df, answer_texts_df = self.get_text_difficulty_and_answer_texts(
            dataset[TRAIN], answer_texts_df, skip_answers_texts
        )
        text_difficulty_df.to_csv(
            os.path.join(
                data_dir,
                f"tf_{dataset_name}_text_difficulty_train.csv",
            ),
            index=False,
        )

        text_difficulty_df, answer_texts_df = self.get_text_difficulty_and_answer_texts(
            dataset[TEST], answer_texts_df, skip_answers_texts
        )
        text_difficulty_df.to_csv(
            os.path.join(
                data_dir,
                f"tf_{dataset_name}_text_difficulty_test.csv",
            ),
            index=False,
        )

        text_difficulty_df, answer_texts_df = self.get_text_difficulty_and_answer_texts(
            dataset[DEV], answer_texts_df, skip_answers_texts
        )
        text_difficulty_df.to_csv(
            os.path.join(
                data_dir,
                f"tf_{dataset_name}_text_difficulty_dev.csv",
            ),
            index=False,
        )

        if not skip_answers_texts:
            answer_texts_df.to_csv(
                os.path.join(
                    data_dir,
                    f"tf_{dataset_name}_answers_texts.csv",
                ),
                index=False,
            )

    def get_text_difficulty_and_answer_texts(
        self, df: pd.DataFrame, answers_text_df: pd.DataFrame, skip_ans_texts: bool
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Get text difficulty and answer texts.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe with text difficulty.
        answers_text_df : pd.DataFrame
            Dataframe with answer texts.
        skip_ans_texts : bool
            Whether to skip answers texts.

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame]
            Dataframes of text difficulty and answer texts.
        """
        text_difficulty_df = pd.DataFrame(columns=self.TF_DF_COLS_TEXT_DIFFICULTY_DF)
        if skip_ans_texts:
            for question, context, q_id, difficulty in df[
                [QUESTION, CONTEXT, Q_ID, DIFFICULTY]
            ].values:
                text_difficulty_df = pd.concat(
                    [
                        text_difficulty_df,
                        self._get_new_row_text_difficulty_df(
                            q_id, question, context, difficulty
                        ),
                    ],
                    ignore_index=True,
                )
        else:
            for (
                correct_option,
                _,
                opt0,
                opt1,
                opt2,
                opt3,
                question,
                context,
                _,
                q_id,
                _,
                difficulty,
            ) in df[DF_COLS].values:
                answers_text_df = pd.concat(
                    [
                        answers_text_df.astype(self.TF_AS_TYPE_DICT),
                        self._get_new_rows_answers_text_df(
                            correct_option, q_id, [opt0, opt1, opt2, opt3]
                        ).astype(self.TF_AS_TYPE_DICT),
                    ],
                    ignore_index=True,
                )
                text_difficulty_df = pd.concat(
                    [
                        text_difficulty_df,
                        self._get_new_row_text_difficulty_df(
                            q_id, question, context, difficulty
                        ),
                    ],
                    ignore_index=True,
                )
        return text_difficulty_df, answers_text_df

    def _get_new_rows_answers_text_df(
        self, correct_ans: str, q_id: str, options: list
    ) -> pd.DataFrame:
        """Append new rows to answers text dataframe.

        Parameters
        ----------
        correct_ans : str
            Correct answer
        q_id : str
            Question ID
        options : list
            List of answer options

        Returns
        -------
        pd.DataFrame
            Dataframe with new row
        """
        out_df = pd.DataFrame(columns=self.TF_DF_COLS_ANS_DF)
        for idx, option in enumerate(options):
            new_row = pd.DataFrame(
                [
                    {
                        TF_CORRECT: idx == correct_ans,
                        TF_DESCRIPTION: option,
                        TF_ANS_ID: idx,
                        TF_QUESTION_ID: q_id,
                    }
                ]
            )
            out_df = pd.concat(
                [
                    out_df.astype(self.TF_AS_TYPE_DICT),
                    new_row.astype(self.TF_AS_TYPE_DICT),
                ],
                ignore_index=True,
            )
        return out_df

    def _get_new_row_text_difficulty_df(
        self, q_id: str, question: str, context: str, difficulty: int
    ) -> pd.DataFrame:
        """Build new DF row for text difficulty.

        Parameters
        ----------
        q_id : str
            Question ID
        question : str
            Question content
        context : str
            Context content
        difficulty : int
            Difficulty level

        Returns
        -------
        pd.DataFrame
            Dataframe with single row
        """
        context = "" if not isinstance(context, str) else context + " "
        return pd.DataFrame(
            [
                {
                    TF_DESCRIPTION: context + question,
                    TF_QUESTION_ID: q_id,
                    DIFFICULTY: difficulty,
                }
            ]
        )
