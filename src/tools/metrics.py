"""Metrics."""

from typing import Callable, Literal

import evaluate
import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score, root_mean_squared_error
from torchmetrics import Accuracy as torchAccuracy

from tools.constants import (
    AM,
    ARC,
    ARC_BALANCED,
    RACE,
    RACE_4K,
    RACE_PP,
    RACE_PP_4K,
    RACE_PP_8K,
    RACE_PP_12K,
)


def compute_metrics_regression(eval_pred, dataset_name: str):
    """Determine which metrics to use for evaluation."""
    predictions, labels = eval_pred
    # continuous predictions
    # r_squared = evaluate.load("r_squared")
    # spearmanr = evaluate.load("spearmanr")

    # discrete predictions
    mapper = get_difficulty_mapper(dataset_name)
    discrete_predictions = [mapper(x) for x in predictions]
    # discrete_r_squared = evaluate.load("r_squared")
    # discrete_spearmanr = evaluate.load("spearmanr")

    return {
        # "r_squared": r_squared.compute(predictions=predictions, references=labels),
        "rmse": root_mean_squared_error(labels, predictions),
        # "spearmanr": spearmanr.compute(predictions=predictions, references=labels)[
        #     "spearmanr"
        # ],
        # "discrete_r_squared": discrete_r_squared.compute(
        #     predictions=discrete_predictions, references=labels
        # ),
        "discrete_rmse": root_mean_squared_error(labels, discrete_predictions),
        # "discrete_spearmanr": discrete_spearmanr.compute(
        #     predictions=discrete_predictions, references=labels
        # )["spearmanr"],
    }


def compute_metrics_classification(eval_pred, num_classes: int):
    """Determine which metrics to use for classification evaluation."""
    accuracy = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc_class = torchAccuracy(task="multiclass", num_classes=num_classes, average=None)

    return {
        "accuracy": accuracy.compute(predictions=predictions, references=labels),
        "f1": f1_metric.compute(predictions=predictions, references=labels),
        "bal_accuracy": balanced_accuracy_score(labels, predictions),
        "bal_f1": f1_score(labels, predictions, average="weighted"),
        "acc_class": acc_class(
            torch.from_numpy(predictions), torch.from_numpy(labels)
        ).tolist(),
    }
    # return accuracy.compute(predictions=predictions, references=labels)


def mapper_race(pred_score: float) -> Literal[0, 1, 2]:
    """Map race difficulty to discrete values.

    Parameters
    ----------
    pred_score : float
        Predicted difficulty score

    Returns
    -------
    Literal[0, 1, 2]
        RACE difficulty label
    """
    if pred_score <= 0.5:
        return 0
    elif pred_score < 1.5:
        return 1
    else:
        return 2


def identity_mapper(pred_score: float) -> float:
    """Return the input value.

    Parameters
    ----------
    pred_score : float
        Predicted difficulty score

    Returns
    -------
    float
        Input value
    """
    return pred_score


def mapper_am(pred_score: float) -> float:
    """Map AM difficulty to itself because continuous.

    Parameters
    ----------
    pred_score : float
        Predicted difficulty score

    Returns
    -------
    float
        Input value
    """
    return identity_mapper(pred_score)


def mapper_arc(pred_score: float) -> Literal[3, 4, 5, 6, 7, 8, 9]:
    """Map ARC difficulty to discrete values.

    Parameters
    ----------
    pred_score : float
        Predicted difficulty score

    Returns
    -------
    Literal[3, 4, 5, 6, 7, 8, 9]
        ARC difficulty label
    """
    if pred_score < 3.5:
        return 3
    elif pred_score >= 8.5:
        return 9
    else:
        return round(pred_score)


def get_difficulty_mapper(dataset: str) -> Callable:
    """Get difficulty mapper from dataset name.

    Parameters
    ----------
    dataset : str
        Dataset name

    Returns
    -------
    Callable
        Difficulty mapper function
    """
    if dataset in {RACE_PP, RACE_PP_4K, RACE_PP_8K, RACE_PP_12K, RACE, RACE_4K}:
        return mapper_race
    if dataset in {ARC, ARC_BALANCED}:
        return mapper_arc
    if dataset == AM:
        return mapper_am
    else:
        return identity_mapper


def get_difficulty_labels(dataset: str) -> list[str]:
    """Get difficulty labels from dataset name.

    Parameters
    ----------
    dataset : str
        Dataset name

    Returns
    -------
    list[str]
        Difficulty labels

    Raises
    ------
    ValueError
        If dataset is AM
    NotImplementedError
        If dataset is not recognized
    """
    if dataset in {RACE, RACE_4K}:
        return ["middle", "high"]
    if dataset in {RACE_PP, RACE_PP_4K, RACE_PP_8K, RACE_PP_12K}:
        return ["middle", "high", "university"]
    if dataset in {ARC, ARC_BALANCED}:
        return [
            "level_3",
            "level_4",
            "level_5",
            "level_6",
            "level_7",
            "level_8",
            "level_9",
        ]
    if dataset in {AM}:
        raise ValueError("AM does not have difficulty labels")
    else:
        raise NotImplementedError
