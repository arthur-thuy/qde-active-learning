"""Module for random prediction trainer."""

# standard library imports
import logging
from typing import Literal

# related third party imports
import numpy as np
import structlog
from yacs.config import CfgNode

# local application/library specific imports
from tools.metrics import compute_metrics_classification, compute_metrics_regression

# set up logger
structlog.stdlib.recreate_defaults(log_level=logging.WARNING)
logger = structlog.get_logger("qdet")

ROUND_FLOAT = 4
METRIC_KEY_PREFIX = "test"


def baseline_procedure(
    cfg: CfgNode, datasets, method: Literal["random", "majority"]
) -> dict:
    """Baseline prediction trainer.

    Can be either random or majority prediction.

    Parameters
    ----------
    cfg : CfgNode
        Config object
    datasets : _type_
        HF datasets
    method : Literal["random", "majority"]
        Method for prediction

    Returns
    -------
    dict
        Metrics
    """
    if method not in ["random", "majority"]:
        raise ValueError(f"Invalid method: {method}")
    ##### EVALUATION #####
    logger.info(
        "Evaluate - start",
        test_size=len(datasets["test"]),
    )
    # evaluate
    train_labels = datasets["train"]["label"]

    # TODO: for marginal random, need class distribution in train set

    if cfg.LOADER.REGRESSION:
        if method == "random":
            eval_preds = np.random.uniform(
                low=np.min(train_labels) - 0.5,
                high=np.max(train_labels) + 0.5,
                size=len(datasets["test"]["label"]),
            )
        else:  # NOTE: majority
            unique, counts = np.unique(train_labels, return_counts=True)
            majority = unique[np.argmax(counts)]
            eval_preds = np.repeat(majority, len(datasets["test"]["label"]))

        eval_pred_label = (eval_preds, datasets["test"]["label"])
        eval_metrics = compute_metrics_regression(
            eval_pred_label, dataset_name=cfg.LOADER.NAME
        )
        eval_metrics = {f"{METRIC_KEY_PREFIX}_{k}": v for k, v in eval_metrics.items()}

        logger.info(
            "Evaluate - end",
            test_rmse=round(eval_metrics["test_rmse"], ROUND_FLOAT),
            test_discrete_rmse=round(eval_metrics["test_discrete_rmse"], ROUND_FLOAT),
        )
    else:  # NOTE: classification
        if method == "random":
            eval_preds = np.random.choice(
                list(set(train_labels)),
                size=len(datasets["test"]["label"]),
                replace=True,
            )
        else:  # NOTE: majority
            NotImplementedError("Marginal probability not implemented yet!")
        eval_pred_label = (eval_preds, datasets["test"]["label"])
        eval_metrics = compute_metrics_classification(
            eval_pred_label, dataset_name=cfg.LOADER.NAME
        )
        eval_metrics = {f"{METRIC_KEY_PREFIX}_{k}": v for k, v in eval_metrics.items()}

        logger.info(
            "Evaluate - end",
            test_acc=round(eval_metrics["test_accuracy"], ROUND_FLOAT),
        )

    metrics = {}
    metrics[datasets["train"].num_rows] = {
        **eval_metrics,
        "dataset_size": datasets["train"].num_rows,
        "test_pred_label": eval_pred_label,
    }

    return metrics
