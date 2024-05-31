"""File with analyzer functionalities."""

# standard library imports
import glob
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Optional, Tuple

# related third party imports
import numpy as np
import scipy
import structlog
import torch
from datasets import ClassLabel
from numpy.typing import ArrayLike, NDArray
from transformers.modelcard import parse_log_history

from tools.metrics import compute_metrics_regression

# local application/library specific imports
from tools.utils import save_checkpoint

# set up logger
logger = structlog.get_logger("qdet")


def mean_stderror(ary: NDArray, axis: Optional[int] = None) -> Tuple[NDArray, NDArray]:
    """Calculate mean and standard error from array.

    Parameters
    ----------
    ary : NDArray
        Output array containing metrics
    axis : Union[Any, None], optional
        Axis to average over, by default None

    Returns
    -------
    Tuple[NDArray, NDArray]
        Tuple of mean and standard error
    """
    mean = np.mean(ary, axis=axis)
    stderror = scipy.stats.sem(ary, ddof=1, axis=axis)
    return mean, stderror


def get_output_paths(experiment: str, config_ids: list[str]) -> list[str]:
    """Get paths to output files.

    Parameters
    ----------
    experiment : str
        Experiment name
    config_ids : list[str]
        List of config IDs

    Returns
    -------
    list[str]
        List of output paths
    """
    # paths
    output_dir = os.path.join("output", experiment)
    output_paths = [
        os.path.join(output_dir, f"{config_id}.pth") for config_id in config_ids
    ]

    return output_paths


def compute_metric_vs_size(
    experiment: str, metric: str, config_ids: list[str], run_id: Optional[int] = None
) -> dict[str, dict[str, NDArray]]:
    """Read output data and get dataset sizes and metrics.

    Parameters
    ----------
    experiment : str
        Experiment name
    metric : str
        Metric name
    config_ids : list[str]
        List of config IDs
    run_id : Optional[int], optional
        Run ID to plot. If None, all runs are averaged, by default None

    Returns
    -------
    dict[str, dict[str, NDArray]]
        Dict of results
    """
    # paths
    output_paths = get_output_paths(experiment, config_ids)

    # max dataset size
    max_ds_size = torch.load(output_paths[0])["max_ds_size"]

    # metrics and dataset sizes
    results = defaultdict(dict)  # type: defaultdict[str, dict[str, NDArray]]
    for config_id, output_path in zip(config_ids, output_paths):
        logger.info(f"Loading checkpoint from '{output_path}'")
        hist_dict = torch.load(output_path)["metrics"]
        dataset_sizes = [epoch["dataset_size"] for epoch in hist_dict["run_1"].values()]

        run_agg_list = list()
        for run_key, run_value in hist_dict.items():
            if run_id is not None and run_key != f"run_{run_id}":
                continue
            run_agg_list.append([epoch[metric] for epoch in run_value.values()])
        mean, stderr = mean_stderror(np.array(run_agg_list), axis=0)
        results[config_id]["mean"] = mean
        results[config_id]["stderr"] = stderr
        results[config_id]["dataset_sizes"] = dataset_sizes
        results[config_id]["max_ds_size"] = max_ds_size

    # for full_data configs, fill in with first regular AL dataset_sizes
    # NOTE: initialise template_dataset_sizes with default -> used if only full_data
    template_dataset_sizes = list(range(0, max_ds_size + 1, 1000))
    for config_id in config_ids:
        if "full_data" not in config_id:
            template_dataset_sizes = results[config_id]["dataset_sizes"]
            break
    for config_id in config_ids:
        if "full_data" in config_id:
            results[config_id]["dataset_sizes"] = template_dataset_sizes

    return dict(results)


def get_labelling_progress(
    experiment: str, config_ids: list[str]
) -> dict[str, dict[str, NDArray]]:
    """Read output data and get labelling progress.

    Parameters
    ----------
    experiment : str
        Experiment name
    config_ids : list[str]
        List of config IDs

    Returns
    -------
    dict[str, dict[str, NDArray]]
        Dict of config IDs and labelling arrays
    """
    # paths
    output_paths = get_output_paths(experiment, config_ids)

    # labelling progress
    labelling_dict = defaultdict(dict)  # type: defaultdict[str, dict[str, NDArray]]
    for config_id, output_path in zip(config_ids, output_paths):
        print(f"=> loading checkpoint from '{output_path}'")
        labelling_dict[config_id] = torch.load(output_path)["labelling_progress"]

    return dict(labelling_dict)


class Dict2Class(object):
    """Turns a dictionary into a class with attributes.

    Parameters
    ----------
    object : dict
        dict of config parameters
    """

    def __init__(self, my_dict: dict):
        for key in my_dict:
            setattr(self, key, my_dict[key])


def get_idx_acquired(
    labelling_progress: NDArray[np.int32], acq_step: int
) -> NDArray[np.int64]:
    """Get indices of observations acquired at some step.

    Parameters
    ----------
    labelling_progress : NDArray[np.int32]
        Labelling progress saved to disk by Baal
    acq_step : int
        Acquisition step to get (0: initial; 1: first acquisition; etc.)

    Returns
    -------
    NDArray[np.int64]
        Indices of observations acquired at some step
    """
    step_idx = labelling_progress.max() - acq_step
    idx_acquired = np.where(labelling_progress == step_idx)[0]
    return idx_acquired


def find_label_map(dataset) -> dict:
    """Find label map dictionary from HF dataset.

    Parameters
    ----------
    dataset : _type_
        HF dataset

    Returns
    -------
    dict
        Label map dictionary (type {int: str})
    """
    if isinstance(dataset.features["label"], ClassLabel):
        label_str = dataset.features["label"].names
    else:
        label_str = np.unique(np.array(dataset["label"]).astype(int)).astype(str)
    label_int = np.unique(np.array(dataset["label"]).astype(int))
    label_map = {k: v for k, v in zip(label_int, label_str)}
    return label_map


def get_train_logs(
    exp_name: str, config_id: str, run_id: int, ds_size: Optional[int] = None
) -> tuple:
    """Get training logs to inspect learning convergence.

    Parameters
    ----------
    exp_name : str
        Experiment name
    config_id : str
        Config ID
    run_id : int
        Run ID
    ds_size : Optional[int], optional
        Dataset size to show convergence for, by default None

    Returns
    -------
    tuple
        Tuple of training logs, lines, and evaluation results
    """
    output_path = get_output_paths(exp_name, [config_id])[0]

    if "full_data" in output_path:
        # NOTE: for full data, there is only one ds_size
        all_metrics = torch.load(output_path)["metrics"][f"run_{run_id}"]
        ds_sizes = list(all_metrics.keys())
        assert len(ds_sizes) == 1
        log_history = all_metrics[ds_sizes[0]]["convergence"]
        logger.info(
            f"Loading convergence for `full_data` config ({ds_sizes[0]} samples)"
        )
    else:
        # NOTE: for active learning, find ds_size in function argument
        assert ds_size is not None, "Provide an active learning dataset size"
        log_history = torch.load(output_path)["metrics"][f"run_{run_id}"][ds_size][
            "convergence"
        ]
        logger.info(
            f"Loading convergence for active learning config ({ds_size} samples)"
        )

    train_log, lines, eval_results = parse_log_history(
        log_history
    )  # NOTE: func from transformers.modelcard
    return train_log, lines, eval_results


def get_single_pred_label(
    exp_name: str, config_id: str, run_id: int, ds_size: Optional[int] = None
) -> tuple:
    """Get predictions and labels.

    Parameters
    ----------
    exp_name : str
        Experiment name
    config_id : str
        Config ID
    run_id : int
        Run ID
    ds_size : Optional[int], optional
        Dataset size to show convergence for, by default None

    Returns
    -------
    tuple
        Tuple of predictions and labels
    """
    output_path = get_output_paths(exp_name, [config_id])[0]

    if "full_data" in output_path:
        # NOTE: for full data, there is only one ds_size
        all_metrics = torch.load(output_path)["metrics"][f"run_{run_id}"]
        ds_sizes = list(all_metrics.keys())
        assert len(ds_sizes) == 1
        test_pred_label = all_metrics[ds_sizes[0]]["test_pred_label"]
        logger.info(
            f"Loading convergence for `full_data` config ({ds_sizes[0]} samples)"
        )
    else:
        # NOTE: for active learning, find ds_size in function argument
        assert ds_size is not None, "Provide an active learning dataset size"
        test_pred_label = torch.load(output_path)["metrics"][f"run_{run_id}"][ds_size][
            "test_pred_label"
        ]
        logger.info(
            f"Loading convergence for active learning config ({ds_size} samples)"
        )
    return test_pred_label


def compute_level_performance(
    experiment: str,
    metric: str,
    config_ids: list[str],
    config_dict: dict,
    label_map: dict,
    run_id: Optional[int] = None,
) -> dict:
    """Get metrics per difficulty level.

    Parameters
    ----------
    experiment : str
        Experiment name
    metric : str
        Metric name
    config_ids : list[str]
        List of config IDs
    config_dict : dict
        Dict of config objects
    label_map : dict
        Mapping integer labels to string labels
    run_id : Optional[int], optional
        Run ID to plot. If None, all runs are averaged, by default None

    Returns
    -------
    dict
        Dict of results per config and level.
    """
    # log
    logger.info(
        f"Computing {'all runs' if run_id is None else f'run {run_id}'} "
        f"of config_ids {config_ids}"
    )

    # paths
    output_paths = get_output_paths(experiment, config_ids)

    # max dataset size
    max_ds_size = torch.load(output_paths[0])["max_ds_size"]

    # metrics and dataset sizes
    results = defaultdict(dict)
    for config_id, output_path in zip(config_ids, output_paths):
        logger.info(f"Loading checkpoint from '{output_path}'")
        hist_dict = torch.load(output_path)["metrics"]
        dataset_name = config_dict[config_id]["LOADER"]["NAME"]

        # initialize dict over runs
        run_agg_dict = {k: [] for k in list(label_map.values())}
        run_agg_dict["pred_parity"] = []
        for run_key, run_value in hist_dict.items():
            if run_id is not None and run_key != f"run_{run_id}":
                continue
            # initialize dict within run
            metric_level_dict = {k: [] for k in list(label_map.values())}
            for ds_value in run_value.values():
                preds, labels = ds_value["test_pred_label"]
                for diff_level in np.nditer(np.unique(labels)):
                    test_metrics = compute_metrics_regression(
                        eval_pred=(
                            preds[labels == diff_level],
                            labels[labels == diff_level],
                        ),
                        dataset_name=dataset_name,
                    )
                    metric_level_dict[str(int(diff_level))].append(test_metrics[metric])

            # compute predictive parity
            pred_parity = np.minimum.reduce(list(metric_level_dict.values())).tolist()
            metric_level_dict["pred_parity"] = pred_parity
            # add within run results to dict over runs
            for k, v in metric_level_dict.items():
                run_agg_dict[k].append(v)

        # average over runs
        for k, v in run_agg_dict.items():
            mean, stderr = mean_stderror(np.array(v), axis=0)
            results[config_id][k] = {"mean": mean, "stderr": stderr}
        history = torch.load(output_path)["metrics"][run_key]  # from last run
        dataset_sizes = [epoch["dataset_size"] for epoch in history.values()]
        results[config_id]["dataset_sizes"] = dataset_sizes
        results[config_id]["max_ds_size"] = max_ds_size

    # for full_data configs, fill in with first regular AL dataset_sizes
    # NOTE: initialise template_dataset_sizes with default -> used if only full_data
    template_dataset_sizes = list(range(0, max_ds_size + 1, 1000))
    for config_id in config_ids:
        if "full_data" not in config_id:
            template_dataset_sizes = results[config_id]["dataset_sizes"]
            break
    for config_id in config_ids:
        if "full_data" in config_id:
            results[config_id]["dataset_sizes"] = template_dataset_sizes

    return dict(results)


def merge_run_results(output_dir: str) -> None:
    """Merge all results from different runs within a config.

    Parameters
    ----------
    output_dir : str
        Path to output directory
    """
    output_paths = glob.glob(os.path.join(output_dir, "*.pth"))

    # check if overlap in run IDs
    run_id_list = [
        re.search(r"run_(\d+)", output_path).group(1) for output_path in output_paths
    ]
    if len(run_id_list) != len(set(run_id_list)):
        raise ValueError("Overlap in run IDs!")

    all_metrics = {}
    all_model_weights = {}
    all_labelling_progress = {}
    for output_path in output_paths:
        run_n = re.search(r"run_(\d+)", output_path).group(1)
        result = torch.load(output_path)

        all_metrics[f"run_{run_n}"] = result["metrics"]
        all_model_weights[f"run_{run_n}"] = result["model"]
        all_labelling_progress[f"run_{run_n}"] = result["labelling_progress"]

    max_ds_size = torch.load(output_paths[0])["max_ds_size"]

    path = Path(output_dir)
    logger.info(
        f"Merging runs {run_id_list} in: "
        f"{os.path.join(path.parent, f'{path.stem}.pth')}"
    )
    save_checkpoint(
        {
            "model": all_model_weights,
            "labelling_progress": all_labelling_progress,
            "metrics": all_metrics,
            "max_ds_size": max_ds_size,
        },
        save_dir=path.parent,
        fname=path.stem,
    )


def merge_all_results(experiment: str, config_ids: list[str]):
    """Merge results for all configs.

    Parameters
    ----------
    experiment : str
        Experiment name
    config_ids : list[str]
        List of config IDs
    """
    # paths
    output_dir = os.path.join("output", experiment)
    output_paths = [os.path.join(output_dir, config_id) for config_id in config_ids]

    for output_path in output_paths:
        merge_run_results(output_path)


def compute_level_acquisitions(
    labelling_dict: dict,
    datasets: dict,
    config_dict: dict,
    config_id: str,
    exp_name: str,
    run_id: Optional[int] = None,
    only_acquisition: bool = True,
) -> dict[str, ArrayLike]:
    """Compute level acquisitions as a function of the dataset size.

    Parameters
    ----------
    labelling_dict : dict
        Dict of `labelling_progress` objects saved to disk by Baal
    datasets : _type_
        Datasets for all runs
    config_dict : dict
        Dict of config objects
    config_id : str
        Configuration ID
    exp_name : str
        Experiment name
    run_id : Optional[int]
        Run ID to plot. If None, all runs are averaged, by default None
    only_acquisition : bool, optional
        Show distribution of samples only acquired at a specific step.
        If False, show distribution of all labeled samples acquired up to that step,
        by default True

    Returns
    -------
    dict[str, ArrayLike]
        Dict of results
    """
    # log
    logger.info(
        f"Computing {'all runs' if run_id is None else f'run {run_id}'} "
        f"of config_id {config_id}"
    )

    # get class proportions
    def _get_bincount_labels(
        idx_acquired: NDArray[np.int64], labels, num_classes: int
    ) -> NDArray[np.int64]:
        """Get bincount of labels acquired at some step.

        Parameters
        ----------
        idx_acquired : NDArray[np.int64]
            Indices acquired at some step
        labels : _type_
            Training dataset labels
        num_classes : int
            Number of classes

        Returns
        -------
        NDArray[np.int64]
            Bincount of labels acquired at some step
        """
        labels_idx = np.array(labels)[idx_acquired]
        class_count = np.bincount(labels_idx, minlength=num_classes)
        return class_count

    # iterate over runs
    run_agg_list = list()
    for run_id_tmp, run_value in labelling_dict[config_id].items():
        if run_id is not None and run_id_tmp != f"run_{run_id}":
            continue
        # get indices acquired at each step
        labelling_progress = run_value
        all_idx_acquired = []
        for i in range(labelling_progress.max()):
            idx_acquired = get_idx_acquired(labelling_progress, i)
            all_idx_acquired.append(idx_acquired)

        num_classes = config_dict[config_id]["MODEL"]["NUM_LABELS"]
        labels = np.array(datasets[run_id_tmp]["train"]["label"], dtype=int)
        label_counts = [
            _get_bincount_labels(idx_acquired, labels, num_classes)
            for idx_acquired in all_idx_acquired
        ]
        label_vstack = np.vstack(label_counts)
        if not only_acquisition:
            label_vstack = label_vstack.cumsum(axis=0)
        label_prop = label_vstack / label_vstack.sum(axis=1, keepdims=True)
        run_agg_list.append(label_prop)

    mean, stderr = mean_stderror(np.array(run_agg_list), axis=0)
    results = {"mean": mean, "stderr": stderr}

    # get dataset size (can take first path because only 1)
    output_path = get_output_paths(exp_name, [config_id])
    max_ds_size = torch.load(output_path[0])["max_ds_size"]
    history = torch.load(output_path[0])["metrics"][run_id_tmp]
    dataset_sizes = [epoch["dataset_size"] for epoch in history.values()]
    results["dataset_sizes"] = dataset_sizes
    results["max_ds_size"] = max_ds_size

    return results
