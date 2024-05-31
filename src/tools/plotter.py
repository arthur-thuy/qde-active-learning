"""File with plotting functionalities."""

# standard library imports
import os
from typing import List, Literal, Optional

import matplotlib.pyplot as plt

# related third party imports
import numpy as np
import seaborn as sns
import structlog
from matplotlib.ticker import PercentFormatter, ScalarFormatter

# local application/library specific imports
from tools.analyzer import (
    compute_level_acquisitions,
    compute_level_performance,
    compute_metric_vs_size,
)
from tools.utils import ensure_dir

# set up logger
logger = structlog.get_logger("qdet")


def plot_level_acquisitions(
    labelling_dict: dict,
    datasets: dict,
    label_map: dict,
    config_dict: dict,
    config_id: str,
    exp_name: str,
    run_id: Optional[int] = None,
    x_axis: Literal["steps", "samples", "percent"] = "samples",
    only_acquisition: bool = True,
    save: bool = False,
    savefig_kwargs: Optional[dict] = None,
) -> tuple:
    """Plot level acquisitions as a function of the dataset size.

    Parameters
    ----------
    labelling_dict : dict
        Dict of `labelling_progress` objects saved to disk by Baal
    datasets : _type_
        Datasets for all runs
    label_map : dict
        Dict mapping labels to class names
    config_dict : dict
        Dict of config objects
    config_id : str
        Configuration ID
    exp_name : str
        Experiment name
    run_id : Optional[int]
        Run ID to plot. If None, all runs are averaged, by default None
    x_axis : Literal["steps", "samples", "percent"], optional
        X-axis content, by default "samples"
    only_acquisition : bool, optional
        Show distribution of samples only acquired at a specific step.
        If False, show distribution of all labeled samples acquired up to that step,
        by default True
    save : bool, optional
        Whether to save the plot to file, by default False
    savefig_kwargs : dict, optional
        Kwargs to save the figure, by default None

    Returns
    -------
    tuple
        Tuple of label counts and label proportions
    """
    if only_acquisition:
        logger.warning(
            "Showing only level distribution acquired at each specific step!"
        )
        ylabel = "Proportion of samples acquired at step"
    else:
        logger.warning(
            "Showing distribution of all labeled samples, "
            "from this or previous acquisitions!"
        )
        ylabel = "Proportion of total labeled samples"

    results = compute_level_acquisitions(
        labelling_dict=labelling_dict,
        datasets=datasets,
        config_dict=config_dict,
        config_id=config_id,
        exp_name=exp_name,
        run_id=run_id,
        only_acquisition=only_acquisition,
    )

    # determine x axis content
    if x_axis == "samples":
        x_range = results["dataset_sizes"]
        xlabel = "Number of labeled samples"
        x_formatter = ScalarFormatter()
    elif x_axis == "steps":
        x_range = list(range(len(results["dataset_sizes"])))
        xlabel = "Number of acquisition steps"
        x_formatter = ScalarFormatter()
    else:  # show_size == "percent"
        x_range = np.array(results["dataset_sizes"]) / results["max_ds_size"] * 100
        xlabel = "Percentage of labeled samples"
        x_formatter = PercentFormatter(decimals=1)

    # plot stacked area chart (switch legend order around)
    _, ax = plt.subplots(figsize=(16 / 3, 10 / 3))
    y = [results["mean"][:, i] for i in range(results["mean"].shape[1])]
    ax.stackplot(x_range, y, labels=label_map)
    handles, labels = plt.gca().get_legend_handles_labels()
    order = list(reversed(range(len(labels))))
    ax.legend(
        [handles[idx] for idx in order],
        [labels[idx] for idx in order],
        loc="center left",
        bbox_to_anchor=(1, 0.5),
    )
    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.grid(linestyle="dashed")
    ax.xaxis.set_major_formatter(x_formatter)
    # get ticks in sans-serif if sans-serif is used
    ax.xaxis.get_major_formatter()._usetex = False
    ax.yaxis.get_major_formatter()._usetex = False
    if save:
        plt.tight_layout()
        ensure_dir(os.path.dirname(savefig_kwargs["fname"]))
        plt.savefig(**savefig_kwargs)

    plt.show()


def plot_level_performance(
    experiment: str,
    metric: str,
    config_ids: List[str],
    config_dict: dict,
    label_map: dict,
    diff_level: Optional[str] = None,
    run_id: Optional[int] = None,
    x_axis: Literal["steps", "samples", "percent"] = "samples",
    stderr: bool = False,
    config2legend: Optional[dict] = None,
    metric2legend: Optional[dict] = None,
    save: bool = False,
    savefig_kwargs: Optional[dict] = None,
):
    """Plot results per difficulty level.

    Parameters
    ----------
    experiment : str
        Experiment name
    metric : str
        Metric name
    config_ids : List[str]
        List of config IDs
    config_dict : dict
        Dict of config objects
    label_map : dict
        Mapping integer labels to string labels
    diff_level : Optional[str], optional
        Difficulty level to plot. If none, all levels plotted, by default None
    run_id : Optional[int], optional
        Run ID to plot. If None, all runs are averaged, by default None
    x_axis : Literal["steps", "samples", "percent"], optional
        X-axis content, by default "samples"
    stderr : bool, optional
        Whether to plot standard error region, by default False
    config2legend : dict, optional
        Dictionary to map config ids to legend names, by default None
    metric2legend : dict, optional
        Dictionary to map metric names to legend names, by default None
    save : bool, optional
        Whether to save the plot to file, by default False
    savefig_kwargs : dict, optional
        Kwargs to save the figure, by default None.

    Raises
    ------
    ValueError
        If multiple configs and multiple difficulty levels
    """
    # checks
    if len(config_ids) > 1 and diff_level is None:
        raise ValueError(
            "Multiple configs and multiple difficulty levels not supported"
        )

    # get results
    level_results = compute_level_performance(
        experiment,
        metric.replace("test_", ""),
        config_ids,
        label_map=label_map,
        config_dict=config_dict,
        run_id=run_id,
    )

    # difficulty levels
    diff_levels = list(label_map.values()) if diff_level is None else [diff_level]

    # find metric name to print
    if metric2legend is not None:
        metric_print = metric2legend.get(metric, metric)
    else:
        metric_print = metric

    # get ylabel
    if diff_level is None:
        ylabel = f"{metric_print} per level"
    else:
        ylabel = f"{metric_print} - level {diff_level}"

    # log
    logger.info(f"Plotting {ylabel}, for config_ids: {config_ids}")

    _, ax = plt.subplots(figsize=(16 / 3, 10 / 3))

    for config_key, config_value in level_results.items():
        # find config name to print
        if config2legend is not None:
            config_id_print = config2legend.get(config_key, config_key)
        else:
            config_id_print = config_key

        # determine x axis content
        if x_axis == "samples":
            x_range = config_value["dataset_sizes"]
            xlabel = "Number of labeled samples"
            x_formatter = ScalarFormatter()
        elif x_axis == "steps":
            x_range = list(range(len(config_value["dataset_sizes"])))
            xlabel = "Number of acquisition steps"
            x_formatter = ScalarFormatter()
        else:  # show_size == "percent"
            x_range = (
                np.array(config_value["dataset_sizes"])
                / config_value["max_ds_size"]
                * 100
            )
            xlabel = "Percentage of labeled samples"
            x_formatter = PercentFormatter(decimals=1)

        # replicate result if full_data
        if "full_data" in config_key:
            for diff_level in diff_levels:
                config_value[diff_level]["mean"] = np.repeat(
                    config_value[diff_level]["mean"], len(x_range)
                )
                config_value[diff_level]["stderr"] = np.repeat(
                    config_value[diff_level]["stderr"], len(x_range)
                )

        for diff_level in diff_levels:
            ax.plot(
                x_range,
                config_value[diff_level]["mean"],
                label=(diff_level if len(diff_levels) > 1 else config_id_print),
            )
            if stderr:
                ax.fill_between(
                    x_range,
                    config_value[diff_level]["mean"]
                    - config_value[diff_level]["stderr"],
                    config_value[diff_level]["mean"]
                    + config_value[diff_level]["stderr"],
                    alpha=0.2,
                )

    ax.legend(loc="upper right")
    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.xaxis.set_major_formatter(x_formatter)
    ax.grid(linestyle="dashed")
    # get ticks in sans-serif if sans-serif is used
    ax.xaxis.get_major_formatter()._usetex = False
    ax.yaxis.get_major_formatter()._usetex = False
    if save:
        plt.tight_layout()
        ensure_dir(os.path.dirname(savefig_kwargs["fname"]))
        plt.savefig(**savefig_kwargs)
    plt.show()


def plot_metric_vs_size(
    exp_name: str,
    metric: str,
    config_ids: list,
    run_id: Optional[int] = None,
    config2legend: Optional[dict] = None,
    metric2legend: Optional[dict] = None,
    stderr: bool = False,
    x_axis: Literal["steps", "samples", "percent"] = "samples",
    switch_axes: bool = False,
    save: bool = False,
    savefig_kwargs: Optional[dict] = None,
) -> None:
    """Plot metric as a function of the AL epochs or dataset size.

    Parameters
    ----------
    exp_name : str
        Experiment name
    metric : str
        Metric to plot
    config_ids : list
        List of config ids
    run_id : Optional[int]
        Run ID to plot. If None, all runs are averaged, by default None
    config2legend : dict, optional
        Dictionary to map config ids to legend names, by default None
    metric2legend : dict, optional
        Dictionary to map metric names to legend names, by default None
    stderr : bool, optional
        Whether to plot standard error bands, by default False
    x_axis : Literal["steps", "samples", "percent"], optional
        What to show on the x-axis, by default "samples"
    switch_axes : bool, optional
        Whether to switch x and y axes, by default False
    save : bool, optional
        Whether to save the plot to file, by default False
    savefig_kwargs : dict, optional
        Kwargs to save the figure, by default None.
    """
    # log
    logger.info(
        f"Computing {'all runs' if run_id is None else f'run {run_id}'} "
        f"of config_id {config_ids}"
    )

    # get plot inputs
    results = compute_metric_vs_size(
        experiment=exp_name, metric=metric, config_ids=config_ids, run_id=run_id
    )
    # pprint(results)

    _, ax = plt.subplots(figsize=(16 / 3, 10 / 3))
    clean_configs = True
    for config_key, config_value in results.items():

        # find config name to print
        if config2legend is not None:
            config_id_print = config2legend.get(config_key, config_key)
        else:
            clean_configs = False
            config_id_print = config_key

        # determine x axis content
        if x_axis == "samples":
            x_range = config_value["dataset_sizes"]
            xlabel = "Number of labeled samples"
            x_formatter = ScalarFormatter()
        elif x_axis == "steps":
            x_range = list(range(len(config_value["dataset_sizes"])))
            xlabel = "Number of acquisition steps"
            x_formatter = ScalarFormatter()
        else:  # show_size == "percent"
            x_range = (
                np.array(config_value["dataset_sizes"])
                / config_value["max_ds_size"]
                * 100
            )
            xlabel = "Percentage of labeled samples"
            x_formatter = PercentFormatter(decimals=1)

        # replicate result if full_data
        if "full_data" in config_key:
            config_value["mean"] = np.repeat(config_value["mean"], len(x_range))
            config_value["stderr"] = np.repeat(config_value["stderr"], len(x_range))

        if not switch_axes:
            ax.plot(x_range, config_value["mean"], label=config_id_print)
            if stderr:
                ax.fill_between(
                    x_range,
                    config_value["mean"] - config_value["stderr"],
                    config_value["mean"] + config_value["stderr"],
                    alpha=0.2,
                )
        else:
            ax.plot(config_value["mean"], x_range, label=config_id_print)
            if stderr:
                ax.fill_betweenx(
                    x_range,
                    config_value["mean"] - config_value["stderr"],
                    config_value["mean"] + config_value["stderr"],
                    alpha=0.2,
                )

    # find metric name to print
    if metric2legend is not None:
        metric_print = metric2legend.get(metric, metric)
    else:
        metric_print = metric

    if not switch_axes:
        ax.set(xlabel=xlabel, ylabel=metric_print)
        ax.xaxis.set_major_formatter(x_formatter)
    else:
        ax.set(ylabel=xlabel, xlabel=metric_print)
        ax.yaxis.set_major_formatter(x_formatter)
        ax.invert_xaxis()
        ax.invert_yaxis()
    ax.grid(linestyle="dashed")
    if clean_configs:
        ax.legend(loc="upper right")
    else:  # NOTE: legend under plot for long config names
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15))
    # get ticks in sans-serif if sans-serif is used
    ax.xaxis.get_major_formatter()._usetex = False
    ax.yaxis.get_major_formatter()._usetex = False
    if save:
        plt.tight_layout()
        ensure_dir(os.path.dirname(savefig_kwargs["fname"]))
        plt.savefig(**savefig_kwargs)

    plt.show()


def plot_active_gain(
    exp_name: str,
    metric: str,
    baseline: str,
    config_ids: list,
    run_id: Optional[int] = None,
    config2legend: Optional[dict] = None,
    metric2legend: Optional[dict] = None,
    x_axis: Literal["steps", "samples", "percent"] = "samples",
    save: bool = False,
    savefig_kwargs: Optional[dict] = None,
) -> None:
    """Plot active gain as a function of the AL epochs.

    Parameters
    ----------
    exp_name : str
        Experiment name
    metric : str
        Metric to plot
    baseline : str
        Baseline config id
    config_ids : list
        List of config ids
    run_id : Optional[int]
        Run ID to plot. If None, all runs are averaged, by default None
    config2legend : dict, optional
        Dictionary to map config ids to legend names, by default None
    metric2legend : dict, optional
        Dictionary to map metric names to legend names, by default None
    x_axis : Literal["steps", "samples", "percent"], optional
        What to show on the x-axis, by default "samples"
    save : bool, optional
        Whether to save the plot to file, by default False
    savefig_kwargs : dict, optional
        Kwargs to save the figure, by default None.
    """
    new_config_ids = config_ids.copy()
    if baseline not in config_ids:
        logger.warning("Baseline not in config_ids, adding it.")
        new_config_ids.append(baseline)
    # get plot inputs
    # get plot inputs
    results = compute_metric_vs_size(
        experiment=exp_name, metric=metric, config_ids=config_ids, run_id=run_id
    )
    # pprint(results)

    _, ax = plt.subplots(figsize=(16 / 3, 10 / 3))
    clean_configs = True
    for config_key, config_value in results.items():

        # find config name to print
        if config2legend is not None:
            config_id_print = config2legend.get(config_key, config_key)
        else:
            clean_configs = False
            config_id_print = config_key

        # determine x axis content
        if x_axis == "samples":
            x_range = config_value["dataset_sizes"]
            xlabel = "Number of labeled samples"
            x_formatter = ScalarFormatter()
        elif x_axis == "steps":
            x_range = list(range(len(config_value["dataset_sizes"])))
            xlabel = "Number of acquisition steps"
            x_formatter = ScalarFormatter()
        else:  # show_size == "percent"
            x_range = (
                np.array(config_value["dataset_sizes"])
                / config_value["max_ds_size"]
                * 100
            )
            xlabel = "Percentage of labeled samples"
            x_formatter = PercentFormatter(decimals=1)

        # replicate baseline result if full_data
        if "full_data" in baseline:
            baseline_mean = np.repeat(results[baseline]["mean"], len(x_range))
        else:
            baseline_mean = results[baseline]["mean"]
        diff_mean = baseline_mean - config_value["mean"]
        ax.plot(x_range, diff_mean, label=config_id_print)

    # find metric name to print
    if metric2legend is not None:
        metric_print = metric2legend.get(metric, metric)
    else:
        metric_print = metric

    ax.set(xlabel=xlabel, ylabel=f"{metric_print} - Active gain")
    ax.grid(linestyle="dashed")
    ax.xaxis.set_major_formatter(x_formatter)
    # ax.legend()
    if clean_configs:
        ax.legend(loc="lower left")
    else:  # NOTE: legend under plot for long config names
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15))
    # get ticks in sans-serif if sans-serif is used
    ax.xaxis.get_major_formatter()._usetex = False
    ax.yaxis.get_major_formatter()._usetex = False
    if save:
        plt.tight_layout()
        ensure_dir(os.path.dirname(savefig_kwargs["fname"]))
        plt.savefig(**savefig_kwargs)

    plt.show()


def plot_history(all_logs: list[dict[str, float]], metric: str) -> None:
    """Plot metric and loss in function of number of epochs.

    Parameters
    ----------
    all_logs : list[dict[str, float]]
        List of dictionaries containing the training logs for each epoch.
    metric : str
        Metric to plot (in addition to loss).
    """

    def _imitate_parsed_metric_name(metric: str) -> str:
        """Imitate parsed metric name from `transformers.modelcard.parse_log_history`.

        Parameters
        ----------
        metric : str
            Metric name

        Returns
        -------
        str
            Parsed metric name
        """
        splits = metric.split("_")
        name = " ".join([part.capitalize() for part in splits[1:]])
        return name

    epochs_arr = [log_epoch["Epoch"] for log_epoch in all_logs]
    train_loss_arr = [log_epoch["Training Loss"] for log_epoch in all_logs]
    train_loss_arr = [  # NOTE: replace "No log" with None for plotting
        None if train_loss == "No log" else train_loss for train_loss in train_loss_arr
    ]
    val_loss_arr = [log_epoch["Validation Loss"] for log_epoch in all_logs]

    parsed_metric = _imitate_parsed_metric_name(metric)
    metric_arr = [log_epoch[parsed_metric] for log_epoch in all_logs]

    plt.figure(figsize=(12, 5))

    # Metric
    plt.subplot(1, 2, 1)
    plt.plot(epochs_arr, metric_arr, color="tab:orange")
    # plt.ylim(0, 1)
    plt.title(parsed_metric)
    plt.ylabel(parsed_metric)
    plt.xlabel("Epochs")
    plt.legend(["valid"], loc="upper right")
    plt.grid(linestyle="dashed")

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_arr, train_loss_arr)
    plt.plot(epochs_arr, val_loss_arr)
    plt.title("Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.legend(["train", "valid"], loc="upper right")
    plt.grid(linestyle="dashed")

    plt.show()


def plot_violinplot_racepp(
    pred_label: tuple,
    label_map: dict,
    save: bool = False,
    savefig_kwargs: Optional[dict] = None,
) -> None:
    """Plot violinplot for RACE++ dataset.

    Parameters
    ----------
    pred_label : tuple
        Tuple of predictions and labels
    label_map : dict
        Mapping integer labels to string labels
    save : bool, optional
        Whether to save plot, by default False
    savefig_kwargs : Optional[dict], optional
        Kwargs to save the figure, by default None
    """
    # NOTE: make function more flexible if want to do ARC dataset
    predictions, labels = pred_label

    # difficulty levels int
    diff_levels = list(label_map.keys())
    data = []
    for diff_level in diff_levels:
        data.append(predictions[labels == diff_level])

    m, b = np.polyfit(labels, predictions, 1)

    _, ax = plt.subplots(figsize=(8, 8))
    sns.violinplot(data, color="#c41331", alpha=0.25)
    ax.plot([-0.5, 2.5], [0.5, 0.5], c="k", alpha=0.25)
    ax.plot([-0.5, 2.5], [1.5, 1.5], c="k", alpha=0.25)
    ax.set_xlabel("Difficulty")
    ax.set_ylabel("Predicted difficulty")
    ax.set_xticks(diff_levels)
    if m and b:
        x0, x1 = -0.5, 2.5
        ax.plot([x0, x1], [x0 * m + b, x1 * m + b], c="#c41331", label="linear fit")
        ax.plot([x0, x1], [x0, x1], "--", c="darkred", label="ideal")
    ax.legend()
    if save:
        plt.tight_layout()
        ensure_dir(os.path.dirname(savefig_kwargs["fname"]))
        plt.savefig(**savefig_kwargs)
    plt.show()


def plot_pred_parity(
    experiment: str,
    metric: str,
    config_ids: List[str],
    config_dict: dict,
    label_map: dict,
    run_id: Optional[int] = None,
    x_axis: Literal["steps", "samples", "percent"] = "samples",
    stderr: bool = False,
    config2legend: Optional[dict] = None,
    metric2legend: Optional[dict] = None,
    save: bool = False,
    savefig_kwargs: Optional[dict] = None,
):
    """Plot predictive parity.

    Parameters
    ----------
    experiment : str
        Experiment name
    metric : str
        Metric name
    config_ids : List[str]
        List of config IDs
    config_dict : dict
        Dict of config objects
    label_map : dict
        Mapping integer labels to string labels
    run_id : Optional[int], optional
        Run ID to plot. If None, all runs are averaged, by default None
    x_axis : Literal["steps", "samples", "percent"], optional
        X-axis content, by default "samples"
    stderr : bool, optional
        Whether to plot standard error region, by default False
    config2legend : dict, optional
        Dictionary to map config ids to legend names, by default None
    metric2legend : dict, optional
        Dictionary to map metric names to legend names, by default None
    save : bool, optional
        Whether to save the plot to file, by default False
    savefig_kwargs : dict, optional
        Kwargs to save the figure, by default None.
    """
    # get results
    level_results = compute_level_performance(
        experiment,
        metric.replace("test_", ""),
        config_ids,
        label_map=label_map,
        config_dict=config_dict,
        run_id=run_id,
    )

    # find metric name to print
    if metric2legend is not None:
        metric_print = metric2legend.get(metric, metric)
    else:
        metric_print = metric

    # get ylabel
    ylabel = f"Predictive parity ({metric_print})"

    # log
    logger.info(f"Plotting {ylabel}, for config_ids: {config_ids}")

    _, ax = plt.subplots(figsize=(16 / 3, 10 / 3))

    for config_key, config_value in level_results.items():
        # find config name to print
        if config2legend is not None:
            config_id_print = config2legend.get(config_key, config_key)
        else:
            config_id_print = config_key

        # determine x axis content
        if x_axis == "samples":
            x_range = config_value["dataset_sizes"]
            xlabel = "Number of labeled samples"
            x_formatter = ScalarFormatter()
        elif x_axis == "steps":
            x_range = list(range(len(config_value["dataset_sizes"])))
            xlabel = "Number of acquisition steps"
            x_formatter = ScalarFormatter()
        else:  # show_size == "percent"
            x_range = (
                np.array(config_value["dataset_sizes"])
                / config_value["max_ds_size"]
                * 100
            )
            xlabel = "Percentage of labeled samples"
            x_formatter = PercentFormatter(decimals=1)

        # replicate result if full_data
        if "full_data" in config_key:
            config_value["pred_parity"]["mean"] = np.repeat(
                config_value["pred_parity"]["mean"], len(x_range)
            )
            config_value["pred_parity"]["stderr"] = np.repeat(
                config_value["pred_parity"]["stderr"], len(x_range)
            )

        ax.plot(
            x_range,
            config_value["pred_parity"]["mean"],
            label=config_id_print,
        )
        if stderr:
            ax.fill_between(
                x_range,
                config_value["pred_parity"]["mean"]
                - config_value["pred_parity"]["stderr"],
                config_value["pred_parity"]["mean"]
                + config_value["pred_parity"]["stderr"],
                alpha=0.2,
            )

    ax.legend(loc="upper right")
    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.xaxis.set_major_formatter(x_formatter)
    ax.grid(linestyle="dashed")
    # get ticks in sans-serif if sans-serif is used
    ax.xaxis.get_major_formatter()._usetex = False
    ax.yaxis.get_major_formatter()._usetex = False
    if save:
        plt.tight_layout()
        ensure_dir(os.path.dirname(savefig_kwargs["fname"]))
        plt.savefig(**savefig_kwargs)
    plt.show()
