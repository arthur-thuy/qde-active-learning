"""Utils file."""

# standard library imports
import os
import random
import time
from pathlib import Path
from typing import Optional, Union

# related third party imports
import numpy as np
import structlog
import torch
from tabulate import tabulate
from torch.backends import cudnn

# local application/library specific imports
# /

# set up logger
logger = structlog.get_logger("qdet")


def ensure_dir(dirname: Union[Path, str]) -> None:
    """Ensure directory exists.

    Parameters
    ----------
    dirname : Union[Path, str]
        Directory to check/create.
    """
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def get_save_path(save_dir: str, fname: str) -> str:
    """Get save path.

    Parameters
    ----------
    save_dir : str
        Directory to save to
    fname : str
        Filename to save to

    Returns
    -------
    str
        Full path to save to
    """
    ensure_dir(save_dir)
    fpath = os.path.join(save_dir, f"{fname}.pth")
    return fpath


def save_checkpoint(state: dict, save_dir: str, fname: str) -> None:
    """Save checkpoint.

    Parameters
    ----------
    state : dict
        Dict to save
    save_dir : str
        Directory to save to
    fname : str
        Filename to save to
    """
    fpath = get_save_path(save_dir, fname)
    torch.save(state, fpath)


def set_device(no_cuda: bool, no_mps: bool) -> torch.device:
    """Set device to run operations on.

    Parameters
    ----------
    no_cuda : bool
        Whether to disable GPU training.
    no_mps : bool
        Whether to disable high-performance training on GPU for MacOS devices.
    Returns
    -------
    torch.device
        Device to run operations on.
    """
    use_cuda = not no_cuda and torch.cuda.is_available()
    use_mps = not no_mps and torch.backends.mps.is_available()
    if use_cuda:
        device = torch.device("cuda")
        logger.info(
            f"GPU name: {torch.cuda.get_device_name(torch.cuda.current_device())}"
        )
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        logger.warning("using CPU, this will be slow")
    return device


def set_seed(seed: int) -> None:
    """Set seed for reproducibility.

    Parameters
    ----------
    seed : int
        Seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.info(f"Set seed ({seed})")


def print_elapsed_time(start_time: float, exp: int, acq_func: str) -> None:
    """Print elapsed time for each experiment of acquiring.

    Parameters
    ----------
    start_time : float
        Starting time (in time.time())
    exp : int
        Experiment iteration
    acq_func : str
        Name of acquisition function
    """
    elp = time.time() - start_time
    print(
        (
            f"Run {exp} finished ({acq_func}): "
            f"{int(elp//3600)}:{int(elp%3600//60)}:{int(elp%60)}"
        )
    )


def get_init_active_idx(
    dataset,
    init_size: int,
    num_classes: Optional[int] = None,
    balanced: bool = False,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Get initial active indices.

    Parameters
    ----------
    dataset :
        HF dataset
    init_size : int
        Initial size
    num_classes : int, optional
        Number of classes
    balanced : bool, optional
        Whether to have balanced classes in the initial active set, by default False

    Returns
    -------
    np.ndarray
        Initial active indices
    """
    if seed is not None:
        set_seed(seed)

    targets = np.array(dataset["label"]).astype(int)

    # NOTE: assume regression task if num_classes=1
    num_classes = (
        num_classes if num_classes > 1 else np.unique(targets, axis=0).shape[0]
    )

    if balanced:
        # find observations per class (need exactly init_size samples in total)
        obs_per_class = {i: init_size // num_classes for i in range(num_classes)}
        if init_size % num_classes != 0:
            extra_idx = np.random.choice(
                np.arange(num_classes), size=(init_size % num_classes), replace=False
            )
            for i in extra_idx:
                obs_per_class[i] += 1

        # sample indices
        initial_idx = np.array([], dtype=int)
        for i in range(num_classes):
            idx = np.random.choice(
                np.where(targets == i)[0], size=obs_per_class[i], replace=False
            )
            initial_idx = np.concatenate((initial_idx, idx))
    else:
        # sample indices
        initial_idx = np.random.choice(targets.shape[0], size=init_size, replace=False)

    print(f"Initial labeled set size: \t\t{initial_idx.shape[0]}")

    # astype(int) is necessary for np.bincount to work because cannot work with floats
    label_props = (
        np.bincount(targets[initial_idx].astype(int), minlength=num_classes)
        / targets[initial_idx].shape[0]
    )
    label_str = np.unique(targets[initial_idx], axis=0)
    if hasattr(dataset.features["label"], "names"):
        label_str = dataset.features["label"].names
    prop_table = tabulate(
        np.column_stack((label_str, label_props * 100)),
        headers=["Label", "Percentage"],
        tablefmt="github",
        floatfmt=".2f",
    )
    print(f"Initial labeled set distribution: \n{prop_table}")
    return initial_idx
