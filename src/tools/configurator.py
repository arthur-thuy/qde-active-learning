"""File with configurator functionalities."""

# standard library imports
import importlib
import os
from pathlib import Path
from typing import Any, Union

# related third party imports
import structlog
import yaml
from yacs.config import CfgNode

# local application/library specific imports
from tools.utils import ensure_dir

# set up logger
logger = structlog.get_logger("qdet")


def save_config(cfg: CfgNode, save_dir: str, fname: str) -> None:
    """Save config.

    Parameters
    ----------
    cfg : CfgNode
        Config object
    save_dir : str
        Directory to save to
    fname : str
        Filename to save to
    """
    ensure_dir(os.path.join(save_dir, "config"))
    fpath = os.path.join(save_dir, "config", f"{fname}.yaml")
    cfg.dump(stream=open(fpath, "w", encoding="utf-8"))


def _merge_config(config_path: Union[str, Path]) -> CfgNode:
    """Load single config file from path.

    Parameters
    ----------
    config_path : Union[str, Path]
        Config file path.

    Returns
    -------
    CfgNode
        Config object.
    """
    logger.info(f"Loading config from '{config_path}'")
    exp_name = os.path.basename(os.path.dirname(config_path))
    config_exp = importlib.import_module(f"config.{exp_name}.config")
    cfg = config_exp.get_cfg_defaults()
    cfg.merge_from_file(config_path)
    return cfg


def _add_derived_configs(
    cfg: CfgNode, config_dir: Union[str, Path], freeze: bool = True
) -> CfgNode:
    """Add derived config variables at runtime.

    Parameters
    ----------
    cfg : CfgNode
        Config object.
    config_dir : Union[str, Path]
        Config directory.
    freeze : bool, optional
        Whether to freeze config object, by default True

    Returns
    -------
    CfgNode
        Config object with derived variables.
    """
    # add derived config variables at runtime
    cfg.ID = create_config_id(cfg)
    cfg.OUTPUT_DIR = f"./output/{config_dir}"
    if freeze:
        cfg.freeze()
    return cfg


def load_configs(fpath: str, freeze: bool = True) -> tuple[CfgNode, ...]:
    """Load one or more config files from path.

    Parameters
    ----------
    fpath : str
        Path to config file or directory.
    freeze : bool, optional
        Whether to freeze config objects, by default True

    Returns
    -------
    tuple[CfgNode, ...]
        Tuple of config objects.

    Raises
    ------
    ValueError
        When fpath is not a valid directory.
    """
    # check if path is valid directory
    config_path_full = Path(os.path.join("config", fpath))
    if not (config_path_full.exists() and config_path_full.is_dir()):
        raise ValueError(f"Invalid config dirname (base): {config_path_full}")
    config_paths = list(config_path_full.glob("*.yaml"))
    config_dir_base = fpath
    # load config files
    configs = tuple([_merge_config(config_path) for config_path in config_paths])
    # add derived config variables
    configs = tuple(
        [_add_derived_configs(config, config_dir_base, freeze) for config in configs]
    )
    return configs


def create_config_id(cfg: CfgNode) -> str:
    """Create identifier for config.

    Parameters
    ----------
    cfg : CfgNode
        Config object.

    Returns
    -------
    str
        Config identifier.
    """
    if cfg.AL.FULL_DATA:
        al_suffix = "full_data"
    else:
        al_suffix = (
            f"{cfg.AL.HEURISTIC}-{'MCD' if cfg.MODEL.MC_DROPOUT else 'STD'}-"
            f"N{cfg.AL.EPOCHS}-Q{cfg.AL.QUERY_SIZE}-I{cfg.AL.INIT_ACTIVE_SIZE}-S{cfg.AL.SUBSET_POOL}-B{cfg.AL.INIT_ACTIVE_BALANCED}"
        )
    cfg_id = (
        f"{cfg.MODEL.NAME}-{'regr' if cfg.LOADER.REGRESSION else 'class'}-" + al_suffix
    )
    return cfg_id


def get_configs_out(experiment: str) -> tuple[dict[str, Any], ...]:
    """Get configs from output directory.

    Parameters
    ----------
    experiment : str
        Experiment name

    Returns
    -------
    Tuple[Dict[str, Any], ...]
        Tuple of config dicts
    """
    config_dir = Path(os.path.join("output", experiment, "config"))
    config_paths = list(config_dir.glob("*.yaml"))

    def _read_yaml_config(config_path: Union[str, Path]) -> dict:
        with open(config_path, "r", encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
        return config

    configs = tuple([_read_yaml_config(config_path) for config_path in config_paths])
    return configs


def get_config_ids(configs: tuple[dict]) -> list[Any]:
    """Get config IDs from configs.

    Parameters
    ----------
    configs : Tuple[dict]
        Tuple of config dicts

    Returns
    -------
    List[Any]
        Tuple of config IDs
    """
    config_ids = [cfg["ID"] for cfg in configs]
    return config_ids


def check_cfg(cfg: CfgNode) -> None:
    """Check config for logical errors.

    Parameters
    ----------
    cfg : CfgNode
        Config object.

    Raises
    ------
    ValueError
        If error in values of EVAL.MC_SAMPLES, MODEL.MC_DROPOUT, and AL.HEURISTIC
    ValueError
        If error in values of LOADER.VAL_SET, TRAIN.EARLY_STOPPING, and TRAIN.PATIENCE
    """
    if cfg.LOADER.REGRESSION and cfg.MODEL.NUM_LABELS > 1:
        raise ValueError(
            f"Regression is set to True, but num_labels is {cfg.MODEL.NUM_LABELS}."
        )
    if not cfg.LOADER.REGRESSION and cfg.MODEL.NUM_LABELS == 1:
        raise ValueError(
            f"Regression is set to False, but num_labels is {cfg.MODEL.NUM_LABELS}."
        )
    if cfg.EVAL.MC_SAMPLES > 1 and not cfg.MODEL.MC_DROPOUT:
        raise ValueError(
            "MC Dropout is disabled, but number of MC samples is larger than 1."
        )
    if cfg.AL.HEURISTIC == "random" and cfg.MODEL.MC_DROPOUT:
        raise ValueError(
            "MC Dropout is enabled, but heuristic is set to random sampling."
        )
    if cfg.MODEL.MC_DROPOUT and cfg.EVAL.MC_SAMPLES == 1:
        raise ValueError("MC Dropout is enabled, but number of MC samples is set to 1.")
    if cfg.TRAIN.EARLY_STOPPING:
        if cfg.TRAIN.PATIENCE is None:
            raise ValueError(
                "Early stopping is enabled, but patience for early stopping is not set."
            )

    logger.info("Checks completed")


def check_dataset_info(configs: tuple[CfgNode, ...]) -> tuple[str, bool, bool]:
    """Check if all configs have the same dataset name and init active balanced.

    Parameters
    ----------
    configs : tuple[CfgNode, ...]
        Tuple of config objects

    Returns
    -------
    tuple[str, bool]
        Dataset name and init active balanced
    """
    names = []
    for cfg in configs:
        names.append(cfg.LOADER.NAME)

    def all_same(items: list) -> bool:
        """Check if all items in list are the same.

        Parameters
        ----------
        items : list
            List to check

        Returns
        -------
        bool
            If yes, all items in list are the same
        """
        return all(x == items[0] for x in items)

    assert all_same(names), "All configs should have the same LOADER.NAME value"

    return names[0]
