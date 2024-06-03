"""Build file for data loader."""

# standard library imports
# /

# related third party imports
from yacs.config import CfgNode

# local application/library specific imports
from data_loader.data_loader import QDET


def build_hf_dataset(loader_cfg: CfgNode, num_classes: int, seed: int):
    """Build the HuggingFace dataset.

    Parameters
    ----------
    loader_cfg : CfgNode
        Data loader config object
    num_classes : int
        Number of classes
    seed : int
        Random seed

    Returns
    -------
    _type_
        Train/Val/Test datasets
    """
    loader = QDET(
        name=loader_cfg.NAME,
        num_classes=num_classes,
        regression=loader_cfg.REGRESSION,
        small_dev=loader_cfg.SMALL_DEV,
        seed=seed,
    )
    return loader.load_all()
