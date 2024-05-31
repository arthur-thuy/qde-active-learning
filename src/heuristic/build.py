"""Build file for active learning."""

from baal.active import get_heuristic
from baal.active.heuristics import AbstractHeuristic
from yacs.config import CfgNode

from tools.registry import Registry

AL_REGISTRY = Registry()


def build_heuristic(al_cfg: CfgNode) -> AbstractHeuristic:
    """Build heuristic from config file.

    Parameters
    ----------
    al_cfg : CfgNode
        Active learning config file

    Returns
    -------
    AbstractHeuristic
        Heuristic
    """
    # print(f"AL_REGISTRY: {AL_REGISTRY}")
    try:
        heuristic = AL_REGISTRY[al_cfg.HEURISTIC](al_cfg)
    except KeyError:
        heuristic = get_heuristic(al_cfg.HEURISTIC)
    return heuristic
