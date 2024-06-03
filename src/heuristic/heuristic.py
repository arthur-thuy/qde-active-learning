"""Active learning heuristics."""

# standard library imports
# /

# related third party imports
import numpy as np
from baal.active.heuristics import (
    BALD,
    AbstractHeuristic,
    Variance,
    require_single_item,
)
from baal.active.heuristics.stochastics import PowerSampling
from yacs.config import CfgNode

# local application/library specific imports
from .build import AL_REGISTRY


@AL_REGISTRY.register("powerbald")
def build_powerbald(al_cfg: CfgNode) -> AbstractHeuristic:
    """Build PowerBALD heuristic.

    Parameters
    ----------
    al_cfg : CfgNode
        Config file

    Returns
    -------
    AbstractHeuristic
        Heuristic
    """
    return PowerSampling(
        BALD(), query_size=al_cfg.QUERY_SIZE, temperature=al_cfg.TEMPERATURE
    )


@AL_REGISTRY.register("powervariance")
def build_powervariance(al_cfg: CfgNode) -> AbstractHeuristic:
    """Build PowerVariance heuristic.

    Parameters
    ----------
    al_cfg : CfgNode
        Config file

    Returns
    -------
    AbstractHeuristic
        Heuristic
    """
    return PowerSampling(
        Variance(), query_size=al_cfg.QUERY_SIZE, temperature=al_cfg.TEMPERATURE
    )


@AL_REGISTRY.register("variance")
def build_variance(al_cfg: CfgNode) -> AbstractHeuristic:  # noqa
    """Build Variance heuristic.

    Parameters
    ----------
    al_cfg : CfgNode
        Config file

    Returns
    -------
    AbstractHeuristic
        Heuristic
    """
    return Variance()


@AL_REGISTRY.register("powergaussentropy")
def build_powergaussentropy(al_cfg: CfgNode) -> AbstractHeuristic:
    """Build PowerGaussEntropy heuristic.

    Parameters
    ----------
    al_cfg : CfgNode
        Config file

    Returns
    -------
    AbstractHeuristic
        Heuristic
    """
    return PowerSampling(
        GaussEntropy(), query_size=al_cfg.QUERY_SIZE, temperature=al_cfg.TEMPERATURE
    )


class GaussEntropy(AbstractHeuristic):
    """Sort by the highest entropy for Gaussian distribution."""

    def __init__(self, reduction: str = "mean"):
        """Sort by the highest entropy for Gaussian distribution.

        Parameters
        ----------
        reduction : str, optional
            function that aggregates the results, by default "mean"
        """
        _help = "Need to reduce the output from [n_sample, n_class] to [n_sample]"
        assert reduction != "none", _help
        super().__init__(shuffle_prop=0.0, reverse=True, reduction=reduction)

    @require_single_item
    def compute_score(self, predictions):
        """Compute acquisition scores.

        Parameters
        ----------
        predictions : _type_
            Predictions (3-dimensional)

        Returns
        -------
        _type_
            Acquisition scores
        """
        assert predictions.ndim >= 3
        return (1 / 2) * np.log(2 * np.pi * np.e * np.var(predictions, -1))
