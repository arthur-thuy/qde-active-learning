"""Execute at import time, needed for decorators to work."""

from .build import AL_REGISTRY, build_heuristic  # noqa: F401
from .heuristic import build_powerbald, build_powervariance  # noqa: F401
