from enum import Enum


__all__ = ["Collectives"]


class Collectives(Enum):
    """Enum for collectives usage during checkpoint save."""

    ALL = "all"
    PLANNING_ONLY = "planning_only"
    METADATA_ONLY = "metadata_only"
    NONE = "none"
