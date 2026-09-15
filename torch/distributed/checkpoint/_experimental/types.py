"""
Type definitions for distributed training and checkpointing.

This module provides type definitions and classes for managing rank information
in distributed training environments, which is essential for proper checkpoint
saving and loading.
"""

from dataclasses import dataclass
from typing import Any
from typing_extensions import TypeAliasType


# Use TypeAliasType so that __module__ is set correctly for public API reexports
# See: https://github.com/pytorch/pytorch/issues/171905
STATE_DICT = TypeAliasType("STATE_DICT", dict[str, Any])


@dataclass
class RankInfo:
    """
    Information about the current rank in a distributed training environment.

    Attributes:
        global_rank: The global rank ID of the current process.
        global_world_size: The total number of processes in the distributed environment.
    """

    global_rank: int
    global_world_size: int
