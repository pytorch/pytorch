"""
Configuration classes for checkpointer construction.

This module provides configuration dataclasses that consolidate all
configuration options needed to construct checkpointers.
"""

from dataclasses import dataclass, field

from .barriers import BarrierConfig
from .checkpoint_process import CheckpointProcessConfig
from .checkpoint_writer import CheckpointWriterConfig
from .staging import CheckpointStagerConfig


@dataclass
class CheckpointerConfig:
    """
    Configuration class for checkpointer construction.

    This class consolidates the core component configuration options needed to construct
    a checkpointer, providing a clean separation of concerns where each component
    manages its own configuration.

    Attributes:
        writer_config: Configuration options for the checkpoint writer component.
        barrier_config: Configuration for barrier construction and arguments.
        staging_config: Configuration options for the async staging component.
        process_config: Configuration options for the async checkpoint process component.

    """

    writer_config: CheckpointWriterConfig = field(
        default_factory=CheckpointWriterConfig
    )
    barrier_config: BarrierConfig = field(default_factory=BarrierConfig)

    # Below configs are used for async checkpointing
    staging_config: CheckpointStagerConfig = field(
        default_factory=CheckpointStagerConfig
    )
    process_config: CheckpointProcessConfig = field(
        default_factory=CheckpointProcessConfig
    )
