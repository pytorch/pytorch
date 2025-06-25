"""
Builder functions for constructing checkpointers.

This module provides factory functions for building checkpointers with various
configurations and sensible defaults.
"""

from typing import Any, Callable, Optional

import torch.distributed as dist
from torch.distributed.checkpoint._experimental.barriers import (
    BarrierConfig,
    create_barrier_from_config,
    DistBarrier,
)

from .checkpoint_process import CheckpointProcess
from .checkpoint_reader import CheckpointReader
from .checkpoint_writer import CheckpointWriter, CheckpointWriterConfig, WriterHook
from .checkpointer import AsyncCheckpointer, SyncCheckpointer
from .config import CheckpointerConfig
from .staging import DefaultStager
from .types import RankInfo


def _get_default_rank_info() -> RankInfo:
    """
    Construct RankInfo from the default process group if it's initialized.

    Returns:
        RankInfo: Either from distributed process group or single-rank fallback.
    """
    if dist.is_initialized():
        return RankInfo(
            global_world_size=dist.get_world_size(),
            global_rank=dist.get_rank(),
        )
    else:
        # Single-rank fallback
        return RankInfo(global_world_size=1, global_rank=0)


def make_sync_checkpointer(
    config: CheckpointerConfig = CheckpointerConfig(),
    rank_info: Optional[RankInfo] = None,
    use_dist_barrier: bool = True,
    commit_hook: Optional[WriterHook] = None,
) -> SyncCheckpointer:
    """
    Factory function to create a SyncCheckpointer instance with sensible defaults.

    This function creates a synchronous checkpointer with default components, automatically
    detecting rank information from the default process group if available, and using the
    provided component configurations.

    Args:
        config: CheckpointerConfig containing component-specific configurations
               (writer_config, staging_config, process_config). Defaults to CheckpointerConfig().
        rank_info: RankInfo for distributed training. Defaults to auto-detection from
                  the default PyTorch distributed process group if initialized, otherwise
                  falls back to single-rank (world_size=1, rank=0).
        use_dist_barrier: Whether to use a distributed barrier for synchronization.
                         If True, a DistBarrier will be used for synchronization.
                         If False, no barrier will be used.
        commit_hook: Optional hook for custom actions before and after checkpoint commits.

    Returns:
        SyncCheckpointer: A configured synchronous checkpointer instance.

    Examples:
        # Simplest usage - auto-detect rank, default config
        checkpointer = make_sync_checkpointer()

        # Explicit rank configuration
        checkpointer = make_sync_checkpointer(
            rank_info=RankInfo(global_world_size=4, global_rank=0)
        )

        # Disable barrier
        checkpointer = make_sync_checkpointer(use_dist_barrier=False)
    """
    if rank_info is None:
        rank_info = _get_default_rank_info()

    reader = CheckpointReader(
        rank_info=rank_info,
    )

    writer = CheckpointWriter(
        config=config.writer_config,
        rank_info=rank_info,
        barrier=DistBarrier() if use_dist_barrier else None,
        commit_hook=commit_hook,
    )

    return SyncCheckpointer(
        reader=reader,
        writer=writer,
    )


def default_subprocess_init_fn(_: Any = None) -> None:
    """Default no-op initialization function for subprocess."""


def default_writer_init_fn(
    rank_info: RankInfo,
    config: Optional[CheckpointWriterConfig] = None,
    barrier_config: Optional[BarrierConfig] = None,
) -> CheckpointWriter:
    """
    Default function to create a CheckpointWriter instance.

    Args:
        rank_info: Information about the current rank in distributed training.
                  This is automatically provided by the checkpoint process.
        config: Optional configuration for the checkpoint writer.
               If None, uses default CheckpointWriterConfig.
        barrier_config: Optional barrier configuration.
                       If None, no barrier will be used.

    Returns:
        CheckpointWriter: A configured checkpoint writer instance.
    """
    if config is None:
        config = CheckpointWriterConfig()

    # Create barrier from config if provided
    barrier = create_barrier_from_config(barrier_config) if barrier_config else None

    return CheckpointWriter(
        config=config,
        rank_info=rank_info,
        barrier=barrier,
        commit_hook=None,
    )


def make_async_checkpointer(
    config: CheckpointerConfig = CheckpointerConfig(),
    rank_info: Optional[RankInfo] = None,
    subprocess_init_fn: Callable[[Any], None] = default_subprocess_init_fn,
    subprocess_init_args: tuple[Any, ...] = (),
    checkpoint_writer_init_fn: Callable[..., CheckpointWriter] = default_writer_init_fn,
    checkpoint_writer_init_args: dict[str, Any] = {},
) -> AsyncCheckpointer:
    """
    Factory function to create an AsyncCheckpointer instance with the provided config.

    This function creates an asynchronous checkpointer using the provided configuration,
    automatically detecting rank information if not provided.

    Args:
        config: CheckpointerConfig containing component-specific configurations.
        rank_info: RankInfo for distributed training. Defaults to auto-detection.
        subprocess_init_fn: Function to initialize the subprocess. Defaults to no-op.
        subprocess_init_args: Arguments to pass to subprocess_init_fn.
        checkpoint_writer_init_fn: Function to create CheckpointWriter instance.
        checkpoint_writer_init_args: Arguments to pass to checkpoint_writer_init_fn.

    Returns:
        AsyncCheckpointer: A configured asynchronous checkpointer instance.

    Examples:
        # Create with default config
        checkpointer = make_async_checkpointer()

        # Create with custom init functions
        checkpointer = make_async_checkpointer(
            subprocess_init_fn=my_subprocess_init_fn,
            checkpoint_writer_init_fn=my_writer_init_fn
        )
    """
    if rank_info is None:
        rank_info = _get_default_rank_info()

    reader = CheckpointReader(
        rank_info=rank_info,
    )

    checkpoint_stager = DefaultStager(config=config.staging_config)
    checkpoint_process = CheckpointProcess(
        rank_info=rank_info,
        config=config.process_config,
        subprocess_init_fn=subprocess_init_fn,
        subprocess_init_args=subprocess_init_args,
        checkpoint_writer_init_fn=checkpoint_writer_init_fn,
        checkpoint_writer_init_args=checkpoint_writer_init_args,
    )

    return AsyncCheckpointer(
        checkpoint_stager=checkpoint_stager,
        checkpoint_process=checkpoint_process,
        reader=reader,
    )
