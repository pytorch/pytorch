"""
Factory functions for creating checkpointer instances with sensible defaults.

This module provides high-level factory functions that simplify the creation
of checkpointer instances by automatically handling component initialization
and configuration with reasonable defaults.
"""

from collections.abc import Callable
from typing import Any

import torch.distributed as dist

from .barriers import create_barrier_from_config
from .checkpoint_process import CheckpointProcess
from .checkpoint_reader import CheckpointReader
from .checkpoint_writer import CheckpointWriter, CheckpointWriterConfig, WriterHook
from .checkpointer import AsyncCheckpointer, SyncCheckpointer
from .config import CheckpointerConfig
from .staging import DefaultStager
from .types import RankInfo


def _get_default_rank_info() -> RankInfo:
    """
    Get default rank information from the current distributed environment.

    Returns:
        RankInfo: Rank information from the default process group if initialized,
                 otherwise single-rank fallback.
    """
    if dist.is_initialized():
        return RankInfo(
            global_world_size=dist.get_world_size(),
            global_rank=dist.get_rank(),
        )
    else:
        # Single-rank fallback
        return RankInfo(global_world_size=1, global_rank=0)


def default_subprocess_init_fn(*_: Any) -> None:
    """Default subprocess initialization function (no-op)."""


def default_writer_init_fn(rank_info: RankInfo) -> CheckpointWriter:
    """Default checkpoint writer initialization function."""
    return CheckpointWriter(
        config=CheckpointWriterConfig(),
        rank_info=rank_info,
    )


def make_sync_checkpointer(
    config: CheckpointerConfig = CheckpointerConfig(),
    rank_info: RankInfo | None = None,
    commit_hook: WriterHook | None = None,
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
        from .barriers import BarrierConfig
        config = CheckpointerConfig(barrier_config=BarrierConfig(barrier_type=None))
        checkpointer = make_sync_checkpointer(config=config)
    """
    if rank_info is None:
        rank_info = _get_default_rank_info()

    reader = CheckpointReader(
        rank_info=rank_info,
    )

    barrier = create_barrier_from_config(config.barrier_config)

    writer = CheckpointWriter(
        config=config.writer_config,
        rank_info=rank_info,
        barrier=barrier,
        commit_hook=commit_hook,
    )

    return SyncCheckpointer(
        writer=writer,
        reader=reader,
    )


def make_async_checkpointer(
    config: CheckpointerConfig = CheckpointerConfig(),
    rank_info: RankInfo | None = None,
    subprocess_init_fn: Callable[..., None] = default_subprocess_init_fn,
    subprocess_init_args: tuple[Any, ...] = (),
    checkpoint_writer_init_fn: Callable[..., CheckpointWriter] = default_writer_init_fn,
    checkpoint_writer_init_args: dict[str, Any] | None = None,
) -> AsyncCheckpointer:
    """
    Factory function to create an AsyncCheckpointer instance with sensible defaults.

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

    checkpoint_stager = DefaultStager(
        config=config.staging_config,
    )

    checkpoint_writer_init_args = checkpoint_writer_init_args or {}

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
