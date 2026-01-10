"""
Checkpoint writer functionality for machine learning models.

This module provides classes for writing checkpoints to storage, including
determining checkpoint layout, configuring the writer, and defining hooks
for custom actions during the checkpoint writing process.
"""

import abc
import logging
import os
from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from .barriers import Barrier
from .types import RankInfo, STATE_DICT


logger = logging.getLogger(__name__)


class WriterHook(abc.ABC):
    """
    Abstract base class for checkpoint commit hooks.

    A commit hook provides callbacks that are executed before and after a checkpoint
    is committed to storage. This allows for custom actions to be performed at specific
    points in the checkpoint writing process, such as metadata updates, cleanup operations,
    or notifications.
    """

    @abc.abstractmethod
    def pre_commit(self, path: str, **kwargs: dict[str, Any]) -> None:
        """
        Performs actions before committing the checkpoint.
        """

    @abc.abstractmethod
    def post_commit(self, path: str, **kwargs: dict[str, Any]) -> None:
        """
        Performs actions after committing the checkpoint.
        """


@dataclass
class CheckpointWriterConfig:
    """
    Configuration options for the CheckpointWriter.

    Attributes:
        write_barrier_timeout_secs: Maximum time in seconds to wait for all ranks
            to reach the checkpoint barrier before timing out. Default is 600 seconds.
    """

    write_barrier_timeout_secs: int = 600


class CheckpointWriter:
    """
    Handles writing state dictionaries to storage.

    This class is responsible for writing model state dictionaries to storage according
    to the specified checkpoint layout. It supports synchronization barriers to ensure
    all ranks in a distributed setting complete their checkpoint operations.
    """

    def __init__(
        self,
        config: CheckpointWriterConfig,
        rank_info: RankInfo,
        barrier: Barrier | None = None,
        commit_hook: WriterHook | None = None,
    ):
        """
        Initialize a CheckpointWriter.

        Args:
            config: Configuration options for the checkpoint writer.
            rank_info: Information about the current rank in a distributed setting.
            barrier: Optional synchronization barrier for distributed checkpointing.
                    Note: The barrier should be initialized with the appropriate barrier_prefix
                    and timeout_secs parameters.
            commit_hook: Optional hook for custom actions before and after checkpoint commits.
        """

        self._config = config
        self._rank_info = rank_info
        self._commit_hook = commit_hook
        self._barrier = barrier

    def write(
        self,
        path: str,
        state_dict: STATE_DICT,
        **kwargs: dict[str, Any],
    ) -> Future[None] | None:
        """
        Writes the state_dict to storage.

        Args:
            path (str): The path to write the checkpoint to.
            state_dict (STATE_DICT): The state_dict to write.
            **kwargs: Additional keyword arguments passed to hooks.

        Returns:
            Optional[Future[None]]: A future for tracking the write operation, if applicable.
        """
        logger.debug(
            "Writing checkpoint to %s for rank %s",
            path,
            self._rank_info.global_rank,
        )
        dir_path = Path(path)
        full_path = dir_path / f"checkpoint_{self._rank_info.global_rank}.pt"
        os.makedirs(
            os.path.dirname(full_path),
            exist_ok=True,
        )
        torch.save(state_dict, full_path)
        logger.debug("Successfully saved checkpoint file to %s", full_path)

        # Execute pre-commit hook if available
        commit_hook = self._commit_hook
        if commit_hook is not None:
            logger.debug("Executing pre-commit hook for %s", path)
            commit_hook.pre_commit(path, **kwargs)

        # Wait for all ranks to finish writing if barrier is available
        barrier = self._barrier
        if barrier is not None:
            logger.info(
                "Waiting for all ranks at barrier with timeout %ss",
                self._config.write_barrier_timeout_secs,
            )
            barrier.execute_barrier()
            logger.info("All ranks passed barrier")
        else:
            logger.info("No barrier configured, skipping synchronization")

        # Execute commit hook if available
        if commit_hook is not None:
            logger.debug("Executing commit hook for %s", path)
            commit_hook.post_commit(path, **kwargs)

        logger.info(
            "Successfully wrote checkpoint to %s for rank %s",
            path,
            self._rank_info.global_rank,
        )
        return None

    def close(self) -> None:
        """
        Close the writer and release any resources.

        This is a no-op for the base CheckpointWriter but may be overridden
        by subclasses that need to perform cleanup.
        """
        logger.debug("Closing checkpoint writer")
