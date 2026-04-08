"""
Experimental staging module for PyTorch Distributed Checkpointing.

This module provides advanced staging capabilities for checkpoints including:
- Asynchronous staging using ThreadPoolExecutor
- Pinned memory allocation for faster CPU-GPU transfers
- Shared memory support for multi-process scenarios
- Non-blocking CUDA operations with stream synchronization
- Caching of frequently used storages for efficient memory management
- Automatic resource cleanup and memory management

Classes:
    CheckpointStager: Abstract base class defining the staging interface
    StagingOptions: Configuration dataclass for staging behavior
    DefaultStager: Default implementation with comprehensive staging features
"""

import abc
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, TypeVar

import torch
from torch.distributed.checkpoint._state_dict_stager import StateDictStager

from .types import STATE_DICT


T = TypeVar("T")


class CheckpointStager(abc.ABC):
    """
    Abstract base class for checkpoint staging implementations.

    CheckpointStager defines the interface that all staging implementations
    must follow. Staging is the process of offloading state dictionaries
    for async checkpointing.
    """

    @abc.abstractmethod
    def stage(
        self,
        state_dict: STATE_DICT,
        **kwargs: Any,
    ) -> STATE_DICT | Future[STATE_DICT]:
        """
        Stage a state dictionary for checkpointing.

        Args:
            state_dict: The state dictionary to stage
            **kwargs: Additional staging parameters

        Returns:
            Either a staged state dictionary (synchronous) or a Future
            that will resolve to the staged state dictionary (asynchronous)
        """

    @abc.abstractmethod
    def close(self) -> None:
        """
        Clean up all resources used by the stager.
        """


@dataclass
class CheckpointStagerConfig:
    """
    Configuration options for checkpoint staging behavior.

    Attributes:
        use_pinned_memory (bool): Enable pinned memory allocation for faster
            CPU-GPU transfers. Requires CUDA to be available. Default: True
        use_shared_memory (bool): Enable shared memory for multi-process
            scenarios. Useful when multiple processes need access to the
            same staged data. Default: True
        use_async_staging (bool): Enable asynchronous staging using a
            background thread pool. Allows overlapping computation with
            staging operations. Requires CUDA. Default: True
        use_non_blocking_copy (bool): Use non-blocking device memory
            copies with stream synchronization. Improves performance by
            allowing CPU work to continue during GPU transfers. Default: True

    Note:
        CUDA-dependent features will raise exception if CUDA is not available.
    """

    use_pinned_memory: bool = True
    use_shared_memory: bool = True
    use_async_staging: bool = True
    use_non_blocking_copy: bool = True


class DefaultStager(CheckpointStager):
    """
    DefaultStager provides a full-featured staging implementation that combines
    multiple optimization techniques for efficient checkpoint preparation.

    The staging process works as follows:
    1. State dictionary is submitted for staging (sync or async)
    2. Tensors are copied from GPU to optimized CPU storage
    3. CUDA operations are synchronized if non-blocking copies are used
    4. Staged state dictionary is returned or made available via Future

    NOTE: state_dict should be deep-copyable object as staging will create a
    copy of it.

    Usage Patterns:
        # Synchronous staging
        stager = DefaultStager(CheckpointStagerConfig(use_async_staging=False))
        staged_dict = stager.stage(state_dict)
        stager.close()

        # Asynchronous staging
        stager = DefaultStager(CheckpointStagerConfig(use_async_staging=True))
        future = stager.stage(state_dict)
        # ... do other work ...
        staged_dict = future.result()
        stager.close()

        # Context manager pattern (recommended)
        with DefaultStager(config) as stager:
            result = stager.stage(state_dict)
            # Automatic cleanup on exit

    Performance Considerations:
        - Async staging provides best performance when model computation
          can overlap with staging operations
        - Pinned memory improves CPU-GPU transfer speeds but uses more memory
        - Shared memory allows efficient IPC to checkpoint process
        - Non-blocking copies reduce GPU idle time during memory transfers

    Thread Safety:
        DefaultStager is not thread-safe. Each thread should use its own
        instance, or external synchronization should be provided.
    """

    def __init__(
        self,
        config: CheckpointStagerConfig = CheckpointStagerConfig(),
    ):
        self._config = config
        self._state_dict_stager = StateDictStager(
            pin_memory=config.use_pinned_memory, share_memory=config.use_shared_memory
        )
        self._staging_executor = None
        self._staging_stream = None

        if self._config.use_async_staging:
            self._staging_executor = ThreadPoolExecutor(max_workers=1)
            if torch.accelerator.is_available():
                # Note: stream needs to be initialized on the main thread after default cuda
                # stream is setup/used to avoid the risk of accidentally reusing the main
                # compute stream or in other cases kernels actually launching from the
                # main thread.
                self._staging_stream = torch.Stream()

        if self._config.use_non_blocking_copy:
            if not torch.accelerator.is_available():
                raise AssertionError(
                    "Non-blocking copy requires that the current accelerator is available."
                )

    def stage(
        self,
        state_dict: STATE_DICT,
        **kwargs: Any,
    ) -> STATE_DICT | Future[STATE_DICT]:
        if self._config.use_async_staging:
            if self._staging_executor is None:
                raise AssertionError(
                    "Staging executor should be initialized for async staging"
                )
            return self._staging_executor.submit(
                self._stage,
                state_dict,
                **kwargs,
            )
        else:
            return self._stage(state_dict, **kwargs)

    def _stage(self, state_dict: STATE_DICT, **kwargs: Any) -> STATE_DICT:
        state_dict = self._state_dict_stager.stage(
            state_dict, non_blocking=self._config.use_non_blocking_copy, **kwargs
        )

        if self._config.use_non_blocking_copy:
            if not (self._staging_stream or not self._config.use_async_staging):
                raise AssertionError(
                    "Non-blocking copy in a background thread for async staging needs staging_stream to be initialized."
                )

            # waits for the enqued copy operations to finish.
            self._staging_stream.synchronize() if self._staging_stream else torch.accelerator.synchronize()

        return state_dict

    def close(self) -> None:
        """
        Clean up all resources used by the DefaultStager. Shuts down the ThreadPoolExecutor
        used for async staging operations and cleans up the underlying StateDictStager's
        cached storages. Should be called when the stager is no longer needed to prevent
        resource leaks, especially in long-running applications. After calling close(),
        the stager should not be used for further staging operations.

        state_dict should be deep-copyable object.

        Example:
            stager = DefaultStager(CheckpointStagerConfig(use_async_staging=True))
            # ... do staging operations ...
            stager.close()  # Clean up all resources
        """
        if self._staging_executor:
            self._staging_executor.shutdown(wait=True)

        self._state_dict_stager.close()
