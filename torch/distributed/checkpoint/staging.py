from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Optional, Union
from typing_extensions import deprecated, Protocol, runtime_checkable

import torch
from torch.distributed._state_dict_utils import _copy_state_dict, _create_cpu_state_dict
from torch.distributed.checkpoint._state_dict_stager import StateDictStager
from torch.distributed.checkpoint.metadata import STATE_DICT_TYPE


__all__ = ["AsyncStager", "BlockingAsyncStager", "DefaultStager", "StagingOptions"]

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
    AsyncStager: Protocol defining the staging interface
    StagingOptions: Configuration dataclass for staging behavior
    DefaultStager: Default implementation with comprehensive staging features
    BlockingAsyncStager: Implementation of AsyncStager which stages the state_dict
    on CPU RAM and blocks until the copy is complete. Please use DefaultStager instead.
"""


@runtime_checkable
class AsyncStager(Protocol):
    """
    This protocol is meant to provide customization and extensibility for dcp.async_save, allowing users
    to customize how data is staged previous to executing the usual dcp.save path in parallel.
    The expected order of operations (concretely defined in `torch.distributed.state_dict_saver.async_save`)
    is the following:

    1. AsyncStager.stage_data(state_dict):
        This call gives the AsyncStager the opportunity to 'stage'
        the state_dict. The expectation and purpose of staging in this context is to create a "training-safe"
        representation of the state dict, meaning that any updates to module data after staging is complete
        should not be reflected in the state dict returned from this method. For example, in the default
        case a copy of the entire state dict is created on CPU RAM and returned here, allowing users
        to continue training without risking changes to data which is being serialized.

    2. dcp.save is called on the state_dict returned from stage in parallel. This call is responsible
        for serializing the state_dict and writing it to storage.

    3. If AsyncStager.should_synchronize_after_execute is True, this method will be called immediately after
        the serialization thread starts and before returning from dcp.async_save. If this is set to False,
        the assumption is the user has defined a custom synchronization point for the the purpose of further
        optimizing save latency in the training loop (for example, by overlapping staging with the
        forward/backward pass), and it is the respondsibility of the user to call `AsyncStager.synchronize_staging`
        at the appropriate time.

    """

    # default to True since the common case is to stage synchronously
    _synchronize_after_execute: bool = True

    @property
    def should_synchronize_after_execute(self) -> bool:
        """
        Whether to synchronize after executing the stage.
        """
        return self._synchronize_after_execute

    def stage(
        self, state_dict: STATE_DICT_TYPE
    ) -> Union[Future[STATE_DICT_TYPE], STATE_DICT_TYPE]:
        """
        Returns a "staged" copy of `state_dict`. The expectation of the staged copy is that it is
        inoculated from any updates incurred after the stage call is complete.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement stage method"
        )

    @deprecated(
        "`synchronize_staging` is deprecated and will be removed in future versions."
        "Please use staging_future from AsyncSaveResponse instead.",
        category=FutureWarning,
    )
    def synchronize_staging(self) -> None:
        """
        In the case `stage` is async in some way, this method should be called to ensure staging
        is complete and it is safe to begin modifying the original `state_dict`
        """

    def close(self) -> None:
        """
        Clean up all resources used by the stager.
        """


@dataclass
class StagingOptions:
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


class DefaultStager(AsyncStager):
    """
    DefaultStager provides a full-featured staging implementation that combines
    multiple optimization techniques for efficient checkpoint preparation.

    The staging process works as follows:
    1. State dictionary is submitted for staging (sync or async)
    2. Tensors are copied from GPU to optimized CPU storage
    3. CUDA operations are synchronized if non-blocking copies are used
    4. Staged state dictionary is returned or made available via Future

    Usage Patterns:
        # Synchronous staging
        stager = DefaultStager(StagingOptions(use_async_staging=False))
        staged_dict = stager.stage(state_dict)
        stager.close()

        # Asynchronous staging
        stager = DefaultStager(StagingOptions(use_async_staging=True))
        future = stager.stage(state_dict)
        # ... do other work ...
        staged_dict = future.result()
        stager.close()

        # Context manager pattern (recommended)
        stager = DefaultStager(config)
        with stager:
        result = stager.stage(state_dict)

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
        config: StagingOptions = StagingOptions(),
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
            assert torch.accelerator.is_available(), (
                "Non-blocking copy requires that the current accelerator is available."
            )

        self._staging_future: Optional[Future[STATE_DICT_TYPE]] = None

    def stage(
        self,
        state_dict: STATE_DICT_TYPE,
        **kwargs: Any,
    ) -> Union[STATE_DICT_TYPE, Future[STATE_DICT_TYPE]]:
        """
        This function is responsible for staging staging the state_dict.
        See class docstring for more details on staging.
        If use_async_staging is True, it will return a Future object that will be
        fulfilled when staging is complete.
        If use_async_staging is False, it will return the fully staged state_dict.

        Args:
            state_dict (STATE_DICT_TYPE): The state_dict to be staged.
        """
        if self._config.use_async_staging:
            assert self._staging_executor is not None
            self._staging_future = self._staging_executor.submit(
                self._stage,
                state_dict,
                **kwargs,
            )
            return self._staging_future
        else:
            return self._stage(state_dict, **kwargs)

    def _stage(self, state_dict: STATE_DICT_TYPE, **kwargs: Any) -> STATE_DICT_TYPE:
        if self._config.use_non_blocking_copy:
            assert self._staging_stream or not self._config.use_async_staging, (
                "Non-blocking copy in a background thread for async staging needs staging_stream to be initialized."
            )
            with (
                self._staging_stream
                if self._staging_stream is not None
                else nullcontext()
            ):
                state_dict = self._state_dict_stager.stage(
                    state_dict, non_blocking=self._config.use_non_blocking_copy
                )
            # waits for the enqued copy operations to finish.
            self._staging_stream.synchronize() if self._staging_stream else torch.accelerator.synchronize()
        else:
            state_dict = self._state_dict_stager.stage(state_dict, non_blocking=False)
        return state_dict

    def close(self) -> None:
        """
        Clean up all resources used by the DefaultStager. Shuts down the ThreadPoolExecutor
        used for async staging operations and cleans up the underlying StateDictStager's
        cached storages. Should be called when the stager is no longer needed to prevent
        resource leaks, especially in long-running applications. After calling close(),
        the stager should not be used for further staging operations.

        Example Usage:
            stager = DefaultStager(StagingOptions(use_async_staging=True))
            future = stager.stage(state_dict)
            result = future.result()
            stager.close()  # Clean up all resources
        """
        if self._staging_executor:
            self._staging_executor.shutdown(wait=True)

    def synchronize_staging(self) -> None:
        """
        When use_async_staging is True, this method will wait until staging is complete.
        If use_async_staging is False, this method is a no-op.
        """
        if self._staging_future is not None:
            self._staging_future.result()


class BlockingAsyncStager(AsyncStager):
    """
    An implementation of AsyncStager which stages the state_dict on CPU RAM and blocks until the copy is complete.
    This implementation also provides an option to optimize stage latency using pinned memory.

    N.B. synchronize_staging is a no-op in this case.


    """

    # default to True since the common case is to stage synchronously
    _synchronize_after_execute: bool = False

    def __init__(
        self,
        cache_staged_state_dict: bool = False,
        type_check: bool = False,
    ):
        """
        Initializes the BlockingAsyncStager.

        Args:
            cache_staged_state_dict: Whether to cache the staged state_dict. This option decreases staging latency
                at the cost of increases memory usage. Additionally, if this parameter is set to True, it's the expectation
                that the stager is maintained and reused for multiple dcp.async_save calls. Default to False.
            type_check: Whether to perform a type check during cpu_offload. Defaults to False.

        """
        self.cache_staged_state_dict = cache_staged_state_dict
        self.type_check = type_check
        self.state_dict_cache: Optional[STATE_DICT_TYPE] = None

    def stage(self, state_dict: STATE_DICT_TYPE) -> STATE_DICT_TYPE:
        """
        Returns a copy of `state_dict` on the CPU.
        """

        if not self.cache_staged_state_dict:
            staged_state_dict = _create_cpu_state_dict(state_dict)
            _copy_state_dict(state_dict, staged_state_dict, type_check=self.type_check)
            return staged_state_dict

        if self.state_dict_cache is None:
            self.state_dict_cache = _create_cpu_state_dict(state_dict, pin_memory=True)
        return _copy_state_dict(state_dict, self.state_dict_cache)

    def synchronize_staging(self) -> None:
        """
        No-op function, since staging is blocking.
        """

    def close(self) -> None:
        pass
