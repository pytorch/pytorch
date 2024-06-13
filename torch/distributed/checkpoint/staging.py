from typing import Optional, runtime_checkable
from typing_extensions import Protocol

from torch.distributed._state_dict_utils import (
    _copy_state_dict,
    _create_cpu_state_dict,
    _offload_state_dict_to_cpu,
)

from torch.distributed.checkpoint.metadata import STATE_DICT_TYPE

__all__ = ["AsyncStager", "BlockingAsyncStager"]


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

    def stage(self, state_dict: STATE_DICT_TYPE) -> STATE_DICT_TYPE:
        """
        Returns a "staged" copy of `state_dict`. The expectation of the staged copy is that it is
        innoculated from any updates incurred after the stage call is complete.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement stage method"
        )

    def synchronize_staging(self) -> None:
        """
        In the case `stage` is async in some way, this method should be called to ensure staging
        is complete and it is safe to begin modifying the original `state_dict`
        """
        pass


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
                that the stager is maintained and re-used for multiple dcp.async_save calls. Default to False.
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
            return _offload_state_dict_to_cpu(state_dict, type_check=self.type_check)

        if self.state_dict_cache is None:
            self.state_dict_cache = _create_cpu_state_dict(state_dict, pin_memory=True)
        return _copy_state_dict(state_dict, self.state_dict_cache)

    def synchronize_staging(self) -> None:
        """
        No-op function, since staging is blocking.
        """
        pass
