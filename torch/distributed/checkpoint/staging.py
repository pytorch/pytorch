from typing import runtime_checkable

from typing_extensions import Protocol

from torch.distributed.checkpoint.metadata import STATE_DICT_TYPE


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

    2. dcp.save is called on the state_dict returned from stage in parallel. This call is respondsible
        for serializing the state_dict and writing it to storage.

    3. If AsyncStager.should_synchronize_after_execute is True, this method will be called immediately after
        the serialization thread starts and before returning from dcp.async_save. If this is set to False,
        the assumption is the user has defined a custom synchronization point for the the purpose of further
        optimizing save latency in the training loop (for example, by overlapping staging with the
        forward/backward pass), and it is respondsibility is on the user to call `AsyncStager.synchronize_staging`
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
