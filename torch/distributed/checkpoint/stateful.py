from typing import Any, runtime_checkable, TypeVar
from typing_extensions import Protocol


__all__ = ["Stateful", "StatefulT"]


@runtime_checkable
class Stateful(Protocol):
    """
    Stateful protocol for objects that can be checkpointed and restored.
    """

    def state_dict(self) -> dict[str, Any]:
        """
        Objects should return their state_dict representation as a dictionary.
        The output of this function will be checkpointed, and later restored in
        `load_state_dict()`.

        .. warning::
            Because of the inplace nature of restoring a checkpoint, this function
            is also called during `torch.distributed.checkpoint.load`.


        Returns:
            Dict: The objects state dict
        """

        ...

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """
        Restore the object's state from the provided state_dict.

        Args:
            state_dict: The state dict to restore from
        """

        ...


StatefulT = TypeVar("StatefulT", bound=Stateful)
