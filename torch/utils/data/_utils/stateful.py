# mypy: allow-untyped-defs
"""
Stateful protocol for objects that can save and restore their state.

This module defines the Stateful protocol that objects can implement to
participate in DataLoader checkpointing when using stateful=True.
"""

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Stateful(Protocol):
    """
    Protocol for objects that can save and restore their state.

    Objects implementing this protocol can participate in DataLoader checkpointing
    when using stateful=True.

    Example:
        class MyDataset:
            def __init__(self):
                self.position = 0

            def state_dict(self) -> dict[str, Any]:
                return {"position": self.position}

            def load_state_dict(self, state_dict: dict[str, Any]) -> None:
                self.position = state_dict["position"]
    """

    def state_dict(self) -> dict[str, Any]:
        """
        Return the state of the object as a dictionary.

        Returns:
            Dict[str, Any]: A dictionary containing the object's state that can
                           be used to restore the object later via load_state_dict.
        """
        ...

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """
        Load the state from a dictionary.

        Args:
            state_dict: A dictionary containing the state to restore, typically
                       returned by a previous call to state_dict().
        """
        ...
