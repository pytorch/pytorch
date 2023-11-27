from typing import Any, Dict, runtime_checkable, TypeVar

from typing_extensions import Protocol


__all__ = ["Stateful", "StatefulT"]


@runtime_checkable
class Stateful(Protocol):
    def state_dict(self, **kwargs) -> Dict[str, Any]:
        ...

    def load_state_dict(self, state_dict: Dict[str, Any], **kwargs) -> None:
        ...


StatefulT = TypeVar("StatefulT", bound=Stateful)
