from typing import Any, Dict, runtime_checkable, TypeVar

from typing_extensions import Protocol


@runtime_checkable
class Stateful(Protocol):
    def state_dict(self) -> Dict[str, Any]:
        ...

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        ...


StatefulT = TypeVar("StatefulT", bound=Stateful)
