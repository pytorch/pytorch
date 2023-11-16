from typing import Any, Dict, TypeVar

from typing_extensions import Protocol, runtime_checkable


@runtime_checkable
class Stateful(Protocol):
    def state_dict(self) -> Dict[str, Any]:
        ...

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        ...


#TODO: Evaluate AppState vs `checkpoint.metadata.State_DICT_TYPE`
T = TypeVar("T", bound=Stateful)
AppState = Dict[str, T]
