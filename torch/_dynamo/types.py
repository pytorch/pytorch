import dataclasses
import types
from typing import Dict, List, Optional, OrderedDict, Union

from typing_extensions import Protocol


class GuardFn(Protocol):
    closure_vars: OrderedDict[str, object]
    code_parts: List[str]
    verbose_code_parts: List[str]
    global_scope: Dict[str, object]

    # maps locals of user function to bool
    def __call__(self, *maybe_dotzero: object, **f_locals: object) -> bool:
        ...


@dataclasses.dataclass
class GuardedCode:
    code: types.CodeType
    check_fn: GuardFn


class DynamoCallbackFn(Protocol):
    def __call__(
        self, frame: types.FrameType, cache_size: int
    ) -> Optional[GuardedCode]:
        ...


DynamoCallback = Union[DynamoCallbackFn, None, bool]


class DynamoGuardHook(Protocol):
    def __call__(
        self,
        guard_fn: GuardFn,
        code: types.CodeType,
        f_locals: Dict[str, object],
        last: bool,
    ) -> None:
        ...
