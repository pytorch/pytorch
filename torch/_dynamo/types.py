import types
from typing import Any, Callable, Dict, Optional, Union, List, OrderedDict

from typing_extensions import Protocol

from torch._dynamo.guards import GuardedCode


class DynamoCallbackFn(Protocol):
    def __call__(
        self, frame: types.FrameType, cache_size: int
    ) -> Optional[GuardedCode]:
        ...


DynamoCallback = Union[DynamoCallbackFn, None, bool]


class GuardFn(Protocol):
    closure_vars: OrderedDict[str, object]
    code_parts: List[str]
    verbose_code_parts: List[str]
    global_scope: Dict[str, object]

    def __call__(*args: object) -> bool:
        ...


class DynamoGuardHook(Protocol):
    def __call__(
        self,
        guard_fn: GuardFn,
        code: types.CodeType,
        f_locals: Dict[str, object],
        last: bool,
    ) -> None:
        ...
