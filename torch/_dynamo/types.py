import types
from typing import Any, Callable, Dict, Optional, Union

from typing_extensions import Protocol

from torch._dynamo.guards import GuardedCode


class DynamoCallbackFn(Protocol):
    def __call__(
        self, frame: types.FrameType, cache_size: int
    ) -> Optional[GuardedCode]:
        ...


DynamoCallback = Union[DynamoCallbackFn, None, bool]


class DynamoGuardHook(Protocol):
    def __call__(
        self,
        guard_fn: Callable,
        code: types.CodeType,
        f_locals: Dict[str, Any],
        last: bool,
    ) -> None:
        ...
