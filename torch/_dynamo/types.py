import types
from typing_extensions import Protocol
from torch._dynamo.guards import GuardedCode
from typing import Callable, Optional, Union, Dict, Any

class DynamoCallbackFn(Protocol):
    def __call__(frame: types.FrameType, cache_size: int) -> Optional[GuardedCode]:
        ...

DynamoCallback = Union[DynamoCallbackFn, None, bool]

class DynamoGuardHook(Protocol):
    def __call__(
        guard_fn: Callable, code: types.CodeType, f_locals: Dict[str, Any], last: bool
    ) -> None:
        ...

