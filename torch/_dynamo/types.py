import dataclasses
import sys
import types
from typing import (
    Any,
    Callable,
    Dict,
    List,
    NamedTuple,
    Optional,
    OrderedDict,
    Protocol,
    Union,
)


if sys.version_info >= (3, 11):
    from torch._C._dynamo import eval_frame

    DynamoFrameType = eval_frame._PyInterpreterFrame
else:
    DynamoFrameType = types.FrameType

import torch

# This class has a `check_fn` field for the guard,
#  and a `code` field for the code object.
CacheEntry = torch._C._dynamo.eval_frame._CacheEntry

# We use a dict to store additional data per frame.
FrameState = Dict[Any, Any]


class GuardFail(NamedTuple):
    # A string repr of the piece of failed guard code we eval-ed
    reason: str
    # A code object where we failed a guard
    orig_code: types.CodeType


class GuardFn(Protocol):
    closure_vars: OrderedDict[str, object]
    args: List[str]
    code_parts: List[str]
    verbose_code_parts: List[str]
    global_scope: Dict[str, object]
    guard_fail_fn: Optional[Callable[[GuardFail], None]]

    # maps locals of user function to bool
    def __call__(self, f_locals: Dict[str, object]) -> bool:
        ...


@dataclasses.dataclass
class GuardedCode:
    code: types.CodeType
    check_fn: GuardFn


class DynamoCallbackFn(Protocol):
    def __call__(
        self,
        frame: DynamoFrameType,
        cache_entry: Optional[CacheEntry],  # type: ignore[valid-type]
        frame_state: FrameState,
    ) -> Optional[GuardedCode]:
        ...


DynamoCallback = Union[DynamoCallbackFn, None, bool]


class DynamoGuardHook(Protocol):
    def __call__(
        self,
        guard_fn: GuardFn,
        code: types.CodeType,
        f_locals: Dict[str, object],
        index: int,
        last: bool,
    ) -> None:
        ...


class ProfilerStartHook(Protocol):
    def __call__(
        self,
        name: str,
        # TODO(whc) how do I annotate a _RecordFunction here?
    ) -> Any:
        ...


class ProfilerEndHook(Protocol):
    def __call__(self, record: Any) -> None:
        ...
