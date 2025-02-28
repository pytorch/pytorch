"""This module contains the core type definitions and protocols used throughout Dynamo.

The types defined here fall into several categories:
- Guard related types (GuardFn, GuardFail, GuardedCode): Used for tracking and managing guards that protect compiled code
- Frame and cache types (FrameState, CacheEntry): Used for managing interpreter frame state and caching
- Callback protocols (DynamoCallbackFn): Define the interface for frame evaluation callbacks
- Hook protocols (DynamoGuardHook, ProfilerStartHook, ProfilerEndHook, BytecodeHook): Define various hook points for
  instrumentation and customization

These types provide the foundational interfaces that enable Dynamo's dynamic compilation and optimization system,
ensuring type safety and clear contracts between different components of the system.
"""

import dataclasses
import types
from typing import Any, Callable, NamedTuple, Optional, Protocol, Union

# CacheEntry has a `guard_manager` field for the guard, and a `code` field for the code object.
from torch._C._dynamo.eval_frame import (
    _CacheEntry as CacheEntry,
    _ExtraState as ExtraState,
    _FrameAction as FrameAction,
    _FrameExecStrategy as FrameExecStrategy,
    _PyInterpreterFrame as DynamoFrameType,
)
from torch._guards import CompileId


# We use a dict to store additional data per frame.
FrameState = dict[Any, Any]


class GuardFail(NamedTuple):
    # A string repr of the piece of failed guard code we eval-ed
    reason: str
    # A code object where we failed a guard
    orig_code: types.CodeType


class GuardFn(Protocol):
    closure_vars: dict[str, object]
    args: list[str]
    code_parts: list[str]
    verbose_code_parts: list[str]
    global_scope: dict[str, object]
    guard_fail_fn: Optional[Callable[[GuardFail], None]]
    cache_entry: Optional[CacheEntry]
    extra_state: Optional[ExtraState]

    # maps locals of user function to bool
    def __call__(self, f_locals: dict[str, object]) -> bool:
        ...


@dataclasses.dataclass
class GuardedCode:
    code: types.CodeType
    guard_manager: GuardFn
    compile_id: CompileId
    trace_annotation: str = "Unknown"


@dataclasses.dataclass
class ConvertFrameReturn:
    # default return is no compiled code (i.e. `return None`):
    # strategy is to skip non-recursively, for all future intercepted frames too

    # eval fram execution strategy for this frame
    frame_exec_strategy: FrameExecStrategy = dataclasses.field(
        default_factory=lambda: FrameExecStrategy(FrameAction.SKIP, FrameAction.DEFAULT)
    )
    # also apply frame_exec strategy to future frames with same code
    apply_to_code: bool = True
    guarded_code: Optional[GuardedCode] = None


def wrap_guarded_code(guarded_code: GuardedCode) -> ConvertFrameReturn:
    return ConvertFrameReturn(
        frame_exec_strategy=FrameExecStrategy(FrameAction.DEFAULT, FrameAction.DEFAULT),
        guarded_code=guarded_code,
    )


class DynamoCallbackFn(Protocol):
    def __call__(
        self,
        frame: DynamoFrameType,
        cache_entry: Optional[CacheEntry],
        frame_state: FrameState,
    ) -> ConvertFrameReturn:
        ...


DynamoCallback = Union[DynamoCallbackFn, None, bool]


class DynamoGuardHook(Protocol):
    def __call__(
        self,
        guard_manager: GuardFn,
        code: types.CodeType,
        f_locals: dict[str, object],
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


class BytecodeHook(Protocol):
    def __call__(
        self, code: types.CodeType, new_code: types.CodeType
    ) -> Optional[types.CodeType]:
        ...
