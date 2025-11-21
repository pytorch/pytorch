import enum
import types
from typing import Optional, overload

from torch._dynamo.guards import GuardManagerWrapper
from torch._dynamo.types import DynamoCallback, DynamoGuardCompleteHook, DynamoGuardHook
from torch._guards import CompileId

def set_eval_frame(callback: DynamoCallback) -> DynamoCallback: ...
def set_skip_guard_eval_unsafe(value: bool) -> bool: ...
def get_eval_frame_callback() -> DynamoCallback: ...
def reset_code(code: types.CodeType) -> None: ...
def unsupported(obj1: object, obj2: object) -> object: ...
def set_code_exec_strategy(
    code: types.CodeType, strategy: _FrameExecStrategy
) -> None: ...
def set_guard_error_hook(hook: DynamoGuardHook) -> None: ...
def set_guard_complete_hook(
    hook: Optional[DynamoGuardCompleteHook],
) -> Optional[DynamoGuardCompleteHook]: ...
def raise_sigtrap() -> None: ...
def set_c_recursion_limit(limit: int) -> None: ...
def get_c_recursion_limit() -> int: ...

class _CacheEntry:
    def check_fn(self, *args: object, **kwargs: object) -> bool: ...
    def update_diff_guard_root_manager(self) -> None: ...
    code: types.CodeType
    compile_id: CompileId
    # If we run into circular issues, just use object
    guard_manager: GuardManagerWrapper
    next: _CacheEntry | None

class _PrecompileEntry:
    guard_manager: GuardManagerWrapper

class _ExtraState:
    def invalidate(
        self, cache_entry: _CacheEntry, guard_manager: GuardManagerWrapper
    ) -> None: ...

class _FrameAction(enum.IntEnum):
    DEFAULT = 0
    SKIP = 1
    RUN_ONLY = 2

class _FrameExecStrategy:
    cur_action: _FrameAction
    recursive_action: _FrameAction

    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(
        self, cur_action: _FrameAction, recursive_action: _FrameAction
    ) -> None: ...

# This is an object that encapsulates the Python FrameType, and exposes
# properties Dynamo cares about for a frame.
class _PyInterpreterFrame:
    f_code: types.CodeType
    f_locals: dict[str, object]
    f_globals: dict[str, object]
    f_builtins: dict[str, object]
    f_lasti: int
    f_lineno: int
    f_back: types.FrameType
    # A tuple containing cell objects captured by this frame.
    closure: tuple[types.CellType]

def _debug_get_cache_entry_list(code: types.CodeType) -> list[_CacheEntry]: ...

py_opcode_caches: list[int]

def code_framelocals_names(code: types.CodeType) -> tuple[str]: ...
def _load_precompile_entry(
    code: types.CodeType,
    guard_manager: GuardManagerWrapper,
    dynamo_code: types.CodeType,
) -> None: ...
def _reset_precompile_entries(code: types.CodeType) -> None: ...
def _debug_get_precompile_entries(code: types.CodeType) -> list[_PrecompileEntry]: ...
