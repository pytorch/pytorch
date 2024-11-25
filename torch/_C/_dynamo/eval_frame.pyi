# mypy: allow-untyped-defs
import types
from typing import Dict, NewType, Tuple

from torch._dynamo.types import DynamoCallback, DynamoGuardHook

# For typechecking
SkipCodeRecursiveFlag = NewType("SkipCodeRecursiveFlag", object)
CacheLimitHitFlag = NewType("CacheLimitHitFlag", object)
# Flag returned by Dynamo tracer to indicate to Dynamo eval frame that we should skip frames recursively.
skip_code_recursive_flag: SkipCodeRecursiveFlag
cache_limit_hit_flag: CacheLimitHitFlag

def set_eval_frame(callback: DynamoCallback) -> DynamoCallback: ...
def set_skip_guard_eval_unsafe(value: bool) -> bool: ...
def get_eval_frame_callback() -> DynamoCallback: ...
def reset_code(code: types.CodeType) -> None: ...
def unsupported(obj1: object, obj2: object) -> object: ...
def skip_code(code: types.CodeType) -> None: ...
def set_guard_error_hook(hook: DynamoGuardHook) -> None: ...
def raise_sigtrap() -> None: ...

class _CacheEntry:
    def check_fn(self, *args, **kwargs): ...
    code: types.CodeType
    next: _CacheEntry | None

class _ExtraState:
    def invalidate(self, cache_entry: _CacheEntry, guard_manager: object): ...

# This is an object that encapsulates the Python FrameType, and exposes
# properties Dynamo cares about for a frame.
class _PyInterpreterFrame:
    f_code: types.CodeType
    f_locals: Dict[str, object]
    f_globals: Dict[str, object]
    f_builtins: Dict[str, object]
    f_lasti: int
    f_lineo: int
    f_back: types.FrameType
    # A tuple containing cell objects captured by this frame.
    closure: Tuple[types.CellType]

def _debug_get_cache_entry_list(code: types.CodeType) -> list[_CacheEntry]: ...

py_opcode_caches: list[int]
