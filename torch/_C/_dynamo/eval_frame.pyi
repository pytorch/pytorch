# mypy: allow-untyped-defs
import types
from typing import NewType

from torch._dynamo.types import DynamoCallback, DynamoGuardHook

# We implement our own FrameType-like type for Python >= 3.11. So it's not actually an alias of FrameType, but still
# exposes the same interface.
_PyInterpreterFrame = NewType("_PyInterpreterFrame", types.FrameType)

# For typechecking
SkipCodeRecursiveFlag = NewType("SkipCodeRecursiveFlag", object)
CacheLimitHitFlag = NewType("CacheLimitHitFlag", object)
# Flag returned by Dynamo tracer to indicate to Dynamo eval frame that we should skip frames recursively.
skip_code_recursive_flag: SkipCodeRecursiveFlag
cache_limit_hit_flag: CacheLimitHitFlag

def set_eval_frame(callback: DynamoCallback) -> DynamoCallback: ...
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
    def invalidate(self, cache_entry: _CacheEntry): ...

def _debug_get_cache_entry_list(code: types.CodeType) -> list[_CacheEntry]: ...

py_opcode_caches: list[int]
