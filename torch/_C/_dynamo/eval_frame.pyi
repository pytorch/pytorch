# mypy: allow-untyped-defs
import types
from typing import List, NewType, Optional

from torch._dynamo.types import DynamoCallback, DynamoGuardHook

# We implement our own FrameType-like type for Python >= 3.11. So it's not actually an alias of FrameType, but still
# exposes the same interface.
_PyInterpreterFrame = NewType("_PyInterpreterFrame", types.FrameType)

def set_eval_frame(callback: DynamoCallback) -> DynamoCallback: ...
def reset_code(code: types.CodeType) -> None: ...
def unsupported(obj1: object, obj2: object) -> object: ...
def skip_code(code: types.CodeType) -> None: ...
def set_guard_error_hook(hook: DynamoGuardHook) -> None: ...

class _CacheEntry:
    def check_fn(self, *args, **kwargs): ...
    code: types.CodeType
    next: Optional[_CacheEntry]

class _ExtraState:
    def invalidate(self, cache_entry: _CacheEntry): ...

def _debug_get_cache_entry_list(code: types.CodeType) -> List[_CacheEntry]: ...

py_opcode_caches: List[int]
