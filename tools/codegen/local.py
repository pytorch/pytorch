import threading
from contextlib import contextmanager
from typing import Optional, Iterator

# Simple dynamic scoping implementation.  The name "parametrize" comes
# from Racket.

class Locals(threading.local):
    use_c10_dispatcher_full: Optional[bool] = None
    hack_const_mutable_self: bool = False
_locals = Locals()

def use_c10_dispatcher_full() -> bool:
    assert _locals.use_c10_dispatcher_full is not None, \
        "need to initialize local.use_c10_dispatcher_full with local.parametrize"
    return _locals.use_c10_dispatcher_full

def hack_const_mutable_self() -> bool:
    return _locals.hack_const_mutable_self

@contextmanager
def parametrize(*, use_c10_dispatcher_full: bool, hack_const_mutable_self: bool) -> Iterator[None]:
    old_use_c10_dispatcher_full = _locals.use_c10_dispatcher_full
    old_hack_const_mutable_self = _locals.hack_const_mutable_self
    try:
        _locals.use_c10_dispatcher_full = use_c10_dispatcher_full
        _locals.hack_const_mutable_self = hack_const_mutable_self
        yield
    finally:
        _locals.use_c10_dispatcher_full = old_use_c10_dispatcher_full
        _locals.hack_const_mutable_self = old_hack_const_mutable_self
