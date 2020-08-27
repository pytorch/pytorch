import threading
from contextlib import contextmanager
from typing import Optional, Iterator

from tools.codegen.model import UseC10Dispatcher

# Simple dynamic scoping implementation.  The name "parametrize" comes
# from Racket.

class Locals(threading.local):
    use_c10_dispatcher: Optional[UseC10Dispatcher] = None
    hack_const_mutable_self: bool = False
_locals = Locals()

def use_c10_dispatcher() -> UseC10Dispatcher:
    assert _locals.use_c10_dispatcher is not None, \
        "need to initialize local.use_c10_dispatcher with local.parametrize"
    return _locals.use_c10_dispatcher

def hack_const_mutable_self() -> bool:
    return _locals.hack_const_mutable_self

@contextmanager
def parametrize(*, use_c10_dispatcher: UseC10Dispatcher, hack_const_mutable_self: bool) -> Iterator[None]:
    old_use_c10_dispatcher = _locals.use_c10_dispatcher
    old_hack_const_mutable_self = _locals.hack_const_mutable_self
    try:
        _locals.use_c10_dispatcher = use_c10_dispatcher
        _locals.hack_const_mutable_self = hack_const_mutable_self
        yield
    finally:
        _locals.use_c10_dispatcher = old_use_c10_dispatcher
        _locals.hack_const_mutable_self = old_hack_const_mutable_self
