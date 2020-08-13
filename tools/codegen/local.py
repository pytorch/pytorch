import threading
from contextlib import contextmanager
from typing import Optional, Iterator

# Simple dynamic scoping implementation.  The name "parametrize" comes
# from Racket.

class Locals(threading.local):
    use_c10_dispatcher_full: Optional[bool] = None
_locals = Locals()

def use_c10_dispatcher_full() -> bool:
    assert _locals.use_c10_dispatcher_full is not None, \
        "need to initialize local.use_c10_dispatcher_full with local.parametrize"
    return _locals.use_c10_dispatcher_full

@contextmanager
def parametrize(use_c10_dispatcher_full: bool) -> Iterator[None]:
    old_use_c10_dispatcher_full = _locals.use_c10_dispatcher_full
    try:
        _locals.use_c10_dispatcher_full = use_c10_dispatcher_full
        yield
    finally:
        _locals.use_c10_dispatcher_full = old_use_c10_dispatcher_full
