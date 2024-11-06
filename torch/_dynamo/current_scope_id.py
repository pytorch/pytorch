import contextlib
import threading
from typing import Generator


# Global variable to identify which SubgraphTracer we are in.
# It is sometimes difficult to find an InstructionTranslator to use.
_current_scope_id = threading.local()


def current_scope_id() -> int:
    global _current_scope_id
    if not hasattr(_current_scope_id, "value"):
        _current_scope_id.value = 1
    return _current_scope_id.value


@contextlib.contextmanager
def enter_new_scope() -> Generator[None, None, None]:
    global _current_scope_id
    try:
        _current_scope_id.value = current_scope_id() + 1
        yield
    finally:
        _current_scope_id.value = current_scope_id() - 1
