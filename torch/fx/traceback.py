from contextlib import contextmanager
from typing import Optional

__all__ = ['override_stack_trace', 'current_stack_trace', 'is_stack_trace_overridden']

active_interpreter = None
is_overriden = False

@contextmanager
def override_stack_trace(interpreter):
    global active_interpreter
    global is_overriden

    saved_intepreter = active_interpreter
    saved_is_overriden = is_overriden
    try:
        active_interpreter = interpreter
        is_overriden = True
        yield
    finally:
        active_interpreter = saved_intepreter
        is_overriden = saved_is_overriden

def current_stack_trace() -> Optional[str]:
    global active_interpreter

    if active_interpreter:
        node = getattr(active_interpreter, "current_node", None)
        return node.stack_trace
    return None

def is_stack_trace_overridden() -> bool:
    global is_overriden

    return is_overriden
