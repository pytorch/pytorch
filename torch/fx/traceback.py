from contextlib import contextmanager
from typing import Optional
from ._compatibility import compatibility

__all__ = ['override_stack_trace', 'current_stack_trace', 'is_stack_trace_overridden']

active_interpreter = None
is_overriden = False

@compatibility(is_backward_compatible=False)
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

@compatibility(is_backward_compatible=False)
def current_stack_trace() -> Optional[str]:
    global active_interpreter

    if is_overriden and active_interpreter:
        node = getattr(active_interpreter, "current_node", None)
        if node is not None:
            return node.stack_trace
    return None

@compatibility(is_backward_compatible=False)
def is_stack_trace_overridden() -> bool:
    global is_overriden

    return is_overriden
