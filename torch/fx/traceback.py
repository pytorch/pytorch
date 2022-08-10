from contextlib import contextmanager
from typing import Optional, List
from ._compatibility import compatibility

__all__ = ['override_stack_trace', 'append_stack_trace', 'format_stack', 'is_stack_trace_overridden']

current_stack: List[str] = []
is_overridden = False

@compatibility(is_backward_compatible=False)
@contextmanager
def override_stack_trace():
    global is_overridden

    saved_is_overridden = is_overridden
    try:
        is_overridden = True
        yield
    finally:
        is_overridden = saved_is_overridden


@compatibility(is_backward_compatible=False)
@contextmanager
def append_stack_trace(stack: Optional[str]):
    global is_overridden
    global current_stack

    if is_overridden and stack:
        try:
            current_stack.append(stack)
            yield
        finally:
            current_stack.pop()
    else:
        yield


@compatibility(is_backward_compatible=False)
def format_stack() -> str:
    global is_overridden
    global current_stack

    if is_overridden:
        return '\n'.join(reversed(current_stack))
    else:
        return ''


@compatibility(is_backward_compatible=False)
def is_stack_trace_overridden() -> bool:
    global is_overridden

    return is_overridden
