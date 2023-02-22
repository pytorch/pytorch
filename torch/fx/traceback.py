import traceback
from contextlib import contextmanager
from typing import Optional, List, Any, Dict
from ._compatibility import compatibility

__all__ = ['override_stack_trace', 'set_stack_trace', 'append_stack_trace', 'format_stack',
           'is_stack_trace_overridden', 'get_current_meta', 'set_current_meta']


current_stack: List[str] = []
current_meta: Dict[str, Any] = {}
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
def set_stack_trace(stack : List[str]):
    global current_stack

    if is_overridden and stack:
        current_stack = stack

@compatibility(is_backward_compatible=False)
@contextmanager
def append_stack_trace(stack : Optional[str]):
    """
    The content of stack here is an entire stacktraces as a string
    """
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
def format_stack() -> List[str]:
    if is_overridden:
        return current_stack.copy()
    else:
        # fallback to traceback.format_stack()
        return traceback.format_list(traceback.extract_stack()[:-1])


@compatibility(is_backward_compatible=False)
def is_stack_trace_overridden() -> bool:
    return is_overridden


@compatibility(is_backward_compatible=False)
@contextmanager
def set_current_meta(meta : Dict[str, Any]):
    global current_meta

    old_meta = current_meta
    if is_overridden and meta:
        try:
            current_meta = meta
            yield
        finally:
            current_meta = old_meta
    else:
        yield


@compatibility(is_backward_compatible=False)
def get_current_meta() -> Dict[str, Any]:
    return current_meta.copy()
