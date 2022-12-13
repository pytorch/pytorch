import traceback
from contextlib import contextmanager
from typing import List, Any, Dict
from ._compatibility import compatibility

__all__ = ['preserve_node_meta', 'has_preserved_node_meta',
           'set_stack_trace', 'format_stack',
           'set_current_meta', 'get_current_meta']

current_meta: Dict[str, Any] = {}
is_enabled = False


@compatibility(is_backward_compatible=False)
@contextmanager
def preserve_node_meta():
    global is_enabled

    saved_is_enabled = is_enabled
    try:
        is_enabled = True
        yield
    finally:
        is_enabled = saved_is_enabled


@compatibility(is_backward_compatible=False)
def set_stack_trace(stack : List[str]):
    global current_meta

    if is_enabled and stack:
        current_meta["stack_trace"] = "".join(stack)


@compatibility(is_backward_compatible=False)
def format_stack() -> List[str]:
    if is_enabled:
        return [current_meta.get("stack_trace", "")]
    else:
        # fallback to traceback.format_stack()
        return traceback.format_list(traceback.extract_stack()[:-1])


@compatibility(is_backward_compatible=False)
def has_preserved_node_meta() -> bool:
    return is_enabled


@compatibility(is_backward_compatible=False)
@contextmanager
def set_current_meta(meta : Dict[str, Any]):
    global current_meta

    if is_enabled and meta:
        saved_meta = current_meta
        try:
            current_meta = meta
            yield
        finally:
            current_meta = saved_meta
    else:
        yield


@compatibility(is_backward_compatible=False)
def get_current_meta() -> Dict[str, Any]:
    return current_meta.copy()
