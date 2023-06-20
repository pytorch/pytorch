import traceback
from contextlib import contextmanager
from typing import List, Any, Dict
from ._compatibility import compatibility

__all__ = ['preserve_node_meta', 'has_preserved_node_meta',
           'set_stack_trace', 'set_bwd_seq_id', 'format_stack',
           'set_current_meta', 'get_current_meta']

current_meta: Dict[str, Any] = {}
should_preserve_node_meta = False
bwd_seq_id = -1


@compatibility(is_backward_compatible=False)
@contextmanager
def preserve_node_meta():
    global should_preserve_node_meta

    saved_should_preserve_node_meta = should_preserve_node_meta
    try:
        should_preserve_node_meta = True
        yield
    finally:
        should_preserve_node_meta = saved_should_preserve_node_meta


@compatibility(is_backward_compatible=False)
def set_stack_trace(stack : List[str]):
    global current_meta

    if should_preserve_node_meta and stack:
        current_meta["stack_trace"] = "".join(stack)

@compatibility(is_backward_compatible=False)
def set_bwd_seq_id(max_fwd_seq_id):
    global current_meta
    global bwd_seq_id

    if should_preserve_node_meta:
        # 1st bwd op is set to seq id of last fwd op
        # Then count down to 0
        if bwd_seq_id == -1:
            bwd_seq_id = max_fwd_seq_id
        assert bwd_seq_id >= 0, f"Unexpected bwd seq id {bwd_seq_id}"
        current_meta["seq_id"] = bwd_seq_id
        bwd_seq_id = bwd_seq_id - 1


@compatibility(is_backward_compatible=False)
def format_stack() -> List[str]:
    if should_preserve_node_meta:
        return [current_meta.get("stack_trace", "")]
    else:
        # fallback to traceback.format_stack()
        return traceback.format_list(traceback.extract_stack()[:-1])


@compatibility(is_backward_compatible=False)
def has_preserved_node_meta() -> bool:
    return should_preserve_node_meta


@compatibility(is_backward_compatible=False)
@contextmanager
def set_current_meta(meta : Dict[str, Any]):
    global current_meta

    if should_preserve_node_meta and meta:
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
