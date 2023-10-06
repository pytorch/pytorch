import traceback
from contextlib import contextmanager
from typing import List, Any, Dict
from ._compatibility import compatibility

__all__ = ['preserve_node_meta', 'has_preserved_node_meta',
           'set_stack_trace', 'set_grad_fn_seq_nr', 'reset_grad_fn_seq_nr',
           'format_stack', 'set_current_meta', 'get_current_meta']

current_meta: Dict[str, Any] = {}
should_preserve_node_meta = False


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
def set_grad_fn_seq_nr(seq_nr):
    global current_meta

    if should_preserve_node_meta:
        # The seq_nr is captured by eager mode in the grad_fn during forward
        current_meta["prev_grad_fn_seq_nr"] = current_meta.get("grad_fn_seq_nr", None)
        current_meta["prev_in_grad_fn"] = current_meta.get("in_grad_fn", None)
        current_meta["grad_fn_seq_nr"] = seq_nr
        current_meta["in_grad_fn"] = True


@compatibility(is_backward_compatible=False)
def reset_grad_fn_seq_nr():
    # NB: reset state properly, this would be helpful towards supporting
    #     reentrant autograd if we actually wanted to do that.
    global current_meta

    if should_preserve_node_meta:
        if current_meta["prev_grad_fn_seq_nr"] is None:
            assert current_meta["prev_in_grad_fn"] is None
            del current_meta["grad_fn_seq_nr"]
            del current_meta["in_grad_fn"]
        current_meta["grad_fn_seq_nr"] = current_meta["prev_grad_fn_seq_nr"]
        current_meta["in_grad_fn"] = current_meta["prev_in_grad_fn"]


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
def set_current_meta(node):
    global current_meta
    if should_preserve_node_meta and node.meta:
        saved_meta = current_meta
        try:
            current_meta = node.meta.copy()

            # Append (node.name, node.target) onto "from_node" for provenance tracking
            if "from_node" not in current_meta:
                current_meta["from_node"] = [(node.name, node.target)]
            elif current_meta["from_node"][-1][0] != node.name:
                current_meta["from_node"].append((node.name, node.target))

            yield
        finally:
            current_meta = saved_meta
    else:
        yield


@compatibility(is_backward_compatible=False)
def get_current_meta() -> Dict[str, Any]:
    return current_meta
