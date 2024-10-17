# type: ignore
import threading
import torch
from ._compile import _disable_dynamo
from ._C import _autograd
# TODO(rzou): why doesn't torch.fx.wrap work directly?
from torch.fx._symbolic_trace import _create_wrapped_func as wrap

"""
TODO(rzou): did we really need a new file? I did it to appease trace_rules.
"""


def python_autograd(saved_state, hooks, nodecalls, num_outputs, arange):
    """Given the state of the autograd graph (the saved tensors/sizes/scalar,
    hooks, and the actual nodes), execute it in Python.

    Compiled Autograd uses the equivalent of torch.fx.symbolic_trace over
    this function to produce a graph that can then be Dynamo'ed.

    NB: Before executing this function (or an acquired graph version of it)
    on real Tensors, please call set_global_nodecalls(nodecalls) to set the
    current autograd nodes structure state. We intentionally hide this state
    from the graph so that Dynamo doesn't need to deal with proxying it into
    the graph.

    TODO(rzou): Compiled Autograd is responsible for calling set_global_nodecalls
    using the current nodecalls data structure. If the user did not specify
    retain_graph=True, then something needs to free it later,
    so we don't end up keeping the nodes around forever.
    """
    node_to_idx_data = {node_id(call.node): idx for idx, call in enumerate(nodecalls)}

    def node_to_idx(node):
        return node_to_idx_data[torch._compiled_autograd.node_id(node)]

    input_buffers = {}

    def lookup_input_buffer(node_idx, num_inputs):
        if node_idx not in input_buffers:
            input_buffers[node_idx] = [None] * num_inputs
        return input_buffers[node_idx]

    saved_state = iter(SavedState(
        nodecalls,
        saved_state[0],
        saved_state[1],
        saved_state[2],
    ))

    graph_outputs = [None] * num_outputs

    for idx, call in enumerate(nodecalls):
        node_idx = arange[idx]
        inputs = lookup_input_buffer(idx, call.node.num_inputs())

        # Given all of the saved state, retrieve the saved state that matters
        # for the current node call.
        apply_state, validate_outputs_state = next(saved_state)

        for hook_idx, input_idx in call.tensor_pre_hooks:
            inputs[input_idx] = call_hook(hooks[hook_idx], inputs[input_idx], hook_type="pre_hook")
        for input_nr, result_idx in call.graph_output:
            graph_outputs[result_idx] = inputs[input_nr]
        if not call.needed:
            continue
        if call.node.is_compiled_autograd_traceable():
            outputs = apply_with_saved(node_idx, inputs, *apply_state)
        else:
            outputs = apply_with_saved_dynamo_disabled(node_idx, inputs, *apply_state)
        outputs = validate_outputs(node_idx, outputs, *validate_outputs_state)
        for hook_idx, input_idx in call.post_hooks:
            call_hook(hooks[hook_idx], outputs, inputs, hook_type="post_hook")
        for output_idx in range(call.node.num_outputs()):
            output = outputs[output_idx]
            next_edge = call.node.next_edge(output_idx)
            if not next_edge.is_valid():
                continue
            next_node = next_edge.function
            input_buffer = lookup_input_buffer(node_to_idx(next_node), next_node.num_inputs())
            updated_buffer = accumulate(input_buffer[next_edge.input_nr], output)
            input_buffer[next_edge.input_nr] = updated_buffer

    return graph_outputs


global_nodecalls = threading.local()


def get_node(idx):
    return global_nodecalls.thread_local[idx].node


def set_global_nodecalls(nodecalls):
    global_nodecalls.thread_local = nodecalls


@wrap
def apply_with_saved(node_idx, inputs, saved_tensors, saved_sizes, saved_scalars):
    """
    Applies the node at global_nodecalls[node_idx] using the inputs and saved values.
    """
    node = get_node(node_idx)
    outputs = _autograd.apply_with_saved(global_nodecalls.thread_local[node_idx], inputs, saved_tensors, list(saved_sizes), saved_scalars)
    return outputs


@_disable_dynamo
@wrap
def apply_with_saved_dynamo_disabled(node_idx, inputs, saved_tensors, saved_sizes, saved_scalars):
    """
    This is apply_with_saved, but also induces a graph break in Dynamo.
    """
    return apply_with_saved(node_idx, inputs, saved_tensors, saved_sizes, saved_scalars)


@wrap
def validate_outputs(node_idx, outputs, saved_tensors, saved_sizes, saved_scalars):
    """
    Validates the outputs of the node at global_nodecalls[node_idx]. This requires
    swizzling out some input metadata state of the next nodes, which is why
    it also accepts some saved variables.
    """
    outputs = _autograd.validate_outputs_with_saved(global_nodecalls.thread_local[node_idx], outputs, saved_tensors, list(saved_sizes), saved_scalars)
    return outputs


def node_id(node):
    if node is None:
        breakpoint()
    assert node is not None
    return _autograd.node_id(node)


def arange(num):
    return list(range(num))


@wrap
def call_hook(*args, **kwargs):
    return torch._dynamo.external_utils.call_hook(*args, **kwargs)


class IterableWrapper:
    def __init__(self, noniterable, size):
        self.noniterable = noniterable
        self.idx = 0
        self.size = size

    def __iter__(self):
        return self

    def __next__(self):
        assert self.idx < self.size
        result = self.noniterable[self.idx]
        self.idx += 1
        return result


class SavedState:
    def __init__(self, nodecalls, tensors, sizes, scalars):
        self.tensors = tensors
        self.sizes = sizes
        self.scalars = scalars
        self.nodecalls = iter(nodecalls)

    def __iter__(self):
        return self

    def __next__(self):
        call = next(self.nodecalls)

        def get_next(collection_info):
            tensors = [next(self.tensors) for _ in range(collection_info.num_saved_tensors)]
            sizes = [next(self.sizes) for _ in range(collection_info.num_saved_sizes)]
            scalars = [next(self.scalars) for _ in range(collection_info.num_saved_ivalues)]
            return (tensors, sizes, scalars)

        saved_state_for_apply = get_next(call.compiled_args_info)
        saved_state_for_validate_output = get_next(call.next_edges_info)
        return saved_state_for_apply, saved_state_for_validate_output


@wrap
def accumulate(old_var, var):
    if old_var is None:
        return var
    if var is None:
        return old_var
    return old_var + var
