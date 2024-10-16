import threading
from ._compile import _allow_in_graph, _disable_dynamo

"""
need this new file to deal with trace_rules
"""

global_input_buffers = threading.local()
global_nodecalls = threading.local()

def get_node(idx):
    return global_nodecalls.thread_local[idx].node

def set_global_nodecalls(nodecalls):
    global_nodecalls.thread_local = nodecalls

def CA_input_buffers_init():
    global_input_buffers.thread_local = InputBuffers()

def CA_input_buffers_lookup(node_idx):
    node = get_node(node_idx)
    print("looking up input buffers for: ", node_id(node))
    print("global_input_buffers: ", global_input_buffers.thread_local)
    result = global_input_buffers.thread_local.lookup(node).buffer
    print("input buffers: ", result)
    return result


from ._C import _autograd


def CA_apply_with_saved(node_idx, inputs, saved_tensors, saved_sizes, saved_scalars):
    node = get_node(node_idx)
    outputs = _autograd.apply_with_saved(global_nodecalls.thread_local[node_idx], inputs, saved_tensors, list(saved_sizes), saved_scalars)
    return outputs

@_disable_dynamo
def CA_apply_with_saved_dynamo_disabled(node_idx, inputs, saved_tensors, saved_sizes, saved_scalars):
    node = get_node(node_idx)
    outputs = _autograd.apply_with_saved(global_nodecalls.thread_local[node_idx], inputs, saved_tensors, list(saved_sizes), saved_scalars)
    return outputs


def CA_validate_outputs(node_idx, outputs):
    pass

def CA_update_input_buffers(node_idx, outputs):
    node = get_node(node_idx)
    for output_idx, output in enumerate(outputs):
        next_edge = node.next_edge(output_idx)
        if next_edge.is_valid() and output is not None:
            next_node = next_edge.function
            print("setting input buffers for: ", node_id(next_node))
            input_buffer = global_input_buffers.thread_local.lookup(next_node)
            input_buffer.add(next_edge.input_nr, output)
            print("global_input_buffers: ", global_input_buffers.thread_local)

def node_id(node):
    if node is None:
        breakpoint()
    assert node is not None
    return _autograd.node_id(node)


class InputBuffers:
    def __init__(self):
        self.dct = {}

    def __repr__(self):
        return repr(self.dct)

    def lookup(self, node):
        key = _autograd.node_id(node)
        if key not in self.dct:
            self.dct[key] = InputBuffer(node.num_inputs())
        return self.dct[key]

    def get(self, node):
        key = _autograd.node_id(node)
        return self.dct[key]


class InputBuffer:
    def __init__(self, size):
        self.buffer = [None] * size

    def __repr__(self):
        return repr(self.buffer)

    def __getitem__(self, pos):
        return self.buffer[pos]

    def add(self, pos, var):
        if var is None:
            return
        old_var = self.buffer[pos]
        if old_var is None:
            self.buffer[pos] = var
        else:
            accumulate(self.buffer, pos, var)


def accumulate(buffer, pos, var):
    # TODO(rzou): some more stuff here
    buffer[pos] = buffer[pos] + var


def validate_outputs(edges, grads, format_error):
    if len(grads) != len(edges):
        raise ValueError(f"Invalid number of gradients - expected {len(edges)}, but got {len(grads)}")

    # TODO(rzou): some more stuff here

    for idx, grad in enumerate(grads):
        edge = edges[idx]
        if not edge.is_valid():
            continue
        metadata = edge.function.input_metadata(edge.input_nr)
        if grad is None:
            continue
        grads[idx] = metadata.maybe_reduce(idx, grad, format_error)
    
    # TODO(rzou): some more stuff here
