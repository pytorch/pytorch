
import torch
from torch.fx import Proxy, GraphModule, Node, symbolic_trace

from enum import Enum, auto

'''
Wrap Graph Output Dynamically

The following code demonstrates how change an existing Graph based on
parameters specified at runtime. We'll let the user specify an
activation function from a predefined Enum list, then we'll symbolically
trace it. Next, we'll create a Proxy from the last operation in the
Graph. We'll call our traced activation function with this Proxy and
insert the ``output`` Node from that call into our Graph. (This final
step will automatically inline the entire traced function.)
'''


# Sample module
class M(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        y = torch.cat([x, y])
        return y

# Symbolically trace an instance of `M`
traced = symbolic_trace(M())

# Selected activation functions
class ActivationFunction(Enum):
    RELU = auto()
    LEAKY_RELU = auto()
    PRELU = auto()

# Map activation function names to their implementation
activation_functions = {
    ActivationFunction.RELU: torch.nn.ReLU(),
    ActivationFunction.LEAKY_RELU: torch.nn.LeakyReLU(),
    ActivationFunction.PRELU: torch.nn.PReLU(),
}

def wrap_in_activation_function(m: GraphModule, fn: ActivationFunction) -> GraphModule:
    # Get output node
    output_node: Optional[Node] = None
    for n in reversed(m.graph.nodes):
        if n.op == "output":
            output_node = n
            break
    assert output_node

    # Get the actual output (the "input" of the output node). This is
    # the Node we want to wrap in a user-specified activation function
    assert len(output_node.all_input_nodes) == 1
    wrap_node = output_node.all_input_nodes[0]

    # Wrap the actual output in a Proxy
    wrap_proxy = Proxy(wrap_node)

    # Get the implementation of the specified activation function and
    # symbolically trace it
    fn_impl = activation_functions[fn]
    fn_impl_traced = symbolic_trace(fn_impl)

    # Call the specified activation function using the Proxy wrapper for
    # `output_op`. The result of this call is another Proxy, which we
    # can hook into our existing Graph.
    with traced.graph.inserting_before(wrap_node):
        fn_impl_output_node = fn_impl_traced(wrap_proxy)
        new_args = (fn_impl_output_node.node,)
        output_node.args = new_args


# Example call
wrap_in_activation_function(traced, ActivationFunction.LEAKY_RELU)
