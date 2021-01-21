import torch
from torch.fx import Proxy, symbolic_trace
from torch.fx.node import map_arg


'''
How to inline a function into an existing Graph

If you're using the default Tracer to symbolically trace, ``torch.nn``
module instances will appear as ``call_module`` calls rather than being
traced through. Let's say this behavior is almost what you need; the
only problem is that there's a single module call that you want to
replace with an inlined trace of the function. Creating a custom Tracer
would be too much. Instead, you can accomplish this using Proxies.

The following code demonstrates how to trace a module and inline it
into an existing Graph using Proxy. We'll trace our Graph, trace our
function, then iterate through our Graph's list of Nodes until we find
the right place to swap out the ``call_module`` Node with an inlined
trace. At that point, we'll create Proxies from the Node's args and
kwargs. Finally, we'll call our traced module with those Proxies and
insert the ``output`` Node from that call into our Graph. (This final
step will automatically inline the entire traced function.)
'''


# Sample module
class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        return self.relu(x) + 1.0

# Symbolically trace an instance of `M`. After tracing, `self.relu` is
# represented as a `call_module` Node. The full operation in the
# generated `forward` function's code will appear as `self.relu(x)`
m = symbolic_trace(M())

# Trace through the ReLU module only. This allows us to get its Graph
# to inline
traced_relu = symbolic_trace(m.relu)

# Insert nodes from the ReLU graph in place of the original call to
# `self.relu`
for node in m.graph.nodes:
    # Find `call_module` Node in `m` that corresponds to `self.relu`.
    # This is the Node we want to swap out for an inlined version of the
    # same call
    if (node.op, node.target) == ("call_module", "relu"):
        with m.graph.inserting_before(node):
            # Create a Proxy from each Node in the current Node's
            # args/kwargs
            proxy_args = map_arg(node.args, Proxy)
            proxy_kwargs = map_arg(node.kwargs, Proxy)
            # Call `traced_relu` with the newly-created Proxy arguments.
            # `traced_relu` is the generic version of the function; by
            # calling it with Proxies created from Nodes in `m`, we're
            # creating a "closure" that includes those Nodes. The result
            # of this call is another Proxy, which we can hook into
            # our existing Graph to complete the function inlining.
            proxy_output = traced_relu(*proxy_args, **proxy_kwargs)
            # Replace the relu `call_module` node with the inlined
            # version of the function
            node.replace_all_uses_with(proxy_output.node)
            # Make sure that the old relu Node is erased
            m.graph.erase_node(node)
