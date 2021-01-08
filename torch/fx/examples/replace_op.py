import torch
from torch.fx import symbolic_trace
import operator


"""
How to replace one op with another
1. Iterate through all nodes in your GraphModule's Graph.
2. Determine if the current node should be replaced. (Suggested: match
on the node's ``target`` attribute).
3. Create a replacement node.
4. Use the fx built-in ``replace_all_uses_with`` to replace the current
node with the replacement.
5. Call ``recompile`` on the GraphModule. This updates the generated
Python code to reflect the new Graph state.

Currently, FX does not provide operator normalization. It's up to the
user to confirm a way to confirm that any replacement operators will
work with the existing operands.

The following code demonstrates an example of replacing any instance of
addition with a bitwise AND.
"""

# Sample module
class M(torch.nn.Module):
    def forward(self, x, y):
        return x + y, torch.add(x, y), x.add(y)

# Symbolically trace an instance of the module
traced = symbolic_trace(M())

# As demonstrated in the above example, there are several different ways
# to denote addition. The possible cases are:
#     1. `x + y` - A `call_function` Node with target
#        `<built-in function add>`. This is `operator.add`, so we can
#         match on equality with that function directly.
#     2. `torch.add(x, y)` - A `call_function` Node with target
#        `<built-in method add of type object at MEMORY-LOCATION-OF-TORCH>`.
#         This is `torch.add`, which we can similarly match directly.
#     3. `x.add(y)` - The Tensor method call, whose target we can match
#         as a string.

patterns = [
    ('call_function', operator.add),
    ('call_function', torch.add),
    ('call_method', 'add')
]

# Go through all the nodes in the Graph
for n in traced.graph.nodes:
    # If the target matches one of the patterns
    to_match = (n.op, n.target)
    if any(to_match == pattern for pattern in patterns):
        # Set the insert point, add the new node, and replace all uses
        # of `n` with the new node
        with traced.graph.inserting_after(n):
            new_node = traced.graph.call_function(torch.bitwise_and, n.args, n.kwargs)
            n.replace_all_uses_with(new_node)
        # Remove the old node from the graph
        traced.graph.erase_node(n)

# Don't forget to recompile!
traced.recompile()
