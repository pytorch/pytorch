import torch
from torch.fx import symbolic_trace
import operator

"""
How to Replace One Op With Another

1. Iterate through all Nodes in your GraphModule's Graph.
2. Determine if the current Node should be replaced. (Suggested: match
on the Node's ``target`` attribute).
3. Create a replacement Node and add it to the Graph.
4. Use the FX built-in ``replace_all_uses_with`` to replace all uses of
the current Node with the replacement.
5. Delete the old Node from the graph.
6. Call ``recompile`` on the GraphModule. This updates the generated
Python code to reflect the new Graph state.

Currently, FX does not provide any way to guarantee that replaced
operators are syntactically valid. It's up to the user to confirm that
any new operators will work with the existing operands.

The following code demonstrates an example of replacing any instance of
addition with a bitwise AND.

To examine how the Graph evolves during op replacement, add the
statement `print(traced.graph)` after the line you want to inspect.
Alternatively, call `traced.graph.print_tabular()` to see the IR in a
tabular format.
"""

# Sample module
class M(torch.nn.Module):
    def forward(self, x, y):
        return x + y, torch.add(x, y), x.add(y)

# Symbolically trace an instance of the module
traced = symbolic_trace(M())

# As demonstrated in the above example, there are several different ways
# to denote addition. The possible cases are:
#     1. `x + y` - A `call_function` Node with target `operator.add`.
#         We can match for equality on that `operator.add` directly.
#     2. `torch.add(x, y)` - A `call_function` Node with target
#         `torch.add`. Similarly, we can match this function directly.
#     3. `x.add(y)` - The Tensor method call, whose target we can match
#         as a string.

patterns = set([operator.add, torch.add, "add"])

# Go through all the nodes in the Graph
for n in traced.graph.nodes:
    # If the target matches one of the patterns
    if any(n.target == pattern for pattern in patterns):
        # Set the insert point, add the new node, and replace all uses
        # of `n` with the new node
        with traced.graph.inserting_after(n):
            new_node = traced.graph.call_function(torch.bitwise_and, n.args, n.kwargs)
            n.replace_all_uses_with(new_node)
        # Remove the old node from the graph
        traced.graph.erase_node(n)

# Don't forget to recompile!
traced.recompile()
