import torch
from torch.fx import symbolic_trace
import re


"""
How to replace one op with another
1. Iterate through all nodes in your GraphModule's Graph.
2. Determine if the current node should be replaced. (Suggested: match
on the node's ``target`` attribute).
3. Create a replacement node.
4. Use the fx built-in ``replace_all_uses_with`` to replace the current
node with the replacement.
5. Call `recompile` on the GraphModule. This updates the generated
Python code to reflect the new Graph state.

The following code demonstrates an example of replacing any instance of
addition with a bitwise AND.
"""

# Sample module
class M(torch.nn.Module):
    def forward(self, x):
        val = torch.neg(x) + torch.relu(x)
        return torch.add(val, val)

# Symbolically trace an instance of the module
traced = symbolic_trace(M())

# If you have ``torch`` imported, you have two ways to denonte addition:
# the ``+`` operator and the method ``torch.add``. ``+`` is a Python
# built-in, so it's represented as a ``call_function`` Node with a target
# of ``<built-in function add>``. ``torch.add`` is part of ``torch``, so
# it becomes a ``call_method`` Node with a target of
# ``<built-in method add of type object at MEMORY-LOCATION-OF-TORCH>``.
#
# To determine whether a given node represents addition, we can match
# on the ``target`` attribute. In this particular case, we have two
# different representations of "addition", so we'll use a regex to match
# on ``target``. (If you want to replace a ``torch``-specific operator,
# you can match on a simple string.)
regexp = re.compile(r"(?<=[\s])add(?=[\>\s])")

# Go through all the nodes in the Graph
for n in traced.graph.nodes:
    # If the target matches the regex
    if regexp.search(str(n.target)):
        # Create a replacement node with the new op
        new_node = traced.graph.call_function(torch.bitwise_and, n.args, n.kwargs)
        # Move the new node to the correct spot
        n.append(new_node)
        # Replace all uses of `n` with the new node
        n.replace_all_uses_with(new_node)
        # Remove the old node from the graph
        traced.graph.erase_node(n)

traced.recompile()
