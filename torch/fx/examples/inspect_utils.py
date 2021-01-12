from tabulate import tabulate

"""
The methods in this file may be used to examine the state of the code
and how the Graph evolves at any time during execution. If you're
unsure of what's happening in an example in this folder, try adding one
of these methods before and after a key line.
"""

def print_IR(graph):
    node_specs = [[n.op, n.name, n.target, n.args, n.kwargs]
                  for n in graph.nodes]
    print(tabulate(node_specs,
                   headers=['opcode', 'name', 'target', 'args', 'kwargs']))
