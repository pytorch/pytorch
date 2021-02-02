import torch
import torch.fx as fx

# An inverse mapping is one that takes a function f(x) and returns a function g
# such that f(g(x)) == x. For example,since log(exp(x)) == x, exp and log are
# inverses.

invert_mapping = {}
def add_inverse(a, b):
    invert_mapping[a] = b
    invert_mapping[b] = a
inverses = [
    (torch.sin, torch.arcsin),
    (torch.cos, torch.arccos),
    (torch.tan, torch.arctan),
    (torch.exp, torch.log),
]
for a, b in inverses:
    add_inverse(a, b)

# The general strategy is that we walk the graph backwards, transforming each
# node into its inverse. To do so, we swap the outputs and inputs of the
# functions, and then we look up its inverse in `invert_mapping`. Note that
# this transform assumes that all operations take in only one input and return
# one output.
def invert(model: torch.nn.Module) -> torch.nn.Module:
    fx_model = fx.symbolic_trace(model)
    new_graph = fx.Graph()  # As we're building up a new graph
    env = {}
    for node in reversed(fx_model.graph.nodes):
        if node.op == 'call_function':
            # This creates a node in the new graph with the inverse function,
            # and passes `env[node.name]` (i.e. the previous output node) as
            # input.
            new_node = new_graph.call_function(invert_mapping[node.target], (env[node.name],))
            env[node.args[0].name] = new_node
        elif node.op == 'output':
            # We turn the output into an input placeholder
            new_node = new_graph.placeholder(node.name)
            env[node.args[0].name] = new_node
        elif node.op == 'placeholder':
            # We turn the input placeholder into an output
            new_graph.output(env[node.name])
        else:
            raise RuntimeError("Not implemented")

    new_graph.lint()
    return fx.GraphModule(fx_model, new_graph)


def f(x):
    return torch.exp(torch.tan(x))

res = invert(f)
print(res.code)
"""
def forward(self, output):
    log_1 = torch.log(output);  output = None
    arctan_1 = torch.arctan(log_1);  log_1 = None
    return arctan_1
"""
print(f(res((torch.arange(5) + 1))))  # [1., 2., 3., 4, 5.]
