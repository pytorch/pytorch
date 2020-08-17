# type: ignore

# Nodes represent a definition of a value in our graph of operators.
class Node:
    def __init__(self, graph, name, op, target, args, kwargs):
        self.graph = graph
        self.name = name  # unique name of value being created
        self.op = op  # the kind of operation = placeholder|call_method|call_module|call_function|getattr
        self.target = target  # for method/module/function, the name of the method/module/function/attr
        # being invoked, e.g add, layer1, or torch.add
        self.args = args
        self.kwargs = kwargs
        self.uses = 0

    def __repr__(self):
        return self.name
