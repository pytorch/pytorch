import dis
import torch
import inspect
import operator

class TraceError(ValueError):
    pass

# Nodes represent a definition of a value in our graph of operators.
# For simplicitly, they also have methods defined on them so that they serve as proxy objects for _building_
# more nodes.
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

    def __getattr__(self, k):
        # note: not added to the graph yet, if this is a method call
        # we peephole optimize to the method invocation
        return Attribute(self, k)

    def __call__(self, *args, **kwargs):
        return self._create_node('call_method', '__call__', [self] + args, kwargs)

    def __iter__(self):
        frame = inspect.currentframe()
        calling_frame = frame.f_back
        inst = list(dis.get_instructions(calling_frame.f_code))[calling_frame.f_lasti // 2]
        if inst.opname == 'UNPACK_SEQUENCE':
            return (self[i] for i in range(inst.argval))
        self._no_control_flow()

    def _no_control_flow(self):
        raise TraceError('symbolically traced variables cannot be used as inputs to control flow')

    def __bool__(self):
        self._no_control_flow()

    def __torch_function__(self, orig_method, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if torch.overrides.is_tensor_method_or_property(orig_method):
            return self.graph._create_node('call_method', orig_method.__name__, args, kwargs)
        else:
            return self.graph._create_node('call_function', orig_method, args, kwargs, name=self.graph._name(orig_method.__name__))

class Attribute(Node):
    def __init__(self, node, attr):
        super().__init__(node.graph, node.graph._name(attr), 'call_function', getattr, [node, attr], {})

    def __call__(self, *args, **kwargs):
        return self.node.graph._create_node('call_method', self.args[1], [self.args[0]] + list(args), kwargs)

reflectable_magic_methods = {
    'add': '{} + {}',
    'sub': '{} - {}',
    'mul': '{} * {}',
    'floordiv': '{} // {}',
    'truediv': '{} / {}',
    'div': '{} / {}',
    'mod': '{} % {}',
    'pow': '{} ** {}',
    'lshift': '{} << {}',
    'rshift': '{} >> {}',
    'and': '{} & {}',
    'or': '{} | {}',
    'xor': '{} ^ {}',
    'getitem': '{}[{}]'
}

magic_methods = dict({
    'eq': '{} == {}',
    'ne': '{} != {}',
    'lt': '{} < {}',
    'gt': '{} > {}',
    'le': '{} <= {}',
    'ge': '{} >= {}',
    'pos': '+{}',
    'neg': '-{}',
    'invert': '~{}'}, **reflectable_magic_methods)

for method in magic_methods:
    def scope(method):
        def impl(*args, **kwargs):
            g = args[0].graph
            target = getattr(operator, method)
            return g._create_node('call_function', target, args, kwargs)
        impl.__name__ = method
        as_magic = f'__{method}__'
        setattr(Node, as_magic, impl)
    scope(method)

for orig_method_name in reflectable_magic_methods:
    def scope(orig_method_name):
        method_name = f'__r{orig_method_name}__'

        def impl(self, rhs):
            target = getattr(operator, orig_method_name)
            return self.graph._create_node('call_function', target, [rhs, self])
        impl.__name__ = method_name
        impl.__qualname__ = method_name
        setattr(Node, method_name, impl)
    scope(orig_method_name)
