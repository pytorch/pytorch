import dis
import torch
import inspect
import operator

from .graph import magic_methods, reflectable_magic_methods
from typing import TYPE_CHECKING, Tuple, Dict, Optional, Iterable, NoReturn, Any
from .node import Target, Node

from .tracer_base import TracerBase

class TraceError(ValueError):
    pass

# Proxy objects are stand-in values for normal values in a PyTorch computation.
# Instead of performing compute they record computation into Graph.
# Each proxy wraps the Node instance that represents the expression that define the
# value.

# Unwrap the proxies inside args, and kwargs, create the resulting node
# and then wrap the result in a proxy.
def _create_proxy(tracer: TracerBase, op: str, target: Target, args_: Tuple[Any, ...], kwargs_: Dict[str, Any], name=None):
    args = tracer.create_arg(args_)
    kwargs = tracer.create_arg(kwargs_)
    assert isinstance(args, tuple)
    assert isinstance(kwargs, dict)
    rn = tracer.create_node(op, target, args, kwargs, name)
    return Proxy(rn, tracer)

class GraphAppendingTracer(TracerBase):
    def __init__(self, graph: Graph):
        self.graph = graph

class Proxy:
    def __init__(self, node: Node, tracer: Optional[TracerBase] = None):
        if tracer is None:
            tracer = GraphAppendingTracer(node.graph)

        self.tracer = tracer
        self.node = node

    def __repr__(self) -> str:
        return f'Proxy({self.node.name})'

    def __getattr__(self, k) -> 'Attribute':
        # note: not added to the graph yet, if this is a method call
        # we peephole optimize to the method invocation
        return Attribute(self, k)

    def __call__(self, *args, **kwargs) -> 'Proxy':
        return _create_proxy(self.tracer, 'call_method', '__call__', (self,) + args, kwargs)

    def __iter__(self) -> Iterable['Proxy']:
        frame = inspect.currentframe()
        assert frame is not None
        calling_frame = frame.f_back
        assert calling_frame is not None
        inst = list(dis.get_instructions(calling_frame.f_code))[calling_frame.f_lasti // 2]
        if inst.opname == 'UNPACK_SEQUENCE':
            return (self[i] for i in range(inst.argval))  # type: ignore
        if inst.opname == 'CALL_FUNCTION_EX':
            self._no_arg_unpack()
        else:
            self._no_control_flow()

    def _no_control_flow(self) -> NoReturn:
        raise TraceError('symbolically traced variables cannot be used as inputs to control flow')

    def _no_arg_unpack(self) -> NoReturn:
        raise TraceError('Proxy object cannot be unpacked as function argument')

    def __bool__(self) -> NoReturn:
        self._no_control_flow()

    def __torch_function__(self, orig_method, types, args=None, kwargs=None):
        args = args if args else ()
        kwargs = kwargs if kwargs else {}
        if torch.overrides.is_tensor_method_or_property(orig_method):
            return _create_proxy(self.tracer, 'call_method', orig_method.__name__, args, kwargs)
        else:
            return _create_proxy(self.tracer, 'call_function', orig_method, args, kwargs,
                                 name=self.tracer.graph._name(orig_method.__name__))

class Attribute(Proxy):
    def __init__(self, root: Proxy, attr: str):
        self.root = root
        self.attr = attr
        self.tracer = root.tracer
        self._node: Optional[Node] = None

    @property
    def node(self):
        # the node for attributes is added lazily, since most will just be method calls
        # which do not rely on the getitem call
        if self._node is None:
            self._node = _create_proxy(self.tracer, 'call_function', getattr, (self.root, self.attr), {}).node
        return self._node

    def __call__(self, *args, **kwargs):
        return _create_proxy(self.tracer, 'call_method', self.attr, (self.root,) + args, kwargs)

for method in magic_methods:
    def scope(method):
        def impl(*args, **kwargs):
            tracer = args[0].tracer
            target = getattr(operator, method)
            return _create_proxy(tracer, 'call_function', target, args, kwargs)
        impl.__name__ = method
        as_magic = f'__{method}__'
        setattr(Proxy, as_magic, impl)
    scope(method)

def _define_reflectable(orig_method_name):
    method_name = f'__r{orig_method_name}__'

    def impl(self, rhs):
        target = getattr(operator, orig_method_name)
        return _create_proxy(self.tracer, 'call_function', target, (rhs, self), {})
    impl.__name__ = method_name
    impl.__qualname__ = method_name
    setattr(Proxy, method_name, impl)

for orig_method_name in reflectable_magic_methods:
    _define_reflectable(orig_method_name)
