import dis
import torch
import inspect
import operator
import traceback

from .graph import magic_methods, reflectable_magic_methods, Graph
from typing import Tuple, Dict, Optional, Iterable, Any, Iterator, Callable
from .node import Target, Node, Argument, base_types, map_aggregate
from ._compatibility import compatibility
from .operator_schemas import check_for_mutable_operation
import torch.fx.traceback as fx_traceback

__all__ = ['TracerBase', 'GraphAppendingTracer', 'TraceError', 'Proxy', 'Attribute', 'ParameterProxy']


@compatibility(is_backward_compatible=False)
class Scope(object):
    """ Scope object that records the module path and the module type
    of a module. Scope is used to track the information of the module
    that contains a Node in a Graph of GraphModule. For example::

        class Sub(torch.nn.Module):
            def forward(self, x):
                # This will be a call_method Node in GraphModule,
                # scope for this would be (module_path="sub", module_type=Sub)
                return x.transpose(1, 2)

        class M(torch.nn.Module):
            def __init__(self):
                self.sub = Sub()

            def forward(self, x):
                # This will be a call_method Node as well,
                # scope for this would be (module_path="", None)
                x = x.transpose(1, 2)
                x = self.sub(x)
                return x

    """

    def __init__(self, module_path: str, module_type: Any):
        super().__init__()
        self.module_path = module_path
        self.module_type = module_type


@compatibility(is_backward_compatible=False)
class ScopeContextManager(object):
    """ A context manager to track the Scope of Node during symbolic tracing.
    When entering a forward function of a Module, we'll update the scope information of
    the current module, and when we exit, we'll restore the previous scope information.
    """

    def __init__(
        self, scope: Scope, current_module: torch.nn.Module, current_module_path: str
    ):
        super().__init__()
        self.prev_module_type = scope.module_type
        self.prev_module_path = scope.module_path
        self.scope = scope
        self.scope.module_path = current_module_path
        self.scope.module_type = type(current_module)

    def __enter__(self):
        return

    def __exit__(self, *args):
        self.scope.module_path = self.prev_module_path
        self.scope.module_type = self.prev_module_type
        return


@compatibility(is_backward_compatible=True)
class TracerBase:
    graph: Graph
    record_stack_traces : bool = False
    # Feature flag for mutable schema checking
    # Enableby default in 1.12
    check_mutable_operations : bool = False
    # Feature flag for assert tracing
    trace_asserts : bool = False
    # Feature flag for proxying accesses to buffer values
    proxy_buffer_attributes : bool = False

    # Name of the function to be traced. It will only be used when
    # ``root`` is an instance of ``nn.Module``
    traced_func_name: str = "forward"

    # Maps the containing module's name to the operator name
    scope : Scope

    # Mapping of node name to module scope
    node_name_to_scope: Dict[str, Tuple[str, type]]

    @compatibility(is_backward_compatible=True)
    def create_node(self, kind : str, target : Target,
                    args : Tuple[Argument, ...], kwargs : Dict[str, Argument], name : Optional[str] = None,
                    type_expr : Optional[Any] = None) -> Node:
        """
        Inserts a graph node given target, args, kwargs, and name.

        This method can be overridden to do extra checking, validation, or
        modification of values used in node creation. For example, one might
        want to disallow in-place operations from being recorded.
        """
        if kind == 'call_function' and self.check_mutable_operations:
            check_for_mutable_operation(target, args, kwargs)

        node = self.graph.create_node(kind, target, args, kwargs, name, type_expr)
        self.node_name_to_scope[node.name] = (
            self.scope.module_path,
            self.scope.module_type,
        )
        node.meta['nn_module_stack'] = {self.scope.module_path : self.scope.module_type}
        return node

    @compatibility(is_backward_compatible=True)
    def proxy(self, node: Node) -> 'Proxy':
        return Proxy(node, self)

    @compatibility(is_backward_compatible=True)
    def create_proxy(self, kind: str, target: Target, args: Tuple[Any, ...], kwargs: Dict[str, Any],
                     name: Optional[str] = None, type_expr : Optional[Any] = None,
                     proxy_factory_fn: Callable[[Node], 'Proxy'] = None):
        '''
        Create a Node from the given arguments, then return the Node
        wrapped in a Proxy object.

        If kind = 'placeholder', then we're creating a Node that
        represents the parameter of a function. If we need to encode
        a default parameter, we use the ``args`` tuple. ``args`` is
        otherwise empty for ``placeholder`` Nodes.
        '''

        args_ = self.create_arg(args)
        kwargs_ = self.create_arg(kwargs)
        assert isinstance(args_, tuple)
        assert isinstance(kwargs_, dict)

        node = self.create_node(kind, target, args_, kwargs_, name, type_expr)

        if not proxy_factory_fn:
            proxy = self.proxy(node)
        else:
            proxy = proxy_factory_fn(node)

        # Optionally set stack trace on the created Node for debugging purposes
        if fx_traceback.is_stack_trace_overridden():
            stacks = fx_traceback.format_stack()
            proxy.node.stack_trace = '\n'.join(reversed(stacks))
        elif self.record_stack_traces:
            user_frame = self._find_user_frame()
            if user_frame:
                walk_stack_gen = traceback.walk_stack(user_frame)
                summary = traceback.StackSummary.extract(walk_stack_gen)  # type: ignore[arg-type]
                tb_lines = summary.format()
                proxy.node.stack_trace = ''.join(tb_lines)

        return proxy

    def _find_user_frame(self):
        """
        Find the Python stack frame executing the user code during
        symbolic tracing.
        """
        # We have to do a little dance here. Basically, walk up the callstack and
        # record the first frame not in the pytorch source. This is the frame executing
        # the user code during tracing.
        frame = inspect.currentframe()

        pt_files = ['torch/fx/proxy.py',
                    'torch/fx/_symbolic_trace.py',
                    'torch/fx/experimental/proxy_tensor.py',
                    'torch/_ops.py',
                    'torch/_tensor.py',
                    'torch/utils/_python_dispatch.py',
                    'torch/_prims_common/wrappers.py',
                    'torch/_refs/__init__.py',
                    'torch/_refs/nn/functional/__init__.py'
                    ]
        while frame:
            frame = frame.f_back
            if frame and all(not frame.f_code.co_filename.endswith(file) for file in pt_files):
                break

        if not frame:
            return None

        return frame

    @compatibility(is_backward_compatible=True)
    def create_arg(self, a: Any) -> Argument:
        """
        A method that lowers the objects seen as arguments during symbolic evaluation
        into Argument types that can be stored in IR.

        Can be override to support more trace-specific types.
        """
        if not isinstance(a, Proxy) and hasattr(a, '__fx_create_arg__'):
            return a.__fx_create_arg__(self)
        # aggregates
        elif isinstance(a, tuple) and hasattr(a, '_fields'):
            # NamedTuple constructors don't seem to like getting a generator
            # expression as an argument to their constructor, so build this
            # intermediate tuple and unpack it into the NamedTuple constructor
            args = tuple(self.create_arg(elem) for elem in a)
            return type(a)(*args)  # type: ignore[arg-type]
        elif isinstance(a, (tuple, list)):
            return type(a)(self.create_arg(elem) for elem in a)
        elif isinstance(a, dict):
            r = {}
            for k, v in a.items():
                # Check for invalid dict keys. We do not want a Proxy to appear
                # anywhere within the key. Since keys can be collection types,
                # we iterate through the key with map_aggregate
                k = self.create_arg(k)

                def no_node(arg):
                    if isinstance(arg, Node):
                        raise RuntimeError("Keys for dictionaries used as an argument cannot contain a "
                                           f"Node. Got key: {k}")
                map_aggregate(k, no_node)

                r[k] = self.create_arg(v)
            return r
        elif isinstance(a, slice):
            return slice(self.create_arg(a.start), self.create_arg(a.stop), self.create_arg(a.step))

        if isinstance(a, Proxy):
            # base case: we unwrap the Proxy object
            return a.node
        elif isinstance(a, base_types) or a is None or a is ...:
            return a
        raise NotImplementedError(f"argument of type: {type(a)}")

    @compatibility(is_backward_compatible=True)
    def to_bool(self, obj: 'Proxy') -> bool:
        """Called when a proxy object is being converted to a boolean, such as
        when used in control flow.  Normally we don't know what to do because
        we don't know the value of the proxy, but a custom tracer can attach more
        information to the graph node using create_node and can choose to return a value.
        """
        raise TraceError('symbolically traced variables cannot be used as inputs to control flow')

    @compatibility(is_backward_compatible=True)
    def iter(self, obj: 'Proxy') -> Iterator:
        """Called when a proxy object is being iterated over, such as
        when used in control flow.  Normally we don't know what to do because
        we don't know the value of the proxy, but a custom tracer can attach more
        information to the graph node using create_node and can choose to return an iterator.
        """
        raise TraceError('Proxy object cannot be iterated. This can be '
                         'attempted when the Proxy is used in a loop or'
                         ' as a *args or **kwargs function argument. '
                         'See the torch.fx docs on pytorch.org for a '
                         'more detailed explanation of what types of '
                         'control flow can be traced, and check out the'
                         ' Proxy docstring for help troubleshooting '
                         'Proxy iteration errors')

    @compatibility(is_backward_compatible=True)
    def keys(self, obj: 'Proxy') -> Any:
        """Called when a proxy object is has the keys() method called.
        This is what happens when ** is called on a proxy. This should return an
        iterator it ** is suppose to work in your custom tracer.
        """
        return Attribute(obj, 'keys')()


# used in Proxy object when just appending to the graph while not tracing.
@compatibility(is_backward_compatible=True)
class GraphAppendingTracer(TracerBase):
    def __init__(self, graph: Graph):
        super().__init__()
        self.graph = graph

@compatibility(is_backward_compatible=False)
def assert_fn(x):
    assert x

@compatibility(is_backward_compatible=True)
class TraceError(ValueError):
    pass

@compatibility(is_backward_compatible=True)
class Proxy:
    """
    ``Proxy`` objects are ``Node`` wrappers that flow through the
    program during symbolic tracing and record all the operations
    (``torch`` function calls, method calls, operators) that they touch
    into the growing FX Graph.

    If you're doing graph transforms, you can wrap your own ``Proxy``
    method around a raw ``Node`` so that you can use the overloaded
    operators to add additional things to a ``Graph``.

    ``Proxy`` objects cannot be iterated. In other words, the symbolic
    tracer will throw an error if a ``Proxy`` is used in a loop or as
    an ``*args``/``**kwargs`` function argument.

    There are two main ways around this:
    1. Factor out the untraceable logic into a top-level function and
    use ``fx.wrap`` on it.
    2. If the control flow is static (i.e. the loop trip count is
    based on some hyperparameter), the code can be kept in its original
    position and refactored into something like::

        for i in range(self.some_hyperparameter):
            indexed_item = proxied_value[i]

    For a more detailed description into the Proxy internals, check out
    the "Proxy" section in `torch/fx/OVERVIEW.md`
    """

    @compatibility(is_backward_compatible=True)
    def __init__(self, node: Node, tracer: 'Optional[TracerBase]' = None):
        if tracer is None:
            # This allows you to create a Proxy object around a raw Node
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
        return self.tracer.create_proxy('call_method', '__call__', (self,) + args, kwargs)

    def __iter__(self) -> Iterable['Proxy']:
        frame = inspect.currentframe()
        assert frame is not None
        calling_frame = frame.f_back
        assert calling_frame is not None
        inst = list(dis.get_instructions(calling_frame.f_code))[calling_frame.f_lasti // 2]
        if inst.opname == 'UNPACK_SEQUENCE':
            return (self[i] for i in range(inst.argval))  # type: ignore[index]

        return self.tracer.iter(self)

    def __bool__(self) -> bool:
        if self.tracer.trace_asserts:
            # check if this boolean is used in an assertion, bytecode pattern for assertions
            # is pretty stable for Python 3.7--3.9
            frame = inspect.currentframe()
            assert frame is not None
            calling_frame = frame.f_back
            assert calling_frame is not None
            insts = list(dis.get_instructions(calling_frame.f_code))
            cur = calling_frame.f_lasti // 2
            inst = insts[cur]

            if inst.opname == 'POP_JUMP_IF_TRUE':
                first = insts[cur + 1]
                assert inst.arg is not None
                last = insts[inst.arg // 2 - 1]
                starts_with_assert = (first.opname == 'LOAD_GLOBAL' and first.argval == 'AssertionError'
                                      or first.opname == 'LOAD_ASSERTION_ERROR')
                if starts_with_assert and last.opname == 'RAISE_VARARGS':
                    self.tracer.create_proxy('call_function', assert_fn, (self,), {})
                    return True

        return self.tracer.to_bool(self)

    @compatibility(is_backward_compatible=True)
    def keys(self):
        return self.tracer.keys(self)

    def __len__(self):
        raise RuntimeError("'len' is not supported in symbolic tracing by default. If you want "
                           "this call to be recorded, please call torch.fx.wrap('len') at "
                           "module scope")

    @classmethod
    def __torch_function__(cls, orig_method, types, args=None, kwargs=None):
        args = args if args else ()
        kwargs = kwargs if kwargs else {}

        tracers : Dict[Any, None] = {}

        def find_tracer(a):
            if isinstance(a, cls):
                tracers[a.tracer] = None
        torch.fx.node.map_aggregate(args, find_tracer)
        torch.fx.node.map_aggregate(kwargs, find_tracer)

        if len(tracers) > 1:
            raise RuntimeError(f'Found multiple different tracers {list(tracers.keys())} while '
                               f'trying to trace operations {orig_method}')
        tracer = next(iter(tracers.keys()))

        if isinstance(orig_method, torch._C.ScriptMethod):
            args = (orig_method.owner,) + args
            return tracer.create_proxy('call_method', orig_method.name, args, kwargs)
        if torch.overrides.is_tensor_method_or_property(orig_method):
            return tracer.create_proxy('call_method', orig_method.__name__, args, kwargs)
        else:
            return tracer.create_proxy('call_function', orig_method, args, kwargs,
                                       name=tracer.graph._target_to_str(orig_method.__name__))


@compatibility(is_backward_compatible=True)
class Attribute(Proxy):
    @compatibility(is_backward_compatible=True)
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
            self._node = self.tracer.create_proxy('call_function', getattr, (self.root, self.attr), {}).node
        return self._node

    def __call__(self, *args, **kwargs):
        return self.tracer.create_proxy('call_method', self.attr, (self.root,) + args, kwargs)


@compatibility(is_backward_compatible=False)
class ParameterProxy(Proxy):
    """
    A special proxy which lets "shape", "size", "dim", and a few other
    attribute accesses pass through to the underlying  module parameter object,
    so that conditional tests on these attributes will not throw exception during tracing
    """
    def __init__(self, tracer: TracerBase, node: Node, name, param):
        super().__init__(node, tracer)
        assert(isinstance(param, torch.nn.Parameter))
        self.param = param
        self.name = name

    def __repr__(self) -> str:
        return f'ParameterProxy({self.name})'

    @property
    def shape(self):
        return self.param.shape

    def size(self):
        return self.param.size()

    def dim(self):
        return self.param.dim()

    @property
    def ndim(self):
        return self.param.ndim

    def numel(self):
        return self.param.numel()

    def nelement(self):
        return self.param.nelement()


for method in magic_methods:
    def _scope(method):
        def impl(*args, **kwargs):
            tracer = args[0].tracer
            target = getattr(operator, method)
            return tracer.create_proxy('call_function', target, args, kwargs)
        impl.__name__ = method
        as_magic = f'__{method.strip("_")}__'
        setattr(Proxy, as_magic, impl)
    _scope(method)

def _define_reflectable(orig_method_name):
    method_name = f'__r{orig_method_name.strip("_")}__'

    def impl(self, rhs):
        target = getattr(operator, orig_method_name)
        return self.tracer.create_proxy('call_function', target, (rhs, self), {})
    impl.__name__ = method_name
    impl.__qualname__ = method_name
    setattr(Proxy, method_name, impl)

for orig_method_name in reflectable_magic_methods:
    _define_reflectable(orig_method_name)
