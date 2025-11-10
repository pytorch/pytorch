# mypy: ignore-errors

import collections
import copy
import dis
import enum
import inspect
import logging
import operator
import sys
import traceback
from collections import OrderedDict
from collections.abc import Callable, Iterator
from dataclasses import fields, is_dataclass
from typing import Any, Optional

import torch
import torch.fx.traceback as fx_traceback
from torch._C import _fx_map_aggregate as map_aggregate, _fx_map_arg as map_arg
from torch._logging import getArtifactLogger
from torch.utils._traceback import CapturedTraceback

from ._compatibility import compatibility
from .graph import Graph, magic_methods, reflectable_magic_methods
from .immutable_collections import immutable_dict, immutable_list
from .node import Argument, base_types, Node, Target
from .operator_schemas import check_for_mutable_operation


__all__ = [
    "TracerBase",
    "GraphAppendingTracer",
    "TraceError",
    "Proxy",
    "MetaProxy",
    "Attribute",
    "ParameterProxy",
    "Scope",
    "ScopeContextManager",
]


log = logging.getLogger(__name__)
annotation_log = getArtifactLogger(__name__, "annotation")


@compatibility(is_backward_compatible=False)
class Scope:
    """Scope object that records the module path and the module type
    of a module. Scope is used to track the information of the module
    that contains a Node in a Graph of GraphModule. For example::

        class Sub(torch.nn.Module):
            def forward(self, x):
                # This will be a call_method Node in GraphModule,
                # scope for this would be (module_path="sub", module_type=Sub)
                return x.transpose(1, 2)


        class M(torch.nn.Module):
            def __init__(self) -> None:
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
class ScopeContextManager:
    """A context manager to track the Scope of Node during symbolic tracing.
    When entering a forward function of a Module, we'll update the scope information of
    the current module, and when we exit, we'll restore the previous scope information.
    """

    def __init__(
        self,
        scope: Scope,
        current_scope: Scope,
    ):
        super().__init__()
        # Keep a copy of prev scope to restore on exit
        self._prev_scope = copy.copy(scope)
        # Update scope to current scope
        scope.module_path = current_scope.module_path
        scope.module_type = current_scope.module_type
        # Save a reference so we can restore it
        self._scope = scope

    def __enter__(self):
        return self._scope

    def __exit__(self, *args):
        self._scope.module_path = self._prev_scope.module_path
        self._scope.module_type = self._prev_scope.module_type
        return


_COPY_META_FIELDS = [
    "nn_module_stack",
    "torch_fn",
    "source_fn_stack",
    "original_aten",
    "recompute",
    "ac_graph_id",
    "has_backward_hook",
    "from_node",
    "quantization_tag",  # TODO deprecated
    "_numeric_debug_handle",  # TODO deprecated
    "custom",
    "partitioner_tag",
]


@compatibility(is_backward_compatible=True)
class TracerBase:
    graph: Graph
    record_stack_traces: bool = False
    # When record_stack_traces is True, only reocrd stack traces
    # with forward function names.
    # This helps when we want stack trace back to model code
    _record_forward_stack_traces_only: bool = False
    # Feature flag for mutable schema checking
    # Enableby default in 1.12
    check_mutable_operations: bool = False
    # Feature flag for assert tracing
    trace_asserts: bool = False
    # Feature flag for proxying accesses to buffer values
    proxy_buffer_attributes: bool = False

    # Name of the function to be traced. It will only be used when
    # ``root`` is an instance of ``nn.Module``
    traced_func_name: str = "forward"

    # Maps the containing module's name to the operator name
    scope: Scope

    # Records the module call stack
    module_stack: OrderedDict[str, tuple[str, Any]]

    # Mapping of node name to module scope
    node_name_to_scope: dict[str, tuple[str, type]]

    @compatibility(is_backward_compatible=True)
    def create_node(
        self,
        kind: str,
        target: Target,
        args: tuple[Argument, ...],
        kwargs: dict[str, Argument],
        name: Optional[str] = None,
        type_expr: Optional[Any] = None,
    ) -> Node:
        """
        Inserts a graph node given target, args, kwargs, and name.

        This method can be overridden to do extra checking, validation, or
        modification of values used in node creation. For example, one might
        want to disallow in-place operations from being recorded.
        """

        if kind == "call_function" and self.check_mutable_operations:
            check_for_mutable_operation(target, args, kwargs)

        node = self.graph.create_node(kind, target, args, kwargs, name, type_expr)
        # TODO node_name_to_scope will be depreciated in favor of
        # node.meta['nn_module_stack']
        self.node_name_to_scope[node.name] = (
            self.scope.module_path,
            self.scope.module_type,
        )

        # Optionally set stack trace on the created Node for debugging purposes
        if fx_traceback.has_preserved_node_meta():
            current_meta: dict[str, Any] = fx_traceback.get_current_meta()

            stack_trace = current_meta.get("stack_trace")
            if stack_trace:
                node.stack_trace = stack_trace
            # Explicitly set the stack_trace, nn_module_stack and source_fn on the node.meta
            # If other meta fields are needed, they can be added here
            for field in _COPY_META_FIELDS:
                if field in current_meta:
                    node.meta[field] = copy.copy(current_meta[field])

            # Here we decrement to account for the sequence_nr having
            # just been incremented while tracing this lowered aten op.
            new_seq_nr = torch.autograd._get_sequence_nr() - 1
            # The sequence_nr increments every time a new autograd Node
            # is created. During the FWD pass we store the sequence_nr
            # corresponding to the last autograd Node created on this fx
            # node's meta.  A single aten op can create multiple autograd
            # nodes as is the case with in-place foreach ops. During the
            # BWD pass we retrieve the sequence_nr stored on the current
            # executing autograd Node. See NOTE [ Sequence Number ].
            if current_meta.get("in_grad_fn", 0) > 0:
                annotation_log.debug("seq_nr from current_meta")
                new_seq_nr = current_meta["grad_fn_seq_nr"][-1]

            # See Note [Functionalization View Replay Annotation]
            # Overriding some node meta with the original node meta of the
            # regenerated node.
            replay_node: Node = fx_traceback.get_current_replay_node()
            if replay_node is not None:
                node.meta["is_functional_regenerated"] = True
                if "seq_nr" in replay_node.meta:
                    annotation_log.debug("seq_nr from replay_node")
                    new_seq_nr = replay_node.meta["seq_nr"]
                if "custom" in replay_node.meta:
                    node.meta["custom"] = replay_node.meta.get("custom")
                if "stack_trace" in replay_node.meta:
                    node.stack_trace = replay_node.meta.get("stack_trace")

            annotation_log.debug("Assigning new_seq_nr %s to %s", new_seq_nr, node.name)
            node.meta["seq_nr"] = new_seq_nr

        elif self.module_stack:
            node.meta["nn_module_stack"] = copy.copy(self.module_stack)

        if self.record_stack_traces and not node.stack_trace:
            user_stack_summary = CapturedTraceback.extract().summary()
            if user_stack_summary:
                user_stack_summary = self._filter_traceback_frames(user_stack_summary)
                if user_stack_summary:
                    node.stack_trace = "".join(user_stack_summary.format()).strip()

        log.debug("create_node %s", node)
        return node

    def _filter_traceback_frames(
        self, user_stack_summary: traceback.StackSummary
    ) -> traceback.StackSummary:
        # This method can be overridden to customize the frame filtering logic
        # for the recorded stack trace
        user_frames = []
        if self._record_forward_stack_traces_only:
            user_frames = [
                frame
                for frame in user_stack_summary
                if (
                    frame.name == "forward"
                    or frame.filename.endswith("torch/__init__.py")
                )
            ]
        else:
            first_forward = -1
            for i, frame in enumerate(user_stack_summary):
                if frame.name == "forward":
                    user_frames = user_stack_summary[i:]
                    first_forward = i
                    break

            # Not having a "forward" call in the stacktrace implies the
            # stacktrace will probably be irrelevant
            if first_forward == -1:
                user_frames = []

        from torch.fx.experimental.symbolic_shapes import uninteresting_files

        user_frames = [
            frame
            for frame in user_frames
            if frame.filename not in uninteresting_files()
        ]

        return traceback.StackSummary.from_list(user_frames)

    @compatibility(is_backward_compatible=True)
    def proxy(self, node: Node) -> "Proxy":
        return Proxy(node, self)

    @compatibility(is_backward_compatible=True)
    def create_proxy(
        self,
        kind: str,
        target: Target,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        name: Optional[str] = None,
        type_expr: Optional[Any] = None,
        # fix noqa when updating bc tests
        proxy_factory_fn: Callable[[Node], "Proxy"] = None,  # noqa: RUF013
    ):
        """
        Create a Node from the given arguments, then return the Node
        wrapped in a Proxy object.

        If kind = 'placeholder', then we're creating a Node that
        represents the parameter of a function. If we need to encode
        a default parameter, we use the ``args`` tuple. ``args`` is
        otherwise empty for ``placeholder`` Nodes.
        """

        args_ = self.create_arg(args)
        kwargs_ = self.create_arg(kwargs)
        assert isinstance(args_, tuple)
        assert isinstance(kwargs_, dict)

        node = self.create_node(kind, target, args_, kwargs_, name, type_expr)

        if not proxy_factory_fn:
            proxy = self.proxy(node)
        else:
            proxy = proxy_factory_fn(node)

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

        pt_files = [
            "torch/fx/proxy.py",
            "torch/fx/_symbolic_trace.py",
            "torch/fx/experimental/proxy_tensor.py",
            "torch/_ops.py",
            "torch/_tensor.py",
            "torch/utils/_python_dispatch.py",
            "torch/_prims_common/wrappers.py",
            "torch/_refs/__init__.py",
            "torch/_refs/nn/functional/__init__.py",
            "torch/utils/_stats.py",
        ]
        while frame:
            frame = frame.f_back
            if frame and all(
                not frame.f_code.co_filename.endswith(file) for file in pt_files
            ):
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
        # IMPORTANT: Are you here because you are trying to proxy a new type into
        # the graph? Please Please Please contact someone on the PyTorch Compiler team;
        # the considerations are subtle.
        #
        # 1) When you add a new type, all of the downstream consumers and pass writers
        # need to handle the new type. torch.fx is intended to be easy to write
        # passes for, so we will push back against new types.
        # 2) In torch.compile's IR, there are only specific operations that go
        # into the graph. In particular, Tensor operations should go into the graph,
        # but non-Tensor operations shouldn't. What that means is that constructors
        # for new types *SHOULD NOT* become nodes in the FX graph.
        handler = _create_arg_bypass.get(type(a))
        if handler is not None:
            # this is just a performance optimization and can be removed if needed
            # for common types, we have a fast path to avoid isinstance() overhead
            # this doesn't remove the checks below since we need to handle subclasses
            return handler(self, a)

        if isinstance(a, Proxy):
            return a.node  # most common arg type goes first
        elif hasattr(a, "__fx_create_arg__"):
            return a.__fx_create_arg__(self)
        # aggregates
        elif isinstance(a, tuple):
            if hasattr(a, "_fields"):
                # NamedTuple constructors don't seem to like getting a generator
                # expression as an argument to their constructor, so build this
                # intermediate tuple and unpack it into the NamedTuple constructor
                args = [self.create_arg(elem) for elem in a]
                return type(a)(*args)  # type: ignore[arg-type]
            return type(a)([self.create_arg(elem) for elem in a])
        elif isinstance(a, list):
            return [self.create_arg(elem) for elem in a]
        elif isinstance(a, dict):
            return _create_arg_dict(self, a)
        elif isinstance(a, slice):
            return slice(
                self.create_arg(a.start),
                self.create_arg(a.stop),
                self.create_arg(a.step),
            )

        elif isinstance(a, range):
            return range(
                self.create_arg(a.start),
                self.create_arg(a.stop),
                self.create_arg(a.step),
            )

        elif isinstance(a, (torch._ops.OpOverload, torch._ops.HigherOrderOperator)):
            return a

        elif is_dataclass(a):
            kwargs = {
                field.name: self.create_arg(getattr(a, field.name))
                for field in fields(a)
            }
            return self.create_node("call_function", a.__class__, (), kwargs)

        elif isinstance(a, (*base_types, enum.Enum)) or a is None or a is ...:
            return a

        raise NotImplementedError(f"argument of type: {type(a)}")

    @compatibility(is_backward_compatible=True)
    def to_bool(self, obj: "Proxy") -> bool:
        """Called when a proxy object is being converted to a boolean, such as
        when used in control flow.  Normally we don't know what to do because
        we don't know the value of the proxy, but a custom tracer can attach more
        information to the graph node using create_node and can choose to return a value.
        """
        raise TraceError(
            "symbolically traced variables cannot be used as inputs to control flow"
        )

    @compatibility(is_backward_compatible=True)
    def iter(self, obj: "Proxy") -> Iterator:
        """Called when a proxy object is being iterated over, such as
        when used in control flow.  Normally we don't know what to do because
        we don't know the value of the proxy, but a custom tracer can attach more
        information to the graph node using create_node and can choose to return an iterator.
        """
        raise TraceError(
            "Proxy object cannot be iterated. This can be "
            "attempted when the Proxy is used in a loop or"
            " as a *args or **kwargs function argument. "
            "See the torch.fx docs on pytorch.org for a "
            "more detailed explanation of what types of "
            "control flow can be traced, and check out the"
            " Proxy docstring for help troubleshooting "
            "Proxy iteration errors"
        )

    @compatibility(is_backward_compatible=True)
    def keys(self, obj: "Proxy") -> Any:
        """Called when a proxy object is has the keys() method called.
        This is what happens when ** is called on a proxy. This should return an
        iterator it ** is suppose to work in your custom tracer.
        """
        return Attribute(obj, "keys")()


# used in Proxy object when just appending to the graph while not tracing.
@compatibility(is_backward_compatible=True)
class GraphAppendingTracer(TracerBase):
    def __init__(self, graph: Graph):
        super().__init__()
        self.graph = graph
        self.scope = Scope("", None)
        self.module_stack = collections.OrderedDict()
        self.node_name_to_scope = {}


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
    the "Proxy" section in `torch/fx/README.md`
    """

    @compatibility(is_backward_compatible=True)
    def __init__(self, node: Node, tracer: "Optional[TracerBase]" = None):
        if tracer is None:
            # This allows you to create a Proxy object around a raw Node
            tracer = GraphAppendingTracer(node.graph)
        self.tracer = tracer
        self.node = node

    def __repr__(self) -> str:
        return f"Proxy({self.node.name})"

    def __getattr__(self, k) -> "Attribute":
        # note: not added to the graph yet, if this is a method call
        # we peephole optimize to the method invocation
        return Attribute(self, k)

    def __getstate__(self) -> dict:
        return self.__dict__

    def __deepcopy__(self, memo) -> dict:
        # We have to explicitly override this method, because otherwise deepcopy
        # will go to __getattr__(self, "__deepcopy__") and return a
        # Attribute(__deepcopy__), and may go into an infinite loop in some cases.
        import copy

        new_dict = {}
        for k, v in self.__dict__.items():
            try:
                new_obj = copy.deepcopy(v, memo)
            except Exception:
                log.warning(
                    "Shallow copy %s of Proxy because it cannot be deepcopied. "
                    "Proxy is created for node %s",
                    k,
                    self.node.name,
                )
                new_obj = copy.copy(v)
            new_dict[k] = new_obj
        assert "node" in new_dict
        assert "tracer" in new_dict
        new_proxy = Proxy(new_dict["node"], new_dict["tracer"])
        for k, v in new_dict.items():
            new_proxy.__dict__[k] = v
        return new_proxy

    def __setstate__(self, d):
        # This is called when being unpickled/loaded.
        self.__dict__ = d

    def __call__(self, *args, **kwargs) -> "Proxy":
        return self.tracer.create_proxy(
            "call_method", "__call__", (self,) + args, kwargs
        )

    def __iter__(self) -> Iterator["Proxy"]:
        frame = inspect.currentframe()
        assert frame is not None
        calling_frame = frame.f_back
        assert calling_frame is not None
        inst_list = list(dis.get_instructions(calling_frame.f_code))
        if sys.version_info >= (3, 11):
            from bisect import bisect_left

            inst_idx = bisect_left(
                inst_list, calling_frame.f_lasti, key=lambda x: x.offset
            )
        else:
            inst_idx = calling_frame.f_lasti // 2
        inst = inst_list[inst_idx]
        if inst.opname == "UNPACK_SEQUENCE":
            return (self[i] for i in range(inst.argval))  # type: ignore[index]

        return self.tracer.iter(self)

    def __abs__(self):
        return self.tracer.create_proxy("call_function", operator.abs, (self,), {})

    def __bool__(self) -> bool:
        if self.tracer.trace_asserts:
            # check if this boolean is used in an assertion, bytecode pattern for assertions
            # is pretty stable for Python 3.7--3.9
            frame = inspect.currentframe()
            assert frame is not None
            calling_frame = frame.f_back
            assert calling_frame is not None
            insts = list(dis.get_instructions(calling_frame.f_code))
            if sys.version_info >= (3, 11):
                from bisect import bisect_left

                cur = bisect_left(insts, calling_frame.f_lasti, key=lambda x: x.offset)
            else:
                cur = calling_frame.f_lasti // 2
            inst = insts[cur]

            if inst.opname == "POP_JUMP_IF_TRUE":
                first = insts[cur + 1]
                assert inst.arg is not None
                last = insts[inst.arg // 2 - 1]
                starts_with_assert = (
                    first.opname == "LOAD_GLOBAL"
                    and first.argval == "AssertionError"
                    or first.opname == "LOAD_ASSERTION_ERROR"
                )
                if starts_with_assert and last.opname == "RAISE_VARARGS":
                    self.tracer.create_proxy("call_function", assert_fn, (self,), {})
                    return True

        return self.tracer.to_bool(self)

    @compatibility(is_backward_compatible=True)
    def keys(self):
        return self.tracer.keys(self)

    def __len__(self):
        raise RuntimeError(
            "'len' is not supported in symbolic tracing by default. If you want "
            "this call to be recorded, please call torch.fx.wrap('len') at "
            "module scope"
        )

    @classmethod
    def __torch_function__(cls, orig_method, types, args=None, kwargs=None):
        args = args if args else ()
        kwargs = kwargs if kwargs else {}

        tracers: dict[Any, None] = {}

        def find_tracer(a):
            if isinstance(a, cls):
                tracers[a.tracer] = None

        map_aggregate(args, find_tracer)
        map_aggregate(kwargs, find_tracer)

        if len(tracers) > 1:
            raise RuntimeError(
                f"Found multiple different tracers {list(tracers.keys())} while "
                f"trying to trace operations {orig_method}"
            )
        tracer = next(iter(tracers.keys()))

        if isinstance(orig_method, torch._C.ScriptMethod):
            args = (orig_method.owner,) + args
            return tracer.create_proxy("call_method", orig_method.name, args, kwargs)
        if torch.overrides.is_tensor_method_or_property(orig_method):
            return tracer.create_proxy(
                "call_method", orig_method.__name__, args, kwargs
            )
        else:
            if isinstance(orig_method, torch._ops.HigherOrderOperator):
                # TODO: Define how to symbolically trace HigherOrderOperators
                raise RuntimeError("Unable to symbolically trace HigherOrderOperators")
            return tracer.create_proxy(
                "call_function",
                orig_method,
                args,
                kwargs,
                name=tracer.graph._target_to_str(orig_method.__name__),
            )


@compatibility(is_backward_compatible=False)
class MetaProxy(Proxy):
    """
    A Proxy subclass that propagates metadata (meta['val']) during graph tracing.
    """

    def __init__(
        self, node: Node, tracer: "Optional[TracerBase]" = None, fake_mode=None
    ):
        super().__init__(node, tracer)
        self.fake_mode = fake_mode

    def __repr__(self) -> str:
        return f"MetaProxy({self.node.name})"

    @classmethod
    def __torch_function__(cls, orig_method, types, args=None, kwargs=None):
        args = args if args else ()
        kwargs = kwargs if kwargs else {}

        meta_proxy = None
        for arg in args:
            if isinstance(arg, MetaProxy):
                meta_proxy = arg
                break

        assert meta_proxy is not None, (
            "No MetaProxy found in arguments, but one is expected."
        )

        proxy = super().__torch_function__(orig_method, types, args, kwargs)
        with meta_proxy.fake_mode:
            proxy.node.meta["val"] = orig_method(
                *[a.node.meta["val"] if isinstance(a, Proxy) else a for a in args],
                **kwargs,
            )
        return MetaProxy(proxy.node, proxy.tracer, meta_proxy.fake_mode)


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
            self._node = self.tracer.create_proxy(
                "call_function", getattr, (self.root, self.attr), {}
            ).node
        return self._node

    def __call__(self, *args, **kwargs):
        return self.tracer.create_proxy(
            "call_method", self.attr, (self.root,) + args, kwargs
        )


@compatibility(is_backward_compatible=False)
class ParameterProxy(Proxy):
    """
    A special proxy which lets "shape", "size", "dim", and a few other
    attribute accesses pass through to the underlying  module parameter object,
    so that conditional tests on these attributes will not throw exception during tracing
    """

    def __init__(self, tracer: TracerBase, node: Node, name, param):
        super().__init__(node, tracer)
        assert isinstance(param, torch.nn.Parameter)
        self.param = param
        self.name = name

    def __repr__(self) -> str:
        return f"ParameterProxy({self.name})"

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
            return tracer.create_proxy("call_function", target, args, kwargs)

        impl.__name__ = method
        as_magic = f"__{method.strip('_')}__"
        setattr(Proxy, as_magic, impl)

    _scope(method)


def _define_reflectable(orig_method_name):
    method_name = f"__r{orig_method_name.strip('_')}__"

    def impl(self, rhs):
        target = getattr(operator, orig_method_name)
        return self.tracer.create_proxy("call_function", target, (rhs, self), {})

    impl.__name__ = method_name
    impl.__qualname__ = method_name
    setattr(Proxy, method_name, impl)


for orig_method_name in reflectable_magic_methods:
    _define_reflectable(orig_method_name)


def _no_nodes_error(arg):
    raise RuntimeError(
        "Keys for dictionaries used as an argument cannot contain a "
        f"Node. Got key: {arg}"
    )


def _create_arg_dict(self, a):
    r = {}
    for k, v in a.items():
        if not isinstance(k, str):
            # Check for invalid dict keys. We do not want a Proxy to appear
            # anywhere within the key. Since keys can be collection types,
            # we iterate through the key with map_arg
            k = self.create_arg(k)
            map_arg(k, _no_nodes_error)
        r[k] = self.create_arg(v)
    return r


_create_arg_bypass = {
    t: lambda self, a: a
    for t in [
        *base_types,
        type(None),
        type(...),
        torch._ops.OpOverload,
        torch._ops.HigherOrderOperator,
    ]
}
_create_arg_bypass[Proxy] = lambda self, a: a.node
_create_arg_bypass[tuple] = lambda self, a: tuple(self.create_arg(elem) for elem in a)
_create_arg_bypass[list] = lambda self, a: [self.create_arg(elem) for elem in a]
_create_arg_bypass[dict] = _create_arg_dict
_create_arg_bypass[immutable_list] = _create_arg_bypass[list]
_create_arg_bypass[immutable_dict] = _create_arg_bypass[dict]
