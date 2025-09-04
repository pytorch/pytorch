# mypy: allow-untyped-defs
"""
This file provides a number of "global" variables/handlers that are actually
thread local and dynamically scoped, with Inductor patching them to various
implementations depending on the situation.

These handlers are interacted with in a fairly stylized way.  Typically,
we will import V from this module::

    from .virtualized import V

Various handlers are accessible as attributes on this module; for example,
you might access ``V.graph.sizevars.size_hint`` to resolve a size hint associated with
a number.

There are a few distinct usage patterns for virtualized global variables:

1. Implicit argument passing.  Examples: ``V.current_node``, ``V.aot_compilation``.
   Use ``V.set_current_node`` to change what the current node is while we're
   executing some region of code, so code inside that region can query ``V.current_node``
   to find out what it is.  This is often more convenient than manually threading
   the current node as an argument through all call stacks.

2. Per-compilation global state.  Examples: ``V.fake_mode``, ``V.graph``.  For a
   given ``compile_fx`` invocation, these typically don't change, but they are
   associated with some internal state so they cannot just be global functions.
   We install these objects at the beginning of compilation and then you can
   conveniently access them without having to pass them around.

3. Alternate define-by-run interpretations.  Examples: ``V.ops``, ``V.kernel``.
   A commonly used IR in Inductor is define-by-run: instead of maintaining
   explicit syntax data structures, we instead represent loop bodies as
   callable functions, which internally invoke operations defined on
   ``V.ops``.  To perform semantic analysis, print or code generate these
   operations, we dynamically patch ``V.ops`` with an alternate handler with
   the intended semantics and then run the callable function.  For example, to
   extract out a traditional (FX) graph representation of the define-by-run
   IR, simply install a handler that records each ``ops`` call to a graph.

   TODO: Define a parent class / protocol that defines all of the operations
   V.ops is expected to support.

It is typically an error to access a virtualized global without having installed
an appropriate handler (you will get a NullHandler), although in some cases we
provide a default implementation.

One last thing: although most virtualized globals are accessed via ``V``, ``ops`` is
ubiquitous enough to have its own top level variable, so you will typically see
``ops.constant(...)`` rather than ``V.ops.constant(...)``.  In fact, these are not
equivalent; the former interface supports arithmetic overloads like ``x + y``
instead of forcing ``ops.add(x, y)``, so it should be preferred.

Some operators are seemingly unused, but they are implicitly used by ops_wrapper.
In particular, we typically have an operator for every basic pointwise PyTorch operation
supported.
"""

from __future__ import annotations

from contextlib import AbstractContextManager, contextmanager
from threading import local
from typing import Any, Callable, cast, Generic, TYPE_CHECKING, TypeVar, Union

from torch.utils._ordered_set import OrderedSet

from .ops_handler import (  # noqa: F401
    DefaultHandler,
    KernelFormatterHandler,
    MockHandler,
    OpsHandler,
    ReductionType,
    StoreMode,
    WrapperHandler,
)


if TYPE_CHECKING:
    import torch
    from torch._inductor.choices import InductorChoices
    from torch._inductor.codegen.cpp_utils import LocalBufferContext
    from torch._inductor.debug import DebugContext
    from torch._inductor.graph import GraphLowering
    from torch._inductor.ir import ExternKernelNode
    from torch._inductor.loop_body import InterpreterShim
    from torch._subclasses import FakeTensorMode

threadlocal = local()

T = TypeVar("T")


class NullHandler:
    """
    Sentinel indicating that a global variable is unset ala None.  Typically,
    attempting to access the global variable before it's set is an error, but with
    NullHandler it won't fail until you try to access an attribute on it.
    """


# If a virtualized value is set to _PoisonedVirtual then any attempt to get the
# value will result an an exception being raised. This is useful if we want to
# trap uninitialized reads of virtualized globals - for example when compiling
# in a subprocess we don't want the child reading globals that weren't copied
# from the parent.
_PoisonedVirtual = object()


class Virtualized(Generic[T]):
    """
    Implements a global variable that redirects via thread local variable
    (NB: construct this class to create the global variable; this is not
    a singleton class!)

    This allows us to swap in different op implementations in codegen.

    NB: Despite the fact that we typically call these "handlers" (e.g., NullHandler is
    the default value of the variable), we sometimes use these variables to
    store other things, like booleans.
    """

    def __init__(self, vname: str, default: Union[Callable[[], T], type[NullHandler]]):
        self._vname = vname
        self._key: str = f"__torchinductor_{vname}"
        self._default = default

    def _set_handler(self, value: T) -> AbstractContextManager[None]:
        prior = self._get_handler(False)
        setattr(threadlocal, self._key, value)

        @contextmanager
        def ctx():
            try:
                yield
            finally:
                self._set_handler(prior)

        return ctx()

    def _get_handler(self, check_poisoned: bool = True) -> T:
        try:
            value = getattr(threadlocal, self._key)
            if check_poisoned and value is _PoisonedVirtual:
                raise RuntimeError(
                    f"Attempt to use poisoned virtualized value '{self._vname}'."
                )
            return value
        except AttributeError:
            # TODO: To be honest, I feel we probably should just error in this
            # case, instead of making a null handler that will probably error
            # when you getattr on it
            return self._default()  # type: ignore[return-value]

    def __getattr__(self, name: str) -> Any:
        return getattr(self._get_handler(), name)


class NullKernelHandler(NullHandler):
    """
    We need access `V.kernel.removed_buffers` in DeferredLine class when there
    is no kernel in the context. This happens when codegening the wrapper.
    Initialize `removed_buffers` and `inplaced_to_remove` explicitly so we don't
    need call 'getattr' with default value which is error prone to typo in
    attribute name.
    """

    def __init__(self):
        super().__init__()
        self.removed_buffers = OrderedSet[Any]()
        self.inplaced_to_remove = OrderedSet[Any]()
        self.index_dtype = "tl.int64"

    def get_index_dtype_as_torch_dtype(self):
        import torch

        if self.index_dtype == "tl.int64":
            return torch.int64
        elif self.index_dtype == "tl.int32":
            return torch.int32
        else:
            raise ValueError(f"Unknown dtype: {self.index_dtype}")


_ops: Virtualized[OpsHandler[Any]] = Virtualized(
    "ops", cast(type[OpsHandler[Any]], MockHandler)
)
_graph: Virtualized[GraphLowering] = Virtualized("graph", NullHandler)
_extern_kernel_nodes: Virtualized[list[ExternKernelNode]] = Virtualized(
    "extern_kernel_nodes", NullHandler
)
_real_inputs: Virtualized[list[torch.Tensor]] = Virtualized("real_inputs", NullHandler)
_fake_mode: Virtualized[FakeTensorMode] = Virtualized("fake_mode", NullHandler)
_kernel: Virtualized[NullKernelHandler] = Virtualized(
    "kernel", NullKernelHandler
)  # TODO: improve type
_debug: Virtualized[DebugContext] = Virtualized("debug", NullHandler)
_interpreter: Virtualized[InterpreterShim] = Virtualized("interpreter", NullHandler)
_aot_compilation: Virtualized[bool] = Virtualized("aot_compilation", NullHandler)
_current_node: Virtualized[torch.fx.Node] = Virtualized("current_node", NullHandler)
_local_buffer_context: Virtualized[LocalBufferContext] = Virtualized(
    "local_buffer_context", NullHandler
)


def _choices_default():
    """
    Lazy init the global choices handler

    We virtualize InductorChoices to allow changing inductor heuristics from out of tree.
    """
    from torch._inductor.choices import InductorChoices

    rv = InductorChoices()
    setattr(threadlocal, _choices._key, rv)
    return rv


_choices: Virtualized[InductorChoices] = Virtualized("choices", _choices_default)


class OpsValue:
    """The return type of most ops calls.

    This exists so we can overload magic methods, and write mathematical
    expressions much more fluently. So instead of

        ops.add(ops.mul(ops.mul(ops.sub(ops.mul(_Ap2, x), _Ap3), x), x), _1)

    we can write

        (_Ap2 * x - _Ap3) * x * x + _1

    """

    value: Any

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return f"OpsValue({self.value!r})"

    def __add__(self, other):
        return ops.add(self, other)

    def __mul__(self, other):
        return ops.mul(self, other)

    def __sub__(self, other):
        return ops.sub(self, other)

    def __neg__(self):
        return ops.neg(self)

    def __truediv__(self, other):
        return ops.truediv(self, other)

    def __floordiv__(self, other):
        return ops.floordiv(self, other)

    def __mod__(self, other):
        return ops.mod(self, other)

    def __pow__(self, other):
        return ops.pow(self, other)

    def __lt__(self, other):
        return ops.lt(self, other)

    def __le__(self, other):
        return ops.le(self, other)

    def __eq__(self, other):
        return ops.eq(self, other)

    def __ne__(self, other):
        return ops.ne(self, other)

    def __gt__(self, other):
        return ops.gt(self, other)

    def __ge__(self, other):
        return ops.ge(self, other)

    def __and__(self, other):
        return ops.bitwise_and(self, other)

    def __or__(self, other):
        return ops.bitwise_or(self, other)

    def __xor__(self, other):
        return ops.bitwise_xor(self, other)

    def __invert__(self):
        return ops.bitwise_not(self)

    def __rshfit__(self, n):
        return ops.bitwise_right_shift(self, n)

    def __lshift__(self, n):
        return ops.bitwise_left_shift(self, n)


class OpsWrapper(DefaultHandler):
    """This wraps any returned IR values into an `OpsValue` instance, so that we
    can overload the magic methods for writing mathematical expressions fluently.
    """

    def _default(self, name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        new_args = [OpsWrapper._unwrap(a) for a in args]
        new_kwargs = {k: OpsWrapper._unwrap(v) for k, v in kwargs.items()}
        return OpsWrapper._wrap(getattr(_ops, name)(*new_args, **new_kwargs))

    @staticmethod
    def _unwrap(x):
        if isinstance(x, (list, tuple)):
            return tuple(OpsWrapper._unwrap(v) for v in x)
        if isinstance(x, OpsValue):
            return x.value
        return x

    @staticmethod
    def _wrap(x):
        if isinstance(x, (list, tuple)):
            return tuple(OpsValue(v) for v in x)
        return OpsValue(x)

    @staticmethod
    def indirect_indexing(index, size, check=True, wrap_neg=True):
        # Returns a sympy value, not IR value
        index = OpsWrapper._unwrap(index)
        return _ops.indirect_indexing(index, size, check, wrap_neg)


ops: OpsHandler[Any] = OpsWrapper()


class _V:
    MockHandler = MockHandler
    KernelFormatterHandler = KernelFormatterHandler
    WrapperHandler = WrapperHandler

    set_ops_handler: Callable[[OpsHandler[Any]], AbstractContextManager[None]] = (
        _ops._set_handler
    )
    get_ops_handler: Callable[[], OpsHandler[Any]] = _ops._get_handler
    set_graph_handler: Callable[[GraphLowering], Any] = _graph._set_handler
    set_extern_kernel_nodes: Callable[[list[ExternKernelNode]], Any] = (
        _extern_kernel_nodes._set_handler
    )
    set_real_inputs: Callable[[Any], Any] = _real_inputs._set_handler
    get_real_inputs: Callable[[], Any] = _real_inputs._get_handler
    set_fake_mode: Callable[[Any], Any] = _fake_mode._set_handler
    get_fake_mode: Callable[[], Any] = _fake_mode._get_handler
    set_kernel_handler: Callable[[Any], Any] = _kernel._set_handler
    set_debug_handler: Callable[[Any], Any] = _debug._set_handler
    set_interpreter_handler: Callable[[Any], Any] = _interpreter._set_handler
    set_aot_compilation: Callable[[bool], Any] = _aot_compilation._set_handler
    get_aot_compilation: Callable[[], Any] = _aot_compilation._get_handler
    set_current_node: Callable[[Any], Any] = _current_node._set_handler
    get_current_node: Callable[[], Any] = _current_node._get_handler
    set_local_buffer_context: Callable[[Any], Any] = _local_buffer_context._set_handler
    get_local_buffer_context: Callable[[], Any] = _local_buffer_context._get_handler
    set_choices_handler: Callable[[Any], Any] = _choices._set_handler

    @property
    def ops(self) -> OpsHandler[Any]:
        """The operator handler specific to the current codegen task"""
        return _ops._get_handler()

    @property
    def graph(self) -> GraphLowering:
        """The graph currently being generated"""
        return _graph._get_handler()

    @property
    def extern_kernel_nodes(self) -> list[ExternKernelNode]:
        """
        The extern_kernel_nodes needed for the entire graph, including the
        subgraphs.
        See `ProxyExecutor Design Note` in ir.py for more details
        """
        return _extern_kernel_nodes._get_handler()

    @property
    def real_inputs(self):
        """non-fake example inputs"""
        return _real_inputs._get_handler()

    @property
    def fake_mode(self):
        """The graph currently being generated"""
        return _fake_mode._get_handler()

    @property
    def kernel(self):
        """The kernel currently being generated"""
        return _kernel._get_handler()

    @property
    def debug(self):
        return _debug._get_handler()

    @property
    def interpreter(self):
        return _interpreter._get_handler()

    @property
    def aot_compilation(self):
        return _aot_compilation._get_handler() is True

    @property
    def current_node(self):
        return _current_node._get_handler()

    @property
    def local_buffer_context(self):
        return _local_buffer_context._get_handler()

    @property
    def choices(self) -> InductorChoices:
        return _choices._get_handler()


V = _V()
