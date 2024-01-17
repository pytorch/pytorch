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

3. Alternate define-by-run interpretations.  Examples: ``V.ops``, ``V.kernel``.  Inductor's
   IR is define-by-run: instead of maintaining explicit syntax data structures,
   we instead represent loop bodies as callable functions, which internally invoke
   operations defined on ``V.ops``.  To perform semantic analysis, print or code
   generate these operations, we dynamically patch ``V.ops`` with an alternate
   handler with the intended semantics and then run the callable function.
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
"""

from __future__ import annotations

import itertools
from contextlib import AbstractContextManager, contextmanager
from itertools import chain
from threading import local
from typing import Any, Callable, Generic, List, Type, TYPE_CHECKING, TypeVar, Union
from unittest.mock import patch

import sympy

from torch._inductor.utils import IndentedBuffer

from torch.fx.graph import inplace_methods, magic_methods

from .utils import reduction_num_outputs, sympy_str, sympy_symbol

if TYPE_CHECKING:
    import torch
    from torch._inductor.debug import DebugContext
    from torch._inductor.graph import GraphLowering
    from torch._inductor.ir import InterpreterShim
    from torch._subclasses import FakeTensorMode

threadlocal = local()

T = TypeVar("T")


class NullHandler:
    """
    Sentinel indicating that a global variable is unset ala None.  Typically,
    attempting to access the global variable when it is an error, but with
    NullHandler it won't fail until you try to access an attribute on it.
    """

    pass


class Virtualized(Generic[T]):
    """
    Implements a global variable that redirects via thread local variable
    (NB: construct this class to create the global variable; this is not
    a singleton class!)

    This allows us to swap in different op implementations in codegen.

    NB: Despite the handler naming, we sometimes use these variables to
    store other things, like booleans.
    """

    def __init__(self, vname: str, default: Union[Callable[[], T], Type[NullHandler]]):
        self._key: str = f"__torchinductor_{vname}"
        self._default = default

    def _set_handler(self, value: T) -> AbstractContextManager[None]:
        prior = self._get_handler()
        setattr(threadlocal, self._key, value)

        @contextmanager
        def ctx():
            try:
                yield
            finally:
                self._set_handler(prior)

        return ctx()

    def _get_handler(self) -> T:
        try:
            return getattr(threadlocal, self._key)
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
        self.removed_buffers = set()
        self.inplaced_to_remove = set()
        self.index_dtype = "tl.int64"


def _arg_str(a) -> str:
    if isinstance(a, sympy.Expr):
        return sympy_str(a)
    return str(a)


class MockHandler:
    def __getattr__(self, name):
        if name == "name":
            return "MockHandler"

        def inner(*args, **kwargs):
            fargs = [_arg_str(a) for a in args]
            fargs.extend(f"{k}={v}" for k, v in kwargs.items())
            return f"ops.{name}({', '.join(fargs)})"

        return inner

    def reduction(
        self, dtype, src_dtype, reduction_type, value
    ) -> Union[tuple[str, ...], str]:
        r = self.__getattr__("reduction")(dtype, src_dtype, reduction_type, value)
        num_values = reduction_num_outputs(reduction_type)
        return [f"{r}[{i}]" for i in range(num_values)] if num_values > 1 else r

    @staticmethod
    def masked(mask, body, other) -> str:
        return f"ops.masked({mask}, {body()}, {other})"

    @staticmethod
    def indirect_indexing(index_var, size, check=True) -> sympy.Symbol:
        return sympy_symbol(f"({str(index_var)})")

    @classmethod
    def _init_cls(cls):
        def make_handler(format_string):
            @staticmethod  # type: ignore[misc]
            def inner(*args):
                return format_string.format(*args)

            return inner

        for name, format_string in chain(
            magic_methods.items(), inplace_methods.items()
        ):
            setattr(cls, name, make_handler(format_string))


class TrivialOpsHandler:
    """
    This is intended to be used as a superclass for ops handlers which do not
    need to do forward dataflow analysis.  This always returns None for
    any operation; the intention is you override it and have various operations
    perform side effects.
    """
    def __getattr__(self, name):
        if name == "name":
            return "TrivialOpsHandler"

        def inner(*args, **kwargs):
            return None

        return inner

    def reduction(
        self, dtype, src_dtype, reduction_type, value
    ) -> Union[tuple[str, ...], str]:
        num_values = reduction_num_outputs(reduction_type)
        return [None] * num_values if num_values > 1 else r

    @staticmethod
    def masked(mask, body, other) -> str:
        # NB: body is a nested function, so call it so we run analysis on it
        body()
        return None

    @staticmethod
    def indirect_indexing(index_var, size, check=True) -> sympy.Symbol:
        # The sympy.Symbol here is meaningless, we just return something to
        # ensure subsequent computation with sympy expressions doesn't choke
        # (as we don't control sympy IR functions)
        return sympy_symbol(f"({str(index_var)})")


class KernelFormatterHandler:
    def __init__(self, parent_handler):
        self.parent_handler = parent_handler
        self.output = IndentedBuffer(1)
        self.var_counter = itertools.count()

    @staticmethod
    def ir_to_string(ir_fn, index, rindex=None) -> str:
        from .ir import FlexibleLayout

        args = [index, rindex] if rindex is not None else [index]
        names = ["index", "rindex"] if rindex is not None else ["index"]
        formatter = KernelFormatterHandler(MockHandler())

        with formatter.output.indent(-1):
            formatter.output.writeline(f"def inner_fn({', '.join(names)}):")
        for name, arg in zip(names, args):
            if arg:
                lhs = ", ".join(
                    [
                        str("_" if isinstance(v, (int, sympy.Integer)) else v)
                        for v in arg
                    ]
                )
                formatter.output.writeline(f"{lhs} = {name}")

        with V.set_ops_handler(formatter), patch.object(
            FlexibleLayout, "allow_indexing", True
        ):
            result = ir_fn(*args)
            return formatter.getvalue(result)

    def __getattr__(self, name) -> Callable[..., str]:
        def inner(*args, **kwargs):
            line = getattr(self.parent_handler, name)(*args, **kwargs)
            if name == "indirect_indexing":
                return line
            # replace line with a new variable name
            varname = f"tmp{next(self.var_counter)}"
            self.output.writeline(f"{varname} = {line}")
            return varname

        return inner

    def reduction(
        self, dtype, src_dtype, reduction_type, value
    ) -> Union[tuple[str, ...], str]:
        line = self.parent_handler.reduction(dtype, src_dtype, reduction_type, value)
        num_values = reduction_num_outputs(reduction_type)
        varnames = [f"tmp{next(self.var_counter)}" for _ in range(num_values)]
        self.output.writeline(f"{','.join(varnames)} = {line}")
        return tuple(varnames) if num_values > 1 else varnames[0]

    def getvalue(self, result):
        self.output.writeline(f"return {result}")
        return self.output.getvalue()


class WrapperHandler:
    def __init__(self, inner):
        self._inner = inner

    def __getattr__(self, item):
        return getattr(self._inner, item)


MockHandler._init_cls()

_ops = Virtualized("ops", MockHandler)  # TODO: improve type
_graph: Virtualized[GraphLowering] = Virtualized("graph", NullHandler)
_real_inputs: Virtualized[List[torch.Tensor]] = Virtualized("real_inputs", NullHandler)
_fake_mode: Virtualized[FakeTensorMode] = Virtualized("fake_mode", NullHandler)
_kernel: Virtualized[NullKernelHandler] = Virtualized(
    "kernel", NullKernelHandler
)  # TODO: improve type
_debug: Virtualized[DebugContext] = Virtualized("debug", NullHandler)
_interpreter: Virtualized[InterpreterShim] = Virtualized("interpreter", NullHandler)
_aot_compilation: Virtualized[bool] = Virtualized("aot_compilation", NullHandler)
_current_node: Virtualized[torch.fx.Node] = Virtualized("current_node", NullHandler)


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


class OpsWrapper:
    """This wraps any returned IR values into an `OpsValue` instance, so that we
    can overload the magic methods for writing mathematical expressions fluently.
    """

    def __getattr__(self, name):
        def inner(*args, **kwargs):
            new_args = [OpsWrapper._unwrap(a) for a in args]
            new_kwargs = {k: OpsWrapper._unwrap(v) for k, v in kwargs.items()}
            return OpsWrapper._wrap(getattr(_ops, name)(*new_args, **new_kwargs))

        return inner

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
    def indirect_indexing(index, size, check=True):
        # Returns a sympy value, not IR value
        index = OpsWrapper._unwrap(index)
        return _ops.indirect_indexing(index, size, check)


ops = OpsWrapper()

_MockHandler = MockHandler


class _V:
    MockHandler = MockHandler
    KernelFormatterHandler = KernelFormatterHandler
    WrapperHandler = WrapperHandler

    set_ops_handler: Callable[[Any], Any] = _ops._set_handler
    get_ops_handler: Callable[[], Any] = _ops._get_handler
    set_graph_handler: Callable[[GraphLowering], Any] = _graph._set_handler
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

    @property
    def ops(self) -> _MockHandler:
        """The operator handler specific to the current codegen task"""
        return _ops._get_handler()

    @property
    def graph(self) -> GraphLowering:
        """The graph currently being generated"""
        return _graph._get_handler()

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
        return _aot_compilation._get_handler()

    @property
    def current_node(self):
        return _current_node._get_handler()


V = _V()
