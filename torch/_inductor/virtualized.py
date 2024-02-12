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

import itertools
from contextlib import AbstractContextManager, contextmanager
from itertools import chain
from threading import local
from typing import (
    Any,
    Callable,
    Generic,
    List,
    Literal,
    Optional,
    Tuple,
    Type,
    TYPE_CHECKING,
    TypeVar,
    Union,
)
from unittest.mock import patch

import sympy
from typing_extensions import Protocol

from torch._inductor.utils import IndentedBuffer

from torch.fx.graph import inplace_methods, magic_methods

from .utils import reduction_num_outputs, sympy_index_symbol, sympy_str

if TYPE_CHECKING:
    import torch
    from torch._inductor.debug import DebugContext
    from torch._inductor.graph import GraphLowering
    from torch._inductor.ir import InterpreterShim
    from torch._subclasses import FakeTensorMode

threadlocal = local()

T = TypeVar("T")
StoreMode = Optional[Literal["atomic_add"]]
ReductionType = Literal[
    "argmax",
    "argmin",
    "welford_reduce",
    "welford_combine",
    "any",
    "max",
    "min",
    "prod",
    "sum",
    "xor_sum",
]


class NullHandler:
    """
    Sentinel indicating that a global variable is unset ala None.  Typically,
    attempting to access the global variable before it's set is an error, but with
    NullHandler it won't fail until you try to access an attribute on it.
    """

    pass


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

    @staticmethod
    def masked(mask, body, other) -> str:
        return f"ops.masked({mask}, {body()}, {other})"

    @staticmethod
    def indirect_indexing(index_var, size, check=True) -> sympy.Symbol:
        return sympy_index_symbol(str(index_var))

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


# Use mypy to check protocol implemented correctly
def _typecheck_MockHandler(h: MockHandler) -> OpsHandler[str]:
    return h


# NB: This is not done as a parent class, because our ops handlers
# implementations make heavy use of __getattr__ magic, and pre-existing
# stubs for methods would interfere with this mechanism.
#
# TODO: A superclass that does desugaring for operations like
# reciprocal/square might be useful.
class OpsHandler(Protocol[T]):
    """
    Protocol describing the set of valid operations on ``torch._inductor.virtualized.ops``,
    as well as the contract for op handlers.  The type T signifies the domain
    of the abstract analysis AKA what all of the functions return / take as arguments
    anywhere compute occurs.

    While these operators are typically dtype polymorphic (e.g., you can use mul
    on both integers and floats), they do NOT do promotion and usually return the
    same dtype as the input.  You are expected to have handled type promotion
    during ATen decompositions.  Most operators correspond exactly to pointwise
    operations as defined by torch, so when in doubt about semantics, check the
    corresponding torch documentation.  These are all scalar operations (so they
    are defined to operate on a single element at a time.)

    For convenience, many operators take a src_dtype which indicates what the dtype
    of the input argument is.  Although in principle this can be derived by an
    analysis, providing this for ops where it is useful helps avoid having to repeatedly
    recompute dtype in code generation.

    Note that this often describes a class of static methods, for stateless
    ops handlers.

    Handlers are often defined using ``__getattr__`` metaprogramming, which means
    that you cannot declare that a type implements a protocol by inheriting from
    it (as the type stubs count as attribute declarations and impede the getattr
    magic method from being called).  Instead, define a function that casts an
    argument of your type to the protocol, which is sufficient to induce mypy to
    test that the protocol is implemented correctly.  Search for ``_typecheck_``
    in this file to see some examples.  If you see an obscure error where a
    class doesn't implement a Protocol, but mypy doesn't say why, check to see
    that ``__getattr__`` is typed correctly (typically, it is not possible to
    type ``__getattr__`` without typing it as ``Callable[..., Any]``)
    """

    def constant(self, value: Union[bool, float, int], dtype: torch.dtype) -> T:
        """Produces a scalar constant of type dtype."""
        ...

    def load_seed(self, name: str, offset: T):
        """Computes inductor_prims.lookup_seed."""
        ...

    def rand(self, seed: T, offset: T) -> T:
        """Computes inductor_prims.random with mode="rand".  offset has dtype int32."""
        ...

    def randn(self, seed: T, offset: T) -> T:
        """Computes inductor_prims.random with mode="randn".  offset has dtype int32."""
        ...

    def randint64(self, seed: T, offset: T, low: T, high: T) -> T:
        """Computes inductor_prims.randint.  offset has dtype int32."""
        ...

    def masked(self, mask: T, body: Callable[[], T], other: T) -> T:
        """
        Computes body, but only perform loads/stores if the boolean mask
        evaluates to true.  For example, you would use this if you needed to
        perform an indirect load that may not be valid on some elements;
        without masking, invalid accesses can cause IMAs.  When mask is true,
        the result is the result of body; otherwise it is other.

        Contrast this with ops.where, which can multiplex between two values
        that have been unconditionally computed.
        """
        ...

    def where(self, condition: T, input: T, other: T) -> T:
        """
        Computes torch.where: when condition is true, return input; otherwise return other.
        """
        ...

    def index_expr(self, expr: sympy.Expr, dtype: torch.dtype) -> T:
        """
        Converts a sympy expression into a scalar of type dtype.  expr is typically
        an indexing expression, thus the name; however, it can also be used in
        non-indexing situations.
        """
        ...

    def to_dtype(
        self, x: T, dtype: torch.dtype, src_dtype: Optional[torch.dtype] = None
    ) -> T:
        """
        Convert x to dtype.  src_dtype can be optionally set to specify what the original
        dtype of x was, which can improve code generation (used by torch to(dtype=dtype)).
        """
        ...

    def to_dtype_bitcast(self, x: T, dtype: torch.dtype, src_dtype: torch.dtype) -> T:
        """
        Reinterpret cast x to dtype (reinterpreting the bits in memory as another dtype.)
        src_dtype must be the original type of x.
        """
        ...

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # These operations are only available in a "kernel" context.  Check
    # torch._inductor.codegen.common.CSEProxy for their typical implementation
    # in op handler (routing to their respective implementations in the kernel
    # handler)
    #
    # Importantly, inside a kernel, indexing and mask variables are available
    # in scope, which are typically used by sympy.Expr indexing.

    def indirect_indexing(
        self, x: T, size: sympy.Expr, check: bool = True
    ) -> sympy.Expr:
        """
        Convert an integral x into a sympy.Expr that can be subsequently used in
        indexing computation.  'size' represents an upper bound on the what valid
        indexes can be; when 'check' is True, we check that the x is in bounds.

        NB: This is typically mandatory to implement for any analysis, because you
        MUST return a valid sympy.Expr of some sort (even if it's a meaningless symbol).
        """
        ...

    def load(self, name: str, index: sympy.Expr) -> T:
        """
        Load from the memory location 'name', offset by some indexing expression 'index'.
        """
        ...

    def store(
        self,
        name: str,
        index: sympy.Expr,
        value: T,
        mode: StoreMode = None,
    ) -> None:
        """
        Store 'value' to the memory location 'name' offset by 'expr'.  If
        specified, 'mode' can require the store to be an atomic addition.
        """
        ...

    # TODO: Better explain how the "collective" semantics of these ops;
    # remember that the input value is a scalar, you can't reduce on it in the
    # traditional sense!
    def reduction(
        self,
        dtype: torch.dtype,
        src_dtype: torch.dtype,
        reduction_type: ReductionType,
        value: T,
    ) -> Union[T, Tuple[T, ...]]:
        """
        Perform a 'reduction_type' reduction on 'value' of dtype 'src_dtype',
        using 'dtype' as the accumulation dtype for the reduction.  The result
        is an intermediate computation which should be stored to the final
        location using 'ops.store_reduction'.

        Valid reduction types are .  For Welford reduction types, this
        function returns multiple outputs; consult reduction_num_outputs to
        determine the amount in metaprogramming applications.
        """
        ...

    # TODO: in practice, this seems to actually return None, but not returning
    # a T makes common __getattr__ idioms not type correctly.  Figure out if
    # this should be returning something.
    def store_reduction(self, name: str, index: sympy.Expr, value: T) -> T:
        """
        Store the fully accumulated result of 'reduction' to the memory
        location 'name' offset by 'expr'.
        """
        ...

    def scan(
        self, dtype: torch.dtype, combine_fn: Callable[[T, T], T], value: T, init: int
    ) -> T:
        """
        Perform an associative scan on 'value'.
        """
        # TODO: Improve the description with some pseudocode
        ...

    def bucketize(
        self,
        values: T,
        offsets_name: str,
        offsets_size: sympy.Expr,
        indexing_dtype: torch.dtype,
        right: bool,
    ) -> T:
        # See [Note: Inductor bucketize op]
        ...

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # The following ops have semantics that correspond exactly to the torch
    # operation with the same corresponding name.

    def abs(self, x0: T) -> T:
        ...

    def exp(self, x0: T) -> T:
        ...

    def exp2(self, x0: T) -> T:
        ...

    def expm1(self, x0: T) -> T:
        ...

    def sqrt(self, x0: T) -> T:
        ...

    def relu(self, x0: T) -> T:
        ...

    def minimum(self, x0: T, x1: T) -> T:
        ...

    def maximum(self, x0: T, x1: T) -> T:
        ...

    def cos(self, x0: T) -> T:
        ...

    def sin(self, x0: T) -> T:
        ...

    def lgamma(self, x0: T) -> T:
        ...

    def erf(self, x0: T) -> T:
        ...

    def cosh(self, x0: T) -> T:
        ...

    def sinh(self, x0: T) -> T:
        ...

    def acos(self, x0: T) -> T:
        ...

    def acosh(self, x0: T) -> T:
        ...

    def asin(self, x0: T) -> T:
        ...

    def asinh(self, x0: T) -> T:
        ...

    def atan2(self, x0: T, x1: T) -> T:
        ...

    def atan(self, x0: T) -> T:
        ...

    def atanh(self, x0: T) -> T:
        ...

    def copysign(self, x0: T, x1: T) -> T:
        ...

    def erfc(self, x0: T) -> T:
        ...

    def erfinv(self, x0: T) -> T:
        ...

    def hypot(self, x0: T, x1: T) -> T:
        ...

    def log10(self, x0: T) -> T:
        ...

    def nextafter(self, x0: T, x1: T) -> T:
        ...

    def logical_and(self, x0: T, x1: T) -> T:
        ...

    def logical_not(self, x0: T) -> T:
        ...

    def logical_or(self, x0: T, x1: T) -> T:
        ...

    def logical_xor(self, x0: T, x1: T) -> T:
        ...

    def bitwise_and(self, x0: T, x1: T) -> T:
        ...

    def bitwise_not(self, x0: T) -> T:
        ...

    def bitwise_or(self, x0: T, x1: T) -> T:
        ...

    def bitwise_xor(self, x0: T, x1: T) -> T:
        ...

    def bitwise_left_shift(self, x0: T, x1: T) -> T:
        ...

    def bitwise_right_shift(self, x0: T, x1: T) -> T:
        ...

    def rsqrt(self, x0: T) -> T:
        ...

    def log1p(self, x0: T) -> T:
        ...

    def tan(self, x0: T) -> T:
        ...

    def tanh(self, x0: T) -> T:
        ...

    def sigmoid(self, x0: T) -> T:
        ...

    def signbit(self, x0: T) -> T:
        ...

    def fmod(self, x0: T, x1: T) -> T:
        ...

    def log(self, x0: T) -> T:
        ...

    def isinf(self, x0: T) -> T:
        ...

    def isnan(self, x0: T) -> T:
        ...

    def round(self, x0: T) -> T:
        ...

    def floor(self, x0: T) -> T:
        ...

    def sign(self, x0: T) -> T:
        ...

    def to_int(self, x0: T) -> T:
        ...

    def trunc(self, x0: T) -> T:
        ...

    def truncdiv(self, x0: T, x1: T) -> T:
        ...

    def ceil(self, x0: T) -> T:
        ...

    def neg(self, x0: T) -> T:
        ...

    def reciprocal(self, x0: T) -> T:
        ...

    def eq(self, x0: T, x1: T) -> T:
        ...

    def ne(self, x0: T, x1: T) -> T:
        ...

    def lt(self, x0: T, x1: T) -> T:
        ...

    def gt(self, x0: T, x1: T) -> T:
        ...

    def le(self, x0: T, x1: T) -> T:
        ...

    def ge(self, x0: T, x1: T) -> T:
        ...

    def add(self, x0: T, x1: T) -> T:
        ...

    def sub(self, x0: T, x1: T) -> T:
        ...

    def mul(self, x0: T, x1: T) -> T:
        ...

    def floordiv(self, x0: T, x1: T) -> T:
        ...

    def truediv(self, x0: T, x1: T) -> T:
        ...

    def div(self, x0: T, x1: T) -> T:
        ...

    def mod(self, x0: T, x1: T) -> T:
        ...

    def pow(self, x0: T, x1: T) -> T:
        ...

    def and_(self, x0: T, x1: T) -> T:
        ...

    def or_(self, x0: T, x1: T) -> T:
        ...

    def xor(self, x0: T, x1: T) -> T:
        ...

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # In CUDA, optimized implementations of other mathematical operations are
    # offered separately via libdevice for double precision computation (in
    # Triton, these go to tl.math rather than tl).  We lower to these
    # operators when doing FP64 on CUDA.  Note that some operators
    # unconditional go to tl.math.
    #
    # TODO(ezyang): Is this really the best way to do this?  What if we have
    # abs internally route to tl.math automatically when given a double
    # precision input?  One reason is that when doing codegen, we often don't
    # know what the dtype of the inputs are!  (In principle we do know, but
    # for many analyses it's not conveniently available.)

    def libdevice_abs(self, x0: T) -> T:
        ...

    def libdevice_exp(self, x0: T) -> T:
        ...

    def libdevice_sqrt(self, x0: T) -> T:
        ...

    def libdevice_cos(self, x0: T) -> T:
        ...

    def libdevice_sin(self, x0: T) -> T:
        ...

    def libdevice_sigmoid(self, x0: T) -> T:
        ...

    def libdevice_log(self, x0: T) -> T:
        ...


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

    def __getattr__(self, name) -> Callable[..., Any]:
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
        self,
        dtype: torch.dtype,
        src_dtype: torch.dtype,
        reduction_type: ReductionType,
        value: Union[str, tuple[str, ...]],
    ) -> Union[str, tuple[str, ...]]:
        line = self.parent_handler.reduction(dtype, src_dtype, reduction_type, value)
        num_values = reduction_num_outputs(reduction_type)
        varnames = [f"tmp{next(self.var_counter)}" for _ in range(num_values)]
        self.output.writeline(f"{','.join(varnames)} = {line}")
        return tuple(varnames) if num_values > 1 else varnames[0]

    def getvalue(self, result):
        self.output.writeline(f"return {result}")
        return self.output.getvalue()


# Use mypy to check protocol implemented correctly
def _typecheck_KernelFormatterHandler(h: KernelFormatterHandler) -> OpsHandler[str]:
    return h


class WrapperHandler(Generic[T]):
    def __init__(self, inner: OpsHandler[T]):
        self._inner = inner

    def __getattr__(self, item):
        return getattr(self._inner, item)


# Use mypy to check protocol implemented correctly
def _typecheck_WrapperHandler(h: WrapperHandler[T]) -> OpsHandler[T]:
    return h


MockHandler._init_cls()

_ops: Virtualized[OpsHandler[Any]] = Virtualized("ops", MockHandler)
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
    def ops(self) -> OpsHandler[Any]:
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
