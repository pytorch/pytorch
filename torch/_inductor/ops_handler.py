# mypy: allow-untyped-defs
import itertools
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Literal,
    Optional,
    Tuple,
    TypeVar,
    Union,
)
from typing_extensions import Protocol
from unittest.mock import patch

import sympy

import torch
import torch.utils._pytree as pytree
from .utils import IndentedBuffer, reduction_num_outputs, sympy_index_symbol, sympy_str

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


def _arg_str(a) -> str:
    if isinstance(a, sympy.Expr):
        return sympy_str(a)
    return str(a)


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

    def trunc_to_int(self, x: T, dtype: torch.dtype) -> T:
        """
        Convert x to dtype with truncation semantics (similar to how the int
        constructor works in Python).  In Inductor codegen, this just decays
        to trunc and then to_dtype, but this composite operation helps
        roundtrips for Sympy evaluation.

        dtype is taken as an explicit parameter because the desired output
        dtype is typically the index dtype, which may vary between int32 and
        int64 depending on if we've shown that all the indexing operations can
        be done in int32.
        """
        ...

    def ceil_to_int(self, x: T, dtype: torch.dtype) -> T:
        """
        Convert x to dtype with ceiling semantics.  See also trunc_to_int.
        """
        ...

    def floor_to_int(self, x: T, dtype: torch.dtype) -> T:
        """
        Convert x to dtype with ceiling semantics.  See also trunc_to_int.
        """
        ...

    def round_to_int(self, x: T, dtype: torch.dtype) -> T:
        """
        Convert x to dtype with round-to-even semantics.  See also trunc_to_int.
        """
        ...

    def to_dtype_bitcast(self, x: T, dtype: torch.dtype, src_dtype: torch.dtype) -> T:
        """
        Reinterpret cast x to dtype (reinterpreting the bits in memory as another dtype.)
        src_dtype must be the original type of x.
        """
        ...

    def identity(self, x: T) -> T:
        """
        Returns x as is.  This is used to trigger CSE.
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
        self,
        dtypes: Tuple[torch.dtype, ...],
        combine_fn: Callable[[Tuple[T, ...], Tuple[T, ...]], Tuple[T, ...]],
        values: Tuple[T, ...],
    ) -> Tuple[T, ...]:
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

    def frexp(self, x0: T):
        ...

    def hypot(self, x0: T, x1: T) -> T:
        ...

    def log10(self, x0: T) -> T:
        ...

    def log2(self, x0: T) -> T:
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

    # NB: this returns a float, like the torch operation
    # This rounds half to even to break ties
    def round(self, x0: T) -> T:
        ...

    # NB: this returns a float, like the torch operation
    def floor(self, x0: T) -> T:
        ...

    def sign(self, x0: T) -> T:
        ...

    # NB: this returns a float, like the torch operation
    def trunc(self, x0: T) -> T:
        ...

    # NB: this returns a float, like the torch operation
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

    # NB: this returns a float, like the torch operation
    def pow(self, x0: T, x1: T) -> T:
        ...

    def and_(self, x0: T, x1: T) -> T:
        ...

    def or_(self, x0: T, x1: T) -> T:
        ...

    def xor(self, x0: T, x1: T) -> T:
        ...

    # These are metaprogrammed by MockHandler._init_cls
    def lshift(self, x0: T, x1: T) -> T:
        ...

    def rshift(self, x0: T, x1: T) -> T:
        ...

    def getitem(self, x0: T, x1: T) -> T:
        # TODO: this is probably just illegal lol
        ...

    def matmul(self, x0: T, x1: T) -> T:
        # TODO: this is probably just illegal lol
        ...

    def invert(self, x0: T) -> T:
        ...

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # These are "special" operators.  These only exist if the target
    # language actually supports the operator.  Keep this in sync with
    # pointwise_overrides_data.

    def airy_ai(self, x: T) -> T:
        ...

    def bessel_j0(self, x: T) -> T:
        ...

    def bessel_j1(self, x: T) -> T:
        ...

    def bessel_y0(self, x: T) -> T:
        ...

    def bessel_y1(self, x: T) -> T:
        ...

    def digamma(self, x: T) -> T:
        ...

    def erfcx(self, x: T) -> T:
        ...

    def fma(self, x: T, y: T, z: T) -> T:
        ...

    def igamma(self, x: T, y: T) -> T:
        ...

    def igammac(self, x: T, y: T) -> T:
        ...

    def gammainc(self, x: T, y: T) -> T:
        ...

    def gammaincc(self, x: T, y: T) -> T:
        ...

    def i0(self, x: T) -> T:
        ...

    def i0e(self, x: T) -> T:
        ...

    def i1(self, x: T) -> T:
        ...

    def i1e(self, x: T) -> T:
        ...

    def log_ndtr(self, x: T) -> T:
        ...

    def modified_bessel_i0(self, x: T) -> T:
        ...

    def modified_bessel_i1(self, x: T) -> T:
        ...

    def modified_bessel_k0(self, x: T) -> T:
        ...

    def modified_bessel_k1(self, x: T) -> T:
        ...

    def ndtr(self, x: T) -> T:
        ...

    def ndtri(self, x: T) -> T:
        ...

    def polygamma(self, x: T, y: T) -> T:
        ...

    def scaled_modified_bessel_k0(self, x: T) -> T:
        ...

    def scaled_modified_bessel_k1(self, x: T) -> T:
        ...

    def spherical_bessel_j0(self, x: T) -> T:
        ...

    def zeta(self, x: T, y: T) -> T:
        ...

    def chebyshev_polynomial_t(self, x: T, y: T) -> T:
        ...

    def chebyshev_polynomial_u(self, x: T, y: T) -> T:
        ...

    def chebyshev_polynomial_v(self, x: T, y: T) -> T:
        ...

    def chebyshev_polynomial_w(self, x: T, y: T) -> T:
        ...

    def legendre_polynomial_p(self, x: T, y: T) -> T:
        ...

    def shifted_chebyshev_polynomial_t(self, x: T, y: T) -> T:
        ...

    def shifted_chebyshev_polynomial_u(self, x: T, y: T) -> T:
        ...

    def shifted_chebyshev_polynomial_v(self, x: T, y: T) -> T:
        ...

    def shifted_chebyshev_polynomial_w(self, x: T, y: T) -> T:
        ...

    def hermite_polynomial_h(self, x: T, y: T) -> T:
        ...

    def hermite_polynomial_he(self, x: T, y: T) -> T:
        ...

    def laguerre_polynomial_l(self, x: T, y: T) -> T:
        ...

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # These operators are a bit special, because they are conventionally
    # natively supported in both Python and C, but the semantics differ so
    # care must be taken

    def truncdiv(self, x0: T, x1: T) -> T:
        """C-style trunc division between integers only.  Computes the true
        division of two numbers and rounds the result to zero.
        """
        ...

    def floordiv(self, x0: T, x1: T) -> T:
        """Python-style floor division between integers only.  Computes the
        true division of two numbers and floors the result.  If you want
        floor division for floats, do regular truediv and floor the result.
        """
        ...

    def truediv(self, x0: T, x1: T) -> T:
        """True division between floats.  Integer inputs are NOT valid.  To
        do Python-style (int, int) -> float division, use int_truediv"""
        ...

    def int_truediv(self, x0: T, x1: T) -> T:
        """True division between integers.  This is NOT the same as promoting
        to float and doing integer division, there is a bespoke algorithm for
        doing the division in higher precision than the above.
        """
        ...

    def div(self, x0: T, x1: T) -> T:
        """TODO: to be removed.  This renders as / no matter what the backend is
        which is incoherent."""
        ...

    def mod(self, x0: T, x1: T) -> T:
        """C-style modulus, take sign from LHS (x0)."""
        ...

    def remainder(self, x0: T, x1: T) -> T:
        """Python-style modulus, take sign from RHS (x1)."""
        ...

    def round_decimal(self, x0: T, x1: T) -> T:
        """Python-style round with decimal argument"""
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


class NoopHandler:
    def __getattr__(self, name):
        if name == "name":
            return "NoopHandler"

        def inner(*args, **kwargs):
            return None

        return inner

    @staticmethod
    def masked(mask, body, other) -> None:
        return None

    @staticmethod
    def frexp(x) -> Tuple[None, None]:
        return (None, None)

    @staticmethod
    def scan(dtypes, combine_fn, values) -> Tuple[None, ...]:
        return tuple(None for i in range(len(values)))

    @staticmethod
    def indirect_indexing(index_var, size, check=True) -> sympy.Symbol:
        return sympy.Integer(0)


# Use mypy to check protocol implemented correctly
def _typecheck_NoopHandler(h: NoopHandler) -> OpsHandler[None]:
    return h


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
    def frexp(x):
        return (f"ops.frexp({x})[0]", f"ops.frexp({x})[1]")

    @staticmethod
    def scan(dtypes, combine_fn, values):
        return tuple(
            f"ops.scan({dtypes}, {combine_fn}, {values})[{i}]"
            for i in range(len(values))
        )

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

        for name, format_string in {
            "add": "{} + {}",
            "sub": "{} - {}",
            "mul": "{} * {}",
            "floordiv": "{} // {}",
            "truediv": "{} / {}",
            "mod": "{} % {}",  # careful, depending on target semantics varies
            "pow": "{} ** {}",
            "lshift": "{} << {}",
            "rshift": "{} >> {}",
            "and_": "{} & {}",
            "or_": "{} | {}",
            "xor": "{} ^ {}",
            "eq": "{} == {}",
            "ne": "{} != {}",
            "lt": "{} < {}",
            "gt": "{} > {}",
            "le": "{} <= {}",
            "ge": "{} >= {}",
            "neg": "-{}",
        }.items():
            setattr(cls, name, make_handler(format_string))


MockHandler._init_cls()


# Use mypy to check protocol implemented correctly
def _typecheck_MockHandler(h: MockHandler) -> OpsHandler[str]:
    return h


class KernelFormatterHandler:
    def __init__(self, parent_handler):
        self.parent_handler = parent_handler
        self.output = IndentedBuffer(1)
        self.var_counter = itertools.count()

    @staticmethod
    def ir_to_string(ir_fn, index, rindex=None) -> str:
        from .ir import FlexibleLayout
        from .virtualized import V

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

            def write(line):
                # replace line with a new variable name
                varname = f"tmp{next(self.var_counter)}"
                self.output.writeline(f"{varname} = {line}")
                return varname

            return pytree.tree_map(write, line)

        return inner

    def reduction(
        self,
        dtype: torch.dtype,
        src_dtype: torch.dtype,
        reduction_type: ReductionType,
        value: Union[str, Tuple[str, ...]],
    ) -> Union[str, Tuple[str, ...]]:
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


class OpCounterCSE:
    """Shim to count how many ops are used"""

    def __init__(self, inner):
        super().__init__()
        self.parent_handler = inner
        self.op_count = 0
        self.var_names = {}

    def __getattr__(self, name):
        def inner(*args, **kwargs):
            val = getattr(self.parent_handler, name)(*args, **kwargs)
            if name == "indirect_indexing":
                return val

            def count(val):
                if val not in self.var_names:
                    varname = f"tmp{self.op_count}"
                    self.op_count += 1
                    self.var_names[val] = varname
                    return varname
                else:
                    return self.var_names[val]

            return pytree.tree_map(count, val)

        return inner


def _typecheck_OpCounterCSE(h: OpCounterCSE) -> OpsHandler[str]:
    return h


class ExtractConstantsHandler(NoopHandler):
    def __init__(self, device):
        self.device = device

    def constant(self, value: Any, dtype: torch.dtype) -> "torch._inductor.ir.Constant":
        from torch._inductor import ir

        return ir.Constant(value=value, dtype=dtype, device=self.device)


def _typecheck_ExtractConstantsHandler(h: ExtractConstantsHandler) -> OpsHandler[Any]:
    return h


class SimpleCSEHandler(WrapperHandler[T]):
    """Wraps the underlying handler with a CSE pass

    NOTE: Compared to codegen level CSE this is simplified as it
    doesn't support stores which require load cache invalidation.
    """

    def __init__(self, inner: OpsHandler[T]):
        super().__init__(inner)
        self.cse_cache: Dict[str, Union[T, Tuple[T, ...]]] = {}
        self.mock = MockHandler()

    def indirect_indexing(self, *args, **kwargs) -> sympy.Expr:
        return super().indirect_indexing(*args, **kwargs)  # type: ignore[misc]

    def store(self, *args, **kwargs) -> T:
        raise NotImplementedError("store not implemented")

    def store_reduction(self, *args, **kwargs) -> T:
        raise NotImplementedError("store not implemented")

    def __getattr__(self, name) -> Callable[..., Any]:
        def inner(*args, **kwargs):
            key = getattr(self.mock, name)(*args, **kwargs)
            val = self.cse_cache.get(key)
            if val is not None:
                return val

            val = getattr(self._inner, name)(*args, **kwargs)
            self.cse_cache[key] = val
            return val

        return inner


def _typecheck_SimpleCSEHandler(h: SimpleCSEHandler[Any]) -> OpsHandler[Any]:
    return h
