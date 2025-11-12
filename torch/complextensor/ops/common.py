from collections.abc import Callable
from contextvars import ContextVar, Token
from typing import Any, overload, TypeAlias
from typing_extensions import Self, TypeIs

import torch
from torch import Tensor
from torch._decomp import get_decompositions
from torch._ops import OpOverload, OpOverloadPacket
from torch._refs import is_complex as _is_complex
from torch.types import Number
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten, tree_map, tree_unflatten

from ..core import ComplexTensor


OpType: TypeAlias = OpOverloadPacket | OpOverload

TableType: TypeAlias = dict[OpType, Callable]

DebugSetType: TypeAlias = set[OpType] | None

# Mapping from ops to implementations
COMPLEX_OPS_TABLE: TableType = {}

COMPLEX_TO_REAL = {
    torch.complex128: torch.float64,
    torch.complex64: torch.float32,
    torch.complex32: torch.float16,
}

REAL_TO_COMPLEX = {v: k for k, v in COMPLEX_TO_REAL.items()}

# Used to promote dtypes in `promote_real_cpu_tensors`
PROMOTE_TYPES_CPU = {
    torch.float16: torch.float32,
    torch.bfloat16: torch.float32,
}


def is_complex_tensor(obj: Any, /) -> TypeIs[ComplexTensor]:
    r"""Returns True if the input is a ComplexTensor, else False

    Args:
        a: any input

    Examples:

        >>> # xdoctest: +SKIP
        >>> from torch.complex import ComplexTensor
        >>> data = torch.zeros((3, 2), dtype=torch.complex64)
        >>> ct = ComplexTensor.from_interleaved(data)
        >>> is_complex_tensor(ct)
        True
    """
    return isinstance(obj, ComplexTensor)


def promote_real_cpu_tensors(
    *tensors: Tensor,
) -> tuple[torch.dtype, tuple[Tensor, ...]]:
    """
    Promotes all tensors to a common dtype.
    Additionally promotes CPU tensors to at least `float32`.
    """
    tensor = next(t for t in tensors if isinstance(t, Tensor))
    out_dt = tensor.dtype
    for t in tensors:
        if isinstance(t, Tensor):
            out_dt = torch.promote_types(out_dt, t.dtype)

    prom_dt = PROMOTE_TYPES_CPU.get(out_dt)
    if prom_dt is None or any(
        t.device.type != "cpu" for t in tensors if isinstance(t, Tensor)
    ):
        return out_dt, tuple(
            t.to(out_dt) if isinstance(t, Tensor) else torch.asarray(t, dtype=out_dt)
            for t in tensors
        )

    return out_dt, tuple(
        t.to(prom_dt) if isinstance(t, Tensor) else torch.asarray(t, dtype=prom_dt)
        for t in tensors
    )


def register_complex(
    op: OpType,
    func_impl: Callable | None = None,
):
    """Decorator to register an implementation for some ops in some dispatch tables"""

    def inner(func):
        if COMPLEX_OPS_TABLE.get(op, func) is not func:
            raise RuntimeError(f"Attempted to register multiple functions for {op}")
        COMPLEX_OPS_TABLE[op] = func
        return func

    if func_impl is None:
        return inner

    return inner(func_impl)


FORCE_TEST_LIST: list[OpType] = []


def register_force_test(op: OpType, *args, **kwargs):
    """Will attempt to test these ops even if they err on "normal" inputs"""
    FORCE_TEST_LIST.append(op)
    return register_complex(op, *args, **kwargs)


DECOMPOSITIONS = get_decompositions(list(torch.ops.aten))  # type: ignore[no-matching-overload]

# Set of ops found to be "problematic" in a debugging context.
DEBUG_SET: ContextVar[DebugSetType] = ContextVar("DEBUG_SET", default=None)


def lookup_complex(func: OpOverload, *args, **kwargs) -> Callable | None:
    """
    Lookup an impl from the table.

    Try the particular overload first, then the overload packet.

    If nothing is found, try the decompositions with both.
    """
    return COMPLEX_OPS_TABLE.get(
        func,
        COMPLEX_OPS_TABLE.get(
            func.overloadpacket,
            DECOMPOSITIONS.get(func, DECOMPOSITIONS.get(func.overloadpacket)),
        ),
    )


def is_complex(x: Any, /) -> bool:
    """Utility to detect if a given object is (known) to be complex."""
    return (isinstance(x, Tensor) and _is_complex(x)) or isinstance(x, complex)


@overload
def split_complex_arg(
    arg: Tensor | ComplexTensor,
) -> tuple[Tensor, Tensor]: ...


@overload
def split_complex_arg(
    arg: complex | Number,
) -> tuple[Number, Number]: ...


def split_complex_arg(
    arg: Tensor | ComplexTensor | complex | Number,
) -> tuple[Tensor, Tensor] | tuple[Number, Number]:
    """
    Split a complex argument into a real/imaginary component.

    If real, use zero for the imaginary part.
    """
    if isinstance(arg, ComplexTensor):
        return split_complex_tensor(arg)
    if isinstance(arg, Tensor):
        if is_complex(arg):
            return arg.real, arg.imag
        return arg, torch.zeros_like(arg)
    # TODO (hameerabbasi): Should there be a `torch.SymComplex`?
    if isinstance(arg, complex):
        return arg.real, arg.imag
    if isinstance(arg, float | torch.SymFloat):
        return arg, 0.0
    if isinstance(arg, int | torch.SymInt):
        return arg, 0
    if isinstance(arg, bool | torch.SymBool):
        return arg, False
    raise TypeError(f"Expected tensor or number got, {type(arg)}")


def split_complex_tensor(complex_tensor: ComplexTensor) -> tuple[Tensor, Tensor]:
    """Split a ComplexTensor into its real and imaginary parts."""
    return complex_tensor.re, complex_tensor.im


def complex_to_real_dtype(dtype: torch.dtype) -> torch.dtype:
    """Convert a complex dtype to the dtype of its real part. Return other dtypes as-is."""
    return COMPLEX_TO_REAL.get(dtype, dtype)


def _get_op_name(op: OpType) -> str:
    """Get the op name from the op."""
    if isinstance(op, OpOverload):
        op = op.overloadpacket
    return str(op).split(".", 1)[1]


def _get_func_name(op: OpType) -> str:
    """Get the name of the implementation function from the op."""
    return f"{_get_op_name(op)}_impl"


def register_error(op: OpType, exc_type: type[Exception] = NotImplementedError):
    msg = f"`aten.{_get_op_name(op)}` not implemented for `{ComplexTensor.__name__}`."

    def ordered_impl(*args, **kwargs):
        raise exc_type(msg)

    func_name = _get_func_name(op)
    ordered_impl.__name__ = func_name
    ordered_impl.__qualname__ = func_name

    return register_force_test(op, ordered_impl)


def register_binary_nonlinear(op: OpType) -> Callable:
    """Register a "multiplication-style" op, e.g. aten.mul, aten.mm, ..."""

    def impl(lhs: ComplexTensor, rhs: ComplexTensor, *args, **kwargs) -> ComplexTensor:
        a_r, a_i = split_complex_arg(lhs)
        b_r, b_i = split_complex_arg(rhs)
        out_dt, (a_r, a_i, b_r, b_i) = promote_real_cpu_tensors(a_r, a_i, b_r, b_i)
        real = op(a_r, b_r, *args, **kwargs) - op(a_i, b_i, *args, **kwargs)
        imag = op(a_r, b_i, *args, **kwargs) + op(a_i, b_r, *args, **kwargs)
        return ComplexTensor(real.to(out_dt), imag.to(out_dt))

    func_name = _get_func_name(op)
    impl.__name__ = func_name
    impl.__qualname__ = func_name

    return register_complex(op, impl)


def register_simple(op: OpType):
    """Register an op which can be applied independently to the real and complex parts to get the result."""

    def impl(
        self: ComplexTensor, *args, dtype: torch.dtype | None = None, **kwargs
    ) -> ComplexTensor:
        x, y = split_complex_tensor(self)
        if dtype is not None and dtype not in COMPLEX_TO_REAL:
            raise RuntimeError(
                "Non-complex `dtype` specified, please write custom impl."
            )

        if dtype in COMPLEX_TO_REAL:
            assert dtype is not None
            kwargs["dtype"] = COMPLEX_TO_REAL[dtype]

        u = op(x, *args, **kwargs)
        v = op(y, *args, **kwargs)

        u_flat, u_spec = tree_flatten(u)
        v_flat, v_spec = tree_flatten(v)
        assert u_spec == v_spec
        out_flat = [
            ComplexTensor(ui, vi) for ui, vi in zip(u_flat, v_flat, strict=False)
        ]
        return tree_unflatten(out_flat, u_spec)

    func_name = _get_func_name(op)
    impl.__name__ = func_name
    impl.__qualname__ = func_name

    return register_complex(op, impl)


def _as_complex_tensor(arg: Tensor | Any) -> Tensor | ComplexTensor | Any:
    """Convert a Tensor with complex dtypes to a ComplexTensor. Pass along other args as-is."""
    if (
        not isinstance(arg, ComplexTensor)
        and isinstance(arg, Tensor)
        and arg.dtype in COMPLEX_TO_REAL
    ):
        return ComplexTensor.from_interleaved(arg)
    return arg


def _as_interleaved(arg: ComplexTensor | Any) -> Tensor | Any:
    """Convert a ComplexTensor to a Tensor with a complex dtype. Pass other arguments as-is."""
    if isinstance(arg, ComplexTensor):
        return arg.as_interleaved()
    return arg


class ComplexTensorMode(TorchDispatchMode):
    _compile: bool
    _debug: bool
    _debug_token: Token[DebugSetType] | None

    """ A TorchDispatchMode to replace any Tensor that has a complex dtype with a ComplexTensor for the computation. """

    def __init__(
        self, _dispatch_key=None, *, _compile: bool = False, _debug: bool = False
    ):
        """Initialize a ComplexTensorMode.

        Args:
            _dispatch_key: passed on to TorchDispatchMode
            _compile: Compile the op before the computation
            _debug: Find inconsistencies with the base Tensor
                    while performing the computation.
        """
        super().__init__(_dispatch_key)
        self._compile = _compile
        self._debug = _debug
        self._debug_token = None

    def __torch_dispatch__(
        self,
        func: OpOverload,
        types: tuple[type],
        args: tuple = (),
        kwargs: dict[str, Any] | None = None,
    ):
        if kwargs is None:
            kwargs = {}

        if self._compile:
            func = torch.compile(func)  # type: ignore[bad-assignment]

        args = tree_map(_as_complex_tensor, args)
        kwargs = tree_map(_as_complex_tensor, kwargs)

        return tree_map(_as_interleaved, func(*args, **kwargs))

    def __enter__(self) -> Self:
        # Note (debugging ops): This block sets the debugging mode
        if self._debug:
            self._debug_token = DEBUG_SET.set(set())
        return super().__enter__()

    def __exit__(self, type_, val, tb):
        # Note (debugging ops): This block resets the debugging mode
        if self._debug_token is not None:
            debug_set = DEBUG_SET.get()
            assert debug_set is not None
            print("\n".join([str(op) for op in debug_set]))
            DEBUG_SET.reset(self._debug_token)
            self._debug_token = None
        return super().__exit__(type_, val, tb)
