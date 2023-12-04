"""
This file does three things:
- Contains the definition of SymNode
- Installs all the magic methods into SymBool, SymFloat, SymFloat at import time
- Does not depend on sympy at import time

As this file is imported from within torch/__init__.py we do not want it to depend on SymPy
to avoid having to load SymPy at import time, as doing so is *very* slow.
"""

import builtins
import itertools
import logging
import math
import operator
import sys
from functools import lru_cache
from typing import Optional, Type, TYPE_CHECKING, Union

# NB: The sym_* functions are used via getattr() and must be imported here.
from torch import (  # noqa: F401
    sym_float,
    sym_ite,
    sym_max,
    sym_min,
    sym_not,
    sym_sqrt,
    SymBool,
    SymFloat,
    SymInt,
)

from torch.fx.experimental._sym_dispatch_mode import (
    handle_sym_dispatch,
    sym_function_mode,
)

if TYPE_CHECKING:
    from torch.fx.experimental.symbolic_shapes import ShapeEnv

log = logging.getLogger(__name__)


__all__ = ["SymNode", "method_to_operator", "magic_methods", "sym_sqrt"]


SymTypes = (SymInt, SymFloat, SymBool)


def _to_symtype(t):
    if t is bool:
        return SymBool
    if t is int:
        return SymInt
    if t is float:
        return SymFloat
    return t


# TODO: An incomplete list
# 1. Set variables to be equal when we do equality
# 2. Specialize on 0/1 when we do subtraction
class SymNode:
    """
    This is a type erased SymInt/SymFloat which we use to do actual operations.
    End users don't touch this.  Magic methods are NOT defined on this object.
    """

    def __init__(
        self,
        expr,
        shape_env,
        pytype,
        hint: Optional[Union[int, float, bool]],
        constant=None,
        fx_node=None,
    ):
        self._expr = expr
        self.shape_env = shape_env
        self.pytype = pytype
        # What's the difference between hint and constant?
        #
        # - A constant is known to be invariant across invocations of the model;
        #   it will always be this value.  We only really know this when we
        #   encounter an honest-to-goodness literal (when wrapping it into
        #   a SymNode, we set constant.)  Most of the time, constant is None
        #
        # - A hint is a *particular* value from the particular run we are
        #   tracing, but it may vary the next time around.  It's useful to
        #   keep this around, as if we need a concrete value from a SymNode,
        #   we will return the hint and guard on the expression that produced
        #   it giving the same hint next time around.  The hint is not
        #   guaranteed to be set either: if you have an unbacked SymNode,
        #   there won't be any hint; it was the result of some tensor-dependent
        #   computation, but we don't know what it actually is because we
        #   haven't actually run the tensor computation.
        #
        # If _hint is None, we will query maybe_evaluate_static(compute_hint=True)
        # in hopes that we've learned enough about the unbacked symints to
        # discharge the hint; otherwise, you're likely to just error out.
        #
        # (A previous version of this system had some optimizations to only
        # recompute when it was possible we had learned enough about the
        # unbacked symint that a hint was now possible, but as we added more
        # potential refinements to unbacked symints this got harder to keep
        # in sync, so we've deleted it for now.)
        if hint is not None:
            assert type(hint) is pytype or type(hint) is _to_symtype(pytype), (
                "Cannot create SymNode of type "
                f"{pytype} with incompatible hint of type {type(hint)}"
            )
        self._hint = hint
        self.constant: Optional[Union[int, float, bool]] = constant

        # Record the FX node of the current node if we are doing translation
        # validation. They will be used for building the input assertions for
        # the translation validation problem.
        self.fx_node = (
            fx_node if self.shape_env._translation_validation_enabled else None
        )

    def with_shape_env(self, shape_env: "ShapeEnv") -> "SymNode":
        return SymNode(
            self._expr, shape_env, self.pytype, self._hint, self.constant, self.fx_node
        )

    @property
    def expr(self):
        return self.shape_env.replace(self._expr)

    # Recompute the hint and see if we've got it now
    # Precondition: self._hint is None
    def _update_hint(self):
        r = self.shape_env._maybe_evaluate_static(self.expr, compute_hint=True)
        if r is not None:
            self._hint = self.pytype(r) if not isinstance(r, SymTypes) else r

    @property
    def hint(self):
        if self._hint is None:
            self._update_hint()
        return self._hint

    def has_hint(self):
        if self._hint is None:
            self._update_hint()
        return self._hint is not None

    def require_hint(self, fallback=None):
        if self._hint is None:
            self._update_hint()
        if self._hint is None:
            if fallback is not None:
                return fallback
            # NB: we expect this to raise
            return self.shape_env.size_hint(self.expr)
        return self._hint

    def maybe_as_int(self):
        if self.expr.is_number:
            return int(self.expr)
        else:
            return None

    def is_int(self):
        return self.pytype is int

    def is_float(self):
        return self.pytype is float

    def is_bool(self):
        return self.pytype is bool

    def wrap_int(self, num):
        assert type(num) is int
        import sympy

        return SymNode(
            sympy.Integer(num), self.shape_env, int, num, constant=num, fx_node=num
        )

    def wrap_float(self, num):
        assert type(num) is float
        import sympy

        return SymNode(
            sympy.Float(num), self.shape_env, float, num, constant=num, fx_node=num
        )

    def wrap_bool(self, num):
        assert type(num) is bool
        import sympy

        return SymNode(
            sympy.true if num else sympy.false,
            self.shape_env,
            bool,
            num,
            constant=num,
            fx_node=num,
        )

    def clone(self):
        return self

    def str(self):
        return f"{self.expr}"

    def __str__(self):
        return self.str()

    def __repr__(self):
        return self.str()

    # These methods call the metaprogrammed methods, they're hand written
    # here so we get good stack traces
    def abs(self) -> "SymNode":
        return self._abs()  # type: ignore[attr-defined]

    def add(self, other) -> "SymNode":
        return self._add(other)  # type: ignore[attr-defined]

    def sub(self, other) -> "SymNode":
        return self._sub(other)  # type: ignore[attr-defined]

    def mul(self, other) -> "SymNode":
        return self._mul(other)  # type: ignore[attr-defined]

    def mod(self, other) -> "SymNode":
        return self._mod(other)  # type: ignore[attr-defined]

    def pow(self, other) -> "SymNode":
        return self._pow(other)  # type: ignore[attr-defined]

    def and_(self, other) -> "SymNode":
        return self._and_(other)  # type: ignore[attr-defined]

    def or_(self, other) -> "SymNode":
        return self._or_(other)  # type: ignore[attr-defined]

    def truediv(self, other) -> "SymNode":
        return self._truediv(other)  # type: ignore[attr-defined]

    def floordiv(self, other) -> "SymNode":
        return self._floordiv(other)  # type: ignore[attr-defined]

    def lshift(self, other) -> "SymNode":
        return self._lshift(other)  # type: ignore[attr-defined]

    def rshift(self, other) -> "SymNode":
        return self._rshift(other)  # type: ignore[attr-defined]

    def sym_not(self) -> "SymNode":  # noqa: F811
        return self._sym_not()  # type: ignore[attr-defined]

    def eq(self, other) -> "SymNode":
        return self._eq(other)  # type: ignore[attr-defined]

    def ne(self, other) -> "SymNode":
        return self._ne(other)  # type: ignore[attr-defined]

    def gt(self, other) -> "SymNode":
        return self._gt(other)  # type: ignore[attr-defined]

    def lt(self, other) -> "SymNode":
        return self._lt(other)  # type: ignore[attr-defined]

    def le(self, other) -> "SymNode":
        return self._le(other)  # type: ignore[attr-defined]

    def ge(self, other) -> "SymNode":
        return self._ge(other)  # type: ignore[attr-defined]

    def floor(self) -> "SymNode":
        return self._floor()  # type: ignore[attr-defined]

    def sym_float(self) -> "SymNode":  # noqa: F811
        return self._sym_float()  # type: ignore[attr-defined]

    def sym_int(self) -> "SymNode":
        return self._sym_int()  # type: ignore[attr-defined]

    def ceil(self) -> "SymNode":
        return self._ceil()  # type: ignore[attr-defined]

    def neg(self) -> "SymNode":
        return self._neg()  # type: ignore[attr-defined]

    def sym_min(self, other) -> "SymNode":  # noqa: F811
        return self._sym_min(other)  # type: ignore[attr-defined]

    def sym_max(self, other) -> "SymNode":  # noqa: F811
        return self._sym_max(other)  # type: ignore[attr-defined]

    def sym_ite(self, then_val, else_val) -> "SymNode":
        return self._sym_ite(then_val, else_val)  # type: ignore[attr-defined]

    def sym_sqrt(self) -> "SymNode":
        return self._sym_sqrt()  # type: ignore[attr-defined]

    def is_contiguous(self, sizes, strides) -> "SymNode":
        return self._is_contiguous(sizes, strides)  # type: ignore[attr-defined]

    def is_channels_last_contiguous_2d(self, sizes, strides) -> "SymNode":
        return self._is_channels_last_contiguous_2d(sizes, strides)  # type: ignore[attr-defined]

    def is_channels_last_contiguous_3d(self, sizes, strides) -> "SymNode":
        return self._is_channels_last_contiguous_3d(sizes, strides)  # type: ignore[attr-defined]

    def is_channels_last_strides_2d(self, sizes, strides) -> "SymNode":
        return self._is_channels_last_strides_2d(sizes, strides)  # type: ignore[attr-defined]

    def is_channels_last_strides_3d(self, sizes, strides) -> "SymNode":
        return self._is_channels_last_strides_3d(sizes, strides)  # type: ignore[attr-defined]

    def is_non_overlapping_and_dense_indicator(self, sizes, strides) -> "SymNode":
        return self._is_non_overlapping_and_dense_indicator(sizes, strides)  # type: ignore[attr-defined]

    # Make C++ happy
    def sym_or(self, other):
        return self.or_(other)

    def sym_and(self, other):
        return self.and_(other)

    def is_non_overlapping_and_dense(self, sizes, strides):
        return self.is_non_overlapping_and_dense_indicator(sizes, strides).eq(to_node(self, 1))  # type: ignore[attr-defined]

    def int_(self):
        return self.guard_int("", 0)  # NB: uses Python backtrace

    # You can manually trigger a guard with this function
    def guard_int(self, file, line):
        # TODO: use the file/line for some useful diagnostic on why a
        # guard occurred
        r = self.shape_env.evaluate_expr(self.expr, self.hint, fx_node=self.fx_node)
        try:
            return int(r)
        except Exception:
            log.warning("Failed to convert to int: %s", r)
            raise

    def guard_float(self, file, line):
        # TODO: use the file/line for some useful diagnostic on why a
        # guard occurred
        r = self.shape_env.evaluate_expr(
            self.expr, self.hint, fx_node=self.fx_node, expect_rational=False
        )
        try:
            return float(r)
        except Exception:
            log.warning("Failed to convert to float: %s", r)
            raise

    def guard_bool(self, file, line):
        # TODO: use the file/line for some useful diagnostic on why a
        # guard occurred
        r = self.shape_env.evaluate_expr(self.expr, self.hint, fx_node=self.fx_node)
        try:
            return bool(r)
        except Exception:
            log.warning("Failed to convert to bool: %s", r)
            raise

    def expect_true(self, file, line):
        if self.has_hint():
            # OK to generate guards
            return self.guard_bool(file, line)
        # Generate a deferred runtime assert (this might actually end up doing
        # a regular guard if we can!)
        # TODO: file/line here is very important, because the assert has been
        # deferred so you can't backtrace easily
        return self.shape_env.defer_runtime_assert(
            self.expr, f"{file}:{line}", fx_node=self.fx_node
        )

    def expect_size(self, file, line):
        from torch.fx.experimental.symbolic_shapes import _advise_is_size

        b = self.ge(self.wrap_int(0))
        # Generate a deferred runtime assert
        r = b.expect_true(file, line)
        # Refine compile time range, but only if it's unbacked.
        # If you refine range for hinted variables, you can end up making
        # improper deductions since compile time reasoning may be
        # incompatible with runtime reasoning.
        if r and not self.has_hint():
            _advise_is_size(SymInt(self))
        return r

    def bool_(self):
        return self.guard_bool("", 0)

    def is_symbolic(self):
        return True

    def singleton_int(self):
        return None

    def is_constant(self):
        return False


# TODO: this probably needs the sizes-strides eval functions
METHOD_TO_OPERATOR = {
    "abs": operator.abs,
    "add": operator.add,
    "and": operator.and_,
    "ceil": math.ceil,
    "eq": operator.eq,
    "floor": math.floor,
    "floordiv": operator.floordiv,
    "ge": operator.ge,
    "gt": operator.gt,
    "le": operator.le,
    "lshift": operator.lshift,
    "lt": operator.lt,
    "mod": operator.mod,
    "mul": operator.mul,
    "ne": operator.ne,
    "neg": operator.neg,
    "or": operator.or_,
    "pow": operator.pow,
    "rshift": operator.rshift,
    "sub": operator.sub,
    "sym_float": sym_float,
    "sym_ite": sym_ite,
    "sym_max": sym_max,
    "sym_min": sym_min,
    "sym_not": sym_not,
    "sym_sqrt": sym_sqrt,
    "truediv": operator.truediv,
}

unary_magic_methods = {
    "abs",
    "sym_float",
    "ceil",
    "floor",
    "neg",
    "sym_sqrt",
    "sym_not",
}

# Most methods are only registered on SymInt and SymFloat
# Some methods are only be registered on SymBool
only_bool_magic_methods = {"and", "or", "sym_not", "sym_ite"}
# Methods that implicitly convert SymBool into SymInt
bool_becomes_int_magic_methods = {"add", "sub", "mul"}
# Methods that are also on SymBool, in addition to on SymInt and SymFloat
also_bool_magic_methods = {"eq"}
bool_magic_methods = only_bool_magic_methods | also_bool_magic_methods


magic_methods_on_operator_with_trailing_underscore = {"and", "or"}


always_float_magic_methods = {"truediv", "sym_float", "sym_sqrt", "pow"}
always_int_magic_methods = {"ceil", "floor"}
always_bool_magic_methods = {
    "eq",
    "ne",
    "gt",
    "lt",
    "le",
    "ge",
    "and",
    "or",
    "sym_not",
    "is_non_overlapping_and_dense",
}

# Methods that have a `__foo__` as well as `__rfoo__`


def _sympy_truediv(a, b):
    from torch.utils._sympy.functions import TrueDiv

    return TrueDiv(a, b)


def _sympy_floordiv(a, b):
    from torch.utils._sympy.functions import FloorDiv

    return FloorDiv(a, b)


def _sympy_mod(a, b):
    from torch.utils._sympy.functions import Mod

    return Mod(a, b)


def _sympy_pow(a, b):
    from torch.utils._sympy.functions import Pow

    return Pow(a, b)


def _sympy_and(a, b):
    import sympy

    return sympy.And(a, b)


def _sympy_or(a, b):
    import sympy

    return sympy.Or(a, b)


def _sympy_lshift(a, b):
    from torch.utils._sympy.functions import LShift

    return LShift(a, b)


def _sympy_rshift(a, b):
    from torch.utils._sympy.functions import RShift

    return RShift(a, b)


reflectable_magic_methods = {
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
    "mod": _sympy_mod,
    "pow": _sympy_pow,
    "and": _sympy_and,
    "or": _sympy_or,
    "truediv": _sympy_truediv,
    "floordiv": _sympy_floordiv,
    "lshift": _sympy_lshift,
    "rshift": _sympy_rshift,
}


def _floor_ceil_helper(a, fn):
    import sympy

    if isinstance(a, sympy.Mul):
        aa = a.args
        if len(aa) == 2 and isinstance(aa[0], sympy.Float) and aa[1].is_integer:
            coef = sympy.Integer(aa[0])
            if aa[0] == coef:  # structural equality test
                return coef * aa[1]
    if (
        isinstance(a, sympy.Float)
        and a == sympy.Integer(a)
        or isinstance(a, sympy.Integer)
    ):
        return sympy.Integer(a)
    return fn(a)


def _sympy_floor(a):
    import sympy

    return _floor_ceil_helper(a, sympy.floor)


def _sympy_ceil(a):
    import sympy

    return _floor_ceil_helper(a, sympy.ceiling)


def _sympy_eq(a, b):
    import sympy

    return sympy.Eq(a, b)


def _sympy_ne(a, b):
    import sympy

    return sympy.Ne(a, b)


def _sympy_gt(a, b):
    import sympy

    return sympy.Gt(a, b)


def _sympy_lt(a, b):
    import sympy

    return sympy.Lt(a, b)


def _sympy_le(a, b):
    import sympy

    return sympy.Le(a, b)


def _sympy_ge(a, b):
    import sympy

    return sympy.Ge(a, b)


def _sympy_min(a, b):
    import sympy

    return sympy.Min(a, b)


def _sympy_max(a, b):
    import sympy

    return sympy.Max(a, b)


def _sympy_ite(a, t, f):
    import sympy

    return sympy.Piecewise((t, a), (f, True))


def _sympy_sqrt(a):
    import sympy

    return sympy.sqrt(a)


def _sympy_abs(a):
    import sympy

    return sympy.Abs(a)


def _sympy_sym_float(a):
    # Cannot use sympy.Float(a) here, coz it expects python literals
    # Multiply by 1.0 to cast to float. This is needed when the input
    # is a SymInt which has the assumption that it is integer and
    # SymPy will otherwise assume that return value cannot be a float.
    return a * 1.0


magic_methods = {
    **reflectable_magic_methods,
    "sym_not": lambda a: ~a,
    "eq": _sympy_eq,
    "ne": _sympy_ne,
    "gt": _sympy_gt,
    "lt": _sympy_lt,
    "le": _sympy_le,
    "ge": _sympy_ge,
    "floor": _sympy_floor,
    "sym_float": _sympy_sym_float,
    "ceil": _sympy_ceil,
    "neg": lambda a: -a,
    "sym_min": _sympy_min,
    "sym_max": _sympy_max,
    "sym_ite": _sympy_ite,
    "sym_sqrt": _sympy_sqrt,
    "abs": _sympy_abs,
}


def sympy_is_contiguous(sizes, strides):
    dim = len(sizes)
    return sympy_is_contiguous_generic(sizes, strides, list(range(dim - 1, -1, -1)))


def sympy_is_contiguous_generic(sizes, strides, dim_order):
    import sympy

    dim = len(sizes)

    if len(dim_order) != dim:
        return sympy.false

    is_contiguous = sympy.true
    z = sympy.Integer(1)
    # Contiguous if the strides make sense (or the dim is size 1)
    for d in dim_order:
        is_contiguous &= sympy.Eq(sizes[d], sympy.Integer(1)) | sympy.Eq(strides[d], z)
        z *= sizes[d]
    # OR if any size is zero
    for d in range(dim):
        is_contiguous |= sympy.Eq(sizes[d], sympy.Integer(0))
    return is_contiguous


# NB: There is a TODO in C++ to allow omitting the batch dim.  If that
# happens you will need to refactor this


def sympy_is_channels_last_contiguous_2d(sizes, strides):
    return sympy_is_contiguous_generic(sizes, strides, [1, 3, 2, 0])


def sympy_is_channels_last_contiguous_3d(sizes, strides):
    return sympy_is_contiguous_generic(sizes, strides, [1, 4, 3, 2, 0])


def sympy_is_channels_last_strides_generic(sizes, strides, dim_order):
    import sympy

    dim = len(sizes)

    if dim != len(dim_order):
        return sympy.false

    m = sympy.Integer(0)
    r = sympy.true

    # special case for trivial C dimension. default to NCHW
    r &= sympy.Ne(strides[1], 0)

    for d in dim_order:
        r &= sympy.Ne(sizes[d], 0) & (strides[d] >= m)
        # Fallback to NCHW as default layout for ambiguous cases
        # This is the flaw of implicit memory_format from strides.
        # N111 tensor with identical strides for size 1 dimension;
        # Two cases could lead us here:
        # a. N111 contiguous Tensor ([N,1,1,1]@[1,1,1,1])
        # b. N11W contiguous Tensor sliced on the W-dimension.
        # ([N,1,1,1]@[W,W,W,W])
        if d == 0:
            r &= sympy.Ne(m, strides[1])
        # This is necessary to:
        # 1. distinguish the memory_format of N1H1;
        #     [H, 1, 1, 1] channels_last stride
        #     [H, H, 1, 1] contiguous stride
        # 2. permutation of 1C1W:
        #     [1, C, 1, H]@[HC, H, H, 1] transpose(1, 3)
        #     [1, H, 1, C]@[HC, 1, H, H] shouldn't be identified as
        #     channels_last
        m = strides[d] * sympy.Max(sizes[d], 1)

    return r


def sympy_is_channels_last_strides_2d(sizes, strides):
    return sympy_is_channels_last_strides_generic(sizes, strides, [1, 3, 2, 0])


def sympy_is_channels_last_strides_3d(sizes, strides):
    return sympy_is_channels_last_strides_generic(sizes, strides, [1, 4, 3, 2, 0])


def _sympy_is_non_overlapping_and_dense_indicator(sizes, strides):
    from torch.utils._sympy.functions import IsNonOverlappingAndDenseIndicator

    return IsNonOverlappingAndDenseIndicator(*sizes, *strides)


sizes_strides_methods = {
    # TODO: These could also be done with indicators, maybe it is better
    # for reasoning to do it that way
    "is_contiguous": sympy_is_contiguous,
    "is_channels_last_contiguous_2d": sympy_is_channels_last_contiguous_2d,
    "is_channels_last_contiguous_3d": sympy_is_channels_last_contiguous_3d,
    "is_channels_last_strides_2d": sympy_is_channels_last_strides_2d,
    "is_channels_last_strides_3d": sympy_is_channels_last_strides_3d,
    "is_non_overlapping_and_dense_indicator": _sympy_is_non_overlapping_and_dense_indicator,
}

alternate_impl_if_hinted_methods = {
    "sym_min": builtins.min,
    "sym_max": builtins.max,
}


def to_node(self, num):
    if isinstance(num, SymTypes):
        return num.node
    elif type(num) is bool:
        return self.wrap_bool(num)
    elif type(num) is int:
        return self.wrap_int(num)
    elif type(num) is float:
        return self.wrap_float(num)
    else:
        # NotImplemented is important so that Python tries the
        # other magic method
        return NotImplemented


def wrap_node(x):
    # TODO: let C++ also take advantage of this
    if isinstance(x, SymNode) and x.constant is not None:
        return x.constant
    if x.is_int():
        return SymInt(x)
    elif x.is_float():
        return SymFloat(x)
    elif x.is_bool():
        return SymBool(x)
    else:
        raise AssertionError(f"unrecognized return type {x}")


def method_to_operator(method):
    return METHOD_TO_OPERATOR[method]


def _make_node_magic(method, func):
    func = lru_cache(256)(func)

    if method in magic_methods_on_operator_with_trailing_underscore:
        method_attr = f"{method}_"
    else:
        method_attr = method

    def binary_magic_impl(self, other):
        from torch.fx.experimental.symbolic_shapes import safe_expand

        op = method_to_operator(method)

        out_hint = None
        if self.hint is not None and other.hint is not None:
            out_hint = op(self.hint, other.hint)

        alternate_impl = alternate_impl_if_hinted_methods.get(method)
        if alternate_impl and out_hint is not None:
            return to_node(self, alternate_impl(wrap_node(self), wrap_node(other)))

        if sym_function_mode():
            return to_node(
                self, handle_sym_dispatch(op, (wrap_node(self), wrap_node(other)), {})
            )
        assert isinstance(other, SymNode)
        # TODO: consider constant prop here
        try:
            out = func(self.expr, other.expr)
        except Exception:
            log.warning("failed to eval %s(%s, %s)", method, self.expr, other.expr)
            raise
        out = safe_expand(out)
        pytype: Type
        # This is not strictly correct. In Python, a**b may return complex when
        # a < 0 and b is a float: (-1)**2.1. Same for sympy.sqrt(-3.14). This
        # returns a float while both arguments are ints: 2**(-1). Also, max and
        # min do not type promote. To avoid having data-dependent control flow
        # here, we just set the type to float if one of the args is a float. In
        # case of a type mismatch, we assume that it will be detected during
        # evaluation.
        if method in always_float_magic_methods:
            pytype = float
        elif method in always_bool_magic_methods:
            pytype = bool
        elif self.pytype is float or other.pytype is float:
            pytype = float
        else:
            pytype = self.pytype

        if (
            pytype is not None
            and out_hint is not None
            and not isinstance(out_hint, SymTypes)
        ):
            out_hint = pytype(out_hint)

        # Create a FX node that corresponds to the operation being applied to
        # this node.
        fx_node, _ = self.shape_env.create_fx_call_function(
            op, (self.fx_node, other.fx_node)
        )
        return SymNode(out, self.shape_env, pytype, out_hint, fx_node=fx_node)

    def unary_magic_impl(self):
        from torch.fx.experimental.symbolic_shapes import safe_expand

        op = method_to_operator(method)
        if sym_function_mode():
            return to_node(self, handle_sym_dispatch(op, (wrap_node(self),), {}))
        # TODO: consider constant prop here
        expr = self.expr
        if method == "floor" or method == "ceiling":
            expr = self.shape_env._simplify_floor_div(expr)

        try:
            out = func(expr)
        except Exception:
            log.warning("failed to eval %s(%s)", method, expr)
            raise

        out_hint = None
        if self.hint is not None:
            out_hint = op(self.hint)
        out = safe_expand(out)
        pytype: Type
        if method in always_int_magic_methods:
            pytype = int
        elif method in always_float_magic_methods:
            pytype = float
        else:
            pytype = self.pytype

        fx_node, _ = self.shape_env.create_fx_call_function(op, (self.fx_node,))
        return SymNode(out, self.shape_env, pytype, out_hint, fx_node=fx_node)

    if method in unary_magic_methods:
        setattr(SymNode, f"_{method_attr}", unary_magic_impl)
    elif method == "sym_ite":

        def sym_ite_impl(pred_node, then_node, else_node):
            from torch.fx.experimental.symbolic_shapes import safe_expand

            out_hint = then_node.hint if pred_node.hint else else_node.hint
            if sym_function_mode():
                return to_node(
                    pred_node,
                    handle_sym_dispatch(
                        sym_ite,
                        (
                            wrap_node(pred_node),
                            wrap_node(then_node),
                            wrap_node(else_node),
                        ),
                        {},
                    ),
                )

            try:
                out = func(pred_node.expr, then_node.expr, else_node.expr)
            except Exception:
                log.warning(
                    "failed to eval %s(%s, %s, %s)",
                    method,
                    pred_node.expr,
                    then_node.expr,
                    else_node.expr,
                )
                raise

            out = safe_expand(out)
            fx_node, _ = pred_node.shape_env.create_fx_call_function(
                sym_ite, (pred_node.fx_node, then_node.fx_node, else_node.fx_node)
            )
            return SymNode(
                out, pred_node.shape_env, then_node.pytype, out_hint, fx_node=fx_node
            )

        setattr(SymNode, f"_{method_attr}", sym_ite_impl)
    else:
        setattr(SymNode, f"_{method_attr}", binary_magic_impl)


def _make_node_sizes_strides(method, func):
    # NB: don't LRU cache, lots of arguments

    def sizes_strides_impl(self, sizes, strides):
        op = getattr(sys.modules[__name__], method)
        if sym_function_mode():
            return to_node(
                self,
                handle_sym_dispatch(
                    op,
                    ([wrap_node(s) for s in sizes], [wrap_node(s) for s in strides]),
                    {},
                ),
            )
        size_exprs = [s.expr for s in sizes]
        stride_exprs = [s.expr for s in strides]
        try:
            out = func(size_exprs, stride_exprs)
        except Exception:
            log.warning("failed to eval %s(%s, %s)", method, size_exprs, stride_exprs)
            raise
        # bool is never expandable

        size_hints = []
        out_hint = None
        for s in sizes:
            if s.hint is None:
                break
            size_hints.append(s.hint)
        else:
            stride_hints = []
            for s in strides:
                if s.hint is None:
                    break
                stride_hints.append(s.hint)
            else:
                out_hint = op(size_hints, stride_hints)

        # NB: This is the indicator function, not the actual bool!
        pytype: Type
        if method.endswith("_indicator"):
            pytype = int
        else:
            pytype = bool
        return SymNode(out, self.shape_env, pytype, out_hint)

    setattr(SymNode, f"_{method}", sizes_strides_impl)

    # TODO: This is technically hotpath, but in the ideal end state
    # guards on this will resolve at a higher level so you never
    # spend time in this code
    def sizes_strides_user(sizes, strides):
        import sympy

        from torch.fx.experimental.symbolic_shapes import (
            eval_is_non_overlapping_and_dense,
        )

        for a in itertools.chain(sizes, strides):
            if isinstance(a, SymInt):
                return wrap_node(
                    getattr(a.node, method)(
                        [to_node(a.node, b) for b in sizes],
                        [to_node(a.node, b) for b in strides],
                    )
                )
        if method == "is_non_overlapping_and_dense_indicator":
            return eval_is_non_overlapping_and_dense(sizes, strides)
        else:
            # TODO: this is an awful implementation
            return bool(
                func(
                    [sympy.sympify(a) for a in sizes],
                    [sympy.sympify(a) for a in strides],
                )
            )

    # Skip for is_non_overlapping_and_dense_indicator
    if not hasattr(sys.modules[__name__], method):
        setattr(sys.modules[__name__], method, sizes_strides_user)


for method, func in magic_methods.items():
    _make_node_magic(method, func)

for method, func in sizes_strides_methods.items():
    _make_node_sizes_strides(method, func)


def _make_user_magic(method, user_type):
    # User magic takes care of wrapping the other operand into a node,
    # so that our internal logic can assume everything is nodes

    if method in magic_methods_on_operator_with_trailing_underscore:
        method_attr = f"{method}_"
    else:
        method_attr = method

    def get_constant(x: Union[SymInt, int, SymFloat, float, SymBool, bool]):
        if isinstance(x, (int, float, bool)):
            return x
        if isinstance(x, SymBool):
            return x.node.guard_bool("", 0)
        raise AssertionError("expect to be called with constant SymBools")

    def is_constant(x):
        if isinstance(x, (int, float, bool)):
            return True
        if isinstance(x, (SymInt, SymFloat, SymBool)):
            return x.node.is_constant()
        return False

    if method in bool_becomes_int_magic_methods:

        def promote(x):
            """Implements True+True=2, which works in python but not sympy"""
            if isinstance(x, SymBool):
                return SymInt(x.node.wrap_int(int(x)))
            return x

    else:

        def promote(x):
            return x

    # Before and after performing the operation, check if any operands are constant.
    # If so, extract out the constant values first. If `self` itself is a
    # constant, then "redispatch" by calling back into the operator. Sometimes
    # this means that operations involving SymBool return plain bools.
    # Alternatively, we could also rewrap into constant Symbool (i.e. by
    # implementing wrap_bool in ConstantSymNodeImpl), but we're not doing that
    # today for no particular reason.
    def unary_magic_impl(self):
        self = promote(self)
        if is_constant(self):
            return (method_to_operator(method))(get_constant(self))
        return wrap_node(getattr(self.node, method_attr)())

    def binary_magic_impl(self, other):
        self = promote(self)
        other = promote(other)
        if is_constant(self):
            return (method_to_operator(method))(get_constant(self), other)
        if is_constant(other):
            other = get_constant(other)
        other_node = to_node(self.node, other)
        if other_node is NotImplemented:
            return NotImplemented
        ret = wrap_node(getattr(self.node, method_attr)(other_node))
        return get_constant(ret) if is_constant(ret) else ret

    def rbinary_magic_impl(self, other):
        self = promote(self)
        other = promote(other)
        if is_constant(self):
            return (method_to_operator(method))(get_constant(self), other)
        if is_constant(other):
            other = get_constant(other)
        other_node = to_node(self.node, other)
        if other_node is NotImplemented:
            return NotImplemented
        ret = wrap_node(getattr(other_node, method_attr)(self.node))
        return get_constant(ret) if is_constant(ret) else ret

    if method in unary_magic_methods:
        setattr(user_type, f"__{method}__", unary_magic_impl)
    elif method == "sym_ite":

        def sym_ite_magic_impl(pred, then_val, else_val):
            pred_node = pred.node
            then_node = to_node(pred_node, then_val)
            else_node = to_node(pred_node, else_val)
            if then_node is NotImplemented or else_node is NotImplemented:
                return NotImplemented
            assert (
                isinstance(then_node, SymNode)
                and isinstance(else_node, SymNode)
                and then_node.pytype == else_node.pytype
            )
            ret = wrap_node(getattr(pred.node, method_attr)(then_node, else_node))
            return get_constant(ret) if ret.node.is_constant() else ret

        setattr(user_type, f"__{method}__", sym_ite_magic_impl)
    else:
        setattr(user_type, f"__{method}__", binary_magic_impl)
        if method in reflectable_magic_methods:
            setattr(user_type, f"__r{method}__", rbinary_magic_impl)


for method, func in magic_methods.items():  # type: ignore[assignment]
    if method in only_bool_magic_methods:
        _make_user_magic(method, SymBool)
        continue
    if method in also_bool_magic_methods or method in bool_becomes_int_magic_methods:
        _make_user_magic(method, SymBool)
    _make_user_magic(method, SymInt)
    _make_user_magic(method, SymFloat)

del method
del func
