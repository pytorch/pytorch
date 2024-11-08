# mypy: allow-untyped-defs

from __future__ import annotations


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
from functools import lru_cache, update_wrapper
from typing import Optional, Type, TYPE_CHECKING, Union

import torch

# NB: The sym_* functions are used via getattr() and must be imported here.
from torch import (  # noqa: F401
    sym_float,
    sym_ite,
    sym_max,
    sym_min,
    sym_not,
    SymBool,
    SymFloat,
    SymInt,
)


if TYPE_CHECKING:
    from torch.fx.experimental.symbolic_shapes import ShapeEnv

log = logging.getLogger(__name__)
sym_node_log = torch._logging.getArtifactLogger(__name__, "sym_node")


__all__ = ["SymNode", "method_to_operator", "magic_methods"]


from torch.types import py_sym_types as SymTypes


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

        def compute_hint():
            from torch.fx.experimental.symbolic_shapes import free_unbacked_symbols

            # This occasionally gets exercised by, e.g.,
            # convert_shape_to_symint.  It's just a nicety so you don't HAVE
            # to have a correct hint on hand when making a SymNode.
            # Don't attempt to compute for unbacked, this can be quite
            # expensive.
            if free_unbacked_symbols(self.expr):
                return None
            hint = self.shape_env._maybe_evaluate_static(self.expr, compute_hint=True)
            if hint is not None:
                hint = self.pytype(hint) if not isinstance(hint, SymTypes) else hint
            return hint

        if hint is not None:
            assert type(hint) is pytype or type(hint) is _to_symtype(pytype), (
                "Cannot create SymNode of type "
                f"{pytype} with incompatible hint of type {type(hint)}"
            )
            if self.shape_env and self.shape_env._translation_validation_enabled:
                # This is technically not TV, but this assert is expensive so
                # let's only do it when we're already doing expensive things
                computed_hint = compute_hint()
                assert (
                    hint == computed_hint
                ), f"{hint} != {computed_hint} (for {self.expr})"
        else:
            hint = compute_hint()
        self._hint = hint
        self.constant: Optional[Union[int, float, bool]] = constant

        # Record the FX node of the current node if we are doing translation
        # validation. They will be used for building the input assertions for
        # the translation validation problem.
        tx_validation_en = (
            self.shape_env and self.shape_env._translation_validation_enabled
        )
        self.fx_node = tx_validation_en and fx_node

    def with_shape_env(self, shape_env: ShapeEnv) -> SymNode:
        return SymNode(
            self._expr, shape_env, self.pytype, self._hint, self.constant, self.fx_node
        )

    def _value_eq(self, other: SymNode) -> bool:
        # Purposely don't include the shape_env in the eq.
        return (
            self._expr == other._expr
            and self.pytype == other.pytype
            and self._hint == other._hint
            and self.constant == other.constant
            and self.fx_node == other.fx_node
        )

    def _value_hash(self) -> int:
        # Purposely don't include the shape_env in the hash.
        return hash((self._expr, self.pytype, self._hint, self.constant, self.fx_node))

    @property
    def expr(self):
        return self.shape_env.replace(self._expr)

    @property
    def hint(self):
        return self._hint

    def has_hint(self):
        return self._hint is not None

    def require_hint(self, fallback=None):
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

    # NB: This does conversions, not sure if this is good or not
    def maybe_as_float(self):
        import sympy

        if isinstance(self.expr, sympy.Float):
            return float(self.expr)
        else:
            return None

    def maybe_as_bool(self):
        import sympy

        if self.expr is sympy.true:
            return True
        elif self.expr is sympy.false:
            return False
        else:
            return None

    def is_int(self):
        return self.pytype is int

    def is_float(self):
        return self.pytype is float

    def is_bool(self):
        return self.pytype is bool

    def is_nested_int(self):
        # Unbacked SymInts cannot be nested int today
        return (
            self._hint is not None
            and isinstance(self._hint, SymInt)
            and self._hint.node.is_nested_int()
        )

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
        rep = [
            f"SymNode({self._expr}, shape_env={self.shape_env}, pytype={self.pytype}",
        ]
        if self._hint is not None:
            rep.append(f"hint={self._hint}")
        if self.constant is not None:
            rep.append(f"constant={self.constant}")
        if self.fx_node is not None:
            rep.append(f"fx_node={self.fx_node}")
        return ", ".join(rep) + ")"

    def _graph_repr(self) -> builtins.str:
        # Representation used by GraphModule to create a pythonic version of a graph
        return self.str()

    # These methods call the metaprogrammed methods, they're hand written
    # here so we get good stack traces
    def abs(self) -> SymNode:
        return self._abs()  # type: ignore[attr-defined]

    def pos(self) -> SymNode:
        return self._pos()  # type: ignore[attr-defined]

    def round(self, ndigits=None) -> SymNode:
        return self._round(ndigits)  # type: ignore[attr-defined]

    def trunc(self) -> SymNode:
        return self._trunc()  # type: ignore[attr-defined]

    def add(self, other) -> SymNode:
        return self._add(other)  # type: ignore[attr-defined]

    def sub(self, other) -> SymNode:
        return self._sub(other)  # type: ignore[attr-defined]

    def mul(self, other) -> SymNode:
        return self._mul(other)  # type: ignore[attr-defined]

    def mod(self, other) -> SymNode:
        return self._mod(other)  # type: ignore[attr-defined]

    def float_pow(self, other) -> SymNode:
        return self._float_pow(other)  # type: ignore[attr-defined]

    def pow_by_natural(self, other) -> SymNode:
        return self._pow_by_natural(other)  # type: ignore[attr-defined]

    def and_(self, other) -> SymNode:
        return self._and_(other)  # type: ignore[attr-defined]

    def or_(self, other) -> SymNode:
        return self._or_(other)  # type: ignore[attr-defined]

    def float_truediv(self, other) -> SymNode:
        return self._float_truediv(other)  # type: ignore[attr-defined]

    def int_truediv(self, other) -> SymNode:
        return self._int_truediv(other)  # type: ignore[attr-defined]

    def int_floordiv(self, other) -> SymNode:
        return self._int_floordiv(other)  # type: ignore[attr-defined]

    def lshift(self, other) -> SymNode:
        return self._lshift(other)  # type: ignore[attr-defined]

    def rshift(self, other) -> SymNode:
        return self._rshift(other)  # type: ignore[attr-defined]

    def sym_not(self) -> SymNode:  # noqa: F811
        return self._sym_not()  # type: ignore[attr-defined]

    def eq(self, other) -> SymNode:
        return self._eq(other)  # type: ignore[attr-defined]

    def ne(self, other) -> SymNode:
        return self._ne(other)  # type: ignore[attr-defined]

    def gt(self, other) -> SymNode:
        return self._gt(other)  # type: ignore[attr-defined]

    def lt(self, other) -> SymNode:
        return self._lt(other)  # type: ignore[attr-defined]

    def le(self, other) -> SymNode:
        return self._le(other)  # type: ignore[attr-defined]

    def ge(self, other) -> SymNode:
        return self._ge(other)  # type: ignore[attr-defined]

    def floor(self) -> SymNode:
        return self._floor()  # type: ignore[attr-defined]

    def is_integer(self) -> SymNode:
        return self._is_integer()  # type: ignore[attr-defined]

    def sym_float(self) -> SymNode:  # noqa: F811
        return self._sym_float()  # type: ignore[attr-defined]

    def sym_int(self) -> SymNode:
        return self._sym_int()  # type: ignore[attr-defined]

    def ceil(self) -> SymNode:
        return self._ceil()  # type: ignore[attr-defined]

    def neg(self) -> SymNode:
        return self._neg()  # type: ignore[attr-defined]

    def sym_min(self, other) -> SymNode:  # noqa: F811
        return self._sym_min(other)  # type: ignore[attr-defined]

    def sym_max(self, other) -> SymNode:  # noqa: F811
        return self._sym_max(other)  # type: ignore[attr-defined]

    def sym_ite(self, then_val, else_val) -> SymNode:
        return self._sym_ite(then_val, else_val)  # type: ignore[attr-defined]

    def is_contiguous(self, sizes, strides) -> SymNode:
        return self._is_contiguous(sizes, strides)  # type: ignore[attr-defined]

    def is_channels_last_contiguous_2d(self, sizes, strides) -> SymNode:
        return self._is_channels_last_contiguous_2d(sizes, strides)  # type: ignore[attr-defined]

    def is_channels_last_contiguous_3d(self, sizes, strides) -> SymNode:
        return self._is_channels_last_contiguous_3d(sizes, strides)  # type: ignore[attr-defined]

    def is_channels_last_strides_2d(self, sizes, strides) -> SymNode:
        return self._is_channels_last_strides_2d(sizes, strides)  # type: ignore[attr-defined]

    def is_channels_last_strides_3d(self, sizes, strides) -> SymNode:
        return self._is_channels_last_strides_3d(sizes, strides)  # type: ignore[attr-defined]

    def is_non_overlapping_and_dense_indicator(self, sizes, strides) -> SymNode:
        return self._is_non_overlapping_and_dense_indicator(sizes, strides)  # type: ignore[attr-defined]

    # Make C++ happy
    def sym_or(self, other):
        return self.or_(other)

    def sym_and(self, other):
        return self.and_(other)

    # There is no int_truediv available from C++
    def truediv(self, other):
        return self.float_truediv(other)

    def floordiv(self, other) -> SymNode:
        return self.int_floordiv(other)

    # We didn't bind integer pow in C++
    def pow(self, other):
        return self.float_pow(other)

    def is_non_overlapping_and_dense(self, sizes, strides):
        return self.is_non_overlapping_and_dense_indicator(sizes, strides).eq(to_node(self, 1))  # type: ignore[attr-defined]

    def int_(self):
        return self.guard_int("", 0)  # NB: uses Python backtrace

    # This one is currently done by hand, but if we add other variadic
    # functions consider factoring it out to be metaprogrammed too.  Note that
    # some load bearing logic is directly in torch.sym_sum

    def sym_sum(self, args) -> SymNode:
        import sympy

        # Inner impl
        from torch.fx.experimental.proxy_tensor import (
            get_proxy_mode,
            handle_sym_dispatch,
        )

        if get_proxy_mode():
            return to_node(
                self,
                handle_sym_dispatch(
                    torch.sym_sum,
                    (tuple(wrap_node(a) for a in args),),
                    {},
                ),
            )
        exprs = [a.expr for a in args]
        out = sympy.Add(*exprs)

        size_hints = []
        out_hint = None
        for a in args:
            if a.hint is None:
                break
            size_hints.append(a.hint)
        else:
            out_hint = sum(size_hints)

        fx_node, _ = self.shape_env._create_fx_call_function(
            torch.sym_sum, (tuple(a.fx_node for a in args),)
        )

        # NB: Only for integers!
        return SymNode(out, self.shape_env, int, out_hint, fx_node=fx_node)

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
        r = self.shape_env.evaluate_expr(self.expr, self.hint, fx_node=self.fx_node)
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
        from torch.fx.experimental.symbolic_shapes import free_unbacked_symbols

        if (
            self.has_hint()
            and not free_unbacked_symbols(self.expr)
            and not self.shape_env.prefer_deferred_runtime_asserts_over_guards
        ):
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

    def guard_size_oblivious(self, file, line):
        """
        Like guard_bool, but if we encounter unbacked symbols, if those symbols
        are size-like, we will treat them as >= 2 for the purposes of the analysis.

        This CHANGES the runtime semantics, but all size-oblivious sites have been
        audited to ensure that the runtime semantics don't change in a material way.
        Acceptable runtime semantic changes are, e.g., squeeze() no longer dropping
        an unbacked one size, or a tensor reporting as non-contiguous even if it's
        contiguous if it would have been reported contiguous due to being empty.
        """
        # TODO: use the file/line for some useful diagnostic on why a
        # guard occurred
        r = self.shape_env.evaluate_expr(
            self.expr, self.hint, fx_node=self.fx_node, size_oblivious=True
        )
        try:
            return bool(r)
        except Exception:
            log.warning("Failed to convert to bool: %s", r)
            raise

    def bool_(self):
        return self.guard_bool("", 0)

    def is_symbolic(self):
        return True

    def nested_int(self):
        return None

    def is_constant(self):
        return False


# TODO: this probably needs the sizes-strides eval functions
METHOD_TO_OPERATOR = {
    "pos": operator.pos,
    "abs": operator.abs,
    "add": operator.add,
    "and": operator.and_,
    "ceil": math.ceil,
    "eq": operator.eq,
    "floor": math.floor,
    "trunc": math.trunc,
    "int_floordiv": operator.floordiv,
    "ge": operator.ge,
    "gt": operator.gt,
    "is_integer": lambda x: x.is_integer(),
    "le": operator.le,
    "lshift": operator.lshift,
    "lt": operator.lt,
    "mod": operator.mod,
    "mul": operator.mul,
    "ne": operator.ne,
    "neg": operator.neg,
    "or": operator.or_,
    "float_pow": operator.pow,
    "pow_by_natural": operator.pow,
    "round": builtins.round,
    "rshift": operator.rshift,
    "sub": operator.sub,
    "sym_float": sym_float,
    "sym_ite": sym_ite,
    "sym_max": sym_max,
    "sym_min": sym_min,
    "sym_not": sym_not,
    "float_truediv": operator.truediv,
    "int_truediv": operator.truediv,
}

unary_magic_methods = {
    "abs",
    "sym_float",
    "sym_int",
    "ceil",
    "floor",
    "neg",
    "sym_not",
    "pos",
    "trunc",
}


# Adding math ops: sqrt, cos, sin, ...
def _get_sym_node_fn(name):
    def fn(self):
        return getattr(self, f"_sym_{name}")()

    return fn


math_op_names = (
    "sqrt",
    "cos",
    "cosh",
    "sin",
    "sinh",
    "tan",
    "tanh",
    "asin",
    "acos",
    "atan",
    "log2",
)
for name in math_op_names:
    sym_name = f"sym_{name}"
    priv_sym_name = f"_{sym_name}"
    setattr(SymNode, sym_name, _get_sym_node_fn(name))
    METHOD_TO_OPERATOR[sym_name] = getattr(torch, priv_sym_name)
    unary_magic_methods.add(sym_name)
    __all__.append(sym_name)


# Unary methods that are not magic methods
unary_nonmagic_methods = {
    "is_integer",
}

unary_methods = unary_magic_methods | unary_nonmagic_methods

# Most methods are only registered on SymInt and SymFloat
# Some methods are only be registered on SymBool
only_bool_magic_methods = {"and", "or", "sym_not", "sym_ite"}
# Methods that implicitly convert SymBool into SymInt
bool_becomes_int_magic_methods = {"add", "sub", "mul"}
# Methods that are also on SymBool, in addition to on SymInt and SymFloat
also_bool_magic_methods = {"eq"}
bool_magic_methods = only_bool_magic_methods | also_bool_magic_methods

# Methods that are only for float
only_float_magic_methods = {"is_integer", "round", "sym_int", "sym_log2"}


magic_methods_on_operator_with_trailing_underscore = {"and", "or"}


always_float_magic_methods = {"int_truediv", "float_truediv", "sym_float", "float_pow"}

for name in math_op_names:
    sym_name = f"sym_{name}"
    always_float_magic_methods.add(sym_name)


always_int_magic_methods = {"ceil", "floor", "trunc", "pow_by_natural"}
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
    "is_integer",
}

# Methods that have a `__foo__` as well as `__rfoo__`


def _sympy_float_truediv(a, b):
    from torch.utils._sympy.functions import FloatTrueDiv

    return FloatTrueDiv(a, b)


def _sympy_int_truediv(a, b):
    from torch.utils._sympy.functions import IntTrueDiv

    return IntTrueDiv(a, b)


def _sympy_floordiv(a, b):
    from torch.utils._sympy.functions import FloorDiv

    return FloorDiv(a, b)


def _sympy_mod(a, b):
    from torch.utils._sympy.functions import Mod, PythonMod

    if a.is_nonnegative and b.is_nonnegative:
        return Mod(a, b)
    else:
        return PythonMod(a, b)


def _sympy_pow_by_natural(a, b):
    from torch.utils._sympy.functions import PowByNatural

    return PowByNatural(a, b)


def _sympy_float_pow(a, b):
    from torch.utils._sympy.functions import FloatPow

    return FloatPow(a, b)


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
    "add": operator.add,
    "sub": operator.sub,
    "mul": operator.mul,
    "mod": _sympy_mod,
    "pow_by_natural": _sympy_pow_by_natural,
    "float_pow": _sympy_float_pow,
    "and": _sympy_and,
    "or": _sympy_or,
    "float_truediv": _sympy_float_truediv,
    "int_truediv": _sympy_int_truediv,
    "int_floordiv": _sympy_floordiv,
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
    from torch.utils._sympy.functions import FloorToInt

    return FloorToInt(a)


# NB: this is Python trunc semantics which returns an int.  Do NOT use this to
# represent torch.trunc (which is float to float)
def _sympy_trunc(a):
    from torch.utils._sympy.functions import TruncToInt

    return TruncToInt(a)


def _sympy_ceil(a):
    from torch.utils._sympy.functions import CeilToInt

    return CeilToInt(a)


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
    from torch.utils._sympy.functions import Min

    return Min(a, b)


def _sympy_max(a, b):
    from torch.utils._sympy.functions import Max

    return Max(a, b)


def _sympy_ite(a, t, f):
    import sympy

    return sympy.Piecewise((t, a), (f, True))


current_module = sys.modules[__name__]


def _get_sym_math_fn(name):
    def fn(a):
        import torch.utils._sympy.functions

        return getattr(torch.utils._sympy.functions, f"OpaqueUnaryFn_{name}")(a)

    return fn


for name in math_op_names:
    priv_sympy_name = f"_sympy_{name}"
    fn = _get_sym_math_fn(name)
    fn.__qualname__ = fn.__name__ = priv_sympy_name
    setattr(current_module, priv_sympy_name, fn)

del fn, name, priv_sympy_name  # type: ignore[possibly-undefined]


def _sympy_abs(a):
    import sympy

    return sympy.Abs(a)


def _sympy_round(number, ndigits=None):
    from torch.utils._sympy.functions import RoundDecimal, RoundToInt

    if ndigits is None:
        return RoundToInt(number)
    else:
        return RoundDecimal(number, ndigits)


def _sympy_sym_float(a):
    from torch.utils._sympy.functions import ToFloat

    # NB: Cannot use a * 1.0 here, because 0 * 1.0 is 0 which incorrectly
    # reports that it is an integer
    return ToFloat(a)


def _sympy_is_integer(a):
    import sympy

    from torch.utils._sympy.functions import ToFloat

    return sympy.Eq(ToFloat(sympy.floor(a)), a)


magic_methods = {
    **reflectable_magic_methods,
    "sym_not": operator.invert,
    "pos": operator.pos,
    "eq": _sympy_eq,
    "ne": _sympy_ne,
    "gt": _sympy_gt,
    "lt": _sympy_lt,
    "le": _sympy_le,
    "ge": _sympy_ge,
    "floor": _sympy_floor,
    "trunc": _sympy_trunc,
    "sym_float": _sympy_sym_float,
    "ceil": _sympy_ceil,
    "neg": operator.neg,
    "sym_min": _sympy_min,
    "sym_max": _sympy_max,
    "sym_ite": _sympy_ite,
    "abs": _sympy_abs,
    "round": _sympy_round,
    "is_integer": _sympy_is_integer,
}


for name in math_op_names:
    sym_name = f"sym_{name}"
    magic_methods[sym_name] = getattr(current_module, f"_sympy_{name}")

del name, sym_name, math_op_names, current_module  # type: ignore[possibly-undefined]


def sympy_is_contiguous(sizes, strides):
    dim = len(sizes)
    return sympy_is_contiguous_generic(sizes, strides, list(range(dim - 1, -1, -1)))


def sympy_is_contiguous_generic(sizes, strides, dim_order):
    import sympy

    dim = len(sizes)

    if len(dim_order) != dim:
        return sympy.false

    is_contiguous = sympy.true
    z = sympy.S.One
    # Contiguous if the strides make sense (or the dim is size 1)
    for d in dim_order:
        is_contiguous &= sympy.Eq(sizes[d], sympy.S.One) | sympy.Eq(strides[d], z)
        z *= sizes[d]
    # OR if any size is zero
    for d in range(dim):
        is_contiguous |= sympy.Eq(sizes[d], sympy.S.Zero)
    return is_contiguous


# NB: There is a TODO in C++ to allow omitting the batch dim.  If that
# happens you will need to refactor this


def sympy_is_channels_last_contiguous_2d(sizes, strides):
    return sympy_is_contiguous_generic(sizes, strides, [1, 3, 2, 0])


def sympy_is_channels_last_contiguous_3d(sizes, strides):
    return sympy_is_contiguous_generic(sizes, strides, [1, 4, 3, 2, 0])


def sympy_is_channels_last_strides_generic(sizes, strides, dim_order):
    import sympy

    from torch.utils._sympy.functions import Max

    dim = len(sizes)

    if dim != len(dim_order):
        return sympy.false

    m = sympy.S.Zero
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
        m = strides[d] * Max(sizes[d], 1)

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
        from torch.fx.experimental.proxy_tensor import (
            get_proxy_mode,
            handle_sym_dispatch,
        )

        op = method_to_operator(method)

        out_hint = None
        if self.hint is not None and other.hint is not None:
            out_hint = op(self.hint, other.hint)

        alternate_impl = alternate_impl_if_hinted_methods.get(method)
        if alternate_impl and out_hint is not None:
            return to_node(self, alternate_impl(wrap_node(self), wrap_node(other)))

        if get_proxy_mode():
            return to_node(
                self, handle_sym_dispatch(op, (wrap_node(self), wrap_node(other)), {})
            )
        assert isinstance(other, SymNode)
        try:
            if method == "mod":
                from torch.utils._sympy.functions import Mod, PythonMod

                # Special handling for mod that requires access to the value
                # ranges
                shape_env = self.shape_env
                if (
                    self.expr.is_nonnegative
                    or shape_env.bound_sympy(self.expr).lower >= 0
                ) and (
                    other.expr.is_nonnegative
                    or shape_env.bound_sympy(other.expr).lower >= 0
                ):
                    out = Mod(self.expr, other.expr)
                else:
                    out = PythonMod(self.expr, other.expr)
            else:
                # TODO: consider constant prop here
                out = func(self.expr, other.expr)
        except Exception:
            log.warning("failed to eval %s(%s, %s)", method, self.expr, other.expr)
            raise
        sym_node_log.debug("%s %s %s -> %s", method, self.expr, other.expr, out)
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
        fx_node, _ = self.shape_env._create_fx_call_function(
            op, (self.fx_node, other.fx_node)
        )
        return SymNode(out, self.shape_env, pytype, out_hint, fx_node=fx_node)

    def unary_magic_impl(self):
        from torch.fx.experimental.proxy_tensor import (
            get_proxy_mode,
            handle_sym_dispatch,
        )

        op = method_to_operator(method)
        if get_proxy_mode():
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
        sym_node_log.debug("%s %s -> %s", func, expr, out)
        out_hint = None
        if self.hint is not None:
            out_hint = op(self.hint)
        pytype: Type
        if method in always_int_magic_methods:
            pytype = int
        elif method in always_bool_magic_methods:
            pytype = bool
        elif method in always_float_magic_methods:
            pytype = float
        else:
            pytype = self.pytype

        fx_node, _ = self.shape_env._create_fx_call_function(op, (self.fx_node,))
        return SymNode(out, self.shape_env, pytype, out_hint, fx_node=fx_node)

    if method in unary_methods:
        setattr(SymNode, f"_{method_attr}", unary_magic_impl)
    elif method == "sym_ite":

        def sym_ite_impl(pred_node, then_node, else_node):
            from torch.fx.experimental.proxy_tensor import (
                get_proxy_mode,
                handle_sym_dispatch,
            )

            out_hint = then_node.hint if pred_node.hint else else_node.hint
            if get_proxy_mode():
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

            fx_node, _ = pred_node.shape_env._create_fx_call_function(
                sym_ite, (pred_node.fx_node, then_node.fx_node, else_node.fx_node)
            )
            return SymNode(
                out, pred_node.shape_env, then_node.pytype, out_hint, fx_node=fx_node
            )

        setattr(SymNode, f"_{method_attr}", sym_ite_impl)
    elif method == "round":

        def round_impl(self, ndigits=None):
            from torch.fx.experimental.proxy_tensor import (
                get_proxy_mode,
                handle_sym_dispatch,
            )

            op = builtins.round
            if get_proxy_mode():
                return to_node(
                    self, handle_sym_dispatch(op, (wrap_node(self), ndigits), {})
                )

            expr = self.expr
            try:
                out = func(expr, ndigits)
            except Exception:
                log.warning("failed to eval %s(%s, ndigits=%s)", method, expr, ndigits)
                raise

            if ndigits is None:
                pytype = int
            else:
                pytype = self.pytype

            out_hint = None
            if self.hint is not None:
                out_hint = op(self.hint, ndigits)

            # Internally, None is used as sentinel to indicate that a something is not a node on an FX graph. At the
            # same time, there is no way to wrap a plain None into an FX node. Thus, there is no way to pass None here
            # without triggering some asserts that check whether we are mixing FX nodes with untracked arguments. The
            # hack down below works, because all round function down the line all take ndigits=None as default in their
            # signature.
            # TODO: Remove the args construction below if a different sentinel is used by FX.
            # ezyang(May 2024): LOL
            args = [self.fx_node]
            if ndigits is not None:
                args.append(ndigits)
            fx_node, _ = self.shape_env._create_fx_call_function(op, tuple(args))
            return SymNode(out, self.shape_env, pytype, out_hint, fx_node=fx_node)

        setattr(SymNode, f"_{method_attr}", round_impl)
    else:
        setattr(SymNode, f"_{method_attr}", binary_magic_impl)


def _make_node_sizes_strides(method, func):
    # NB: don't LRU cache, lots of arguments

    def sizes_strides_impl(self, sizes, strides):
        from torch.fx.experimental.proxy_tensor import (
            get_proxy_mode,
            handle_sym_dispatch,
        )

        op = getattr(sys.modules[__name__], method)
        if get_proxy_mode():
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
        method_attr = f"sym_{method}"
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

    # Promotion rules for binary operations.  NB: we preserve PYTHON semantics
    #   - if args are same type, do nothing
    #   - if one arg is float, promote other arg to float
    #       - nb: this applies to floordiv, even though output is integral
    #       (it's still float)
    #   - pow is funny business
    #       - if both ints
    #       - trigger a guard on exponent >= 0
    #           - if non-negative, output is int
    #           - otherwise, output is float
    #   - otherwise, promote other arg to float
    #       - nb: complex is impossible to handle correctly lol, with
    #       negative base and integral float need to diverge semantics and
    #       just always return complex.  Neener neener pretend this problem
    #       doesn't exist
    #   - equality is pain: Python does the fancy thing where it unpacks the
    #     mantissa from the float and then compares that against the int.
    #     Which means it is able to tell that
    #     9007199254740993 != 9007199254740992. (rather than if the LHS was
    #     promoted to float, in which case it would have truncated to the RHS
    #     and subsequently been equal).  We'll model this exactly by having
    #     special mixed type equality operations.  Unfortunately, we need to
    #     do this for all comparison operations (maybe I'll only implement
    #     compare)
    #   - sym_ite mumble mumble really shouldn't allow mixed but whatever

    if method in bool_becomes_int_magic_methods:

        def promote(x):
            """Implements True+True=2, which works in python but not sympy"""
            if isinstance(x, SymBool):
                return SymInt(x.node.wrap_int(int(x)))
            return x

    else:

        def promote(x):
            return x

    def promote2(self, other):
        # TODO: Remove eq and other relations from this list.
        # CPython has fancy implementations for these to get as much precision
        # as possible instead of just promoting to float64 and praying, so we
        # need to handle them specially too.
        # Also, note that int_truediv doesn't go through this path: both
        # arguments are "int" so there isn't any promotion
        if method not in [
            "add",
            "sub",
            "mul",
            "mod",
            "float_pow",
            "float_truediv",
            "int_floordiv",
            "sym_min",
            "sym_max",
            # TODO: remove these
            "eq",
            "ne",
            "gt",
            "lt",
            "le",
            "ge",
        ]:
            return self, other
        f_self = isinstance(self, (float, torch.SymFloat))
        f_other = isinstance(other, (float, torch.SymFloat))
        if f_self or f_other:
            if not f_self:
                self = torch.sym_float(self)
            if not f_other:
                other = torch.sym_float(other)
        return self, other

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
        if not isinstance(other, (int, float, bool, SymInt, SymFloat, SymBool)):
            return NotImplemented
        sym_node_log.debug("MAGIC %s %s %s", method, self, other)
        self = promote(self)
        other = promote(other)
        self, other = promote2(self, other)
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
        if not isinstance(other, (int, float, bool, SymInt, SymFloat, SymBool)):
            return NotImplemented
        self = promote(self)
        other = promote(other)
        self, other = promote2(self, other)
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
    elif method in unary_nonmagic_methods:
        orig = getattr(user_type, method)
        setattr(user_type, method, update_wrapper(unary_magic_impl, orig))
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
    elif method == "round":

        def round_magic_impl(self, ndigits=None):
            if is_constant(self):
                return builtins.round(get_constant(self), ndigits)

            return wrap_node(getattr(self.node, method)(ndigits))

        setattr(user_type, f"__{method}__", round_magic_impl)
    else:
        setattr(user_type, f"__{method}__", binary_magic_impl)
        if method in reflectable_magic_methods:
            setattr(user_type, f"__r{method}__", rbinary_magic_impl)


for method, func in magic_methods.items():  # type: ignore[assignment]
    if method in only_bool_magic_methods:
        _make_user_magic(method, SymBool)
        continue
    if method in only_float_magic_methods:
        _make_user_magic(method, SymFloat)
        continue
    if method in also_bool_magic_methods or method in bool_becomes_int_magic_methods:
        _make_user_magic(method, SymBool)
    _make_user_magic(method, SymInt)
    _make_user_magic(method, SymFloat)

del method
del func
