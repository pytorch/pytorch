from __future__ import annotations


"""
``torch.fx.experimental.symbolic_shapes`` provides interfaces for interacting with
our symbolic shapes reasoning system that is used heavily in torch.compile.  Although
this is not generally considered public API, when writing framework code in PyTorch
as well as extensions to PyTorch (e.g., in custom operator implementations), you may
need to make use of these APIs to setup dynamic shapes support appropriately.
"""

import abc
import atexit
import collections
import functools
import inspect
import itertools
import logging
import math
import operator
import os
import re
import sys
import threading
import traceback
from collections import defaultdict
from contextlib import _GeneratorContextManager, contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Callable,
    cast,
    Counter,
    DefaultDict,
    Dict,
    Iterator,
    List,
    Mapping,
    NoReturn,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    TYPE_CHECKING,
    TypeVar,
    Union,
)
from typing_extensions import deprecated, TypeAlias, TypeGuard

import torch
import torch.fx
import torch.fx.traceback as fx_traceback
import torch.utils._pytree as pytree

# NB: The sym_* functions are used via getattr() and must be imported here.
from torch import SymBool, SymFloat, SymInt
from torch._guards import ShapeGuard, SLoc, Source, TracingContext
from torch._logging import dtrace_structured, LazyString, structured, trace_structured
from torch._subclasses.meta_utils import is_sparse_any
from torch._utils_internal import signpost_event
from torch.fx.experimental import _config as config
from torch.fx.experimental.recording import (
    FakeTensorMeta,
    record_shapeenv_event,
    replay_shape_env_events,
    shape_env_check_state_equal,
    ShapeEnvEvent,
)
from torch.fx.experimental.sym_node import SymNode, SymTypes
from torch.utils._ordered_set import OrderedSet
from torch.utils._python_dispatch import is_traceable_wrapper_subclass
from torch.utils._sympy.functions import (
    Application,
    CeilToInt,
    CleanDiv,
    FloorDiv,
    FloorToInt,
    IsNonOverlappingAndDenseIndicator,
    Mod,
    PythonMod,
)
from torch.utils._sympy.numbers import int_oo
from torch.utils._sympy.printers import CppPrinter, PythonPrinter
from torch.utils._sympy.singleton_int import SingletonInt
from torch.utils._sympy.solve import try_solve
from torch.utils._sympy.symbol import make_symbol, symbol_is_type, SymT
from torch.utils._sympy.value_ranges import (
    bound_sympy,
    SymPyValueRangeAnalysis,
    ValueRangeError,
    ValueRanges,
)
from torch.utils._traceback import CapturedTraceback, format_frame


if TYPE_CHECKING:
    import types

    from torch import Tensor
    from torch._subclasses.fake_tensor import FakeTensor
    from torch.types import BoolLikeType


InputList = List
DimList = List

log = logging.getLogger(__name__)

import sympy
from sympy import S


class GuardOnDataDependentSymNode(RuntimeError):
    cond: sympy.Basic

    def __init__(self, cond: sympy.Basic, *args: Any) -> None:
        super().__init__(*args)
        self.cond = cond


class PendingUnbackedSymbolNotFound(RuntimeError):
    pass


aten = torch._ops.ops.aten  # type: ignore[has-type]

__all__ = [
    "has_symbolic_sizes_strides",
    "create_contiguous",
    "ShapeEnv",
    "is_concrete_int",
    "is_concrete_float",
    "guard_int",
    "guard_float",
    "guard_scalar",
    "canonicalize_bool_expr",
    "hint_int",
    "SYMPY_INTERP",
    "free_symbols",
    "is_symbol_binding_fx_node",
    "is_concrete_bool",
    "is_nested_int",
    "SHAPEENV_EVENT_KEY",
    "CURRENT_NODE_KEY",
    "has_free_symbols",
    "has_free_unbacked_symbols",
    "sym_eq",
    "SymbolicContext",
    "StatelessSymbolicContext",
    "StatefulSymbolicContext",
    "SubclassSymbolicContext",
    "statically_known_true",
    "guard_size_oblivious",
    "check_consistent",
    "compute_unbacked_bindings",
    "ConvertIntKey",
    "rebind_unbacked",
    "resolve_unbacked_bindings",
    "is_accessor_node",
    "ValueRangesSLoc",
    "SymIntEqByExpr",
]

# FX node metadata keys for symbolic shape FX graph.
SHAPEENV_EVENT_KEY = "shapeenv_event"
CURRENT_NODE_KEY = "current_node"


def log_lru_cache_stats(wrapped_f: functools._lru_cache_wrapper[object]) -> None:
    log.debug(
        "lru_cache_stats %s: %s", wrapped_f.__name__, wrapped_f.cumulative_cache_info()  # type: ignore[attr-defined]
    )


# Note about Sympy Expr/SympyBoolean/Basic typing: the Sympy hierarchy is
#
#   Basic
#       Expr
#       SympyBoolean
#           Relational
#
# Notably, Expr and SympyBoolean are not related.  So use Basic when the
# expression could denote int, float OR bool, and otherwise use the more
# specific Expr for int/float and SympyBoolean for bool.
#
# In obscure Meta only situations, sympy.logic.boolalg doesn't exist at runtime.
# So make sure only type checker evaluates this alias.
# Xref: https://www.internalfb.com/diff/D53324783
SympyBoolean: TypeAlias = "sympy.logic.boolalg.Boolean"


_T = TypeVar("_T")
_SympyT = TypeVar("_SympyT", sympy.Expr, SympyBoolean, sympy.Basic)


class SymIntEqByExpr:
    """
    This is a wrapper around SymInt which has alternative semantics for
    equality.  Specifically, instead of erroring or guarding, we
    instead will hash/compare equality based on the underlying sympy
    expression; e.g., s0 and s1 will always compare as False.

    NB: This does NOT do fancy analysis that maybe_evaluate_static does;
    we can only reason through equalities that occur because to expressions
    canonicalize to the same expression via regular simplification.
    """

    val: Union[torch.SymInt, int]

    def __init__(self, val: Union[torch.SymInt, int]) -> None:
        self.val = val

    def __repr__(self) -> str:
        return repr(self.val)

    def _extract(self) -> sympy.Expr:
        if isinstance(self.val, torch.SymInt):
            return self.val.node.expr
        else:
            return sympy.Integer(self.val)

    def __eq__(self, other: object) -> bool:
        assert isinstance(other, SymIntEqByExpr)

        # int equality fastpath
        if type(self.val) is int and type(other.val) is int:
            return self.val == other.val

        return self._extract() == other._extract()

    def __hash__(self) -> int:
        return hash(self._extract())


def _nested_int_aware_sort(
    tup: Tuple[Union[SymInt, int], int]
) -> Tuple[int, Union[SymInt, int], int]:
    return (
        # Order nested ints by their coefficients.
        # 1 here to order nested ints after non-nested-ints.
        (1, tup[0].node.nested_int_coeff(), tup[1])
        if is_nested_int(tup[0])
        else (0, *tup)
    )


# Wrapper on lru_cache that reports statistics at process end
def lru_cache(
    maxsize: Optional[int],
) -> Callable[[Callable[..., _T]], functools._lru_cache_wrapper[_T]]:
    def inner(f: Callable[..., _T]) -> functools._lru_cache_wrapper[_T]:
        wrapped_f = functools.lru_cache(maxsize)(f)
        old_cache_clear = wrapped_f.cache_clear
        prev_hits = 0
        prev_misses = 0

        # TODO: There's a ref-cycle here (wrapped_f -> cumulative_cache_info
        # -> wrapped_f) but cannot be solved with weakref as wrapped_f is not
        # weakref'able on some versions of Python

        def cumulative_cache_info() -> functools._CacheInfo:
            cur = wrapped_f.cache_info()
            return functools._CacheInfo(
                prev_hits + cur.hits,
                prev_misses + cur.misses,
                cur.maxsize,
                cur.currsize,
            )

        def new_cache_clear() -> None:
            nonlocal prev_hits, prev_misses
            cur = wrapped_f.cache_info()
            prev_hits += cur.hits
            prev_misses += cur.misses
            old_cache_clear()

        wrapped_f.cache_clear = new_cache_clear  # type: ignore[attr-defined, method-assign]
        wrapped_f.cumulative_cache_info = cumulative_cache_info  # type: ignore[attr-defined, method-assign]
        if log.isEnabledFor(logging.DEBUG):
            atexit.register(log_lru_cache_stats, wrapped_f)  # type: ignore[arg-type]
        return wrapped_f

    return inner


# These are modules that contain generic code for interacting with ShapeEnv
# which are unlikely to identify a particular interesting guard statement
@lru_cache(None)
def uninteresting_files() -> Set[str]:
    import torch._dynamo.eval_frame
    import torch._inductor.sizevars
    import torch._library.custom_ops
    import torch._library.fake_impl
    import torch._subclasses.fake_tensor
    import torch._subclasses.meta_utils

    mods = [
        sys.modules[__name__],
        torch.fx.experimental.recording,
        torch.fx.experimental.sym_node,
        torch.fx.interpreter,
        torch,
        torch._dynamo.eval_frame,
        torch._inductor.sizevars,
        torch._library.custom_ops,
        torch._library.fake_impl,
        torch._subclasses.meta_utils,
        torch._subclasses.fake_tensor,
    ]
    import torch._dynamo.guards

    return (
        {inspect.getfile(m) for m in mods}
        | torch._dynamo.guards.uninteresting_files()
        | {"<string>"}
    )


class ConstraintViolationError(RuntimeError):
    pass


def has_symbolic_sizes_strides(elem: torch.Tensor) -> bool:
    return elem._has_symbolic_sizes_strides


Int: TypeAlias = Union[torch.SymInt, int]


def create_contiguous(shape: Sequence[Int]) -> List[Int]:
    strides: List[Int] = [1]
    for dim in reversed(shape[:-1]):
        strides.append(dim * strides[-1])  # type: ignore[operator]
    return list(reversed(strides))


def hint_int(a: Union[torch.SymInt, int], fallback: Optional[int] = None) -> int:
    """
    Retrieve the hint for an int (based on the underlying real values as observed
    at runtime).  If no hint is available (e.g., because data dependent shapes),
    if fallback is not None, use that instead (otherwise raise an error).
    """
    if isinstance(a, torch.SymInt):
        return a.node.require_hint(fallback)
    assert type(a) is int, a
    return a


Scalar: TypeAlias = Union[torch.SymInt, torch.SymFloat, torch.SymBool, int, float, bool]


def has_hint(a: Scalar) -> bool:
    if isinstance(a, SymTypes):
        return a.node.has_hint()
    return True


def is_concrete_int(a: Union[int, SymInt]) -> bool:
    """
    Utility to check if underlying object
    in SymInt is concrete value. Also returns
    true if integer is passed in.

    Args:
        a (SymInt or int): Object to test if it int
    """
    assert isinstance(a, (SymInt, int))

    if isinstance(a, int):
        return True

    if isinstance(a.node.expr, sympy.core.numbers.Integer):
        return True

    return False


def is_concrete_float(a: Union[float, SymFloat]) -> bool:
    r"""Utility to check if underlying object
    in SymInt is concrete value. Also returns
    true if integer is passed in.

    Args:
        a (SymInt or float): Object to test if it float
    """
    assert isinstance(a, (SymFloat, float))

    if isinstance(a, float):
        return True

    if isinstance(a.node.expr, sympy.core.numbers.Float):
        return True

    return False


def guard_size_oblivious(expr: Union[torch.SymBool, bool]) -> bool:
    """
    Perform a guard on a symbolic boolean expression in a size oblivious way.
    This is typically used when a non-oblivious test would result in a guard
    on a data dependent value of which we don't know the value of at compile time.
    When a guard is tested this way, we may diverge in behavior from how regular
    PyTorch semantics would treat it.  For more information, see
    https://github.com/pytorch/pytorch/pull/118579
    """
    if isinstance(expr, torch.SymBool):
        return expr.node.guard_size_oblivious("", 0)
    else:
        assert isinstance(expr, bool), expr
        return expr


def _guard_sizes_oblivious(
    lhs_sizes: Sequence[Union[torch.SymInt, bool]],
    rhs_sizes: Sequence[Union[torch.SymInt, bool]],
) -> bool:
    """
    Leverage guard_size_oblivious to compare if two lists of int/symint are equal.
    Useful to compare sizes, strides etc.
    """

    return len(lhs_sizes) == len(rhs_sizes) and all(
        guard_size_oblivious(lhs_item == rhs_item)
        for lhs_item, rhs_item in zip(lhs_sizes, rhs_sizes)
    )


def check_consistent(new: _T, old: _T) -> None:
    """
    Test that two "meta" values (typically either Tensor or SymInt) have
    the same values, e.g., after retracing.  If we don't understand the
    quantities in question, we'll just skip the consistency check.
    """
    # TODO: do boolean equality test too, see
    # https://github.com/pytorch/pytorch/issues/124110
    scalar_types = (torch.SymInt, torch.SymFloat, int, float)

    if isinstance(new, torch.Tensor):
        assert isinstance(old, torch.Tensor)
        torch._check(
            old.dim() == new.dim(), lambda: f"{old.shape} != {new.shape} (old != new)"
        )
        # Do this manually so that each individual test is irrefutable
        # (TODO: should be a helper for this, maybe sym_eq?  That
        # gives us a compound expression and I'm not sure it
        # simplifies right now)
        for i, j in zip(old.shape, new.shape):
            torch._check(i == j, lambda: f"{old.shape} != {new.shape} (old != new)")
    # NB: bool is subclass of int
    elif isinstance(new, scalar_types) and not isinstance(new, bool):
        assert isinstance(old, scalar_types) and not isinstance(
            old, bool
        ), f"{old} != {new}"
        torch._check(old == new, lambda: f"{old} != {new} (old != new)")


def resolve_unbacked_bindings(
    shape_env: Optional[ShapeEnv],
    bindings: Optional[Dict[sympy.Symbol, pytree.KeyPath]],
) -> Optional[Dict[sympy.Symbol, pytree.KeyPath]]:
    if bindings is None:
        return None
    assert shape_env is not None
    return {shape_env.unbacked_renamings.get(k, k): v for k, v in bindings.items()}


Result: TypeAlias = Union[torch.Tensor, Tuple[torch.Tensor, ...]]


def rebind_unbacked(
    shape_env: Optional[ShapeEnv], n: torch.fx.Node, result: Result
) -> None:
    """
    Suppose we are retracing a pre-existing FX graph that previously had
    fake tensor propagation (and therefore unbacked SymInts).  When we retrace,
    we re-propagate fake tensors, which results in new unbacked SymInts.
    When this happens, we need to tell the shape environment about the equivalence
    of the old and new unbacked SymInts.  Pass us the old torch.fx.Node (which
    has the old binding information) and the new result (which we can extract the
    new unbacked SymInts out from).
    """

    # Inputs never need rebinding
    if n.op == "placeholder":
        return

    if bindings := resolve_unbacked_bindings(
        shape_env, n.meta.get("unbacked_bindings")
    ):
        assert shape_env is not None
        for raw_u0, path in bindings.items():
            u1 = pytree.key_get(result, path)
            # Sometimes, things were previously unbacked bindings become constants.
            # There are two situations this can happen.
            #
            # First, you might have a runtime assert that causes the
            # constant-ification.  In this case, the /binding/ itself will
            # still be an unbacked symbol (because we will only force it
            # to be a constant later in fake tensor propagation).  In this
            # case, u1 is a SymInt and we still do all our work as normal.
            #
            # But second, it might be that fake tensor propagation DIRECTLY
            # converted the unbacked SymInt into a constant.  This happens
            # more rarely, but we have identified two situations it can
            # validly occur:
            #
            # - If you have a tensor_version operator, these are initially
            #   allocated as unbacked SymInts, but after AOTAutograd they
            #   get forced specialized to specific values.  In this case,
            #   there is no reason to do runtime asserts on them, this is
            #   just a hack to properly keep track of them to start.
            #
            # - If you have an item() call on a constant tensor, the result
            #   of the item() call is constant and we do not need runtime
            #   asserts on this symbol.  In
            #   https://github.com/pytorch/pytorch/issues/140625 we have a
            #   case where in the initial trace of the program we are unable
            #   to determine that torch.tensor is constant, but then
            #   subsequent passes cause torch.tensor to become a constant and
            #   then the unbacked symbol goes poof.
            #
            # In all of these cases, it is no longer necessary to generate
            # deferred runtime asserts, since other subsystems (e.g., the
            # constant-ification pass) ensure that the quantity is now truly
            # static and cannot change at runtime.  So it's OK to discard
            # in these situations.
            #
            # There is one more hazard (re
            # https://github.com/pytorch/pytorch/issues/141248), the problem
            # is that you can end up with "dangling" unbacked symbols that
            # exist in the ShapeEnv but are never bound anywhere.  You might
            # like an invariant that unbacked symbols never get lost.  But
            # we do not have this invariant, so do not try to enforce it.
            if isinstance(u1, int):
                log.info(
                    "rebind_unbacked: discard %s %s %s -> %s",
                    n.target,
                    raw_u0,
                    path,
                    u1,
                )
                continue

            # We only care about rebinding unbacked things
            if u1.node.hint is not None:
                continue

            raw_u1 = u1.node.expr
            # Simplify SymBool binding
            if (
                isinstance(raw_u1, sympy.Piecewise)
                and len(raw_u1.args) == 2
                and (
                    raw_u1_args0 := cast(
                        Tuple[sympy.Basic, sympy.Basic], raw_u1.args[0]
                    )
                )
                and raw_u1_args0[0] == 1
                and isinstance(eq := raw_u1_args0[1], sympy.Eq)
                and isinstance(new_raw_u1 := eq.lhs, sympy.Symbol)
                and shape_env.var_to_range[new_raw_u1].issubset(ValueRanges(0, 1))
                and eq.rhs == 1
                and cast(Tuple[sympy.Basic, sympy.Basic], raw_u1.args[1]) == (0, True)
            ):
                # This is what the pattern match above is testing
                repacked = _sympy_cast_symbool_to_symint_guardless(
                    sympy.Eq(new_raw_u1, 1)
                )
                assert repacked == raw_u1, f"{repacked} != {raw_u1}"
                # Cancel the to_int(to_bool(x)). This is sound because x in
                # [0, 1]
                raw_u1 = new_raw_u1

            if not isinstance(raw_u1, sympy.Symbol):
                assert (
                    not raw_u1.free_symbols
                ), f"should have been constant, but got {raw_u1}"
                continue

            # The old and new could be the same if you improperly hit the memo
            # while retracing.  Make sure you updated FakeTensorMode.epoch
            assert raw_u0 != raw_u1, f"{raw_u0} possible memo disaster"
            # Reuse the OLD symbol name
            shape_env._rename_unbacked_to(raw_u1, raw_u0)


# NB: You could try to expand this to cover more cases by simply
# detecting whenever you have an int output, but this is a bit
# dangerous in case someone adds a function that returns an int but is
# mutating.  So manually whitelist for now.
def is_accessor_node(node: torch.fx.Node) -> bool:
    # Dynamo only exercised condition
    if (
        node.op == "call_method"
        and isinstance(node.args[0], torch.fx.Node)
        and isinstance(node.args[0].meta.get("example_value"), torch.Tensor)
        and node.target in ["size", "stride", "storage_offset", "item"]
    ):
        return True
    if node.op == "call_function" and node.target in [
        torch.ops.aten.sym_size,
        torch.ops.aten.sym_size.default,
        torch.ops.aten.sym_size.int,
        torch.ops.aten.sym_stride,
        torch.ops.aten.sym_stride.default,
        torch.ops.aten.sym_stride.int,
        torch.ops.aten.sym_storage_offset,
        torch.ops.aten.sym_storage_offset.default,
        torch.ops.aten.sym_numel.default,
    ]:
        return True
    return False


def canonicalize_bool_expr(expr: _T) -> _T:
    """
    Canonicalize a boolean expression by transforming it into a lt / le
    inequality and moving all the non-constant terms to the rhs.
    We canonicalize And / Ors / Not via cnf and then canonicalize their subexpr
    recursively
    nb. sympy.Rel.canonical is not good enough https://github.com/sympy/sympy/issues/25924

    Args:
        expr (sympy.Expr): Expression to canonicalize
    """
    # Canonicalise an inequality by transforming it into a lt / le
    # inequality and moving all the non-constant terms to the rhs
    # We canonicalise And / Ors / Not via cnf
    # nb. Relational.canonical in sympy is broken
    # https://github.com/sympy/sympy/issues/25924

    if not isinstance(
        expr, (sympy.Rel, sympy.And, sympy.Or, sympy.Not, sympy.Eq, sympy.Ne)
    ):
        return expr

    if isinstance(expr, (sympy.And, sympy.Or, sympy.Not)):
        expr = sympy.logic.boolalg.to_cnf(expr)
    return _canonicalize_bool_expr_impl(expr)  # type: ignore[arg-type, return-value]


def _sympy_from_args(
    cls: Union[Type[sympy.Add], Type[sympy.Mul]],
    args: List[sympy.Expr],
    sort: bool = True,
    is_commutative: Optional[bool] = None,
) -> sympy.Expr:
    if not args:
        return cls.identity  # type: ignore[union-attr]
    # These args are already in canonical form, so we avoid calling
    # Add(*args) to avoid expensive Add.flatten operation
    if sort:
        if cls is sympy.Add:
            sort_fn = sympy.core.add._addsort
        elif cls is sympy.Mul:
            sort_fn = sympy.core.mul._mulsort
        else:
            raise ValueError(f"Unknown cls: {cls}")

        # we don't support non commutative with sort
        assert is_commutative is True
        if args[0].is_Number:
            rest = args[1:]
            sort_fn(rest)
            return cls._from_args([args[0]] + rest, is_commutative=is_commutative)  # type: ignore[attr-defined]
        else:
            args = args.copy()
            sort_fn(args)
            return cls._from_args(args, is_commutative=is_commutative)  # type: ignore[attr-defined]
    else:
        # if the args are already sorted, we create directly
        return cls._from_args(args, is_commutative=is_commutative)  # type: ignore[attr-defined]


def _canonicalize_bool_expr_impl(expr: SympyBoolean) -> SympyBoolean:
    """
    After canonicalization, we are guaranteed to have eliminated Ge/Gt relations
    (rewriting them to Le/Lt, respectively).
    """
    if isinstance(expr, (sympy.And, sympy.Or)):
        return type(expr)(*map(canonicalize_bool_expr, expr.args))

    opposite = {sympy.Gt: sympy.Lt, sympy.Ge: sympy.Le}
    t: Union[Type[Any]]
    if isinstance(expr, tuple(opposite.keys())):
        rhs = expr.lhs - expr.rhs  # type: ignore[attr-defined]
        t = opposite[type(expr)]  # type: ignore[index]
    else:
        assert isinstance(expr, (sympy.Lt, sympy.Le, sympy.Eq, sympy.Ne))
        rhs = expr.rhs - expr.lhs
        t = type(expr)

    def is_neg(t: sympy.Expr) -> bool:
        return (t.is_Number and t.is_negative) or (
            isinstance(t, sympy.Mul) and t.args[0].is_Number and t.args[0].is_negative
        )

    lhs = S.Zero
    rhs = _reduce_to_lowest_terms(rhs)
    if isinstance(rhs, sympy.Add):
        pos = []
        neg = []
        for term in rhs.args:
            if is_neg(term):
                neg.append(-term)
            else:
                pos.append(term)
        # these are already sorted
        rhs = _sympy_from_args(sympy.Add, pos, sort=False, is_commutative=True)
        # the terms were changed, so needs a sorting
        lhs = _sympy_from_args(sympy.Add, neg, sort=True, is_commutative=True)
    elif is_neg(rhs):
        # lhs == 0
        lhs, rhs = -rhs, S.Zero
    # We don't have to evaluate here because lhs, rhs came from a Boolean
    # and it was already simplified
    return t(lhs, rhs, evaluate=False)


def _reduce_to_lowest_terms(expr: sympy.Expr) -> sympy.Expr:
    """
    Eliminates any integer factor from a given expression.
    E.g., 6x + 4y reduces to 3x + 2y.

    Useful when an expression is == or != to 0.
    """

    def integer_coefficient(x: sympy.Expr) -> int:
        if x.is_Integer:
            return abs(int(x))
        elif x.is_Mul:
            # If one of the args of a Mul is an Integer, it is the
            # first arg. eg: args(2*x*3*y) == (6, x, y)
            return abs(int(x.args[0])) if x.args[0].is_Integer else 1  # type: ignore[call-overload]
        else:
            return 1

    def div_by_factor(x: sympy.Expr, factor: int) -> sympy.Expr:
        if x.is_Integer:
            return x / factor
        elif x.is_Mul:
            if x.args[0] != factor:
                args = [x.args[0] / sympy.Integer(factor), *x.args[1:]]
            else:
                # Mul._from_args require a canonical list of args
                # so we remove the first arg (x.args[0] / factor) if it was 1
                args = list(x.args[1:])
            return _sympy_from_args(sympy.Mul, args, is_commutative=x.is_commutative)
        else:
            raise AssertionError(f"illegal arg to div_by_factor: {x}")

    if expr.is_Add:
        atoms = cast(Sequence[sympy.Expr], expr.args)
        factor = functools.reduce(math.gcd, map(integer_coefficient, atoms))
        if factor == 1:
            return expr
        atoms = [div_by_factor(x, factor) for x in atoms]
        return _sympy_from_args(
            sympy.Add, atoms, sort=True, is_commutative=expr.is_commutative
        )
    elif expr.is_Integer:
        return S.One
    elif expr.is_Mul:
        return div_by_factor(expr, integer_coefficient(expr))
    return expr


def is_concrete_bool(a: Union[bool, SymBool]) -> bool:
    """
    Utility to check if underlying object
    in SymBool is concrete value. Also returns
    true if integer is passed in.

    Args:
        a (SymBool or bool): Object to test if it bool
    """
    assert isinstance(a, (SymBool, bool))

    if isinstance(a, bool):
        return True

    if isinstance(
        a.node.expr, (sympy.logic.boolalg.BooleanTrue, sympy.logic.boolalg.BooleanFalse)
    ):
        return True

    return False


def is_nested_int(s: Union[int, SymInt]) -> TypeGuard[SymInt]:
    return isinstance(s, torch.SymInt) and s.node.is_nested_int()


IterateExprsAtom: TypeAlias = Union[
    SymInt, SymFloat, SymBool, int, float, bool, sympy.Basic, torch.Tensor
]
IterateExprs: TypeAlias = Union[IterateExprsAtom, Sequence[IterateExprsAtom]]


def _iterate_exprs(val: IterateExprs) -> Iterator[sympy.Basic]:
    if isinstance(val, SymTypes):
        # This allow applies to the jagged layout NestedTensor case as
        # nested ints are not symbolic
        if is_symbolic(val):
            yield val.node.expr
    elif isinstance(val, sympy.Basic):
        yield val
    elif isinstance(val, (int, float, bool)):
        pass
    elif isinstance(val, (tuple, list)):
        for s in val:
            yield from _iterate_exprs(s)
    elif is_sparse_any(val):
        yield from _iterate_exprs(val.size())
    elif isinstance(val, torch.Tensor):
        yield from _iterate_exprs(val.size())
        yield from _iterate_exprs(val.stride())
        yield from _iterate_exprs(val.storage_offset())
    elif val is None:
        pass
    else:
        raise AssertionError(f"cannot extract sympy expressions from {val} {type(val)}")


def free_symbols(val: IterateExprs) -> OrderedSet[sympy.Symbol]:
    if val is None:
        return OrderedSet()
    itr = _iterate_exprs(val)
    # we need at least 1 to call union, so we hand code the identity
    try:
        first_expr = next(itr)
    except StopIteration:
        return OrderedSet()

    # TODO: Apparently, returning an OrderedSet here breaks
    # python test/distributed/_tensor/test_dtensor_compile.py TestDTensorCompile.test_dtensor_dynamic
    return first_expr.free_symbols.union(*(e.free_symbols for e in itr))  # type: ignore[return-value]


def has_free_symbols(val: IterateExprs) -> bool:
    """Faster version of bool(free_symbols(val))"""
    return not all(e.is_number for e in _iterate_exprs(val))


def has_free_unbacked_symbols(x: IterateExprs) -> bool:
    """Faster version of bool(free_unbacked_symbols(val))"""
    from sympy.core.traversal import iterargs

    for s in _iterate_exprs(x):
        for arg in iterargs(s):
            if arg.is_Symbol and symbol_is_type(
                arg, (SymT.UNBACKED_INT, SymT.UNBACKED_FLOAT)
            ):
                return True
    return False


# Like free_symbols, but filtered to only report unbacked symbols
def free_unbacked_symbols(x: IterateExprs) -> OrderedSet[sympy.Symbol]:
    # NB: keep synced with is_unbacked_symint
    return OrderedSet(
        s
        for s in free_symbols(x)
        if symbol_is_type(s, (SymT.UNBACKED_INT, SymT.UNBACKED_FLOAT))
    )


# WARNING: Don't use this on Dynamo produced graphs, they don't have meta
# setup!
def is_symbol_binding_fx_node(node: torch.fx.Node) -> Optional[sympy.Symbol]:
    if (
        "val" in node.meta
        and isinstance(node.meta["val"], torch.SymInt)
        and isinstance(node.meta["val"].node.expr, sympy.Symbol)
        and (
            node.op == "placeholder"
            or free_unbacked_symbols(node.meta["val"].node.expr)
        )
    ):
        return node.meta["val"].node.expr
    return None


def find_symbol_binding_fx_nodes(
    graph: torch.fx.Graph,
) -> Dict[sympy.Symbol, torch.fx.Node]:
    r = {}
    # NB: Prefer first occurrence of symbol
    for node in graph.nodes:
        if (s := is_symbol_binding_fx_node(node)) is not None and s not in r:
            r[s] = node
    return r


# Analogous to ConvertIntSource
@dataclass(frozen=True)
class ConvertIntKey:
    def __str__(self) -> str:
        return ".cast_symbool_to_symint_guardless()"

    def get(self, b: bool) -> Union[int, SymInt]:
        """Get the int value from bool"""
        return cast_symbool_to_symint_guardless(b)


@dataclass(frozen=True)
class CallMethodKey:
    name: str

    def __str__(self) -> str:
        return f".{self.name}()"

    def get(self, o: Any) -> Any:
        """Call the method on object"""
        return getattr(o, self.name)()


@dataclass(frozen=True)
class InnerTensorKey:
    inner_name: str

    def __str__(self) -> str:
        return f".{self.inner_name}"

    def get(self, o: Any) -> Any:
        """Get the inner tensor attribute"""
        return getattr(o, self.inner_name)


@dataclass(frozen=True)
class DivideByKey:
    divisor: int

    def __str__(self) -> str:
        return f".__floordiv__({self.divisor})"

    def get(self, o: int) -> int:
        """Divide object by divisor"""
        return o // self.divisor


def compute_unbacked_bindings(
    shape_env: Optional[ShapeEnv],
    example_value: object,
    old_example_value: Optional[object] = None,
    peek: bool = False,
) -> Optional[Dict[sympy.Symbol, pytree.KeyPath]]:
    """
    After having run fake tensor propagation and producing example_value
    result, traverse example_value looking for freshly bound unbacked
    symbols and record their paths for later.  It is an error if
    we have allocated an unbacked SymInt but it cannot be found in
    example_value.  (NB: this means if you have a multi-output
    function, you must call this on the tuple of tensor output, you
    cannot wait!)

    The peek parameter lets you check out what the bindings are without
    changing the affected list.  This is primarily useful for ensuring
    unbacked_var_to_val is promptly populated when propagate_real_tensors is on.
    """
    if shape_env is None:
        return None

    fs = shape_env.pending_fresh_unbacked_symbols
    pending = set(fs)
    if not pending:
        return None

    if not peek:
        log.info("compute_unbacked_bindings %s", fs)
        fs.clear()

    def free_unbacked_symbols_with_path(
        a: object, path: pytree.KeyPath, real: Optional[object] = None
    ) -> Dict[sympy.Symbol, pytree.KeyPath]:
        assert shape_env is not None
        r = {}
        if isinstance(a, (tuple, list)):
            # NB: real is apparently not always a tuple/list here
            # python test/inductor/test_torchinductor.py CpuTests.test_index_propagation_nested_indirect_indexing_cpu
            for i in range(len(a)):
                r.update(
                    free_unbacked_symbols_with_path(
                        a[i],
                        path + (pytree.SequenceKey(i),),
                        real=real[i] if real is not None else None,  # type: ignore[index]
                    )
                )
        elif is_traceable_wrapper_subclass(a):
            # TODO: Determine if this is correct
            attrs, _ = a.__tensor_flatten__()
            for attr in attrs:
                sub = getattr(a, attr)
                r.update(
                    free_unbacked_symbols_with_path(sub, path + (InnerTensorKey(attr),))
                )
        elif isinstance(a, torch.Tensor):
            from torch._subclasses.fake_tensor import FakeTensor

            assert isinstance(a, FakeTensor)
            r.update(
                free_unbacked_symbols_with_path(
                    a.size(),
                    path + (CallMethodKey("size"),),
                    real=a.real_tensor.size() if a.real_tensor is not None else None,
                )
            )
            r.update(
                free_unbacked_symbols_with_path(
                    a.stride(),
                    path + (CallMethodKey("stride"),),
                    real=a.real_tensor.stride() if a.real_tensor is not None else None,
                )
            )
            r.update(
                free_unbacked_symbols_with_path(
                    a.storage_offset(),
                    path + (CallMethodKey("storage_offset"),),
                    real=a.real_tensor.storage_offset()
                    if a.real_tensor is not None
                    else None,
                )
            )

        # NB: Intentionally access _expr, not expr, do not want
        # simplification!
        elif (
            isinstance(a, (torch.SymInt, torch.SymFloat))
            and isinstance(s := a.node._expr, sympy.Symbol)
            and s in pending
        ):
            r[s] = path
            if real is not None:
                assert isinstance(real, (int, float))
                shape_env.set_unbacked_var_to_val(s, real)
            pending.remove(s)
        # When an unbacked SymInt is perfectly divisible by an integer
        # constant, we replace it with the integer constant to improve
        # reasoning capabilities.  However, in synthetic examples, it is
        # then possible that the factor never is explicitly allocated.
        # Fortunately, we can compute it by division.
        elif (
            isinstance(a, torch.SymInt)
            and isinstance(s := a.node._expr, sympy.Mul)
            and len(s.args) == 2
            and isinstance(lhs := s.args[0], sympy.Integer)
            and isinstance(rhs := s.args[1], sympy.Symbol)
            and rhs in pending
        ):
            # TODO: DivideByKey needs to test divisibility at runtime!
            r[rhs] = path + (DivideByKey(int(lhs)),)
            if real is not None:
                assert isinstance(real, int)
                shape_env.set_unbacked_var_to_val(rhs, real // int(lhs))
            pending.remove(rhs)
        # The annoyance here arises from the fact that SymBool is
        # allocated by allocating a SymInt and then testing if it's equal
        # to one.  So you have a complicated binding site logic for this.
        elif (
            isinstance(a, torch.SymBool)
            and isinstance(s := a.node._expr, sympy.Eq)
            # This must match create_unbacked_symbool EXACTLY
            and isinstance(s.lhs, sympy.Symbol)
            and s.rhs == 1
            and s.lhs in pending
        ):
            r[s.lhs] = path + (ConvertIntKey(),)
            if real is not None:
                assert type(real) is bool
                shape_env.set_unbacked_var_to_val(s, int(real))
            pending.remove(s.lhs)

        return r

    symbol_to_path = free_unbacked_symbols_with_path(example_value, ())
    if not peek and pending:
        extra = (
            repr((example_value.stride(), example_value.storage_offset()))
            if isinstance(example_value, torch.Tensor)
            else ""
        )
        raise PendingUnbackedSymbolNotFound(
            f"Pending unbacked symbols {pending} not in returned outputs {example_value} {extra}.\n"
            "Did you accidentally call new_dynamic_size() or item() more times "
            "than you needed to in your fake implementation?\n"
            "For more help, see https://docs.google.com/document/d/1RWrH-3wLEpzR9kCS6gGBNen_-Fs-8PVbWWFE5AcgeWE/edit"
        )

    # Why do we have to do some rebinding here?  If the original FX node
    # wasn't a binding site because you had a memo hit, but post
    # translation you aren't a memo hit anymore, there's now a new binding
    # site... but we know (because it's the same FX node) that the value
    # is actually the same, they're just not obviously equal anymore.
    #
    # The logic here is written carefully, because unlike the
    # bind_unbacked case, we are not guaranteed to have a symbol for
    # old_sym.  If we have a symbol, do regular rename unbacked to; but if
    # we don't, we need to specially eliminate the fresh unbacked symbol
    # (NB: we are /trusting/ that the memoization is correct, and that we
    # don't need to generate a new runtime assert.  This is load bearing,
    # as repropagation can happen after we've frozen runtime asserts.)
    if old_example_value is not None:
        for keypath in symbol_to_path.values():
            old_sym = pytree.key_get(old_example_value, keypath)
            new_sym = pytree.key_get(example_value, keypath)
            if isinstance(new_sym, SymTypes) and isinstance(
                new_s := new_sym.node.expr, sympy.Symbol
            ):
                if (
                    isinstance(old_sym, SymTypes)
                    and (old_s := old_sym.node.expr) != new_s
                ):
                    if isinstance(old_s, sympy.Symbol):
                        shape_env._rename_unbacked_to(new_s, old_s)
                    else:
                        shape_env._eliminate_unbacked(new_s, old_s)
                elif not isinstance(old_sym, SymTypes):
                    shape_env._eliminate_unbacked(new_s, sympy.sympify(old_sym))

    return symbol_to_path


def definitely_true(a: BoolLikeType) -> bool:
    """
    Returns True only if we can tell that a is True, possibly introducing
    a guard in the process.  If a depends on some unbacked SymInt, we may
    return False even though there may exist a possible value of the SymInt
    that would cause the expression to return True.

    When is it appropriate to use definitely_true?  First, if you can use
    a higher level combinator prefer using those instead, they are definitely
    safe (modulo short-circuiting).
    Second, it can be used if the program would behave equivalently if
    definitely_true always returned False. Finally, it even
    be OK if the program wouldn't behave equivalently, so long as the
    change is semantics preserving.  It can be semantics preserving if
    the program errors in more cases than it did previously (but otherwise
    behaves identically), or if it changes some quantity in a way that
    doesn't matter (e.g., strides often fall in this bucket.)
    """
    if isinstance(a, SymBool):
        if a.node.has_hint():
            return guard_bool(a)
        else:
            return False
    return bool(a)


def definitely_false(a: BoolLikeType) -> bool:
    """
    Returns True only if we can tell that a is False, possibly introducing
    a guard in the process.  If a depends on some unbacked SymInt, we may
    return False even though there may exist a possible value of the SymInt
    that would cause the expression a to be False.  See definitely_true
    for more usage guidance.
    """
    if isinstance(a, SymBool):
        if a.node.has_hint():
            return not guard_bool(a)
        else:
            return False
    return not bool(a)


def statically_known_true(x: Union[bool, SymBool]) -> bool:
    """
    Returns True if x can be simplified to a constant and is true.

    .. note::
        This function doesn't introduce new guards, so the expression may end
        up evaluating to true at runtime even if this function returns False.

    Args:
        x (bool, SymBool): The expression to try statically evaluating
    """
    if isinstance(x, SymBool):
        expr = x.node.expr
        shape_env = x.node.shape_env
        try:
            simplified = shape_env._maybe_evaluate_static(expr)
            if simplified is not None:
                return bool(simplified)
        except Exception:
            log.debug("Could not simplify %s", expr)
        return False
    assert isinstance(x, bool)
    return x


def sym_eq(x: _T, y: _T) -> Union[bool, SymBool]:
    """
    Like ==, but when run on list/tuple, it will recursively test equality
    and use sym_and to join the results together, without guarding.
    """
    if (isinstance(x, tuple) and isinstance(y, tuple)) or (
        isinstance(x, list) and isinstance(y, list)
    ):
        if len(x) != len(y):
            return False
        return functools.reduce(operator.and_, map(sym_eq, x, y), True)
    elif isinstance(x, (int, torch.SymInt)) and isinstance(y, (int, torch.SymInt)):
        return x == y
    else:
        raise AssertionError(f"unexpected sym_eq between {type(x)} {type(y)}")


def guard_scalar(
    a: Union[SymBool, SymInt, SymFloat, int, bool, float]
) -> Union[bool, int, float]:
    if isinstance(a, (SymBool, bool)):
        return guard_bool(a)
    elif isinstance(a, (SymInt, int)):
        return guard_int(a)
    elif isinstance(a, (SymFloat, float)):
        return guard_float(a)
    else:
        raise AssertionError(f"unrecognized scalar {a}")


def _constrain_symbol_range(
    shape_env: ShapeEnv, s: sympy.Symbol, compiler_min: int, compiler_max: int
) -> None:
    shape_env.constrain_symbol_range(s, compiler_min, compiler_max)


def _advise_is_size(a: SymInt) -> None:
    """
    Don't use this directly; use torch._check_is_size instead.

    This is a softer version of _constrain_range_for_size (with min=0,
    max=Inf).  Instead of forcibly constraining a variable (and erroring if we
    failed to constrain it), it will simply advise us that a size is
    constrained in some way.  We will always defer a runtime assert for this
    constraint if we cannot prove it at compile-time, but we we only
    *sometimes* learn useful extra information at compile-time with this
    information.  This is in contrast to constrain_range_for_size, where if
    you don't call that on a fresh unbacked symint, chances are we will choke.

    TODO: Make Dynamo handle this appropriately if this is seen in Dynamo-ed
    code.  Right now this is only really used in code with AOTAutograd trace
    through, so it is not a big problem that this isn't supported, but in
    principle all of this code should be Dynamo'able too.

    TODO: I didn't support min/max because I didn't have a use case where this
    actually helped.  In principle we can support it, it just makes the
    implementation below more complicated.
    """

    # This must always succeed, because the sole allowed caller _check_is_size
    # was responsible for expect_true'ing this
    # This assert triggers expensive sym compute, do not do it until its cheap.
    # assert a >= 0

    # NB: it's important not to constrain range for size for *hinted* SymInts,
    # because it is not only unsound, it will immediately trip our asserts
    # that hints have to be consistent with static analysis!  If you somehow
    # have an unbounded SymInt that later constrains to 1, this will be
    # inconsistent with the range
    if (
        isinstance(a, SymInt)
        and isinstance(a.node, SymNode)
        and isinstance(a.node.expr, sympy.Symbol)
        and a.node.shape_env.is_unbacked_symint(a.node.expr)
    ):
        _constrain_range_for_size(a)


def _constrain_range_for_size(
    a: SymInt, min: Optional[int] = None, max: Optional[int] = None
) -> None:
    """
    This function is NOT INTENDED to be used by itself.
    """

    if isinstance(a, (SymFloat, SymBool)):
        raise ValueError("Constraining SymFloat/SymBool is nyi")

    assert isinstance(a, SymInt), "can only constrain range for SymInt"
    assert isinstance(a.node.expr, sympy.Symbol), f"constraining non-Symbols NYI: {a}"

    a.node.shape_env._constrain_range_for_size(a.node.expr, min, max)


# inclusive both ways
def constrain_range(
    a: SymInt, *, min: Optional[int], max: Optional[int] = None
) -> None:
    """
    Applies a constraint that the passed in SymInt must lie between min-max
    inclusive-inclusive, WITHOUT introducing a guard on the SymInt (meaning
    that it can be used on unbacked SymInts).  If min/max are None, we assume
    that the dimension is unbounded in that direction.  Repeated application
    of constrain_range intersects the ranges.  This is a fairly low level API
    that doesn't have a lot of safety guarantees (TODO: provide higher level
    APIs).

    Currently, we use this API in the following circumstance: when we allocate
    an unbacked SymInt, denoting an integer quantity which is data dependent,
    we ordinarily do not know anything about what values it may take.  This
    means that any sort of guard on it will immediately fail.  However, in
    many cases, we know something about the unbacked SymInt: for example, we
    know that nonzero(x).size(0) must be >= 0.  We use constrain_range to
    narrow the possible range, declaring that negative symbols are impossible.
    This permits to definitely answer True to queries like 'nnz >= 0', even if
    we don't know what the actual (hinted) value of 'nnz' is.  In fact, we
    actually use constrain_range to unsoundly discharge common guards: for an
    unbacked SymInt produced by nonzero, we will also assume that it is not
    equal to 0/1 (even though these are perfectly possible values at runtime),
    because we generally expect graphs that are valid for N=2 to also be valid
    for N=1.
    """
    if min is None:
        min = -int_oo
    if max is None:
        max = int_oo

    if max < min:
        raise ValueError(
            "Maximum value to constrain_as_size can't be less than the specified min value, "
            "received min={min} and max={max}"
        )

    if isinstance(a, int):
        if not (min <= a <= max):
            raise ValueError(f"Invalid value {a} for range [{min}:{max}]")
        return

    a.node.shape_env._constrain_range(a.node.expr, min, max)


def constrain_unify(a: torch.SymInt, b: torch.SymInt) -> None:
    """
    Given two SymInts, constrain them so that they must be equal.  NB:
    this will not work with SymInts that represent nontrivial expressions
    (yet!)
    """
    if not isinstance(a, SymInt):
        if not isinstance(b, SymInt):
            assert a == b
            return
        else:
            shape_env = b.node.shape_env
    else:
        shape_env = a.node.shape_env

    shape_env._constrain_unify(a, b)


# Assume that a boolean is true for the purposes of subsequent symbolic
# reasoning.  This will keep track of corresponding runtime checks to verify
# that the result is upheld: either as a regular guard, or as a special set
# of asserts which are triggered when an unbacked SymInt is allocated.
#
# DO NOT use this function for these cases:
#
#  - This is inappropriate for "branching" conditions (where both
#    true and false result in valid programs).  We will always assume
#    the condition evaluates true, and so it will never be possible
#    to trace the false condition when you use it.  For true branching
#    on unbacked SymInts, you must use torch.cond; if you incorrectly
#    use expect_true in this case, you will make the false branch
#    unreachable (as we will simply assume that only the true branch
#    is ever exercised).
#
#  - This is inappropriate for situations where you know some other system
#    invariant guarantees that this property holds, since you don't
#    really need to insert a runtime check in that case.  Use something
#    like constrain_range in that case.
#
# This API has a hitch.  To avoid having to reimplement error reporting
# capabilities, this function CAN return False.  The invariant is that
# the surrounding code must raise an error when this function returns
# False.  This is quite low level, so we recommend using other functions
# like check() which enforce this in a more intuitive way.
#
# By the way, this name is a nod to the __builtin_expect macro,
# which is used similarly (but unlike __builtin_expect, you MUST fail
# in the unlikely branch.)  (I think expect is a good name; in recent
# versions of C++, this is replaced with [[likely]], which is weaker
# and not accurate for this function!)
def expect_true(a: Union[SymBool, bool], skip: int = 0) -> bool:
    if isinstance(a, SymBool):
        # TODO: check perf implications of this
        frame = inspect.currentframe()
        for _ in range(skip + 1):  # always run this loop at least once
            if frame is None:
                break
            frame = frame.f_back
        return a.node.expect_true(
            frame.f_code.co_filename if frame else "", frame.f_lineno if frame else 0
        )
    assert type(a) is bool, a
    return a


def guard_bool(a: Union[SymBool, bool]) -> bool:
    if isinstance(a, SymBool):
        return a.node.guard_bool("", 0)  # NB: uses Python backtrace
    assert type(a) is bool, a
    return a


def guard_int(a: Union[SymInt, int]) -> int:
    if isinstance(a, SymInt):
        return a.node.guard_int("", 0)  # NB: uses Python backtrace
    assert type(a) is int, a
    return a


def guard_float(a: Union[SymFloat, float]) -> float:
    if isinstance(a, SymFloat):
        return a.node.guard_float("", 0)  # NB: uses Python backtrace
    assert isinstance(a, float), a
    return a


# Given a GraphModule, return all the FakeTensors for all the placeholders
def fx_placeholder_vals(gm: torch.fx.GraphModule) -> List[object]:
    return [n.meta["val"] for n in gm.graph.nodes if n.op == "placeholder"]


def fx_placeholder_targets(gm: torch.fx.GraphModule) -> List[str]:
    return [n.target for n in gm.graph.nodes if n.op == "placeholder"]


# Given a GraphModule and arguments to run it with, evaluate that the guards
# for its associated ShapeEnv are satisfied by the passed arguments.  This
# WILL check for duck sizing.
def eval_guards(
    gm: torch.fx.GraphModule, *args: Tensor, ignore_static: bool = True
) -> bool:
    return gm.shape_env.evaluate_guards_for_args(  # type: ignore[operator, union-attr]
        fx_placeholder_vals(gm), args, ignore_static=ignore_static
    )


def bind_symbols(gm: torch.fx.GraphModule, *args: Tensor) -> Dict[sympy.Symbol, int]:
    return gm.shape_env.bind_symbols(fx_placeholder_vals(gm), args)  # type: ignore[operator, union-attr]


class DimDynamic(Enum):
    """
    Controls how to perform symbol allocation for a dimension.  It is always
    sound to default this to DYNAMIC, but the policies DUCK and STATIC can
    result in better trace-time and compile-time performance, as they reduce
    the number of allocated symbols and generally make your graph more static.

    NB: If we notice you've applied a constraint to the dimension, we will
    force it to DYNAMIC for simplicity.

    DimDynamic is controlled by a variety of higher level UX features.
    Currently:

    - In eager mode, the default policy is DUCK.
        - The default is changed to STATIC with assume_static_by_default.
        - An individual dim is marked DYNAMIC if you mark_dynamic_dim.
    - In export mode, the default policy is STATIC.
        - An individual dim is marked DYNAMIC if you specify it in
          dynamic_shapes passed to export.
    """

    # Treat the dimension symbolically
    DYNAMIC = 0
    # Treat the dimension symbolically, but if its hint matches another
    # dynamic dimension, unify the two symbols ("duck sizing")
    DUCK = 1
    # Treat the dimension statically based on its hint
    STATIC = 2
    # Treat the dimension as a size-like unbacked
    SIZE_LIKE_UNBACKED = 3
    # Infer the strides from stride. If size is static, strides will be static as well.
    INFER_STRIDE = 4
    # Like SIZE_LIKE_UNBACKED, but there's a hint
    OBLIVIOUS_SIZE = 5


# NB: These constraints affect both clients and backends: given some
# constraint C, the client must pass inputs that satisfy the constraint,
# while a backend must not introduce guards BEYOND this constraint.
# For clarity, we document the implications on both sides for both the client
# and the backend.
#
# NB: These constraints are on a *single* dimension.  In principle, we could
# also have multi-dimension constraints, but our guess is that this is not
# actually useful and so we are not supporting it right now.
#
# NB: Strict constraints are typically only suitable for export, as in eager
# a backend like inductor may validly introduce extra, discretionary guards
# to improve performance of code.  A StrictMinMaxConstraint would be brittle
# under future optimizations performed by inductor; we don't guarantee
# eager code with StrictMinMaxConstraint will keep working in the future!


@dataclass(frozen=True)
class Constraint:
    warn_only: bool


@dataclass(frozen=True)
class StrictMinMaxConstraint(Constraint):
    """
    For clients: the size at this dimension must be within 'vr' (which
    specifies a lower and upper bound, inclusive-inclusive) AND it
    must be non-negative and should not be 0 or 1 (but see NB below).

    For backends: there must not be any guards on this dimension which
    are not implied by the given lower and upper bound.  Regardless of
    the lower bound, the backend can assume the size is non-negative
    and that it is not 0 or 1.

    An unbounded StrictMinMaxConstraint can be thought of as a strict version
    of "RelaxedUnspecConstraint".

    NB: Export will often unsoundly assume that a graph works for 0/1, even
    though at trace time we assumed size is not 0 or 1.  The idea is that
    if we produce a graph that works for a range of values, it will be OK
    for N=0/1 too.
    """

    vr: ValueRanges

    def render(self, source: Source) -> str:
        """Format the constrain equation"""
        # TODO: better printing for -oo and oo
        return f"{self.vr.lower} <= {source.name()} <= {self.vr.upper}"


@dataclass(frozen=True)
class RelaxedUnspecConstraint(Constraint):
    """
    For clients: no explicit constraint; constraint is whatever is implicitly
    inferred by guards from tracing.

    For backends: there must exist at least TWO possible values for the
    size at this dimension which satisfy the guards for this dimension.

    In other words, this constraint helps us distinguish between "we don't
    care if this dimension specializes or not" versus "this dimension must be
    unspecialized."  However, this constraint doesn't say very much about what
    specialization is permitted; for example, if we guard on a size being
    even, this would still be acceptable under an unspec constraint.  This
    makes RelaxedUnspecConstraint useful for eager mode, where your backend compiler
    may add constraints to otherwise dynamic dimensions; we can't assert that
    there are NO guards as this is brittle because compilers should be able to
    add extra constraints.  If you want to assert that there are no guards,
    use StrictMinMaxConstraint with an unbounded ValueRanges.
    """

    def render(self, source: Source) -> str:
        return f"RelaxedUnspecConstraint({source.name()})"


# NB: None here indicates the client constraint is whatever is implicitly
# inferred by guards from tracing, and that a backend can add whatever guards
# it wants (including fully specializing the value).
DimConstraint = Union[StrictMinMaxConstraint, RelaxedUnspecConstraint, None]


@dataclass(frozen=True)
class EqualityConstraint(Constraint):
    """
    Represent and decide various kinds of equality constraints between input sources.

    A "source pair" is a pair of input sources for dynamic dimensions that
    are specified equal. We represent `source_pairs` in a union-find forest
    so that we can efficiently check whether two such sources are transitively equal.

    A "derived equality" relates an input source to an expression over a root.
    The root can be another input source, corresponding to some dynamic dimension,
    or a phantom symbol that does not directly represent any dynamic dimension. We
    represent `derived_equalities` involving input sources in a transitively-closed map
    so that we can efficiently check whether an input source is transitively equal to
    a given expression over another input source.
    (NOTE: In contrast, it is easy to decide whether an input source is transitively equal
    to a given expression over a phantom symbol; such expressions are already in canonical
    form and so the problem reduces to symbolic expression equality.)
    """

    source_pairs: List[Tuple[Source, Source]]
    derived_equalities: List[
        Tuple[Source, Union[Source, sympy.Symbol], Callable[[sympy.Expr], sympy.Expr]]
    ]
    phantom_symbols: List[sympy.Symbol]
    relaxed_sources: Set[Source]

    _parents: Dict[Source, Source] = field(init=False)
    _defs: Dict[Source, sympy.Expr] = field(init=False)

    def __post_init__(self) -> None:
        """
        Pre-processing to answer queries `is_equal` and `is_derived` below.

        Example: Suppose we are given:
          source_pairs [a = b, b = c]
          derived_equalities [d = c + 1, e = d - 1]
        We first construct a union find with source_pairs:
          _parents = {a: a, b: a, c: a}
        Then we compute canonical symbolic expressions, recursively applying derived_equalities
        until we bottom out:
          _defs = {d: c + 1, e: (c + 1) - 1 aka c}
        """

        # self._parents is a map from input sources to input sources where, conceptually,
        # these are directed edges in a union-find forest
        _parents: Dict[Source, Source] = {}
        object.__setattr__(self, "_parents", _parents)
        # self._defs is a map from input sources to "canonical" symbolic expressions,
        # i.e., unary expressions with symbols that corresponds to regular Dims (i.e.,
        # not derived Dims)
        _defs: Dict[Source, sympy.Expr] = {}
        object.__setattr__(self, "_defs", _defs)

        for source1, source2 in self.source_pairs:
            # preprocess into a union-find forest
            self._union(self._find(source1), self._find(source2))
        for source, root, fn in self.derived_equalities:
            # preprocess into a transitively-closed map
            # NOTE(avik): we reuse the union-find forest for canonicalizing input sources
            if isinstance(root, sympy.Symbol):
                self._defs[self._find(source)] = fn(root)
            else:
                self._defs[self._find(source)] = fn(self._rewrite(root))

    def _find(self, source: Source) -> Source:
        # chase edges to find the root of this equivalence class
        if source in self._parents:
            return self._find(self._parents[source])
        else:
            return source

    def _union(self, root1: Source, root2: Source) -> None:
        # merge two equivalence classes by adding an edge from one root to the other
        if root1 != root2:
            self._parents[root1] = root2

    def _rewrite(self, src: Source) -> sympy.Expr:
        # always represent the given source by the root of its equivalence class
        src = self._find(src)
        if src in self._defs:
            # simply look up the definition if it exists
            # NOTE(avik): This works because definitions are always transitively-closed;
            # otherwise we would have to do recursive rewriting.
            return self._defs[src]
        else:
            # otherwise, create a symbol representing the source
            return sympy.Symbol(src.name())

    def is_equal(self, source1: Source, source2: Source) -> bool:
        return (
            # check whether source1 and source2 have the same root
            # or are relaxed
            (src1 := self._find(source1)) in self.relaxed_sources
            or (src2 := self._find(source2)) in self.relaxed_sources
            or src1 == src2
            # check whether source1 is derived equal to source2
            or self.is_derived(source1, source2, lambda x: x)
        )

    def is_derived(
        self, src: Source, symbol_src: Source, fn: Callable[[sympy.Expr], sympy.Expr]
    ) -> bool:
        # check whether both src and symbol_src have the same definition
        return self._rewrite(src) == fn(self._rewrite(symbol_src))


def _assert_symbol_context(symbolic_context: object) -> TypeGuard[SymbolicContext]:
    assert isinstance(
        symbolic_context, SymbolicContext
    ), "Invalid symbolic_context object"
    assert (
        type(symbolic_context) is not SymbolicContext
    ), "Illegal usage of symbolic_context ABC"
    return True


def _is_supported_equivalence(expr: sympy.Expr) -> bool:
    # Currently supported Dim ops are linear expressions with integer coefficients.
    # So check that expr only contains +, *, ints, and a single occurrence of a symbol.
    # (See also documentation of dynamic_shapes._DerivedDim.)
    if isinstance(expr, (sympy.Add, sympy.Mul)):
        if len(expr.args) > 2:
            return False
        lhs, rhs = expr.args
        return (_is_supported_equivalence(lhs) and isinstance(rhs, sympy.Integer)) or (
            isinstance(lhs, sympy.Integer) and _is_supported_equivalence(rhs)
        )
    return isinstance(expr, sympy.Symbol)


def _has_uninterpretable_sympy_function(expr: sympy.Basic) -> bool:
    """
    Add functions that our sympy interpreter can't reify into FX nodes
    """
    return expr.has(
        torch.utils._sympy.functions.ToFloat,
        torch.utils._sympy.functions.TruncToInt,
        torch.utils._sympy.functions.CeilToInt,
    )


@dataclass(frozen=True)
class SymbolicContext:
    """
    Data structure specifying how we should create symbols in
    ``create_symbolic_sizes_strides_storage_offset``; e.g., should
    they be static or dynamic.

    This is an abstract base class because we are probably going to add
    another version of this that says "use exactly these SymInts, don't
    allocate fresh symbols."
    """


@dataclass(frozen=True)
class StatelessSymbolicContext(SymbolicContext):
    """
    Create symbols in ``create_symbolic_sizes_strides_storage_offset`` via
    a symbolic_context determination as given by ``DimDynamic`` and ``DimConstraint``.
    This will cause fresh symbols to be allocated
    """

    dynamic_sizes: DimList[DimDynamic]
    dynamic_strides: DimList[DimDynamic] = None  # type: ignore[assignment]
    constraint_sizes: DimList[DimConstraint] = None  # type: ignore[assignment]
    constraint_strides: DimList[DimConstraint] = None  # type: ignore[assignment]
    # If the tensor is a view, this should be populated for the base. It contains
    # information on how to allocate symbols when recursively fakeifying the base
    # during view fake-ification.
    view_base_context: Optional[SymbolicContext] = None
    # TODO: add storage offset and stride symbolic_context

    def __post_init__(self) -> None:
        if self.dynamic_strides is None:
            object.__setattr__(
                self,
                "dynamic_strides",
                [DimDynamic.INFER_STRIDE] * len(self.dynamic_sizes),
            )
        if self.constraint_sizes is None:
            object.__setattr__(
                self, "constraint_sizes", [None] * len(self.dynamic_sizes)
            )
        if self.constraint_strides is None:
            object.__setattr__(
                self, "constraint_strides", [None] * len(self.dynamic_sizes)
            )
        assert all(
            stride in (DimDynamic.INFER_STRIDE, DimDynamic.DYNAMIC, DimDynamic.DUCK)
            for stride in self.dynamic_strides
        )


# note [Tensor Fakification and Symbol Caching]
#
# As of the time of this note, dynamo creates a fresh fake tensor mode for backends.
# The reason we do this is because there are certain classes of operations, namely,
# metadata mutations, that change tensor size, stride, etc. This means that the fake tensor
# state at the end of a dynamo trace is different than the fake tensor state at the beginning
# of a trace. Backends like aot_autograd need a fresh fake tensor to correctly track metadata mutation,
# view relationships, etc.
#
# As we create a new fake mode, we also lose the memoization that comes with it. Rather than
# transfer the memoization cache, we instead transfer the shape env. However, with this
# comes nuance - as dynamo is selective in how it makes symbolic shapes. Due to strategies in
# automatic dynamic and constraints, the policy for which dims are dynamic is nuanced and varies across
# recompilations.
#
# In order to preserve the symbolic decisions made during dynamo tensor fakification, we pass
# a StatefulSymbolicContext at creation time. This object is tracked, per tensor, on the TracingContext.
# The lifecycle of this object should match the lifecycle of the original dynamo tracked tensor, and it is
# safe to reuse this object as many times as necessary to create a fake tensor. Fake tensors
# created with new fake modes should produce the same exact symbols as the original, providing the same shape_env
# is used.
# TODO(voz): Shape env validation
@dataclass(frozen=True)
class StatefulSymbolicContext(StatelessSymbolicContext):
    """
    Create symbols in ``create_symbolic_sizes_strides_storage_offset`` via
    a symbolic_context determination as given by a cache of Source:Symbol. A cache hit
    will reuse a stored symbol, and a cache miss will write to this cache.

    This behaves like StatelessSymbolicContext, except the cache supersedes the
    other values - dynamic_sizes and constraint_sizes will not be read if we cache
    hit.

    It is the cache owners responsibility to maintain the lifecycle of the cache
    w/r/t different shape_envs, clearing, etc.
    """

    tensor_source: Source = None  # type: ignore[assignment]
    # Why is this keyd on int first?
    # That integer is actually the id of the shape_env. This cache short-circuits symbol
    # creation, and we must store it per shape env. Now, while tracing invariants are a single
    # shape env per tracing context, and every new frame gets a new shape_env. So where would we have
    # multiple shape envs? The answer lies in recording. When we are replaying, replay_shape_env_events
    # is invoked, and creates a new shape_env. Replaying events against this new shape_env will
    # cause it to fail with unknown symbols, as the symbols cached here will skip creation, and never
    # get recorded in var_to_val, etc.
    # TODO(voz): consider a weakref to the shape_env here
    shape_env_to_source_to_symbol_cache: Dict[int, Dict[str, sympy.Expr]] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        super().__post_init__()
        # The None default is annoying, but required because of dataclass limitations
        assert self.tensor_source is not None
        if not self.shape_env_to_source_to_symbol_cache:
            object.__setattr__(self, "shape_env_to_source_to_symbol_cache", {})


@dataclass(frozen=True)
class SubclassSymbolicContext(StatefulSymbolicContext):
    """
    The correct symbolic context for a given inner tensor of a traceable tensor subclass
    may differ from that of the outer symbolic context. This structure allows for this
    flexibility, with inner symbolic contexts mapped via attr -> symbolic context.
    """

    inner_contexts: Dict[str, SymbolicContext] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.inner_contexts is None:
            self.inner_contexts = {}


def is_symbolic(
    val: Union[int, SymInt, float, SymFloat, bool, SymBool]
) -> TypeGuard[Union[SymInt, SymFloat, SymBool]]:
    if isinstance(val, (int, float, bool)):
        return False
    return val.node.is_symbolic()


IndicatorTypes = (IsNonOverlappingAndDenseIndicator,)


def _expandsums(args: List[sympy.Expr]) -> Tuple[sympy.Expr, bool]:
    adds, other = [], []
    for arg in args:
        if arg.is_Add:
            adds.append(arg)
        else:
            other.append(arg)

    result = [sympy.Mul(*other)]
    for add in adds:
        result = [a * b for a, b in itertools.product(result, add.args)]

    result = sympy.Add(*result)
    return result, len(adds) > 1 or (len(adds) > 0 and len(other) > 0)


def _fast_expand(expr: _SympyT) -> _SympyT:
    # The expand algorithm in sympy is slow due to all the features is supports
    # For eg: e^(-x)*(x-1)/(x+1) is expanded to (x-1)/(e^x + e^x*x) if x is
    # positive and (e^(-x)*x-e^(-x))/(x+1) if x is negative. We do not implement
    # such features here to avoid expensive checks. We also make sure that we
    # only re-create the objects if any of the args changed to avoid expensive
    # checks when re-creating objects.
    new_args = [_fast_expand(arg) for arg in expr.args]  # type: ignore[arg-type]
    if any(arg is not new_arg for arg, new_arg in zip(expr.args, new_args)):
        return _fast_expand(expr.func(*new_args))

    if expr.is_Pow:
        base: sympy.Expr
        exp: sympy.Expr
        base, exp = expr.args  # type: ignore[assignment]
        if exp.is_Integer and base.is_Add:
            if exp > 1:
                return sympy.expand_multinomial(expr, deep=False)
            elif exp < 0:
                return S.One / sympy.expand_multinomial(S.One / expr, deep=False)
    elif expr.is_Mul:
        num: List[sympy.Expr] = []
        den: List[sympy.Expr] = []
        for arg in expr.args:
            if arg.is_Pow and arg.args[1] == -1:
                den.append(S.One / arg)  # type: ignore[operator, arg-type]
            else:
                num.append(arg)  # type: ignore[arg-type]

        num, num_changed = _expandsums(num)
        den, den_changed = _expandsums(den)
        if num_changed or den_changed:
            return num / den

    return expr


@lru_cache(256)
def safe_expand(r: _SympyT) -> _SympyT:
    """
    Expand the given symbolic expression by recursively rewriting product of
    sums into sum of products (with the product being either a multiplication or
    exponentiation).

    NOTE: using this on an intermediate expression may prevent simplification
    down the line, e.g., if we eagerly expand `(a + b)^2` into `a^2 + 2ab + b^2`,
    we won't be able to simplify `(a^2 + 2ab + b^2) / (a + b)` as easily.
    """
    if hasattr(r, "expand"):
        try:
            return _fast_expand(r)
        except RecursionError:
            log.warning("RecursionError in _fast_expand(%s)", r)
            return r
    else:
        return r


@lru_cache(None)
def _maybe_evaluate_static_worker(
    expr: _SympyT,
    symbol_info: Tuple[Tuple[sympy.Symbol, ValueRanges, sympy.Integer, bool], ...],
    unbacked_only: bool,
    size_oblivious: bool,
) -> Optional[_SympyT]:
    """
    This variant of ShapeEnv._maybe_evaluate_static has no dependence on
    ShapeEnv and thus can be cached indefinitely.  It does the "heavy" lifting
    for static evaluation, including nontrivial reliance on Sympy simplification
    that occurs when we reallocate the symbols
    """

    # Simplify making use of value range lower bound
    new_shape_env = {}
    new_range_env = {}
    for idx, sinfo in enumerate(symbol_info):
        k, vr, val, is_size_like = sinfo
        if isinstance(val, SingletonInt):
            # Skip var_ranges logic for SingletonInt which is only used
            # for jagged layout NestedTensors today
            continue
        if size_oblivious and is_size_like:
            lower = max(2, vr.lower)
            # Clamping size-oblivious to some quantity below sys.maxsize
            # helps us determine that f(u0) != sys.maxsize, which is a
            # test that is looking for sys.maxsize as a sentinel, but you
            # don't really want to worry about it for unbacked SymInts.
            # This is similar to the flavor where size oblivious omits
            # 0/1, it changes semantics but in a benign way.
            upper = min(2**48, vr.upper)
            # This is a bit dodgy: what this means is that there was a
            # size-like unbacked symbol whose upper bound < 2.  This
            # causes... problems.
            if lower <= upper:
                vr = ValueRanges(lower, upper)
        else:
            lower = vr.lower
        # Don't do anything if we don't have a nontrivial lower bound
        # Also don't do anything if we asked only to simplify unbacked
        # SymInt
        if lower is -int_oo or (unbacked_only and val is not None) or not vr.is_int:
            new_range_env[k] = vr
            continue
        # The goal is to take our symbols which have various lower bounds
        # and reallocate them into new symbols which are exactly positive;
        # e.g., if we have s0 in [2, inf], we want to turn it into ess0 in
        # [1, inf], where s0 = ess0 + 1.  This gives the most information
        # to sympy for subsequent simplifications.
        #
        # Positive means >= 1
        # Positive - 1 means >= 0
        # Positive + lower - 1 means >= lower
        # The new symbol 's' is "too low", so when we substitute it in
        # we have to increase it by offset (and conversely, the new
        # variables have to have their value range bounds adjusted as
        # well)
        s = sympy.Symbol(f"evaluate_static_shape_{idx}", positive=True, integer=True)

        # Note:
        #   Offset might be a fraction(e.g. aten.split.Tensor), but shapes are always integers.
        #   Sympy might give unexepected results when comparing an integer with a non-integer
        #   Therefore, we cast offset to int here.
        #   For example:
        #       shape_0 = sympy.Symbol("shape_0", positive=True, integer=True)
        #       expr = sympy.Eq(shape_0 - 1/3, 4)
        #       expr.xreplace({}) # False
        offset = int(lower - 1)
        new_shape_env[k] = s + offset
        new_range_env[s] = SymPyValueRangeAnalysis.add(vr, -offset)

    # TODO: remove this try catch (esp for unbacked_only)
    try:
        new_expr = expr.xreplace(new_shape_env)
    except RecursionError:
        log.warning("RecursionError in sympy.xreplace(%s, %s)", expr, new_shape_env)
        return None

    # We need to canonicalize, as after expand we may have something like `a + b = a` and
    # sympy will not simplify the a. The two appeareances of the a will then make value ranges
    # analysis give lose bounds
    new_expr = canonicalize_bool_expr(safe_expand(new_expr))
    if new_expr.is_number:
        return new_expr

    # Check if the range can solve it statically
    out = bound_sympy(new_expr, new_range_env)
    if out.is_singleton():
        return out.lower

    return new_expr if unbacked_only else None


def error() -> NoReturn:
    raise AssertionError("shouldn't be hit")


# TODO: Deduplicate this with torch/_prims_common/__init__.py
def eval_is_non_overlapping_and_dense(
    sizes: Sequence[int], strides: Sequence[int]
) -> int:
    return int(guard_bool(_eval_is_non_overlapping_and_dense(sizes, strides)))


def _eval_is_non_overlapping_and_dense(
    sizes: Sequence[int], strides: Sequence[int]
) -> bool:
    dim = len(sizes)

    # Short-circuits for tensors of rank one, which are
    # non-overlapping and "dense" if their stride is one
    # or it is a 0/1 element tensor
    if dim == 1:
        return strides[0] == 1 or sizes[0] < 2

    # Checks that there exists a permutation of the strides s.t. the tensor would be contiguous
    # Sorts (length, stride) pairs by stride
    lengths_and_strides = sorted(zip(sizes, strides), key=operator.itemgetter(1))

    # Unlike the C++ code, we don't move the 0/1 size dimensions to the
    # end.  So we have to keep going for this code.
    expected_stride = 1
    for length, stride in lengths_and_strides:
        if length == 1:
            continue

        if stride != expected_stride:
            return False

        expected_stride *= length

    return True


def _sympy_cast_symbool_to_symint_guardless(x: SympyBoolean) -> sympy.Expr:
    return sympy.Piecewise((1, x), (0, True))


def cast_symbool_to_symint_guardless(
    symbool: Union[bool, torch.SymBool]
) -> Union[int, torch.SymInt]:
    if isinstance(symbool, bool):
        return 1 if symbool else 0
    int_sym = _sympy_cast_symbool_to_symint_guardless(symbool.node.expr)
    return symbool.node.shape_env.create_symintnode(
        int_sym, hint=int(symbool.node.require_hint()) if has_hint(symbool) else None
    )


SYMPY_INTERP = {
    "IsNonOverlappingAndDenseIndicator": eval_is_non_overlapping_and_dense,
    "cast_symbool_to_symint_guardless": cast_symbool_to_symint_guardless,
    "math": math,
}


def _lru_cache(
    fn: Callable[..., _T], maxsize: Optional[int] = None
) -> functools._lru_cache_wrapper[_T]:
    """
    Wrapper around lru_cache that clears when new info about shapes has been
    updated.

    Use lru_cache if the output is always the same, regardless of the
    constraints we know now (i.e. evaluate_expr)

    Use _lru_cache otherwise.

    Also note that this depends on _update_version_counter being called on the
    shape environment whenever the constraints are updated, otherwise the cache
    will not be cleared.
    """
    fn_cache = lru_cache(maxsize)(fn)
    prior_version = 0

    if config.validate_shape_env_version_key:
        prior_key = None

        @functools.wraps(fn)
        def wrapper(self: ShapeEnv, *args: Any, **kwargs: Any) -> _T:
            nonlocal prior_version, prior_key
            if prior_key is None:
                prior_key = self._get_key()

            if prior_version != self._version_counter:
                fn_cache.cache_clear()
                prior_version = self._version_counter
                prior_key = self._get_key()
            else:
                assert (
                    prior_key == self._get_key()
                ), "ShapeEnv cache key changed without version being updated!"

            return fn_cache(self, *args, **kwargs)

    else:

        @functools.wraps(fn)
        def wrapper(self: ShapeEnv, *args: Any, **kwargs: Any) -> _T:  # type: ignore[misc]
            nonlocal prior_version
            if prior_version != self._version_counter:
                fn_cache.cache_clear()
                prior_version = self._version_counter

            return fn_cache(self, *args, **kwargs)

    wrapper.cache_clear = fn_cache.cache_clear  # type: ignore[attr-defined]
    wrapper.cache_info = fn_cache.cache_info  # type: ignore[attr-defined]
    return wrapper  # type: ignore[return-value]


# This is pretty similar to ShapeGuard but it also comes with a message,
# and is exclusively used for things that MUST be true (unlike guards,
# which can evaluate False, in which case you just choose not to use
# a particular specialization)
@dataclass(frozen=True)
class RuntimeAssert:
    expr: SympyBoolean
    msg: str = field(repr=False)
    stack: CapturedTraceback = field(repr=False)


# Used for printing SymExprs in compile_fx
class SymExprPrinter(PythonPrinter):
    def _print_Float(self, expr: sympy.Float) -> str:
        return str(float(expr))


class _ShapeGuardPrinter(abc.ABC):
    def __init__(
        self,
        symbol_to_source: Mapping[sympy.Symbol, List[Source]],
        source_ref: Callable[[Source], str],
        var_to_sources: Mapping[sympy.Symbol, List[Source]],
    ) -> None:
        self.symbol_to_source = symbol_to_source
        self.source_ref = source_ref
        self.var_to_sources = var_to_sources
        super().__init__()

    def _print_Float(self, expr: sympy.Float) -> str:
        return str(float(expr))

    def _print_Symbol(self, expr: sympy.Symbol) -> str:
        assert isinstance(expr, sympy.Symbol), str(type(expr))

        def repr_symbol_to_source() -> str:
            return repr(
                {
                    symbol: [s.name() for s in sources]
                    for symbol, sources in self.symbol_to_source.items()
                }
            )

        assert self.symbol_to_source.get(expr), (
            f"{expr} (could be from {[s.name() for s in self.var_to_sources[expr]]}) "
            f"not in {repr_symbol_to_source()}.  If this assert is failing, it could be "
            "due to the issue described in https://github.com/pytorch/pytorch/pull/90665"
        )
        return self.print_source(self.symbol_to_source[expr][0])

    @abc.abstractmethod
    def print_source(self, source: Source) -> str:
        ...


class ShapeGuardPythonPrinter(_ShapeGuardPrinter, PythonPrinter):
    def __init__(self, *args: Any) -> None:
        super().__init__(*args)
        self._print_cache: Dict[sympy.Expr, str] = {}

    def print_source(self, source: Source) -> str:
        return self.source_ref(source)

    def doprint(self, expr: sympy.Expr) -> str:
        val = self._print_cache.get(expr, None)
        if val is not None:
            return val
        else:
            res = super().doprint(expr)
            self._print_cache[expr] = res
            return res


@deprecated(
    "`torch.fx.experimental.symbolic_shapes.ShapeGuardPrinter` is deprecated, "
    "please use `torch.fx.experimental.symbolic_shapes.ShapeGuardPythonPrinter` instead.",
    category=FutureWarning,
)
class ShapeGuardPrinter(ShapeGuardPythonPrinter):
    pass


class ShapeGuardCppPrinter(_ShapeGuardPrinter, CppPrinter):
    def __init__(self, *args: Any) -> None:
        self.all_symbols: Set[str] = set()
        self.source_to_symbol: Dict[Source, sympy.Symbol] = {}
        super().__init__(*args)

    def print_source(self, source: Source) -> str:
        if source in self.source_to_symbol:
            return self.source_to_symbol[source].name

        source_name = source.name()
        mangled_name = re.sub("[^0-9a-zA-Z_]+", "_", source_name)
        old_mangled_name = mangled_name
        count = 0
        while mangled_name in self.all_symbols:
            mangled_name = f"{old_mangled_name}_{count}"
            count += 1
        self.source_to_symbol[source] = sympy.Symbol(mangled_name)
        self.all_symbols.add(mangled_name)
        return mangled_name


# A dataclass for storing shape guards
@dataclass(frozen=True)
class _ShapeGuardsHelper:
    exprs: List[str]


# A dataclass for storing C++ expressions and helper variables
@dataclass(frozen=True)
class _CppShapeGuardsHelper(_ShapeGuardsHelper):
    source_to_symbol: Dict[Source, sympy.Symbol]


class LoggingShapeGuardPrinter(ShapeGuardPythonPrinter):
    def __init__(self, var_to_sources: Mapping[sympy.Symbol, List[Source]]):
        super().__init__(var_to_sources, lambda n: n.name(), var_to_sources)


class DynamicDimConstraintPrinter(PythonPrinter):
    """
    Printer for dynamic dim constraints.
    - Instead of symbol s_k it prints its source t.size()[i]
    - Instead of Eq(_, _), Mod(_, _), etc. it prints _ == _, _ % _, etc.

    We use this to suggest code for specifying dynamic dim constraints.
    """

    def __init__(
        self,
        symbol_to_source: Dict[sympy.Symbol, List[Source]],
        source_name_to_debug_name: Mapping[str, str],
    ):
        super().__init__()
        self.symbol_to_source = symbol_to_source
        self.source_name_to_debug_name = source_name_to_debug_name

    def _print_Symbol(self, expr: sympy.Symbol) -> str:
        assert isinstance(expr, sympy.Symbol), str(type(expr))
        assert self.symbol_to_source.get(
            expr
        ), f"Unknown symbol {expr} created by constraints solver"
        return self.symbol_to_source[expr][0].name()


class DimConstraints:
    """
    Custom solver for a system of constraints on symbolic dimensions.
    Solutions are "static" values or simplified "dynamic" constraints.
    """

    def __init__(
        self,
        symbol_to_source: Dict[sympy.Symbol, List[Source]],
        var_to_val: Mapping[sympy.Symbol, sympy.Integer],
        marked_dynamic: Set[sympy.Symbol],
        source_name_to_debug_name: Mapping[str, str],
    ) -> None:
        # We try to solve systems of inequalities with 1 free variable.
        self._univariate_inequalities: Dict[
            sympy.Symbol, Set[SympyBoolean]
        ] = defaultdict(set)
        # Among them, we prioritize solving for a free variable that has equalities.
        # NOTE: _symbols_with_equalities is always a subset of _univariate_inequalities.keys()
        # and removing a symbol from the former => removing it from the latter.
        self._symbols_with_equalities: Set[sympy.Symbol] = set()
        # A solution of a free variable with equalities becomes a substitution.
        # We use these substitutions to simplify other constraints.
        # NOTE: removing a symbol from _symbols_with_equalities => adding it to _substitutions.
        self._substitutions: Dict[sympy.Symbol, sympy.Integer] = {}

        # In general, constraints may have // and % operations.
        # Of course, // can be expressed in terms of / and %.
        # Our inequality solver can handle / but not %. So we need to transform them away.
        # We do so by using the values of variables as hints to evaluate %.
        # For soundness we record additional congruence guards and solve them separately.
        self._var_to_val: Mapping[sympy.Symbol, sympy.Integer] = var_to_val
        self._congruences: DefaultDict[sympy.Symbol, Set[sympy.Expr]] = defaultdict(set)

        # We do not try to (directly) solve inequalities with > 1 free variables.
        # NOTE: free variables in these inequalities cannot also be in _substitutions.
        self._multivariate_inequalities: Set[SympyBoolean] = set()

        # We park external equalities between free variables here.
        self._symbolic_equivalences: List[Tuple[Source, sympy.Expr]] = []

        # Solutions come in two forms:
        # - (static) specializations
        # - (dynamic) inequalities / congruences
        self._static_results: Set[str] = set()
        self._dynamic_results: Set[str] = set()

        # printer for solutions
        self._dcp = DynamicDimConstraintPrinter(
            symbol_to_source, source_name_to_debug_name
        )

        # inconsistencies found on substituting with concrete values / static solutions
        self._inconsistencies: List[str] = []

        # symbols that are marked dynamic
        self._marked_dynamic = marked_dynamic

        # track supported sympy functions and subtract from list of all sympy functions
        self._supported_sympy_functions: Set[sympy.Function] = {
            Application,
            Mod,
            PythonMod,
            FloorDiv,
        }
        self._enumerate_sympy_functions()

    def rewrite_with_congruences(self, s: sympy.Symbol, expr: _SympyT) -> _SympyT:
        """
        Eliminate expressions of the form b // d and b % d while adding congruences of the form b % d == k.
        This leaves rational operators (in particular of the form b / d) that our inequality solver can handle.
        We solve the added congruences separately (using our congruence solver, see below).
        """

        def mod_handler(*args: sympy.Expr) -> sympy.Expr:
            # Suppose that we have an expression of the form b % d with free variable s.
            # Using the value of s as a "hint," we can evaluate b % d to a value k.
            # Then we can rewrite b % d to k while adding the guard b % d == k.

            # NOTE(avik): This abstraction is provably sound but, in general, incomplete. It is complete IFF
            # the original expression always evaluates to a constant value (i.e., it does not vary with s).
            # In other words,
            # - solutions of s with the rewritten expression are guaranteed to also be solutions of s with
            #   the original expression;
            # - while it may be possible to find solutions of s with the original expression that are not
            #   solutions with the rewritten expression, in that case the original expression cannot evaluate
            #   to the same value for all solutions of s.
            #
            # Should we be worried about this incompleteness? No, because of the following reasons:
            # 1. It unblocks dramatic simplification that would not be otherwise possible with current tech
            #    (i.e., "don't let perfect be the enemy of the good").
            # 2. We already have a tradition of using hints to add guards in the compiler for making progress.
            # 3. We have not yet seen a counterexample arise in practice! In particular, any congruence guards
            #    we generate (or simplify to) seem to be of the form b % d == k where k is a constant.
            #
            # Here's a theoretical counterexample: 3*s % (s + 1) == s - 2, that is satisfied by all s >= 2.
            # With any hint (say) s = k, we'd rewrite this to: 3*s % (s + 1) == k - 2. But, substituting, we
            # would then get k - 2 == s - 2, and thus s = k as the (only, constant) solution!
            base, divisor = args
            base, divisor = self.rewrite_with_congruences(
                s, base
            ), self.rewrite_with_congruences(s, divisor)
            mod_reduced = base.xreplace(self._var_to_val) % divisor.xreplace(
                self._var_to_val
            )
            congruence = (base - mod_reduced) % divisor
            if congruence != 0:
                self._congruences[s].add(congruence)
            return mod_reduced

        def floor_div_handler(*args: sympy.Expr) -> sympy.Expr:
            # Suppose that we have an expression of the form b // d with free variable s.
            # Using the value of s, we can evaluate b % d to a value k.
            # Then we can rewrite b // d to (b - k) / d, while adding the guard b % d == k.

            # NOTE(avik): This is exactly equivalent to rewriting b // d as (b - (b % d)) / d
            # and eliminating b % d as above.
            base, divisor = args
            base, divisor = self.rewrite_with_congruences(
                s, base
            ), self.rewrite_with_congruences(s, divisor)
            mod_reduced = base.xreplace(self._var_to_val) % divisor.xreplace(
                self._var_to_val
            )
            congruence = (base - mod_reduced) % divisor
            if congruence != 0:
                self._congruences[s].add(congruence)
            # NB: Must not be CleanDiv, it needs to be regular sympy division
            # so inequality solver works.  This is sort of problematic for
            # is_integer tests though haha
            return (base - mod_reduced) / divisor

        if expr.has(Mod):
            expr = expr.replace(Mod, mod_handler)
        # 7 // -3 is -3, 7 % -3 is -2, and 7 - (-2) / -3 is -3.0 so negative
        # arguments should be OK.
        if expr.has(PythonMod):
            expr = expr.replace(PythonMod, mod_handler)
        if expr.has(FloorDiv):
            expr = expr.replace(FloorDiv, floor_div_handler)
        return expr

    def _enumerate_sympy_functions(self) -> None:
        module = torch.utils._sympy.functions
        all_functions = set()
        for attr in dir(module):
            if isinstance(func := getattr(module, attr), sympy.FunctionClass):
                all_functions.add(func)
        self._unsupported_sympy_functions = all_functions.difference(
            self._supported_sympy_functions
        )

    def _has_unsupported_sympy_function(self, expr: sympy.Basic) -> bool:
        """
        Tracks list of sympy.Functions the export solver doesn't know how to handle.
        """
        return expr.has(*self._unsupported_sympy_functions)

    def add(self, expr: SympyBoolean) -> bool:
        """Add an expression to the set of constraints.

        Return whether the expression is a trivial constraint (i.e., an obvious tautology).
        """
        if expr == sympy.true:
            return True
        orig_expr = expr
        orig_reduced = orig_expr.xreplace(self._var_to_val)
        # TODO(avik): https://github.com/pytorch/pytorch/issues/101093
        # It is possible that `expr` will fail the consistency check because of
        # precision errors. Specifically, on substituting its free symbols with
        # their concrete values, we might end up comparing floats. Until we have
        # a fix for this issue, we delay raising such failures. See solve().
        if orig_reduced == sympy.false:
            self._inconsistencies.append(f"{orig_expr} is inconsistent!")
        if isinstance(expr, sympy.Ne) or self._has_unsupported_sympy_function(expr):
            # we're not going to do anything useful with these, so drop them
            return False
        free_symbols = expr.free_symbols
        assert free_symbols, f"Did not expect constraint with no free variables: {expr}"
        if len(free_symbols) > 1:
            # multivariate: record and move on
            self._multivariate_inequalities.add(expr)
        else:
            # univariate: can solve these immediately
            s = next(iter(free_symbols))
            # eliminate // and % (see documentation of `rewrite_with_congruences` above)
            old_n_congruences = len(self._congruences[s])
            expr = self.rewrite_with_congruences(s, expr)
            new_n_congruences = len(self._congruences[s])
            if expr == sympy.true:
                return old_n_congruences == new_n_congruences
            reduced = expr.xreplace(self._var_to_val)
            if reduced == sympy.false:
                self._inconsistencies.append(
                    f"{expr}, obtained by rewriting {orig_expr} with congruences, "
                    "is inconsistent!"
                )
            if isinstance(expr, sympy.Eq):
                # special status for symbols that have equalities (see `solve` below)
                self._symbols_with_equalities.add(s)
            self._univariate_inequalities[s].add(expr)
        return False

    def add_equality(self, source: Source, expr: sympy.Expr) -> None:
        """Add an equality constraint"""
        if expr.is_number:
            # specialization, right here
            self._static_results.add(f"{source.name()} == {expr}")
        else:
            # these will resolve to either specializations or dynamic equality constraints
            self._symbolic_equivalences.append((source, expr))

    def _reduce_congruences(self) -> Dict[sympy.Symbol, Set[sympy.Expr]]:
        reduced_congruences: Dict[sympy.Symbol, Set[sympy.Expr]] = {}
        for s, congruences in self._congruences.items():
            remainder_modulus_pairs = []
            congruences_to_check = set()
            for congruence in congruences:
                base, divisor = congruence.args
                # We are given a congruence of the form base % divisor == 0 with a free variable s. So:
                # - we transform this into an equation of the form base = divisor * tmp;
                # - we solve this equation for s to get a linear solution with free variable tmp.
                tmp = sympy.Symbol("reduce_congruences_tmp", integer=True)
                symbol, solution = sympy.solve_linear(base - divisor * tmp, symbols=[s])
                # See https://docs.sympy.org/latest/modules/solvers/solvers.html#sympy.solvers.solvers.solve_linear
                # for how to interpret the results.
                if s == symbol:
                    # This means the solution is of the form s = modulus*tmp + remainder.
                    modulus, remainder = sympy.polys.polytools.div(solution, tmp)
                    if isinstance(modulus, sympy.Integer) and isinstance(
                        remainder, sympy.Integer
                    ):
                        # Make sure 0 <= remainder <= modulus.
                        remainder = remainder % modulus
                        remainder_modulus_pairs.append((remainder, modulus))
                        continue
                # This means that we did not get a unique solution to the equation.
                # No problem, we will check it.
                congruences_to_check.add(congruence)
            # Finally we solve for a congruence s such that s = r_i mod m_i for each (r_i, m_i).
            # The solution will be a congruence of the form s = r mod m.
            # NOTE(avik): Since the given m_i may not be pairwise coprime, we can't just use CRT.
            if remainder_modulus_pairs:
                remainder, modulus = sympy.ntheory.modular.solve_congruence(
                    *remainder_modulus_pairs
                )
                reduced_congruences[s] = {(s - remainder) % modulus}
                substitution = {
                    s: modulus * sympy.Symbol("tmp", integer=True) + remainder
                }
                reduced_congruences[s].update(
                    congruence
                    for congruence in congruences_to_check
                    if not sympy.checksol(congruence, substitution)
                )
            else:
                reduced_congruences[s] = congruences_to_check

        return reduced_congruences

    def _raise_inconsistencies(self) -> None:
        if self._inconsistencies:
            msg = "\n".join(self._inconsistencies)
            self._inconsistencies.clear()
            raise ValueError(f"The following inconsistencies were found:\n{msg}")

    def solve(self) -> None:
        """Solve the system of constraint equations to find simplified constraints"""
        self._raise_inconsistencies()
        # as long as there are symbols with equalities, solve for them
        # NOTE(avik): this is guaranteed to terminate (#iterations <= #symbols)
        while self._symbols_with_equalities:
            s = self._symbols_with_equalities.pop()
            exprs = self._univariate_inequalities.pop(s)
            solution = sympy.solvers.inequalities.reduce_inequalities(exprs, s)
            if isinstance(solution, sympy.And):
                solution = next(
                    (arg for arg in solution.args if isinstance(arg, sympy.Eq)),
                    solution,
                )
            assert isinstance(
                solution, sympy.Eq
            ), f"Expected an equality constraint for {s}, got {solution}"
            symbol, val = solution.args
            assert symbol == s, f"Expected a constraint on {s} instead of on {symbol}"
            # because this is univariate, the solution is a specialization
            self._static_results.add(
                f"{self._dcp.symbol_to_source[s][0].name()} == {val}"
            )
            # add this as a substitution to simplify other constraints
            self._substitutions[s] = val  # type: ignore[assignment]

            # simplify multivariate inequalities: some of them will now become univariate!
            multivariate_inequalities = self._multivariate_inequalities
            self._multivariate_inequalities = set()
            for expr in multivariate_inequalities:
                self.add(expr.xreplace({s: self._substitutions[s]}))
            self._raise_inconsistencies()

        # solve linear congruences
        # NOTE(avik): We do not need to solve them for symbols that have already been specialized.
        reduced_congruences = self._reduce_congruences()
        for s, congruences in reduced_congruences.items():
            for congruence in congruences:
                # any congruence that cannot be checked becomes a dynamic constraint as well
                if s not in self._substitutions or not sympy.checksol(
                    congruence, {s: self._substitutions[s]}
                ):
                    if self._is_supported_congruence(congruence):
                        base, divisor = congruence.args
                        tmp_name = "_" + str(
                            self._dcp.source_name_to_debug_name.get(
                                self._dcp.symbol_to_source[s][0].name(),
                                self._dcp.symbol_to_source[s][0].name(),
                            )
                        )
                        tmp = sympy.Symbol(tmp_name, integer=True)
                        from torch._dynamo.source import ConstantSource

                        self._dcp.symbol_to_source[tmp] = [ConstantSource(tmp_name)]
                        r = try_solve(sympy.Eq(base, divisor * tmp), s)
                        assert r is not None
                        self._dynamic_results.add(self._dcp.doprint(sympy.Eq(s, r[1])))

        # remaining symbols have only pure inequalities (no equalities)
        for s, exprs in self._univariate_inequalities.items():
            try:
                solution = sympy.solvers.inequalities.reduce_inequalities(exprs, s)
                # because this is univariate, the solution is a dynamic (range) constraint
                if isinstance(solution, sympy.Or):
                    solution = next(
                        iter(
                            arg
                            for arg in solution.args
                            if arg.xreplace(self._var_to_val)
                        )
                    )
                if isinstance(solution, sympy.And):
                    for arg in solution.args:
                        self._dynamic_results.add(self._dcp.doprint(arg))
                else:
                    self._dynamic_results.add(self._dcp.doprint(solution))
            except (NotImplementedError, AssertionError) as e:
                log.warning("Failed to reduce inequalities: %s", e)
                for expr2 in exprs:
                    self._dynamic_results.add(self._dcp.doprint(expr2))

        # simplify symbolic equivalences: some of them will now become specializations!
        symbolic_equivalences = self._symbolic_equivalences
        self._symbolic_equivalences = []
        for source, expr3 in symbolic_equivalences:
            self.add_equality(source, expr3.xreplace(self._substitutions))

        # remaining symbolic equivalences become dynamic equality constraints
        for source, expr3 in self._symbolic_equivalences:
            self._dynamic_results.add(f"{source.name()} == {self._dcp.doprint(expr3)}")

    @classmethod
    def _is_supported_congruence(cls, congruence: sympy.Expr) -> bool:
        base, divisor = congruence.args
        # Congruences that can be currently expressed with supported Dim ops are
        # of the form (x + a) % b == 0, where x is a Dim and a and b are constants.
        # This allows us to derive x as b*y - a for some Dim y.
        # (See also documentation of dynamic_shapes._DerivedDim.)
        if isinstance(base, sympy.Add):
            lhs, rhs = base.args
            cond = (
                isinstance(lhs, sympy.Symbol) and isinstance(rhs, sympy.Integer)
            ) or (isinstance(lhs, sympy.Integer) and isinstance(rhs, sympy.Symbol))
        else:
            cond = isinstance(base, sympy.Symbol)
        cond = cond and isinstance(divisor, sympy.Integer)
        return cond

    def forced_specializations(self) -> Dict[str, sympy.Expr]:
        """Returns a dictionary of the names of symbols to their specialized value"""

        def debug_name(src: Source) -> str:
            name = src.name()
            if self._dcp.source_name_to_debug_name:
                return f"{self._dcp.source_name_to_debug_name[name]} = {name}"
            else:
                return name

        return {
            debug_name(self._dcp.symbol_to_source[s][0]): val
            for s, val in self._substitutions.items()
            if s in self._marked_dynamic
        }

    def _is_derived_dim(
        self, dim: object
    ) -> TypeGuard[torch.export.dynamic_shapes._DerivedDim]:
        return isinstance(dim, torch.export.dynamic_shapes._DerivedDim)

    def _is_dim(self, dim: object) -> TypeGuard[torch.export.dynamic_shapes._Dim]:
        return isinstance(dim, torch.export.dynamic_shapes._Dim) and not isinstance(
            dim, torch.export.dynamic_shapes._DerivedDim
        )

    def _process_derived_dim_roots(
        self,
        results: Dict[str, Dict[str, Any]],
        name_to_dim: Dict[str, Any],
    ) -> None:
        """
        Here we resolve 2 concerns with derived dims suggested fixes: 1) newly introduced roots,
        and 2) root swapping.

        1) Newly introduced roots appear with modulo guards, e.g. Mod(dx, 2) = 0 suggests
        dx is a derived dim equal to 2 * _dx, introducing a new root _dx. Currently the final
        suggested fixes handle this correctly, but we can get intermediate results that look like
        {"dy": {"eq": "dx + 1"}, "dx": {"eq": "2 * _dx + 1, "min": 3, "max": 15}}
        and this routine prettifies this by unifying to a single root, and making each suggestion
        either a derived dim or min/max range, not both.

        2) With suggested fixes for derived dims, roots can be swapped,
        e.g. dx, dx - 1 -> dy + 1, dy. Here we don't want to print out the attached name,
        since this leads to messages like "dx - 1 = Dim("dx - 1", ...)".
        Instead we evaluate the new root value, and remove results for its derivations.

        First we find all the original roots (specified in dynamic_shapes), that are found in the
        values of results (i.e. used for computing suggesting fix values). These original roots
        (suppose `dx`) are either specialized, unchanged, refined, or swapped
        (expressed as a derived dim). If any of the first 3 cases happen, we suggest `dx`'s value
        in results, and remove suggestions for derivations of `dx`, assuming the derived relation
        is valid. If swapped, we find the new root, and use the fix to evaluate `dx`'s new value,
        and then do the same with `dx`'s derivations.

        Assuming the originally specified derived relations are correct is valid, because:
            1) if the relations are plain wrong (e.g. input shape = (6, 4) with spec (dx, dx - 1))
               produce_guards() will catch this and crash before hand.
            2) if the relations are numerically correct but do not match the emitted guard,
               for example:

                    def forward(self, x, y):
                        return x.reshape([-1]) + y  # guard: s0 * 2 = s1
                    inputs = (torch.randn(6, 2), torch.randn(12))
                    dx = Dim("dx", min=2, max=32)
                    dynamic_shapes={"x": (dx, 2), "y": (dx + 6, )}  # this matches values but not op

               then this leads to 2 linear equations, and a) produce_guards() is able to solve for
               the unique solution of dx = 6 and specialize, and b) the export constraint solver will
               raise an issue due to range constraints (a unique solution means not all values in a
               range satisfy a guard) and also force specializations.
        """
        from torch.export.dynamic_shapes import Dim

        def _check_same_range(c: Mapping[str, int], dim: object) -> bool:
            # returns True if c & dim are both min/max ranges with same values
            return (
                self._is_dim(dim)
                and ("min" in c or "max" in c)
                and (
                    (dim.min < 2 and c.get("min", 2) == 2) or dim.min == c.get("min", 2)  # type: ignore[attr-defined]
                )  # let pass if analysis min = 2 and specified min = 0/1
                and dim.max == c.get("max", int_oo)  # type: ignore[attr-defined]
            )

        # 1) newly introduced roots
        # this part we handle adding newly introduced roots
        # these arise from guards like "x.shape[0] % 3 == 0"
        # leading to suggested fixes like "dx = 3*_dx"
        # extract _dx, and find appropriate min/max values
        #
        # before, we have something like:
        # {"dx": {"eq": 3*_dx+1, "min": 4, "max": 10}, "dy": dx+1, "dz": dx+2}
        # we want instead:
        # {"_dx": {"min": 1, "max": 4}, "dx": 3*_dx+1, "dy": 3*_dx+2, "dz": 3*_dx+3}
        introduced_roots: Dict[str, str] = {}  # map new root -> old root
        for k, c in list(results.items()):
            if "eq" in c and isinstance(c["eq"], sympy.Expr):  # derived dim
                root = next(iter(c["eq"].free_symbols))
                if str(root) not in name_to_dim:
                    introduced_roots[str(root)] = k
                    # calculate necessary min & max
                    modulus, remainder = sympy.polys.polytools.div(c["eq"], root)
                    c_min = c.get("min", 2)
                    min_ = math.ceil((c_min - remainder) / modulus)
                    c_max = c.get("max", int_oo)
                    max_ = math.floor((c_max - remainder) / modulus)
                    # create result & dim
                    results[str(root)] = {"min": min_, "max": max_}
                    name_to_dim[str(root)] = Dim(str(root), min=min_, max=max_)
                    # remove old root min/max bounds
                    c.pop("min", None)
                    c.pop("max", None)

        # alter derivations that depend on old root, to unify to new root
        # e.g. dx=3*_dx+1, dy=dx+1 -> dy=3*_dx+2
        for old_root in introduced_roots.values():
            for k, c in list(results.items()):
                if (
                    "eq" in c
                    and isinstance(c["eq"], sympy.Expr)
                    and str(symbol := next(iter(c["eq"].free_symbols))) == old_root
                ):  # derived dim with root = old_root
                    new_root_expr = results[str(old_root)]["eq"]  # dx=3*_dx+1
                    new_expr = c["eq"].subs({symbol: new_root_expr})  # dy=(3*_dx+1)+1
                    c["eq"] = new_expr

        # 2) root swapping
        # collect all the original roots that are used for calculating values of suggested fixes
        # this consists of:
        # 1) {"dx": {"min": ..., "max": ...}} -> dx: refined root dim
        # 2) {"dy": "dx + 1"} -> dx: root for suggested fix
        modified_roots: Set[str] = set()
        for k, c in results.items():
            if k not in name_to_dim:  # _dynamo.export() may handle source directly
                continue
            if self._is_dim(name_to_dim[k]) and ("min" in c or "max" in c):  # case 1)
                modified_roots.add(k)
            elif "eq" in c and isinstance(c["eq"], sympy.Expr):  # case 2)
                root = next(iter(c["eq"].free_symbols))
                assert root is not None
                modified_roots.add(str(root))

        # exclude newly introduced roots, we've already processed these
        modified_roots = modified_roots.difference(introduced_roots)

        # evaluate the new value for each root
        # this is now either 1) unchanged, 2) refined with a new range,
        # or 3) specialized to a concrete value
        modified_root_values: Dict[str, Dict[str, Any]] = {}
        for mroot in modified_roots:
            swapped_root = True
            if mroot in results:
                c = results[mroot]
                if ("min" in c or "max" in c) or isinstance(  # range
                    c["eq"], int
                ):  # specialized
                    # here, the original root is a root Dim or concrete value in results.
                    # if it is a derived dim, it is swapped, and we handle that below.
                    if not _check_same_range(
                        c, name_to_dim[mroot]
                    ):  # ignore if unchanged
                        modified_root_values[mroot] = c
                    swapped_root = False

            if swapped_root:
                # if the original root has been swapped in results, that means the new root
                # is a range (if it had specialized, the original root would have too).
                # find this new root, and solve for the original root's range.
                for k, c in results.items():
                    if k not in name_to_dim:
                        continue
                    dim = name_to_dim[k]
                    if (
                        dim.__class__.__name__ == "_DerivedDim"
                        and dim.root.__name__ == mroot
                    ):
                        # only look for min/max root, otherwise root would have specialized
                        if "min" in c or "max" in c:
                            expr = sympy.sympify(k)
                            s = next(iter(expr.free_symbols))
                            result = {
                                "min": try_solve(sympy.Eq(expr, c["min"]), s)[1],  # type: ignore[arg-type, index]
                                "max": try_solve(sympy.Eq(expr, c["max"]), s)[1],  # type: ignore[arg-type, index]
                            }
                            if not _check_same_range(
                                result, name_to_dim[mroot]  # type: ignore[index, arg-type]
                            ):  # ignore if unchanged
                                modified_root_values[mroot] = result  # type: ignore[index]
                                break

        # filter out results where the key is a derived dim (e.g. {"dx - 1" : 4})
        # we only want to suggest fixes for the root, to avoid derived names.
        # also, remove anything in modified_roots, since we either add new modified values after this,
        # or have decided they are unchanged.
        for k in list(results.keys()):
            if k not in name_to_dim:
                continue
            if self._is_derived_dim(name_to_dim[k]) or k in modified_roots:
                del results[k]

        # update results with modified root values
        # now results has the following properties:
        # - only contains original roots as keys
        # - each root is now either specialized, refined, or derived from another original root
        results.update(modified_root_values)

    def prettify_results(
        self,
        original_signature: inspect.Signature,
        dynamic_shapes: Union[Dict[str, Any], Tuple[Any], List[Any]],
        constraint_violation_error: object,
        forced_specializations: Dict[str, str],
    ) -> str:
        """Format a message for constraint violation erros"""
        from torch.export.dynamic_shapes import _get_dim_name_mapping

        if not self._dcp.source_name_to_debug_name:
            # nothing to do
            return ""

        def transform(s: str, inverse: bool = False) -> str:
            for k, v in self._dcp.source_name_to_debug_name.items():
                s = s.replace(k, v) if not inverse else s.replace(v, k)
            return s

        results: DefaultDict[str, Dict[str, Any]] = defaultdict(dict)
        if dynamic_shapes is None:
            dynamic_shapes = {}

        def flip(op: str) -> str:
            if op == "<=":
                return ">="
            if op == ">=":
                return "<="
            if op == "<":
                return ">"
            if op == ">":
                return "<"
            assert op == "=="
            return op

        def relation_with_digit(expr: str, op: str, digit: int) -> None:
            if op == "<=":
                results[expr]["max"] = digit
            elif op == "<":
                results[expr]["max"] = digit - 1
            elif op == ">=":
                results[expr]["min"] = digit
            elif op == ">":
                results[expr]["min"] = digit + 1
            else:
                assert op == "=="
                results[expr]["eq"] = digit

        # retrieve dynamic shapes
        name_to_dim = _get_dim_name_mapping(dynamic_shapes)

        for s in self._static_results.union(self._dynamic_results):
            t = transform(s)
            if t == s:
                continue
            left, op, right = re.split(r"( == | <= | >= | < | > )", t)
            op = op.strip()
            if op == "==" and left == right:
                continue
            if right.isdigit():
                relation_with_digit(left, op, int(right))
            elif left.isdigit():
                relation_with_digit(right, flip(op), int(left))
            else:
                assert op == "==", t
                try:
                    results[left]["eq"] = sympy.sympify(right)
                except TypeError as e:  # rhs source is not linked to Dim name
                    pass

        # order forced specializations based on name
        forced_specializations = {
            k: forced_specializations[k]
            for k in sorted(
                forced_specializations.keys(),
                key=lambda x: x.split(" = ")[1],
            )
        }

        buf = ""
        if forced_specializations:
            debug_names = set()
            for k in forced_specializations:
                dim = name_to_dim[k.split(" = ")[0]]
                if self._is_derived_dim(dim):
                    debug_names.add(dim.root.__name__)  # type: ignore[attr-defined]
                else:
                    debug_names.add(dim.__name__)

            buf += (
                f"Specializations unexpectedly required ({', '.join(sorted(debug_names))})! "
                'For more information, run with TORCH_LOGS="+dynamic".\n'
            )
            for s, val in forced_specializations.items():
                buf += f"  - solving the guards generated for {s} resulted in a specialized value of {val}.\n"

        self._process_derived_dim_roots(results, name_to_dim)

        dims = []
        others = []

        # order results by source name
        results2 = {
            k: results[k]
            for k in sorted(
                results.keys(),
                key=lambda x: transform(x, inverse=True),
            )
        }
        for k, c in results2.items():
            if "eq" in c:
                other = c["eq"]
                if isinstance(other, int):
                    others.append(f"{k} = {other}")
                elif _is_supported_equivalence(other):
                    others.append(f"{k} = {other}")
            else:
                min_ = c.get("min", None)
                if min_ == 2:
                    min_ = None
                max_ = c.get("max", None)
                if min_ is not None and max_ is not None:
                    dims.append(f"{k} = Dim('{k}', min={min_}, max={max_})")
                elif min_ is not None:
                    dims.append(f"{k} = Dim('{k}', min={min_})")
                elif max_ is not None:
                    dims.append(f"{k} = Dim('{k}', max={max_})")
                else:
                    dims.append(f"{k} = Dim('{k}')")

        # results2 will get filtered out if no new suggestions,
        # this can happen if guards are too complex.
        # in that case don't suggest fix
        if dims or others:
            buf += "\nSuggested fixes:\n  "
            buf += "\n  ".join(dims + others)

        return buf


TLS = threading.local()


@dataclass(frozen=True)
class ShapeEnvSettings:
    """
    Encapsulates all shape env settings that could potentially affect
    FakeTensor dispatch. Used when creating dispatch cache keys.
    """

    allow_scalar_outputs: bool
    allow_dynamic_output_shape_ops: bool
    assume_static_by_default: bool
    specialize_zero_one: bool
    duck_shape: bool
    prefer_deferred_runtime_asserts_over_guards: bool
    allow_complex_guards_as_runtime_asserts: bool


@dataclass
class ValueRangesSLoc:
    """
    Locations of the guards that triggered lower and upper bound.
    """

    lower: SLoc
    upper: SLoc


@contextmanager
def _suppress_guards(shape_env: ShapeEnv) -> Iterator[None]:
    shape_env._suppress_guards_enter()
    try:
        yield
    finally:
        shape_env._suppress_guards_exit()


class ShapeEnv:
    # This is a wrapper over the actual __init__ function.
    #
    # Where to add a new constructor parameter to ShapeEnv?
    # =====================================================
    # This __init__ function should be used only for parameters related to event recording.
    # These are parameters that we don't wish to pass down the road to new ShapeEnv instances
    # created from replaying events.
    #
    # If you wish to add a parameter to the constructor of ShapeEnv, unrelated to event
    # recording, do so in the _init function.
    def __init__(
        self,
        *,
        should_record_events: Optional[bool] = None,
        tracked_fakes: Optional[List[Any]] = None,
        **kwargs: Any,
    ) -> None:
        self._init(**kwargs)

        # Disable event recording when replaying.
        kwargs["should_record_events"] = False

        from torch.fx.experimental.validator import translation_validation_enabled

        self._translation_validation_enabled = translation_validation_enabled()

        # If not specified, enable event recording if both:
        #   - Translation validation is on
        #   - Translation validation bisection is not disabled
        self.should_record_events = (
            should_record_events
            if should_record_events is not None
            else (
                self._translation_validation_enabled
                and not config.translation_validation_no_bisect
            )
        )

        # Enable event recording check if both:
        #   - It should record events
        #   - The recording check is enabled
        self.check_recorded_events = (
            self.should_record_events and config.check_shape_env_recorded_events
        )

        # This will make sure we only record the top-level function call.
        self.is_recording = False
        # Keep track of the list of tracked fakes.
        self.tracked_fakes = tracked_fakes
        # List of events for reconstructing ShapeEnv at arbitrary points in time.
        self.events: List[ShapeEnvEvent] = (
            [ShapeEnvEvent(ShapeEnv, kwargs=kwargs)]
            if self.should_record_events
            else []
        )

        # FakeTensor per-ShapeEnv operation cache. This is used for caching
        # operations that contain symbolic shapes which have guards on the
        # ShapeEnv (so are ShapeEnv-dependent).
        #
        # NOTE: It's important that SymNodes in this cache have their ShapeEnv
        # stripped otherwise you end up with cycles which can only be cleaned
        # with the GC.
        self.fake_tensor_cache: Dict[
            torch._subclasses.fake_tensor._DispatchCacheKey,
            torch._subclasses.fake_tensor._DispatchCacheEntry,
        ] = {}

    # Pro-tip: if you add new field to ShapeEnv, this affects some accept
    # tests.  Accept their output with:
    #
    #   EXPECTTEST_ACCEPT=1 python test/dynamo/test_dynamic_shapes.py -k test_shape_env_equal
    #
    def _init(
        self,
        *,
        allow_scalar_outputs: bool = True,
        allow_dynamic_output_shape_ops: bool = True,
        # NB: These are legacy configuration that help us make good choices
        # when the constraint/dynamic dims are not explicitly passed to us.
        # Ideally we will fix all call sites to be explicit and not have
        # implicit choices, but this apparently was pretty involved.
        assume_static_by_default: bool = False,
        # Note - On 0/1 specialization
        #
        # The following options affect decisions we make about eager
        # specialization.  Disabling them will increase trace time (as we do
        # more symbolic reasoning) and can also harm the quality of generated
        # code (because inductor may not be able to specialize for bounds
        # being equal--although if we later respecialize because of a guard,
        # your code may be just as good as it was before.)
        #
        # When True, eagerly specialize input sizes which have 0/1.
        specialize_zero_one: bool = True,
        # When True, assume input sizes which have the same size are
        # symbolically equal.
        duck_shape: Optional[bool] = None,
        # For debugging
        co_fields: Optional[Dict[str, str]] = None,
        # When True, whenever safe, we will generate a deferred runtime assert
        # instead of a guard whenever we know that an expression must be True,
        # otherwise it would be an error, even for backed SymInts (where we
        # could ostensibly unconditionally generate guards).  This is useful
        # for export, where preventing "error checking" sizes from showing up
        # in guards is helpful, since these guards in some sense are overly
        # pedantic.  See also https://github.com/pytorch/pytorch/issues/121749
        prefer_deferred_runtime_asserts_over_guards: bool = False,
        # When True, does not emit or raise constraint violation errors on
        # implicit guards generated by ops, and defers to runtime assertions
        # in the graph instead. For export.
        allow_complex_guards_as_runtime_asserts: bool = False,
        # XXX Add any new settings that could affect FakeTensor evaluation
        # to: torch._subclasses.fake_tensor._ShapeEnvSettings
    ) -> None:
        if duck_shape is None:
            duck_shape = config.use_duck_shape

        self.settings = ShapeEnvSettings(
            # Not directly used by ShapeEnv; indirectly used by FakeTensor
            allow_scalar_outputs=allow_scalar_outputs,
            allow_dynamic_output_shape_ops=allow_dynamic_output_shape_ops,
            # End
            assume_static_by_default=assume_static_by_default,
            specialize_zero_one=specialize_zero_one,
            duck_shape=duck_shape,
            prefer_deferred_runtime_asserts_over_guards=prefer_deferred_runtime_asserts_over_guards,
            allow_complex_guards_as_runtime_asserts=allow_complex_guards_as_runtime_asserts,
        )

        self.guards: List[ShapeGuard] = []
        self.axioms: Dict[sympy.Expr, sympy.Expr] = {}
        # Maps symbolic ints to their original concrete values
        # Currently populated from tensors
        self.var_to_val: Dict[sympy.Symbol, sympy.Integer] = {}
        # Like var_to_val, but only set when propagate_real_tensors is on.
        # Used as last resort to avoid GuardOnDataDependent error
        self.unbacked_var_to_val: Dict[sympy.Symbol, sympy.Integer] = {}
        # Like above, but used exclusively for OBLIVIOUS_SIZE.  These
        # potentially could be put together but I am not sure, writing out
        # the logic individually before abstracting.
        self.oblivious_var_to_val: Dict[sympy.Symbol, sympy.Integer] = {}
        # Maps symbolic ints to their min/max range.  These ranges
        # are conservative: the int MUST fall in the range, but the
        # range may contain ints which may not actually appear in
        # practice
        self.var_to_range: Dict[sympy.Symbol, ValueRanges] = {}
        self.var_to_range_sloc: Dict[sympy.Symbol, ValueRangesSLoc] = {}
        self.source_name_to_debug_name: Dict[str, str] = {}
        self.var_to_sources: Dict[sympy.Symbol, List[Source]] = {}
        self.var_to_stack: Dict[sympy.Symbol, CapturedTraceback] = {}
        # Maps a source to the *original* symbol that was assigned to it
        self.source_to_var: Dict[str, sympy.Symbol] = {}
        # Maps from sympy ints to expressions representing them
        # Populated from equality guards (i.e. a.shape[0] == b.shape[0])
        self.replacements: Dict[sympy.Symbol, sympy.Expr] = {}
        # The sloc of the guard that triggered this replacement to be added
        self.replacements_slocs: Dict[sympy.Symbol, SLoc] = {}
        self.unbacked_renamings: Dict[sympy.Symbol, sympy.Symbol] = {}
        # Set holds a % b expressions that evaluate to 0.
        self.divisible: Set[sympy.Expr] = set()
        # Set that holds "size-like" symbols.  When we perform
        # "size-oblivious" tests, these can be assumed to be >= 2.
        self.size_like: Set[sympy.Symbol] = set()
        # Duck-shaping says that if two input tensors have the same size,
        # they get assigned the same symbolic variable
        self.val_to_var: Dict[int, sympy.Symbol] = {}
        if specialize_zero_one:
            self.val_to_var = {0: sympy.S.Zero, 1: sympy.S.One}
        self.unbacked_symfloat_counter = itertools.count()
        self.unbacked_symint_counter = itertools.count()
        # Similar to guards, but these MUST evaluate to true and can
        # only be evaluated at runtime midway through (i.e., they always
        # involve unbacked symints)
        #
        # For efficiency reasons, we index in the following way.  Suppose you have
        # a runtime assert i0 + i1 <= s1.  We pick the most recently allocated
        # symbol in the source expression and add the assert to the list for
        # that symbol e.g., {i1: [i0 + i1 <= s1]}.
        #
        # We access the runtime asserts in two situations:
        #
        #   - When we are guarding on an expression, we will attempt to
        #     statically evaluate it, in case the unbacked SymInts can
        #     simplify away.  If we have a runtime assert, we may be able
        #     to discharge the guard entirely.  We only need to attempt
        #     runtime asserts that mention freevars of the expression in
        #     question.
        #
        #   - When we are performing codegen (in Inductor for eager, or
        #     when finalizing the export FX graph), we need to know what
        #     extra runtime asserts to insert.  Whenever an unbacked
        #     SymInt comes into scope, all runtime asserts involving it
        #     become eligible for insertion (so long as all of their other
        #     free unbacked symbols are also in scope).  We technically
        #     can handle any choice of key by kicking inexpressible asserts
        #     to the next unbacked symbol to wait on, but if we choose the
        #     latest key, an assert will only show up at the moment when
        #     we can actually codegen it.
        self.deferred_runtime_asserts: Dict[
            Optional[sympy.Symbol], List[RuntimeAssert]
        ] = {}
        # This exists so we can efficiently invalidate the cache (it's used as
        # part of the cache key); otherwise we'd have to iterate through
        # deferred_runtime_asserts to compute its length
        self.num_deferred_runtime_asserts = 0
        self.log = log
        self.log.info("create_env")
        self.frozen = False
        self.runtime_asserts_frozen = False
        self.dim_constraints: Optional[DimConstraints] = None
        self.counter: Counter[str] = collections.Counter()
        # Mapping from sympy.Symbol to the number of guards which mention this
        # symbol
        self.symbol_guard_counter: Counter[sympy.Symbol] = collections.Counter()
        # A selection of important fields on co_field; solely used for
        # signpost_event
        self.co_fields = co_fields if co_fields else {}

        # Whenever we allocate a fresh unbacked Symbol, we add it to this
        # pending list.  Unbacked symbol allocation can occur at unpredictable
        # points during meta tensor propagation, but at some point, we
        # have to know what the binding site for an unbacked symbol is, and
        # this is computed when we actually place the node in the graph. The
        # important thing is that we always actually handle every unaccounted
        # for unbacked symbol, so this list helps us keep track of them and
        # then make sure they are all accounted for.
        #
        # We could potentially give rise to errors earlier by lexically
        # scoping when we do propagation, and only allowing unbacked symbols
        # to be allocated at this point in time.  However this is inconvenient
        # to do in Dynamo, because fake tensor propagation is far from when we
        # analyze binding sites (set_example_value), so we do it in a more
        # mutatey way.
        #
        # NB: fresh unbacked symbols NEVER get substitutions applied to them,
        # they are binding sites!
        self.pending_fresh_unbacked_symbols: List[sympy.Symbol] = []

        # Version counter used to invalidate cached values
        self._prev_cache_key = self._get_key()
        self._version_counter = 0

        # Each time divisible is changed this should be set to True, this is set in _update_version_counter.
        self._resimplify_floor_div_axioms = True

        # Cache for FX nodes.
        # Maps an already built node a tuple of:
        #   1. node's target
        #   2. list of arguments
        # This drastically reduces the size of the FX graph, avoiding
        # duplicated nodes.
        self.fx_node_cache: Dict[Tuple[Callable, Tuple[Any, ...]], torch.fx.Node] = {}
        self.source_to_symbol: Dict[str, sympy.Symbol] = {}

        # Suppose you want to replace an unbacked symbol with another
        # unbacked symbol.  This is error prone because you can cause
        # references to unbacked symbols to time travel backwards.  E.g.,
        #
        # u1 = x.item()
        # ... use of u1 ...
        # u2 = y.item()
        # u3 = z.item()
        # torch._check(u1 == u2 + u3)
        #
        # If you replace u1 with u2 + u3, then the use of u1 now
        # references u2 and u3 prior to them actually being bound at
        # runtime.
        #
        # To control for this, we track the order unbacked symbols
        # were allocated, and only allow substitutions if they respect
        # the dependency from this order; an unbacked symbol can only
        # be substituted with unbacked symbols that come before it in the
        # order.
        #
        # This also imposes an ordering on the unbacked symbol binding
        # sites themselves: you are not allowed to reorder unbacked symbol
        # bindings.  At the moment, this is not tracked, but we potentially
        # could track this at the IR level using a higher order operator
        # with something like effect token tracking.
        self.unbacked_alloc_order: Dict[sympy.Symbol, int] = {}

        from torch.fx.experimental.validator import translation_validation_enabled

        self._translation_validation_enabled = translation_validation_enabled()

        if self._translation_validation_enabled:
            from torch.fx.experimental.validator import TranslationValidator

            self.validator = TranslationValidator()
            self.graph = torch.fx.Graph()
            # Create an output graph and start inserting before that.
            # This is needed when 'deepcopy'-ing this object.
            self.graph.inserting_before(self.graph.output(None))

            # Mapping of each node name to the node itself.
            #
            # This is useful for matching an FX node from a recorded ShapeEnv.graph
            # to the FX node of the ShapeEnv we are running the event on.
            #
            # Whenever you add a node to self.graph, you must add a mapping to this
            # variable. Otherwise, the built FX graph on the replayed ShapeEnv will
            # not be valid.
            self.name_to_node: Dict[str, torch.fx.Node] = {}

    @property
    def allow_scalar_outputs(self) -> bool:
        return self.settings.allow_scalar_outputs

    @property
    def allow_dynamic_output_shape_ops(self) -> bool:
        return self.settings.allow_dynamic_output_shape_ops

    @property
    def assume_static_by_default(self) -> bool:
        return self.settings.assume_static_by_default

    @property
    def specialize_zero_one(self) -> bool:
        return self.settings.specialize_zero_one

    @property
    def duck_shape(self) -> bool:
        return self.settings.duck_shape

    @property
    def prefer_deferred_runtime_asserts_over_guards(self) -> bool:
        return self.settings.prefer_deferred_runtime_asserts_over_guards

    @property
    def allow_complex_guards_as_runtime_asserts(self) -> bool:
        return self.settings.allow_complex_guards_as_runtime_asserts

    def check_equal(self, other: ShapeEnv) -> None:
        """Compare another ShapeEnv for equivalence"""
        # ShapeEnv fields that are not relevant for the outcome of
        # ShapeEnv.produce_guards call:
        #   - Debugging variables
        #   - Translation validation related variables
        #   - Events recording related variables
        non_state_variable_names = (
            "counter",
            "log",
            "var_to_stack",
            "fx_node_cache",
            "graph",
            "validator",
            "check_recorded_events",
            "should_record_events",
            "is_recording",
            "tracked_fakes",
            "events",
            "source_name_to_debug_name",
            "_prev_cache_key",
            "_version_counter",
            "dim_constraints",
            # source locations are OK to diverge
            "var_to_range_sloc",
            "replacements_slocs",
            "_resimplify_floor_div_axioms",
        )

        # Mapping of the value of each to-be-compared field into the values that
        # should actually be compared.
        #
        # You should modify this if, for example, the field that holds state and
        # debugging information. e.g. ShapeGuard holds the actual guard (sympy.Expr)
        # and the stack when it was added to the set of guards. In order to compare
        # it, we throw away the stack information.
        def map_value(key: str, value: Any) -> Any:
            if key in ("unbacked_symfloat_counter", "unbacked_symint_counter"):
                from copy import copy

                # For itertools.count(), we compare the next integer returned
                # by the count iterators. Not that we need to copy the iterator
                # first. Otherwise we are mutating the object.
                return next(copy(value))
            elif key == "guards":
                # Transform the list of ShapeGuard into a list of expressions.
                return [g.expr for g in value]
            elif key == "deferred_runtime_asserts":
                # Transform the list of RuntimeAsserts into a list of expressions.
                return {s: [ra.expr for ra in ras] for s, ras in value.items()}
            elif key == "name_to_node":
                # Compare just the set of keys is the same.
                return set(value.keys())
            elif key in (
                "symbol_guard_counter",
                "pending_fresh_unbacked_symbols",
                "fake_tensor_cache",
            ):
                # Skip this for comparisons
                return None
            return value

        shape_env_check_state_equal(self, other, non_state_variable_names, map_value)

    def _snapshot_tracked_fakes(self) -> Optional[List[Any]]:
        if self.tracked_fakes is None:
            return None

        from torch._dynamo.variables.builder import TrackedFake

        def maybe_transform_fake(fake: TrackedFake) -> TrackedFake:
            inner_fake = (
                fake.fake
                if isinstance(fake.fake, (torch.SymInt, torch.SymFloat))
                else FakeTensorMeta.from_fake(fake.fake)
            )
            # Even though TrackedFake accepts either a Union[SymInt, FakeTensor], here we give it a
            # FakeTensorMeta for two reasons:
            #   1. this is all the information we need when recording ShapeEnvEvents.
            #   2. it works even if each TrackedFake changes its metadata.
            return TrackedFake(inner_fake, fake.source, fake.symbolic_context)  # type: ignore[arg-type]

        return [maybe_transform_fake(fake) for fake in self.tracked_fakes]

    def _last_event_index(self) -> int:
        return len(self.events) - 1

    @contextmanager
    def _recording(self) -> Iterator[None]:
        self.is_recording = True
        try:
            yield
        finally:
            self.is_recording = False

    @record_shapeenv_event()
    def _eliminate_unbacked(self, orig_s: sympy.Symbol, new_s: sympy.Expr) -> None:
        self._set_replacement(orig_s, new_s, "eliminate_unbacked")

    @record_shapeenv_event()
    def set_unbacked_var_to_val(self, k: sympy.Symbol, v: int) -> None:
        """Used only when propagate_real_tensors; registers a value for an
        unbacked symbol, which can be used last resort to resolve hints."""
        log.info("set_unbacked_var_to_val %s = %s", k, v)
        self.unbacked_var_to_val[k] = sympy.sympify(v)

    # Unlike set_replacement, this records a shapeenv event
    @record_shapeenv_event()
    def _rename_unbacked_to(self, orig_s: sympy.Symbol, new_s: sympy.Symbol) -> None:
        assert isinstance(orig_s, sympy.Symbol), orig_s
        assert isinstance(new_s, sympy.Symbol), new_s
        assert free_unbacked_symbols(new_s), new_s
        assert free_unbacked_symbols(orig_s), orig_s
        dest = self.replacements.get(orig_s)
        if dest is not None:
            assert not free_unbacked_symbols(dest), f"{orig_s} -> {dest}"
        self._set_replacement(orig_s, new_s, "rename_unbacked_to")
        self.unbacked_renamings[orig_s] = new_s
        if dest is not None:
            self._set_replacement(new_s, dest, "rename_unbacked_to_dest")

    @record_shapeenv_event()
    def _constrain_range_for_size(
        self, a: sympy.Symbol, min: Optional[int] = None, max: Optional[int] = None
    ) -> None:
        if min is None:
            min = 0
        if max is None:
            max = int_oo

        if max < min:
            raise ValueError(
                "Maximum value to constrain_as_size can't be less than the specified min value, "
                "received min={min} and max={max}"
            )

        self.constrain_symbol_range(
            a,
            compiler_min=min,
            compiler_max=max,
        )
        self.size_like.add(a)

    @record_shapeenv_event()
    def _constrain_range(self, a: sympy.Expr, min: int, max: int) -> None:
        if isinstance(a, sympy.Integer):
            if not (min <= int(a) <= max):
                raise ValueRangeError(f"Invalid value {int(a)} for range [{min}:{max}]")
            return

        # TODO: Shouldn't we install a guard if the symbol is backed?  Or is the
        # semantics that this is an "unchecked" assert (but it this actually
        # something useful?  Might be better to restrict only for unbacked
        # SymInt).
        if isinstance(a, sympy.Symbol):
            self.constrain_symbol_range(
                a,
                compiler_min=min,
                compiler_max=max,
            )

    @record_shapeenv_event()
    def _constrain_unify(self, a: SymInt, b: SymInt) -> None:
        """
        Given two SymInts, constrain them so that they must be equal.  NB:
        this will not work with SymInts that represent nontrivial expressions
        (yet!)
        """
        # TODO: this does not install a deferred runtime assert yet

        # TODO: Maybe dedupe this with _maybe_guard_rel?
        # Update Feb 2024: this is extra important to do, this doesn't handle
        # unbacked replacements properly nor does it generate deferred runtime
        # asserts
        if not isinstance(a, SymInt):
            if not isinstance(b, SymInt):
                assert a == b
            else:
                assert isinstance(
                    b.node.expr, sympy.Symbol
                ), "constraining non-Symbols NYI"
                assert b.node.shape_env is self
                self.replacements[b.node.expr] = sympy.Integer(a)
        else:
            # TODO: Actually, we can support this as long as one of them is a symbol.
            # NB: We can't actually do "unification" as our operators are not
            # injective
            assert isinstance(a.node.expr, sympy.Symbol), "constraining non-Symbols NYI"
            assert a.node.shape_env is self
            if not isinstance(b, SymInt):
                self.replacements[a.node.expr] = sympy.Integer(b)
            else:
                assert a.node.shape_env is b.node.shape_env
                assert isinstance(
                    b.node.expr, sympy.Symbol
                ), "constraining non-Symbols NYI"
                new_var = self._find(a.node.expr)
                self.replacements[b.node.expr] = new_var

    def _ignore_fresh_unbacked_symbols_tls(self) -> bool:
        return getattr(TLS, "ignore_fresh_unbacked_symbols", False)

    @record_shapeenv_event()
    def _ignore_fresh_unbacked_symbols_set(self, b: bool) -> bool:
        prev = self._ignore_fresh_unbacked_symbols_tls()
        TLS.ignore_fresh_unbacked_symbols = b
        return prev

    @contextmanager
    def ignore_fresh_unbacked_symbols(self) -> Iterator[None]:
        """
        Indicates that the newly allocated unbacked SymInts are being
        discarded
        """
        prev = self._ignore_fresh_unbacked_symbols_set(True)
        try:
            yield
        finally:
            self._ignore_fresh_unbacked_symbols_set(prev)

    @record_shapeenv_event()
    def freeze(self) -> None:
        """Freeze this ShapeEnv to stop accumulating guards

        A frozen ShapeEnv will ignore any further guards generated on it and
        only emit a warning which may lead to accuracy problems.
        """
        self.frozen = True

    @record_shapeenv_event()
    def freeze_runtime_asserts(self) -> None:
        """Freeze this ShapeEnv to stop adding deferred runtime asserts.

        We will error if you try to install a new runtime assert when it is
        frozen.  This would indicate a lowering violation, or perhaps something
        we know statically is already True but we are checking it again in a way
        that is not clearly dischargeable.
        """
        # self.prefer_deferred_runtime_asserts_over_guards = False
        self.runtime_asserts_frozen = True

    def _create_symbol_for_source(self, source: Source) -> Optional[sympy.Symbol]:
        if not self._translation_validation_enabled:
            return None
        srcname = source.name()
        if source not in self.source_to_symbol:
            self.source_to_symbol[srcname] = sympy.Symbol(srcname, integer=True)
        return self.source_to_symbol[srcname]

    def _add_z3var(self, symbol: sympy.Symbol, type: Type) -> None:
        if self._translation_validation_enabled:
            self.validator.add_var(symbol, type)

    def _add_target_expr(self, expr: SympyBoolean) -> None:
        if self._translation_validation_enabled:
            self.validator.add_target_expr(expr)

    def _add_assertion(self, expr: SympyBoolean) -> None:
        if self._translation_validation_enabled:
            self.validator.add_assertion(expr)

    def _check_translation_validate(self) -> None:
        if self._translation_validation_enabled:
            self.validator.validate()

    @record_shapeenv_event()
    def _create_fx_call_function(
        self,
        op: Callable,
        args: Tuple,
    ) -> Tuple[Optional[torch.fx.Node], bool]:
        # Cache this tuple in order to avoid duplicated nodes.
        node_key = (op, args)
        # Flags whether the returned node was cached or not.
        fresh = False

        if self._translation_validation_enabled and node_key not in self.fx_node_cache:
            # Presence of None in the arguments implies that we should ignore this operation.
            if any(a is None for a in args):
                # We check if we are not mixing SymNode that should not be ignored
                # (fx_node is not None) with those that should (fx_node is None).
                assert all(not isinstance(a, torch.fx.Node) for a in args)
                return None, fresh

            fresh = True

            # If translation validation is enabled, all arguments must have its
            # own FX node.
            assert all(
                a is not None for a in args
            ), f"missing arg in FX graph ({op.__name__}): {args}"
            node = self.fx_node_cache[node_key] = self.graph.call_function(op, args)
            self.name_to_node[node.name] = node

        return self.fx_node_cache.get(node_key, None), fresh

    def _create_fx_placeholder_and_z3var(
        self,
        symbol: sympy.Symbol,
        type: Type,
    ) -> Optional[torch.fx.Node]:
        if not self._translation_validation_enabled:
            return None

        node_key = (self.graph.placeholder, (symbol,))

        # Check if we haven't added this symbol already.
        # If so, skip the placeholder creation, as it
        # generates invalid Python code.
        if node_key not in self.fx_node_cache:
            # Add a Z3 variable according to 'type'.
            self._add_z3var(symbol, type)
            # Create the FX placeholder out of a mangled name.
            mangled_name = re.sub(
                r"[^a-zA-Z0-9]", "_", re.sub(r"[()]", "", symbol.name)
            )
            node = self.fx_node_cache[node_key] = self.graph.placeholder(mangled_name)
            self.name_to_node[node.name] = node
            # Attach the 'symbol' to the placeholder so that we can retrieve
            # the Z3 variable later.
            node.meta["symbol"] = symbol

        return self.fx_node_cache[node_key]

    def _remove_fx_node(self, node: Optional[torch.fx.Node]) -> None:
        if self._translation_validation_enabled and node is not None:
            self.name_to_node.pop(node.name)
            self.graph.erase_node(node)

    def _add_fx_node_metadata(self, node: torch.fx.Node) -> None:
        from torch._dynamo.utils import get_current_node

        if self.should_record_events:
            node.meta[SHAPEENV_EVENT_KEY] = self._last_event_index()
            node.meta[CURRENT_NODE_KEY] = get_current_node()

    def _suppress_guards_tls(self) -> bool:
        return getattr(TLS, "suppress_guards", False)

    @record_shapeenv_event()
    def _suppress_guards_enter(self) -> None:
        if not hasattr(TLS, "suppress_guards_stack"):
            TLS.suppress_guards_stack = []
        old = self._suppress_guards_tls()
        TLS.suppress_guards_stack.append(old)
        TLS.suppress_guards = True

    @record_shapeenv_event()
    def _suppress_guards_exit(self) -> None:
        old = (
            TLS.suppress_guards_stack.pop()
            if len(TLS.suppress_guards_stack) > 0
            else False
        )
        TLS.suppress_guards = old

    def suppress_guards(self) -> _GeneratorContextManager[None]:
        """Context manager to ignore all guards generated inside"""
        return _suppress_guards(self)

    def _get_key(self) -> Tuple[int, int, int, int]:
        """
        Defines the current "state" of the guards we've accumulated in this ShapeEnv.
        Determines when we need to invalidate our cache
        """
        return (
            len(self.replacements),
            len(self.divisible),
            self.num_deferred_runtime_asserts,
            len(self.unbacked_var_to_val),
        )

    def _update_version_counter(self) -> None:
        # if the change to shape env effects self.divisible set
        # _resimplify_floor_div_axioms.
        # This is used to trigger a resimplication of FloorDiv to CleanDivs
        # in implication inside the function resimplify_floor_div.
        if len(self.divisible) != self._prev_cache_key[1]:
            self._resimplify_floor_div_axioms = True

        # The shape environment is queried orders of magnitude more often than
        # it is changed, so we summarise the cache key into a linearly
        # increasing version counter which is cheaper to check in _lru_cache

        # Only update version counter if the state actually changed
        cur_key = self._get_key()

        if self._prev_cache_key != cur_key:
            self._prev_cache_key = cur_key
            self._version_counter += 1

    def _produce_dyn_sizes(
        self,
        ex_size: Sequence[Union[int, SymInt]],
        source: Source,
        symbolic_context: SymbolicContext,
    ) -> List[sympy.Expr]:
        return self._produce_dyn_sizes_from_int_tuple(
            tuple(ex_size), source, symbolic_context
        )

    def _produce_dyn_sizes_from_int_tuple(
        self,
        tensor_size: Sequence[Union[int, SymInt]],
        source: Source,
        symbolic_context: SymbolicContext,
    ) -> List[sympy.Expr]:
        assert all(
            not is_symbolic(val) for val in tensor_size
        ), f"Expect size to be a plain tuple of ints but got {tensor_size}"
        from torch._dynamo.source import TensorProperty, TensorPropertySource

        _assert_symbol_context(symbolic_context)
        dynamic_dims = symbolic_context.dynamic_sizes  # type: ignore[attr-defined]
        constraint_dims = symbolic_context.constraint_sizes  # type: ignore[attr-defined]
        size = []
        for i, val in enumerate(tensor_size):
            size.append(
                self.create_symbol(
                    val,
                    TensorPropertySource(source, TensorProperty.SIZE, i),
                    dynamic_dims[i],
                    constraint_dims[i],
                    symbolic_context=symbolic_context,
                )
            )
        return size

    def create_symbolic_sizes_strides_storage_offset(
        self,
        ex: torch.Tensor,
        source: Source,
        *,
        symbolic_context: Optional[SymbolicContext] = None,
    ) -> Tuple[
        Tuple[Union[int, SymInt], ...],
        Tuple[Union[int, SymInt], ...],
        Union[int, SymInt],
    ]:
        """
        Returns a list of symbolic sizes and strides for the given tensor.
        We try our best to express stride in terms of the sizes, so as to not
        introduce new symbolic variables.
        """

        ex_size = tuple(
            self._maybe_specialize_sym_int_with_hint(sz) for sz in ex.size()
        )
        ex_stride = tuple(
            self._maybe_specialize_sym_int_with_hint(sd) for sd in ex.stride()
        )
        ex_storage_offset = self._maybe_specialize_sym_int_with_hint(
            ex.storage_offset()
        )

        return self._create_symbolic_sizes_strides_storage_offset(
            ex_size,
            ex_stride,
            ex_storage_offset,
            [_is_dim_dynamic(ex, i) for i in range(ex.dim())],
            source,
            symbolic_context=symbolic_context,
        )

    # Dynamo may want to wrap FakeTensors with SymInt sizes up e.g. make_fx(opt_f(), tracing_mode="symbolic").
    # We create symbols in shape_env using the backed hints behind SymInt.

    # Case 1: when SymInt is backed, dynamo can proceed with FakeTensors that have concrete shape.
    # produce_guards will trigger specializations on the outer stuff

    # Case 2: when the SymInt is unbacked, we will throw an data dependent error in require_hint().
    #
    # It's probably good for now but it's important to note that this approach has implications for
    # the original shape_env when checking guards in different order.

    # Example:
    # ---------
    # Consider a function "opt_f" as shown below:

    # @torch.compile()
    # def opt_f(x: bool, y: Tensor):
    #   if x == True:
    #     return y + torch.randn([4])
    #   else:
    #     return y
    # Depending on the sequence of calls, we might install two different sets of guards:

    # 1. opt_f(False, y):
    #    - "x == False" (always works for any size y)

    # 2. opt_f(True, y):
    #    - Triggers recompilation and results in guards like:
    #      - "x == True and y.size(0) == 4"
    #      - (or "y.size(0) == 4 and x == True")

    # The order of checking the guards matters. In this specific example:
    # If True branch guard check precedes False branch and for True branch, y.size(0) check precedes x == True,
    # we may have an unnessary shape speciliazation for y.
    def _maybe_specialize_sym_int_with_hint(
        self, maybe_sym: Union[int, SymInt]
    ) -> Union[int, SymInt]:
        assert isinstance(maybe_sym, (int, torch.SymInt))
        if is_symbolic(maybe_sym):
            assert (
                maybe_sym.node.shape_env is not self
            ), "expect the symbol is created from an shape env other than current one."
            return maybe_sym.node.require_hint()
        return maybe_sym

    @record_shapeenv_event()
    def _create_symbolic_sizes_strides_storage_offset(
        self,
        # NB: SymInt is allowed here due to nested int, normally you don't
        # actually pass true symbolic sizes to this function
        ex_size: Sequence[Union[int, SymInt]],
        ex_stride: Sequence[Union[int, SymInt]],
        ex_storage_offset: Union[int, SymInt],
        is_dim_dynamic: Sequence[bool],
        source: Source,
        *,
        symbolic_context: Optional[SymbolicContext] = None,
    ) -> Tuple[
        Tuple[Union[int, SymInt], ...],
        Tuple[Union[int, SymInt], ...],
        Union[int, SymInt],
    ]:
        dim = len(ex_size)

        # Reimplement the legacy behavior
        if symbolic_context is None:
            constraint_sizes: List[DimConstraint] = [None] * dim
            constraint_strides: List[DimConstraint] = [None] * dim
            dynamic_dims = []
            dynamic_strides = []
            for i in range(dim):
                # NB: This is encapsulation breaking!  Legacy behavior was
                # bad.
                if is_dim_dynamic[i]:
                    r = DimDynamic.DYNAMIC
                elif self.assume_static_by_default:
                    r = DimDynamic.STATIC
                else:
                    r = DimDynamic.DUCK
                dynamic_dims.append(r)
                dynamic_strides.append(r)
            dynamic_dims = [DimDynamic.DUCK] * dim
            dynamic_strides = [DimDynamic.INFER_STRIDE] * dim
            # symbolic_context is None - set one
            symbolic_context = StatelessSymbolicContext(
                dynamic_sizes=dynamic_dims,
                dynamic_strides=dynamic_strides,
                constraint_sizes=constraint_sizes,
                constraint_strides=constraint_strides,
            )
        # We got a StatelessSymbolicContext
        _assert_symbol_context(symbolic_context)
        constraint_sizes = symbolic_context.constraint_sizes  # type: ignore[attr-defined]
        constraint_strides = symbolic_context.constraint_strides  # type: ignore[attr-defined]
        dynamic_sizes = symbolic_context.dynamic_sizes  # type: ignore[attr-defined]
        dynamic_strides = symbolic_context.dynamic_strides  # type: ignore[attr-defined]

        # TODO: make this configurable from outside symbolic_context; we made a symbolic_context
        # decision here where if all sizes are static, we are going to
        # specialize all of the inner strides/offset too. We don't have to
        # do this, and arguably we should ALWAYS allow for dynamic offset,
        # this is cheap.
        # TODO: This should be DYNAMIC, using DUCK for BC
        dynamic_offset = (
            DimDynamic.STATIC
            if all(r == DimDynamic.STATIC for r in dynamic_sizes)
            else DimDynamic.DUCK
        )
        are_sizes_static = all(r == DimDynamic.STATIC for r in dynamic_sizes)

        assert len(dynamic_sizes) == dim, f"{len(dynamic_sizes)} != {dim}"
        assert len(dynamic_strides) == dim, f"{len(dynamic_sizes)} != {dim}"
        assert len(constraint_sizes) == dim
        assert len(constraint_strides) == dim

        from torch._dynamo.source import TensorProperty, TensorPropertySource

        size: List[sympy.Expr] = self._produce_dyn_sizes_from_int_tuple(
            ex_size, source, symbolic_context
        )
        stride: List[Optional[sympy.Expr]] = [None] * len(size)
        for i, val in enumerate(ex_stride):
            if val in (0, 1):
                stride[i] = sympy.Integer(val)
        while any(x is None for x in stride):
            candidates = {
                ex_size[i] * ex_stride[i]: size[i] * stride[i]
                for i in range(len(size))
                if stride[i] is not None
            }

            # iterate over unbound strides in sorted order
            val_list = sorted(
                [(ex_stride[i], i) for i in range(len(stride)) if stride[i] is None],
                key=_nested_int_aware_sort,
            )
            for _, i in val_list:
                # Set stride to a candidate only for DimDynamic.INFER_STRIDE
                if (
                    stride[i] is None
                    and dynamic_strides[i] == DimDynamic.INFER_STRIDE
                    and ex_stride[i] in candidates
                ):
                    stride[i] = candidates[ex_stride[i]]
                    candidates[ex_size[i] * ex_stride[i]] = size[i] * stride[i]  # type: ignore[operator]

            if any(x is None for x in stride):
                # bind the smallest unbound stride to a new variable
                val, i = min(
                    [
                        (ex_stride[i], i)
                        for i in range(len(stride))
                        if stride[i] is None
                    ],
                    key=_nested_int_aware_sort,
                )
                # Set INFER_STRIDE to STATIC or DUCK depending on sizes
                dyn_stride = dynamic_strides[i]
                if dynamic_strides[i] == DimDynamic.INFER_STRIDE:
                    dyn_stride = (
                        DimDynamic.STATIC if are_sizes_static else DimDynamic.DUCK
                    )
                stride[i] = self.create_symbol(
                    val,
                    TensorPropertySource(source, TensorProperty.STRIDE, i),
                    dynamic_dim=dyn_stride,
                    constraint_dim=constraint_strides[i],
                    symbolic_context=symbolic_context,
                )
        assert all(x is not None for x in stride)

        sym_sizes = [
            self.create_symintnode(
                sym,
                hint=hint,
                source=TensorPropertySource(source, TensorProperty.SIZE, i),
            )
            for i, (sym, hint) in enumerate(zip(size, ex_size))
        ]
        sym_stride = []
        for i, stride_expr in enumerate(stride):
            # NB: Don't duck size the stride; instead use the expression
            # we computed
            assert stride_expr is not None
            sym_stride.append(
                self.create_symintnode(
                    stride_expr,
                    hint=ex_stride[i],
                    source=TensorPropertySource(source, TensorProperty.STRIDE, i),
                )
            )
        sym_storage_offset = self.create_symintnode(
            self.create_symbol(
                ex_storage_offset,
                TensorPropertySource(source, TensorProperty.STORAGE_OFFSET),
                dynamic_dim=dynamic_offset,
                constraint_dim=None,
                symbolic_context=symbolic_context,
            ),
            hint=ex_storage_offset,
            source=TensorPropertySource(source, TensorProperty.STORAGE_OFFSET),
        )
        return tuple(sym_sizes), tuple(sym_stride), sym_storage_offset

    @record_shapeenv_event()
    def create_symintnode(
        self,
        sym: sympy.Expr,
        *,
        hint: Optional[int],
        source: Optional[Source] = None,
    ) -> Union[int, SymInt]:
        """Create a SymInt value from a symbolic expression

        If you know what the current hint value of the SymInt to be created
        is, pass it into hint.  Otherwise, pass None and we will make our best
        guess

        """
        if self._translation_validation_enabled and source is not None:
            # Create a new symbol for this source.
            symbol = self._create_symbol_for_source(source)
            assert symbol is not None

            # Create a new FX placeholder and Z3 variable for 'symbol'.
            fx_node = self._create_fx_placeholder_and_z3var(symbol, int)

            # Add an equality assertion for the newly created symbol and 'sym'.
            self._add_assertion(sympy.Eq(symbol, sym))
        else:
            fx_node = None

        out: Union[int, SymInt]
        if isinstance(sym, sympy.Integer):
            if hint is not None:
                assert int(sym) == hint
            out = int(sym)
        else:
            # How can this occur? When we mark_unbacked, we end up with a real
            # tensor that has hints for all sizes, but we MUST NOT create a
            # SymNode with a hint, because we're hiding the hint from our eyes
            # with the unbacked Symbol.  And in fact, the hint compute may be
            # inconsistent with size oblivious tests.
            if free_unbacked_symbols(sym):
                hint = None
            out = SymInt(SymNode(sym, self, int, hint, fx_node=fx_node))
        return out

    @record_shapeenv_event()
    def create_symfloatnode(
        self,
        sym: sympy.Expr,
        *,
        hint: Optional[int],
        source: Optional[Source] = None,
    ) -> Union[float, SymFloat]:
        """Create a SymFloat value from a symbolic expression"""
        if self._translation_validation_enabled and source is not None:
            # Create a new symbol for this source.
            symbol = self._create_symbol_for_source(source)
            assert symbol is not None

            # Create a new FX placeholder and Z3 variable for 'symbol'.
            fx_node = self._create_fx_placeholder_and_z3var(symbol, float)

            # Add an equality assertion for the newly created symbol and 'sym'.
            self._add_assertion(sympy.Eq(symbol, sym))
        else:
            fx_node = None

        out: Union[float, SymFloat]
        if isinstance(sym, sympy.Float):
            if hint is not None:
                assert float(sym) == hint
            out = float(sym)
        else:
            # You could give this the same treatment as SymInt above if
            # you supported mark_unbacked on a float, but it's a kind of
            # strange thing to do though because floats don't get 0/1
            # specialization anyway
            if free_unbacked_symbols(sym):
                assert hint is None, sym
            out = SymFloat(SymNode(sym, self, float, hint, fx_node=fx_node))
        return out

    @record_shapeenv_event()
    def create_unspecified_symint_and_symbol(
        self, value: int, source: Source, dynamic_dim: DimDynamic
    ) -> Union[int, SymInt]:
        """Create a SymInt wrapping a new unspecified symbol"""
        return self.create_symintnode(
            self.create_unspecified_symbol(
                value,
                source=source,
                dynamic_dim=dynamic_dim,
            ),
            hint=value,
            source=source,
        )

    def create_symboolnode(self, sym: sympy.Expr) -> SymBool:
        """Create a SymBool object from a sympy boolean expression"""
        # This function is only being used in serialization, so we do not track it
        # for validation.
        return SymBool(SymNode(sym, self, bool, None))

    def _log_create_unbacked_symbol(
        self,
        prefix: str,
        symbol: sympy.Symbol,
        vr: ValueRanges,
        source: Optional[Source] = None,
    ) -> None:
        is_debug = config.extended_debug_create_symbol is not None and str(
            symbol
        ) in config.extended_debug_create_symbol.split(",")
        sloc: Union[str, SLoc]
        if source is None:
            sloc, maybe_extra_debug = self._get_stack_summary(is_debug)
        else:
            sloc, maybe_extra_debug = source.name(), ""
        log.info(
            "%s %s [%s, %s] %s%s",
            prefix,
            symbol,
            vr.lower,
            vr.upper,
            sloc,
            maybe_extra_debug,
            stack_info=is_debug,
        )
        trace_structured(
            "create_unbacked_symbol",
            metadata_fn=lambda: {
                "symbol": str(symbol),
                "vr": f"[{vr.lower}, {vr.upper}]",
                "user_stack": structured.from_traceback(TracingContext.extract_stack()),
                "stack": structured.from_traceback(
                    CapturedTraceback.extract(skip=1).summary()
                ),
            },
        )

    @record_shapeenv_event()
    def create_unbacked_symfloat(self) -> SymFloat:
        """Create a symbolic float without a hint value"""
        symbol: sympy.Symbol = make_symbol(
            SymT.UNBACKED_FLOAT, next(self.unbacked_symfloat_counter)
        )
        self.counter["create_unbacked_symbol"] += 1
        if not self._ignore_fresh_unbacked_symbols_tls():
            self.pending_fresh_unbacked_symbols.append(symbol)
        self.var_to_stack[symbol] = CapturedTraceback.extract(skip=1)
        vr = self.var_to_range[symbol] = ValueRanges.unknown()
        assert vr.is_float
        sloc = self._get_sloc()
        self.var_to_range_sloc[symbol] = ValueRangesSLoc(sloc, sloc)

        # Create a new FX placeholder and Z3 variable for 'symbol'.
        fx_node = self._create_fx_placeholder_and_z3var(symbol, float)

        self._log_create_unbacked_symbol("create_unbacked_symfloat", symbol, vr)

        return SymFloat(SymNode(symbol, self, float, None, fx_node=fx_node))

    @record_shapeenv_event()
    def create_unbacked_symint(self, source: Optional[Source] = None) -> SymInt:
        """Create a symbolic integer without a hint value"""
        symbol: sympy.Symbol = make_symbol(
            SymT.UNBACKED_INT, next(self.unbacked_symint_counter), integer=True
        )
        if not self._ignore_fresh_unbacked_symbols_tls():
            self.pending_fresh_unbacked_symbols.append(symbol)
        self.counter["create_unbacked_symbol"] += 1
        self.var_to_stack[symbol] = CapturedTraceback.extract(skip=1)
        vr = self.var_to_range[symbol] = self._default_unspecified_value_range()
        assert vr.is_int
        sloc = self._get_sloc()
        self.var_to_range_sloc[symbol] = ValueRangesSLoc(sloc, sloc)

        # Create a new FX placeholder and Z3 variable for 'symbol'.
        fx_node = self._create_fx_placeholder_and_z3var(symbol, int)

        self._log_create_unbacked_symbol("create_unbacked_symint", symbol, vr, source)

        return SymInt(SymNode(symbol, self, int, None, fx_node=fx_node))

    def is_unbacked_symint(self, symbol: sympy.Symbol) -> bool:
        """Check if a sympy symbol matches the naming convention for unbacked symbols"""
        return symbol_is_type(symbol, SymT.UNBACKED_INT)

    @record_shapeenv_event()
    def create_unbacked_symbool(self) -> SymBool:
        """Create a symbolic boolean without a hint value"""
        symbol: sympy.Symbol = make_symbol(
            SymT.UNBACKED_INT, next(self.unbacked_symint_counter), integer=True
        )
        if not self._ignore_fresh_unbacked_symbols_tls():
            self.pending_fresh_unbacked_symbols.append(symbol)
        self.counter["create_unbacked_symbol"] += 1
        self.var_to_stack[symbol] = CapturedTraceback.extract(skip=1)
        vr = self.var_to_range[symbol] = ValueRanges(0, 1)
        assert vr.is_int
        sloc = self._get_sloc("default value range for unbacked SymBool")
        self.var_to_range_sloc[symbol] = ValueRangesSLoc(sloc, sloc)

        # Create a new FX placeholder and Z3 variable for 'symbol'.
        fx_node = self._create_fx_placeholder_and_z3var(symbol, bool)

        self._log_create_unbacked_symbol("create_unbacked_symbool", symbol, vr)

        return SymBool(SymNode(sympy.Eq(symbol, 1), self, bool, None, fx_node=fx_node))

    @record_shapeenv_event()
    def create_unspecified_symbol(
        self,
        val: Union[int, SymInt, float, SymFloat],
        source: Source,
        dynamic_dim: DimDynamic = DimDynamic.DUCK,
        constraint_dim: DimConstraint = None,  # NB: includes None
        symbolic_context: Optional[StatelessSymbolicContext] = None,
    ) -> sympy.Expr:
        """
        Create a symbol with an unspecified value

        Compared to standard symbols we do not assume the value is positive,
        nor do we specialze on zero or one values.
        """
        # 'positive' is None for unspecified symbols, since we can't
        # assume that it will be neither positive nor negative.

        # We don't want to specialize zero one val for unspecified symbol
        # so that we can always get a new symbol despite val.
        return self.create_symbol(
            val,
            source,
            dynamic_dim,
            constraint_dim,
            positive=None,
            do_not_specialize_zero_one=True,
            symbolic_context=symbolic_context,
        )

    @record_shapeenv_event()
    def create_symbol(
        self,
        val: int,
        source: Source,
        dynamic_dim: DimDynamic = DimDynamic.DUCK,
        constraint_dim: DimConstraint = None,  # NB: includes None
        positive: Optional[bool] = True,
        do_not_specialize_zero_one: bool = False,
        symbolic_context: Optional[StatelessSymbolicContext] = None,
    ) -> sympy.Expr:
        """Create a new symbol which is tracked by this ShapeEnv"""
        # check if constraint_dim is actually static integer
        if (
            isinstance(constraint_dim, StrictMinMaxConstraint)
            and constraint_dim.vr.lower == constraint_dim.vr.upper
        ):
            dynamic_dim = DimDynamic.STATIC
            if constraint_dim.vr.lower != val:
                raise ConstraintViolationError(
                    f"Static shape constraint of {constraint_dim.vr.lower} does not match input size of {val}, "
                    f"for {source.name()}"
                )
            if symbolic_context:
                from torch._dynamo.source import TensorPropertySource

                assert isinstance(source, TensorPropertySource)
                # TODO: storage_offset handling?
                assert source.idx is not None
                symbolic_context.dynamic_sizes[source.idx] = dynamic_dim
                symbolic_context.constraint_sizes[source.idx] = None
            constraint_dim = None

        # see note [Tensor Fakification and Symbol Caching]
        source_name = source.name()
        if (
            isinstance(symbolic_context, StatefulSymbolicContext)
            and id(self) not in symbolic_context.shape_env_to_source_to_symbol_cache
        ):
            symbolic_context.shape_env_to_source_to_symbol_cache[id(self)] = {}

        if (
            isinstance(symbolic_context, StatefulSymbolicContext)
            and source_name
            and (
                source_name
                in symbolic_context.shape_env_to_source_to_symbol_cache[id(self)]
            )
        ):
            return symbolic_context.shape_env_to_source_to_symbol_cache[id(self)][
                source_name
            ]

        if dynamic_dim in (DimDynamic.SIZE_LIKE_UNBACKED, DimDynamic.OBLIVIOUS_SIZE):
            out = self.create_unbacked_symint(source).node.expr
            self._constrain_range_for_size(out)
            if isinstance(symbolic_context, StatefulSymbolicContext) and source_name:
                symbolic_context.shape_env_to_source_to_symbol_cache[id(self)][
                    source_name
                ] = out
            if dynamic_dim is DimDynamic.OBLIVIOUS_SIZE:
                self.oblivious_var_to_val[out] = val
            return out

        if do_not_specialize_zero_one:
            specialize_zero_one = False
        else:
            specialize_zero_one = self.specialize_zero_one

        assert isinstance(source, Source), f"{type(source)} {source}"
        assert not (positive and val < 0), f"positive set for negative value: {val}"
        # It's always sound to allocate a symbol as DYNAMIC.  If the user
        # constrained the symbol, force the symbolic_context to DYNAMIC, because our
        # constraint code will do weird stuff if, e.g., it's duck shaped
        if constraint_dim is not None:
            dynamic_dim = DimDynamic.DYNAMIC

        if dynamic_dim is DimDynamic.STATIC:
            out = sympy.Integer(val)
            if isinstance(symbolic_context, StatefulSymbolicContext) and source_name:
                symbolic_context.shape_env_to_source_to_symbol_cache[id(self)][
                    source_name
                ] = out
            return out

        elif dynamic_dim is DimDynamic.DUCK:
            # duck_shape can be used to globally turn off duck shaping, even
            # if it was requested
            duck = self.duck_shape
        elif dynamic_dim is DimDynamic.DYNAMIC:
            duck = False
        else:
            raise AssertionError(f"unhandled dynamic_dim {dynamic_dim}")

        sloc = self._get_sloc()

        if val in (0, 1) and specialize_zero_one:
            r = self.val_to_var[val]
        elif not duck or val not in self.val_to_var:
            # If we're not duck shaping, we always create a new symbol
            # Even if we're duck shaping, if we haven't seen this particular
            # value before, we also create a new symbol
            if type(val) is int or is_nested_int(val):
                sympy_expr = make_symbol(
                    SymT.SIZE, len(self.var_to_val), positive=positive, integer=True
                )
            else:
                sympy_expr = make_symbol(
                    SymT.FLOAT, len(self.var_to_val), positive=positive, real=True
                )
            self.source_to_var[source_name] = sympy_expr
            # We always associate vars to vals
            if isinstance(val, int):
                self.var_to_val[sympy_expr] = sympy.Integer(val)
            elif isinstance(val, float):
                self.var_to_val[sympy_expr] = sympy.Float(val)
            else:
                # Only used for jagged layout nested tensors
                self.var_to_val[sympy_expr] = SingletonInt(
                    val.node.nested_int(), coeff=val.node.nested_int_coeff()
                )

            # Do the appending later, because we always want to populate this
            self.var_to_sources[sympy_expr] = []
            # Create a Z3 variable for the new symbol.
            self._add_z3var(sympy_expr, int)

            if duck:
                # Make sure to reuse this symbol for subsequent duck shaping
                self.val_to_var[val] = sympy_expr

            if isinstance(val, int):
                if positive:
                    # Add assertions for the newly created symbols
                    self._add_assertion(sympy_expr > 1)

                    # Apply default range, which assumes not zero-one
                    self.var_to_range[sympy_expr] = self._default_value_range()
                    self.var_to_range_sloc[sympy_expr] = ValueRangesSLoc(
                        self._get_sloc(
                            "user code shown is first use of this value--the guard itself is not "
                            "due user code but due to 0/1 specialization in the framework; to "
                            "avoid specialization try torch._dynamo.mark_unbacked(tensor, dim)"
                            if self.specialize_zero_one
                            else None
                        ),
                        sloc,
                    )
                else:
                    self.var_to_range[
                        sympy_expr
                    ] = self._default_unspecified_value_range()
                    self.var_to_range_sloc[sympy_expr] = ValueRangesSLoc(sloc, sloc)

                # Small performance optimization: if we have a min-max constraint,
                # we can proactively narrow to that range
                if isinstance(constraint_dim, StrictMinMaxConstraint):
                    assert not duck
                    self._update_var_to_range(
                        sympy_expr, constraint_dim.vr, is_constraint=True
                    )

                vr = self.var_to_range[sympy_expr]
                assert vr.is_int

                if val not in vr:
                    raise ConstraintViolationError(
                        f"{val} not in range [{vr.lower}, {vr.upper}]"
                    )

                range_str = f"[{vr.lower}, {vr.upper}]"
            elif isinstance(val, float):
                self.var_to_range[sympy_expr] = vr = ValueRanges(-sympy.oo, sympy.oo)
                self.var_to_range_sloc[sympy_expr] = ValueRangesSLoc(sloc, sloc)
                range_str = f"[{vr.lower}, {vr.upper}]"
                assert vr.is_float
            else:
                # Skip var_range logic for SingletonInt
                # Only used for jagged layout nested tensors
                range_str = ""

            r = sympy_expr

            is_debug = config.extended_debug_create_symbol is not None and str(
                sympy_expr
            ) in config.extended_debug_create_symbol.split(",")
            maybe_more_info = ""
            if not is_debug and os.getenv("TORCHDYNAMO_EXTENDED_ADVICE", "1") not in (
                "0",
                "",
            ):
                maybe_more_info = (
                    ", for more info run with "
                    f'TORCHDYNAMO_EXTENDED_DEBUG_CREATE_SYMBOL="{sympy_expr}" '
                    "or to suppress this message run with "
                    'TORCHDYNAMO_EXTENDED_ADVICE="0"'
                )
            sloc, maybe_extra_debug = self._get_stack_summary(is_debug)
            self.log.info(
                "create_symbol %s = %s for %s %s %s%s%s",
                sympy_expr,
                val,
                source.name(),
                range_str,
                sloc,
                maybe_more_info,
                maybe_extra_debug,
                stack_info=is_debug,
            )
            trace_structured(
                "create_symbol",
                metadata_fn=lambda: {
                    "symbol": str(sympy_expr),
                    "val": repr(val),
                    "vr": range_str,
                    "source": source.name(),
                    "user_stack": structured.from_traceback(
                        TracingContext.extract_stack()
                    ),
                    "stack": structured.from_traceback(
                        CapturedTraceback.extract(skip=1).summary()
                    ),
                },
            )

            self.counter["create_symbol"] += 1
        else:
            # This implements duck-shaping: input sizes that match are assigned
            # the same symint
            r = self.val_to_var[val]
            self.source_to_var[source_name] = r
            self.log.debug("create_symbol %s duck sized %s", r, source.name())

        if isinstance(r, sympy.Symbol):
            r_sources = self.var_to_sources[r]
            r_sources.append(source)
            if not source.is_ephemeral() and r_sources[0].is_ephemeral():
                # prefer non-ephemeral source first since it may be guarded on later
                r_sources[0], r_sources[-1] = r_sources[-1], r_sources[0]

            # This ensures we get zeros in symbol_guard_counts, which makes
            # some queries simpler (since we will accumulate mass on 0 this
            # way)
            self.symbol_guard_counter[r] = 0

        if isinstance(symbolic_context, StatefulSymbolicContext) and source_name:
            symbolic_context.shape_env_to_source_to_symbol_cache[id(self)][
                source_name
            ] = r
        return r

    def add_var_to_val(self, expr: sympy.Symbol, val: int) -> None:
        """Adds a new symbol to the symbolic environment."""
        log.debug("add_var_to_val %s %s", expr, val, stack_info=True)
        assert expr not in self.var_to_val, f"{expr} already exists"
        self.var_to_val[expr] = sympy.Integer(val)

    def _debug_name(self, source: Source) -> str:
        src_name = source.name()
        return self.source_name_to_debug_name.get(src_name, src_name)

    def _render_range_for_constraint_violation(
        self, source: Source, c: Union[StrictMinMaxConstraint, RelaxedUnspecConstraint]
    ) -> str:
        if isinstance(c, StrictMinMaxConstraint):
            lower, upper = c.vr.lower, c.vr.upper
            default = self._default_value_range()
            if lower <= default.lower:
                lower = None
            if upper >= default.upper:
                upper = None
            c_render = (
                f"{self._debug_name(source)} = {source.name()} in the specified range"
            )
            if lower is not None and upper is not None:
                c_render += f" {lower} <= {self._debug_name(source)} <= {upper}"
            elif lower is None and upper is not None:
                c_render += f" {self._debug_name(source)} <= {upper}"
            elif lower is not None and upper is None:
                c_render += f" {lower} <= {self._debug_name(source)}"
            return c_render
        return c.render(source)

    def produce_guards(self, *args: Any, **kwargs: Any) -> _ShapeGuardsHelper:
        """
        Like produce_guards_verbose, but only returns the non-verbose guard expressions
        (no verbose guards produced.)
        """
        return self.produce_guards_verbose(*args, **kwargs)[0]

    def produce_guards_verbose(
        self,
        placeholders: Sequence[FakeTensor],
        sources: Sequence[Source],
        source_ref: Callable[[Source], str] = lambda n: n.name(),
        *,
        guards: Optional[List[ShapeGuard]] = None,
        input_contexts: Optional[DimList[SymbolicContext]] = None,
        # Encodes user-specified input shape equations of the form s = s' and s = fn(s').
        # (See docs on EqualityConstraint for details of the encoding.)
        equalities_inputs: Optional[EqualityConstraint] = None,
        _simplified: bool = False,
        # Indicates if we should produce guards for known static values.
        ignore_static: bool = True,
        lang: str = "python",
    ) -> Tuple[_ShapeGuardsHelper, List[str]]:  # regular, verbose
        """
        Generates a list of guards strings which, when evaluated in a context that
        defines tensors for all the sources, returns True or False depending
        on if the guards in the list evaluated to True or not.  Primarily used by Dynamo,
        but this is also helpful for manual testing of guards (see
        evaluate_guards_for_args)

        For convenience in testing, a source is allowed to be a str,
        in which case we will assume it is a LocalSource

        simplified lets you omit duck sizing, equality and 0/1 guards.
        This is useful for testing when you don't care about the boilerplate
        guards, and it may be helpful for user output too (be careful though;
        some equality guards are nontrivial!  It would be nice to get simplified
        output to print them too).  It's private because it's not
        intended for normal use
        """
        self.log.info("produce_guards")

        # Check if we get to the same ShapeEnv state by replaying the recorded events.
        # This will create a new ShapeEnv instance, and call all recorded function
        # calls on this new instance. Finally, it will check whether this new instance
        # has equal state.
        #
        # It's important that we do it in the begining of this function, since it modifies
        # self.dim_constraints through its execution. Changes that happen in this method
        # aren't interesting, since this is the function call we wish to reproduce at the
        # end. If we wish to simply reproduce ShapeEnv instances even after this call,
        # this method should also be recorded.
        if self.check_recorded_events:
            shape_env = replay_shape_env_events(self.events)
            self.check_equal(shape_env)

        assert len(placeholders) == len(
            sources
        ), f"len({placeholders}) != len({sources})"
        Tensorlike = (torch.Tensor, FakeTensorMeta)

        def _create_no_constraints_context(t: Tensor) -> StatelessSymbolicContext:
            return StatelessSymbolicContext(
                # Ignored; only the constraints part is relevant below.
                dynamic_sizes=[DimDynamic.DYNAMIC] * t.dim(),
                dynamic_strides=[DimDynamic.INFER_STRIDE] * t.dim(),
                constraint_sizes=[None] * t.dim(),
                constraint_strides=[None] * t.dim(),
            )

        # Expand optional inputs, or verify invariants are upheld
        if input_contexts is None:
            input_contexts = [
                _create_no_constraints_context(t) if isinstance(t, Tensorlike) else None
                for t in placeholders
            ]
        else:
            assert len(input_contexts) == len(placeholders)
            for i, (t, context) in enumerate(zip(placeholders, input_contexts)):
                if isinstance(t, Tensorlike):
                    if context is None:
                        input_contexts[i] = _create_no_constraints_context(t)
                else:
                    assert isinstance(t, (SymInt, int, SymFloat, float))
                    assert not isinstance(context, list)

        # It took a lot of sweat to figure out the algorithm here.  Let's
        # explain how it works.
        #
        # The ShapeEnv lifecycle looks something like this:
        #
        # - For each input, you either generate a fresh Sympy symbol (s0) to
        #   represent its value (a binding site), or you reuse some
        #   preexisting symbol or expression, skipping the symbol allocation
        #   (e.g., duck sizing to a preexisting symbol, or expressing a
        #   stride as a multiplication of a separate stride and size.)
        #   Naively, you might expect to bind a fresh Sympy symbol for
        #   every input, but this is fairly wasteful as most of these
        #   symbols immediately simplify away, and if you don't eagerly
        #   specialize, e.g., 0/1 symbols, you end up with very complicated
        #   expressions that are not optimizable in practice.
        #
        # - You perform some compute on these symbols, occasionally
        #   introducing guards on boolean expressions on these symbols.
        #   In particular, whenever we guard on equality (_maybe_guard_rel),
        #   we can simplify shapes; e.g., when s0 == s1 * 2, we can now
        #   replace all occurrences of s0 with s1 * 2.  Sometimes, a
        #   boolean expression evaluation doesn't introduce a guard, as
        #   the guard is already entailed by the simplifications we have
        #   applied.
        #
        # - In the end, you have a bunch of replacements (saying how to
        #   simplify shapes) and a bunch of guards (all the equality guards
        #   are trivial, because they're covered by the replacements).
        #
        # From the ShapeEnv, we must generate a Python expression that, when
        # evaluated on a set of inputs, tells us whether or not these boolean
        # expressions would have evaluated in the same way.  However,
        # we cannot easily compute this, as we elide recording boolean
        # expressions when we think they are vacuously true.  Thus, we seek
        # an approximation: we must generate an expression, if true, would have
        # produced an "equivalent" ShapeEnv, which would answer guard
        # expressions in the same way.
        #
        # Our notion of equivalence is a bit subtle.  For example, consider
        # the ShapeEnv created from an input of size (5, 4) versus (4, 4)
        # (no other guards.)  Duck sizing would generate (s0, s1) in the first
        # case but (s0, s0) in the second.  We do NOT assume that size
        # variables are disjoint; so in fact a graph that assumes the input
        # could be (s0, s1) subsumes (s0, s0) (setting s0 == s1), but not
        # vice versa.  However, consider an analogous case (1,) versus (2,).
        # Duck sizing generates (1,) and (s0,); the (s0,) graph does NOT
        # subsume the (1,) graph because we assume that any size variables
        # is NOT 0/1 (and make simplifications according to this; e.g., if
        # we queried s0 == 0, we would immediately return False without
        # returning a guard.)
        #
        # So, it is perhaps easier to flip things on their head: the guard
        # expressions we generate here say what simplifications are valid,
        # and what are not. Below, we explain each of the guard expressions
        # we generate

        # TODO: Make this more efficient by binding all the size/stride/offsets
        # to locals before performing tests on them.

        from torch._dynamo.source import TensorProperty, TensorPropertySource

        # Actual codegen must be delayed as we don't necessarily know what
        # the symbol mapping is
        input_guards = []

        symbol_to_source: Dict[sympy.Symbol, List[Source]] = collections.defaultdict(
            list
        )
        symbol_to_constraints: DefaultDict[
            sympy.Symbol, Set[Constraint]
        ] = collections.defaultdict(set)
        constraint_violations: List[Tuple[bool, str, Callable[[], str]]] = []

        py_printer = ShapeGuardPythonPrinter(
            symbol_to_source, source_ref, self.var_to_sources
        )
        if lang == "cpp":
            printer = ShapeGuardCppPrinter(
                symbol_to_source, source_ref, self.var_to_sources
            )
        elif lang == "python":
            printer = py_printer
        else:
            raise NotImplementedError(f"Unknown lang: {lang}")

        def record_constraint_violation(
            warn_only: bool,
            debug_name: str,
            msg: str,
            hint: Optional[Callable[[], str]] = None,
        ) -> None:
            constraint_violations.append(
                (warn_only, debug_name, lambda: f"{msg}{hint()}" if hint else msg)
            )

        def is_dim(src: object) -> TypeGuard[TensorPropertySource]:
            return (
                isinstance(src, TensorPropertySource)
                and src.prop is TensorProperty.SIZE
            )

        if equalities_inputs:
            source_index = {}
            for i, src in enumerate(sources):
                source_index[src.name()] = i

            def get_expression(tensor_dim_src: Source) -> sympy.Expr:
                fake = placeholders[source_index[tensor_dim_src.base.name()]]  # type: ignore[attr-defined]
                assert tensor_dim_src.idx is not None  # type: ignore[attr-defined]
                symint = fake.shape[tensor_dim_src.idx]  # type: ignore[attr-defined]
                if isinstance(symint, torch.SymInt):
                    return symint.node.expr
                else:
                    assert type(symint) is int, f"Expected int, got {type(symint)}"
                    return sympy.Integer(symint)

            for src1, src2 in equalities_inputs.source_pairs:
                expr1, expr2 = get_expression(src1), get_expression(src2)  # type: ignore[]
                # Check whether given input shape values satisfy a specified equation s = s'.
                # - Raise when the equation was violated by the given input shape values.
                # - Otherwise issue a guard to constrain them.
                concrete_val = self.evaluate_expr(sympy.Eq(expr1, expr2))
                if not concrete_val:
                    raise ConstraintViolationError(
                        f"{src1.name()} = {expr1 if isinstance(expr1, int) else expr1.xreplace(self.var_to_val)}"
                        " is not equal to "
                        f"{src2.name()} = {expr2 if isinstance(expr2, int) else expr2.xreplace(self.var_to_val)}"
                    )

            for srcEq, root, fn in equalities_inputs.derived_equalities:
                expr1 = get_expression(srcEq)
                # recall that root is either a phantom symbol or an input source
                expr2, debug_name = (
                    (root, self.var_to_sources[root][0].name())
                    if isinstance(root, sympy.Symbol)
                    else (get_expression(root), self._debug_name(root))
                )
                expr2_ = fn(expr2)
                # Check whether given input shape values satisfy a specified equation s = fn(s').
                # - Raise when the equation was violated by the given input shape values.
                # - Otherwise issue a guard to constrain them.
                concrete_val = self.evaluate_expr(sympy.Eq(expr1, expr2_))
                if not concrete_val:
                    raise ConstraintViolationError(
                        f"Expected input {srcEq.name()} to be equal to "
                        f"{fn(sympy.Symbol(debug_name))}, "
                        f"where {debug_name} = {expr2.xreplace(self.var_to_val)}, "
                        f"but got {expr1.xreplace(self.var_to_val)}"
                    )

            for phantom_symbol in equalities_inputs.phantom_symbols:
                # we created additional phantom symbols that are not input shape dimensions
                symbol_to_source[phantom_symbol].extend(
                    self.var_to_sources[phantom_symbol]
                )

        # How do we know what the value of s0 is?  Fresh variables can only be
        # bound by inputs, so there MUST be some other input which binds the
        # variable.  If there is no such input, this is an error in our
        # system.  We record where all symbols come from, to help you diagnose
        # why those symbols didn't occur.
        #
        # In fact, generally speaking it is only possible for the "outermost"
        # user of a ShapeEnv to evaluate the guards, because some inputs may
        # not be available to inner levels.  For example, Dynamo can guard on
        # tensors that never actually become graph arguments (they are
        # pruned).  In this case, only Dynamo knows about these arguments.
        def track_symint(
            source: Source, val: Union[SymInt, int], constraint: DimConstraint = None
        ) -> None:
            log.debug("track_symint %s %s %s", LazyString(source.name), val, constraint)
            assert not isinstance(val, SymInt) or is_symbolic(val)

            if isinstance(val, SymInt) and val.node.maybe_as_int() is not None:
                val = val.node.maybe_as_int()

            if isinstance(val, SymInt):
                s = val.node.expr
                if isinstance(s, sympy.Symbol):
                    symbol_to_source[s].append(source)
                    if constraint is not None and not isinstance(
                        constraint, RelaxedUnspecConstraint
                    ):
                        symbol_to_constraints[s].add(constraint)
                else:
                    constraint_violated = False
                    if isinstance(constraint, StrictMinMaxConstraint):
                        # try inferring the ranges of the expr s
                        sym_vrs = {
                            x: self.var_to_range.get(x, None) for x in s.free_symbols
                        }
                        if any(vr is None for vr in sym_vrs.values()):
                            # some of the free symbols in s don't have ranges
                            constraint_violated = True
                    elif isinstance(constraint, RelaxedUnspecConstraint):
                        if s.is_number:
                            i = int(s)
                            # Don't complain about 0/1 specialization, we
                            # expect to have to compile in this case anyway
                            if i not in (0, 1):
                                constraint_violated = True
                    if constraint_violated:
                        assert constraint is not None

                        def hint(s: sympy.Expr) -> str:
                            sexpr = py_printer.doprint(s)
                            return f"{sexpr}."

                        var_with_range = self._render_range_for_constraint_violation(
                            source, constraint
                        )
                        msg = (
                            f"Not all values of {var_with_range} are valid because "
                            f"{self._debug_name(source)} was inferred to be equal to "
                        )
                        record_constraint_violation(
                            constraint.warn_only,
                            self._debug_name(source),
                            msg,
                            hint=functools.partial(hint, s),
                        )

                input_guards.append((source, s))
            else:
                s = sympy.Integer(val)
                input_guards.append((source, s))
                constraint_violated = False
                if isinstance(constraint, StrictMinMaxConstraint):
                    if not (
                        s == constraint.vr.lower == constraint.vr.upper
                    ):  # allow static constraints
                        constraint_violated = True
                elif isinstance(constraint, RelaxedUnspecConstraint):
                    # Don't complain about 0/1 specialization, we
                    # expect to have to compile in this case anyway
                    if val not in (0, 1):
                        constraint_violated = True
                if constraint_violated:
                    assert constraint is not None
                    var_with_range = self._render_range_for_constraint_violation(
                        source, constraint
                    )
                    msg = (
                        f"Not all values of {var_with_range} are valid because "
                        f"{self._debug_name(source)} was inferred to be a constant ({val})."
                    )
                    record_constraint_violation(
                        constraint.warn_only, self._debug_name(source), msg
                    )

        def track_symfloat(source: Source, val: Union[float, SymFloat]) -> None:
            log.debug("track_symfloat %s %s", LazyString(source.name), val)
            assert not isinstance(val, SymFloat) or is_symbolic(val)

            if isinstance(val, SymFloat) and val.node.maybe_as_float() is not None:
                val = val.node.maybe_as_float()

            if isinstance(val, SymFloat):
                s = val.node.expr
                if isinstance(s, sympy.Symbol):
                    symbol_to_source[s].append(source)
                input_guards.append((source, s))
            else:
                s = sympy.Float(val)
                input_guards.append((source, s))

        for t, source, context in zip(placeholders, sources, input_contexts):
            if isinstance(source, str):
                from torch._dynamo.source import LocalSource

                source = LocalSource(source)
            assert isinstance(source, Source)
            if t is None:
                continue
            if isinstance(t, (SymInt, int)):
                track_symint(source, t)
                continue
            elif isinstance(t, (SymFloat, float)):
                track_symfloat(source, t)
                continue
            assert isinstance(t, Tensorlike)
            if is_traceable_wrapper_subclass(t):
                from torch._dynamo.source import AttrSource

                assert isinstance(context, SubclassSymbolicContext)

                # For subclasses, we need to track symints on BOTH the outer
                # and inner tensors.
                # TODO: type this better
                sources_tensors_constraints: List[Tuple[Source, Any, Any, Any]] = [
                    (source, t, context.constraint_sizes, context.constraint_strides)
                ]
                attrs, _ = t.__tensor_flatten__()
                for attr in attrs:
                    inner_t = getattr(t, attr)
                    inner_context = context.inner_contexts[attr]
                    sources_tensors_constraints.append(
                        (
                            AttrSource(source, attr),
                            inner_t,
                            inner_context.constraint_sizes,  # type: ignore[attr-defined]
                            inner_context.constraint_strides,  # type: ignore[attr-defined]
                        )
                    )
            else:
                sources_tensors_constraints = [
                    (source, t, context.constraint_sizes, context.constraint_strides)  # type: ignore[attr-defined]
                ]

            for (
                src,
                curr_t,
                constraint_size,
                constraint_stride,
            ) in sources_tensors_constraints:
                if is_sparse_any(curr_t):
                    for i, ss in enumerate(curr_t.size()):
                        property_source = TensorPropertySource(
                            src, TensorProperty.SIZE, i
                        )
                        track_symint(property_source, ss, constraint_size[i])
                else:
                    for i, ss in enumerate(curr_t.size()):
                        property_source = TensorPropertySource(
                            src, TensorProperty.SIZE, i
                        )
                        track_symint(property_source, ss, constraint_size[i])
                    for i, ss in enumerate(curr_t.stride()):
                        property_source = TensorPropertySource(
                            src, TensorProperty.STRIDE, i
                        )
                        track_symint(property_source, ss, constraint_stride[i])
                    track_symint(
                        TensorPropertySource(src, TensorProperty.STORAGE_OFFSET),
                        curr_t.storage_offset(),
                    )

        # 1. Every input must equal the final simplified symbolic expression
        #    stored on the placeholder.  Given a placeholder (s0*2, s1),
        #    if we have an input (2, 3), we must show s0*2 == 2 and s1 == 3.
        #    This does a lot of work: it covers duck sizing and equality guards.
        exprs = []
        verbose_exprs = []
        self.dim_constraints = DimConstraints(
            symbol_to_source,
            self.var_to_val,
            set(symbol_to_constraints.keys()),
            self.source_name_to_debug_name,
        )

        if not _simplified:
            for source, expr in input_guards:
                srcname = source.name()
                if self._translation_validation_enabled:
                    # Ignore sources that were not turned into SymInts.
                    if srcname in self.source_to_symbol:
                        self._add_target_expr(
                            sympy.Eq(self.source_to_symbol[srcname], expr)
                        )

                # Small optimization
                if (
                    isinstance(expr, sympy.Symbol)
                    and symbol_to_source.get(expr)
                    and source == symbol_to_source[expr][0]
                ):
                    continue

                # This logic excludes static values found on tensors from guarding, because
                # dynamo's check_tensor_fn does that (see guards.cpp).
                # However, for non tensor sources, we still need to guard here.
                if ignore_static and isinstance(source, TensorPropertySource):
                    if expr.is_number:
                        self.log.debug(
                            "Skipping guard %s", f"{source_ref(source)} == {expr}"
                        )
                        continue

                if is_dim(source):
                    self.dim_constraints.add_equality(source, expr)

                res = f"{py_printer.print_source(source)} == {py_printer.doprint(expr)}"
                exprs.append(
                    f"{printer.print_source(source)} == {printer.doprint(expr)}"
                )

                if (s0 := self.source_to_var.get(srcname)) is not None:
                    if source != self.var_to_sources[s0][0]:
                        verbose_exprs.append(
                            f"{res}  # duck sizing added this equality because these "
                            f"variables had the same size {self.var_to_val[s0]} "
                            "(to avoid this specialization, set torch.fx.experimental._config.use_duck_shape = False)"
                        )
                    elif (sloc := self.replacements_slocs.get(s0)) is not None:
                        verbose_exprs.append(f"{res}  # {sloc}")
                    else:
                        verbose_exprs.append(
                            f"{res}  # (unknown var {s0}, please file a bug)"
                        )
                else:
                    verbose_exprs.append(
                        f"{res}  # (unknown source {srcname}, please file a bug)"
                    )

                if (
                    isinstance(source, TensorPropertySource)
                    and source.prop is TensorProperty.SIZE
                    and equalities_inputs
                    and len(expr.free_symbols) == 1
                ):
                    symbol = next(iter(expr.free_symbols))
                    if (
                        isinstance(expr, sympy.Symbol)
                        and expr in symbol_to_constraints
                        and not equalities_inputs.is_equal(
                            source, symbol_to_source[expr][0]
                        )
                    ):
                        msg = (
                            f"The values of {self._debug_name(source)} = {source.name()} and "
                            f"{self._debug_name(symbol_to_source[expr][0])} = {symbol_to_source[expr][0].name()} "
                            "must always be equal."
                        )
                        record_constraint_violation(
                            equalities_inputs.warn_only, self._debug_name(source), msg
                        )

                    if (
                        not isinstance(expr, sympy.Symbol)
                        and symbol in symbol_to_constraints
                        and not equalities_inputs.is_derived(
                            source,
                            symbol_to_source[symbol][0],
                            lambda x: expr.xreplace({symbol: x}),
                        )
                    ):
                        src = symbol_to_source[symbol][0]
                        msg = (
                            f"The values of {self._debug_name(source)} = {source.name()} must always be related to "
                            f"the values of {self._debug_name(src)} = {src.name()} by "
                            f"{self._debug_name(source)} = {expr.xreplace({symbol: sympy.sympify(self._debug_name(src))})}."
                        )
                        record_constraint_violation(
                            equalities_inputs.warn_only, self._debug_name(source), msg
                        )

                # NB: Not necessary to report constraint violations here:
                # constraints are guaranteed to be on symbols (we've already
                # caught constants and non-atomic expressions), so we only
                # have relational constraints, but we don't support those
                # at the moment

        # 2. Every guard must evaluate to True (but remember many guards
        #    like s0 == s1*2 because trivial due to simplification)
        issued = set()

        def issue_guard(guard: ShapeGuard) -> None:
            expr = self.simplify(guard.expr)

            # Avoid re-issueing the same guard.
            if expr in issued:
                return

            issued.add(expr)

            try:
                is_trivial = False
                if any(
                    is_dim(source)
                    for s in expr.free_symbols
                    for source in symbol_to_source[s]
                ):
                    assert self.dim_constraints is not None
                    is_trivial = self.dim_constraints.add(expr)
                guard_expr = py_printer.doprint(expr)
                verbose_exprs.append(f"{guard_expr}  # {guard.sloc}")
                exprs.append(printer.doprint(expr))
                self._add_target_expr(expr)
                # A non-relational constraint on a single sizevar can violate
                # a constraint
                if not is_trivial and len(expr.free_symbols) == 1:
                    symbol = next(iter(expr.free_symbols))
                    source = symbol_to_source[symbol][0]
                    constraints = symbol_to_constraints[symbol]
                    for c in constraints:
                        if isinstance(c, StrictMinMaxConstraint):
                            var_with_range = (
                                self._render_range_for_constraint_violation(source, c)
                            )
                            msg = (
                                f"Not all values of {var_with_range} "
                                f"satisfy the generated guard {guard_expr}."
                            )
                            record_constraint_violation(
                                c.warn_only, self._debug_name(source), msg
                            )
                        elif isinstance(c, RelaxedUnspecConstraint):
                            # This is fine, we allow guards here as long as it
                            # didn't constrain it to one value  (we don't
                            # actually know this; this depends on our
                            # ValueRanges reasoning capability)
                            pass
                        else:
                            raise AssertionError(f"unrecognized constraint {c}")
            except Exception:
                self.log.warning("Failing guard allocated at %s", guard.sloc)
                raise

        # First, issue all guards.
        # This removes all the checks that follow from bounds
        # We could simply emit those and also the bounds 2 <= size when necessary
        for guard in guards if guards is not None else self.guards:
            if self._maybe_evaluate_static(guard.expr, axioms=()) is not None:
                continue
            issue_guard(guard)

        # Because there are guards that export's constraint solver can suggest good fixes for, that we may have
        # deferred as runtime asserts, and that produce_guards() alone won't do anything with (e.g. divisiblity guards),
        # we want to send runtime asserts to export's constraint solver too. These will still stay in the graph as asserts,
        # but export's constraint solver can decide whether to do anything with them (i.e. raise an error and provide
        # suggested fixes, or decide it's out of scope and leave as a runtime assert in the graph).
        for ra in self.deferred_runtime_asserts.get(None, []):
            if self._maybe_evaluate_static(ra.expr, axioms=()) is not None:
                continue
            expr = self.simplify(ra.expr)
            self.dim_constraints.add(expr)

        # 3. Every symbol must be within its value range (this handles 0/1
        # specialization too).
        for symbol, sources in symbol_to_source.items():
            r = self.var_to_range.get(symbol)
            if r is None:
                continue
            vr_sloc = self.var_to_range_sloc[symbol]

            assert sources
            bounds = []
            rf = source_ref(sources[0])
            if r.lower not in (-sympy.oo, -int_oo):
                if any(is_dim(source) for source in sources):
                    self.dim_constraints.add(sympy.Ge(symbol, r.lower))
                # Only print lower bound in simplified mode if it is not the
                # default
                if not _simplified or r.lower != self._default_value_range().lower:
                    bounds.append(sympy.Le(r.lower, symbol, evaluate=False))
                verbose_exprs.append(f"{r.lower} <= {rf}  # {vr_sloc.lower}")
            if r.upper not in (sympy.oo, int_oo):
                if any(is_dim(source) for source in sources):
                    self.dim_constraints.add(sympy.Le(symbol, r.upper))
                # nontrivial upper bound is always interesting
                bounds.append(sympy.Le(symbol, r.upper, evaluate=False))
                verbose_exprs.append(f"{rf} <= {r.upper}  # {vr_sloc.upper}")
            if bounds:
                bound = sympy.And(*bounds, evaluate=False)
                exprs.append(printer.doprint(bound))
                # NB: verbose_exprs are done above

                # Check constraints
                constraints = symbol_to_constraints[symbol]
                for c in constraints:
                    if isinstance(c, StrictMinMaxConstraint):
                        # TODO: With int_oo, I think this condition is a noop
                        # now
                        if not (c.vr & self._default_value_range()).issubset(r):
                            source = sources[0]

                            expr = sympy.And(
                                sympy.Le(r.lower, symbol), sympy.Le(symbol, r.upper)
                            )
                            guard_expr = py_printer.doprint(expr)
                            var_with_range = (
                                self._render_range_for_constraint_violation(source, c)
                            )
                            msg = f"Not all values of {var_with_range} satisfy the generated guard {guard_expr}"
                            record_constraint_violation(
                                c.warn_only,
                                self._debug_name(source),
                                msg,
                            )
            # We NaN specialize, which means similar to 0/1 specialization we
            # should assume that the float is NOT nan.  This is load bearing
            # if you have something like an equality guard, nan will play
            # merry hell with the reasoning.
            if symbol_is_type(symbol, SymT.FLOAT):
                res = f"not __math_isnan({py_printer.print_source(sources[0])})"
                verbose_exprs.append(
                    f"{res}  # implicit guard for float input due to NaN specialization in the framework"
                )
                if lang == "python":
                    exprs.append(res)
                elif lang == "cpp":
                    exprs.append(f"~std::isnan({printer.print_source(sources[0])})")
                else:
                    raise NotImplementedError(f"Unimplemented for lang: {lang}")

        if constraint_violations:
            warn_msgs: List[str] = []
            error_msgs: List[str] = []
            debug_names = set()
            for warn_only, debug_name, msg_cb in constraint_violations:
                if warn_only:
                    str_msg = f"  {len(warn_msgs) + 1}. {msg_cb()}"
                    warn_msgs.append(str_msg)
                else:
                    str_msg = f"  - {msg_cb()}"
                    error_msgs.append(str_msg)
                    debug_names.add(debug_name)
            if len(error_msgs) > 0:
                debug_names_str = ", ".join(sorted(debug_names))
                err = "\n".join(error_msgs)
                raise ConstraintViolationError(
                    f"Constraints violated ({debug_names_str})! "
                    'For more information, run with TORCH_LOGS="+dynamic".\n'
                    f"{err}"
                )
            elif len(warn_msgs) > 0:
                log.debug("%s Warning only constraints violated", len(warn_msgs))

        signpost_event(
            "dynamic",
            "produce_guards",
            {
                **self.co_fields,
                **self.counter,
                "num_guards": len(exprs),
                "free_symbols": sum(1 for v in symbol_to_source.values() if v),
                # The keys are meaningless from an aggregate perspective, so
                # don't include them.  Biggest first.
                "symbol_guard_counts": sorted(
                    self.symbol_guard_counter.values(), reverse=True
                ),
            },
        )

        if self._translation_validation_enabled:
            from torch.fx.experimental.validator import PopulateValidator

            # Add all deferred runtime assertions; these are not technically
            # handled by produce_guards but we need to put them in the target
            # set
            for ras in self.deferred_runtime_asserts.values():
                for ra in ras:
                    self._add_target_expr(ra.expr)

            # Add value range bound guards for all symbols with no trivial bounds.
            # Reason: '_maybe_evaluate_static' may eliminate guards based on the
            # refined value ranges.
            for sym, vr in self.var_to_range.items():
                if vr.lower not in (-sympy.oo, -int_oo):
                    self._add_target_expr(sympy.Le(vr.lower, sym))
                if vr.upper not in (sympy.oo, int_oo):
                    self._add_target_expr(sympy.Le(sym, vr.upper))

            # Before validating, populate the input of the validator with the
            # built FX graph.
            with fx_traceback.preserve_node_meta():
                PopulateValidator(self.graph, self.validator).run()

        # Only run translation validation when we are not passing custom guards
        if guards is None:
            self._check_translation_validate()

        helper: _ShapeGuardsHelper
        if lang == "cpp":
            helper = _CppShapeGuardsHelper(exprs, printer.source_to_symbol)
        else:
            helper = _ShapeGuardsHelper(exprs)
        return helper, verbose_exprs

    def produce_guards_expression(
        self,
        placeholders: Sequence[Union[SymInt, FakeTensor]],
        *,
        guards: Optional[List[ShapeGuard]] = None,
        ignore_static: bool = True,
    ) -> Optional[str]:
        """
        Expected to be used with evaluate_guards_expression(). Produces the guards
        for the given placeholders and returns a string expression to be evaluated
        by evaluate_guards_expression given concrete values for the placeholders.
        """
        from torch._dynamo.source import LocalSource

        arg_names = [f"t{i}" for i in range(len(placeholders))]
        produced_guards = self.produce_guards(
            placeholders,
            [LocalSource(a) for a in arg_names],
            guards=guards,
            ignore_static=ignore_static,
        )
        if produced_guards.exprs:
            return " and ".join(produced_guards.exprs)
        return None

    def evaluate_symexpr(self, code: str) -> Union[int, float, bool]:
        """
        To be used by compile_fx to evaluate symexprs
        """
        args = {str(e): val for e, val in self.var_to_val.items()}
        return eval(code, SYMPY_INTERP, args)

    def deserialize_symexpr(self, code: str) -> Union[SymInt, SymFloat, SymBool]:
        """
        To be used by compile_fx to deserialize symexprs
        """
        args = {
            str(e): SymInt(SymNode(e, self, int, int(val), fx_node=None))
            for e, val in self.var_to_val.items()
        }
        return eval(code, SYMPY_INTERP, args)

    def evaluate_guards_expression(self, code: str, args: Sequence[object]) -> bool:
        """
        Expected to be used with produce_guards_expression(). Evaluates an expression
        generated by produce_guards_expression for the given concrete args.
        """
        arg_names = [f"t{i}" for i in range(len(args))]
        return eval(code, SYMPY_INTERP, {"L": dict(zip(arg_names, args))})

    def evaluate_guards_for_args(
        self,
        placeholders: Sequence[FakeTensor],
        args: Sequence[Tensor],
        *,
        ignore_static: bool = True,
    ) -> bool:
        """Generate guards for a graph's placeholder values and evaluate the guards with args"""
        code = self.produce_guards_expression(placeholders, ignore_static=ignore_static)
        if code:
            return self.evaluate_guards_expression(code, args)
        return True

    def get_pruned_guards(self, symints: Sequence[torch.SymInt]) -> List[ShapeGuard]:
        """
        Get a list of guards, but pruned so it only provides guards that
        reference symints from the passed in input
        """
        symints = {
            s.node.expr for s in symints if isinstance(s.node.expr, sympy.Symbol)
        }
        guards = [
            g for g in self.guards if all(s in symints for s in g.expr.free_symbols)
        ]
        return guards

    def bind_symbols(
        self, placeholders: Sequence[FakeTensor], args: Sequence[Tensor]
    ) -> Dict[sympy.Symbol, int]:
        """
        Given a paired list of placeholders (fake tensors with
        symbolic sizes) and concrete arguments (regular tensors
        with real sizes), returns a dictionary mapping each
        symbol to its real value.  So for example, if you
        have a placeholder with size (s0, s1), binding
        (2, 4) to it will give you {s0: 2, s1: 4}.  This is
        not guaranteed to bind ALL symbols in the ShapeEnv;
        we can't bind a symbol if it doesn't occur in any placeholder,
        and symbols that already have replacements won't get bindings.

        This is a little duplicative with evaluate_guards but
        it's different enough that it seemed cleanest to make
        another copy.  This assumes the guards are already checked,
        though if it's cheap we'll check for shenanigans
        """
        bindings: Dict[sympy.Symbol, int] = {}

        def bind_symint(arg: object, val: object) -> None:
            if isinstance(val, SymInt):
                assert isinstance(arg, int)
                s = val.node.expr

                if isinstance(s, sympy.Symbol):
                    if s in bindings:
                        assert bindings[s] == arg, f"{bindings[s]} != {arg}"
                    else:
                        bindings[s] = arg
                elif isinstance(-s, sympy.Symbol):
                    if -s in bindings:
                        assert bindings[-s] == -arg, f"{bindings[-s]} != {-arg}"
                    else:
                        bindings[-s] = -arg

        for t, arg in zip(placeholders, args):
            if t is None:
                continue
            if isinstance(t, SymInt):
                bind_symint(arg, t)
                continue
            assert isinstance(t, torch.Tensor)
            for i, s in enumerate(t.size()):
                bind_symint(arg.size(i), s)
            for i, s in enumerate(t.stride()):
                bind_symint(arg.stride(i), s)
            bind_symint(arg.storage_offset(), t.storage_offset())

        return bindings

    def get_nontrivial_guards(self) -> List[SympyBoolean]:
        """Returns a list of guard expressions that aren't statically known (i.e. not trivial)"""
        return [
            self.simplify(guard.expr)
            for guard in self.guards
            if self._maybe_evaluate_static(guard.expr, axioms=()) is None
        ]

    def format_guards(self, verbose: bool = False) -> str:
        """Format this shape env's guard expressions with optional traceback info if verbose"""

        return "\n".join(
            f" - {guard.expr}{' ' + str(guard.sloc) if verbose else ''}"
            for guard in self.guards
        )

    def bound_sympy(
        self, expr: sympy.Expr, size_oblivious: bool = False
    ) -> ValueRanges:
        """Given a sympy expression, computes a ValueRanges bound for what values it can be"""
        # TODO: maybe it's guaranteed x in is var_to_range?
        var_to_range = {x: self.var_to_range.get(x, None) for x in expr.free_symbols}
        if size_oblivious:
            # Clamp values of size-like variables
            # NB: discarding the old upper bound in intentional, per
            # https://github.com/pytorch/pytorch/pull/123675
            for x in self.size_like & var_to_range.keys():
                if var_to_range[x] is not None:
                    # NB: do NOT set upper to 2 ** 48, we're using this solely
                    # to determine if we can do size-like replacement, the
                    # upper bound is irrelevant here
                    var_to_range[x] = ValueRanges(2, int_oo)
        return bound_sympy(expr, var_to_range)  # type: ignore[arg-type]

    @_lru_cache
    def get_axioms(
        self,
        symbols: Optional[Tuple[sympy.Symbol]] = None,
        compute_hint: bool = False,
    ) -> Tuple[SympyBoolean, ...]:
        """
        Given the symbols in an expression, it returns all the runtime asserts that have those symbols
        concatenated with all the guards.
        If symbols is None, it returns all the runtime asserts (and all the guards)
        """
        if symbols is None:
            runtime_asserts = (
                r.expr for rs in self.deferred_runtime_asserts.values() for r in rs
            )
        else:
            runtime_asserts = (
                r.expr
                for s in symbols
                if s not in self.var_to_val
                for r in self.deferred_runtime_asserts.get(s, ())
            )
        guards: Iterator[SympyBoolean] = (g.expr for g in self.guards)
        axioms: Iterator[SympyBoolean] = itertools.chain(guards, runtime_asserts)
        if compute_hint:
            axioms = (
                canonicalize_bool_expr(a.xreplace(self.var_to_val)) for a in axioms
            )
        return tuple(dict.fromkeys(axioms).keys())

    @lru_cache(None)
    def get_implications(
        self, e: SympyBoolean
    ) -> Tuple[Tuple[SympyBoolean, sympy.logic.boolalg.BooleanAtom], ...]:
        """Given a expression, it returns a list of predicates that follow from it"""
        equiv: Dict[SympyBoolean, sympy.logic.boolalg.BooleanAtom] = {}

        def add_expr(expr: SympyBoolean) -> None:
            expr = canonicalize_bool_expr(expr)
            if isinstance(expr, (sympy.Eq, sympy.Ne)):
                # No need to canonicalize
                # TODO We could further canonicalize Eq ordering the lhs and rhs somehow
                # With this, we could remove the need for the commutativity part
                opposite = sympy.Eq if isinstance(expr, sympy.Ne) else sympy.Ne
                # Commutativity of == and !=
                equiv[type(expr)(expr.lhs, expr.rhs, evaluate=False)] = sympy.true
                equiv[type(expr)(expr.rhs, expr.lhs, evaluate=False)] = sympy.true
                equiv[opposite(expr.lhs, expr.rhs, evaluate=False)] = sympy.false
                equiv[opposite(expr.rhs, expr.lhs, evaluate=False)] = sympy.false
            else:
                # Expr and negation
                equiv[expr] = sympy.true
                # we do not pass evaluate=False like others on purpose here!
                # we want not(a<b) to be a>=b and not ~(a<b).
                equiv[canonicalize_bool_expr(sympy.Not(expr))] = sympy.false

        add_expr(e)
        # Other relational expressions this expression implies
        if isinstance(e, sympy.Eq):
            add_expr(sympy.Le(e.lhs, e.rhs, evaluate=False))
            add_expr(sympy.Ge(e.lhs, e.rhs, evaluate=False))
        elif isinstance(e, sympy.Lt):
            add_expr(sympy.Le(e.lhs, e.rhs, evaluate=False))
            add_expr(sympy.Ne(e.lhs, e.rhs, evaluate=False))
            if e.lhs.is_integer and e.rhs.is_integer:  # type: ignore[attr-defined]
                add_expr(sympy.Le(e.lhs, e.rhs - 1, evaluate=False))
        elif isinstance(e, sympy.Le):
            add_expr(sympy.Lt(e.lhs, e.rhs + 1, evaluate=False))

        return tuple(equiv.items())

    @_lru_cache
    def _maybe_evaluate_static(
        self,
        expr: sympy.Basic,
        *,
        unbacked_only: bool = False,
        compute_hint: bool = False,
        size_oblivious: bool = False,
        axioms: Optional[Tuple[SympyBoolean]] = None,
        var_to_range: Optional[Tuple[Tuple[sympy.Symbol, ValueRanges]]] = None,
    ) -> Optional[sympy.Basic]:
        """
        Tries to evaluate expr without introducing guards

        If unbacked_only == True, then we only do substitutions on
        unbacked SymInts (leaving regular hinted integers alone).  This could
        result in an expression that still contains backed SymInts, which you
        could then potentially guard on.

        Use compute_hint == True if you are trying to compute a non-binding
        hint for the particular hint values of backed and unbacked SymInts,
        e.g., if s0 happens to be 3 this run, compute_hint will subsitute s0 with 3.
        """

        # axioms with compute hint NYE
        assert not compute_hint or not axioms
        expr = self.simplify(expr)

        if compute_hint:
            expr = expr.xreplace(self.var_to_val).xreplace(self.unbacked_var_to_val)

        expr = canonicalize_bool_expr(expr)

        def resimplify_floor_div(axioms: Dict[sympy.Expr, sympy.Expr]) -> None:
            if not self._resimplify_floor_div_axioms:
                return
            self._resimplify_floor_div_axioms = False
            new_items = {}
            for k, v in axioms.items():
                # A FloorDiv in implications could have became CleanDiv at this point, due to new facts
                # to the shapeEnv. This handles such issue but its not ideal. This is the only expression
                # simplification that depends on the global state of shape env.
                # TODO try to get rid of CleanDiv since it breaks the invariant thats simplifications of sympy
                # expressions only depend on the expression itself.
                if k.has(FloorDiv):
                    new_items.update({self.simplify(k): v})
            axioms.update(new_items)

        # Pattern matching
        if axioms is None:
            resimplify_floor_div(self.axioms)
            subst = self.axioms
        else:
            subst = {}
            for e in axioms:
                if e.free_symbols.issubset(expr.free_symbols):
                    subst.update(dict(self.get_implications(self.simplify(e))))

            resimplify_floor_div(subst)

        expr = expr.xreplace(subst)
        # TODO: compute hint might have gotten broken here

        fs = expr.free_symbols

        if not fs and (expr.is_number or expr.is_Boolean):
            return expr

        if var_to_range is None:
            var_ranges = self.var_to_range
        else:
            var_ranges = dict(var_to_range)

        symbol_info = tuple(
            (s, var_ranges.get(s), self.var_to_val.get(s), s in self.size_like)
            for s in sorted(fs, key=lambda s: str(s))  # TODO: speed up sort?
        )

        r = _maybe_evaluate_static_worker(
            expr, symbol_info, unbacked_only, size_oblivious
        )
        return r

    @_lru_cache
    def replace(self, expr: _SympyT) -> _SympyT:
        """Apply symbol replacements to any symbols in the given expression"""
        replacements = {}
        for s in expr.free_symbols:
            r = self._find(s)
            # Micro-optimization: only do replacements if r and s are different
            # Otherwise, xreplace is not a no-op and will trigger expensive
            # assumption queries if expr has a relational node.
            if not r.is_Symbol or r != s:
                replacements[s] = r
        if replacements:
            return safe_expand(expr.xreplace(replacements))
        else:
            return expr

    @_lru_cache
    def _update_divisible(self) -> None:
        new_divisible = set()
        for k in self.divisible:
            res = self.replace(k)
            if not res.is_number:
                new_divisible.add(k)

        self.divisible = new_divisible
        self._update_version_counter()

    @_lru_cache
    def simplify(self, expr: _SympyT) -> _SympyT:
        """Use known constraints and replacements to simplify the given expr"""
        expr = safe_expand(expr)
        expr = self.replace(expr)
        # TODO it would seem that this pass is not necessary given the
        # below replacement of // with /, but for nested FloorDivs
        # the non-recursive replacement doesn't work, and
        # recursive makes it hard to look up divisibility,
        # because existing divisibility info has FloorDiv in it, not /
        # for now just do a separate pass to catch common nested case
        if expr.has(FloorDiv):
            self._update_divisible()
            div_replacements = {}
            for atom in expr.atoms(FloorDiv):
                base, divisor = atom.args
                if isinstance(divisor, FloorDiv):
                    base1, divisor1 = divisor.args
                    if (
                        self.replace(Mod(base, divisor)) in self.divisible
                        and base == base1
                        and self.replace(Mod(base1, divisor1)) in self.divisible
                    ):
                        div_replacements[atom] = divisor1
            if div_replacements:
                expr = expr.xreplace(div_replacements)
                expr = safe_expand(expr)
        if expr.has(FloorDiv):
            div_replacements = {}
            pows = expr.atoms(sympy.Pow)
            rationals = expr.atoms(sympy.Rational).difference(expr.atoms(sympy.Integer))
            for fd in expr.atoms(FloorDiv):
                base, divisor = fd.args
                if self.replace(Mod(base, divisor)) in self.divisible:
                    div_replacements[fd] = CleanDiv(base, divisor)
            if div_replacements:
                new_expr = expr.xreplace(div_replacements)
                new_expr = safe_expand(new_expr)
                new_pows = new_expr.atoms(sympy.Pow)
                new_rationals = new_expr.atoms(sympy.Rational).difference(
                    new_expr.atoms(sympy.Integer)
                )
                # divisions simplified away
                if new_pows.issubset(pows) and new_rationals.issubset(rationals):
                    expr = new_expr
        return expr

    # TODO: overload for allow_none literal
    @lru_cache(256)
    def size_hint(
        self, expr: sympy.Basic, *, allow_none: bool = False
    ) -> Optional[sympy.Basic]:
        """
        Gets a size hint for a given expression from the underlying shapes we had.
        Does not introduce a guard, so only use this when you can guarantee that
        your code is still valid for arbitrary shapes (such as optimization decisions)
        """
        result_expr = safe_expand(expr).xreplace(self.var_to_val)
        if not result_expr.is_number:
            from torch.utils._sympy.singleton_int import SingletonInt

            if isinstance(result_expr, SingletonInt):
                return None
            r = self._maybe_evaluate_static(result_expr, compute_hint=True)
            if r is not None:
                return r
            if allow_none:
                return None

            if self.oblivious_var_to_val:
                # See https://github.com/pytorch/pytorch/issues/137100#issuecomment-2495778113
                correct_hint = result_expr.xreplace(self.oblivious_var_to_val)
                counterfactual_hint = result_expr.xreplace(
                    {k: max(v, 2) for k, v in self.oblivious_var_to_val.items()}
                )
                if (
                    not correct_hint.free_symbols
                    and not counterfactual_hint.free_symbols
                ):
                    if correct_hint == counterfactual_hint:
                        log.info("oblivious_size hit %s -> %s", expr, correct_hint)
                        return correct_hint
                    else:
                        log.info(
                            "oblivious_size counterfactual failed %s -> %s != %s",
                            expr,
                            correct_hint,
                            counterfactual_hint,
                        )
                else:
                    log.info(
                        "oblivious_size miss %s -> %s (counterfactual: %s)",
                        expr,
                        correct_hint,
                        counterfactual_hint,
                    )

            if self.unbacked_var_to_val:
                unsound_expr = result_expr.xreplace(self.unbacked_var_to_val)
                if not unsound_expr.free_symbols:
                    log.warning(
                        "propagate_real_tensors size_hint(%s) -> %s", expr, unsound_expr
                    )
                    trace_structured(
                        "propagate_real_tensors",
                        metadata_fn=lambda: {
                            "expr": repr(expr),
                            "result": repr(unsound_expr),
                            "stack": structured.from_traceback(
                                CapturedTraceback.extract(skip=1).summary()
                            ),
                        },
                    )
                    self.defer_runtime_assert(
                        sympy.Eq(result_expr, unsound_expr),
                        f"propagate_real_tensors: {result_expr} == {unsound_expr}",
                    )
                    return unsound_expr

            raise self._make_data_dependent_error(result_expr, expr)
        return result_expr

    # NB: keep in sync with size_hint
    @lru_cache(256)
    def has_hint(self, expr: sympy.Expr) -> bool:
        result_expr = safe_expand(expr).xreplace(self.var_to_val)
        return (
            result_expr.is_number
            or self._maybe_evaluate_static(result_expr) is not None
        )

    def _make_data_dependent_error(
        self,
        expr: sympy.Basic,
        unhinted_expr: sympy.Basic,
        *,
        size_oblivious_result: Optional[sympy.Basic] = None,
    ) -> GuardOnDataDependentSymNode:
        # TODO: in a Dynamo context, having user code, and having the
        # name of the local, will be much better
        size_like_symbols = []
        for s in expr.free_symbols:
            stacktrace = "".join(self.var_to_stack[s].format())
            self.log.debug(
                "Data dependent variable '%s' allocated at:\n%s", s, stacktrace
            )
            if s in self.size_like:
                size_like_symbols.append(s)
        size_oblivious_result_msg = ""
        if size_oblivious_result is not None:
            size_oblivious_result_msg = (
                f"ATTENTION: guard_size_oblivious would fix the error, evaluating expression to {size_oblivious_result}.\n"
                "Maybe you need to add guard_size_oblivious to framework code, see doc below for more guidance.\n\n"
            )
        sloc, maybe_extra_debug = self._get_stack_summary(True)
        if expr.is_integer:  # type: ignore[attr-defined]
            desc = (
                "Could not extract specialized integer from data-dependent expression"
            )
        else:
            desc = "Could not guard on data-dependent expression"
        msg = (
            f"{desc} {expr} (unhinted: {unhinted_expr}).  "
            f"(Size-like symbols: {', '.join(map(str, size_like_symbols)) or 'none'})\n\n"
            f"{size_oblivious_result_msg}"
            f"Caused by: {sloc}\n"
            'For more information, run with TORCH_LOGS="dynamic"\n'
            "For extended logs when we create symbols, also add "
            f"TORCHDYNAMO_EXTENDED_DEBUG_CREATE_SYMBOL=\"{','.join(map(str, expr.free_symbols))}\"\n"
            "If you suspect the guard was triggered from C++, add TORCHDYNAMO_EXTENDED_DEBUG_CPP=1\n"
            "For more debugging help, see "
            "https://docs.google.com/document/d/1HSuTTVvYH1pTew89Rtpeu84Ht3nQEFTYhAX3Ypa_xJs/edit?usp=sharing\n"
            + maybe_extra_debug
            # TODO: Help text about how to use our runtime tests to fix this
            # problem
        )
        return GuardOnDataDependentSymNode(expr, msg)

    def _update_var_to_range(
        self,
        symbol: sympy.Symbol,
        vr: ValueRanges,
        vr_sloc: Optional[ValueRangesSLoc] = None,
        *,
        is_constraint: bool = False,
    ) -> None:
        lower, upper = vr.lower, vr.upper

        # If we have a size-like unbacked SymInt, refuse to refine the range to be
        # less than two.  This is because when we intersect this range
        # with [2, inf] for size oblivious tests, the range would be
        # unsatisfiable.  In other words, once you have a size-like
        # unbacked SymInt, we can never learn that it is exactly zero or one,
        # because we would now give inconsistent results for all size
        # oblivous tests!
        if upper < 2 and symbol in self.size_like:
            vr = ValueRanges(lower, 2)

        # Updates the range and the guards corresponding to each bound of the symbol.
        if symbol not in self.var_to_range:
            self.log.debug("_update_var_to_range %s = %s (new)", symbol, vr)
            self.var_to_range[symbol] = vr
            if vr_sloc is None:
                sloc = self._get_sloc()
                vr_sloc = ValueRangesSLoc(sloc, sloc)
            self.var_to_range_sloc[symbol] = vr_sloc
        else:
            old = self.var_to_range[symbol]
            new = old & vr
            if new != old:
                if vr_sloc is None:
                    sloc = self._get_sloc()
                    vr_sloc = ValueRangesSLoc(sloc, sloc)
                if new.lower != old.lower:
                    self.var_to_range_sloc[symbol].lower = vr_sloc.lower
                if new.upper != old.upper:
                    self.var_to_range_sloc[symbol].upper = vr_sloc.upper
                self.var_to_range[symbol] = new
                self.log.debug("_update_var_to_range %s = %s (update)", symbol, new)

        if (v := self.var_to_val.get(symbol)) is not None:
            r = self.var_to_range[symbol]
            if v not in r:
                # For constraint failure, delay this for later
                # TODO: Rework all of this, the constraint logic is very
                # duplicative with regular reasoning
                if not is_constraint:
                    assert v in r, f"{v} not in {r}"

    def _set_replacement(self, a: sympy.Symbol, tgt: sympy.Expr, msg: str) -> None:
        """
        Adds or updates a replacement for a symbol.
        Use this instead of `self.replacements[a] = tgt`.
        """
        if tgt == self.replacements.get(a, None):
            return

        if a in tgt.free_symbols:
            return

        # Precondition: a == tgt
        assert isinstance(a, sympy.Symbol)

        if (
            self.allow_complex_guards_as_runtime_asserts
            and not _is_supported_equivalence(tgt)
        ):
            return  # continuing leads to placeholder shapes having complex expressions that we can't resolve

        # Handles nested tensor symbolic variables which don't have
        # var_to_range bounds
        tgt_bound = None
        if a in self.var_to_range:
            src_bound = self.var_to_range[a]

            # First, refine the value range of a based on the computed value range
            # of tgt.  This is always OK to do, even if we decide not to do the
            # substitution in the end.  This might be a no-op, if a already has
            # a tighter bound
            tgt_bound = self.bound_sympy(tgt)
            self._update_var_to_range(a, tgt_bound)

            # Next, check if we can update the range of free symbols in tgt
            # based on the range in a. But only do it if:
            #  - the source bound non-trivially improves over what we get out of
            #    the existing bounds.
            #  - the replacement is univariate and we can invert the tgt expression
            if not tgt_bound.issubset(src_bound) and len(tgt.free_symbols) == 1:
                b = next(iter(tgt.free_symbols))
                # Try to invert the equality
                r = try_solve(sympy.Eq(a, tgt), b, floordiv_inequality=False)
                if r is not None:
                    self.log.debug(
                        "set_replacement: solve for %s in %s == %s gives %s",
                        b,
                        a,
                        tgt,
                        r,
                    )
                    # The solution here can be non-integral, for example, if
                    # we have s0 = 2*s1, then s1 = s0/2.  What we would like
                    # to do is calculated the bounds in arbitrary precision,
                    # and then requantize the bound to integers when we are
                    # done.
                    rat_b_bound = self.bound_sympy(r[1])
                    b_bound = ValueRanges(
                        CeilToInt(rat_b_bound.lower), FloorToInt(rat_b_bound.upper)
                    )
                    self._update_var_to_range(b, b_bound, self.var_to_range_sloc[a])
                    tgt_bound = self.bound_sympy(tgt)
                    assert tgt_bound.issubset(src_bound)

            # TODO: Should we propagate size-like-ness?
            #
            # Pros: if u0 is size-like, intuitively u0 == u1 should cause u1
            # to become size-like.
            #
            # Cons: if u0 is size-like, what about u0 - 1 == u1?  You CAN'T
            # propagate in this case, because what if u0 == 0, then u1 is negative
            # and clearly isn't a size.  So, at minimum, any f(x) whose value
            # range isn't [0, inf] given x in [0, inf] cannot propagate
            # size-like-ness.  But there are many situations where you could
            # imagine u1 is going to be size-like and actually you just didn't
            # have a refined enough value range on u0.  Since even innocuous
            # looking arithmetic operations can destroy size-like-ness, it's
            # best to not propagate it at all and force the user to annotate it
            # as necessary.
            #
            # Compromise: we preserve size-like-ness only for exact equality
            # and nothing else.
            if a in self.size_like and isinstance(tgt, sympy.Symbol):
                self.size_like.add(tgt)
            elif isinstance(tgt, sympy.Symbol) and tgt in self.size_like:
                self.size_like.add(a)

            # Now, decide if we will do the substitution.
            #
            #  - If the source has a non-trivial range, only substitute if
            #    we preserve this range.  Note that we may have propagated
            #    the src_range to free variables in tgt when tgt is univariate
            #    and we could find an inverse, which helps us achieve this.
            #    This ensures we never "forget" about user defined ranges,
            #    even if they end up being defined on composite formulas
            #    like s0 + s1.
            #
            #  - If the variable is unbacked, only substitute if the substitution
            #    would preserve the bounds also under size-like-ness conditions.

            if not tgt_bound.issubset(src_bound):
                self.log.debug(
                    "skipped set_replacement %s = %s (%s) [%s not subset of %s]",
                    a,
                    tgt,
                    msg,
                    tgt_bound,
                    src_bound,
                )
                return
            elif a in self.size_like:
                tgt_bound_so = self.bound_sympy(tgt, size_oblivious=True)
                src_bound_so = self.bound_sympy(a, size_oblivious=True)
                if not tgt_bound_so.issubset(src_bound_so):
                    self.log.debug(
                        "skipped set_replacement %s = %s (%s) "
                        "[%s not subset of %s (size-oblivious conditions)]",
                        a,
                        tgt,
                        msg,
                        tgt_bound_so,
                        src_bound_so,
                    )
                    return

        if isinstance(tgt, (sympy.Integer, sympy.Float)):
            # specializing to a constant, which is likely unexpected (unless
            # you specified dynamic=True)

            user_tb = TracingContext.extract_stack()
            trace_structured(
                "symbolic_shape_specialization",
                metadata_fn=lambda: {
                    "symbol": repr(a),
                    "sources": [s.name() for s in self.var_to_sources.get(a, [])],
                    "value": repr(tgt),
                    "reason": msg,
                    "stack": structured.from_traceback(
                        CapturedTraceback.extract(skip=1).summary()
                    ),
                    "user_stack": structured.from_traceback(user_tb)
                    if user_tb
                    else None,
                },
            )

            if config.print_specializations:
                self.log.warning(
                    "Specializing %s to %s", self.var_to_sources[a][0].name(), tgt
                )
                self.log.debug("SPECIALIZATION", stack_info=True)
        log.info("set_replacement %s = %s (%s) %s", a, tgt, msg, tgt_bound)
        self.replacements[a] = tgt
        # NB: the replacement may get refined, but the user will find the
        # FIRST one most useful (TODO: Maybe we could consider tracking all of
        # them)
        if a not in self.replacements_slocs:
            self.replacements_slocs[a] = self._get_sloc()
        self._update_version_counter()

        # When specializing 'a == tgt', the equality should be also conveyed to
        # Z3, in case an expression uses 'a'.
        self._add_target_expr(sympy.Eq(a, tgt, evaluate=False))

    def _add_divisible(self, expr: sympy.Expr) -> None:
        self.divisible.add(expr)
        self._update_version_counter()

    @_lru_cache
    @record_shapeenv_event()
    def _find(self, a: sympy.Symbol) -> sympy.Expr:
        """
        Implements a DSU-like algorithm to find the variable that represents a
        Also handles transitive non-identity replacements.

        a: b + c
        c: d
        """
        if a not in self.replacements:
            return a
        res = self.replacements[a]
        cur_replace = {s: self._find(s) for s in res.free_symbols}
        replaced, changed = self.replacements[a]._xreplace(cur_replace)
        if changed:
            self._set_replacement(a, replaced, "find")
        return self.replacements[a]

    @lru_cache(256)
    def _maybe_guard_rel(self, expr: sympy.Rel) -> None:
        """
        The relational guard is guarded to be true.  Use this information to
        simplify shapes (i.e. a == b or a % 5 == 0)
        """
        assert isinstance(expr, sympy.Rel)

        # A good example of what goes wrong if you don't do this is
        # python test/functorch/test_aotdispatch.py -k
        # test_aot_autograd_symbolic_module_exhaustive_nn_LazyConv3d_cpu_float32
        if isinstance(expr, sympy.Ne):
            return

        free = list(expr.free_symbols)

        assert (
            len(free) > 0
        ), f"The expression should not be static by this point: {expr}"
        # In case of really gnarly expression, we don't blow up
        if len(free) > 5:
            return

        # Prioritize unbacked symints for solving by ordering them last.
        # Prefer to simplify out lexicographically higher symbols (i.e. simplify out s4 over s3).
        #   (NB: this unfortunately isn't strictly equivalent to simplifying out newer symbols)
        # Prefer to simplify out symbols with ephemeral sources.
        def _smart_symbol_sort(x: sympy.Symbol) -> Tuple[int, int, str]:
            has_only_ephemeral_sources = x in self.var_to_sources and all(
                s.is_ephemeral() for s in self.var_to_sources[x]
            )
            # NB: size_hint is int, not sympy.Expr, do not use int_oo here
            hint_size = self.size_hint(x, allow_none=True)
            if hint_size is None:
                size = sys.maxsize
            elif symbol_is_type(x, SymT.SIZE):
                assert isinstance(hint_size, sympy.Expr)
                size = int(hint_size)
            else:
                size = sys.maxsize
            name = x.name
            # 1 puts ephemeral sourced symbols first when sorting in reverse
            return (1 if has_only_ephemeral_sources else 0, size, name)

        free = sorted(free, key=_smart_symbol_sort, reverse=True)  # type: ignore[attr-defined]
        lhs = expr.lhs
        rhs = expr.rhs

        self._refine_ranges(expr)

        # The rest of this stuff is for equality only
        if not isinstance(expr, sympy.Eq):
            return

        if not expr.has(Mod):
            try:
                floor_div_atoms = lhs.atoms(FloorDiv).union(rhs.atoms(FloorDiv))
                if len(floor_div_atoms) > 0 and any(
                    a.divisor != 1 for a in floor_div_atoms
                ):
                    raise NotImplementedError

                # Never replace unbacked symbols with other unbacked symbols.
                # This is error prone because you can cause references to
                # unbacked symbols to time travel backwards.  E.g.,
                #
                # u1 = x.item()
                # ... use of u1 ...
                # u2 = y.item()
                # u3 = z.item()
                # torch._check(u1 == u2 + u3)
                #
                # If you replace u1 with u2 + u3, then the use of u1 now
                # references u2 and u3 prior to them actually being bound at
                # runtime.  It's pretty inconvenient to setup control
                # dependencies for substitutions, so ban it entirely.
                def trivial_solve(lhs: sympy.Expr, rhs: sympy.Expr) -> bool:
                    if isinstance(lhs, sympy.Symbol):
                        if free_unbacked_symbols(lhs) and not free_unbacked_symbols(
                            rhs
                        ):
                            return True
                        if symbol_is_type(lhs, SymT.FLOAT):
                            return True
                        # TODO: Maybe trivial solutions for int should also be
                        # done?
                    return False

                # short-circuit when no solving is needed
                if trivial_solve(lhs, rhs):
                    self._set_replacement(lhs, self._find(rhs), "trivial_lhs")
                elif trivial_solve(rhs, lhs):
                    self._set_replacement(rhs, self._find(lhs), "trivial_rhs")
                else:
                    r = try_solve(expr, free[0], floordiv_inequality=False)
                    if r is not None and all(
                        t.is_integer for t in sympy.preorder_traversal(r[1])
                    ):
                        new_var = self._find(r[1])
                        ok = len(free_unbacked_symbols(new_var)) == 0
                        if ok:
                            self._set_replacement(free[0], new_var, "solve")
            except NotImplementedError:
                pass
        if expr.has(Mod):
            mod_expr = next(iter(expr.atoms(Mod)))
            try:
                r = try_solve(expr, mod_expr, floordiv_inequality=False)
                if r is not None and r[1] == 0:
                    self._add_divisible(mod_expr)
                    # This is a little bit of extra logic to make things like
                    # torch.empty(i0, q).view(c, -1, q) work out
                    p, q = mod_expr.args
                    if (
                        isinstance(q, sympy.Number)
                        and isinstance(p, sympy.Mul)
                        and len(p.args) == 2
                    ):
                        c, i0 = p.args
                        # Given Mod(c * i0, q) == 0
                        if (
                            isinstance(c, sympy.Number)
                            and isinstance(i0, sympy.Symbol)
                            and self.is_unbacked_symint(i0)
                        ):
                            # We have Mod(i0, q / c) == 0, which means we can
                            # rewrite i0 as (q / gcd(q, c)) * i1
                            d = q / sympy.gcd(q, c)  # TODO: CleanDiv?
                            i1 = self.create_unbacked_symint().node.expr
                            # Propagate the value ranges.  It doesn't really
                            # matter if we use truediv or floordiv, because we
                            # have established divisibility.
                            self._update_var_to_range(
                                i1,
                                SymPyValueRangeAnalysis.floordiv(
                                    self.var_to_range[i0], ValueRanges.wrap(d)
                                ),
                            )
                            # Propagate size-like-ness
                            if i0 in self.size_like:
                                self.size_like.add(i1)
                            self._set_replacement(i0, d * i1, "divisibility")

            except NotImplementedError:
                pass
        return

    # See: Note - On 0/1 specialization
    def _default_value_range(self) -> ValueRanges:
        lower = 2 if self.specialize_zero_one else 0
        return ValueRanges(lower, int_oo)

    def _default_unspecified_value_range(self) -> ValueRanges:
        return ValueRanges.unknown_int()

    @_lru_cache
    def _simplify_floor_div(self, expr: sympy.Expr) -> sympy.Expr:
        floor_divs = tuple(expr.atoms(FloorDiv))
        # we expect floor_divs to be exact,
        # and thus add the guards for the exact floordivs,
        # even if tracing doesn't require them otherwise
        for fd in reversed(floor_divs):
            base, divisor = fd.args
            mod_expr = Mod(base, divisor)
            eq_expr = sympy.Eq(mod_expr, 0)
            # add necessary mod guards
            self.evaluate_expr(eq_expr)
        return self.simplify(expr)

    # We're about to add a guard/runtime assert, check if the ShapeEnv is frozen
    # and if so issue a warning
    def _check_frozen(self, expr: sympy.Basic, concrete_val: sympy.Basic) -> None:
        if self.frozen:
            self.counter["ignored_backward_guard"] += 1
            signpost_event(
                "dynamic",
                "evaluate_expr_frozen",
                {
                    **self.co_fields,
                    "ignored_guard": f"{expr} == {concrete_val}",
                    # no version = original state (this signpost is expected)
                    # version 2 = dynamic backwards is eagerly compiled
                    "version": 2,
                },
            )
            log.warning(
                "Ignored guard %s == %s, this could result in accuracy problems",
                expr,
                concrete_val,
                # only print stack trace when debug mode is on (e.g. TORCH_LOGS="dynamic")
                stack_info=True if log.getEffectiveLevel() < logging.WARNING else False,
            )

    def _get_stack_summary(
        self, is_debug: bool = False, framework_loc: Optional[str] = None
    ) -> Tuple[SLoc, str]:
        floc: Optional[Union[str, traceback.FrameSummary]] = framework_loc
        if floc is None:
            frame = inspect.currentframe()
            try:
                while frame is not None:
                    if frame.f_code.co_filename not in uninteresting_files():
                        floc = traceback.FrameSummary(
                            frame.f_code.co_filename,
                            frame.f_lineno,
                            frame.f_code.co_name,
                        )
                        break
                    frame = frame.f_back
            finally:
                del frame

        # NB: this stack is truncated, but it's fine because the main
        # stack_info will give you the rest of the info you need
        maybe_user_loc = None
        user_tb = TracingContext.extract_stack()
        if user_tb:
            idx = len(user_tb) - 1
            while idx > 0 and user_tb[idx].filename in uninteresting_files():
                idx -= 1
            maybe_user_loc = format_frame(user_tb[idx], line=True)

        maybe_extra_debug = ""
        if is_debug and user_tb:
            maybe_extra_debug = (
                "\nUser Stack (most recent call last):\n"
                + "  (snipped, see stack below for prefix)\n"
                + "".join(traceback.format_list(user_tb))
            )
        if is_debug and config.extended_debug_cpp:
            cpp_stack = CapturedTraceback.extract(cpp=True)
            maybe_extra_debug += "\nC++ stack trace:\n" + "".join(cpp_stack.format())
        elif is_debug:
            maybe_extra_debug += (
                "\nFor C++ stack trace, run with " "TORCHDYNAMO_EXTENDED_DEBUG_CPP=1"
            )

        return SLoc(floc, maybe_user_loc), maybe_extra_debug

    # Pass in framework_loc to override the framework location info
    def _get_sloc(self, framework_loc: Optional[str] = None) -> SLoc:
        sloc, _ = self._get_stack_summary(framework_loc=framework_loc)
        return sloc

    def _log_guard(self, prefix: str, g: SympyBoolean, forcing_spec: bool) -> None:
        dtrace_structured(
            "guard_added",
            metadata_fn=lambda: {
                "expr": str(g),
                "stack": structured.from_traceback(
                    CapturedTraceback.extract(skip=1).summary()
                ),
                "symbol_to_sources": {
                    str(v): k
                    for k, v in self.source_to_var.items()
                    if v in g.free_symbols
                },
            },
        )
        trace_structured(
            "guard_added_fast",
            metadata_fn=lambda: {
                "expr": str(g),
                "user_stack": structured.from_traceback(TracingContext.extract_stack()),
                "stack": structured.from_traceback(
                    CapturedTraceback.extract(skip=1).summary()
                ),
            },
        )
        if self.log.isEnabledFor(logging.INFO):
            str_g = str(g)
            is_debug = (
                config.extended_debug_guard_added is not None
                and str_g == config.extended_debug_guard_added
            )
            sloc, maybe_extra_debug = self._get_stack_summary(is_debug)
            maybe_more_info = ""
            if not is_debug:
                maybe_more_info = (
                    ", for more info run with "
                    f'TORCHDYNAMO_EXTENDED_DEBUG_GUARD_ADDED="{str_g}"'
                )
            self.log.info(
                "%s %s [guard added] %s%s%s",
                prefix if not forcing_spec else f"{prefix} (forcing_spec)",
                str_g,
                sloc,
                maybe_more_info,
                maybe_extra_debug,
                stack_info=is_debug,
            )

    @lru_cache(256)
    @record_shapeenv_event(save_tracked_fakes=True)
    def evaluate_expr(
        self,
        orig_expr: sympy.Basic,
        hint: Optional[Union[int, bool, float]] = None,
        fx_node: Optional[torch.fx.Node] = None,
        size_oblivious: bool = False,
        *,
        forcing_spec: bool = False,
    ) -> sympy.Basic:
        try:
            return self._evaluate_expr(
                orig_expr, hint, fx_node, size_oblivious, forcing_spec=forcing_spec
            )
        except Exception:
            self.log.warning(
                "failed during evaluate_expr(%s, hint=%s, size_oblivious=%s, forcing_spec=%s",
                orig_expr,
                hint,
                size_oblivious,
                forcing_spec,
            )
            raise

    def _evaluate_expr(
        self,
        orig_expr: sympy.Basic,
        hint: Optional[Union[bool, int, float]] = None,
        fx_node: Optional[torch.fx.Node] = None,
        size_oblivious: bool = False,
        *,
        forcing_spec: bool = False,
    ) -> sympy.Basic:
        """
        Given an expression, evaluates it, adding guards if necessary
        """

        # TODO: split conjunctions and evaluate them separately

        if isinstance(
            orig_expr,
            (sympy.logic.boolalg.BooleanTrue, sympy.logic.boolalg.BooleanFalse),
        ):
            return orig_expr

        # Don't track this one
        @functools.lru_cache(None)
        def compute_concrete_val() -> sympy.Basic:
            if hint is None:
                # This is only ever called for expressions WITHOUT unbacked
                # symbols
                r = self.size_hint(orig_expr)
                assert r is not None
                return r
            else:
                return sympy.sympify(hint)

        concrete_val: Optional[sympy.Basic]

        # Check if:
        #   1. 'translation_validation' is set
        #   2. the corresponding 'fx_node' is not 'None'
        #   3. the guard should not be suppressed
        #   4. the guard doesn't contain backed symfloat symbols
        #      since z3 can't handle floats
        #
        # If all of the above check, we create an FX node representing the
        # actual expression to be guarded.
        node = None
        fresh = False
        if (
            self._translation_validation_enabled
            and fx_node is not None
            and not self._suppress_guards_tls()
            and not size_oblivious
            and not any(symbol_is_type(s, SymT.FLOAT) for s in orig_expr.free_symbols)
        ):
            # TODO: does this even worked with unbacked :think:
            concrete_val = compute_concrete_val()
            if concrete_val is sympy.true:
                node, fresh = self._create_fx_call_function(torch._assert, (fx_node,))
            elif concrete_val is sympy.false:
                neg, _ = self._create_fx_call_function(operator.not_, (fx_node,))
                node, fresh = self._create_fx_call_function(torch._assert, (neg,))
            else:
                eql, _ = self._create_fx_call_function(
                    operator.eq, (fx_node, concrete_val)
                )
                node, fresh = self._create_fx_call_function(torch._assert, (eql,))

            assert node is not None
            # If this is a fresh node, we have to remember the event index that
            # corresponds to this assertion node.
            # Reason: so that, given an assertion node, we can replay the ShapeEnv
            # events until the point where this assertion node was freshly created.
            if fresh:
                self._add_fx_node_metadata(node)

        # After creating the FX node corresponding to orig_expr, we must make sure that
        # no error will be raised until the end of this function.
        #
        # Reason: the translation validation may become invalid otherwise.
        #
        # If an error is raised before the end of this function, we remove the FX node
        # inserted, and re-raise the error.
        guard = None

        try:
            if orig_expr.is_number:
                self.log.debug("eval %s [trivial]", orig_expr)
                if hint is not None:
                    assert orig_expr == hint, f"{orig_expr} != {hint}"
                return orig_expr

            expr = orig_expr

            static_expr = self._maybe_evaluate_static(
                expr, size_oblivious=size_oblivious
            )
            if static_expr is not None:
                self.log.debug(
                    "eval %s == %s [statically known]", orig_expr, static_expr
                )
                if hint is not None:
                    assert static_expr == hint, f"{static_expr} != {hint}"
                return static_expr

            transmute_into_runtime_assert = False

            concrete_val = None
            if not (expr.free_symbols <= self.var_to_val.keys()):
                # TODO: dedupe this with _maybe_evaluate_static
                # Attempt to eliminate the unbacked SymInt
                new_expr = self._maybe_evaluate_static(expr, unbacked_only=True)
                assert new_expr is not None
                if not (new_expr.free_symbols <= self.var_to_val.keys()):
                    size_oblivious_result = None
                    if not size_oblivious:
                        size_oblivious_result = self._maybe_evaluate_static(
                            expr, size_oblivious=True
                        )

                    ok = False

                    # Last ditch
                    if (
                        self.oblivious_var_to_val
                        and not (
                            correct_hint := orig_expr.xreplace(
                                self.oblivious_var_to_val
                            )
                        ).free_symbols
                        and not (
                            counterfactual_hint := orig_expr.xreplace(
                                {
                                    k: max(2, v)
                                    for k, v in self.oblivious_var_to_val.items()
                                }
                            )
                        ).free_symbols
                        and correct_hint == counterfactual_hint
                    ):
                        # TODO: better logging
                        log.info(
                            "oblivious_size %s -> %s (passed counterfactual)",
                            orig_expr,
                            correct_hint,
                        )
                        concrete_val = correct_hint
                        # NB: do NOT transmute into runtime assert
                        ok = True

                    if (
                        not ok
                        and self.unbacked_var_to_val
                        and not (
                            unsound_result := orig_expr.xreplace(
                                self.unbacked_var_to_val
                            )
                        ).free_symbols
                    ):
                        log.warning(
                            "propagate_real_tensors evaluate_expr(%s) -> %s",
                            orig_expr,
                            unsound_result,
                        )
                        trace_structured(
                            "propagate_real_tensors",
                            metadata_fn=lambda: {
                                "expr": repr(orig_expr),
                                "result": repr(unsound_result),
                                "stack": structured.from_traceback(
                                    CapturedTraceback.extract(skip=1).summary()
                                ),
                            },
                        )
                        transmute_into_runtime_assert = True
                        concrete_val = unsound_result
                        ok = True

                    if not ok:
                        raise self._make_data_dependent_error(
                            expr.xreplace(self.var_to_val),
                            expr,
                            size_oblivious_result=size_oblivious_result,
                        )
                else:
                    expr = new_expr

            if concrete_val is None:
                concrete_val = compute_concrete_val()
            self._check_frozen(expr, concrete_val)

            if (
                config.inject_EVALUATE_EXPR_flip_equality_TESTING_ONLY
                and isinstance(hint, bool)
                and isinstance(expr, (sympy.Eq, sympy.Ne))
            ):
                expr = sympy.Not(expr)

            # Turn this into a boolean expression, no longer need to consult
            # concrete_val
            if concrete_val is sympy.true:
                g = cast(SympyBoolean, expr)
            elif concrete_val is sympy.false:
                g = sympy.Not(expr)
            else:
                g = sympy.Eq(expr, concrete_val)  # type: ignore[arg-type]

            if transmute_into_runtime_assert:
                self.defer_runtime_assert(
                    g, f"propagate_real_tensors: {orig_expr} == {concrete_val}"
                )
                return concrete_val

            if not self._suppress_guards_tls():
                if isinstance(g, sympy.Rel):
                    # TODO: If we successfully eliminate a symbol via equality, it
                    # is not actually necessary to save a guard for the equality,
                    # as we will implicitly generate a guard when we match that
                    # input against the symbol.  Probably the easiest way to
                    # implement this is to have maybe_guard_rel return a bool
                    # saying if it "subsumed" the guard (and therefore the guard
                    # is no longer necessary)
                    self._maybe_guard_rel(g)

                if not self.allow_complex_guards_as_runtime_asserts:
                    # at this point, we've evaluated the concrete expr value, and have
                    # flipped/negated the guard if necessary. Now we know what to guard
                    # or defer to runtime assert on.
                    guard = ShapeGuard(g, self._get_sloc())
                    self.guards.append(guard)
                    self.axioms.update(dict(self.get_implications(self.simplify(g))))
                else:
                    # it's fine to defer simple guards here without checking,
                    # the _maybe_guard_rel() call above will set replacements if possible,
                    # and so the result here will be statically known
                    self.defer_runtime_assert(g, f"evaluate_expr: {orig_expr}")

        except Exception:
            if fresh:
                self._remove_fx_node(node)
            raise
        else:
            if not self._suppress_guards_tls():
                if guard is not None:  # we might have deferred this to runtime assert
                    self._log_guard("eval", g, forcing_spec=forcing_spec)

                    for s in g.free_symbols:
                        self.symbol_guard_counter[s] += 1
                        # Forcing_spec to avoid infinite recursion
                        if (
                            not forcing_spec
                            and config.symbol_guard_limit_before_specialize is not None
                            and self.symbol_guard_counter[s]
                            > config.symbol_guard_limit_before_specialize
                        ):
                            # Force specialization
                            self.log.info(
                                "symbol_guard_limit_before_specialize=%s exceeded on %s",
                                config.symbol_guard_limit_before_specialize,
                                s,
                            )
                            self.evaluate_expr(s, forcing_spec=True)
            else:
                self._log_guard("eval [guard suppressed]", g, forcing_spec=forcing_spec)

        return concrete_val

    def cleanup(self) -> None:
        """
        Break reference cycles.

        This destroys the stacks. If you really want to keep them, we
        just need some way to break references on code objects.
        """
        for s in self.var_to_stack.values():
            s.cleanup()
        for ras in self.deferred_runtime_asserts.values():
            for ra in ras:
                ra.stack.cleanup()

    @lru_cache(256)
    @record_shapeenv_event(save_tracked_fakes=True)
    def defer_runtime_assert(
        self, orig_expr: SympyBoolean, msg: str, fx_node: Optional[torch.fx.Node] = None
    ) -> bool:
        """Create an assert that is checked at runtime

        Args:
            orig_expr (sympy.Expr): Boolean expression to assert is true
            msg (str): Message to display on assertion failure
            fx_node (Optional, torch.fx.Node): node in ``self.graph`` corresponding
                to the expression, if applicable

        """
        expr = orig_expr

        # TODO: split conjunctions and evaluate them separately

        static_expr = self._maybe_evaluate_static(expr)
        if static_expr is not None:
            self.log.debug(
                "runtime_assert %s == %s [statically known]", orig_expr, static_expr
            )
            # TODO: assert bool(static_expr)
            return bool(static_expr)

        # Attempt to eliminate the unbacked SymInt
        new_expr = self._maybe_evaluate_static(expr, unbacked_only=True)
        assert new_expr is not None
        if (
            not self.prefer_deferred_runtime_asserts_over_guards
            and new_expr.free_symbols <= self.var_to_val.keys()
        ):
            # Do a normal guard
            return self.evaluate_expr(new_expr, fx_node=fx_node)
        # NB: Don't use new_expr as expr; it could contain gunk like shape0
        # which we don't want to guard on

        if (
            self._translation_validation_enabled
            and fx_node is not None
            and not self._suppress_guards_tls()
        ):
            node, fresh = self._create_fx_call_function(torch._assert, (fx_node,))
            assert node is not None
            if fresh:
                self._add_fx_node_metadata(node)

        if not self._suppress_guards_tls():
            # If you're here because of this assert, read Note [Backwards runtime asserts]
            # in torch/_inductor/graph.py
            if self.runtime_asserts_frozen:
                log.warning("runtime_asserts_frozen but then got %s", expr)
            self._check_frozen(expr, sympy.true)
            # eliminate symbols on equality tests / refine ranges
            if isinstance(expr, sympy.Rel):
                self._maybe_guard_rel(expr)

            # canonicalise to remove equations that are trivially equal
            orig_expr = expr
            expr = canonicalize_bool_expr(expr)
            stack = CapturedTraceback.extract(skip=1)
            ra = RuntimeAssert(expr, msg, stack)
            # TODO: Do this in a way that is less janky than int(s.name[1:])
            cands = sorted(
                (s for s in expr.free_symbols if symbol_is_type(s, SymT.UNBACKED_INT)),
                key=lambda s: int(s.name[1:]),
            )
            # Is None when prefer_deferred_runtime_asserts_over_guards=True
            # and the guard in question has no unbacked SymInts in front
            ix = cands[-1] if cands else None
            self.deferred_runtime_asserts.setdefault(ix, []).append(ra)
            self.axioms.update(dict(self.get_implications(self.simplify(expr))))
            self.num_deferred_runtime_asserts += 1
            self._update_version_counter()
            self._log_guard("runtime_assert", orig_expr, forcing_spec=False)
        else:
            self._log_guard(
                "runtime_assert [guard suppressed]", orig_expr, forcing_spec=False
            )

        return True

    # Refines the ranges of the variables present in 'guard'.
    #
    # This function tries to refine the range of the variables inside
    # 'guard' by reasoning about it. Specifically, when 'guard' is a
    # 'sympy.Relational' operation.
    #
    # It does mainly 3 things:
    #   1. Tries to isolate a variable in the left-hand side
    #   2. Compute the value range of the right-hand side
    #   3. Update the value range of the variable, if better
    def _refine_ranges(self, expr: SympyBoolean) -> None:
        expr = self.simplify(expr)

        for symbol in expr.free_symbols:
            assert isinstance(symbol, sympy.Symbol)

            if isinstance(self.var_to_val.get(symbol, None), SingletonInt):
                # Skip var_to_range logic for SingletonInt which is only used
                # for jagged layout NestedTensors today
                continue

            r = try_solve(expr, symbol)

            if r is None or not (symbol.is_integer and r[1].is_integer):
                # Range refinement only supports integer symbols for now.
                # There are lots of SymPy bugs when it comes to comparing
                # reals and integers, so we skip that for now.
                continue

            r_expr, rhs = r
            vr = self.var_to_range[symbol]
            lower, upper = vr.lower, vr.upper

            rhs_vr = bound_sympy(rhs, self.var_to_range)

            # Let's suppose that we have a preexisting range for x [0, 100].
            # Now, we issue a guard x > y, where the range for y is [50, 150].
            # Then, lower = 0, rhs_vr.lower = 50 and therefore refinement can happen,
            # refining x to [51, 100], since x must be greater than y, but the lowest
            # y could be is 50.
            #
            # sympy.Eq may update both lower and upper bounds.
            # sympy.G{t,e} may update the lower bound, only.
            # sympy.L{t,e} may update the upper bound, only.
            if lower < rhs_vr.lower and isinstance(
                r_expr, (sympy.Eq, sympy.Ge, sympy.Gt)
            ):
                # Strictly greater relations allow us to refine a bit more, since
                # x < y implies that the lower bound for x is: y + 1.
                lower = rhs_vr.lower + int(isinstance(r_expr, sympy.Gt))
            if upper > rhs_vr.upper and isinstance(
                r_expr, (sympy.Eq, sympy.Le, sympy.Lt)
            ):
                upper = rhs_vr.upper - int(isinstance(r_expr, sympy.Lt))

            # Do nothing if the new value range is no better than what we already have.
            if vr == ValueRanges(lower, upper):
                continue

            # Updates the range and the guards corresponding to each bound of the symbol.
            self._update_var_to_range(symbol, ValueRanges(lower, upper))
            # If the range is refined to singleton, set replacement
            if self.var_to_range[symbol].is_singleton():
                self._set_replacement(
                    symbol,
                    self.var_to_range[symbol].lower,
                    "range_refined_to_singleton",
                )

            # Clears the cache, since this update can change the result.
            self._maybe_evaluate_static.cache_clear()

    @lru_cache(maxsize=None)
    @record_shapeenv_event()
    def constrain_symbol_range(
        self, s: sympy.Symbol, compiler_min: int, compiler_max: int
    ) -> None:
        upd_vr = ValueRanges(compiler_min, compiler_max)
        old_vr = self.var_to_range.get(s, ValueRanges.unknown())
        self._update_var_to_range(s, upd_vr)
        if (new_vr := self.var_to_range[s]) != old_vr:
            log.info(
                "constrain_symbol_range %s [%s, %s]", s, new_vr.lower, new_vr.upper
            )


def _is_int(expr: object) -> bool:
    return isinstance(expr, SymInt) and expr.node.expr.is_number


# WARNING: This is legacy, DO NOT USE
def _is_dim_dynamic(t: torch.Tensor, d: int) -> bool:
    return hasattr(t, "_dynamo_dynamic_indices") and d in t._dynamo_dynamic_indices


class PropagateUnbackedSymInts(torch.fx.Interpreter):
    def run_node(self, n: torch.fx.Node) -> Result:
        """
        Run an FX node, propagating unbacked Symbol bindings to the new fake tensor
        """
        from torch._guards import detect_fake_mode

        result = super().run_node(n)
        rebind_unbacked(detect_fake_mode().shape_env, n, result)
        return result


def _find_user_code_frame() -> Optional[types.FrameType]:
    frame = inspect.currentframe()
    while frame is not None:
        if not frame.f_code.co_filename.startswith(
            os.path.dirname(inspect.getfile(torch)) + os.path.sep
        ):
            break
        frame = frame.f_back
    return frame


def _blame_user_code(e: Exception, frame: types.FrameType) -> None:
    frame_summary = traceback.FrameSummary(
        frame.f_code.co_filename,
        frame.f_lineno,
        frame.f_code.co_name,
    )
    msg = e.args[0]
    msg += "\n\nThe following call raised this error:\n" + "".join(
        traceback.StackSummary.from_list([frame_summary]).format()
    )
    e.args = (msg,)


class _PythonMsgPrinter(PythonPrinter):
    """
    Util printer that replaces sympy symbols with their source-level names
    and renders sympy relational operators (e.g., Eq, Ne, Ge, Le) inline
    (i.e., as ==, !=, >, <).
    """

    def __init__(self, src_map: Dict[str, List[str]]) -> None:
        super().__init__()
        self.src_map = src_map

    def _print_Symbol(self, sym: sympy.Symbol) -> str:
        return self.src_map[sym.name][0]


def _suggest_torch_checks(
    e: GuardOnDataDependentSymNode, src_map: DefaultDict[str, List[str]]
) -> None:
    # extract the unresolved condition on unbacked symints in the error
    cond = e.cond
    diff = ", ".join(s.name for s in cond.free_symbols if s.name not in src_map)
    if diff:
        log.warning("Unable to find user code corresponding to {%s}", diff)
        return
    printer = _PythonMsgPrinter(src_map)
    msg = e.args[0]
    msg += "\nTo fix the error, insert one of the following checks before this call:"
    # suggested fixes to resolve `cond`` are to tell the compiler to assume
    # either `cond` or its negation (the user will need to select which)
    suggested_fixes = [
        f"torch._check({printer.doprint(cond)})",
        f"torch._check({printer.doprint(sympy.Not(cond))})",
    ]
    for i, fix in enumerate(suggested_fixes):
        msg += f"\n  {i + 1}. {fix}"
    src_mapped = ", ".join(
        f"`{s}` with {' or '.join(src_map[s])}"
        for s in sorted(s.name for s in cond.free_symbols)
    )
    msg += f"\n\n(These suggested fixes were derived by replacing {src_mapped} in {cond} and its negation.)"
    e.args = (msg,)


def _suggest_fixes_for_data_dependent_error_non_strict(
    e: GuardOnDataDependentSymNode,
) -> None:
    """
    Given a raised data-dependent error, add the following to the error message:
    1. the closest user code location that raised the error;
    2. suggested fixes for the error in terms of live variables at that location.
    """

    # walk the stack up from the data-dependent error until a non-torch frame is found
    frame = _find_user_code_frame()
    if frame is not None:
        # add frame info to error message
        _blame_user_code(e, frame)

        # map symbol names reachable via frame locals to their source-level names
        src_map = defaultdict(list)
        for var, val in frame.f_locals.items():
            try:
                tree_leaves_with_path = pytree.tree_leaves_with_path(val)
            except ValueError:
                log.warning(
                    "pytree.tree_leaves_with_path failed for value of type {%s} in local variable {%s}",
                    type(val),
                    var,
                )
                continue
            # figure out how to access any symbol inside `val` through `var`
            for path, leaf in tree_leaves_with_path:
                name = var + pytree.keystr(path)
                if isinstance(leaf, torch.SymInt):
                    src_map[str(leaf.node.expr)].append(name)
                elif isinstance(leaf, torch.Tensor):
                    for i, dim in enumerate(leaf.shape):
                        if isinstance(dim, torch.SymInt):
                            src_map[str(dim.node.expr)].append(f"{name}.shape[{i}]")

        # add suggested torch.check()s based on `src_map` to the error message
        # replacing unbacked symints in the unresolved condition in the error
        _suggest_torch_checks(e, src_map)
