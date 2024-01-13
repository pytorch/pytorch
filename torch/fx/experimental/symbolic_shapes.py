import builtins
import collections
import functools
import inspect
import itertools
import logging
import math
import operator
import re
import sys
import threading
import traceback
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from typing import (
    Any,
    cast,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    Union,
    TYPE_CHECKING
)

import torch
import torch.fx
import torch.fx.traceback as fx_traceback
from torch.fx.experimental import _config as config

from torch.fx.experimental.recording import (
    FakeTensorMeta,
    ShapeEnvEvent,
    record_shapeenv_event,
    replay_shape_env_events,
    shape_env_check_state_equal
)
from torch.fx.experimental.sym_node import SymNode, SymTypes

# NB: The sym_* functions are used via getattr() and must be imported here.
from torch import SymBool, SymFloat, SymInt
from torch._guards import ShapeGuard, Source, TracingContext
from torch.utils._python_dispatch import is_traceable_wrapper_subclass
from torch.utils._sympy.functions import FloorDiv, Mod, IsNonOverlappingAndDenseIndicator
from torch.utils._sympy.solve import try_solve
from torch.utils._sympy.value_ranges import bound_sympy, SymPyValueRangeAnalysis, ValueRanges, ValueRangeError
from torch.utils._sympy.singleton_int import SingletonInt
from torch.utils._traceback import format_frame, CapturedTraceback
from torch._utils_internal import signpost_event

from torch._logging import LazyString

if TYPE_CHECKING:
    from torch._dynamo.source import TensorPropertySource

InputList = List
DimList = List

log = logging.getLogger(__name__)

class GuardOnDataDependentSymNode(RuntimeError):
    pass

import sympy
from sympy.printing.str import StrPrinter
from sympy.printing.precedence import precedence, PRECEDENCE

aten = torch._ops.ops.aten  # type: ignore[has-type]

__all__ = [
    "has_symbolic_sizes_strides", "create_contiguous", "ShapeEnv", "is_concrete_int",
    "guard_int", "guard_float", "guard_scalar", "canonicalize_bool_expr",
    "hint_int", "SYMPY_INTERP", "free_symbols", "is_symbol_binding_fx_node",
    "is_concrete_bool", "is_singleton", "SHAPEENV_EVENT_KEY", "CURRENT_NODE_KEY",
    "has_free_symbols", "sym_eq", "SymbolicContext", "StatelessSymbolicContext",
    "StatefulSymbolicContext", "SubclassSymbolicContext"
]

# FX node metadata keys for symbolic shape FX graph.
SHAPEENV_EVENT_KEY = "shapeenv_event"
CURRENT_NODE_KEY = "current_node"

# These are modules that contain generic code for interacting with ShapeEnv
# which are unlikely to identify a particular interesting guard statement
@lru_cache(None)
def uninteresting_files():
    import torch._inductor.sizevars
    import torch._library.abstract_impl
    mods = [
        sys.modules[__name__],
        torch.fx.experimental.recording,
        torch.fx.experimental.sym_node,
        torch,
        torch._inductor.sizevars,
        torch._library.abstract_impl,
    ]
    return {inspect.getfile(m) for m in mods}

# We don't bother with the metaclass as all of the dispatching logic happens
# entirely from Python
#
# Didn't bother with ancestors for now, unlikely to have multiple modes for
# symints right now

class ConstraintViolationError(RuntimeError):
    pass

def has_symbolic_sizes_strides(elem):
    return elem._has_symbolic_sizes_strides

def create_contiguous(shape):
    strides = [1]
    for dim in reversed(shape[:-1]):
        strides.append(dim * strides[-1])
    return list(reversed(strides))

def hint_int(a, fallback=None):
    """
    Retrieve the hint for an int (based on the underlying real values as observed
    at runtime).  If no hint is available (e.g., because data dependent shapes),
    if fallback is not None, use that instead (otherwise raise an error).
    """
    if isinstance(a, torch.SymInt):
        return a.node.require_hint(fallback)
    assert type(a) is int, a
    return a

def has_hint(a):
    if isinstance(a, SymTypes):
        return a.node.has_hint()
    return True

def is_concrete_int(a: Union[int, SymInt]):
    r""" Utility to check if underlying object
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

def canonicalize_bool_expr(expr: sympy.Expr):
    r""" Canonicalize a boolean expression by transforming it into a lt / le
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

    if not isinstance(expr, (sympy.Rel, sympy.And, sympy.Or, sympy.Not, sympy.Eq, sympy.Ne)):
        return expr

    if isinstance(expr, (sympy.And, sympy.Or, sympy.Not)):
        expr = sympy.logic.boolalg.to_cnf(expr)
    return _canonicalize_bool_expr_impl(expr)

def _canonicalize_bool_expr_impl(expr: sympy.Expr):
    if isinstance(expr, (sympy.And, sympy.Or)):
        return type(expr)(*map(canonicalize_bool_expr, expr.args))

    opposite = {sympy.Gt: sympy.Lt, sympy.Ge: sympy.Le}
    if isinstance(expr, tuple(opposite.keys())):
        lhs = expr.rhs - expr.lhs
        t = opposite[type(expr)]
    else:
        assert isinstance(expr, (sympy.Lt, sympy.Le, sympy.Eq, sympy.Ne))
        lhs = expr.lhs - expr.rhs
        t = type(expr)
    rhs = 0
    if isinstance(lhs, sympy.Add):
        cts = []
        variables = []
        for term in lhs.args:
            if term.is_number:
                cts.append(term)
            else:
                variables.append(term)
        lhs = sympy.Add(*variables)
        rhs = -sympy.Add(*cts)
    return t(lhs, rhs)

def is_concrete_bool(a: Union[bool, SymBool]):
    r""" Utility to check if underlying object
    in SymBool is concrete value. Also returns
    true if integer is passed in.
    Args:
        a (SymBool or bool): Object to test if it bool
    """
    assert isinstance(a, (SymBool, bool))

    if isinstance(a, bool):
        return True

    if isinstance(a.node.expr, (sympy.logic.boolalg.BooleanTrue, sympy.logic.boolalg.BooleanFalse)):
        return True

    return False

def is_singleton(s):
    # check for SingletonSymNode
    if not isinstance(s, torch.SymInt):
        return False
    if s.node.singleton_int() is not None:
        return True

    # check for symbolic variable wrapping a SingletonSymNode (fake-ifying causes this)
    return (
        s.node.is_symbolic()
        and s.node.hint is not None
        and isinstance(s.node.hint, torch.SymInt)
        and s.node.hint.node.singleton_int() is not None
    )

def _iterate_exprs(val: Union[SymInt, torch.Tensor]) -> Iterable[sympy.Basic]:
    if isinstance(val, SymTypes):
        # This allow applies to the jagged layout NestedTensor case as
        # singleton ints are not symbolic
        if is_symbolic(val):
            yield val.node.expr
    elif isinstance(val, sympy.Basic):
        yield val
    elif isinstance(val, (int, float, bool)):
        pass
    elif isinstance(val, torch.Tensor):
        yield from _iterate_exprs(val.size())
        yield from _iterate_exprs(val.stride())
        yield from _iterate_exprs(val.storage_offset())
    elif isinstance(val, (tuple, list)):
        for s in val:
            yield from _iterate_exprs(s)
    else:
        raise AssertionError(f"cannot extract sympy expressions from {val} {type(val)}")

def free_symbols(val: Union[SymInt, torch.Tensor]) -> Set[sympy.Symbol]:
    itr = _iterate_exprs(val)
    # we need at least 1 to call union, so we hand code the identity
    try:
        first_expr = next(itr)
    except StopIteration:
        return set()

    return first_expr.free_symbols.union(*(e.free_symbols for e in itr))

def has_free_symbols(val: Union[SymInt, torch.Tensor]) -> bool:
    """Faster version of bool(free_symbols(val))"""
    return not all(e.is_number for e in _iterate_exprs(val))

# Like free_symbols, but filtered to only report unbacked symbols
def free_unbacked_symbols(x):
    # NB: keep synced with is_unbacked_symint
    return {s for s in free_symbols(x) if s.name.startswith(("i", "f"))}

# WARNING: Don't use this on Dynamo produced graphs, they don't have meta
# setup!
def is_symbol_binding_fx_node(node) -> Optional[sympy.Symbol]:
    if (
        node.op == "placeholder" and
        "val" in node.meta and
        isinstance(node.meta["val"], torch.SymInt) and
        isinstance(node.meta["val"].node.expr, sympy.Symbol)
    ):
        return node.meta["val"].node.expr
    return None

def find_symbol_binding_fx_nodes(graph):
    return {
        node.meta["val"].node.expr: node
        for node in graph.nodes
        if is_symbol_binding_fx_node(node)
    }

def definitely_true(a):
    """
    Returns True only if we can tell that a is True, possibly introducing
    a guard in the process.  If a depends on some unbacked SymInt, we may
    return False even though there may exist a possible value of the SymInt
    that would cause the expression to return True.

    When is it appropriate to use definitely_true?  First, if you can use
    a higher level combinator like parallel_or/parallel_and, prefer using
    those instead, they are definitely safe (modulo short-circuiting).
    Second, it can be used if the program would behave equivalently if
    definitely_true always returned False (parallel_or/parallel_and are
    examples of this pattern, modulo short-circuiting).  Finally, it even
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

def definitely_false(a):
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

# TODO: could improve parallel_or/parallel_and by avoiding guards
# if there exists a quantity that can be handled un-guardedly.  However,
# for backed SymInts, avoiding guards doesn't really matter in practice,
# so I chose not to do it.

def parallel_or(*args):
    """
    Evaluate the logical OR of several arguments, avoiding guarding on
    unbacked SymInts if another argument is definitely True.
    """
    if any(definitely_true(a) for a in args):
        return True
    return any(args)

def parallel_and(*args):
    """
    Evaluate the logical FALSE of several arguments, avoiding guarding on
    unbacked SymInts if another argument is definitely False.
    """
    if any(definitely_false(a) for a in args):
        return False
    return all(args)

def sym_eq(x, y):
    """
    Like ==, but when run on list/tuple, it will recursively test equality
    and use sym_and to join the results together, without guarding.
    """
    if (isinstance(x, tuple) and isinstance(y, tuple)) or (isinstance(x, list) and isinstance(y, list)):
        if len(x) != len(y):
            return False
        return functools.reduce(operator.and_, map(sym_eq, x, y), True)
    elif isinstance(x, (int, torch.SymInt)) and isinstance(y, (int, torch.SymInt)):
        return x == y
    else:
        raise AssertionError(f"unexpected sym_eq between {type(x)} {type(y)}")

def guard_scalar(a):
    if isinstance(a, (SymBool, bool)):
        return guard_bool(a)
    elif isinstance(a, (SymInt, int)):
        return guard_int(a)
    elif isinstance(a, (SymFloat, float)):
        return guard_float(a)
    else:
        raise AssertionError(f"unrecognized scalar {a}")


@record_shapeenv_event()
def _constrain_symbol_range(shape_env, s: sympy.Symbol, compiler_min: int, compiler_max: int, runtime_min: int, runtime_max: int):
    log.debug("_constrain_symbol_range %s [%s, %s] [%s, %s]", s, compiler_min, compiler_max, runtime_min, runtime_max)
    if r := shape_env.var_to_range.get(s, None):
        shape_env.var_to_range[s] = ValueRanges(
            builtins.max(r.lower, compiler_min), builtins.min(r.upper, compiler_max)
        )
    else:
        shape_env.var_to_range[s] = ValueRanges(compiler_min, compiler_max)

    if r := shape_env.runtime_var_to_range.get(s, None):
        shape_env.runtime_var_to_range[s] = ValueRanges(
            builtins.max(r.lower, runtime_min), builtins.min(r.upper, runtime_max)
        )
    else:
        shape_env.runtime_var_to_range[s] = ValueRanges(runtime_min, runtime_max)


def _advise_is_size(a):
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
    assert a >= 0

    # NB: it's important not to constrain range for size for *hinted* SymInts,
    # because it is not only unsound, it will immediately trip our asserts
    # that hints have to be consistent with static analysis!  If you somehow
    # have an unbounded SymInt that later constrains to 1, this will be
    # inconsistent with the range
    if (
        isinstance(a, SymInt)
        and isinstance(a.node, SymNode)
        and not a.node.has_hint()
        and isinstance(a.node.expr, sympy.Symbol)
    ):
        _constrain_range_for_size(a)

@record_shapeenv_event()
def _constrain_range_for_size(a, min: Optional[int] = None, max: Optional[int] = None):
    """
    This function is NOT INTENDED to be used by itself.
    """

    if isinstance(a, (SymFloat, SymBool)):
        raise ValueError("Constraining SymFloat/SymBool is nyi")

    assert isinstance(a, SymInt), "can only constrain range for SymInt"
    assert isinstance(a.node.expr, sympy.Symbol), "constraining non-Symbols NYI"

    if min is None:
        min = 0
    if max is None:
        max = sympy.oo

    if max <= 2:
        raise ValueError(f"Maximum value to constrain_as_size must be greater than 2, but was {max}")

    if max < min:
        raise ValueError(
            "Maximum value to constrain_as_size can't be less than the specified min value, "
            "received min={min} and max={max}"
        )

    compiler_min = 2 if min < 2 else min

    _constrain_symbol_range(
        a.node.shape_env,
        a.node.expr,
        compiler_min=compiler_min,
        compiler_max=max,
        runtime_min=min,
        runtime_max=max
    )


# inclusive both ways
@record_shapeenv_event()
def constrain_range(a, *, min: Optional[int], max: Optional[int] = None):
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

    .. warning::
        If you use constrain_range in the context of tracing, we do NOT check
        that the constraint was actually valid at runtime!  In fact, we
        cannot (easily) do so, as we currently unsoundly assume that unbacked
        SymInt can never be zero/one, even if it may actually take on these
        values at runtime (we assume that a graph that is valid for N=2 will
        also be valid for N=1).
    """
    if min is None:
        min = -sympy.oo
    if max is None:
        max = sympy.oo

    if max < min:
        raise ValueError(
            "Maximum value to constrain_as_size can't be less than the specified min value, "
            "received min={min} and max={max}"
        )

    if isinstance(a, int):
        if not (min <= a <= max):
            raise ValueError(f"Invalid value {a} for range [{min}:{max}]")
        return

    if isinstance(a.node.expr, sympy.Integer):
        if not (min <= int(a.node.expr) <= max):
            raise ValueRangeError(f"Invalid value {int(a.node.expr)} for range [{min}:{max}]")
        return
    assert isinstance(a.node.expr, sympy.Symbol), "constraining non-Symbols NYI"

    # TODO: Shouldn't we install a guard if the symbol is backed?  Or is the
    # semantics that this is an "unchecked" assert (but it this actually
    # something useful?  Might be better to restrict only for unbacked
    # SymInt).
    _constrain_symbol_range(
        a.node.shape_env,
        a.node.expr,
        compiler_min=min,
        compiler_max=max,
        runtime_min=min,
        runtime_max=max
    )


@record_shapeenv_event()
def constrain_unify(a, b):
    """
    Given two SymInts, constrain them so that they must be equal.  NB:
    this will not work with SymInts that represent nontrivial expressions
    (yet!)
    """
    # TODO: Maybe dedupe this with _maybe_guard_eq?
    if not isinstance(a, SymInt):
        if not isinstance(b, SymInt):
            assert a == b
        else:
            assert isinstance(b.node.expr, sympy.Symbol), "constraining non-Symbols NYI"
            shape_env = b.node.shape_env
            shape_env.replacements[b.node.expr] = sympy.Integer(a)
    else:
        # TODO: Actually, we can support this as long as one of them is a symbol.
        # NB: We can't actually do "unification" as our operators are not
        # injective
        assert isinstance(a.node.expr, sympy.Symbol), "constraining non-Symbols NYI"
        shape_env = a.node.shape_env
        if not isinstance(b, SymInt):
            shape_env.replacements[a.node.expr] = sympy.Integer(b)
        else:
            assert a.node.shape_env is b.node.shape_env
            assert isinstance(b.node.expr, sympy.Symbol), "constraining non-Symbols NYI"
            new_var = shape_env._find(a.node.expr)
            shape_env.replacements[b.node.expr] = new_var

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
def expect_true(a, skip: int = 0):
    if isinstance(a, SymBool):
        # TODO: check perf implications of this
        frame = inspect.currentframe()
        for _ in range(skip + 1):  # always run this loop at least once
            frame = frame.f_back
        return a.node.expect_true(frame.f_code.co_filename, frame.f_lineno)
    assert type(a) is bool, a
    return a

def guard_bool(a):
    if isinstance(a, SymBool):
        return a.node.guard_bool("", 0)  # NB: uses Python backtrace
    assert type(a) is bool, a
    return a

def guard_int(a):
    if isinstance(a, SymInt):
        return a.node.guard_int("", 0)  # NB: uses Python backtrace
    assert type(a) is int, a
    return a

def guard_float(a):
    if isinstance(a, SymFloat):
        return a.node.guard_float("", 0)  # NB: uses Python backtrace
    assert isinstance(a, float), a
    return a

# Given a GraphModule, return all the FakeTensors for all the placeholders
def fx_placeholder_vals(gm):
    return [n.meta['val'] for n in gm.graph.nodes if n.op == "placeholder"]

def fx_placeholder_targets(gm):
    return [n.target for n in gm.graph.nodes if n.op == "placeholder"]

# Given a GraphModule and arguments to run it with, evaluate that the guards
# for its associated ShapeEnv are satisfied by the passed arguments.  This
# WILL check for duck sizing.
def eval_guards(gm, *args, ignore_static=True):
    return gm.shape_env.evaluate_guards_for_args(fx_placeholder_vals(gm), args, ignore_static=ignore_static)

def bind_symbols(gm, *args):
    return gm.shape_env.bind_symbols(fx_placeholder_vals(gm), args)

def _assert_bound_is_rational(expr: sympy.Expr, bound: ValueRanges):
    """
    We assert that the bounds are either Boolean, or not finite, or can be computed
    in exact prevision via rational arithmetic.
    The only exception to this is the rare case when the user calls `sqrt(s0)`
    sqrt is turned into sympy.Pow so we just match for that (it matches more things, but still)
    """
    assert bound.lower.is_rational or bound.lower.is_Boolean or not bound.lower.is_finite or expr.has(sympy.Pow), (bound, expr)
    assert bound.upper.is_rational or bound.upper.is_Boolean or not bound.upper.is_finite or expr.has(sympy.Pow), (bound, expr)

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
        - An individual dim is marked DYNAMIC if you mention it as dynamic_dim
          in the constraints kwarg.
    """
    # Treat the dimension symbolically
    DYNAMIC = 0
    # Treat the dimension symbolically, but if its hint matches another
    # dynamic dimension, unify the two symbols ("duck sizing")
    DUCK = 1
    # Treat the dimension statically based on its hint
    STATIC = 2


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

    def render(self, source: Source):
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
    def render(self, source: Source):
        return f"RelaxedUnspecConstraint({source.name()})"

# NB: None here indicates the client constraint is whatever is implicitly
# inferred by guards from tracing, and that a backend can add whatever guards
# it wants (including fully specializing the value).
DimConstraint = Union[StrictMinMaxConstraint, RelaxedUnspecConstraint, None]

@dataclass(frozen=True)
class EqualityConstraint(Constraint):
    """
    Given pairs of sources corresponding to pairs of dynamic dimensions that
    are specified equal, represent them in a union-find data structure so that
    we can efficiently check whether two such sources are transitively equal.
    """
    source_pairs: List[Tuple[Source, Source]]

    def __post_init__(self):
        object.__setattr__(self, "_parents", {})
        for source1, source2 in self.source_pairs:
            self._union(self._find(source1), self._find(source2))

    def _find(self, source):
        if source in self._parents:
            return self._find(self._parents[source])
        else:
            return source

    def _union(self, root1, root2):
        if root1 != root2:
            self._parents[root1] = root2

    def render(self):
        buf = ", ".join(
            f"{source1.name()} == {source2.name()}"
            for (source1, source2) in self.source_pairs
        )
        return "{" + buf + "}"

    def is_equal(self, source1, source2):
        return self._find(source1) == self._find(source2)


def _assert_symbol_context(symbolic_context):
    assert isinstance(symbolic_context, SymbolicContext), "Invalid symbolic_context object"
    assert type(symbolic_context) is not SymbolicContext, "Illegal usage of symbolic_context ABC"


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
    pass


@dataclass(frozen=True)
class StatelessSymbolicContext(SymbolicContext):
    """
    Create symbols in ``create_symbolic_sizes_strides_storage_offset`` via
    a symbolic_context determination as given by ``DimDynamic`` and ``DimConstraint``.
    This will cause fresh symbols to be allocated
    """
    dynamic_sizes: DimList[DimDynamic]
    constraint_sizes: DimList[DimConstraint] = None
    # TODO: add storage offset and stride symbolic_context

    def __post_init__(self):
        if self.constraint_sizes is None:
            object.__setattr__(self, 'constraint_sizes', [None] * len(self.dynamic_sizes))


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
    tensor_source: Source = None
    # Why is this keyd on int first?
    # That integer is actually the id of the shape_env. This cache short-circuits symbol
    # creation, and we must store it per shape env. Now, while tracing invariants are a single
    # shape env per tracing context, and every new frame gets a new shape_env. So where would we have
    # multiple shape envs? The answer lies in recording. When we are replaying, replay_shape_env_events
    # is invoked, and creates a new shape_env. Replaying events against this new shape_env will
    # cause it to fail with unknown symbols, as the symbols cached here will skip creation, and never
    # get recorded in var_to_val, etc.
    # TODO(voz): consider a weakref to the shape_env here
    shape_env_to_source_to_symbol_cache : Dict[int, Dict["TensorPropertySource", "sympy.Expr"]] = None

    def __post_init__(self):
        # The None default is annoying, but required because of dataclass limitations
        assert self.tensor_source is not None
        if not self.shape_env_to_source_to_symbol_cache:
            object.__setattr__(self, 'shape_env_to_source_to_symbol_cache', {})


@dataclass(frozen=True)
class SubclassSymbolicContext(StatefulSymbolicContext):
    """
    The correct symbolic context for a given inner tensor of a traceable tensor subclass
    may differ from that of the outer symbolic context. This structure allows for this
    flexibility, with inner symbolic contexts mapped via attr -> symbolic context.
    """
    inner_contexts: Dict[str, SymbolicContext] = None

    def __post_init__(self):
        if self.inner_contexts is None:
            self.inner_contexts = {}


def is_symbolic(val: Union[int, SymInt, float, SymFloat, bool, SymBool]) -> bool:
    if isinstance(val, (int, float, bool)):
        return False
    return val.node.is_symbolic()

IndicatorTypes = (IsNonOverlappingAndDenseIndicator,)

@lru_cache(256)
def safe_expand(r):
    if hasattr(r, 'expand'):
        try:
            return sympy.expand(r)
        except RecursionError:
            log.warning("RecursionError in sympy.expand(%s)", r)
            return r
    else:
        return r

def error():
    raise AssertionError("shouldn't be hit")


# TODO: Deduplicate this with torch/_prims_common/__init__.py
def eval_is_non_overlapping_and_dense(sizes, strides):
    return int(guard_bool(_eval_is_non_overlapping_and_dense(sizes, strides)))

def _eval_is_non_overlapping_and_dense(sizes, strides):
    dim = len(sizes)

    # Short-circuits for tensors of rank one, which are
    # non-overlapping and "dense" if their stride is one
    # or it is a 0/1 element tensor
    if dim == 1:
        return strides[0] == 1 or sizes[0] < 2

    # Checks that there exists a permutation of the strides s.t. the tensor would be contiguous
    # Sorts (length, stride) pairs by stride
    lengths_and_strides = sorted(
        zip(sizes, strides), key=operator.itemgetter(1)
    )

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


def cast_symbool_to_symint_guardless(symbool: torch.SymBool) -> torch.SymInt:
    int_sym = sympy.Piecewise((1, symbool.node.expr), (0, True))
    return symbool.node.shape_env.create_symintnode(int_sym, hint=int(symbool.node.require_hint()))

SYMPY_INTERP = {
    'Abs': operator.abs,
    'Eq': operator.eq,
    'Ne': operator.ne,
    'Gt': operator.gt,
    'Lt': operator.lt,
    'Le': operator.le,
    'Ge': operator.ge,
    'Min': min,
    'Max': max,
    'Mod': operator.mod,
    'FloorDiv': operator.floordiv,
    'TrueDiv': operator.truediv,
    'IsNonOverlappingAndDenseIndicator': eval_is_non_overlapping_and_dense,
    'floor': math.floor,
    'ceiling': math.ceil,
    'cast_symbool_to_symint_guardless': cast_symbool_to_symint_guardless,
    'Round': builtins.round,
    'RoundDecimal': builtins.round,
}


def _lru_cache(fn, maxsize=None):
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

    if config.validate_shape_env_verison_key:
        prior_key = None

        @functools.wraps(fn)
        def wrapper(self, *args, **kwargs):
            nonlocal prior_version, prior_key
            if prior_key is None:
                prior_key = self._get_key()

            if prior_version != self._version_counter:
                fn_cache.cache_clear()
                prior_version = self._version_counter
                prior_key = self._get_key()
            else:
                assert prior_key == self._get_key(), \
                    "ShapeEnv cache key changed without version being updated!"

            return fn_cache(self, *args, **kwargs)

    else:

        @functools.wraps(fn)
        def wrapper(self, *args, **kwargs):
            nonlocal prior_version
            if prior_version != self._version_counter:
                fn_cache.cache_clear()
                prior_version = self._version_counter

            return fn_cache(self, *args, **kwargs)

    wrapper.cache_clear = fn_cache.cache_clear
    wrapper.cache_info = fn_cache.cache_info  # type: ignore[attr-defined]
    return wrapper


# This is pretty similar to ShapeGuard but it also comes with a message,
# and is exclusively used for things that MUST be true (unlike guards,
# which can evaluate False, in which case you just choose not to use
# a particular specialization)
@dataclass(frozen=True)
class RuntimeAssert:
    expr: sympy.Expr
    msg: str = field(repr=False)
    stack: str = field(repr=False)


class ShapeGuardPrinter(StrPrinter):
    def __init__(
        self,
        symbol_to_source,
        source_ref,
        var_to_sources,
    ):
        super().__init__()
        self.symbol_to_source = symbol_to_source
        self.source_ref = source_ref
        self.var_to_sources = var_to_sources

    def _print_Not(self, expr):
        return 'not %s' % (self.parenthesize(expr.args[0], PRECEDENCE["Not"]))

    def _print_And(self, expr):
        return self.stringify(expr.args, " and ", PRECEDENCE["And"])

    def _print_Or(self, expr):
        return self.stringify(expr.args, " or ", PRECEDENCE["Or"])

    def _print_Symbol(self, expr) -> str:
        assert isinstance(expr, sympy.Symbol), str(type(expr))

        def repr_symbol_to_source():
            return repr({
                symbol: [s.name() for s in sources]
                for symbol, sources in self.symbol_to_source.items()
            })

        assert self.symbol_to_source.get(expr), (
            f"{expr} (could be from {[s.name() for s in self.var_to_sources[expr]]}) "
            f"not in {repr_symbol_to_source()}.  If this assert is failing, it could be "
            "due to the issue described in https://github.com/pytorch/pytorch/pull/90665"
        )
        return self.source_ref(self.symbol_to_source[expr][0])


class LoggingShapeGuardPrinter(ShapeGuardPrinter):
    def __init__(self, var_to_sources):
        super().__init__(var_to_sources, lambda n: n.name(), var_to_sources)


class DynamicDimConstraintPrinter(StrPrinter):
    """
    Printer for dynamic dim constraints.
    - Instead of t.size()[d] it prints dynamic_dim(t, d)
    - Instead of Eq(_, _), Mod(_, _), etc. it prints _ == _, _ % _, etc.

    We use this to suggest code for specifying dynamic dim constraints.
    """
    def __init__(self, symbol_to_source, source_name_to_debug_name):
        super().__init__()
        self.symbol_to_source = symbol_to_source
        self.source_name_to_debug_name = source_name_to_debug_name

    def print_source(self, source) -> str:
        if self.source_name_to_debug_name:
            return source.name()
        return f"dynamic_dim({source.base.name()}, {source.idx})"

    def _print_Symbol(self, expr) -> str:
        assert isinstance(expr, sympy.Symbol), str(type(expr))
        assert self.symbol_to_source.get(expr), (
            f"Unknown symbol {expr} created by constraints solver"
        )
        return self.print_source(self.symbol_to_source[expr][0])

    def _print_Relational(self, expr):
        return '{} {} {}'.format(
            self.parenthesize(expr.lhs, precedence(expr)),
            expr.rel_op,
            self.parenthesize(expr.rhs, precedence(expr))
        )


class DimConstraints:
    """
    Custom solver for a system of constraints on symbolic dimensions.
    Solutions are "static" values or simplified "dynamic" constraints.
    """

    def __init__(self, symbol_to_source, var_to_val, marked_dynamic, source_name_to_debug_name):
        # We try to solve systems of inequalities with 1 free variable.
        self._univariate_inequalities: Dict[sympy.Symbol, Set[sympy.Expr]] = defaultdict(set)
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
        self._var_to_val: Dict[sympy.Symbol, sympy.Integer] = var_to_val
        self._congruences: Set[sympy.Expr] = defaultdict(set)

        # We do not try to (directly) solve inequalities with > 1 free variables.
        # NOTE: free variables in these inequalities cannot also be in _substitutions.
        self._multivariate_inequalities: Set[sympy.Expr] = set()

        # We park external equalities between free variables here.
        self._symbolic_equivalences: List[Tuple[Source, sympy.Expr]] = []

        # Solutions come in two forms:
        # - (static) specializations
        # - (dynamic) inequalities / congruences
        self._static_results: Set[str] = set()
        self._dynamic_results: Set[str] = set()

        # printer for solutions
        self._dcp = DynamicDimConstraintPrinter(symbol_to_source, source_name_to_debug_name)

        # inconsistencies found on substituting with concrete values / static solutions
        self._inconsistencies: List[str] = []

        # symbols that are marked dynamic
        self._marked_dynamic = marked_dynamic

    def rewrite_with_congruences(self, s, expr):
        """
        Eliminate expressions of the form b // d and b % d while adding congruences of the form b % d == k.
        This leaves rational operators (in particular of the form b / d) that our inequality solver can handle.
        We solve the added congruences separately (using our congruence solver, see below).
        """
        def mod_handler(*args):
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
            base, divisor = self.rewrite_with_congruences(s, base), self.rewrite_with_congruences(s, divisor)
            mod_reduced = base.subs(self._var_to_val) % divisor.subs(self._var_to_val)
            congruence = (base - mod_reduced) % divisor
            if congruence != 0:
                self._congruences[s].add(congruence)
            return mod_reduced

        def floor_div_handler(*args):
            # Suppose that we have an expression of the form b // d with free variable s.
            # Using the value of s, we can evaluate b % d to a value k.
            # Then we can rewrite b // d to (b - k) / d, while adding the guard b % d == k.

            # NOTE(avik): This is exactly equivalent to rewriting b // d as (b - (b % d)) / d
            # and eliminating b % d as above.
            base, divisor = args
            base, divisor = self.rewrite_with_congruences(s, base), self.rewrite_with_congruences(s, divisor)
            mod_reduced = base.subs(self._var_to_val) % divisor.subs(self._var_to_val)
            congruence = (base - mod_reduced) % divisor
            if congruence != 0:
                self._congruences[s].add(congruence)
            return (base - mod_reduced) / divisor

        if expr.has(Mod):
            expr = expr.replace(Mod, mod_handler)
        if expr.has(FloorDiv):
            expr = expr.replace(FloorDiv, floor_div_handler)
        return expr

    def add(self, expr) -> bool:
        # Add an expression to the set of constraints.
        # Return whether the expression is a trivial constraint (i.e., an obvious tautology).
        if expr == sympy.true:
            return True
        orig_expr = expr
        orig_reduced = orig_expr.subs(self._var_to_val)
        # TODO(avik): https://github.com/pytorch/pytorch/issues/101093
        # It is possible that `expr` will fail the consistency check because of
        # precision errors. Specifically, on substituting its free symbols with
        # their concrete values, we might end up comparing floats. Until we have
        # a fix for this issue, we delay raising such failures. See solve().
        if orig_reduced == sympy.false:
            self._inconsistencies.append(f"{orig_expr} is inconsistent!")
        free_symbols = expr.free_symbols
        assert free_symbols, f"Did not expect constraint with no free variables: {expr}"
        if len(free_symbols) > 1:
            # multivariate: record and move on
            self._multivariate_inequalities.add(expr)
        else:
            # univariate: can solve these immediately
            s = next(iter(free_symbols))
            # eliminate // and % (see documentation of `rewrite_with_congruences` above)
            expr = self.rewrite_with_congruences(s, expr)
            if expr == sympy.true:
                return True
            reduced = expr.subs(self._var_to_val)
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

    def add_equality(self, source, expr):
        if expr.is_number:
            # specialization, right here
            self._static_results.add(f"{source.name()} == {expr}")
        else:
            # these will resolve to either specializations or dynamic equality constraints
            self._symbolic_equivalences.append((source, expr))

    def reduce_congruences(self):
        reduced_congruences = {}
        for s, congruences in self._congruences.items():
            remainder_modulus_pairs = []
            congruences_to_check = set()
            for congruence in congruences:
                base, divisor = congruence.args
                # We are given a congruence of the form base % divisor == 0 with a free variable s. So:
                # - we transform this into an equation of the form base = divisor * tmp;
                # - we solve this equation for s to get a linear solution with free variable tmp.
                tmp = sympy.Symbol("tmp", integer=True)
                symbol, solution = sympy.solve_linear(base - divisor * tmp, symbols=[s])
                # See https://docs.sympy.org/latest/modules/solvers/solvers.html#sympy.solvers.solvers.solve_linear
                # for how to interpret the results.
                if s == symbol:
                    # This means the solution is of the form s = modulus*tmp + remainder.
                    modulus, remainder = sympy.polys.polytools.div(solution, tmp)
                    if isinstance(modulus, sympy.Integer) and isinstance(remainder, sympy.Integer):
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
                remainder, modulus = sympy.ntheory.modular.solve_congruence(*remainder_modulus_pairs)
                reduced_congruences[s] = {(s - remainder) % modulus}
                substitution = {s: modulus * sympy.Symbol("tmp", integer=True) + remainder}
                reduced_congruences[s].update(
                    congruence for congruence in congruences_to_check
                    if not sympy.checksol(congruence, substitution)
                )
            else:
                reduced_congruences[s] = congruences_to_check

        return reduced_congruences

    def raise_inconsistencies(self):
        if self._inconsistencies:
            msg = "\n".join(self._inconsistencies)
            self._inconsistencies.clear()
            raise ValueError(f"The following inconsistencies were found:\n{msg}")

    def _force_specialization(self, s):
        val = self._var_to_val[s]
        self._static_results.add(f"{self._dcp.symbol_to_source[s][0].name()} == {val}")
        self._substitutions[s] = val

    def specialize_divisor_symbols(self):
        for expr in self._multivariate_inequalities:
            for atom in expr.atoms(FloorDiv, Mod):
                _, divisor = atom.args
                for s in divisor.free_symbols:
                    self._force_specialization(s)

        multivariate_inequalities = self._multivariate_inequalities
        self._multivariate_inequalities = set()
        for expr in multivariate_inequalities:
            self.add(expr.subs(self._substitutions))
        self.raise_inconsistencies()
        self._univariate_inequalities = {
            s: exprs
            for s, exprs in self._univariate_inequalities.items()
            if s not in self._substitutions
        }
        self._congruences = {
            s: congruences
            for s, congruences in self._congruences.items()
            if s not in self._substitutions
        }

    def solve(self, disable_congruences=True, disable_equivalences=True):
        self.raise_inconsistencies()
        # as long as there are symbols with equalities, solve for them
        # NOTE(avik): this is guaranteed to terminate (#iterations <= #symbols)
        while self._symbols_with_equalities:
            s = self._symbols_with_equalities.pop()
            exprs = self._univariate_inequalities.pop(s)
            solution = sympy.solvers.inequalities.reduce_inequalities(exprs, s)
            if isinstance(solution, sympy.And):
                solution = next((arg for arg in solution.args if isinstance(arg, sympy.Eq)), solution)
            assert isinstance(solution, sympy.Eq), f"Expected an equality constraint for {s}, got {solution}"
            symbol, val = solution.args
            assert symbol == s, f"Expected a constraint on {s} instead of on {symbol}"
            # because this is univariate, the solution is a specialization
            self._static_results.add(f"{self._dcp.symbol_to_source[s][0].name()} == {val}")
            # add this as a substitution to simplify other constraints
            self._substitutions[s] = val

            # simplify multivariate inequalities: some of them will now become univariate!
            multivariate_inequalities = self._multivariate_inequalities
            self._multivariate_inequalities = set()
            for expr in multivariate_inequalities:
                self.add(expr.subs(s, self._substitutions[s]))
            self.raise_inconsistencies()

        self.specialize_divisor_symbols()

        # solve linear congruences
        # NOTE(avik): We do not need to solve them for symbols that have already been specialized.
        reduced_congruences = self.reduce_congruences()
        for s, congruences in reduced_congruences.items():
            for congruence in congruences:
                # any congruence that cannot be checked becomes a dynamic constraint as well
                if s not in self._substitutions or not sympy.checksol(congruence, {s: self._substitutions[s]}):
                    if disable_congruences:
                        self._force_specialization(s)
                        self._univariate_inequalities.pop(s, None)
                    else:
                        self._dynamic_results.add(self._dcp.doprint(sympy.Eq(congruence, 0)))

        # remaining symbols have only pure inequalities (no equalities)
        for s, exprs in self._univariate_inequalities.items():
            try:
                solution = sympy.solvers.inequalities.reduce_inequalities(exprs, s)
                # because this is univariate, the solution is a dynamic (range) constraint
                if isinstance(solution, sympy.And):
                    for arg in solution.args:
                        self._dynamic_results.add(self._dcp.doprint(arg))
                else:
                    self._dynamic_results.add(self._dcp.doprint(solution))
            except (NotImplementedError, AssertionError) as e:
                log.warning("Failed to reduce inequalities: %s", e)
                for expr in exprs:
                    self._dynamic_results.add(self._dcp.doprint(expr))

        # simplify symbolic equivalences: some of them will now become specializations!
        symbolic_equivalences = self._symbolic_equivalences
        self._symbolic_equivalences = []
        for source, expr in symbolic_equivalences:
            if disable_equivalences and not isinstance(expr, sympy.Symbol):
                for s in expr.free_symbols:
                    self._force_specialization(s)
                    sexpr = self._dcp._print_Symbol(s)
                    self._dynamic_results = {r for r in self._dynamic_results if sexpr not in r}
            self.add_equality(source, expr.subs(self._substitutions))

        # remaining symbolic equivalences become dynamic equality constraints
        for source, expr in self._symbolic_equivalences:
            self._dynamic_results.add(f"{self._dcp.print_source(source)} == {self._dcp.doprint(expr)}")

    def forced_specializations(self):
        def debug_name(src):
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

    def remove_redundant_dynamic_results(self):
        candidates_for_removal = []
        dynamic_results = set()
        for dc in self._dynamic_results:
            # Instead of 2 <= dynamic_dim(...) simply suggest dynamic_dim(...).
            # There is no change in behavior since 2 is the default lower bound.
            dc_ = re.sub(r"2 <= dynamic_dim(.+)", r"dynamic_dim\1", dc)
            if dc != dc_:
                candidates_for_removal.append(dc_)
            else:
                dynamic_results.add(dc_)
        for dc in candidates_for_removal:
            # remove dynamic_dim(t, 0) as a constraint when dynamic_dim(t, 0) also
            # appears as part of another constraint
            found = False
            for other_dc in dynamic_results:
                if dc in other_dc:
                    found = True
            if not found:
                dynamic_results.add(dc)
        self._dynamic_results = dynamic_results

    def prettify_results(
        self,
        original_signature: inspect.Signature,
        constraint_violation_error=None,
        forced_specializations=None,
    ):
        if self._dcp.source_name_to_debug_name:
            def transform(s):
                for k, v in self._dcp.source_name_to_debug_name.items():
                    s = s.replace(k, v)
                return s

            results = defaultdict(dict)

            def flip(op):
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

            def relation_with_digit(expr, op, digit):
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

            for s in self._static_results.union(self._dynamic_results):
                t = transform(s)
                if t == s:
                    continue
                left, op, right = t.split(" ")
                if op == "==" and left == right:
                    continue
                if right.isdigit():
                    relation_with_digit(left, op, int(right))
                elif left.isdigit():
                    relation_with_digit(right, flip(op), int(left))
                else:
                    assert op == "=="
                    results[left]["eq"] = right

            buf = ""
            debug_names = set()
            if forced_specializations:
                debug_names.update(k.split(" = ")[0] for k in forced_specializations.keys())
                buf += (
                    f"Specializations unexpectedly required ({', '.join(debug_names)})! "
                    "For more information, run with TORCH_LOGS=\"+dynamic\".\n"
                )
                for s, val in forced_specializations.items():
                    buf += f"  - {s} must be specialized to {val} because the guards generated for it are too complex.\n"

            dims = []
            others = []
            match = None
            if constraint_violation_error:
                match = re.search(r"Constraints violated \((.*)\)", constraint_violation_error.args[0])
            if match is not None:
                debug_names.update(match.expand(r'\1').split(', '))

            for k, c in results.items():
                if k not in debug_names:
                    continue
                if "eq" in c:
                    other = c["eq"]
                    if isinstance(other, int):
                        others.append(f"{k} = None  # {other}")
                    else:
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

            buf += "\nSuggested fixes:\n  "
            buf += "\n  ".join(dims + others)

            return buf

        # Note: Model inputs are wrapped as LocalSource in dynamo.
        # LocalSource.name() wraps the name with L[""]. We use regular
        # expression to do the replacement to avoid traversing up
        # the source hierarchy manually.
        def extract_and_rewrite_local(dc):
            match = re.search(r"L\['(.+?)'\]", dc)
            if match is None:
                return
            arg = match.expand(r'\1')
            dc = re.sub(r"L\['(.+?)'\]", r'\1', dc)
            return arg, dc

        def group(results, args_index):
            groups = defaultdict(list)
            for dc in results:
                local = extract_and_rewrite_local(dc)
                if local is None:
                    # This can happen, e.g., with `assume_constant_result`.
                    # In that case, we drop the constraint.
                    # TODO(avik) Maybe we should generate an assertion here?
                    continue
                arg, dc = local
                if arg in args_index:
                    groups[args_index[arg]].append(dc)
                else:
                    # This can happen, e.g., with decorators that change the signature.
                    # In that case, we drop the constraint. Seems hard to do better. :/
                    # TODO(avik) Maybe warn that `arg` in not in `signature`?
                    continue
            sorted_groups = []
            for idx, dcs in sorted(groups.items()):
                _, arg = idx
                sorted_groups.append((arg, sorted(dcs)))
            return sorted_groups

        signature = original_signature.replace(return_annotation=inspect.Signature.empty)
        args_index = {}
        for i, arg in enumerate(signature.parameters.keys()):
            args_index[arg] = (i, arg)

        def print_results(grouped, indent, result_fn):
            nonlocal buf

            space = False
            for arg, results in grouped:
                if space:
                    buf += "\n"
                else:
                    space = True
                buf += f"\n{indent}# {arg}:"
                for result in results:
                    buf += f"\n{indent}{result_fn(result)}"

        buf = ""
        if forced_specializations:
            buf += (
                "Some dynamic dimensions need to be specialized because "
                "the constraints inferred for them are too complex to specify.\n"
            )
            for s, val in forced_specializations.items():
                buf += f"  - {s}, which was marked dynamic, must be specialized to {val}.\n"
        indent = 4 * " "
        if self._static_results:
            grouped_static_results = group(self._static_results, args_index)
            buf += "\nThe following dimensions have been specialized and CANNOT be dynamic."
            buf += f"\n```\ndef specializations{str(signature)}:"
            print_results(
                grouped_static_results,
                indent,
                lambda result: f"assert {result}",
            )
            buf += "\n```\n"
        if self._dynamic_results:
            grouped_dynamic_results = group(self._dynamic_results, args_index)
            buf += "\nThe following dimensions CAN be dynamic."
            buf += "\nPlease use the following code to specify the constraints they must satisfy:"
            buf += f"\n```\ndef specify_constraints{str(signature)}:"
            buf += f"\n{indent}return ["
            print_results(
                grouped_dynamic_results,
                indent * 2,
                lambda result: f"{result},",
            )
            buf += f"\n{indent}]\n```\n"
        return buf



TLS = threading.local()


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
        self, *,
        should_record_events: Optional[bool] = None,
        tracked_fakes: Optional[List[Any]] = None,
        **kwargs
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
        self.is_recording = not self.should_record_events
        # Keep track of the list of tracked fakes.
        self.tracked_fakes = tracked_fakes
        # List of events for reconstructing ShapeEnv at arbitrary points in time.
        self.events: List[ShapeEnvEvent] = (
            [ShapeEnvEvent(ShapeEnv, kwargs=kwargs)] if self.should_record_events else []
        )

    def _init(
        self, *,
        allow_scalar_outputs=True,
        allow_dynamic_output_shape_ops=True,
        # NB: These are legacy configuration that help us make good choices
        # when the constraint/dynamic dims are not explicitly passed to us.
        # Ideally we will fix all call sites to be explicit and not have
        # implicit choices, but this apparently was pretty involved.
        assume_static_by_default=False,
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
        specialize_zero_one=True,
        # When True, assume input sizes which have the same size are
        # symbolically equal.
        duck_shape=True,
        # For debugging
        co_fields=None,
    ):
        # Not directly used by ShapeEnv; indirectly used by FakeTensor
        self.allow_scalar_outputs = allow_scalar_outputs
        self.allow_dynamic_output_shape_ops = allow_dynamic_output_shape_ops
        self.guards: List[ShapeGuard] = []
        # Maps symbolic ints to their original concrete values
        # Currently populated from tensors
        self.var_to_val: Dict[sympy.Symbol, sympy.Integer] = {}
        # Maps symbolic ints to their min/max range.  These ranges
        # are conservative: the int MUST fall in the range, but the
        # range may contain ints which may not actually appear in
        # practice
        self.var_to_range: Dict[sympy.Symbol, ValueRanges] = {}
        self.source_name_to_debug_name: Dict[str, str] = {}
        # Maps symbolic ints to their min/max range for runtime checks.
        # This is because we assume a graph generated with N=2 is general enough
        # for N < 2. Therefore, it will be too strict to assert N=2 at runtime.
        self.runtime_var_to_range: Dict[sympy.Symbol, ValueRanges] = {}
        self.var_to_sources: Dict[sympy.Symbol, List[Source]] = {}
        self.var_to_stack: Dict[sympy.Symbol, CapturedTraceback] = {}
        # Maps symbolic ints to the guards that refine their lower/upper
        # bound. If one of them is None, it means that there are no guards
        # that refine that respective bound.
        self.var_to_guards: Dict[sympy.Symbol, Tuple[Optional[ShapeGuard], Optional[ShapeGuard]]] = {}
        # Maps from sympy ints to expressions representing them
        # Populated from equality guards (i.e. a.shape[0] == b.shape[0])
        self.replacements: Dict[sympy.Symbol, sympy.Expr] = {}  #
        # Set holds a % b expressions that evaluate to 0.
        self.divisible: Set[sympy.Expr] = set()
        # Duck-shaping says that if two input tensors have the same size,
        # they get assigned the same symbolic variable
        self.val_to_var: Dict[int, sympy.Expr] = {}
        if specialize_zero_one:
            self.val_to_var = {0: sympy.Integer(0), 1: sympy.Integer(1)}
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
        self.deferred_runtime_asserts: Dict[sympy.Symbol, List[RuntimeAssert]] = {}
        # This exists so we can efficiently invalidate the cache (it's used as
        # part of the cache key); otherwise we'd have to iterate through
        # deferred_runtime_asserts to compute its length
        self.num_deferred_runtime_asserts = 0
        self.assume_static_by_default = assume_static_by_default
        self.specialize_zero_one = specialize_zero_one
        self.duck_shape = duck_shape
        self.log = log
        self.log.info("create_env")
        self.frozen = False
        self.dim_constraints: Optional[DimConstraints] = None
        self.counter = collections.Counter()
        # A selection of important fields on co_field; solely used for
        # signpost_event
        self.co_fields = co_fields if co_fields else {}

        # Version counter used to invalidate cached values
        self._prev_cache_key = self._get_key()
        self._version_counter = 0

        # Cache for FX nodes.
        # Maps an already built node a tuple of:
        #   1. node's target
        #   2. list of arguments
        # This drastically reduces the size of the FX graph, avoiding
        # duplicated nodes.
        self.fx_node_cache: Dict[Tuple[Callable, Tuple[Any, ...]], torch.fx.Node] = {}
        self.source_to_symbol: Dict[str, sympy.Symbol] = {}

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

    def check_equal(self, other: "ShapeEnv") -> None:
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
            elif key == "var_to_guards":
                # Transform the tuple of optional ShapeGuards of each entry into
                # a tuple of optional expressions.
                return {
                    s: (
                        lb.expr if lb is not None else None,
                        ub.expr if ub is not None else None,
                    )
                    for s, (lb, ub) in value.items()
                }
            elif key == "deferred_runtime_asserts":
                # Transform the list of RuntimeAsserts into a list of expressions.
                return {s: [ra.expr for ra in ras] for s, ras in value.items()}
            elif key == "name_to_node":
                # Compare just the set of keys is the same.
                return set(value.keys())
            return value

        shape_env_check_state_equal(self, other, non_state_variable_names, map_value)

    def snapshot_tracked_fakes(self) -> Optional[List[Any]]:
        if self.tracked_fakes is None:
            return None

        from torch._dynamo.variables.builder import TrackedFake

        def maybe_transform_fake(fake: TrackedFake):
            inner_fake = fake.fake \
                if isinstance(fake.fake, torch.SymInt) \
                else FakeTensorMeta.from_fake(fake.fake)
            # Even though TrackedFake accepts either a Union[SymInt, FakeTensor], here we give it a
            # FakeTensorMeta for two reasons:
            #   1. this is all the information we need when recording ShapeEnvEvents.
            #   2. it works even if each TrackedFake changes its metadata.
            return TrackedFake(inner_fake, fake.source, fake.symbolic_context)  # type: ignore[arg-type]

        return [maybe_transform_fake(fake) for fake in self.tracked_fakes]

    def inc_tracked_fakes_length(self) -> None:
        self.tracked_fakes_length += 1

    def set_tracked_fakes_length(self, i: int) -> None:
        self.tracked_fakes_length = i

    def last_event_index(self) -> int:
        return len(self.events) - 1

    @contextmanager
    def recording(self):
        self.is_recording = True
        try:
            yield
        finally:
            self.is_recording = False

    @record_shapeenv_event()
    def freeze(self):
        self.frozen = True

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

    def _add_target_expr(self, expr) -> None:
        if self._translation_validation_enabled:
            self.validator.add_target_expr(expr)

    def _add_assertion(self, expr) -> None:
        if self._translation_validation_enabled:
            self.validator.add_assertion(expr)

    def _check_translation_validate(self) -> None:
        if self._translation_validation_enabled:
            self.validator.validate()

    @record_shapeenv_event()
    def create_fx_call_function(
            self,
            op: Callable,
            args: Tuple,
    ) -> Tuple[Optional[torch.fx.Node], bool]:
        # Cache this tuple in order to avoid duplicated nodes.
        node_key = (op, args)
        # Flags whether the returned node was cached or not.
        fresh = False

        if self._translation_validation_enabled and node_key not in self.fx_node_cache:
            from torch.fx.experimental.validator import z3op

            # Presence of None in the arguments implies that we should ignore this operation.
            if any(a is None for a in args):
                # We check if we are not mixing SymNode that should not be ignored
                # (fx_node is not None) with those that should (fx_node is None).
                assert all(not isinstance(a, torch.fx.Node) for a in args)
                return None, fresh

            fresh = True
            lifted_op = z3op(op, self.validator)

            # If translation validation is enabled, all arguments must have its
            # own FX node.
            assert all(a is not None for a in args), f"missing arg in FX graph ({op.__name__}): {args}"
            node = self.fx_node_cache[node_key] = self.graph.call_function(lifted_op, args)
            self.name_to_node[node.name] = node

        return self.fx_node_cache.get(node_key, None), fresh

    def create_fx_placeholder_and_z3var(
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
            mangled_name = re.sub(r'[^a-zA-Z0-9]', '_', re.sub(r'[()]', '', symbol.name))
            node = self.fx_node_cache[node_key] = self.graph.placeholder(mangled_name)
            self.name_to_node[node.name] = node
            # Attach the 'symbol' to the placeholder so that we can retrieve
            # the Z3 variable later.
            node.meta["symbol"] = symbol

        return self.fx_node_cache[node_key]

    def remove_fx_node(self, node: Optional[torch.fx.Node]) -> None:
        if self._translation_validation_enabled and node is not None:
            self.name_to_node.pop(node.name)
            self.graph.erase_node(node)

    def add_fx_node_metadata(self, node: torch.fx.Node) -> None:
        from torch._dynamo.utils import get_current_node

        if self.should_record_events:
            node.meta[SHAPEENV_EVENT_KEY] = self.last_event_index()
            node.meta[CURRENT_NODE_KEY] = get_current_node()

    def _suppress_guards_tls(self):
        return getattr(TLS, "suppress_guards", False)

    @record_shapeenv_event()
    def suppress_guards_enter(self):
        TLS.suppress_guards = True

    @record_shapeenv_event()
    def suppress_guards_exit(self):
        TLS.suppress_guards = False

    @contextmanager
    def suppress_guards(self):
        self.suppress_guards_enter()
        try:
            yield
        finally:
            self.suppress_guards_exit()

    def _get_key(self):
        """
        Defines the current "state" of the guards we've accumulated in this ShapeEnv.
        Determines when we need to invalidate our cache
        """
        return (len(self.replacements), len(self.divisible), self.num_deferred_runtime_asserts)

    def _update_version_counter(self):
        # The shape environment is queried orders of magnitude more often than
        # it is changed, so we summarise the cache key into a linearly
        # increasing version counter which is cheaper to check in _lru_cache

        # Only update version counter if the state actually changed
        cur_key = self._get_key()
        if self._prev_cache_key != cur_key:
            self._prev_cache_key = cur_key
            self._version_counter += 1

    def _produce_dyn_sizes(self,
                           ex_size: Sequence[int],
                           source: Source,
                           symbolic_context: SymbolicContext
                           ) -> List[sympy.Expr]:
        return self._produce_dyn_sizes_from_int_tuple(tuple(ex_size), source, symbolic_context)

    def _produce_dyn_sizes_from_int_tuple(self,
                                          tensor_size: Tuple[int],
                                          source: Source,
                                          symbolic_context: SymbolicContext,
                                          ) -> List[sympy.Expr]:
        assert all(not is_symbolic(val) for val in tensor_size), f"Expect size to be a plain tuple of ints but got {tensor_size}"
        from torch._dynamo.source import TensorPropertySource, TensorProperty
        _assert_symbol_context(symbolic_context)
        dynamic_dims = symbolic_context.dynamic_sizes
        constraint_dims = symbolic_context.constraint_sizes
        size = []
        for i, val in enumerate(tensor_size):
            size.append(self.create_symbol(
                val,
                TensorPropertySource(source, TensorProperty.SIZE, i),
                dynamic_dims[i],
                constraint_dims[i],
                symbolic_context=symbolic_context
            ))
        return size

    def create_symbolic_sizes_strides_storage_offset(
        self,
        ex: torch.Tensor,
        source: Source,
        *,
        symbolic_context: Optional[SymbolicContext] = None,
    ):
        """
        Returns a list of symbolic sizes and strides for the given tensor.
        We try our best to express stride in terms of the sizes, so as to not
        introduce new symbolic variables.
        """

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
        def maybe_specialize_sym_int_with_hint(maybe_sym) -> int:
            assert isinstance(maybe_sym, (int, torch.SymInt))
            if is_symbolic(maybe_sym):
                assert maybe_sym.node.shape_env is not self, \
                    "expect the symbol is created from an shape env other than current one."
                return maybe_sym.node.require_hint()
            return maybe_sym

        ex_size = tuple(maybe_specialize_sym_int_with_hint(sz) for sz in ex.size())
        ex_stride = tuple(maybe_specialize_sym_int_with_hint(sd) for sd in ex.stride())
        ex_storage_offset = maybe_specialize_sym_int_with_hint(ex.storage_offset())

        return self._create_symbolic_sizes_strides_storage_offset(
            ex_size,
            ex_stride,
            ex_storage_offset,
            [_is_dim_dynamic(ex, i) for i in range(ex.dim())],
            source,
            symbolic_context=symbolic_context,
        )

    @record_shapeenv_event()
    def _create_symbolic_sizes_strides_storage_offset(
        self,
        ex_size: Sequence[int],
        ex_stride: Sequence[int],
        ex_storage_offset: int,
        is_dim_dynamic: Sequence[bool],
        source: Source,
        *,
        symbolic_context: Optional[SymbolicContext] = None,
    ):
        dim = len(ex_size)

        # Reimplement the legacy behavior
        if symbolic_context is None:
            constraint_dims = [None] * dim
            dynamic_dims = []
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
            dynamic_dims = [DimDynamic.DUCK] * dim
            # symbolic_context is None - set one
            symbolic_context = StatelessSymbolicContext(dynamic_sizes=dynamic_dims, constraint_sizes=constraint_dims)
        # We got a StatelessSymbolicContext
        _assert_symbol_context(symbolic_context)
        constraint_dims = symbolic_context.constraint_sizes
        dynamic_dims = symbolic_context.dynamic_sizes

        # TODO: make this configurable from outside symbolic_context; we made a symbolic_context
        # decision here where if all sizes are static, we are going to
        # specialize all of the inner strides/offset too. We don't have to
        # do this, and arguably we should ALWAYS allow for dynamic offset,
        # this is cheap.
        # TODO: This should be DYNAMIC, using DUCK for BC
        dynamic_strides_offset = DimDynamic.STATIC if all(r == DimDynamic.STATIC for r in dynamic_dims) else DimDynamic.DUCK

        assert len(dynamic_dims) == dim, f"{len(dynamic_dims)} != {dim}"
        assert len(constraint_dims) == dim

        from torch._dynamo.source import TensorPropertySource, TensorProperty
        size: List[sympy.Expr] = self._produce_dyn_sizes_from_int_tuple(ex_size, source, symbolic_context)
        stride: List[Optional[sympy.Expr]] = [None] * len(size)
        for i, val in enumerate(ex_stride):
            if val in (0, 1):
                stride[i] = sympy.Integer(val)
        while any(x is None for x in stride):
            candidates = {
                ex_size[i] * ex_stride[i]: size[i] * stride[i]
                for i in range(len(size))
                if stride[i] is not None and ex_stride[i] >= 0
            }
            # iterate over unbound strides in sorted order
            val_list = sorted(
                [(ex_stride[i], i) for i in range(len(stride)) if stride[i] is None],
                key=lambda tup: (
                    # Order singletons by their coefficients.
                    # 1 here to order singletons after non-singletons.
                    (1, tup[0].node.singleton_coeff(), tup[1]) if is_singleton(tup[0])
                    else (0, *tup)
                )
            )
            for _, i in val_list:
                if stride[i] is None and ex_stride[i] in candidates:
                    stride[i] = candidates[ex_stride[i]]
                    candidates[ex_size[i] * ex_stride[i]] = size[i] * stride[i]

            if any(x is None for x in stride):
                # bind the smallest unbound stride to a new variable
                val, i = min(
                    [
                        (ex_stride[i], i)
                        for i in range(len(stride))
                        if stride[i] is None
                    ]
                )
                stride[i] = self.create_symbol(
                    val,
                    TensorPropertySource(source, TensorProperty.STRIDE, i),
                    dynamic_dim=dynamic_strides_offset,
                    constraint_dim=None,
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
            sym_stride.append(self.create_symintnode(
                stride_expr, hint=ex_stride[i], source=TensorPropertySource(source, TensorProperty.STRIDE, i)))
        sym_storage_offset = self.create_symintnode(
            self.create_symbol(
                ex_storage_offset,
                TensorPropertySource(source, TensorProperty.STORAGE_OFFSET),
                dynamic_dim=dynamic_strides_offset,
                constraint_dim=None,
                symbolic_context=symbolic_context
            ),
            hint=ex_storage_offset,
            source=TensorPropertySource(source, TensorProperty.STORAGE_OFFSET))
        return tuple(sym_sizes), tuple(sym_stride), sym_storage_offset

    # If you know what the current hint value of the SymInt to be created
    # is, pass it into hint.  Otherwise, pass None and we will make our best
    # guess
    @record_shapeenv_event()
    def create_symintnode(
            self,
            sym: "sympy.Expr",
            *,
            hint: Optional[int],
            source: Optional[Source] = None,
    ):
        source_name = source.name() if source else None

        if self._translation_validation_enabled and source is not None:
            # Create a new symbol for this source.
            symbol = self._create_symbol_for_source(source)
            assert symbol is not None

            # Create a new FX placeholder and Z3 variable for 'symbol'.
            fx_node = self.create_fx_placeholder_and_z3var(symbol, int)

            # Add an equality assertion for the newly created symbol and 'sym'.
            self._add_assertion(sympy.Eq(symbol, sym))
        else:
            fx_node = None

        if isinstance(sym, sympy.Integer):
            if hint is not None:
                assert int(sym) == hint
            out = int(sym)
        else:
            out = SymInt(SymNode(sym, self, int, hint, fx_node=fx_node))
        return out

    @record_shapeenv_event()
    def create_unspecified_symint_and_symbol(self, value, source, dynamic_dim):
        return self.create_symintnode(
            self.create_unspecified_symbol(
                value,
                source=source,
                dynamic_dim=dynamic_dim,
            ),
            hint=value,
            source=source,
        )

    def create_symboolnode(self, sym: "sympy.Expr"):
        # This function is only being used in serialization, so we do not track it
        # for validation.
        return SymBool(SymNode(sym, self, bool, None))

    @record_shapeenv_event()
    def create_unbacked_symfloat(self):
        symbol: sympy.Symbol = sympy.Symbol(f"f{next(self.unbacked_symfloat_counter)}")
        self.counter["create_unbacked_symbol"] += 1
        self.var_to_stack[symbol] = CapturedTraceback.extract(skip=1)
        self.var_to_range[symbol] = ValueRanges.unknown()

        # Create a new FX placeholder and Z3 variable for 'symbol'.
        fx_node = self.create_fx_placeholder_and_z3var(symbol, float)

        return SymFloat(SymNode(symbol, self, float, None, fx_node=fx_node))

    @record_shapeenv_event()
    def create_unbacked_symint(self):
        symbol: sympy.Symbol = sympy.Symbol(f"i{next(self.unbacked_symint_counter)}", integer=True)
        self.counter["create_unbacked_symbol"] += 1
        self.var_to_stack[symbol] = CapturedTraceback.extract(skip=1)
        vr = self.var_to_range[symbol] = self._default_unspecified_value_range()

        # Create a new FX placeholder and Z3 variable for 'symbol'.
        fx_node = self.create_fx_placeholder_and_z3var(symbol, int)

        fsummary, user_tb, maybe_user_loc = self._get_stack_summary()
        log.info("create_unbacked_symbol %s [%s, %s]%s (%s)", symbol, vr.lower, vr.upper, maybe_user_loc, format_frame(fsummary))

        return SymInt(SymNode(symbol, self, int, None, fx_node=fx_node))

    def is_unbacked_symint(self, symbol: sympy.Symbol) -> bool:
        # NB: keep synced with free_unbacked_symbols
        return str(symbol).startswith("i")

    @record_shapeenv_event()
    def create_unbacked_symbool(self):
        symbol: sympy.Symbol = sympy.Symbol(f"i{next(self.unbacked_symint_counter)}", integer=True)
        self.counter["create_unbacked_symbol"] += 1
        self.var_to_stack[symbol] = CapturedTraceback.extract(skip=1)
        self.var_to_range[symbol] = ValueRanges(0, 1)

        # Create a new FX placeholder and Z3 variable for 'symbol'.
        fx_node = self.create_fx_placeholder_and_z3var(symbol, bool)

        return SymBool(SymNode(sympy.Eq(symbol, 1), self, bool, None, fx_node=fx_node))

    @record_shapeenv_event()
    def create_unspecified_symbol(
        self,
        val: Union[int, SymInt],
        source: Source,
        dynamic_dim: DimDynamic = DimDynamic.DUCK,
        constraint_dim: DimConstraint = None,  # NB: includes None
    ) -> "sympy.Expr":
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
            symbolic_context=None)

    @record_shapeenv_event()
    def create_symbol(
        self,
        val: int,
        source: Source,
        dynamic_dim: DimDynamic = DimDynamic.DUCK,
        constraint_dim: DimConstraint = None,  # NB: includes None
        positive: Optional[bool] = True,
        do_not_specialize_zero_one: bool = False,
        symbolic_context=None,
    ) -> "sympy.Expr":
        # see note [Tensor Fakification and Symbol Caching]
        source_name = source.name()
        if (isinstance(symbolic_context, StatefulSymbolicContext)
                and id(self) not in symbolic_context.shape_env_to_source_to_symbol_cache):
            symbolic_context.shape_env_to_source_to_symbol_cache[id(self)] = {}

        if (isinstance(symbolic_context, StatefulSymbolicContext)
                and source_name
                and (source_name in symbolic_context.shape_env_to_source_to_symbol_cache[id(self)])):
            return symbolic_context.shape_env_to_source_to_symbol_cache[id(self)][source_name]

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
                symbolic_context.shape_env_to_source_to_symbol_cache[id(self)][source_name] = out
            return out

        elif dynamic_dim is DimDynamic.DUCK:
            # duck_shape can be used to globally turn off duck shaping, even
            # if it was requested
            duck = self.duck_shape
        elif dynamic_dim is DimDynamic.DYNAMIC:
            duck = False
        else:
            raise AssertionError(f"unhandled dynamic_dim {dynamic_dim}")

        if val in (0, 1) and specialize_zero_one:
            r = self.val_to_var[val]
        elif not duck or val not in self.val_to_var:
            # If we're not duck shaping, we always create a new symbol
            # Even if we're duck shaping, if we haven't seen this particular
            # value before, we also create a new symbol
            sympy_expr = sympy.Symbol(f"s{len(self.var_to_val)}", positive=positive, integer=True)
            # We always associate vars to vals
            if isinstance(val, int):
                self.var_to_val[sympy_expr] = sympy.Integer(val)
            else:
                # Only used for jagged layout nested tensors
                self.var_to_val[sympy_expr] = SingletonInt(val.node.singleton_int(), coeff=val.node.singleton_coeff())

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
                else:
                    self.var_to_range[sympy_expr] = self._default_unspecified_value_range()

                # Small performance optimization: if we have a min-max constraint,
                # we can proactively narrow to that range
                if isinstance(constraint_dim, StrictMinMaxConstraint):
                    assert not duck
                    self.var_to_range[sympy_expr] &= constraint_dim.vr

                vr = self.var_to_range[sympy_expr]

                if val not in vr:
                    raise ConstraintViolationError(f"{val} not in range [{vr.lower}, {vr.upper}]")

                # Initialize default runtime range to match compile time range,
                # for backed SymInts (this is allowed to diverge for unbacked)
                self.runtime_var_to_range[sympy_expr] = vr

                range_str = f"[{vr.lower}, {vr.upper}]"
            else:
                # Skip var_range logic for SingletonInt
                # Only used for jagged layout nested tensors
                range_str = ""

            r = sympy_expr
            self.log.info("create_symbol %s = %s for %s %s", sympy_expr, val, source.name(), range_str)
            self.counter["create_symbol"] += 1
        else:
            # This implements duck-shaping: input sizes that match are assigned
            # the same symint
            r = self.val_to_var[val]
            self.log.debug("create_symbol %s duck sized %s", r, source.name())

        if isinstance(r, sympy.Symbol):
            self.var_to_sources[r].append(source)

        if isinstance(symbolic_context, StatefulSymbolicContext) and source_name:
            symbolic_context.shape_env_to_source_to_symbol_cache[id(self)][source_name] = r
        return r

    def debug_name(self, source):
        src_name = source.name()
        return self.source_name_to_debug_name.get(src_name, src_name)

    def render_range_for_constraint_violation(self, source, c):
        if isinstance(c, StrictMinMaxConstraint):
            lower, upper = c.vr.lower, c.vr.upper
            default = self._default_value_range()
            if lower <= default.lower:
                lower = None
            if upper >= default.upper:
                upper = None
            c_render = f"{self.debug_name(source)} = {source.name()} in the specified range"
            if lower is not None and upper is not None:
                c_render += f" {lower} <= {self.debug_name(source)} <= {upper}"
            elif lower is None and upper is not None:
                c_render += f" {self.debug_name(source)} <= {upper}"
            elif lower is not None and upper is None:
                c_render += f" {lower} <= {self.debug_name(source)}"
            return c_render
        return c.render(source)

    # Generates a list of guards strings which, when evaluated in a context that
    # defines tensors for all the sources, returns True or False depending
    # on if the guards in the list evaluated to True or not.  Primarily used by Dynamo,
    # but this is also helpful for manual testing of guards (see
    # evaluate_guards_for_args)
    #
    # For convenience in testing, a source is allowed to be a str,
    # in which case we will assume it is a LocalSource
    #
    # simplified lets you omit duck sizing, equality and 0/1 guards.
    # This is useful for testing when you don't care about the boilerplate
    # guards, and it may be helpful for user output too (be careful though;
    # some equality guards are nontrivial!  It would be nice to get simplified
    # output to print them too).  It's private because it's not
    # intended for normal use
    def produce_guards(
        self,
        placeholders,
        sources,
        source_ref=lambda n: n.name(),
        *,
        input_contexts: Optional[DimList[SymbolicContext]] = None,
        equalities_inputs: Optional[Set[Tuple[Source, Source]]] = None,
        _simplified=False,
        # Indicates if we should produce guards for known static values.
        ignore_static=True,
    ) -> List[str]:
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

        assert len(placeholders) == len(sources)
        Tensorlike = (torch.Tensor, FakeTensorMeta)

        def _create_no_constraints_context(t):
            return StatelessSymbolicContext(
                # Ignored; only the constraints part is relevant below.
                dynamic_sizes=[DimDynamic.DYNAMIC] * t.dim(),
                constraint_sizes=[None] * t.dim()
            )

        # Expand optional inputs, or verify invariants are upheld
        if input_contexts is None:
            input_contexts = [
                _create_no_constraints_context(t) if isinstance(t, Tensorlike)
                else None for t in placeholders
            ]
        else:
            assert len(input_contexts) == len(placeholders)
            for i, (t, context) in enumerate(zip(placeholders, input_contexts)):
                if isinstance(t, Tensorlike):
                    if context is None:
                        input_contexts[i] = _create_no_constraints_context(t)
                else:
                    assert isinstance(t, (SymInt, int))
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
        #   In particular, whenever we guard on equality (_maybe_guard_eq),
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
        # and what are not.  Below, we explain each of the guard expressions
        # we generate

        # TODO: Make this more efficient by binding all the size/stride/offsets
        # to locals before performing tests on them.

        from torch._dynamo.source import TensorPropertySource, TensorProperty, NegateSource

        # Actual codegen must be delayed as we don't necessarily know what
        # the symbol mapping is
        input_guards = []

        symbol_to_source = collections.defaultdict(list)
        symbol_to_constraints = collections.defaultdict(set)
        constraint_violations : List[Tuple[bool, Callable[[], str]]] = []

        def record_constraint_violation(warn_only, debug_name, msg, hint=None):
            constraint_violations.append(
                (warn_only, debug_name, lambda: f"{msg}{hint()}" if hint else msg)
            )

        def is_dim(src):
            return isinstance(src, TensorPropertySource) and src.prop is TensorProperty.SIZE

        if equalities_inputs:
            source_index = {}
            for i, src in enumerate(sources):
                source_index[src.name()] = i

            def get_symbol(tensor_dim_src):
                fake = placeholders[source_index[tensor_dim_src.base.name()]]
                symint = fake.shape[tensor_dim_src.idx]
                assert isinstance(symint, torch.SymInt)
                return symint.node.expr

            for src1, src2 in equalities_inputs.source_pairs:
                s1, s2 = get_symbol(src1), get_symbol(src2)
                concrete_val = self.evaluate_expr(sympy.Eq(s1, s2))
                if not concrete_val:
                    raise ConstraintViolationError(
                        f"{src1.name()} = {self.var_to_val[s1]}"
                        " is not equal to "
                        f"{src2.name()} = {self.var_to_val[s2]}"
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
        def track_symint(source, val, constraint=None):
            log.debug("track_symint %s %s %s", LazyString(source.name), val, constraint)
            assert not isinstance(val, SymInt) or is_symbolic(val)

            if isinstance(val, SymInt) and val.node.maybe_as_int() is not None:
                val = val.node.maybe_as_int()

            if isinstance(val, SymInt):
                s = val.node.expr
                if isinstance(s, sympy.Symbol):
                    symbol_to_source[s].append(source)
                    if constraint is not None:
                        symbol_to_constraints[s].add(constraint)
                elif isinstance(-s, sympy.Symbol):
                    symbol_to_source[-s].append(NegateSource(source))
                else:
                    constraint_violated = False
                    if isinstance(constraint, StrictMinMaxConstraint):
                        # try inferring the ranges of the expr s
                        sym_vrs = {x: self.var_to_range.get(x, None) for x in s.free_symbols}
                        if all(vr is not None for vr in sym_vrs.values()):
                            expr_vr = bound_sympy(s, sym_vrs)
                            if (expr_vr != constraint.vr):
                                # the expr and constrain ranges don't match
                                constraint_violated = True
                        else:
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
                        def hint(s):
                            sexpr = ShapeGuardPrinter(symbol_to_source, source_ref, self.var_to_sources).doprint(s)
                            return f"{sexpr}."

                        var_with_range = self.render_range_for_constraint_violation(source, constraint)
                        msg = (
                            f"Not all values of {var_with_range} are valid because "
                            f"{self.debug_name(source)} was inferred to be equal to "
                        )
                        record_constraint_violation(
                            constraint.warn_only,
                            self.debug_name(source),
                            msg,
                            hint=functools.partial(hint, s),
                        )

                input_guards.append((source, s))
            else:
                s = sympy.Integer(val)
                input_guards.append((source, s))
                constraint_violated = False
                if isinstance(constraint, StrictMinMaxConstraint):
                    constraint_violated = True
                elif isinstance(constraint, RelaxedUnspecConstraint):
                    # Don't complain about 0/1 specialization, we
                    # expect to have to compile in this case anyway
                    if val not in (0, 1):
                        constraint_violated = True
                if constraint_violated:
                    var_with_range = self.render_range_for_constraint_violation(source, constraint)
                    msg = (
                        f"Not all values of {var_with_range} are valid because "
                        f"{self.debug_name(source)} was inferred to be a constant ({val})."
                    )
                    record_constraint_violation(constraint.warn_only, self.debug_name(source), msg)

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
            assert isinstance(t, Tensorlike)
            if is_traceable_wrapper_subclass(t):
                from torch._dynamo.source import AttrSource

                assert isinstance(context, SubclassSymbolicContext)

                # For subclasses, we need to track symints on BOTH the outer
                # and inner tensors.
                sources_tensors_constraints = [
                    (source, t, context.constraint_sizes)
                ]
                attrs, _ = t.__tensor_flatten__()
                for attr in attrs:
                    inner_t = getattr(t, attr)
                    inner_context = context.inner_contexts[attr]
                    sources_tensors_constraints.append((
                        AttrSource(source, attr),
                        inner_t,
                        inner_context.constraint_sizes
                    ))
            else:
                sources_tensors_constraints = [(source, t, context.constraint_sizes)]

            for src, curr_t, constraint in sources_tensors_constraints:
                for i, ss in enumerate(curr_t.size()):
                    property_source = TensorPropertySource(src, TensorProperty.SIZE, i)
                    track_symint(property_source, ss, constraint[i])
                for i, ss in enumerate(curr_t.stride()):
                    track_symint(TensorPropertySource(src, TensorProperty.STRIDE, i), ss)
                track_symint(TensorPropertySource(src, TensorProperty.STORAGE_OFFSET), curr_t.storage_offset())

        # 1. Every input must equal the final simplified symbolic expression
        #    stored on the placeholder.  Given a placeholder (s0*2, s1),
        #    if we have an input (2, 3), we must show s0*2 == 2 and s1 == 3.
        #    This does a lot of work: it covers duck sizing and equality guards.
        exprs = []
        self.dim_constraints = DimConstraints(
            symbol_to_source,
            self.var_to_val,
            set(symbol_to_constraints.keys()),
            self.source_name_to_debug_name,
        )

        if not _simplified:
            for source, expr in input_guards:
                if self._translation_validation_enabled:
                    # Ignore sources that were not turned into SymInts.
                    srcname = source.name()
                    if srcname in self.source_to_symbol:
                        self._add_target_expr(sympy.Eq(self.source_to_symbol[srcname], expr))

                # Small optimization
                if (
                    isinstance(expr, sympy.Symbol) and
                    symbol_to_source.get(expr) and
                    source == symbol_to_source[expr][0]
                ):
                    continue

                # This logic excludes static values found on tensors from guarding, because
                # dynamo's check_tensor_fn does that (see guards.cpp).
                # However, for non tensor sources, we still need to guard here.
                if ignore_static and isinstance(source, TensorPropertySource):
                    if expr.is_number:
                        self.log.debug("Skipping guard %s", f"{source_ref(source)} == {expr}")
                        continue

                if is_dim(source):
                    self.dim_constraints.add_equality(source, expr)

                sexpr = ShapeGuardPrinter(symbol_to_source, source_ref, self.var_to_sources).doprint(expr)
                exprs.append(f"{source_ref(source)} == {sexpr}")
                if (
                    isinstance(expr, sympy.Symbol) and
                    expr in symbol_to_constraints and
                    isinstance(source, TensorPropertySource)
                    and source.prop is TensorProperty.SIZE
                    and equalities_inputs and
                    not equalities_inputs.is_equal(source, symbol_to_source[expr][0])
                ):
                    msg = (
                        f"The values of {self.debug_name(source)} = {source.name()} and "
                        f"{self.debug_name(symbol_to_source[expr][0])} = {symbol_to_source[expr][0].name()} "
                        "must always be equal."
                    )
                    record_constraint_violation(equalities_inputs.warn_only, self.debug_name(source), msg)
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
                if any(is_dim(source) for s in expr.free_symbols for source in symbol_to_source[s]):
                    is_trivial = self.dim_constraints.add(expr)
                guard_expr = ShapeGuardPrinter(symbol_to_source, source_ref, self.var_to_sources).doprint(expr)
                exprs.append(guard_expr)
                self._add_target_expr(expr)
                # A non-relational constraint on a single sizevar can violate
                # a constraint
                if not is_trivial and len(expr.free_symbols) == 1:
                    symbol = next(iter(expr.free_symbols))
                    source = symbol_to_source[symbol][0]
                    constraints = symbol_to_constraints[symbol]
                    for c in constraints:
                        if isinstance(c, StrictMinMaxConstraint):
                            var_with_range = self.render_range_for_constraint_violation(source, c)
                            msg = (
                                f"Not all values of {var_with_range} "
                                f"satisfy the generated guard {guard_expr}."
                            )
                            record_constraint_violation(c.warn_only, self.debug_name(source), msg)
                        elif isinstance(c, RelaxedUnspecConstraint):
                            # This is fine, we allow guards here as long as it
                            # didn't constrain it to one value  (we don't
                            # actually know this; this depends on our
                            # ValueRanges reasoning capability)
                            pass
                        else:
                            raise AssertionError(f"unrecognized constraint {c}")
            except Exception:
                self.log.warning("Failing guard allocated at: \n%s", ''.join(guard.stack.format()))
                raise

        # First, issue all the non-trivial guards.
        for guard in self.guards:
            if self._maybe_evaluate_static(guard.expr) is not None:
                continue
            issue_guard(guard)

        # Then, issue the guards that refine the value range of tracked symbols.
        # We need to explicitly issue these guards, since they are the ones that
        # guarantee the symbol's value range. Plus, due to the updated value
        # range, they may be skipped in the previous step.
        for symbol, guards in self.var_to_guards.items():
            if symbol not in symbol_to_source:
                continue
            for guard in guards:
                if guard is not None:
                    issue_guard(guard)

        # 3. Every symbol must be within its value range (this handles 0/1
        # specialization too).  NB: because we never update value ranges
        # except in case of explicit user annotation, these are not included
        # in simplified.  However, when we start updating value ranges
        # these should probably get reported in tests too
        if not _simplified:
            for symbol, sources in symbol_to_source.items():
                r = self.runtime_var_to_range.get(symbol)
                if r is None:
                    if symbol not in self.var_to_range:
                        continue
                    r = self.var_to_range[symbol]

                assert sources
                assert symbol.is_integer
                g_lower, g_upper = self.var_to_guards.get(symbol, (None, None))
                bounds = []
                if r.lower != -sympy.oo and g_lower is None:
                    if any(is_dim(source) for source in sources):
                        self.dim_constraints.add(sympy.Ge(symbol, r.lower))
                    bounds.append(str(r.lower))
                bounds.append(source_ref(sources[0]))
                # NB: This looks like an off-by-one error but it's not: the
                # upper bound may be sys.maxsize - 1 because we intentionally
                # exclude sys.maxsize from our bounds to deal with direct
                # == INT_MAX guards, but it's still dumb to actually test it.
                # Note that you can be off by a pretty large constant and it
                # won't matter because sizes in practice will be no where near
                # the 64-bit limit.
                if r.upper != sympy.oo and r.upper < sys.maxsize - 1 and g_upper is None:
                    if any(is_dim(source) for source in sources):
                        self.dim_constraints.add(sympy.Le(symbol, r.upper))
                    bounds.append(str(r.upper))
                if len(bounds) > 1:
                    exprs.append(" <= ".join(bounds))

        if constraint_violations:
            warn_msgs = []
            error_msgs = []
            debug_names = set()
            for warn_only, debug_name, msg in constraint_violations:
                if warn_only:
                    msg = f"  {len(warn_msgs) + 1}. {msg()}"
                    warn_msgs.append(msg)
                else:
                    msg = f"  - {msg()}"
                    error_msgs.append(msg)
                    debug_names.add(debug_name)
            if len(error_msgs) > 0:
                debug_names = ', '.join(debug_names)
                err = '\n'.join(error_msgs)
                raise ConstraintViolationError(
                    f"Constraints violated ({debug_names})! "
                    "For more information, run with TORCH_LOGS=\"+dynamic\".\n"
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
            #
            # NB: do NOT use runtime var ranges, they're unsound!  You will
            # only get correct TV with the compile-time ranges.
            for sym, vr in self.var_to_range.items():
                if vr.lower != -sympy.oo:
                    self._add_target_expr(sympy.Le(vr.lower, sym))
                if vr.upper != sympy.oo:
                    self._add_target_expr(sympy.Le(sym, vr.upper))

            # Before validating, populate the input of the validator with the
            # built FX graph.
            with fx_traceback.preserve_node_meta():
                PopulateValidator(self.graph, self.validator).run()

        self._check_translation_validate()
        return exprs

    def produce_guards_expression(self, placeholders, ignore_static=True):
        """
        Expected to be used with evaluate_guards_expression(). Produces the guards
        for the given placeholders and returns a string expression to be evaluated
        by evaluate_guards_expression given concrete values for the placeholders.
        """
        from torch._dynamo.source import LocalSource
        arg_names = [f"t{i}" for i in range(len(placeholders))]
        guards = self.produce_guards(placeholders, [LocalSource(a) for a in arg_names], ignore_static=ignore_static)
        if guards:
            return " and ".join(guards)
        return None

    def evaluate_guards_expression(self, code, args):
        """
        Expected to be used with produce_guards_expression(). Evaluates an expression
        generated by produce_guards_expression for the given concrete args.
        """
        arg_names = [f"t{i}" for i in range(len(args))]
        return eval(code, SYMPY_INTERP, {"L": dict(zip(arg_names, args))})

    def evaluate_guards_for_args(self, placeholders, args, *, ignore_static=True):
        code = self.produce_guards_expression(placeholders, ignore_static=ignore_static)
        if code:
            return self.evaluate_guards_expression(code, args)
        return True

    def bind_symbols(self, placeholders, args):
        # Given a paired list of placeholders (fake tensors with
        # symbolic sizes) and concrete arguments (regular tensors
        # with real sizes), returns a dictionary mapping each
        # symbol to its real value.  So for example, if you
        # have a placeholder with size (s0, s1), binding
        # (2, 4) to it will give you {s0: 2, s1: 4}.  This is
        # not guaranteed to bind ALL symbols in the ShapeEnv;
        # we can't bind a symbol if it doesn't occur in any placeholder,
        # and symbols that already have replacements won't get bindings.

        # This is a little duplicative with evaluate_guards but
        # it's different enough that it seemed cleanest to make
        # another copy.  This assumes the guards are already checked,
        # though if it's cheap we'll check for shenanigans
        bindings: Dict[sympy.Symbol, int] = {}

        def bind_symint(arg, val):
            if isinstance(val, SymInt):
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

    def get_nontrivial_guards(self):
        return [self.simplify(guard.expr) for guard in self.guards if self._maybe_evaluate_static(guard.expr) is None]

    def format_guards(self, verbose=False):
        def format_tb(tb):
            if not verbose:
                return ""
            return f"\n   Guarded at:\n{''.join('   ' + l for l in tb.format())}"

        return '\n'.join(f" - {guard.expr}{format_tb(guard.stack)}" for guard in self.guards)

    def get_shape_groups(self):
        shape_groups = collections.defaultdict(list)
        for k, v in self.replacements.items():
            shape_groups[v].append(k)
        return shape_groups

    @_lru_cache
    def _maybe_evaluate_static(
        self, expr: "sympy.Expr", *, unbacked_only: bool = False, compute_hint: bool = False,
        expect_rational=True,
    ) -> "Optional[sympy.Expr]":
        """
        Tries to evaluate expr without introducing guards

        If unbacked_only == True, then we only do substitutions on
        unbacked SymInts (leaving regular hinted integers alone).  This could
        result in an expression that still contains backed SymInts, which you
        could then potentially guard on.

        Use compute_hint == True if you are trying to compute a non-binding
        hint for the particular hint values of backed SymInts, e.g., if
        s0 happens to be 3 this run, compute_hint will subsitute s0 with 3.
        """
        expr = self.simplify(expr)

        if compute_hint:
            expr = expr.xreplace(self.var_to_val)

        expr = canonicalize_bool_expr(expr)

        symbols = list(expr.free_symbols)

        # Apply known runtime asserts
        for s in symbols:
            # Unbacked symints only
            if s in self.var_to_val:
                continue
            subst = {}
            for ra in self.deferred_runtime_asserts.get(s, ()):
                if compute_hint:
                    e = canonicalize_bool_expr(ra.expr.xreplace(self.var_to_val))
                else:
                    e = ra.expr
                # e is already canonical
                subst[e] = sympy.true
                subst[canonicalize_bool_expr(sympy.Not(e))] = sympy.false
                if isinstance(e, sympy.Eq):
                    subst[sympy.Le(e.lhs, e.rhs)] = sympy.true
                    subst[sympy.Le(-e.lhs, -e.rhs)] = sympy.true
                    subst[sympy.Lt(e.lhs, e.rhs)] = sympy.false
                    subst[sympy.Lt(-e.lhs, -e.rhs)] = sympy.false

            # NB: this helps us deal with And/Or connectives
            expr = expr.subs(subst)

        # Simplify making use of value range lower bound
        new_shape_env = {}
        new_range_env = {}
        for idx, k in enumerate(symbols):
            if isinstance(self.var_to_val.get(k, None), SingletonInt):
                # Skip var_to_range logic for SingletonInt which is only used
                # for jagged layout NestedTensors today
                continue
            vr = self.var_to_range[k]
            # Don't do anything if we don't have a nontrivial lower bound
            # Also don't do anything if we asked only to simplify unbacked
            # SymInt
            if (
                vr.lower < (-sys.maxsize - 1) // 2 or
                (unbacked_only and k in self.var_to_val)
            ):
                new_range_env[k] = vr
                continue
            # Positive means >= 1
            # Positive - 1 means >= 0
            # Positive + lower - 1 means >= lower
            # The new symbol 's' is "too low", so when we substitute it in
            # we have to increase it by offset (and conversely, the new
            # variables have to have their value range bounds adjusted as
            # well)
            s = sympy.Symbol(f"shape_{idx}", positive=True, integer=True)
            offset = vr.lower - 1
            new_shape_env[k] = s + offset
            new_range_env[s] = SymPyValueRangeAnalysis.add(vr, -offset)

        def replace(expr, repl):
            return expr.xreplace(repl)

        try:
            new_expr = replace(expr, new_shape_env)
        except RecursionError:
            log.warning("RecursionError in sympy.xreplace(%s, %s)", expr, new_shape_env)
            self.counter["sympy_recursion_error"] += 1
            return None

        floor_div_replace = {}
        for atom in new_expr.atoms(FloorDiv):
            floor_div_replace[atom] = sympy.floor(atom.args[0] / atom.args[1])
        new_expr = safe_expand(new_expr.xreplace(floor_div_replace))
        # TODO: when unbacked_only, can sometimes early return even when there
        # are still free symbols
        if new_expr.is_number:
            return new_expr

        # Check if the range can solve it statically
        out = bound_sympy(new_expr, new_range_env)
        if expect_rational:
            _assert_bound_is_rational(new_expr, out)
            if out.is_singleton():
                return out.lower

        return new_expr if unbacked_only else None

    @_lru_cache
    def replace(self, expr: "sympy.Expr") -> "sympy.Expr":
        replacements = {s: self._find(cast(sympy.Symbol, s)) for s in expr.free_symbols}
        return safe_expand(expr.xreplace(replacements))

    @_lru_cache
    def _update_divisible(self):
        new_divisible = set()
        for k in self.divisible:
            res = self.replace(k)
            if not res.is_number:
                new_divisible.add(k)

        self.divisible = new_divisible
        self._update_version_counter()

    @_lru_cache
    def simplify(self, expr: "sympy.Expr") -> "sympy.Expr":
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
                    if self.replace(Mod(base, divisor)) in self.divisible and \
                            base == base1 and self.replace(Mod(base1, divisor1)) in self.divisible:
                        div_replacements[atom] = divisor1
            expr = expr.xreplace(div_replacements)
            expr = safe_expand(expr)
        if expr.has(FloorDiv):
            div_replacements = {}
            pows = expr.atoms(sympy.Pow)
            rationals = expr.atoms(sympy.Rational).difference(expr.atoms(sympy.Integer))
            for fd in expr.atoms(FloorDiv):
                base, divisor = fd.args
                if self.replace(Mod(base, divisor)) in self.divisible:
                    div_replacements[fd] = base / divisor
            new_expr = expr.xreplace(div_replacements)
            new_expr = safe_expand(new_expr)
            new_pows = new_expr.atoms(sympy.Pow)
            new_rationals = new_expr.atoms(sympy.Rational).difference(new_expr.atoms(sympy.Integer))
            # divisions simplified away
            if new_pows.issubset(pows) and new_rationals.issubset(rationals):
                expr = new_expr
        return expr

    @lru_cache(256)
    def size_hint(self, expr: "sympy.Expr", *, allow_none=False):
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
            raise self._make_data_dependent_error(result_expr, expr)
        return result_expr

    # NB: keep in sync with size_hint
    @lru_cache(256)
    def has_hint(self, expr: "sympy.Expr"):
        result_expr = safe_expand(expr).xreplace(self.var_to_val)
        return result_expr.is_number or self._maybe_evaluate_static(result_expr) is not None

    def _make_data_dependent_error(self, expr, unhinted_expr):
        # TODO: in a Dynamo context, having user code, and having the
        # name of the local, will be much better
        for s in expr.free_symbols:
            stacktrace = ''.join(self.var_to_stack[s].format())
            self.log.debug("Data dependent variable '%s' allocated at:\n%s", s, stacktrace)
        return GuardOnDataDependentSymNode(
            "It appears that you're trying to get a value out of symbolic int/float "
            "whose value is data-dependent (and thus we do not know the true value.)  "
            f"The expression we were trying to evaluate is {expr} (unhinted: {unhinted_expr}).  "
            "For more information, run with TORCH_LOGS=\"+dynamic\".\n"
            # TODO: Help text about how to use our runtime tests to fix this
            # problem
        )

    def _set_replacement(self, a: "sympy.Symbol", expr: "sympy.Expr") -> None:
        """
        Adds or updates a replacement for a symbol.
        Use this instead of `self.replacements[a] = expr`.
        """
        if config.print_specializations and isinstance(expr, (sympy.Integer, sympy.Float)):
            # specializing to a constant, which is likely unexpected

            # NOTE(avik): It is possible that we try logging the same specialization multiple times, e.g.,
            # when adding a to self.replacements, and again when simplifying an expression containing a.
            # Thus to avoid duplication, checking whether a is in self.replacements isn't enough; if it is,
            # it must not already map to `expr`. Fortunately this check is cheap because `expr` is a constant.
            if a not in self.replacements or expr != self.replacements[a]:
                self.log.warning("Specializing %s to %s", self.var_to_sources[a][0].name(), expr)
                self.log.debug("SPECIALIZATION", stack_info=True)
        log.info("set_replacement %s = %s", a, expr)
        self.replacements[a] = expr
        self._update_version_counter()

        # When specializing 'a == expr', the equality should be also conveyed to
        # Z3, in case an expression uses 'a'.
        self._add_target_expr(sympy.Eq(a, expr))

    def _add_divisible(self, expr: "sympy.Expr"):
        self.divisible.add(expr)
        self._update_version_counter()

    @_lru_cache
    @record_shapeenv_event()
    def _find(self, a: "sympy.Symbol") -> "sympy.Expr":
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
        self._set_replacement(a, self.replacements[a].xreplace(cur_replace))
        return self.replacements[a]

    @lru_cache(256)
    def _maybe_guard_eq(self, expr: Union["sympy.Eq", "sympy.Ne"], concrete_bool: bool) -> None:
        """
        Evaluates the result of an eq call. If true, uses information to
        simplify shapes (i.e. a == b or a % 5 == 0)
        """
        assert type(concrete_bool) is bool
        if isinstance(expr, sympy.Eq):
            if not concrete_bool:
                return
        # NB: Apparently this is load bearing; to see what test fails if
        # you comment it out run:
        # python test/functorch/test_aotdispatch.py -k
        # test_aot_autograd_symbolic_module_exhaustive_nn_LazyConv3d_cpu_float32
        elif isinstance(expr, sympy.Ne):
            if concrete_bool:
                return
        free = list(expr.free_symbols)

        assert len(free) > 0, f"The expression should not be static by this point: {expr}"
        # In case of really gnarly expression, we don't blow up
        if len(free) > 5:
            return
        # NB: prioritize unbacked symints for solving by ordering them last
        free = sorted(free, key=lambda x: (self.size_hint(x, allow_none=True) or sys.maxsize, x.name), reverse=True)  # type: ignore[attr-defined]
        lhs = expr.lhs
        rhs = expr.rhs
        if not expr.has(Mod):
            try:
                floor_div_atoms = lhs.atoms(FloorDiv).union(rhs.atoms(FloorDiv))
                if len(floor_div_atoms) > 0 and any(a.divisor != 1 for a in floor_div_atoms):
                    raise NotImplementedError
                # short-circuit when no solving is needed
                if isinstance(lhs, sympy.Symbol) and free_unbacked_symbols(lhs):
                    self._set_replacement(lhs, self._find(rhs))
                elif isinstance(rhs, sympy.Symbol) and free_unbacked_symbols(rhs):
                    self._set_replacement(rhs, self._find(lhs))
                else:
                    r = try_solve(expr, free[0], floordiv_inequality=False)
                    if r is not None and all(t.is_integer for t in sympy.preorder_traversal(r[1])):
                        new_var = self._find(r[1])
                        ok = False
                        if self.is_unbacked_symint(free[0]):
                            # If you have i0 + i1 + i2 = s0, don't substitute i2 =
                            # s0 - i0 - i1.  Arguably this should be OK but the
                            # runtime assert machinery is very delicate right now
                            # so this causes things to fail e.g.,
                            # test_split_unbacked_sizes
                            ok = len(free_unbacked_symbols(new_var)) <= 1
                        else:
                            # Never substitute backed with unbacked
                            ok = len(free_unbacked_symbols(new_var)) == 0
                        if ok:
                            self._set_replacement(cast(sympy.Symbol, free[0]), new_var)
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
                    if isinstance(q, sympy.Number) and isinstance(p, sympy.Mul) and len(p.args) == 2:
                        c, i0 = p.args
                        # Given Mod(c * i0, q) == 0
                        if (
                            isinstance(c, sympy.Number) and
                            isinstance(i0, sympy.Symbol) and
                            self.is_unbacked_symint(i0)
                        ):
                            # We have Mod(i0, q / c) == 0, which means we can
                            # rewrite i0 as (q / gcd(q, c)) * i1
                            d = q / sympy.gcd(q, c)
                            i1 = self.create_unbacked_symint().node.expr
                            # Propagate the value ranges.  It doesn't really
                            # matter if we use truediv or floordiv, because we
                            # have established divisibility.
                            self.var_to_range[i1] = SymPyValueRangeAnalysis.truediv(
                                self.var_to_range[i0], ValueRanges.wrap(d)
                            )
                            self.runtime_var_to_range[i1] = SymPyValueRangeAnalysis.truediv(
                                self.runtime_var_to_range[i0], ValueRanges.wrap(d)
                            )
                            self._set_replacement(i0, d * i1)

            except NotImplementedError:
                pass
        return

    # See: Note - On 0/1 specialization
    # NB: sys.maxsize is NOT allowed for sizes, because we use MAX_INT
    # as a sentinel sometimes.  Your sizevar isn't going to be
    # anywhere near the max 64-bit integer anyway.
    def _default_value_range(self) -> ValueRanges:
        lower = 2 if self.specialize_zero_one else 0
        return ValueRanges(lower, sys.maxsize - 1)

    def _default_unspecified_value_range(self) -> ValueRanges:
        return ValueRanges(-sys.maxsize - 1, sys.maxsize)

    @_lru_cache
    def _simplify_floor_div(self, expr):
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
    def _check_frozen(self, expr, concrete_val):
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
            log.warning("Ignored guard %s == %s, this could result in accuracy problems", expr, concrete_val)


    def _get_stack_summary(self):
        fsummary = None
        frame = inspect.currentframe()
        try:
            while frame is not None:
                if frame.f_code.co_filename not in uninteresting_files():
                    fsummary = traceback.FrameSummary(
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
        maybe_user_loc = ""
        user_tb = TracingContext.extract_stack()
        if user_tb:
            maybe_user_loc = " at " + format_frame(user_tb[-1])

        return fsummary, user_tb, maybe_user_loc

    def _log_guard(self, prefix: str, g):
        if self.log.isEnabledFor(logging.INFO):
            fsummary, user_tb, maybe_user_loc = self._get_stack_summary()

            # TODO: make this an artifact
            is_debug = False
            maybe_extra_debug = ""
            if is_debug and user_tb:
                maybe_extra_debug = (
                    '\nUser Stack (most recent call last):\n' +
                    '  (snipped, see stack below for prefix)\n' +
                    ''.join(traceback.format_list(user_tb))
                )
            self.log.info(
                "%s %s [guard added]%s (%s)%s",
                prefix,
                g,
                maybe_user_loc,
                format_frame(fsummary),
                maybe_extra_debug,
                stack_info=is_debug,
            )

    @lru_cache(256)
    @record_shapeenv_event(save_tracked_fakes=True)
    def evaluate_expr(self, orig_expr: "sympy.Expr", hint=None, fx_node=None,
                      expect_rational=True):
        """
        Given an expression, evaluates it, adding guards if necessary
        """
        if hint is None:
            concrete_val = self.size_hint(orig_expr)
        else:
            concrete_val = sympy.sympify(hint)

        # Check if:
        #   1. 'translation_validation' is set
        #   2. the corresponding 'fx_node' is not 'None'
        #   3. the guard should not be suppressed
        #
        # If all of the above check, we create an FX node representing the
        # actual expression to be guarded.
        node = None
        fresh = False
        if (
                self._translation_validation_enabled
                and fx_node is not None
                and not self._suppress_guards_tls()
        ):
            if concrete_val is sympy.true:
                node, fresh = self.create_fx_call_function(torch._assert, (fx_node,))
            elif concrete_val is sympy.false:
                neg, _ = self.create_fx_call_function(operator.not_, (fx_node,))
                node, fresh = self.create_fx_call_function(torch._assert, (neg,))
            else:
                eql, _ = self.create_fx_call_function(operator.eq, (fx_node, concrete_val))
                node, fresh = self.create_fx_call_function(torch._assert, (eql,))

            assert node is not None
            # If this is a fresh node, we have to remember the event index that
            # corresponds to this assertion node.
            # Reason: so that, given an assertion node, we can replay the ShapeEnv
            # events until the point where this assertion node was freshly created.
            if fresh:
                self.add_fx_node_metadata(node)

        # After creating the FX node corresponding to orig_expr, we must make sure that
        # no error will be raised until the end of this function.
        #
        # Reason: the translation validation may become invalid otherwise.
        #
        # If an error is raised before the end of this function, we remove the FX node
        # inserted, and re-raise the error.
        guard = None
        tb = None

        try:
            if orig_expr.is_number:
                self.log.debug("eval %s [trivial]", orig_expr)
                # NB: don't test float as there may be precision issues
                if isinstance(hint, (int, bool)):
                    assert orig_expr == hint, f"{orig_expr} != {hint}"
                return orig_expr

            expr = orig_expr

            static_expr = self._maybe_evaluate_static(expr,
                                                      expect_rational=expect_rational)
            if static_expr is not None:
                self.log.debug("eval %s == %s [statically known]", orig_expr, static_expr)
                # NB: don't test float as there may be precision issues
                if isinstance(hint, (int, bool)):
                    assert static_expr == hint, f"{static_expr} != {hint}"
                return static_expr

            if not (expr.free_symbols <= self.var_to_val.keys()):
                # TODO: dedupe this with _maybe_evaluate_static
                # Attempt to eliminate the unbacked SymInt
                new_expr = self._maybe_evaluate_static(expr, unbacked_only=True)
                if not (new_expr.free_symbols <= self.var_to_val.keys()):
                    raise self._make_data_dependent_error(expr.xreplace(self.var_to_val), expr)
                expr = new_expr

            self._check_frozen(expr, concrete_val)

            if (
                    config.inject_EVALUATE_EXPR_flip_equality_TESTING_ONLY
                    and isinstance(hint, bool)
                    and isinstance(expr, (sympy.Eq, sympy.Ne))
            ):
                expr = sympy.Not(expr)

            if isinstance(expr, (sympy.Eq, sympy.Ne)):
                self._maybe_guard_eq(expr, bool(concrete_val))
                # TODO: If we successfully eliminate a symbol via equality, it
                # is not actually necessary to save a guard for the equality,
                # as we will implicitly generate a guard when we match that
                # input against the symbol
            elif isinstance(concrete_val, sympy.Integer):
                # WARNING: we cannot actually do simplifications on guards
                # on floating point values, because Sympy generally does not
                # think expressions on integers can ever be equal to floating
                # point (e.g., sympy.Eq(s0/6, 0.5) evaluates to False).  Without
                # very clear algebraic laws that hold for floating point, such
                # simplifications are error prone anyway, so be sure not to
                # maybe_guard_eq in those cases.
                self._maybe_guard_eq(sympy.Eq(expr, concrete_val), True)

            if concrete_val is sympy.true:
                g = expr
            elif concrete_val is sympy.false:
                g = sympy.Not(expr)
            else:
                g = sympy.Eq(expr, concrete_val)  # type: ignore[arg-type]

            if not self._suppress_guards_tls():
                stack = CapturedTraceback.extract(skip=1)
                guard = ShapeGuard(g, stack)
                self.guards.append(guard)
        except Exception:
            if fresh:
                self.remove_fx_node(node)
            raise
        else:
            if not self._suppress_guards_tls():
                assert guard is not None

                self.refine_ranges(guard)

                self._log_guard("eval", g)
            else:
                self.log.debug("eval %s [guard suppressed]", g)

        return concrete_val

    def cleanup(self):
        # Break reference cycles.
        # This destroys the stacks. If you really want to keep them, we
        # just need some way to break references on code objects.
        for g in self.guards:
            g.stack.cleanup()
        for s in self.var_to_stack.values():
            s.cleanup()
        for ras in self.deferred_runtime_asserts.values():
            for ra in ras:
                ra.stack.cleanup()

    @record_shapeenv_event(save_tracked_fakes=True)
    def defer_runtime_assert(self, orig_expr: "sympy.Expr", msg, fx_node=None):
        expr = orig_expr

        static_expr = self._maybe_evaluate_static(expr)
        if static_expr is not None:
            self.log.debug("runtime_assert %s == %s [statically known]", orig_expr, static_expr)
            return static_expr

        # Attempt to eliminate the unbacked SymInt
        new_expr = self._maybe_evaluate_static(expr, unbacked_only=True)
        if new_expr.free_symbols <= self.var_to_val.keys():
            # Do a normal guard
            return self.evaluate_expr(new_expr, fx_node=fx_node)
        # NB: Don't use new_expr as expr; it could contain gunk like shape0
        # which we don't want to guard on

        # OK, we're definitely doing a runtime assert now
        if (
            self._translation_validation_enabled
            and fx_node is not None
            and not self._suppress_guards_tls()
        ):
            node, fresh = self.create_fx_call_function(torch._assert, (fx_node,))
            assert node is not None
            if fresh:
                self.add_fx_node_metadata(node)

        self._check_frozen(expr, sympy.true)

        # eliminate symbols on equality tests
        if isinstance(expr, sympy.Eq):
            self._maybe_guard_eq(expr, True)

        if not self._suppress_guards_tls():
            # canonicalise to remove equations that are trivially equal
            expr = canonicalize_bool_expr(expr)
            stack = CapturedTraceback.extract(skip=1)
            ra = RuntimeAssert(expr, msg, stack)
            # TODO: Do this in a way that is less janky than int(s.name[1:])
            cands = sorted([s for s in expr.free_symbols if s.name.startswith("i")], key=lambda s: int(s.name[1:]))
            self.deferred_runtime_asserts.setdefault(cands[-1], []).append(ra)
            self.num_deferred_runtime_asserts += 1
            self._update_version_counter()
            # TODO: refine ranges
            # Unfortunately, range refinement is probably going to not
            # work most of the time, because we don't support symbols
            # in ranges.  For example, i0 <= s0 is un-rangeable, because
            # we can't put s0 in the range.  So this is not very high
            # priority at the moment.
            self._log_guard("runtime_assert", expr)
        else:
            self.log.debug("runtime_assert %s [guard suppressed]", expr)

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
    def refine_ranges(self, guard: ShapeGuard) -> None:
        expr = self.simplify(guard.expr)

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
            _assert_bound_is_rational(rhs, rhs_vr)
            lower_guard, upper_guard = self.var_to_guards.get(symbol, (None, None))

            # Let's suppose that we have a preexisting range for x [0, 100].
            # Now, we issue a guard x > y, where the range for y is [50, 150].
            # Then, lower = 0, rhs_vr.lower = 50 and therefore refinement can happen,
            # refining x to [51, 100], since x must be greater than y, but the lowest
            # y could be is 50.
            #
            # sympy.Eq may update both lower and upper bounds.
            # sympy.G{t,e} may update the lower bound, only.
            # sympy.L{t,e} may update the upper bound, only.
            if lower < rhs_vr.lower and isinstance(r_expr, (sympy.Eq, sympy.Ge, sympy.Gt)):
                # Strictly greater relations allow us to refine a bit more, since
                # x < y implies that the lower bound for x is: y + 1.
                lower = rhs_vr.lower + int(isinstance(r_expr, sympy.Gt))
                lower_guard = guard
            if upper > rhs_vr.upper and isinstance(r_expr, (sympy.Eq, sympy.Le, sympy.Lt)):
                upper = rhs_vr.upper - int(isinstance(r_expr, sympy.Lt))
                upper_guard = guard

            # Do nothing if the new value range is no better than what we already have.
            if vr == ValueRanges(lower, upper):
                continue

            # Updates the range and the guards corresponding to each bound of the symbol.
            self.var_to_range[symbol] = ValueRanges(lower, upper)
            self.var_to_guards[symbol] = (lower_guard, upper_guard)
            # Clears the cache, since this update can change the result.
            self._maybe_evaluate_static.cache_clear()

def _is_int(expr):
    return isinstance(expr, SymInt) and expr.node.expr.is_number

# WARNING: This is legacy, DO NOT USE
def _is_dim_dynamic(t, d):
    return hasattr(t, "_dynamo_dynamic_indices") and d in t._dynamo_dynamic_indices
