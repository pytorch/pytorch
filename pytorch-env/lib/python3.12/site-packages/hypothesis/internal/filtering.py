# This file is part of Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Copyright the Hypothesis Authors.
# Individual contributors are listed in AUTHORS.rst and the git log.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Tools for understanding predicates, to satisfy them by construction.

For example::

    integers().filter(lambda x: x >= 0) -> integers(min_value=0)

This is intractable in general, but reasonably easy for simple cases involving
numeric bounds, strings with length or regex constraints, and collection lengths -
and those are precisely the most common cases.  When they arise in e.g. Pandas
dataframes, it's also pretty painful to do the constructive version by hand in
a library; so we prefer to share all the implementation effort here.
See https://github.com/HypothesisWorks/hypothesis/issues/2701 for details.
"""

import ast
import inspect
import math
import operator
from collections.abc import Collection
from decimal import Decimal
from fractions import Fraction
from functools import partial
from typing import Any, Callable, NamedTuple, Optional, TypeVar

from hypothesis.internal.compat import ceil, floor
from hypothesis.internal.floats import next_down, next_up
from hypothesis.internal.reflection import (
    extract_lambda_source,
    get_pretty_function_description,
)

Ex = TypeVar("Ex")
Predicate = Callable[[Ex], bool]


class ConstructivePredicate(NamedTuple):
    """Return kwargs to the appropriate strategy, and the predicate if needed.

    For example::

        integers().filter(lambda x: x >= 0)
        -> {"min_value": 0"}, None

        integers().filter(lambda x: x >= 0 and x % 7)
        -> {"min_value": 0}, lambda x: x % 7

    At least in principle - for now we usually return the predicate unchanged
    if needed.

    We have a separate get-predicate frontend for each "group" of strategies; e.g.
    for each numeric type, for strings, for bytes, for collection sizes, etc.
    """

    kwargs: dict[str, Any]
    predicate: Optional[Predicate]

    @classmethod
    def unchanged(cls, predicate: Predicate) -> "ConstructivePredicate":
        return cls({}, predicate)

    def __repr__(self) -> str:
        fn = get_pretty_function_description(self.predicate)
        return f"{self.__class__.__name__}(kwargs={self.kwargs!r}, predicate={fn})"


ARG = object()


def convert(node: ast.AST, argname: str) -> object:
    if isinstance(node, ast.Name):
        if node.id != argname:
            raise ValueError("Non-local variable")
        return ARG
    if isinstance(node, ast.Call):
        if (
            isinstance(node.func, ast.Name)
            and node.func.id == "len"
            and len(node.args) == 1
        ):
            # error unless comparison is to the len *of the lambda arg*
            return convert(node.args[0], argname)
    return ast.literal_eval(node)


def comp_to_kwargs(x: ast.AST, op: ast.AST, y: ast.AST, *, argname: str) -> dict:
    a = convert(x, argname)
    b = convert(y, argname)
    num = (int, float)
    if not (a is ARG and isinstance(b, num)) and not (isinstance(a, num) and b is ARG):
        # It would be possible to work out if comparisons between two literals
        # are always true or false, but it's too rare to be worth the complexity.
        # (and we can't even do `arg == arg`, because what if it's NaN?)
        raise ValueError("Can't analyse this comparison")

    of_len = {"len": True} if isinstance(x, ast.Call) or isinstance(y, ast.Call) else {}

    if isinstance(op, ast.Lt):
        if a is ARG:
            return {"max_value": b, "exclude_max": True, **of_len}
        return {"min_value": a, "exclude_min": True, **of_len}
    elif isinstance(op, ast.LtE):
        if a is ARG:
            return {"max_value": b, **of_len}
        return {"min_value": a, **of_len}
    elif isinstance(op, ast.Eq):
        if a is ARG:
            return {"min_value": b, "max_value": b, **of_len}
        return {"min_value": a, "max_value": a, **of_len}
    elif isinstance(op, ast.GtE):
        if a is ARG:
            return {"min_value": b, **of_len}
        return {"max_value": a, **of_len}
    elif isinstance(op, ast.Gt):
        if a is ARG:
            return {"min_value": b, "exclude_min": True, **of_len}
        return {"max_value": a, "exclude_max": True, **of_len}
    raise ValueError("Unhandled comparison operator")  # e.g. ast.Ne


def merge_preds(*con_predicates: ConstructivePredicate) -> ConstructivePredicate:
    # This function is just kinda messy.  Unfortunately the neatest way
    # to do this is just to roll out each case and handle them in turn.
    base = {
        "min_value": -math.inf,
        "max_value": math.inf,
        "exclude_min": False,
        "exclude_max": False,
    }
    predicate = None
    for kw, p in con_predicates:
        assert (
            not p or not predicate or p is predicate
        ), "Can't merge two partially-constructive preds"
        predicate = p or predicate
        if "min_value" in kw:
            if kw["min_value"] > base["min_value"]:
                base["exclude_min"] = kw.get("exclude_min", False)
                base["min_value"] = kw["min_value"]
            elif kw["min_value"] == base["min_value"]:
                base["exclude_min"] |= kw.get("exclude_min", False)
        if "max_value" in kw:
            if kw["max_value"] < base["max_value"]:
                base["exclude_max"] = kw.get("exclude_max", False)
                base["max_value"] = kw["max_value"]
            elif kw["max_value"] == base["max_value"]:
                base["exclude_max"] |= kw.get("exclude_max", False)

    has_len = {"len" in kw for kw, _ in con_predicates if kw}
    assert len(has_len) <= 1, "can't mix numeric with length constraints"
    if has_len == {True}:
        base["len"] = True

    if not base["exclude_min"]:
        del base["exclude_min"]
        if base["min_value"] == -math.inf:
            del base["min_value"]
    if not base["exclude_max"]:
        del base["exclude_max"]
        if base["max_value"] == math.inf:
            del base["max_value"]
    return ConstructivePredicate(base, predicate)


def numeric_bounds_from_ast(
    tree: ast.AST, argname: str, fallback: ConstructivePredicate
) -> ConstructivePredicate:
    """Take an AST; return a ConstructivePredicate.

    >>> lambda x: x >= 0
    {"min_value": 0}, None
    >>> lambda x: x < 10
    {"max_value": 10, "exclude_max": True}, None
    >>> lambda x: len(x) >= 5
    {"min_value": 5, "len": True}, None
    >>> lambda x: x >= y
    {}, lambda x: x >= y

    See also https://greentreesnakes.readthedocs.io/en/latest/
    """
    if isinstance(tree, ast.Compare):
        ops = tree.ops
        vals = tree.comparators
        comparisons = [(tree.left, ops[0], vals[0])]
        for i, (op, val) in enumerate(zip(ops[1:], vals[1:]), start=1):
            comparisons.append((vals[i - 1], op, val))
        bounds = []
        for comp in comparisons:
            try:
                kwargs = comp_to_kwargs(*comp, argname=argname)
                # Because `len` could be redefined in the enclosing scope, we *always*
                # have to apply the condition as a filter, in addition to rewriting.
                pred = fallback.predicate if "len" in kwargs else None
                bounds.append(ConstructivePredicate(kwargs, pred))
            except ValueError:
                bounds.append(fallback)
        return merge_preds(*bounds)

    if isinstance(tree, ast.BoolOp) and isinstance(tree.op, ast.And):
        return merge_preds(
            *(numeric_bounds_from_ast(node, argname, fallback) for node in tree.values)
        )

    return fallback


def get_numeric_predicate_bounds(predicate: Predicate) -> ConstructivePredicate:
    """Shared logic for understanding numeric bounds.

    We then specialise this in the other functions below, to ensure that e.g.
    all the values are representable in the types that we're planning to generate
    so that the strategy validation doesn't complain.
    """
    unchanged = ConstructivePredicate.unchanged(predicate)
    if (
        isinstance(predicate, partial)
        and len(predicate.args) == 1
        and not predicate.keywords
    ):
        arg = predicate.args[0]
        if (
            (isinstance(arg, Decimal) and Decimal.is_snan(arg))
            or not isinstance(arg, (int, float, Fraction, Decimal))
            or math.isnan(arg)
        ):
            return unchanged
        options = {
            # We're talking about op(arg, x) - the reverse of our usual intuition!
            operator.lt: {"min_value": arg, "exclude_min": True},  # lambda x: arg < x
            operator.le: {"min_value": arg},  #                      lambda x: arg <= x
            operator.eq: {"min_value": arg, "max_value": arg},  #    lambda x: arg == x
            operator.ge: {"max_value": arg},  #                      lambda x: arg >= x
            operator.gt: {"max_value": arg, "exclude_max": True},  # lambda x: arg > x
            # Special-case our default predicates for length bounds
            min_len: {"min_value": arg, "len": True},
            max_len: {"max_value": arg, "len": True},
        }
        if predicate.func in options:
            return ConstructivePredicate(options[predicate.func], None)

    # This section is a little complicated, but stepping through with comments should
    # help to clarify it.  We start by finding the source code for our predicate and
    # parsing it to an abstract syntax tree; if this fails for any reason we bail out
    # and fall back to standard rejection sampling (a running theme).
    try:
        if predicate.__name__ == "<lambda>":
            source = extract_lambda_source(predicate)
        else:
            source = inspect.getsource(predicate)
        tree: ast.AST = ast.parse(source)
    except Exception:
        return unchanged

    # Dig down to the relevant subtree - our tree is probably a Module containing
    # either a FunctionDef, or an Expr which in turn contains a lambda definition.
    while isinstance(tree, ast.Module) and len(tree.body) == 1:
        tree = tree.body[0]
    while isinstance(tree, ast.Expr):
        tree = tree.value

    if isinstance(tree, ast.Lambda) and len(tree.args.args) == 1:
        return numeric_bounds_from_ast(tree.body, tree.args.args[0].arg, unchanged)
    elif isinstance(tree, ast.FunctionDef) and len(tree.args.args) == 1:
        if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Return):
            # If the body of the function is anything but `return <expr>`,
            # i.e. as simple as a lambda, we can't process it (yet).
            return unchanged
        argname = tree.args.args[0].arg
        body = tree.body[0].value
        assert isinstance(body, ast.AST)
        return numeric_bounds_from_ast(body, argname, unchanged)
    return unchanged


def get_integer_predicate_bounds(predicate: Predicate) -> ConstructivePredicate:
    kwargs, predicate = get_numeric_predicate_bounds(predicate)  # type: ignore

    if "min_value" in kwargs:
        if kwargs["min_value"] == -math.inf:
            del kwargs["min_value"]
        elif math.isinf(kwargs["min_value"]):
            return ConstructivePredicate({"min_value": 1, "max_value": -1}, None)
        elif kwargs["min_value"] != int(kwargs["min_value"]):
            kwargs["min_value"] = ceil(kwargs["min_value"])
        elif kwargs.get("exclude_min", False):
            kwargs["min_value"] = int(kwargs["min_value"]) + 1

    if "max_value" in kwargs:
        if kwargs["max_value"] == math.inf:
            del kwargs["max_value"]
        elif math.isinf(kwargs["max_value"]):
            return ConstructivePredicate({"min_value": 1, "max_value": -1}, None)
        elif kwargs["max_value"] != int(kwargs["max_value"]):
            kwargs["max_value"] = floor(kwargs["max_value"])
        elif kwargs.get("exclude_max", False):
            kwargs["max_value"] = int(kwargs["max_value"]) - 1

    kw_categories = {"min_value", "max_value", "len"}
    kwargs = {k: v for k, v in kwargs.items() if k in kw_categories}
    return ConstructivePredicate(kwargs, predicate)


def get_float_predicate_bounds(predicate: Predicate) -> ConstructivePredicate:
    kwargs, predicate = get_numeric_predicate_bounds(predicate)  # type: ignore

    if "min_value" in kwargs:
        min_value = kwargs["min_value"]
        kwargs["min_value"] = float(kwargs["min_value"])
        if min_value < kwargs["min_value"] or (
            min_value == kwargs["min_value"] and kwargs.get("exclude_min", False)
        ):
            kwargs["min_value"] = next_up(kwargs["min_value"])

    if "max_value" in kwargs:
        max_value = kwargs["max_value"]
        kwargs["max_value"] = float(kwargs["max_value"])
        if max_value > kwargs["max_value"] or (
            max_value == kwargs["max_value"] and kwargs.get("exclude_max", False)
        ):
            kwargs["max_value"] = next_down(kwargs["max_value"])

    kwargs = {k: v for k, v in kwargs.items() if k in {"min_value", "max_value"}}
    return ConstructivePredicate(kwargs, predicate)


def max_len(size: int, element: Collection[object]) -> bool:
    return len(element) <= size


def min_len(size: int, element: Collection[object]) -> bool:
    return size <= len(element)
