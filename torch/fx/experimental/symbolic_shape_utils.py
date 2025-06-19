from __future__ import annotations

import functools
import inspect
import math
import os
import traceback
import types
from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Any, Optional, TypeGuard, Union, cast

import sympy
import torch
import torch.utils._pytree as pytree
from sympy import S
from torch import SymBool, SymFloat, SymInt
from torch._subclasses.meta_utils import is_sparse_any
from torch.fx.experimental.sym_node import SymNode, SymTypes
from torch.utils._python_dispatch import is_traceable_wrapper_subclass
from torch.utils._sympy.functions import CleanDiv
from typing_extensions import TypeGuard

if TYPE_CHECKING:
    from torch.types import BoolLikeType, IntLikeType


# Forward declarations for functions that will be available when this module is imported
# into symbolic_shapes.py. These will be monkey-patched at import time.
guard_bool = None  # type: ignore[assignment]
is_nested_int = None  # type: ignore[assignment]


def is_symbolic(val: object) -> bool:
    """Check if a value is symbolic."""
    return isinstance(val, (SymInt, SymFloat, SymBool)) and hasattr(val, 'node')


def canonicalize_bool_expr(expr: "sympy.logic.boolalg.Boolean") -> "sympy.logic.boolalg.Boolean":
    """Canonicalize a boolean expression."""
    return SymbolicShapeUtils.canonicalize_bool_expr_impl(expr)


# Type aliases
IterateExprs = Union[
    SymInt, SymFloat, SymBool, sympy.Basic, int, float, bool,
    tuple, list, torch.Tensor, None, torch.Generator
]

SympyBoolean = "sympy.logic.boolalg.Boolean"


class SymbolicContext:
    """Abstract base class for symbolic contexts."""
    pass


class InnerTensorKey:
    """Key for accessing inner tensor attributes."""
    def __init__(self, attr: str):
        self.attr = attr


class CallMethodKey:
    """Key for method calls."""
    def __init__(self, method: str):
        self.method = method


class DivideByKey:
    """Key for division operations."""
    def __init__(self, divisor: "IntLikeType"):
        self.divisor = divisor


class ConvertIntKey:
    """Key for integer conversion."""
    pass


class SymbolicShapeUtils:
    """Utility helper class containing non-public functions for symbolic shape operations."""

    @staticmethod
    def nested_int_aware_sort(
        tup: tuple["IntLikeType", int],
    ) -> tuple[int, "IntLikeType", int]:
        return (
            # Order nested ints by their coefficients.
            # 1 here to order nested ints after non-nested-ints.
            (1, tup[0].node.nested_int_coeff(), tup[1])
            if is_nested_int(tup[0])
            else (0, *tup)
        )

    @staticmethod
    def sympy_from_args(
        cls: type[Union[sympy.Add, sympy.Mul]],
        args: list[sympy.Expr],
        sort: bool = True,
        is_commutative: Optional[bool] = None,
    ) -> sympy.Expr:
        """
        Create a sympy expression from a list of arguments, optimizing for performance.

        This function creates a sympy Add or Mul expression from a list of arguments
        while avoiding expensive operations like flattening. It handles sorting the
        arguments appropriately based on the expression type.

        Args:
            cls: The sympy class to create (Add or Mul)
            args: List of sympy expressions to combine
            sort: Whether to sort the arguments (default: True)
            is_commutative: Whether the operation is commutative (default: None)

        Returns:
            A sympy expression of type cls combining all arguments

        Raises:
            ValueError: If cls is not sympy.Add or sympy.Mul
        """

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

    @staticmethod
    def canonicalize_bool_expr_impl(expr: SympyBoolean) -> SympyBoolean:
        """
        After canonicalization, we are guaranteed to have eliminated Ge/Gt relations
        (rewriting them to Le/Lt, respectively).
        """
        if isinstance(expr, (sympy.And, sympy.Or)):
            return type(expr)(*map(canonicalize_bool_expr, expr.args))

        opposite = {sympy.Gt: sympy.Lt, sympy.Ge: sympy.Le}
        t: Union[type[Any]]
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
        rhs = SymbolicShapeUtils.reduce_to_lowest_terms(rhs)
        if isinstance(rhs, sympy.Add):
            pos = []
            neg = []
            for term in rhs.args:
                if is_neg(term):
                    neg.append(-term)
                else:
                    pos.append(term)
            # these are already sorted
            rhs = SymbolicShapeUtils.sympy_from_args(sympy.Add, pos, sort=False, is_commutative=True)
            # the terms were changed, so needs a sorting
            lhs = SymbolicShapeUtils.sympy_from_args(sympy.Add, neg, sort=True, is_commutative=True)
        elif is_neg(rhs):
            # lhs == 0
            lhs, rhs = -rhs, S.Zero
        # We don't have to evaluate here because lhs, rhs came from a Boolean
        # and it was already simplified
        return t(lhs, rhs, evaluate=False)

    @staticmethod
    def reduce_to_lowest_terms(expr: sympy.Expr) -> sympy.Expr:
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
                return SymbolicShapeUtils.sympy_from_args(sympy.Mul, args, is_commutative=x.is_commutative)
            else:
                raise AssertionError(f"illegal arg to div_by_factor: {x}")

        if expr.is_Add:
            atoms = cast(Sequence[sympy.Expr], expr.args)
            factor = functools.reduce(math.gcd, map(integer_coefficient, atoms))
            if factor == 1:
                return expr
            atoms = [div_by_factor(x, factor) for x in atoms]
            return SymbolicShapeUtils.sympy_from_args(
                sympy.Add, atoms, sort=True, is_commutative=expr.is_commutative
            )
        elif expr.is_Integer:
            return S.One
        elif expr.is_Mul:
            return div_by_factor(expr, integer_coefficient(expr))
        return expr

    @staticmethod
    def iterate_exprs(val: IterateExprs) -> Iterator[sympy.Basic]:
        """
        Recursively iterate through a value and yield all sympy expressions contained within it.

        This function traverses various data structures (tensors, lists, tuples, etc.) and extracts
        any symbolic expressions they contain. It's used for operations like finding free symbols
        in complex nested structures.

        Args:
            val: The value to extract sympy expressions from. Can be a symbolic type (SymInt, SymFloat, SymBool),
                 a sympy expression, a primitive type (int, float, bool), a container (tuple, list),
                 a sparse tensor, a regular tensor, None, or a torch.Generator.

        Yields:
            sympy.Basic: Each sympy expression found in the value.

        Raises:
            AssertionError: If the value is of an unsupported type.
        """
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
                yield from SymbolicShapeUtils.iterate_exprs(s)
        elif is_sparse_any(val):
            yield from SymbolicShapeUtils.iterate_exprs(val.size())
        elif isinstance(val, torch.Tensor):
            yield from SymbolicShapeUtils.iterate_exprs(val.size())
            yield from SymbolicShapeUtils.iterate_exprs(val.stride())
            yield from SymbolicShapeUtils.iterate_exprs(val.storage_offset())
        elif val is None:
            pass
        # see Note: [Generator arguments in AOTDispatcher]
        elif isinstance(val, torch.Generator):
            pass
        else:
            raise AssertionError(f"cannot extract sympy expressions from {val} {type(val)}")

    @staticmethod
    def free_unbacked_symbols_with_path(
        a: object,
        path: pytree.KeyPath,
        real: Optional[object] = None,
        shape_env: Optional["ShapeEnv"] = None,
        pending: Optional[set[sympy.Symbol]] = None,
        simplify: bool = False,
    ) -> dict[sympy.Symbol, pytree.KeyPath]:
        """
        Recursively traverses a structure to find unbacked symbols and their access paths.

        This function walks through tensors, lists, tuples, and symbolic values to locate
        unbacked symbols that are in the pending set, and returns a mapping from those
        symbols to their access paths in the structure.

        Args:
            a: The object to traverse (tensor, list, tuple, SymInt, etc.)
            path: The current path in the object tree
            real: Optional real tensor corresponding to the fake tensor being traversed
            shape_env: Optional ShapeEnv to register unbacked values with
            pending: Set of unbacked symbols to look for (will be modified in-place)
            simplify: Whether to use simplified expressions

        Returns:
            A dictionary mapping unbacked symbols to their access paths
        """
        go = functools.partial(
            SymbolicShapeUtils.free_unbacked_symbols_with_path,
            shape_env=shape_env,
            pending=pending,
            simplify=simplify,
        )

        def expr(s: Union[SymInt, SymFloat, SymBool]) -> sympy.Expr:
            if simplify:
                return s.node.expr
            # (When called from compute_unbacked_bindings)
            # NB: Intentionally access _expr, not expr, do not want
            # simplification!
            return s.node._expr

        if pending is None:
            pending = set()
        r = {}
        if isinstance(a, (tuple, list)):
            # NB: real is apparently not always a tuple/list here
            # python test/inductor/test_torchinductor.py CpuTests.test_index_propagation_nested_indirect_indexing_cpu
            for i in range(len(a)):
                r.update(
                    go(
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
                r.update(go(sub, path + (InnerTensorKey(attr),)))
        elif isinstance(a, torch.Tensor):
            from torch._subclasses.fake_tensor import FakeTensor

            assert isinstance(a, FakeTensor)
            r.update(
                go(
                    a.size(),
                    path + (CallMethodKey("size"),),
                    real=a.real_tensor.size() if a.real_tensor is not None else None,
                )
            )
            if a.layout not in [
                torch.sparse_csr,
                torch.sparse_csc,
                torch.sparse_bsr,
                torch.sparse_bsc,
            ]:
                r.update(
                    go(
                        a.stride(),
                        path + (CallMethodKey("stride"),),
                        real=a.real_tensor.stride() if a.real_tensor is not None else None,
                    )
                )
            r.update(
                go(
                    a.storage_offset(),
                    path + (CallMethodKey("storage_offset"),),
                    real=(
                        a.real_tensor.storage_offset()
                        if a.real_tensor is not None
                        else None
                    ),
                )
            )

        elif (
            isinstance(a, (torch.SymInt, torch.SymFloat))
            and isinstance(s := expr(a), sympy.Symbol)
            and s in pending
        ):
            r[s] = path
            if shape_env and real is not None:
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
            and isinstance(s := expr(a), sympy.Mul)
            and len(s.args) == 2
            and isinstance(lhs := s.args[0], (sympy.Integer, sympy.Symbol))
            and isinstance(rhs := s.args[1], sympy.Symbol)
            # support exactly one unbacked for now
            and ((rhs in pending) ^ (lhs in pending))
            # support constant coefficient or backed symbolic coefficient
            and (
                isinstance(coeff := lhs if lhs not in pending else rhs, sympy.Integer)
                or shape_env
                and coeff in shape_env.var_to_val
            )
        ):

            def _symint_wrap(s: sympy.Symbol) -> SymInt:
                return shape_env.create_symintnode(  # type: ignore[union-attr]
                    s,
                    hint=int(shape_env.var_to_val[s]),  # type: ignore[union-attr]
                    source=shape_env.var_to_sources.get(s, [None])[0],  # type: ignore[union-attr]
                )

            unbacked = lhs if lhs in pending else rhs
            divisor: "IntLikeType" = (
                int(coeff)
                if shape_env and isinstance(coeff, sympy.Integer)
                else _symint_wrap(coeff)
            )
            # TODO: DivideByKey needs to test divisibility at runtime!
            r[unbacked] = path + (DivideByKey(divisor),)
            if real is not None:
                assert isinstance(real, int)
                val = (
                    real // int(coeff)
                    if isinstance(coeff, sympy.Integer)
                    else CleanDiv(real, coeff)
                )
                if shape_env:
                    shape_env.set_unbacked_var_to_val(unbacked, val)
            pending.remove(unbacked)
        # The annoyance here arises from the fact that SymBool is
        # allocated by allocating a SymInt and then testing if it's equal
        # to one.  So you have a complicated binding site logic for this.
        elif (
            isinstance(a, torch.SymBool)
            and isinstance(s := expr(a), sympy.Eq)
            # This must match create_unbacked_symbool EXACTLY
            and isinstance(s.lhs, sympy.Symbol)
            and s.rhs == 1
            and s.lhs in pending
        ):
            r[s.lhs] = path + (ConvertIntKey(),)
            if real is not None:
                assert type(real) is bool
                if shape_env:
                    shape_env.set_unbacked_var_to_val(s, int(real))
            pending.remove(s.lhs)

        return r

    @staticmethod
    def static_eval_sym_bool(x: SymBool) -> Optional[bool]:
        assert isinstance(x, SymBool)
        expr = x.node.expr

        try:
            # Shape env access is inside the try on purpose. xla symnode does not
            # have it on its attributes.
            shape_env = x.node.shape_env
            simplified = shape_env._maybe_evaluate_static(expr)
            if simplified is not None:
                return bool(simplified)
            else:
                return None
        except Exception:
            # log.debug("Could not simplify %s", expr)
            return None

    @staticmethod
    def guard_or(a: "BoolLikeType", default: bool) -> bool:
        """
        Try to guard a, if data dependent error encountered just return default.
        """
        if not isinstance(a, SymBool):
            assert isinstance(a, bool)
            return a

        # if backed_size_oblivious is True we treat backed as unbacked here.
        if torch.fx.experimental._config.backed_size_oblivious:
            result = SymbolicShapeUtils.static_eval_sym_bool(a)
            return result if result is not None else default

        shape_env = getattr(a.node, "shape_env", None)

        # xla symnode path.
        if shape_env is None:
            return guard_bool(a)

        sym_node = a.node
        r = sym_node.shape_env.evaluate_sym_node(
            sym_node, size_oblivious=False, fallback_value=default
        )
        return bool(r)

    @staticmethod
    def is_int(expr: object) -> bool:
        return isinstance(expr, SymInt) and expr.node.expr.is_number

    @staticmethod
    def is_dim_dynamic(t: torch.Tensor, d: int) -> bool:
        # WARNING: This is legacy, DO NOT USE
        return hasattr(t, "_dynamo_dynamic_indices") and d in t._dynamo_dynamic_indices

    @staticmethod
    def find_user_code_frame() -> Optional[types.FrameType]:
        frame = inspect.currentframe()
        while frame is not None:
            if not frame.f_code.co_filename.startswith(
                os.path.dirname(inspect.getfile(torch)) + os.path.sep
            ):
                break
            frame = frame.f_back
        return frame

    @staticmethod
    def blame_user_code(e: Exception, frame: types.FrameType) -> None:
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

    @staticmethod
    def assert_symbol_context(symbolic_context: object) -> TypeGuard[SymbolicContext]:
        assert isinstance(
            symbolic_context, SymbolicContext
        ), "Invalid symbolic_context object"
        assert (
            type(symbolic_context) is not SymbolicContext
        ), "Illegal usage of symbolic_context ABC"
        return True

    @staticmethod
    def is_non_negative_check(cond: sympy.Basic) -> Optional[str]:
        """
        Check if a condition (SymPy expression) is checking for non-negative values (>= 0).
        Returns the variable name if it's a non-negative check (>= 0), None otherwise.
        """
        if isinstance(cond, sympy.Rel):
            if cond.rel_op == ">=" and cond.rhs == 0:
                return str(cond.lhs)
        return None

    @staticmethod
    def remove_effect_token_unbacked_bindings(
        node: torch.fx.Node,
    ) -> "collections.abc.Generator[None, None, None]":
        """
        Temporarily modifies unbacked_bindings in a node's metadata by removing the first element
        of each path, which corresponds to an effect token.

        This is used when processing nodes that have effect tokens as the first element in their
        unbacked_bindings paths. The context manager ensures that the original bindings are
        restored after the operation is complete.

        Args:
            node: The FX node whose unbacked_bindings will be temporarily modified

        Yields:
            None
        """
        old_bindings = node.meta.get("unbacked_bindings", {})

        # Remove the extra layer for effect token
        new_bindings = {k: path[1:] if path else path for k, path in old_bindings.items()}

        node.meta["unbacked_bindings"] = new_bindings

        try:
            yield
        finally:
            node.meta["unbacked_bindings"] = old_bindings

    @staticmethod
    def get_placeholder_expr(sym_node: SymNode) -> sympy.Expr:
        # This helper function is used in passes that insert runtime assertions in the graph.
        # When accessing expressions representing input placeholders, we do not apply replacements
        # since those inputs should be seen by assertions that use them to be inserted. The only replacement
        # that we apply is unbacked renaming.
        shape_env = sym_node.shape_env
        result = sym_node._expr
        if result in shape_env.unbacked_renamings:
            return shape_env.unbacked_renamings[result]
        return result

    @staticmethod
    def is_supported_equivalence(expr: sympy.Expr) -> bool:
        # Currently supported Dim ops are linear expressions with integer coefficients.
        # So check that expr only contains +, *, ints, and a single occurrence of a symbol.
        # (See also documentation of dynamic_shapes._DerivedDim.)
        if isinstance(expr, (sympy.Add, sympy.Mul)):
            if len(expr.args) > 2:
                return False
            lhs, rhs = expr.args
            return (SymbolicShapeUtils.is_supported_equivalence(lhs) and isinstance(rhs, sympy.Integer)) or (
                isinstance(lhs, sympy.Integer) and SymbolicShapeUtils.is_supported_equivalence(rhs)
            )
        return isinstance(expr, sympy.Symbol)

    @staticmethod
    def has_uninterpretable_sympy_function(expr: sympy.Basic) -> bool:
        """
        Add functions that our sympy interpreter can't reify into FX nodes
        """
        return expr.has(
            torch.utils._sympy.functions.ToFloat,
            torch.utils._sympy.functions.TruncToInt,
            torch.utils._sympy.functions.CeilToInt,
        )