# mypy: allow-untyped-defs
import builtins
import functools
import logging
import math
import operator
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Optional, Union

import sympy

import torch
import torch.fx
import torch.fx.traceback as fx_traceback
from torch._dynamo.exc import TorchDynamoException
from torch._dynamo.utils import dynamo_timed
from torch.fx.node import Argument, Target
from torch.utils._sympy.interp import sympy_interp


log = logging.getLogger(__name__)

try:
    import z3  # type: ignore[import]

    # Translation Validation for Dynamo guards
    # ========================================
    #
    # Checks whether optimizations applied to the collected guards are
    # valid. In other words, whether the guard function we actually run
    # does not have false positives (unsound).
    #
    # In order to do so, we build the guards using 2 different information
    # attached to each 'SymNode':
    #   1. SymPy expressions
    #   2. FX nodes
    #
    # SymPy expressions have implicit optimizations baked within itself,
    # which may have a few bugs. On the other hand, we build the FX graph
    # manually, with no optimizations enabled. This gives us access to
    # the "ground truth".
    #
    # We then convert into Z3 expressions both the SymPy expressions
    # (see [Note: SympyToZ3]) that reach 'ShapeEnv.produce_guards' function
    # and the FX nodes (see [Note: PopulateValidator]) that go through
    # 'ShapeEnv.evaluate_expr' function. Finally, we run the validation.
    # (see [Note: TranslationValidator])
    # Better Z3 to string implementation (for a small fraction of Z3).
    #
    # Here are the things we clean before showing the Z3 expression:
    #   - Rename a few ops (e.g. "Distinct" ==> "!=")
    #
    #   - Ignore ToInt and ToReal operations:
    #     usually they don't really matter
    #
    #   - Transform (ToInt (/ ...)) into (idiv ...):
    #     this is the pattern for floor division
    #
    #   - Collect a chain of the same operations into one
    def z3str(e: z3.ExprRef) -> str:
        assert z3.is_expr(e), f"unsupported expression type: {e}"

        def get_args_str(e: z3.ExprRef) -> list[str]:
            return [z3str(e.arg(i)) for i in range(e.num_args())]

        # First, we simplify the given expression.
        # This is done using rewriting rules, so shouldn't take long.
        e = z3.simplify(e)

        # Only support function applications.
        # Even Z3 "variables" are, in fact, function applications.
        if not z3.is_app(e):
            raise ValueError(f"can't print Z3 expression: {e}")

        if z3.is_int_value(e) or z3.is_rational_value(e):
            return e.as_string()  # type: ignore[attr-defined]

        decl = e.decl()
        kind = decl.kind()
        op = str(decl)
        args = get_args_str(e)

        if kind == z3.Z3_OP_POWER:
            op = "pow"

        elif kind in (z3.Z3_OP_ADD, z3.Z3_OP_MUL):
            # Collect the arguments of chains of ADD and MUL.
            # This is safe, since they are associative.

            def collect_str_args(e):
                if not (z3.is_app(e) and e.decl().kind() == kind):
                    return [z3str(e)]
                else:
                    return [
                        x
                        for i in range(e.num_args())
                        for x in collect_str_args(e.arg(i))
                    ]

            args = collect_str_args(e)

        elif kind == z3.Z3_OP_NOT:
            # Revert some conversions that z3.simplify applies:
            #   - a != b ==> (Not (== a b)) ==> (!= a b)
            #   - a < b ==> (Not (<= b a)) ==> (> b a)
            #   - a > b ==> (Not (<= a b)) ==> (> a b)

            assert e.num_args() == 1
            arg = e.arg(0)

            assert z3.is_app(arg)
            argkind = arg.decl().kind()

            logic_inverse = {
                z3.Z3_OP_EQ: "!=",
                z3.Z3_OP_LE: ">",
                z3.Z3_OP_GE: "<",
            }

            if argkind in logic_inverse:
                op = logic_inverse[argkind]
                args = get_args_str(arg)

        elif kind in (z3.Z3_OP_TO_INT, z3.Z3_OP_TO_REAL):
            assert e.num_args() == 1
            argstr = z3str(e.arg(0))

            # Check if it's the floor division pattern.
            if argstr.startswith("(/"):
                return "(idiv" + argstr[2:]

            # Otherwise, just ignore it.
            return argstr

        elif kind == z3.Z3_OP_UNINTERPRETED:
            assert e.num_args() == 0
            return str(decl)

        string = op + " " + " ".join(args)
        return f"({string.rstrip()})"

    # We need to convert to/from BitVec in order to use z3 bitwise ops.
    # We assume that integers are 64 bit.
    # If all args are boolean, then use the boolean bitwise op implementation instead, if provided.
    def _bitwise_op(bitwise_func, bool_func):
        @functools.wraps(bitwise_func)
        def wrapper(self, *args):
            if bool_func is not None and all(
                isinstance(arg, z3.BoolRef) for arg in args
            ):
                return bool_func(*args)

            wrapped_args = tuple(z3.Int2BV(a, 64) for a in args)
            return z3.BV2Int(bitwise_func(*wrapped_args))

        return wrapper

    # Implementation of Python semantics as Z3 expressions.
    #
    # Z3 Real-Int theory has operators with semantics that differ that of
    # Python. Therefore, in order to get it right, we need to implement
    # the (Python) semantics we are relying on in Z3.
    @dataclass
    class _Z3Ops:
        # Validator used for adding assertions as needed.
        # e.g. div(a, b) requires b != 0.
        validator: "TranslationValidator"

        # The 2 functions below are used for conditionally casting between
        # integer and reals.
        #
        # Returns a real expression from 'x'.
        @staticmethod
        def to_real(x: z3.ArithRef) -> z3.ArithRef:
            return x if x.is_real() else z3.ToReal(x)

        # Returns an integer expression from 'x'.
        @staticmethod
        def to_int(x: z3.ArithRef) -> z3.ArithRef:
            return x if x.is_int() else z3.ToInt(x)

        def sym_sum(self, args: z3.ArithRef) -> z3.ArithRef:
            # pyrefly: ignore
            return sum(args)

        # Implements Python division semantics.
        def div(self, numerator: z3.ArithRef, denominator: z3.ArithRef) -> z3.ArithRef:
            self.validator.add_assertion(denominator != 0)  # type: ignore[arg-type]
            return _Z3Ops.to_real(numerator) / _Z3Ops.to_real(denominator)

        def floor(self, number: z3.ArithRef) -> z3.ArithRef:
            # Z3 ToInt function rounds a real number towards negative infinity.
            return _Z3Ops.to_int(number)

        # Python semantics for 'FloorDiv' states that before applying the floor
        # function, the operands are converted to their common type.
        def floordiv(
            self, numerator: z3.ArithRef, denominator: z3.ArithRef
        ) -> z3.ArithRef:
            cast_result_to_real = numerator.is_real() or denominator.is_real()
            result = _Z3Ops.to_int(self.div(numerator, denominator))
            # Since the 'result' is already an integer, we just have to check
            # whether we should cast it to real.
            return _Z3Ops.to_real(result) if cast_result_to_real else result

        def ceil(self, number: z3.ArithRef) -> z3.ArithRef:
            return z3.If(self.floor(number) < number, self.floor(number + 1), number)  # type: ignore[return-value]

        def trunc(self, number: z3.ArithRef) -> z3.ArithRef:
            return z3.If(number >= 0, self.floor(number), self.ceil(number))  # type: ignore[return-value]

        def max(self, a: z3.ArithRef, b: z3.ArithRef) -> z3.ArithRef:
            return z3.If(a > b, a, b)  # type: ignore[return-value]

        def min(self, a: z3.ArithRef, b: z3.ArithRef) -> z3.ArithRef:
            return z3.If(a < b, a, b)  # type: ignore[return-value]

        # Python semantics for 'Mod' is defined as: p % q = p - floordiv(p, q) * q
        # It should work with both integer and reals.
        def mod(self, p: z3.ArithRef, q: z3.ArithRef) -> z3.ArithRef:
            return p - self.floordiv(p, q) * q

        def pow(self, base: z3.ArithRef, exp: z3.ArithRef) -> z3.ArithRef:
            # Z3 can't handle complex numbers very well.
            self.validator.add_assertion(z3.Or(base != 0, exp > 0))  # type: ignore[arg-type]
            return base**exp

        def sqrt(self, number: z3.ArithRef) -> z3.ArithRef:
            # Square-root:
            # 1. Only work with reals
            number = _Z3Ops.to_real(number)
            # 2. The number should be positive or zero.
            #    Otherwise, Z3 returns 'unknown'.
            self.validator.add_assertion(number >= 0)
            return number**0.5

        def abs(self, number: z3.ArithRef) -> z3.ArithRef:
            return z3.Abs(number)

        def round_to_int(self, number: z3.ArithRef) -> z3.ArithRef:
            # Pythons builtin 'round' implements the 'round half to even' strategy
            # See https://en.wikipedia.org/wiki/Rounding#Rounding_half_to_even
            # z3 has an equivalent z3.fpRoundToIntegral(z3.RoundNearestTiesToEven(), ...), but this only applies to
            # floating point numbers, which is different from real numbers that we are dealing with here.
            # Instead, we implement 'round half to even' in terms of 'round half up' (floor(x + 0.5)) and
            # 'round half down' (ceil(x - 0.5)).
            # Assuming 'round half up' is the default case, we need to correct ..., -3.5, -1.5, 0.5, 2.5, 4.5, ...
            # to round down, i.e. use the 'round half down' strategy
            return z3.If(
                self.mod(number, z3.IntVal(2)) == 0.5,
                self.ceil(number - 0.5),
                self.floor(number + 0.5),
            )

        bitwise_and = _bitwise_op(operator.and_, z3.And)
        bitwise_or = _bitwise_op(operator.or_, z3.Or)
        lshift = _bitwise_op(operator.lshift, None)
        rshift = _bitwise_op(operator.rshift, None)

    # Lifts a callable to be used in Z3.
    #
    # This function replaces the given 'op' by a function that:
    #
    #   1. Lifts the arguments into Z3 (i.e. make them inhabitants of Z3)
    #
    #   2. Calls an operation that corresponds to 'op', but works with Z3
    #      inhabitants (left as is if it works as is)
    def z3op(op: Callable, validator: "TranslationValidator") -> Callable:
        # Operations that have booleans as their argument.
        # This is needed because the argument of some FX nodes were
        # literal integers, instead of booleans. So, whenever this flag
        # is set, we also convert ints to booleans.
        boolean_ops = {operator.not_}
        as_bool = op in boolean_ops

        # Lifts the function into 'z3.ExprRef' domain.
        def lift(func):
            def wrap(a) -> z3.ExprRef:
                if isinstance(a, (z3.ArithRef, z3.BoolRef)):
                    return a
                # Convert it into a Z3 value, if it is some of the supported
                # types below.
                if isinstance(a, bool) or (as_bool and isinstance(a, int)):
                    return z3.BoolVal(bool(a))
                if isinstance(a, (int, sympy.Integer)):
                    return z3.IntVal(int(a))
                if isinstance(a, (float, sympy.Float)):
                    return z3.RealVal(float(a))
                raise ValueError(f"can't lift type: {type(a)}")

            @functools.wraps(func)
            def wrapper(*args):
                # Lifts the arguments into a list of Z3 inhabitants.
                if len(args) == 1 and isinstance(args[0], (list, tuple)):
                    wrapped_args = (tuple(wrap(a) for a in args[0]),)
                else:
                    wrapped_args = tuple(wrap(a) for a in args)
                # Run the function on the Z3 expressions.
                return func(*wrapped_args)

            return wrapper

        ops = _Z3Ops(validator)
        replacement_map = {
            # Operator module.
            operator.not_: lift(z3.Not),
            operator.and_: lift(ops.bitwise_and),
            operator.or_: lift(ops.bitwise_or),
            operator.lshift: lift(ops.lshift),
            operator.rshift: lift(ops.rshift),
            operator.floordiv: lift(ops.floordiv),
            operator.truediv: lift(ops.div),
            operator.mod: lift(ops.mod),
            operator.abs: lift(ops.abs),
            builtins.round: lift(ops.round_to_int),
            # Math module.
            math.ceil: lift(ops.ceil),
            math.floor: lift(ops.floor),
            math.trunc: lift(ops.trunc),
            # Torch module.
            torch.sym_float: lift(ops.to_real),
            torch.sym_max: lift(ops.max),
            torch.sym_min: lift(ops.min),
            torch.sym_sum: lift(ops.sym_sum),
            torch.sym_ite: lift(lambda b, t, f: t if b else f),
            torch._sym_sqrt: lift(ops.sqrt),  # type: ignore[attr-defined]
            # Not lifted because we only use this function as a
            # marker for adding the expression as validator input.
            torch._assert: torch._assert,
        }
        return replacement_map[op] if op in replacement_map else lift(op)

    # Processes an FX graph, populating the given validator.
    #
    # [Note: PopulateValidator]
    # This class walks through each node in the FX graph, translating
    # them into the Z3 world.
    #
    # Then, whenever it finds an 'torch._assert' call_function operation,
    # it adds the Z3 expression corresponding to the argument as validator
    # input.
    class PopulateValidator(torch.fx.Interpreter):
        def __init__(self, graph: torch.fx.Graph, validator: "TranslationValidator"):
            # Reference to the translation validator.
            self.validator = validator

            # Build the graph module and call `Interpreter` constructor.
            module = torch.fx.GraphModule(root={}, graph=graph)
            super().__init__(module, garbage_collect_values=True)

        def placeholder(
            self, target: Target, args: tuple[Argument, ...], kwargs: dict[str, Any]
        ) -> Any:
            symbol = fx_traceback.get_current_meta()["symbol"]
            return self.validator.z3var(symbol)

        def call_function(
            self, target: Target, args: tuple[Argument, ...], kwargs: dict[str, Any]
        ) -> Any:
            if target is not torch._assert:
                # Lift and runs the node target function
                return super().call_function(z3op(target, self.validator), args, kwargs)  # type: ignore[arg-type]
            # Adds the Z3 expression corresponding to the first argument
            # as a validator input.
            assert len(args) == 1, (
                f"expected 1 argument on assertion. Got: {len(args)} "
            )
            self.validator.add_source_expr(args[0])  # type: ignore[arg-type]

    # Translates SymPy expressions into Z3 expressions.
    #
    # [Note: SympyToZ3]
    # At the time of the translation, all free variables present in the
    # SymPy expression being translated must be already mapped to a Z3
    # integer variable.
    class SympyToZ3:
        OPERATOR_HANDLES = {"add", "mul", "eq", "ne", "lt", "gt", "le", "ge"}

        def __init__(
            self,
            validator: "TranslationValidator",
        ) -> None:
            self._validator = validator
            self._ops = _Z3Ops(self._validator)

        def constant(self, value: Any, dtype: torch.dtype) -> z3.ExprRef:
            # TODO: Probably OK to relax this and allow lower precision
            if dtype is torch.int64:
                return z3.IntVal(int(value))
            if dtype is torch.double:
                return z3.RealVal(float(value))
            if dtype is torch.bool:
                return z3.BoolVal(bool(value))
            raise ValueError(f"unsupported dtype (SympyToZ3): {dtype}")

        def to_dtype(self, x: z3.ArithRef, dtype: torch.dtype) -> z3.ArithRef:
            if dtype == torch.float64:
                return z3.ToReal(x)
            raise NotImplementedError(f"to_dtype {dtype} NYI")

        def trunc_to_int(self, x: z3.ArithRef, dtype: torch.dtype) -> z3.ArithRef:
            return z3.ToInt(x)

        def round_to_int(self, x: z3.ArithRef, dtype: torch.dtype) -> z3.ArithRef:
            return self._ops.round_to_int(x)

        def int_truediv(
            self, numerator: z3.ArithRef, denominator: z3.ArithRef
        ) -> z3.ArithRef:
            return self._ops.div(numerator, denominator)

        def truediv(
            self, numerator: z3.ArithRef, denominator: z3.ArithRef
        ) -> z3.ArithRef:
            return self._ops.div(numerator, denominator)

        def floordiv(
            self, numerator: z3.ArithRef, denominator: z3.ArithRef
        ) -> z3.ArithRef:
            return self._ops.floordiv(numerator, denominator)

        def div(self, numerator: z3.ArithRef, denominator: z3.ArithRef) -> z3.ArithRef:
            return self._ops.floordiv(numerator, denominator)

        def pow(self, base: z3.ArithRef, exp: z3.ArithRef) -> z3.ArithRef:
            return self._ops.pow(base, exp)

        def pow_by_natural(self, base: z3.ArithRef, exp: z3.ArithRef) -> z3.ArithRef:
            return self._ops.pow(base, exp)

        def mod(self, p: z3.ArithRef, q: z3.ArithRef) -> z3.ArithRef:
            return self._ops.mod(p, q)

        def ceil_to_int(self, x: z3.ArithRef, dtype: torch.dtype) -> z3.ArithRef:
            return self._ops.ceil(x)

        def floor_to_int(self, x: z3.ArithRef, dtype: torch.dtype) -> z3.ArithRef:
            return self._ops.floor(x)

        def __getattr__(self, name: str) -> Any:
            REPLACEMENT = {
                "and_": z3.And,
                "or_": z3.Or,
                "not_": z3.Not,
                "bitwise_and": self._ops.bitwise_and,
                "bitwise_or": self._ops.bitwise_or,
                "lshift": self._ops.lshift,
                "rshift": self._ops.rshift,
                "floor": self._ops.floor,
                "ceil": self._ops.ceil,
                "minimum": self._ops.min,
                "maximum": self._ops.max,
            }

            if name in REPLACEMENT:
                return REPLACEMENT[name]
            if name in self.OPERATOR_HANDLES:
                return getattr(operator, name)
            raise AttributeError(f"unhandled operator: {name}")

        def run(self, expr: sympy.Basic) -> z3.ExprRef:
            return sympy_interp(self, self._validator.symbols, expr)  # type: ignore[arg-type]

    # Dynamo guards translation validator.
    #
    # [Note: TranslationValidator]
    # Verifies whether the guards issued by 'ShapeEnv.produce_guards' are sound.
    # That is: whether those (target) guards only yield TRUE whenever the original,
    # unoptimized, (source) guards yield TRUE.
    #
    # More concretely, given 'source' and 'target' guard expressions, we wish to
    # check whether the following expression holds:
    #
    # Not(And(source)) AND And(target)
    #
    # i.e. whether there is an assignment of the free variables where the opposite
    # happens: target is TRUE, but source is FALSE.
    class TranslationValidator:
        def __init__(self) -> None:
            log.debug("new instance")

            # Mapping of SymPy symbols to Z3 variables.
            self.symbols: dict[sympy.Symbol, z3.ExprRef] = {}

            # Set of source Z3 expressions.
            # They represent the generated guards without any kind of
            # simplification or transformation.
            self._source_exprs: set[z3.BoolRef] = set()

            # Set of target Z3 expressions.
            # They represent the actual checked guards at runtime. They might
            # be simplified or transformed versions of the source guards.
            self._target_exprs: set[z3.BoolRef] = set()

            # Set of Z3 expressions representing assertions over both the
            # source and target expressions.
            self._assertions: set[z3.BoolRef] = set()

        # Retrieves the corresponding Z3 variable.
        def z3var(self, symbol: sympy.Symbol) -> z3.ExprRef:
            assert symbol in self.symbols, f"Z3 variable not found for: {symbol}"
            return self.symbols[symbol]

        # Create a variable in Z3 of 'type' for 'symbol', if it doesn't already exists.
        def add_var(self, symbol: sympy.Symbol, type: type) -> z3.ExprRef:
            if symbol in self.symbols:
                return self.symbols[symbol]

            log.debug("new variable: %s (%s)", symbol.name, type.__name__)

            if type is int:
                var = z3.Int(symbol.name)

                # If 'symbol' is positive (SymPy assumption), we have to
                # convey it to Z3 as well.
                if symbol.is_positive:  # type: ignore[attr-defined]
                    self._target_exprs.add(var > 0)
            elif type is float:
                var = z3.Real(symbol.name)
            elif type is bool:
                var = z3.Bool(symbol.name)
            else:
                raise RuntimeError(f"unsupported type for Z3 variable: {type}")

            self.symbols[symbol] = var
            return var

        # Checks whether all symbols were already added.
        def _check_freesymbols(self, e: sympy.Basic) -> None:
            for s in e.free_symbols:
                assert isinstance(s, sympy.Symbol)
                # Call 'z3var' just to check whether there's already a
                # Z3 variable corresponding to 's'.
                self.z3var(s)

        def to_z3_boolean_expr(self, e: sympy.Basic) -> z3.BoolRef:
            z3expr = SympyToZ3(self).run(e)
            assert isinstance(z3expr, z3.BoolRef), (
                f"expected boolean expression. Got: {z3expr}"
            )
            return z3expr

        def add_source_expr(self, e: z3.BoolRef) -> None:
            if e not in self._source_exprs:
                log.debug("add source guard: %s", z3str(e))
            self._source_exprs.add(e)

        def add_target_expr(self, e: "sympy.logic.boolalg.Boolean") -> None:
            self._check_freesymbols(e)
            z3expr = self.to_z3_boolean_expr(e)
            if e not in self._target_exprs:
                log.debug("add target guard: %s", z3str(z3expr))
            self._target_exprs.add(z3expr)

        def add_assertion(self, e: Union[z3.BoolRef, sympy.Basic]) -> None:
            if isinstance(e, sympy.Basic):
                self._check_freesymbols(e)
                ref = self.to_z3_boolean_expr(e)
            else:
                ref = e
            assert isinstance(ref, z3.BoolRef)
            if ref not in self._assertions:
                log.debug("add assertion: %s", z3str(ref))
            self._assertions.add(ref)

        def validate(self) -> None:
            with dynamo_timed("TranslationValidator.validate"):
                return self._validate()

        def _validate(self) -> None:
            if len(self._source_exprs) == 0 or len(self._target_exprs) == 0:
                # If there are no source/target expressions, there's nothing we really
                # wish to prove. So, we just return.
                return None

            # Here, we use "QF_NRA" logic for the solver:
            #   "Quantifier-free Non-linear Real Arithmetic".
            #
            # Most of the guards expressions have:
            #   1. arithmetic between integer and reals
            #   2. no quantifiers
            #   3. potentially non-linear.
            #
            # Although there's also "QF_NIRA" (mixed integer-real arithmetic),
            # "QF_NRA" seems to work better on 'dynamo/test_dynamic_shapes.py'.
            solver = z3.SolverFor("QF_NRA")
            # Set a timeout for finding a solution.
            solver.set(timeout=translation_validation_timeout())

            # Add all the assertions to the solver.
            for assertion in self._assertions:
                solver.add(assertion)

            # "Is there any case where it's TRUE for the target expressions,
            #  but FALSE for the source expressions?"
            solver.add(z3.Not(z3.And(*self._source_exprs)))
            solver.add(*self._target_exprs)

            log.debug("translation validation: start")
            r = solver.check()
            if r == z3.sat:
                # Target expressions are unsound.
                # Log the found model and the source expressions that failed.
                model = solver.model()
                raise ValidationException(
                    model,
                    self._assertions,
                    self._target_exprs,
                    failed_source_exprs=[
                        inp for inp in self._source_exprs if not model.evaluate(inp)
                    ],
                )
            else:
                if r == z3.unknown:
                    # Could not find a solution. It didn't fail, but it also
                    # didn't succeed. Canceling the validation execution (keyboard
                    # interrupt) also gets to this branch.
                    log.warning(
                        "translation validation: could not validate: got z3.unknown"
                    )
                else:
                    # Target expressions are sound.
                    assert r == z3.unsat
                    log.debug("translation validation: success")

except ImportError:
    _HAS_Z3 = False

    __all__ = [
        "translation_validation_enabled",
        "translation_validation_timeout",
        "ValidationException",
        "BisectValidationException",
    ]

else:
    _HAS_Z3 = True

    __all__ = [
        "z3str",
        "z3op",
        "PopulateValidator",
        "SympyToZ3",
        "TranslationValidator",
        "translation_validation_enabled",
        "translation_validation_timeout",
        "ValidationException",
        "BisectValidationException",
    ]

from torch.fx.experimental import _config as config


def translation_validation_enabled() -> bool:
    # Checks every time this function is called, in case the Dynamo
    # option is set, but Z3 is not installed.
    _assert_z3_installed_if_tv_set()
    return _HAS_Z3 and config.translation_validation


def translation_validation_timeout() -> int:
    return config.translation_validation_timeout


def _assert_z3_installed_if_tv_set():
    assert _HAS_Z3 or not config.translation_validation, (
        "translation validation requires Z3 package. Please, either install "
        "z3-solver or disable translation validation."
    )


class ValidationException(TorchDynamoException):
    def __init__(self, model, assertions, target_exprs, failed_source_exprs):
        assert _HAS_Z3

        def symbolstr(sym) -> str:
            return f"{sym}: {model[sym]}"

        def joinlines(xs) -> str:
            return "\n".join(f"  ==> {x}" for x in xs)

        model_str = joinlines(sorted(map(symbolstr, model)))
        assertions_str = joinlines(sorted(map(z3str, assertions)))
        target_exprs_str = joinlines(sorted(map(z3str, target_exprs)))
        failed_source_exprs_str = joinlines(sorted(map(z3str, failed_source_exprs)))

        self.msg = "translation validation failed."
        self.details = f"""\
Model:
{model_str}

Assertions:
{assertions_str}

Target Expressions:
{target_exprs_str}

Failed Source Expressions:
{failed_source_exprs_str}"""

    def __str__(self):
        return f"{self.msg}\n\n{self.details}"


class BisectValidationException(TorchDynamoException):
    def __init__(self, validation_exc, expr, failed_action, traced_node):
        self.msg = f"translation validation failed when {failed_action}: {expr}"
        self.details = f"""\
Failure occurred while running node:
    {traced_node.format_node()}

{validation_exc.details}"""

    def __str__(self):
        return f"{self.msg}\n\n{self.details}"


# Checks when this module is loaded.
_assert_z3_installed_if_tv_set()


# Translation validation bisection.
#
# Bisect into the torch._assert nodes recorded in the shape_env FX graph, and raise
# the earliest ValidationException.
#
# As guards are added by ShapeEnv.evaluate_expr calls, some simplification errors
# might be silently happening. This function tries to nail down exactly at which
# point things went wrong from a validation perspective.
def bisect(shape_env):
    from torch.fx.experimental.recording import (
        FakeTensorMeta,
        replay_shape_env_events,
        ShapeEnvEvent,
    )
    from torch.fx.experimental.symbolic_shapes import (
        CURRENT_NODE_KEY,
        ShapeEnv,
        SHAPEENV_EVENT_KEY,
    )

    events = shape_env.events

    # Retrieves the ShapeEnvEvent associated with node.
    def get_node_event(node: torch.fx.Node) -> ShapeEnvEvent:
        assert SHAPEENV_EVENT_KEY in node.meta
        return events[node.meta[SHAPEENV_EVENT_KEY]]

    # Creates a new instance of fake, but updating every symbolic value's ShapeEnv
    # reference to the one given as argument.
    #
    # This is needed so as not to simplify a symbolic expression using a ShapeEnv
    # "from the future", where it may have a different set of replacements.
    def new_with_shape_env(shape_env: ShapeEnv, fake) -> Any:
        if isinstance(fake, int):
            return fake
        if isinstance(fake, torch.SymInt):
            return torch.SymInt(fake.node.with_shape_env(shape_env))
        if isinstance(fake, torch.SymFloat):
            return torch.SymFloat(fake.node.with_shape_env(shape_env))
        assert isinstance(fake, FakeTensorMeta)
        return FakeTensorMeta(
            tuple(new_with_shape_env(shape_env, s) for s in fake.size()),
            tuple(new_with_shape_env(shape_env, s) for s in fake.stride()),
            new_with_shape_env(shape_env, fake.storage_offset()),
            fake.is_nested,
        )

    # Checks whether the given shape_env fails when produce_guards is called.
    def check_shapeenv_fails(
        shape_env: ShapeEnv, tracked_fakes: Optional[list[Any]]
    ) -> Optional[ValidationException]:
        assert tracked_fakes is not None
        try:
            # This produce_guards call is a best-effort replication, since we
            # don't populate EqualityConstraint list. Reason: we would also have
            # to save OutputGraph.tracked_fakes_id_to_source.
            shape_env.produce_guards(
                [new_with_shape_env(shape_env, a.fake) for a in tracked_fakes],
                [a.source for a in tracked_fakes],
                input_contexts=[a.symbolic_context for a in tracked_fakes],
            )
            return None
        except ValidationException as e:
            return e

    # Checks whether the ShapeEnv reconstructed by replaying the events until
    # node is created fails when produce_guards is called.
    def check_node_fails(node: torch.fx.Node) -> Optional[ValidationException]:
        number = node.meta[SHAPEENV_EVENT_KEY]
        # Reconstruct shape_env until the event at event_number.
        shape_env = replay_shape_env_events(events[: number + 1])
        shape_env.graph.lint()
        return check_shapeenv_fails(shape_env, events[number].tracked_fakes)

    last_exception = check_shapeenv_fails(
        shape_env, shape_env._snapshot_tracked_fakes()
    )

    if not last_exception:
        # We don't actually fail due to a produce_guards call.
        # Stop and don't bisect.
        log.info("translation validation succeeded: no errors found.")
        return

    if not shape_env.should_record_events or config.translation_validation_no_bisect:
        # Bisection is off.
        # Return the last ValidationException we got.
        raise last_exception

    # Cache the raised exception (if any) at each bisection point.
    exception = {}

    # Bisection happens on the assertion nodes of the recorded FX graph for
    # dynamic shapes.
    assert_nodes = [
        node for node in shape_env.graph.nodes if node.target is torch._assert
    ]

    # Preparing the indices for binary search.
    # The overall invariants are
    # - for all i < left, assert_node[i] doesn't fail
    # - for all i >= right, assert_node[i] fails
    # - `right in exception` always holds
    # - `left <= right` always holds
    left, mid, right = 0, 0, len(assert_nodes) - 1
    exception[right] = check_node_fails(assert_nodes[right])

    while left < right:
        mid = (left + right) // 2

        node = assert_nodes[mid]
        log.debug("bisecting at %s: %s", mid, get_node_event(node))

        # Check whether the new shape_env raises a ValidationException or not.
        exception[mid] = check_node_fails(node)

        if exception[mid]:
            right = mid
        else:
            left = mid + 1

    assert left in exception and isinstance(exception[left], ValidationException)

    node = assert_nodes[left]
    event = get_node_event(node)

    if event.is_evaluate_expr():
        failed_action = "evaluating"
    else:
        assert event.is_defer_runtime_assert(), f"unexpected event type: {event}"
        failed_action = "adding runtime assert"

    args = event.args
    assert args is not None
    assert len(args) >= 2, (
        f"bisecting expects {event.name} to have at least 2 positional arguments. "
        f"Got: {len(args)}"
    )
    assert isinstance(args[1], sympy.Basic), (
        f"bisecting expects {event.name} to have a SymPy expression as its second argument. "
        f"Got: {type(args[1])}"
    )

    raise BisectValidationException(
        exception[left],
        expr=args[1],
        failed_action=failed_action,
        traced_node=node.meta[CURRENT_NODE_KEY],
    )
