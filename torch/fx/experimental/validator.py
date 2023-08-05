import functools
import logging
import math
import operator
import sympy

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Set, Tuple, Type, Union

import torch
from torch._dynamo.exc import TorchDynamoException
import torch.fx
import torch.fx.traceback as fx_traceback

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

        def get_args_str(e: z3.ExprRef) -> List[str]:
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

    # Implementation of Python semantics as Z3 expressions.
    #
    # Z3 Real-Int theory has operators with semantics that differ that of
    # Python. Therefore, in order to get it right, we need to implement
    # the (Python) semantics we are relying on in Z3.
    @dataclass
    class Z3Ops:
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

        # Implements Python division semantics.
        def div(self, numerator: z3.ArithRef, denominator: z3.ArithRef) -> z3.ArithRef:
            self.validator.add_assertion(denominator != 0)  # type: ignore[arg-type]
            return Z3Ops.to_real(numerator) / Z3Ops.to_real(denominator)

        def floor(self, number: z3.ArithRef) -> z3.ArithRef:
            # Z3 ToInt function rounds a real number towards negative infinity.
            return Z3Ops.to_int(number)

        # Python semantics for 'FloorDiv' states that before applying the floor
        # function, the operands are converted to their common type.
        def floordiv(self, numerator: z3.ArithRef, denominator: z3.ArithRef) -> z3.ArithRef:
            cast_result_to_real = numerator.is_real() or denominator.is_real()
            result = Z3Ops.to_int(self.div(numerator, denominator))
            # Since the 'result' is already an integer, we just have to check
            # whether we should cast it to real.
            return Z3Ops.to_real(result) if cast_result_to_real else result

        def ceil(self, number: z3.ArithRef) -> z3.ArithRef:
            return z3.If(
                self.floor(number) < number,
                self.floor(number + 1),
                number
            )  # type: ignore[return-value]

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
            return base ** exp

        def sqrt(self, number: z3.ArithRef) -> z3.ArithRef:
            # Square-root:
            # 1. Only work with reals
            number = Z3Ops.to_real(number)
            # 2. The number should be positive or zero.
            #    Otherwise, Z3 returns 'unknown'.
            self.validator.add_assertion(number >= 0)
            return number ** 0.5

    # Lifts a callable to be used in Z3.
    #
    # This function replaces the given 'op' by a function that:
    #
    #   1. Lifts the arguments into Z3 (i.e. make them inhabitants of Z3)
    #
    #   2. Calls an operation that corresponds to 'op', but works with Z3
    #      inhabitants (left as is if it works as is)
    def z3op(op: Callable, validator: "TranslationValidator") -> Callable:
        from torch.fx.experimental.symbolic_shapes import sym_sqrt

        # Operations that have booleans as their argument.
        # This is needed because the argument of some FX nodes were
        # literal integers, instead of booleans. So, whenever this flag
        # is set, we also convert ints to booleans.
        boolean_ops = {operator.not_, operator.and_, operator.or_}
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
                wrapped_args = (wrap(a) for a in args)
                # Run the function on the Z3 expressions.
                return func(*wrapped_args)

            return wrapper

        ops = Z3Ops(validator)
        replacement_map = {
            # Operator module.
            operator.not_: lift(z3.Not),
            operator.and_: lift(z3.And),
            operator.or_: lift(z3.Or),
            operator.floordiv: lift(ops.floordiv),
            operator.truediv: lift(ops.div),
            operator.mod: lift(ops.mod),

            # Math module.
            math.ceil: lift(ops.ceil),
            math.floor: lift(ops.floor),

            # Torch module.
            torch.sym_float: lift(ops.to_real),
            torch.sym_max: lift(ops.max),
            torch.sym_min: lift(ops.min),
            sym_sqrt: lift(ops.sqrt),
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

        def placeholder(self, target: Target, args: Tuple[Argument, ...], kwargs: Dict[str, Any]) -> Any:
            symbol = fx_traceback.get_current_meta()["symbol"]
            return self.validator.z3var(symbol)

        def call_function(self, target: Target, args: Tuple[Argument, ...], kwargs: Dict[str, Any]) -> Any:
            if target != torch._assert:
                # Actually runs the node target function (which is already
                # lifted) with its arguments.
                return super().call_function(target, args, kwargs)
            # Adds the Z3 expression corresponding to the first argument
            # as a validator input.
            assert len(args) == 1, f"expected 1 argument on assertion. Got: {len(args)} "
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
            self._ops = Z3Ops(self._validator)

        def constant(self, value: Any, dtype: torch.dtype) -> z3.ExprRef:
            if dtype is torch.int64:
                return z3.IntVal(int(value))
            if dtype is torch.double:
                return z3.RealVal(float(value))
            if dtype is torch.bool:
                return z3.BoolVal(bool(value))
            raise ValueError(f"unsupported dtype (SympyToZ3): {dtype}")

        def truediv(self, numerator: z3.ArithRef, denominator: z3.ArithRef) -> z3.ArithRef:
            return self._ops.div(numerator, denominator)

        def floordiv(self, numerator: z3.ArithRef, denominator: z3.ArithRef) -> z3.ArithRef:
            return self._ops.floordiv(numerator, denominator)

        def div(self, numerator: z3.ArithRef, denominator: z3.ArithRef) -> z3.ArithRef:
            return self._ops.floordiv(numerator, denominator)

        def pow(self, base: z3.ArithRef, exp: z3.ArithRef) -> z3.ArithRef:
            return self._ops.pow(base, exp)

        def mod(self, p: z3.ArithRef, q: z3.ArithRef) -> z3.ArithRef:
            return self._ops.mod(p, q)

        def __getattr__(self, name: str) -> Any:
            REPLACEMENT = {
                "and_": z3.And,
                "or_": z3.Or,
                "not_": z3.Not,
                "floor": self._ops.floor,
                "ceil": self._ops.ceil,
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
            self.symbols: Dict[sympy.Symbol, z3.ExprRef] = {}

            # Set of source Z3 expressions.
            # They represent the generated guards without any kind of
            # simplification or transformation.
            self._source_exprs: Set[z3.BoolRef] = set()

            # Set of target Z3 expressions.
            # They represent the actual checked guards at runtime. They might
            # be simplified or transformed versions of the source guards.
            self._target_exprs: Set[z3.BoolRef] = set()

            # Set of Z3 expressions representing assertions over both the
            # source and target expressions.
            self._assertions: Set[z3.BoolRef] = set()

        # Retrieves the corresponding Z3 variable.
        def z3var(self, symbol: sympy.Symbol) -> z3.ExprRef:
            assert symbol in self.symbols, f"Z3 variable not found for: {symbol}"
            return self.symbols[symbol]

        # Create a variable in Z3 of 'type' for 'symbol', if it doesn't already exists.
        def add_var(self, symbol: sympy.Symbol, type: Type) -> z3.ExprRef:
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
            assert isinstance(z3expr, z3.BoolRef), f"expected boolean expression. Got: {z3expr}"
            return z3expr

        def add_source_expr(self, e: z3.BoolRef) -> None:
            if e not in self._source_exprs:
                log.debug("add source guard: %s", z3str(e))
            self._source_exprs.add(e)

        def add_target_expr(self, e: sympy.Expr) -> None:
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
            from torch._dynamo.utils import dynamo_timed

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
            r = dynamo_timed()(solver.check)()
            if r == z3.sat:
                # Target expressions are unsound.
                # Log the found model and the source expressions that failed.
                model = solver.model()
                raise ValidationException(
                    model, self._assertions, self._target_exprs,
                    failed_source_exprs=[
                        inp for inp in self._source_exprs if not model.evaluate(inp)
                    ]
                )
            else:
                if r == z3.unknown:
                    # Could not find a solution. It didn't fail, but it also
                    # didn't succeed. Canceling the validation execution (keyboard
                    # interrupt) also gets to this branch.
                    log.warning("translation validation: could not validate: got z3.unknown")
                else:
                    # Target expressions are sound.
                    assert r == z3.unsat
                    log.debug("translation validation: success")


    class ValidationException(TorchDynamoException):
        def __init__(
            self,
            model,
            assertions: Iterable[z3.ExprRef],
            target_exprs: Iterable[z3.ExprRef],
            failed_source_exprs: Iterable[z3.ExprRef]
        ) -> None:
            model_str = self._joinlines(model, lambda sym: f"{sym}: {model[sym]}")
            assertions_str = self._joinlines(assertions, z3str)
            target_exprs_str = self._joinlines(target_exprs, z3str)
            failed_source_exprs_str = self._joinlines(failed_source_exprs, z3str)

            super().__init__(
                "translation validation failed.\n\n"
                "Model:\n" + model_str + "\n\n"
                "Assertions:\n" + assertions_str + "\n\n"
                "Target Expressions:\n" + target_exprs_str + "\n\n"
                "Failed Source Expressions:\n" + failed_source_exprs_str + "\n\n"
            )

        def _joinlines(self, xs: Iterable[Any], f: Callable[[Any], str] = lambda x: x) -> str:
            return "\n".join(f"  ==> {f(x)}" for x in xs)

except ImportError:
    _HAS_Z3 = False
else:
    _HAS_Z3 = True

from torch._dynamo import config

def translation_validation_enabled() -> bool:
    # Checks everytime this function is called, in case the Dynamo
    # option is set, but Z3 is not installed.
    assert_z3_installed_if_tv_set()
    return _HAS_Z3 and config.translation_validation


def translation_validation_timeout() -> int:
    return config.translation_validation_timeout


def assert_z3_installed_if_tv_set():
    assert _HAS_Z3 or not config.translation_validation, (
        "translation validation requires Z3 package. Please, either install "
        "z3-solver or disable translation validation."
    )

# Checks when this module is loaded.
assert_z3_installed_if_tv_set()
