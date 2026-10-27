"""Model exact Decimal values without treating them as generic Python constants.

Arithmetic is folded only in fresh explicit contexts, with no FX operations in
their bodies. Such contexts are private to the trace and reconstructed with
their final flags and traps. Ambient arithmetic remains unsupported.
"""

import decimal
import operator
from typing import Any, NoReturn, TYPE_CHECKING

from .. import graph_break_hints
from ..exc import raise_observed_exception, unimplemented
from ..guards import GuardBuilder, install_guard
from ..source import AttrSource, CallFunctionNoArgsSource
from .base import (
    AsPythonConstantNotImplementedError,
    NO_SUCH_SUBOBJ,
    ValueMutationNew,
    VariableTracker,
)
from .constant import ConstantVariable
from .ctx_manager import ContextWrappingVariable


if TYPE_CHECKING:
    from torch._dynamo.codegen import PyCodegen
    from torch._dynamo.symbolic_convert import InstructionTranslatorBase


_CONTEXT_TYPE = decimal.Context
_CONTEXT_FIELDS = ("prec", "rounding", "Emin", "Emax", "capitals", "clamp")
_SIGNALS = (
    decimal.Clamped,
    decimal.InvalidOperation,
    decimal.DivisionByZero,
    decimal.Inexact,
    decimal.Rounded,
    decimal.Subnormal,
    decimal.Overflow,
    decimal.Underflow,
    decimal.FloatOperation,
)


def _context_state(context: decimal.Context) -> tuple[Any, ...]:
    return (
        *(getattr(context, name) for name in _CONTEXT_FIELDS),
        tuple(context.flags[signal] for signal in _SIGNALS),
        tuple(context.traps[signal] for signal in _SIGNALS),
    )


def _default_context_state() -> tuple[Any, ...]:
    return _context_state(_CONTEXT_TYPE())


def _context_capitals() -> int:
    return decimal.getcontext().capitals


def _reconstruct_context(state: tuple[Any, ...]) -> decimal.Context:
    return _CONTEXT_TYPE(
        **dict(zip(_CONTEXT_FIELDS, state[:6])),
        flags=[signal for signal, enabled in zip(_SIGNALS, state[6]) if enabled],
        traps=[signal for signal, enabled in zip(_SIGNALS, state[7]) if enabled],
    )


def _unsupported(operation: str) -> NoReturn:
    unimplemented(
        gb_type="Unsupported Decimal context operation",
        context=operation,
        explanation="Dynamo only supports constant Decimal operations in a fresh, explicit local context.",
        hints=[*graph_break_hints.SUPPORTABLE],
    )


def _unwrap(value: VariableTracker) -> Any:
    if isinstance(value, (DecimalVariable, DecimalContextVariable)):
        return value.value
    try:
        result = value.as_python_constant()
    except AsPythonConstantNotImplementedError:
        _unsupported("Decimal operation on a non-constant argument")

    def safe(obj: Any) -> bool:
        if type(obj) in (int, float, str, bool, type(None)):
            return True
        if type(obj) in (tuple, list):
            return all(safe(item) for item in obj)
        if type(obj) is dict:
            return all(safe(key) and safe(item) for key, item in obj.items())
        return any(obj is signal for signal in _SIGNALS)

    if not safe(result):
        _unsupported("Decimal operation on a non-literal argument")
    return result


def _wrap(tx: "InstructionTranslatorBase", value: Any) -> VariableTracker:
    if type(value) is decimal.Decimal:
        return DecimalVariable(value)
    if type(value) is decimal.Context:
        return DecimalContextVariable(value)
    return VariableTracker.build(tx, value)


def _invoke(
    tx: "InstructionTranslatorBase", fn: Any, *args: Any, **kwargs: Any
) -> VariableTracker:
    if tx.output.decimal_contexts:
        tx.output.side_effects.check_allowed_side_effect(tx.output.decimal_contexts[-1])
    try:
        result = fn(*args, **kwargs)
    except (ArithmeticError, TypeError, ValueError) as error:
        raise_observed_exception(type(error), tx, args=list(error.args))
    return _wrap(tx, result)


class DecimalVariable(VariableTracker):
    _cpython_type = decimal.Decimal
    _nonvar_fields = {"value", *VariableTracker._nonvar_fields}

    def __init__(self, value: decimal.Decimal, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.value = value

    def python_type(self) -> type:
        return decimal.Decimal

    def get_real_python_backed_value(self) -> object:
        # NaN container comparisons depend on identity, unlike numeric equality.
        # Pin sourced NaNs before exposing them to identity checks. Tuple-value
        # guards still protect their values against address reuse.
        if self.source is None or not self.value.is_nan():
            return NO_SUCH_SUBOBJ
        install_guard(self.source.make_guard(GuardBuilder.ID_MATCH))
        return self.value

    def hash_impl(self, tx: "InstructionTranslatorBase") -> tuple[int, bool]:
        value = _invoke(tx, hash, self.value).as_python_constant()
        if self.value.is_qnan():
            # Quiet NaNs use an identity hash, unlike finite Decimals.
            if self.source is None:
                return value, True
            source = CallFunctionNoArgsSource(AttrSource(self.source, "__hash__"))
            install_guard(source.make_guard(GuardBuilder.EQUALS_MATCH))
        return value, False

    def tp_richcompare_impl(
        self,
        tx: "InstructionTranslatorBase",
        other: VariableTracker,
        op: str,
    ) -> VariableTracker:
        other = other.realize()
        if isinstance(other, DecimalVariable):
            value = other.value
        elif other.is_python_constant():
            value = other.as_python_constant()
            if type(value) not in (int, float, complex, bool, str, bytes, type(None)):
                _unsupported("Decimal comparison with a non-literal argument")
        else:
            # Decimal handles Rational operands before trying reflected slots.
            # Returning NotImplemented here could silently change that order.
            _unsupported("Decimal comparison with an unmodeled operand")
        method = getattr(self.value, op)
        if tx.output.decimal_contexts:
            return _invoke(tx, method, value)
        # Comparing floats or NaNs can set flags even without raising.
        probe = _CONTEXT_TYPE(traps=list(_SIGNALS))
        try:
            with decimal.localcontext(probe) as probe:
                result = method(value)
        except decimal.DecimalException:
            _unsupported("Decimal comparison with context-dependent errors")
        if any(probe.flags.values()):
            _unsupported("Decimal comparison with context-dependent flags")
        return ConstantVariable.create(result)

    @classmethod
    def create(
        cls,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        context = kwargs.get("context", args[1] if len(args) > 1 else None)
        explicit = isinstance(context, DecimalContextVariable)
        active = bool(tx.output.decimal_contexts)
        if explicit:
            tx.output.side_effects.check_allowed_side_effect(context)
        if context is not None and not explicit and not context.is_constant_none():
            _unsupported("Decimal with an unmodeled context")
        values = [_unwrap(arg) for arg in args]
        keywords = {name: _unwrap(arg) for name, arg in kwargs.items()}
        if not explicit and not active:
            # A disposable context detects flag-setting/trapping constructors
            # without touching the caller's live context during tracing.
            if (
                values
                and type(values[0]) is float
                or "value" in keywords
                and type(keywords["value"]) is float
            ):
                _unsupported("Decimal(float)")
            probe = _CONTEXT_TYPE(traps=list(_SIGNALS))
            probe.clear_flags()
            try:
                with decimal.localcontext(probe) as probe:
                    value = decimal.Decimal(*values, **keywords)
            except decimal.DecimalException:
                _unsupported("Decimal construction with context-dependent errors")
            except (TypeError, ValueError) as error:
                raise_observed_exception(type(error), tx, args=list(error.args))
            if any(probe.flags.values()):
                _unsupported("Decimal construction with context-dependent flags")
            result = cls(value)
        else:
            result = _invoke(tx, decimal.Decimal, *values, **keywords)
        original = kwargs.get("value", args[0] if args else None)
        if (
            isinstance(original, cls)
            and isinstance(result, cls)
            and result.value is original.value
        ):
            return original
        return result

    def reconstruct(self, codegen: "PyCodegen") -> None:
        codegen.add_push_null(lambda: codegen.load_import_from("decimal", "Decimal"))
        codegen.append_output(codegen.create_load_const(tuple(self.value.as_tuple())))
        codegen.call_function(1, False)

    def nb_float_impl(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        return _invoke(tx, float, self.value)

    def nb_int_impl(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        return _invoke(tx, int, self.value)

    def nb_bool_impl(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        return ConstantVariable.create(bool(self.value))

    def call_method(
        self,
        tx: "InstructionTranslatorBase",
        name: str,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        if name in (
            "__hash__",
            "__eq__",
            "__ne__",
            "__lt__",
            "__le__",
            "__gt__",
            "__ge__",
        ):
            return super().call_method(tx, name, args, kwargs)
        independent = (
            "as_tuple",
            "is_finite",
            "is_infinite",
            "is_nan",
            "is_snan",
            "is_qnan",
            "is_signed",
            "is_zero",
            "copy_abs",
            "copy_negate",
            "copy_sign",
        )
        contextual = (
            "sqrt",
            "exp",
            "ln",
            "log10",
            "quantize",
            "normalize",
            "to_integral",
            "to_integral_exact",
            "to_integral_value",
        )
        if name not in independent and (
            name not in contextual or not tx.output.decimal_contexts
        ):
            _unsupported(f"Decimal.{name}")
        for argument in (*args, *kwargs.values()):
            if isinstance(argument, DecimalContextVariable):
                tx.output.side_effects.check_allowed_side_effect(argument)
        return _invoke(
            tx,
            getattr(self.value, name),
            *[_unwrap(arg) for arg in args],
            **{key: _unwrap(arg) for key, arg in kwargs.items()},
        )

    def tp_str_impl(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        if not tx.output.decimal_contexts:
            source = CallFunctionNoArgsSource(
                AttrSource(tx.import_source(__name__), "_context_capitals")
            )
            install_guard(source.make_guard(GuardBuilder.EQUALS_MATCH))
        return ConstantVariable.create(str(self.value))

    def tp_repr_impl(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        if not tx.output.decimal_contexts:
            source = CallFunctionNoArgsSource(
                AttrSource(tx.import_source(__name__), "_context_capitals")
            )
            install_guard(source.make_guard(GuardBuilder.EQUALS_MATCH))
        return ConstantVariable.create(repr(self.value))

    def _binary(
        self,
        tx: "InstructionTranslatorBase",
        other: VariableTracker,
        op: Any,
        reverse: bool,
    ) -> VariableTracker:
        if not tx.output.decimal_contexts:
            _unsupported("Decimal arithmetic outside an explicit local context")
        left, right = self.value, _unwrap(other)
        return _invoke(tx, op, right, left) if reverse else _invoke(tx, op, left, right)

    def nb_add_impl(
        self,
        tx: "InstructionTranslatorBase",
        other: VariableTracker,
        reverse: bool = False,
    ) -> VariableTracker:
        return self._binary(tx, other, operator.add, reverse)

    def nb_subtract_impl(
        self,
        tx: "InstructionTranslatorBase",
        other: VariableTracker,
        reverse: bool = False,
    ) -> VariableTracker:
        return self._binary(tx, other, operator.sub, reverse)

    def nb_multiply_impl(
        self,
        tx: "InstructionTranslatorBase",
        other: VariableTracker,
        reverse: bool = False,
    ) -> VariableTracker:
        return self._binary(tx, other, operator.mul, reverse)

    def nb_true_divide_impl(
        self,
        tx: "InstructionTranslatorBase",
        other: VariableTracker,
        reverse: bool = False,
    ) -> VariableTracker:
        return self._binary(tx, other, operator.truediv, reverse)

    def nb_power_impl(
        self,
        tx: "InstructionTranslatorBase",
        other: VariableTracker,
        z: VariableTracker | None,
        reverse: bool = False,
    ) -> VariableTracker:
        if z is not None and not z.is_constant_none():
            _unsupported("three-argument Decimal power")
        return self._binary(tx, other, operator.pow, reverse)


class DecimalContextVariable(VariableTracker):
    _cpython_type = decimal.Context
    _nonvar_fields = {"value", *VariableTracker._nonvar_fields}

    def __init__(self, value: decimal.Context, **kwargs: Any) -> None:
        super().__init__(mutation_type=ValueMutationNew(), **kwargs)
        self.value = value

    @classmethod
    def create(
        cls,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        source = CallFunctionNoArgsSource(
            AttrSource(tx.import_source(__name__), "_default_context_state")
        )
        install_guard(source.make_guard(GuardBuilder.EQUALS_MATCH))
        return _invoke(
            tx,
            decimal.Context,
            *[_unwrap(arg) for arg in args],
            **{name: _unwrap(arg) for name, arg in kwargs.items()},
        )

    def python_type(self) -> type:
        return decimal.Context

    def reconstruct(self, codegen: "PyCodegen") -> None:
        codegen.add_push_null(
            lambda: codegen.load_import_from(__name__, "_reconstruct_context")
        )
        codegen.append_output(codegen.create_load_const(_context_state(self.value)))
        codegen.call_function(1, False)

    def tp_getattro_impl(
        self, tx: "InstructionTranslatorBase", name: str
    ) -> VariableTracker:
        if name in ("flags", "traps"):
            _unsupported(f"Context.{name} access during tracing")
        if name in _CONTEXT_FIELDS:
            return VariableTracker.build(tx, getattr(self.value, name))
        return super().tp_getattro_impl(tx, name)

    def call_method(
        self,
        tx: "InstructionTranslatorBase",
        name: str,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        if name not in (
            "copy",
            "clear_flags",
            "clear_traps",
            "create_decimal",
            "create_decimal_from_float",
        ):
            _unsupported(f"Context.{name}")
        tx.output.side_effects.check_allowed_side_effect(self)
        return _invoke(
            tx,
            getattr(self.value, name),
            *[_unwrap(arg) for arg in args],
            **{key: _unwrap(arg) for key, arg in kwargs.items()},
        )


class DecimalLocalContextVariable(ContextWrappingVariable):
    _nonvar_fields = {"kwargs", "node_count", *ContextWrappingVariable._nonvar_fields}

    def __init__(self, context: DecimalContextVariable, kwargs: dict[str, Any]) -> None:
        super().__init__(target_values=None)
        self.context = context
        self.kwargs = kwargs
        self.node_count = 0

    @classmethod
    def create(
        cls,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        if len(args) != 1 or not isinstance(args[0], DecimalContextVariable):
            _unsupported("localcontext without a fresh explicit Context")
        keywords = {key: _unwrap(value) for key, value in kwargs.items()}
        # Validate without entering or changing the caller's active context.
        try:
            decimal.localcontext(args[0].value, **keywords)
        except (TypeError, ValueError) as error:
            raise_observed_exception(type(error), tx, args=list(error.args))
        return cls(args[0], keywords)

    def python_type(self) -> type:
        return type(decimal.localcontext())

    def enter(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        if getattr(self, "cleanup_fn", None) is not None:
            _unsupported("reentrant Decimal localcontext manager")
        self.node_count = len(tx.output.graph.nodes)
        manager = decimal.localcontext(self.context.value, **self.kwargs)
        context = DecimalContextVariable(manager.__enter__())
        tx.output.decimal_contexts.append(context)

        def cleanup() -> None:
            manager.__exit__(None, None, None)
            tx.output.decimal_contexts.remove(context)

        self.set_cleanup_hook(tx, cleanup)
        return context

    def exit(
        self, tx: "InstructionTranslatorBase", *args: VariableTracker
    ) -> VariableTracker:
        self.cleanup_assert()
        if len(tx.output.graph.nodes) != self.node_count:
            _unsupported("tensor operations within a Decimal local context")
        return ConstantVariable.create(None)

    def supports_graph_breaks(self) -> bool:
        return False

    def reconstruct(self, codegen: "PyCodegen") -> None:
        _unsupported("escaping Decimal localcontext manager")
