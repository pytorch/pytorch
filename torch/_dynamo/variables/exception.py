"""Variable tracker for exception instances.

Lives in its own module (rather than misc.py) so user_defined.py can use
ExceptionVariable as a base class without an import cycle: misc.py subclasses
UserDefinedObjectVariable.
"""

import traceback
from typing import Any, TYPE_CHECKING

from .. import graph_break_hints, variables
from ..exc import raise_observed_exception, raise_type_error, unimplemented
from ..source import AttrSource
from ..utils import istype, unpack_iterable
from .base import VariableTracker
from .constant import ConstantVariable
from .object_protocol import generic_str, object_richcompare


if TYPE_CHECKING:
    from torch._dynamo.codegen import PyCodegen
    from torch._dynamo.symbolic_convert import InstructionTranslatorBase


class ExceptionVariable(VariableTracker):
    # _PyExc_BaseException: https://github.com/python/cpython/blob/v3.13.0/Objects/exceptions.c
    _cpython_type = BaseException

    # The ExceptionVariable corresponds to the BaseException class in Python
    def __init__(
        self,
        exc_type: Any = None,
        args: list[VariableTracker] | None = None,
        init_kwargs: dict[str, VariableTracker] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.exc_type = exc_type
        self.args = args if args is not None else []
        if init_kwargs:
            unimplemented(
                gb_type="Keyword args passed to exception constructor",
                context=f"{self} with kwargs {init_kwargs}",
                explanation="Dynamo does not know how to handle keyword args passed to an exception constructor",
                hints=[*graph_break_hints.SUPPORTABLE],
            )
        # When raising a new exception while another exception is already being
        # handled, the new exception's __context__ attribute is automatically
        # set to the handled exception.
        self.__context__: VariableTracker = ConstantVariable.create(None)
        # Set when user raised an exception from another:
        # raise ... from ...
        self.__cause__: VariableTracker = ConstantVariable.create(None)
        # Boolean flag that controls whether the __context__ attribute is set
        self.__suppress_context__: VariableTracker = ConstantVariable.create(False)
        # Contains the call stack where the exception was raised.
        self.__traceback__: VariableTracker = ConstantVariable.create(None)
        # The user stack at the time this exception was first raised.
        # Used to preserve the original exception location when re-raising.
        self.python_stack: traceback.StackSummary | None = None

    def set_context(self, context: VariableTracker) -> None:
        self.__context__ = context

    def reconstruct(self, codegen: "PyCodegen") -> None:
        codegen.add_push_null(
            lambda: codegen.load_import_from("builtins", self.exc_type.__name__)
        )
        codegen.foreach(self.args)
        codegen.call_function(len(self.args), False)

        def codegen_attr(name: str) -> None:
            attr = getattr(self, name)
            if istype(attr, ConstantVariable):
                if attr.value not in (True, False, None):
                    raise AssertionError(
                        f"attr.value must be True, False, or None, got {attr}"
                    )
            else:
                codegen.dup_top()
                codegen(attr)
                codegen.extend_output(codegen.rot_n(2))
                codegen.store_attr(name)

        codegen_attr("__context__")
        codegen_attr("__cause__")
        codegen_attr("__suppress_context__")

    def python_type(self) -> type:
        return self.exc_type

    def richcompare_impl(
        self, tx: "InstructionTranslatorBase", other: "VariableTracker", op: str
    ) -> "VariableTracker":
        from .object_protocol import object_richcompare

        return object_richcompare(self, tx, other, op)

    def call_method(
        self,
        tx: "InstructionTranslatorBase",
        name: str,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        if name == "__setattr__":
            name = args[0].as_python_constant()
            val = args[1]
            if name == "__context__":
                # Constant can be either an Exceptior or None
                if not (
                    val.is_constant_none()
                    or isinstance(
                        val,
                        (
                            variables.ExceptionVariable,
                            variables.UserDefinedExceptionClassVariable,
                            variables.UserDefinedExceptionObjectVariable,
                        ),
                    )
                ):
                    raise_type_error(
                        tx,
                        "exception context must be None or derive from BaseException",
                    )
                self.set_context(val)
            elif name == "__cause__":
                if val.is_constant_none() or isinstance(
                    val,
                    (
                        variables.BuiltinVariable,
                        variables.ExceptionVariable,
                        variables.UserDefinedExceptionClassVariable,
                        variables.UserDefinedExceptionObjectVariable,
                    ),
                ):
                    self.__cause__ = val
                    self.__suppress_context__ = variables.ConstantVariable.create(True)
                else:
                    raise_type_error(
                        tx, "exception cause must be None or derive from BaseException"
                    )
            elif name == "__suppress_context__":
                if val.is_constant_match(True, False):
                    self.__suppress_context__ = val
                else:
                    raise_type_error(
                        tx, "exception cause must be None or derive from BaseException"
                    )
            elif name == "__traceback__":
                if not variables.TracebackVariable.is_valid_traceback(val):
                    raise_type_error(tx, "__traceback__ must be a traceback or None")
                self.__traceback__ = val
            elif name == "args":
                # CPython coerces any iterable to a tuple (PySequence_Tuple).
                self.args = unpack_iterable(tx, val)
            else:
                # Arbitrary user attribute -> store in the instance __dict__
                # via the side effects table.
                se = tx.output.side_effects
                if not se.is_attribute_mutation(self):
                    se.track_attribute_mutation_new(self)
                se.store_instance_dict_attr(self, name, val)
            return variables.ConstantVariable.create(None)
        elif name == "__setstate__":
            if len(args) != 1:
                raise_type_error(
                    tx, f"__setstate__() takes exactly one argument ({len(args)} given)"
                )
            [state] = args
            # BaseException.__setstate__(None) is a documented no-op.
            if state.is_constant_none():
                return variables.ConstantVariable.create(None)
            if not isinstance(state, variables.ConstDictVariable):
                raise_type_error(tx, "state is not a dictionary")
            for key, value in state.keys_as_python_constant().items():
                self.call_method(
                    tx, "__setattr__", [ConstantVariable.create(key), value], {}
                )
            return variables.ConstantVariable.create(None)
        elif name == "with_traceback":
            if len(args) != 1:
                raise_type_error(
                    tx,
                    f"with_traceback() takes exactly one argument ({len(args)} given)",
                )
            [tb] = args
            if not variables.TracebackVariable.is_valid_traceback(tb):
                raise_type_error(tx, "__traceback__ must be a traceback or None")
            self.__traceback__ = tb
            return self
        else:
            return super().call_method(tx, name, args, kwargs)

    def getattro_impl(
        self, tx: "InstructionTranslatorBase", name: str
    ) -> VariableTracker:
        if name == "__class__":
            return VariableTracker.build(tx, self.exc_type)
        elif name == "__context__":
            return self.__context__
        elif name == "__cause__":
            return self.__cause__
        elif name == "__suppress_context__":
            return self.__suppress_context__
        elif name == "__traceback__":
            return self.__traceback__
        elif name == "args":
            return VariableTracker.build(
                tx,
                tuple(self.args),
                source=self.source and AttrSource(self.source, "args"),
            )
        try:
            # Custom attributes are stored in the side effects instance dict and
            # resolved by generic_getattr before reaching here, so a fall-through
            # to the generic lookup that finds nothing means the attribute is
            # genuinely absent -- match CPython's BaseException tp_getattro
            # (PyObject_GenericGetAttr) and raise AttributeError.
            return super().getattro_impl(tx, name)
        except NotImplementedError:
            raise_observed_exception(
                AttributeError,
                tx,
                args=[f"'{self.exc_type.__name__}' object has no attribute '{name}'"],
            )

    def str_impl(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        # ref: https://github.com/python/cpython/blob/v3.13.3/Objects/exceptions.c#L118-L129
        if len(self.args) == 0:
            return VariableTracker.build(tx, "")
        elif len(self.args) == 1:
            return generic_str(tx, self.args[0])
        else:
            from . import TupleVariable

            tuple_var = variables.TupleVariable(list(self.args))
            return generic_str(tx, tuple_var)

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.exc_type})"

    __repr__ = __str__

    @staticmethod
    def _debug_format_arg(arg: VariableTracker) -> str:
        try:
            return repr(arg.as_python_constant())
        except Exception:
            return arg.debug_repr()

    def debug_repr(self) -> str:
        args = ", ".join(self._debug_format_arg(arg) for arg in self.args)
        return f"{self.python_type_name()}({args})"

    def repr_impl(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        # ref: BaseException_repr in https://github.com/python/cpython/blob/3.13/Objects/exceptions.c#L135-L142
        return VariableTracker.build(tx, self.debug_repr())
