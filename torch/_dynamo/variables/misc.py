"""
This module contains miscellaneous variable tracker implementations for various Python types
and features used in Dynamo's symbolic execution. These classes help track and propagate
information about different kinds of variables during graph capture.

Key classes include:
- SuperVariable: Handles super() calls and method resolution
- ExceptionVariable: Tracks exception objects
- RandomVariable: Manages random number generators
- GetAttrVariable: Tracks attribute access
- MethodWrapperVariable: Handles method wrappers
- PythonModuleVariable: Tracks Python modules
- NumpyVariable: Handles numpy functions and types
- StringFormatVariable: Manages string formatting
- DebuggingVariable: Handles print and logging
"""

import dataclasses
import enum
import functools
import inspect
import itertools
import logging
import random
import re
import sys
import traceback
import types
import weakref
from collections.abc import Callable, Sequence
from random import Random
from types import BuiltinFunctionType
from typing import Any, Literal, NoReturn, TYPE_CHECKING, TypeGuard, Union

import torch._C
import torch._numpy as tnp
import torch.utils._pytree as pytree
from torch._dynamo.variables.base import MutationType
from torch._dynamo.variables.lists import TupleVariable
from torch._guards import Source

from .. import config, graph_break_hints, trace_rules, variables
from ..bytecode_transformation import (
    create_call_function,
    create_call_function_ex,
    create_instruction,
)
from ..create_parameter_op import do_not_convert_to_tracable_parameter
from ..exc import raise_observed_exception, unimplemented
from ..guards import GuardBuilder, install_guard
from ..mutation_guard import unpatched_nn_module_init
from ..source import (
    AttrSource,
    DictGetItemSource,
    GenericAttrSource,
    GetItemSource,
    TypeMROSource,
    TypeSource,
    WeakRefCallSource,
)
from ..utils import (
    check_unspec_or_constant_args,
    cmp_name_to_op_mapping,
    identity,
    is_tensor_base_attr_getter,
    istype,
    list_methods,
    proxy_args_kwargs,
    raise_args_mismatch,
    tuple_methods,
)
from .base import (
    AsPythonConstantNotImplementedError,
    raise_type_error_exc,
    VariableTracker,
)
from .constant import CONSTANT_VARIABLE_FALSE, CONSTANT_VARIABLE_NONE, ConstantVariable
from .functions import NestedUserFunctionVariable, UserFunctionVariable
from .user_defined import call_random_fn, is_standard_setattr, UserDefinedObjectVariable


if TYPE_CHECKING:
    from torch._dynamo.codegen import PyCodegen
    from torch._dynamo.symbolic_convert import InstructionTranslator


class NO_SUCH_SUBOBJ:
    pass


class SuperVariable(VariableTracker):
    _nonvar_fields = {
        *VariableTracker._nonvar_fields,
    }

    def __init__(
        self,
        typevar: VariableTracker,
        objvar: VariableTracker | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        # typevar is the first argument to super(). In the case where no argument
        # is provided to super(), it is the __class__ object where
        # the super() function is being called
        self.typevar = typevar
        # objvar here must be an instance or subtype of typevar.
        # In the case where super() is called without arguments, it is the first argument
        # to the current function where super() is called from (self for regular method,
        # cls for a classmethod)
        self.objvar = objvar

    def reconstruct(self, codegen: "PyCodegen") -> None:
        codegen.add_push_null(lambda: codegen(variables.BuiltinVariable(super)))
        codegen(self.typevar)
        if self.objvar is not None:
            codegen(self.objvar)
            codegen.extend_output(create_call_function(2, False))
        else:
            codegen.extend_output(create_call_function(1, False))

    def _resolved_getattr_and_source(
        self, tx: "InstructionTranslator", name: str
    ) -> tuple[Any, AttrSource | None]:
        if not self.objvar:
            unimplemented(
                gb_type="1-arg super not implemented",
                context="",
                explanation=f"Dynamo failed to trace attribute `{name}` accessed "
                f"via `super()` (for type `{self.typevar}` and object `{self.objvar}`) "
                "because one-argument of super() is not supported.",
                hints=[
                    "Use two-argument super(type, object_or_type).",
                ],
            )
        assert self.objvar is not None
        search_type = self.typevar.as_python_constant()

        # The rest of this function does two things:
        #   - Walk the mro to find where the attribute comes from to be
        #     able to provide accurate source
        #   - Call the getattr to get the object

        # Find the class object, where the function lives.
        # When objvar is "self", use type(self), when objvar is "cls", use it as-is
        type_to_use = self.objvar.python_type()
        type_to_use_source: Source | None = (
            TypeSource(self.objvar.source) if self.objvar.source else None
        )
        if issubclass(type_to_use, type):
            type_to_use = self.objvar.value  # type: ignore[attr-defined]
            type_to_use_source = self.objvar.source

        source = None
        search_mro = type_to_use.__mro__

        try:
            start_index = search_mro.index(search_type) + 1
        except ValueError:
            # Corner case where the typevar is not in the mro of the objvar
            # https://github.com/python/cpython/blob/3.11/Objects/typeobject.c#L8843-L8844
            return getattr(super(search_type, type_to_use), name), None
        # Implemented based on https://github.com/python/cpython/blob/3.11/Objects/typeobject.c#L8812
        # super has its getattro implementation. The key point is that instead of calling getattr, it checks the
        # attribute in the class __dict__
        for index in range(start_index, len(search_mro)):
            # Dont call getattr, just check the __dict__ of the class
            if resolved_getattr := search_mro[index].__dict__.get(name, NO_SUCH_SUBOBJ):
                if resolved_getattr is not NO_SUCH_SUBOBJ:
                    # Equivalent of something like type(L['self']).__mro__[1].attr_name
                    if type_to_use_source:
                        source = AttrSource(
                            GetItemSource(TypeMROSource(type_to_use_source), index),
                            name,
                        )
                    return resolved_getattr, source

        unimplemented(
            gb_type="Unable to resolve super getattr",
            context="",
            explanation=f"Dynamo failed to trace attribute `{name}` accessed "
            f"via `super()` (for type `{self.typevar}` and object `{self.objvar}`) "
            "because the resolved attribute type is not supported.",
            hints=[
                "Ensure the attribute exists in the parent class.",
                "Check the arguments passed to `super()`.",
            ],
        )

    def var_getattr(self, tx: "InstructionTranslator", name: str) -> VariableTracker:
        # Check if getattr is a constant. If not, delay the actual work by
        # wrapping the result in GetAttrVariable. Mostly super is called with a
        # method, so most of the work is delayed to call_function.
        #
        # We could have just implemented a const_getattr. However, super is
        # special when it comes to finding sources. Compared to other VTs, super
        # requires the attr name to walk the mro and find the actual source (and
        # not just AttrSource).
        value, source = self._resolved_getattr_and_source(tx, name)
        if not variables.ConstantVariable.is_literal(value):
            return GetAttrVariable(self, name)
        if source:
            install_guard(source.make_guard(GuardBuilder.CONSTANT_MATCH))
        return variables.ConstantVariable.create(value, source=source)

    def call_method(
        self,
        tx: "InstructionTranslator",
        name: str,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        inner_fn, source = self._resolved_getattr_and_source(tx, name)
        assert self.objvar is not None
        # This essentially simulates CPython's `super_getattro`:
        # https://github.com/python/cpython/blob/a1c52d1265c65bcf0d9edf87e143843ad54f9b8f/Objects/typeobject.c#L11138-L11168
        # where `inner_fn` is the VT for `res = _super_lookup_descr(...)`.
        #
        # However, `res`'s type needs to be checked for `tp_descr_get`, and
        # applied if it has one. We currently don't have polyfills for all the
        # relevant `tp_descr_get`, so we explicitly handle the cases we care
        # about here (e.g., note the staticmethod, classmethod cases).
        if inner_fn is object.__init__:
            return LambdaVariable(identity)
        elif inner_fn is torch.nn.Module.__init__:
            objvar = self.objvar
            from ..side_effects import AttributeMutationNew

            if (
                isinstance(objvar, variables.UserDefinedObjectVariable)
                and isinstance(objvar.mutation_type, AttributeMutationNew)
                and not (args or kwargs)
            ):
                with do_not_convert_to_tracable_parameter():
                    fn_vt = VariableTracker.build(
                        tx, unpatched_nn_module_init, source=source
                    )
                    return fn_vt.call_function(tx, [self.objvar] + args, kwargs)
            else:
                unimplemented(
                    gb_type="Unsupported super().__init__() call",
                    context=f"call_method {self} {name} {args} {kwargs}",
                    explanation="Dynamo encountered a super().__init__() call "
                    f"on {objvar} that resolved to a `torch.nn.Module.__init__()` "
                    "call that we cannot trace.",
                    hints=[*graph_break_hints.DIFFICULT],
                )
        elif (
            self.objvar.source
            and hasattr(inner_fn, "__name__")
            and inner_fn.__name__ == "__new__"
            and variables.UserDefinedClassVariable.is_supported_new_method(inner_fn)
        ):
            user_cls = inner_fn.__self__
            if hasattr(user_cls, "__module__") and user_cls.__module__ == "builtins":
                user_cls_vt: VariableTracker = VariableTracker.build(tx, user_cls)
            else:
                assert source is not None
                user_cls_source = source.member
                user_cls_vt = variables.UserDefinedClassVariable(
                    user_cls, source=user_cls_source
                )
            return user_cls_vt.call_method(tx, "__new__", args, kwargs)
        elif isinstance(inner_fn, staticmethod) and isinstance(
            inner_fn.__func__, types.FunctionType
        ):
            fn_vt = VariableTracker.build(
                tx, inner_fn.__func__, source=source, realize=True
            )
            return fn_vt.call_function(tx, args, kwargs)
        elif isinstance(inner_fn, classmethod) and isinstance(
            inner_fn.__func__, types.FunctionType
        ):
            if isinstance(self.objvar, variables.UserDefinedClassVariable):
                # super().classmethod is called from a classmethod itself. So,
                # super was converted to super(__class__, cls) in bytecode and
                # therefore we have to propagate the cls.
                cls_variable = self.objvar
            else:
                # current function is an instance method, therefore super was
                # converted to super(__class__, self). We have to find
                # type(self) to bind the cls to the parent classmethod.
                # Note that it can't be the self.typevar because __class__ is
                # the class where the method is defined, which could be
                # different from type(self) with polymorphism.
                cls_source = None
                if self.objvar.source:
                    cls_source = TypeSource(self.objvar.source)
                cls_variable = VariableTracker.build(
                    tx,
                    self.objvar.value_type,  # type: ignore[attr-defined]
                    cls_source,
                )
            assert source is not None
            fn_vt = VariableTracker.build(
                tx,
                inner_fn.__func__,
                source=AttrSource(source, "__func__"),
                realize=True,
            )
            return fn_vt.call_function(tx, [cls_variable, *args], kwargs)
        elif isinstance(inner_fn, types.FunctionType):
            fn_vt = VariableTracker.build(tx, inner_fn, source=source, realize=True)
            return fn_vt.call_function(tx, [self.objvar] + args, kwargs)
        elif isinstance(inner_fn, types.MethodType):
            return variables.UserMethodVariable(
                inner_fn.__func__, self.objvar, source=source
            ).call_function(tx, args, kwargs)
        elif is_standard_setattr(inner_fn) and isinstance(
            self.objvar, UserDefinedObjectVariable
        ):
            # type: ignore[arg-type]
            return self.objvar.method_setattr_standard(tx, *args, **kwargs)
        elif inner_fn is object.__delattr__:
            attr = args[0]
            try:
                attr = attr.as_python_constant()
            except NotImplementedError as exc:
                unimplemented(
                    gb_type="Non-constant attribute given to `super().__delattr__()`",
                    context=f"call_method {self} {name}",
                    explanation="Dynamo requires the attribute name passed to "
                    "`super().__delattr__(...)` to be a constant (string).",
                    hints=[
                        "Ensure the attribute name is a string literal or a constant variable."
                    ],
                    from_exc=exc,
                )
            if not tx.output.side_effects.is_attribute_mutation(self.objvar):
                unimplemented(
                    gb_type="Attempted super().__delattr__() on an object without mutation tracking",
                    context=f"call_method {self} {name}",
                    explanation="Dynamo needs to track mutations on an object "
                    "before `super().__delattr__` can be used on it. But the "
                    f"object ({self.objvar}) doesn't have attribute mutation "
                    "tracking enabled.",
                    hints=[
                        "Ensure the object is tracked by Dynamo's side effect system.",
                        *graph_break_hints.DYNAMO_BUG,
                    ],
                )
            assert isinstance(attr, str)
            tx.output.side_effects.store_attr(
                self.objvar, attr, variables.DeletedVariable()
            )
            return variables.CONSTANT_VARIABLE_NONE
        elif (
            isinstance(self.objvar, variables.UserDefinedDictVariable)
            and inner_fn in self.objvar._dict_methods
        ):
            return self.objvar._dict_vt.call_method(tx, name, args, kwargs)
        elif (
            isinstance(self.objvar, variables.UserDefinedSetVariable)
            and inner_fn in self.objvar._set_methods
        ):
            return self.objvar._set_vt.call_method(tx, name, args, kwargs)
        elif (
            isinstance(self.objvar, variables.UserDefinedTupleVariable)
            and inner_fn in tuple_methods
        ):
            return self.objvar._tuple_vt.call_method(tx, name, args, kwargs)
        elif (
            isinstance(self.objvar, variables.UserDefinedListVariable)
            and inner_fn in list_methods
        ):
            return self.objvar._list_vt.call_method(tx, name, args, kwargs)
        elif inner_fn is object.__getattribute__:
            # object.__getattribute__ has no side-effects. We can directly call
            # __getattribute__ to access the attribute.
            attr_name = args[0].value  # type: ignore[attr-defined]
            if tx.output.side_effects.has_pending_mutation_of_attr(
                self.objvar, attr_name
            ):
                result = tx.output.side_effects.load_attr(
                    self.objvar, attr_name, deleted_ok=True
                )
                if isinstance(result, variables.DeletedVariable):
                    raise_observed_exception(AttributeError, tx)
                return result

            attr_value = None
            try:
                # NB - use object.__getattribute__ to prevent running any user code
                # type: ignore[attr-defined]
                attr_value = object.__getattribute__(self.objvar.value, attr_name)
            except AttributeError:
                raise_observed_exception(AttributeError, tx)

            attr_source = None
            if self.objvar.source is not None:
                # setup a object.__getattribute__(self.objvar, name) source
                attr_source = GenericAttrSource(self.objvar.source, attr_name)
            return VariableTracker.build(tx, attr_value, attr_source)
        elif inner_fn is torch._C._disabled_torch_function_impl:
            # See `THPModule_disable_torch_function` for the C impl.
            # The signature of _disabled_torch_function_impl is similar to
            # `__torch_function__`, just without the first `cls` argument:
            #  * (func, types, args, kwargs)
            func = args[0]
            # pyrefly: ignore [implicit-any]
            tf_kwargs = {}
            tf_args = args[2].items  # type: ignore[attr-defined]
            # type: ignore[attr-defined]
            for hash_key_vt, value_vt in args[3].items.items():
                key_str = hash_key_vt.vt.as_python_constant()
                tf_kwargs[key_str] = value_vt

            tx_old = tx.symbolic_torch_function_state.torch_function_subclass_enabled
            tx.symbolic_torch_function_state.torch_function_subclass_enabled = False
            try:
                return func.call_function(tx, tf_args, tf_kwargs)
            finally:
                tx.symbolic_torch_function_state.torch_function_subclass_enabled = (
                    tx_old
                )
        elif (
            isinstance(inner_fn, types.MethodDescriptorType)
            and inner_fn in trace_rules.get_tensor_method()
        ):
            # FunctionType but implementation is in C, we support some of these,
            # e.g., tensor ops like `torch.Tensor.to`.
            fn_var = VariableTracker.build(tx, inner_fn, source, realize=True)
            return fn_var.call_function(tx, [self.objvar] + args, kwargs)

        unimplemented(
            gb_type="Attempted to call a super() attribute that is "
            "not a function or method",
            context=f"call_method {self} {name}",
            explanation="Dynamo does not know how to trace the call "
            f"`super().{name}()` because `super().{name}` is not a "
            "function or method attribute.",
            hints=[
                "Ensure the attribute accessed via `super()` is a standard method or function.",
            ],
        )


class FrameSummaryVariable(VariableTracker):
    def __init__(self, frame_summary: traceback.FrameSummary, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.frame_summary = frame_summary

    def python_type(self) -> type:
        return traceback.FrameSummary

    def var_getattr(self, tx: "InstructionTranslator", name: str) -> VariableTracker:
        if name == "lineno":
            return VariableTracker.build(tx, self.frame_summary.lineno)
        elif name == "filename":
            return VariableTracker.build(tx, self.frame_summary.filename)
        elif name == "name":
            return VariableTracker.build(tx, self.frame_summary.name)
        elif name == "line":
            return VariableTracker.build(tx, self.frame_summary.line)
        return super().var_getattr(tx, name)


class TracebackVariable(VariableTracker):
    def __init__(
        self,
        frame_summary: FrameSummaryVariable,
        tb_next: Union["TracebackVariable", ConstantVariable],
        **kwargs: Any,
    ) -> None:
        # The traceback holds four attributes:
        #  - tb_frame
        #  - tb_lineno
        #  - tb_lasti
        #  - tb_next

        super().__init__(**kwargs)
        self.frame_summary = frame_summary
        # the next traceback in the chain
        assert tb_next is not None
        self.tb_next = tb_next

    @classmethod
    def from_frame_summary(
        cls,
        frame_summary: traceback.FrameSummary,
        tb_next: Union["TracebackVariable", ConstantVariable],
    ) -> "TracebackVariable":
        return cls(FrameSummaryVariable(frame_summary), tb_next=tb_next)

    @staticmethod
    def is_valid_traceback(obj: VariableTracker) -> bool:
        return istype(obj, TracebackVariable) or (
            istype(obj, ConstantVariable) and obj.is_constant_none()
        )

    def extract_tb(self) -> list[traceback.FrameSummary | FrameSummaryVariable]:
        if istype(self.tb_next, ConstantVariable):
            return [self.frame_summary]
        return [self.frame_summary] + self.tb_next.extract_tb()

    def has_reference_cycle(self, tb: VariableTracker) -> bool:
        # checks if `tb` is in the chain of tb_next starting from `self`
        curr_tb: TracebackVariable | ConstantVariable = self
        while istype(curr_tb, TracebackVariable):
            if curr_tb is tb:
                return True
            curr_tb = curr_tb.tb_next
        return False

    def python_type(self) -> type[types.TracebackType]:
        return types.TracebackType

    def call_setattr(
        self,
        tx: "InstructionTranslator",
        name_var: VariableTracker,
        val: VariableTracker,
    ) -> VariableTracker:
        name = name_var.as_python_constant()
        if name == "tb_next":
            if not self.is_valid_traceback(val):
                raise_observed_exception(TypeError, tx)
            assert isinstance(val, (TracebackVariable, ConstantVariable))
            if self.has_reference_cycle(val) or (
                istype(val, TracebackVariable) and val.has_reference_cycle(self)
            ):
                raise_observed_exception(ValueError, tx)
            self.tb_next = val
        return variables.CONSTANT_VARIABLE_NONE

    def var_getattr(self, tx: "InstructionTranslator", name: str) -> VariableTracker:
        if name == "tb_next":
            return self.tb_next
        elif name == "tb_lineno":
            return self.frame_summary.var_getattr(tx, "lineno")
        elif name == "frame_summary":
            return self.frame_summary
        elif name == "tb_lasti":
            unimplemented(
                gb_type="traceback.tb_lasti not supported",
                context=f"{self} accessing 'tb_lasti'",
                explanation="Dynamo does not support accessing the tb_lasti attribute of traceback objects.",
                hints=[*graph_break_hints.SUPPORTABLE],
            )
        return super().var_getattr(tx, name)

    def call_method(
        self,
        tx: "InstructionTranslator",
        name: str,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        if name == "__eq__":
            # Two traceback variables are only equal if they are the same object
            return VariableTracker.build(tx, self is args[0])
        elif name == "__setattr__":
            return self.call_setattr(tx, *args)
        return super().call_method(tx, name, args, kwargs)


class ExceptionVariable(VariableTracker):
    # The ExceptionVariable corresponds to the BaseException class in Python
    def __init__(
        self,
        exc_type: Any,
        args: tuple[VariableTracker, ...],
        init_kwargs: dict[str, VariableTracker] | None = None,
        source: Source | None = None,
        mutation_type: MutationType | None = None,
    ) -> None:
        super().__init__(source=source, mutation_type=mutation_type)
        self.exc_type = exc_type
        self.args = args
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
        self.__context__: VariableTracker = CONSTANT_VARIABLE_NONE
        # Set when user raised an exception from another:
        # raise ... from ...
        self.__cause__: VariableTracker = CONSTANT_VARIABLE_NONE
        # Boolean flag that controls whether the __context__ attribute is set
        self.__suppress_context__: VariableTracker = CONSTANT_VARIABLE_FALSE
        # Contains the call stack where the exception was raised.
        self.__traceback__: VariableTracker = CONSTANT_VARIABLE_NONE
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
                assert attr.value in (True, False, None), attr
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

    def call_setattr(
        self,
        tx: "InstructionTranslator",
        name_var: VariableTracker,
        val: VariableTracker,
    ) -> VariableTracker:
        def raise_error(msg: str) -> NoReturn:
            raise_observed_exception(
                TypeError, tx, args=[VariableTracker.build(tx, msg)]
            )

        name = name_var.as_python_constant()
        if name == "__context__":
            # Constant can be either an Exceptior or None
            assert isinstance(val, (ExceptionVariable, ConstantVariable))
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
                self.__suppress_context__ = variables.CONSTANT_VARIABLE_TRUE
            else:
                raise_error("exception cause must be None or derive from BaseException")
        elif name == "__suppress_context__":
            if val.is_constant_match(True, False):
                self.__suppress_context__ = val
            else:
                raise_error("exception cause must be None or derive from BaseException")
        elif name == "__traceback__":
            if not TracebackVariable.is_valid_traceback(val):
                raise_observed_exception(
                    TypeError,
                    tx,
                    args=[
                        VariableTracker.build(
                            tx, "__traceback__ must be a traceback object or None"
                        )
                    ],
                )
            self.__traceback__ = val
        else:
            unimplemented(
                gb_type="Unsupported attribute assignment on Exception object",
                context=f"call_setattr {self} {name}",
                explanation="Dynamo does not support setting the attribute "
                f"'{name}' on tracked exception objects. Only `__context__`, "
                "`__cause__`, `__suppress_context__`, and `__traceback__` are supported.",
                hints=[*graph_break_hints.SUPPORTABLE],
            )
        return variables.CONSTANT_VARIABLE_NONE

    def call_method(
        self,
        tx: "InstructionTranslator",
        name: str,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        if name == "__setattr__":
            return self.call_setattr(tx, *args)
        elif name == "with_traceback":
            [tb] = args
            self.call_setattr(tx, VariableTracker.build(tx, "__traceback__"), tb)
            return self
        else:
            return super().call_method(tx, name, args, kwargs)

    def var_getattr(self, tx: "InstructionTranslator", name: str) -> VariableTracker:
        if name == "__context__":
            return self.__context__
        elif name == "__cause__":
            return self.__cause__
        elif name == "__suppress_context__":
            return self.__suppress_context__
        elif name == "__traceback__":
            return self.__traceback__
        elif name == "args":
            return variables.ListVariable(list(self.args), source=self.source)
        return super().var_getattr(tx, name)

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.exc_type})"

    __repr__ = __str__


class UnknownVariable(VariableTracker):
    """
    It could be anything!
    """


class DelayGraphBreakVariable(UnknownVariable):
    """
    Used to insert a dummy variable in the stack to do the graph break at CALL_FUNCTION.
    """

    def __init__(self, msg: str | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.msg = msg

    def call_function(
        self,
        tx: "InstructionTranslator",
        args: Sequence[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        name = "" if self.source is None else self.source.name
        unimplemented(
            gb_type="Unsupported function call (delayed)",
            context=f"source: {self.source}",
            explanation="Dynamo determined that a graph break should occur "
            f"when calling `{name}`. Reason: {self.msg}",
            hints=[],
        )


class ComptimeVariable(VariableTracker):
    """
    This variable is special, it lets you execute arbitrary code at
    Dynamo compile time
    """

    def reconstruct(self, codegen: "PyCodegen") -> None:
        raise NotImplementedError("comptime is special form")

    def var_getattr(self, tx: "InstructionTranslator", name: str) -> VariableTracker:
        from ..comptime import comptime

        assert self.source is not None
        # To support the comptime.print_graph convenience accessors
        return VariableTracker.build(
            tx, getattr(comptime, name), source=AttrSource(self.source, name)
        )

    def call_function(
        self,
        tx: "InstructionTranslator",
        args: Sequence[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        from ..comptime import ComptimeContext

        # TODO: support an expression form as well
        # Second argument is runtime lambda, ignored
        if kwargs or len(args) > 2:
            raise_args_mismatch(
                tx,
                "comptime()",
                "at most 2 args and 0 kwargs",
                f"{len(args)} args and {len(kwargs)} kwargs",
            )
        fn = args[0]
        if isinstance(fn, UserFunctionVariable):
            fn.get_function()(ComptimeContext(tx))
        elif isinstance(fn, NestedUserFunctionVariable):
            # We have to manually bind the freevars ourselves
            code = fn.get_code()
            if fn.closure:
                raise_type_error_exc(
                    tx,
                    f"comptime function must not have free variables, but these variables were free: {code.co_freevars}",
                )
            func = types.FunctionType(
                code,
                fn.f_globals,
                fn.fn_name.as_python_constant(),
                # type: ignore[attr-defined]
                tuple(fn.defaults.items) if fn.defaults else None,
                # We could automatically promote free variables into
                # ComptimeVar but this is confusing if you access
                # a free variable that we actually DO have the runtime
                # value for
                # tuple(make_cell(ComptimeVar(i)) for i in fn.closure.items)
                (),
            )
            func(ComptimeContext(tx))
        else:
            raise RuntimeError(f"unsupported argument to comptime: {type(fn)}")

        return variables.CONSTANT_VARIABLE_NONE


class CellVariable(VariableTracker):
    # If the cell existed before Dynamo tracing started, this will be the
    # VariableTracker that represents the cell content.
    #
    # Note that all mutation to the cell (i.e., its content) will be buffered in
    # SideEffects, rather than being reflected here. One can think of
    # `CellVariable` as a special case for `UserDefinedObjectVariable`.
    pre_existing_contents: VariableTracker | None

    # This is set when this cell can be referenced via `LOAD/STORE_DEREF` in the
    # root frame via this name (e.g., the name is in `co_cellvars/co_freevars`).
    local_name: str | None = None

    def __init__(
        self, pre_existing_contents: VariableTracker | None = None, **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.pre_existing_contents = pre_existing_contents


class NewGlobalVariable(VariableTracker):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)


def produce_trampoline_autograd_apply(fn_cls: Any) -> Callable[..., Any]:
    def trampoline_autograd_apply(*args: Any, **kwargs: Any) -> Any:
        return fn_cls.apply(*args, **kwargs)

    # type: ignore[attr-defined]
    trampoline_autograd_apply._origin = produce_trampoline_autograd_apply
    return trampoline_autograd_apply


class AutogradFunctionVariable(VariableTracker):
    """represents a torch.autograd.Function subclass"""

    _nonvar_fields = {
        "fn_cls",
        *VariableTracker._nonvar_fields,
    }

    def __init__(self, fn_cls: Any, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.fn_cls = fn_cls

    def call_apply(
        self,
        tx: "InstructionTranslator",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        requires_grad = False

        def visit(vt: VariableTracker) -> None:
            nonlocal requires_grad
            if vt.is_tensor():
                # type: ignore[attr-defined]
                if vt.requires_grad is not False:
                    requires_grad = True
            if isinstance(vt, variables.NNModuleVariable):
                if vt.is_training(tx):
                    requires_grad = True

        VariableTracker.visit(visit, (args, kwargs))

        if requires_grad and torch.is_grad_enabled():
            source = self.source

            from torch._functorch.autograd_function import (
                autograd_function_forward_rewritten,
            )
            from torch.autograd.function import _is_setup_context_defined

            forward_fn = self.fn_cls.forward

            is_setup_ctx_defined = _is_setup_context_defined(self.fn_cls.setup_context)
            if is_setup_ctx_defined:
                # If setup_context is defined, we generate a new forward function which includes
                # the original forward and setup_context function, and trace the new forward function.
                forward_fn = autograd_function_forward_rewritten(
                    self.fn_cls.forward, self.fn_cls.setup_context
                )
                # The forward points to a new function now, so we can't use the
                # old source. Later on, we guard specifically on
                # is_setup_ctx_defined
                source = None

            vjp_fn = self.fn_cls.vjp  # type: ignore[attr-defined]
            if vjp_fn is not torch.autograd.Function.vjp:
                unimplemented(
                    gb_type="Unsupported custom vjp",
                    context=f"call_apply {self} {args} {kwargs}",
                    explanation="Dynamo does not support tracing "
                    "`torch.autograd.Function` subclasses that define "
                    "a custom `vjp` method.",
                    hints=[
                        "Remove the custom `vjp` method if possible.",
                        "Use standard `backward` instead if applicable.",
                        *graph_break_hints.SUPPORTABLE,
                    ],
                )

            jvp_fn = self.fn_cls.jvp  # type: ignore[attr-defined]
            if jvp_fn is not torch.autograd.Function.jvp:
                unimplemented(
                    gb_type="Unsupported custom jvp",
                    context=f"call_apply {self} {args} {kwargs}",
                    explanation="Dynamo does not support tracing "
                    "`torch.autograd.Function` subclasses that define "
                    "a custom `jvp` method.",
                    hints=[
                        "Remove the custom `jvp` method if possible.",
                        *graph_break_hints.SUPPORTABLE,
                    ],
                )

            from .higher_order_ops import AutogradFunctionApplyVariable

            if source is None and not is_setup_ctx_defined:
                source = AttrSource(
                    tx.import_source(self.fn_cls.__module__), self.fn_cls.__name__
                )
            apply_source = source and AttrSource(source, member="apply")
            val = AutogradFunctionApplyVariable(
                forward_fn,
                self.fn_cls.backward,
                source,
                source=apply_source,
            ).call_function(tx, args, kwargs)
            if self.source and is_setup_ctx_defined:
                fwd_src = AttrSource(self.source, "forward")
                install_guard(fwd_src.make_guard(GuardBuilder.CLOSURE_MATCH))
                setup_ctx_src = AttrSource(self.source, "setup_context")
                install_guard(setup_ctx_src.make_guard(GuardBuilder.CLOSURE_MATCH))

            return val

        if self.source:
            source = AttrSource(self.source, "forward")
        else:
            source = None

        fn = self.fn_cls.forward
        ctx = AutogradFunctionContextVariable.create(tx, args, kwargs)
        args = [ctx, *args]
        if isinstance(fn, types.FunctionType):
            sig = inspect.signature(fn)
            if len(args) - 1 == len(sig.parameters):
                args = args[1:]  # Don't use context
            fn_vt = VariableTracker.build(tx, fn, source=source, realize=True)
            return fn_vt.call_function(tx, args, kwargs)
        elif isinstance(fn, types.MethodType):
            return variables.UserMethodVariable(
                fn.__func__,
                variables.UserDefinedClassVariable(self.fn_cls),
                source=source,
            ).call_function(tx, args, kwargs)
        else:
            unimplemented(
                gb_type="Non-function or method in subclass of torch.autograd.Function",
                context=f"call_apply {self} {args} {kwargs}",
                explanation="Dynamo requires the `forward` attribute of a "
                "`torch.autograd.Function` subclass to be a standard Python "
                f"function or method. Found type `{type(fn).__name__}` instead.",
                hints=[
                    "Ensure the `forward` method is defined as a regular "
                    "function or instance method."
                ],
            )

    def call_backward(
        self,
        tx: "InstructionTranslator",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        fn = self.fn_cls.backward
        # type: ignore[attr-defined]
        assert type(args[0].value) is torch._dynamo.external_utils.FakeBackwardCFunction
        assert isinstance(fn, types.FunctionType)
        assert self.source is not None
        fn_source = AttrSource(self.source, "backward")
        fn_vt = VariableTracker.build(tx, fn, source=fn_source, realize=True)
        return fn_vt.call_function(tx, args, kwargs)

    def call_function(
        self,
        tx: "InstructionTranslator",
        args: Sequence[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> "AutogradFunctionVariable":
        return AutogradFunctionVariable(self.fn_cls)

    def call_method(
        self,
        tx: "InstructionTranslator",
        name: str,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        from .builder import wrap_fx_proxy

        if name == "apply":
            if trace_rules.is_callable_allowed(self.fn_cls):
                trampoline_autograd_apply = produce_trampoline_autograd_apply(
                    self.fn_cls
                )
                return wrap_fx_proxy(
                    tx=tx,
                    proxy=tx.output.create_proxy(
                        "call_function",
                        trampoline_autograd_apply,
                        *proxy_args_kwargs(args, kwargs),
                    ),
                )
            else:
                return self.call_apply(tx, args, kwargs)

        elif name == "backward":
            return self.call_backward(tx, args, kwargs)
        else:
            source = AttrSource(self.source, name) if self.source is not None else None
            try:
                obj = inspect.getattr_static(self.fn_cls, name)
            except AttributeError:
                obj = None

            if isinstance(obj, staticmethod):
                func = obj.__get__(self.fn_cls)
                traced = trace_rules.lookup(func)
                assert traced is not None
                if source is not None:
                    return (
                        # type: ignore[attr-defined]
                        traced.create_with_source(func, source=source).call_function(
                            tx, args, kwargs
                        )
                    )
                else:
                    # type: ignore[misc]
                    return traced(func).call_function(tx, args, kwargs)
            elif isinstance(obj, classmethod):
                return variables.UserMethodVariable(
                    obj.__func__, self, source=source
                ).call_function(tx, args, kwargs)
            else:
                unimplemented(
                    gb_type="Unsupported autograd.Function method",
                    context=f"call_method {self} {name}",
                    explanation="Dynamo does not support calling the method "
                    f"`{name}` directly on the `torch.autograd.Function` "
                    "instance. Supported methods include `apply`, `backward`, "
                    "static methods, and class methods.",
                    hints=[
                        "Ensure the method is decorated with `@staticmethod` "
                        "or `@classmethod` if it's meant to be called on the class.",
                    ],
                )


@dataclasses.dataclass
class SavedTensorBox:
    tensors: list[VariableTracker] = dataclasses.field(default_factory=list)


class AutogradFunctionContextVariable(UserDefinedObjectVariable):
    """
    Tracks an autograd.Function() context using mutation tracking in side_effects.py
    """

    _nonvar_fields = {
        "proxy",
        "inference",
        "saved_tensors",
        *UserDefinedObjectVariable._nonvar_fields,
    }

    def __init__(
        self,
        value: Any,
        value_type: type | None = None,
        inference: bool = False,
        saved_tensors: Any | None = None,
        needs_input_grad: tuple[bool, ...] | None = None,
        non_differentiable: Any | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(value=value, value_type=value_type, **kwargs)
        self.inference = inference
        self.saved_tensors = saved_tensors
        self.needs_input_grad = needs_input_grad
        self.non_differentiable = non_differentiable

    @staticmethod
    def create(
        tx: "InstructionTranslator",
        args: Sequence[VariableTracker] | None = None,
        kwargs: dict[str, VariableTracker] | None = None,
    ) -> VariableTracker:
        needs_input_grad = None
        if args and not kwargs:
            # type: ignore[attr-defined]
            needs_input_grad = tuple(x.is_tensor() and x.requires_grad for x in args)
        out = tx.output.side_effects.track_object_new(
            None,
            torch.autograd.function.FunctionCtx,
            functools.partial(
                AutogradFunctionContextVariable,
                inference=True,
                saved_tensors=SavedTensorBox(),
                needs_input_grad=needs_input_grad,
            ),
            {},
        )
        return out

    def as_proxy(self) -> Any:
        # type: ignore[attr-defined]
        if self.proxy is None:
            unimplemented(
                gb_type="proxy not set",
                context=f"as_proxy {self}",
                explanation="Dynamo requires the autograd.Function context "
                "to be initialized with a proxy.",
                hints=[*graph_break_hints.DYNAMO_BUG],
            )
        # type: ignore[attr-defined]
        return self.proxy

    def call_method(
        self,
        tx: "InstructionTranslator",
        name: str,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        if name == "__setattr__":
            return super().call_method(tx, name, args, kwargs)
        elif name == "mark_non_differentiable":
            if kwargs:
                raise_args_mismatch(tx, name, "0 kwargs", f"{len(kwargs)} kwargs")
            self.non_differentiable = proxy_args_kwargs(args, {})[0]
            return variables.CONSTANT_VARIABLE_NONE

        if name != "save_for_backward":
            unimplemented(
                gb_type="Unsupported autograd.Function context method",
                context=f"call_method {self} {name}",
                explanation="Dynamo does not support calling the method "
                f"`{name}` on `autograd.Function` context objects. Supported "
                "methods are `__setattr__`, `save_for_backward` and "
                "`mark_non_differentiable`.",
                hints=[*graph_break_hints.SUPPORTABLE],
            )
        if self.saved_tensors is None:
            unimplemented(
                gb_type="Unsupported autograd.Function context `save_for_backward`",
                context=f"call_method {self} {name}",
                explanation="Dynamo requires the `saved_tensors` attribute "
                "to be initialized on the `autograd.Function` context object.",
                hints=[
                    "Ensure that the `saved_tensors` attribute is properly "
                    "initialized before calling `save_for_backward`. "
                    "`save_for_backward` only supported on a newly constructed `torch.autograd.function.FunctionCtx`.",
                ],
            )
        assert self.saved_tensors is not None
        if not self.inference:
            if kwargs or not self.source:
                raise_type_error_exc(
                    tx, "save_for_backward() requires a source and no keyword arguments"
                )
            tx.output.side_effects.track_save_for_backward(self, args)

        # In eager mode, multiple calls to .save_for_backward() will overwrite previous calls.
        if len(self.saved_tensors.tensors) > 0:
            # pyrefly: ignore [implicit-any]
            self.saved_tensors.tensors = []
        for arg in args:
            self.saved_tensors.tensors.append(arg)
        return variables.CONSTANT_VARIABLE_NONE

    def var_getattr(self, tx: "InstructionTranslator", name: str) -> VariableTracker:
        if name in ["save_for_backward", "mark_non_differentiable"]:
            return LambdaVariable(
                lambda *args, **kwargs: self.call_method(tx, name, list(args), kwargs)
            )
        if name == "saved_tensors" and self.saved_tensors is not None:
            return variables.TupleVariable(list(self.saved_tensors.tensors))
        if name == "needs_input_grad":
            if self.needs_input_grad is not None:
                return variables.ConstantVariable.create(self.needs_input_grad)
            if self.source:
                source = AttrSource(self.source, "needs_input_grad")
                # type: ignore[attr-defined]
                return VariableTracker.build(tx, self.value.needs_input_grad, source)

        return super().var_getattr(tx, name)


class AutogradEngineVariable(UserDefinedObjectVariable):
    """
    Represents a torch._C._ImperativeEngine instance.
    """

    def __init__(
        self,
        value: torch._C._ImperativeEngine,
        value_type: type[torch._C._ImperativeEngine] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(value=value, value_type=value_type, **kwargs)

    def call_method(
        self,
        tx: "InstructionTranslator",
        name: str,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        if name == "queue_callback":
            if torch._dynamo.compiled_autograd.in_compiled_autograd_region:
                assert tx.one_graph or tx.error_on_graph_break, (
                    "queue_callback() is only supported when Compiled Autograd is enabled with fullgraph=True"
                )
                # queue_callback is a method-wrapper, no need to insert a guard.
                fn_vt = VariableTracker.build(
                    tx,
                    torch._dynamo.external_utils.FakeCompiledAutogradEngine.queue_callback,
                )
                return fn_vt.call_function(
                    tx,
                    (tx.output.side_effects.get_ca_final_callbacks_var(), *args),
                    kwargs,
                )
            else:
                unimplemented(
                    gb_type="Unsupported torch._C._ImperativeEngine.queue_callback()",
                    context=f"call_method {self} {name}",
                    explanation="queue_callback() is only supported when "
                    "Compiled Autograd is enabled with fullgraph=True.",
                    hints=[],
                )
        else:
            unimplemented(
                gb_type="Unsupported torch._C._ImperativeEngine method",
                context=f"call_method {self} {name}",
                explanation="Dynamo only supports the `queue_callback` method "
                f"on a torch._C._ImperativeEngine instance, but found: `{name}`.",
                hints=[],
            )


class LambdaVariable(VariableTracker):
    # TODO: change to Ts = TypeVarTuple("Ts") for py 3.11+
    def __init__(self, fn: Callable[..., VariableTracker], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.fn = fn

    def call_function(
        self,
        tx: "InstructionTranslator",
        args: Sequence[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        return self.fn(*args, **kwargs)


class GetAttrVariable(VariableTracker):
    _nonvar_fields = {
        "name",
        "py_type",
        *VariableTracker._nonvar_fields,
    }

    def __init__(
        self,
        obj: VariableTracker,
        name: str,
        py_type: type | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        assert isinstance(obj, VariableTracker)
        assert isinstance(name, str)
        self.obj = obj
        self.name = name
        self.py_type = py_type  # In some cases we know the type (ex. tensor methods)

    def python_type(self) -> type:
        if self.py_type is not None:
            return self.py_type
        else:
            return super().python_type()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.obj}, {self.name})"

    @staticmethod
    def create_getattr_proxy(base_proxy: torch.fx.Proxy, attr: str) -> Any:
        return getattr(base_proxy, attr)

    def as_proxy(self) -> Any:
        return GetAttrVariable.create_getattr_proxy(self.obj.as_proxy(), self.name)

    def as_python_constant(self) -> Any:
        constant = self.obj.as_python_constant()
        try:
            return getattr(constant, self.name)
        except AttributeError:
            raise NotImplementedError(f"{self} is not a constant") from None

    def const_getattr(self, tx: "InstructionTranslator", name: str) -> Any:
        if not isinstance(self.obj, variables.NNModuleVariable):
            raise NotImplementedError
        step1 = tx.output.get_submodule(self.obj.module_key)
        if self.name not in step1.__dict__:
            raise NotImplementedError
        step2 = inspect.getattr_static(step1, self.name)
        if name not in step2.__dict__:
            raise NotImplementedError
        return inspect.getattr_static(step2, name)

    def reconstruct(self, codegen: "PyCodegen") -> None:
        codegen(self.obj)
        codegen.extend_output(codegen.create_load_attrs(self.name))

    def call_function(
        self,
        tx: "InstructionTranslator",
        args: Sequence[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        return self.obj.call_method(tx, self.name, list(args), kwargs)

    def call_method(
        self,
        tx: "InstructionTranslator",
        name: str,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        if (
            name in ("__getitem__", "get")
            and self.name == "__dict__"
            and not kwargs
            and args[0].is_python_constant()
            and isinstance(
                self.obj,
                (
                    variables.UserDefinedObjectVariable,
                    variables.NNModuleVariable,
                    variables.UserDefinedClassVariable,
                ),
            )
        ):
            obj = self.obj
            key = args[0].as_python_constant()
            if obj.has_key_in_generic_dict(tx, key):
                if tx.output.side_effects.has_pending_mutation_of_attr(obj, key):
                    return tx.output.side_effects.load_attr(obj, key)

                # For instance dicts, read directly from __dict__
                if isinstance(obj.value.__dict__, dict):
                    raw_value = obj.value.__dict__[key]
                    raw_source = (
                        DictGetItemSource(AttrSource(obj.source, "__dict__"), key)
                        if obj.source
                        else None
                    )
                    return VariableTracker.build(tx, raw_value, raw_source)

                return obj.var_getattr(tx, key)

            # Return the default value for get
            if name == "get":
                if len(args) == 2:
                    return args[1]
                else:
                    return variables.CONSTANT_VARIABLE_NONE

        elif (
            name == "__contains__"
            and self.name == "__dict__"
            and len(args) == 1
            and args[0].is_python_constant()
            and not kwargs
            and isinstance(
                self.obj,
                (
                    variables.UserDefinedObjectVariable,
                    variables.NNModuleVariable,
                    variables.UserDefinedClassVariable,
                ),
            )
        ):
            obj = self.obj
            key = args[0].as_python_constant()
            if obj.has_key_in_generic_dict(tx, key):
                return variables.CONSTANT_VARIABLE_TRUE
            else:
                return variables.CONSTANT_VARIABLE_FALSE

        elif name == "__setitem__" and self.name == "__dict__" and not kwargs:
            if isinstance(self.obj, variables.UserDefinedObjectVariable):
                # Bypass any custom setattr as we are updating the `__dict__` itself
                return self.obj.method_setattr_standard(
                    tx, args[0], args[1], directly_update_dict=True
                )
            if isinstance(self.obj, variables.NNModuleVariable):
                # This matches how `setattr` is handled for NNModuleVariable
                self.obj.convert_to_unspecialized(tx)

        return super().call_method(tx, name, args, kwargs)

    def get_forwarded_dict(self, tx: "InstructionTranslator") -> VariableTracker:
        assert (
            self.name == "__dict__"
            and isinstance(self.obj, variables.UserDefinedClassVariable)
            and not tx.output.side_effects.has_pending_mutation(self.obj)
        )
        self.obj.ban_mutation = True
        return VariableTracker.build(tx, self.obj.value.__dict__, self.source)


class MethodWrapperVariable(VariableTracker):
    def __init__(self, method_wrapper: types.MethodWrapperType, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.method_wrapper = method_wrapper

    def call_function(
        self,
        tx: "InstructionTranslator",
        args: Sequence[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        if is_tensor_base_attr_getter(self.method_wrapper) and isinstance(
            args[0], variables.TensorVariable
        ):
            if not (len(args) == 1 and len(kwargs) == 0):
                raise_type_error_exc(
                    tx, "tensor attribute getter takes exactly one argument"
                )
            # type: ignore[arg-type, attr-defined]
            return args[0].var_getattr(tx, self.method_wrapper.__self__.__name__)

        # method-wrapper variables are common in __init__ calls. For example,
        # str("foo").__init__ is a method-wrapper. These method wrappers point
        # to C functions.  Here we intercept if these method-wrappers are from
        # builtins and then call the function counterpart directly by obtaining
        # the self object.
        self_obj = self.method_wrapper.__self__
        wrapper_name = self.method_wrapper.__name__
        # TODO(dynamo-team) - We can perhaps expand the scope to more names and
        # more builtins.
        if wrapper_name == "__init__":
            fn_obj = type(self_obj).__init__
            if fn_obj is object.__init__:
                return VariableTracker.build(tx, object).call_method(
                    tx,
                    wrapper_name,
                    # type: ignore[arg-type, list-item]
                    [self_obj, *args],
                    kwargs,
                )
        elif (
            sys.version_info >= (3, 14)
            # for some reason, even if the below check passes,
            # self.method_wrapper may not be the same as type.__dict__["__annotations__"].__get__
            and self_obj is type.__dict__["__annotations__"]
            and wrapper_name == "__get__"
        ):
            from .builder import SourcelessBuilder

            if len(args) == 1 and not kwargs:
                try:
                    return SourcelessBuilder.create(
                        tx, self.method_wrapper(args[0].as_python_constant())
                    )
                except AttributeError:
                    raise_observed_exception(AttributeError, tx)
                except AsPythonConstantNotImplementedError:
                    pass

            unimplemented(
                gb_type="unsupported type.__dict__['__annotations__'].__get__ call",
                context=f"call_function {self}, args: {args}, kwargs: {kwargs}",
                explanation="`torch.compile` only supports calling type.__dict__['__annotations__'].__get__ "
                "on a single constant argument (i.e. a type).",
                hints=[
                    "Make sure your call to type.__dict__['__annotations__'] only has "
                    "one positional argument (no keyword arguments).",
                    "Make sure the argument to type.__dict__['__annotations__'] is a constant "
                    "(i.e. type). For example, `object`, `int`, `MyCustomClass`.",
                    *graph_break_hints.SUPPORTABLE,
                ],
            )
        elif (self_obj is type.__dict__["__mro__"] and wrapper_name == "__get__") or (
            self_obj is type.__dict__["__dict__"] and wrapper_name == "__get__"
        ):
            attr_name = (
                "__mro__" if self_obj is type.__dict__["__mro__"] else "__dict__"
            )

            if len(args) == 1 and not kwargs:
                try:
                    value = self.method_wrapper(args[0].as_python_constant())
                except AsPythonConstantNotImplementedError:
                    pass
                else:
                    # Use a sourced variable when the descriptor is the
                    # standard one from type (not overridden by a metaclass).
                    source = args[0].source
                    if source is not None:
                        cls_val = args[0].as_python_constant()
                        static_desc = inspect.getattr_static(type(cls_val), attr_name)
                        if static_desc is self_obj:
                            if attr_name == "__mro__":
                                source = TypeMROSource(source)
                            else:
                                source = AttrSource(source, attr_name)
                            return VariableTracker.build(tx, value, source)

                    from .builder import SourcelessBuilder

                    return SourcelessBuilder.create(tx, value)

            unimplemented(
                gb_type=f"unsupported type.__dict__['{attr_name}'].__get__ call",
                context=f"call_function {self}, args: {args}, kwargs: {kwargs}",
                explanation=f"`torch.compile` only supports calling type.__dict__['{attr_name}'].__get__ "
                "on a single constant argument (i.e. a type).",
                hints=[
                    f"Make sure your call to type.__dict__['{attr_name}'].__get__ only has "
                    "one positional argument (no keyword arguments).",
                    f"Make sure the argument to type.__dict__['{attr_name}'].__get__ is a constant "
                    "(i.e. type). For example, `object`, `int`, `MyCustomClass`.",
                    *graph_break_hints.SUPPORTABLE,
                ],
            )

        return super().call_function(tx, args, kwargs)

    def is_python_constant(self) -> Literal[True]:
        return True

    def as_python_constant(self) -> types.MethodWrapperType:
        return self.method_wrapper

    def is_python_hashable(self) -> Literal[True]:
        return True

    def get_python_hash(self) -> int:
        return hash(self.as_python_constant())

    def is_python_equal(self, other: object) -> bool:
        return (
            isinstance(other, VariableTracker)
            and self.as_python_constant() == other.as_python_constant()
        )


class GetSetDescriptorVariable(VariableTracker):
    def __init__(self, desc: types.GetSetDescriptorType, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.desc = desc

    def var_getattr(self, tx: "InstructionTranslator", name: str) -> VariableTracker:
        if name == "__get__" and self.source:
            source = AttrSource(self.source, "__get__")
            return VariableTracker.build(tx, self.desc.__get__, source)
        elif name in ("__objclass__", "__name__"):
            source = self.source and AttrSource(self.source, name)
            return VariableTracker.build(tx, getattr(self.desc, name), source)
        else:
            return super().var_getattr(tx, name)

    def is_python_constant(self) -> Literal[True]:
        return True

    def as_python_constant(self) -> types.GetSetDescriptorType:
        return self.desc


class PythonModuleVariable(VariableTracker):
    _nonvar_fields = {
        "value",
        "is_torch",
        *VariableTracker._nonvar_fields,
    }

    def __init__(self, value: types.ModuleType, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.value = value
        self.is_torch = self.value is torch or self.value.__name__.startswith("torch.")

    def python_type(self) -> type[types.ModuleType]:
        return types.ModuleType

    def as_python_constant(self) -> types.ModuleType:
        return self.value

    def __repr__(self) -> str:
        return f"PythonModuleVariable({self.value})"

    def call_obj_hasattr(
        self, tx: "InstructionTranslator", name: str
    ) -> "ConstantVariable":
        result = hasattr(self.value, name)
        return VariableTracker.build(tx, result)

    def var_getattr(self, tx: "InstructionTranslator", name: str) -> VariableTracker:
        if tx.output.side_effects.has_pending_mutation_of_attr(self, name):
            return tx.output.side_effects.load_attr(self, name)

        attr_value = None
        if self.is_torch or name not in self.value.__dict__:
            try:
                attr_value = getattr(self.value, name)
            except AttributeError:
                raise_observed_exception(AttributeError, tx)
        else:
            attr_value = self.value.__dict__[name]

        source = self.source and AttrSource(self.source, name)
        return VariableTracker.build(tx, attr_value, source)


class TypingVariable(VariableTracker):
    def __init__(self, value: Any, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.value = value

    def call_method(
        self,
        tx: "InstructionTranslator",
        name: str,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # Create a new typing variable, e.g., `List[int]`
        if name == "__getitem__" and len(args) == 1:
            new_typing = self.value[args[0].as_python_constant()]
            return TypingVariable(new_typing)
        elif name == "__eq__":
            if len(args) == 1 and not kwargs:
                result = istype(args[0], TypingVariable) and self.value == args[0].value
                return variables.ConstantVariable.create(result)
        unimplemented(
            gb_type="unsupported method call on `typing` variable",
            context=f"typing variable: {self.value}, method name: {name}, args: {args}, kwargs: {kwargs}",
            explanation=f"`torch.compile` does not support method call `{name}` on `typing` variable f{self.value}.",
            hints=[
                f"Avoid calling the {name} method on {self.value}.",
                *graph_break_hints.SUPPORTABLE,
            ],
        )

    def var_getattr(self, tx: "InstructionTranslator", name: str) -> VariableTracker:
        from .builder import SourcelessBuilder, VariableBuilder

        if name in cmp_name_to_op_mapping:
            return variables.GetAttrVariable(self, name)

        if tx.output.side_effects.has_pending_mutation_of_attr(self, name):
            return tx.output.side_effects.load_attr(self, name)

        value = getattr(self.value, name)
        if self.source:
            attr_source = AttrSource(self.source, name)
            return VariableBuilder(tx, attr_source)(value)
        else:
            return SourcelessBuilder.create(tx, value)

    def as_python_constant(self) -> Any:
        return self.value

    def reconstruct(self, codegen: "PyCodegen") -> None:
        if not isinstance(self.value, types.GenericAlias):
            return super().reconstruct(codegen)
        # We're just trying to load the type here. Reconstructing the type from
        # scratch is tricky - for a type like `typing.List[int]` we'd need to
        # deconstruct the origin and args.  The origin for `List[int]` is `list`
        # and the args is `(int,)`. When we recombine those we get the parts
        # back and need to emit code for:
        #
        #     `typing.List[int]`
        #
        # But it's # worse than that - what if `typing` isn't in the globals (or
        # was loaded like `import typing as _typing ; _typing.List[int]`?) so we
        # really need to do something like:
        #
        #   `sys.modules["typing"].List[int]`
        #
        # Argh - but what if they rewrote the global `int`?  So we have to do:
        #
        #   `sys.modules["typing"].List[sys.modules["builtins"].int]`
        #
        # But where do we get `sys`? What if they never imported it or have
        # something ELSE called `sys`?
        #
        # Let's skip all that noise and just emit it as a simple const.
        #
        codegen.append_output(codegen.create_load_const(self.value))

    def is_python_hashable(self) -> Literal[True]:
        return True

    def get_python_hash(self) -> int:
        return hash(self.as_python_constant())

    def is_python_equal(self, other: object) -> bool:
        return (
            isinstance(other, VariableTracker)
            and self.as_python_constant() == other.as_python_constant()
        )


@functools.lru_cache(maxsize=1)
def get_np_to_tnp_map() -> dict[types.BuiltinFunctionType, types.FunctionType]:
    """
    This generates a mapping from numpy modules to their torch._numpy
    modules equivalents.
    """
    from ..utils import NP_TO_TNP_MODULE

    np_fn_to_tnp_fn = {}

    for np_mod, tnp_mod in NP_TO_TNP_MODULE.items():
        for fn_name, tnp_fn in tnp_mod.__dict__.items():
            if callable(tnp_fn):
                # some internal details do leak from tnp
                # which are not part of numpy API.
                if np_fn := getattr(np_mod, fn_name, None):
                    np_fn_to_tnp_fn[np_fn] = tnp_fn

    return np_fn_to_tnp_fn


@functools.lru_cache(maxsize=1)
def get_tnp_to_np_map() -> dict[types.FunctionType, types.BuiltinFunctionType]:
    """
    This is just the reverse mapping of get_np_to_tnp_map() - mapping from
    torch._numpy modules to numpy equivalents.
    """
    m = get_np_to_tnp_map()
    return {v: k for k, v in m.items()}


class NumpyVariable(VariableTracker):
    """
    Wrapper around `numpy.*`. Currently, is able to trace a small subset of numpy functions as well as numpy dtypes.
    """

    constant_fold_functions = (tnp.issubdtype,)

    def __init__(self, value: Any, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.value = value

    @classmethod
    def can_constant_fold_through(cls, fn: types.FunctionType) -> bool:
        mod = fn.__module__.split(".")
        assert len(mod) >= 2 and mod[:2] == ["torch", "_numpy"]
        return fn in cls.constant_fold_functions

    @classmethod
    def get_constant_collection_for_func(cls, fn: types.FunctionType) -> Any:
        mod = fn.__module__.split(".")
        assert len(mod) >= 2 and mod[:2] == ["torch", "_numpy"]
        return np_constant_collections_map.get(fn)

    def call_function(
        self,
        tx: "InstructionTranslator",
        args: Sequence[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        if not config.trace_numpy:
            unimplemented(
                gb_type="attempted to trace numpy function with config.trace_numpy=False",
                context=f"numpy function: {self.value}, args: {args}, kwargs: {kwargs}",
                explanation=f"Attempted to trace numpy function {self.value} "
                "while `torch._dynamo.config.trace_numpy` was set to False.",
                hints=[
                    "Set `torch._dynamo.config.trace_numpy` to True to trace numpy functions.",
                ],
            )

        from ..utils import numpy_to_tensor_wrapper
        from .tensor import NumpyNdarrayVariable

        func = get_np_to_tnp_map().get(self.value)
        if func is None:
            unimplemented(
                gb_type="attempted to trace numpy function unsupported by PyTorch",
                context=f"numpy function: {self.value}, args: {args}, kwargs: {kwargs} (corresponding torch function: {func})",
                explanation=f"Can't find numpy numpy function {self.value} in torch._numpy.",
                hints=[
                    *graph_break_hints.SUPPORTABLE,
                ],
            )

        # We are dealing with a function that produces a const collection type (np.dtype, np.iinfo/np.finfo)
        assert func is not None
        if (
            collection_variable_typ := self.get_constant_collection_for_func(func)
        ) is not None:
            try:
                return collection_variable_typ(
                    self.value(
                        *[x.as_python_constant() for x in args],
                        **{k: v.as_python_constant() for k, v in kwargs.items()},
                    )
                )
            except AsPythonConstantNotImplementedError:
                unimplemented(
                    gb_type="numpy function that produces a const collection type encountered non-const arguments",
                    context=f"numpy function: {self.value}, args: {args}, kwargs: {kwargs} (corresponding torch function: {func})",
                    explanation=f"numpy function {self.value} that produces a const collection type "
                    "(e.g. np.dtype, np.iinfo/np.finfo) "
                    "received arguments that are not constant.",
                    hints=[
                        *graph_break_hints.USER_ERROR,
                    ],
                )
        else:
            if (
                func.__module__ == "torch._numpy.random"
                and config.use_numpy_random_stream
            ):
                unimplemented(
                    gb_type="attempted to trace torch._numpy.random function with config.use_numpy_random_stream=True",
                    context=f"numpy function: {self.value}, args: {args}, kwargs: {kwargs} (corresponding torch function: {func})",
                    explanation=f"Attempted to trace {self.value} when `torch._dynamo.config.use_numpy_random_stream` "
                    "is set to True.",
                    hints=[
                        "Set `torch._dynamo.config.use_numpy_random_stream` to False.",
                        f"Avoid calling {self.value}.",
                    ],
                )

            args, kwargs = NumpyNdarrayVariable.patch_args(func.__name__, args, kwargs)

            if self.can_constant_fold_through(func) and (
                check_unspec_or_constant_args(args, kwargs)
            ):
                # constant fold
                return VariableTracker.build(
                    tx,
                    self.as_python_constant()(
                        *[x.as_python_constant() for x in args],
                        **{k: v.as_python_constant() for k, v in kwargs.items()},
                    ),
                )

            # TODO Add all the functions that go from constants to constants to can_constant_fold_through
            proxy = tx.output.create_proxy(
                "call_function",
                numpy_to_tensor_wrapper(func),
                *proxy_args_kwargs(args, kwargs),
            )
            return NumpyNdarrayVariable.create(tx, proxy)

    def call_method(
        self,
        tx: "InstructionTranslator",
        name: str,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        unimplemented(
            gb_type="attempted to trace numpy.* function as a method",
            context=f"numpy function: {self.value}, args: {args}, kwargs: {kwargs}",
            explanation="Tracing numpy.* functions as methods is not supported.",
            hints=[
                *graph_break_hints.DIFFICULT,
            ],
        )

    def as_python_constant(self) -> BuiltinFunctionType:
        return self.value

    def as_proxy(self) -> Any:
        if config.trace_numpy:
            # Can replace with EnumType once we drop 3.10 support
            if isinstance(self.value, enum.EnumMeta):
                # This is mostly for np._CopyMode
                return self.value
            if isinstance(self.value, type):
                # This handles numpy dtype attributes such as np.float32
                # We return a string as we don't want to serialize non-PyTorch objects in the output FX graph
                # In torch/_numpy we normalize strings to their dtypes when the input is a dtype, as NumPy does
                return self.value.__name__

        return super().as_proxy()

    def is_python_hashable(self) -> Literal[True]:
        return True

    def get_python_hash(self) -> int:
        return hash(self.as_python_constant())

    def is_python_equal(self, other: object) -> bool:
        return (
            isinstance(other, VariableTracker)
            and self.as_python_constant() == other.as_python_constant()
        )


# Used to keep track of NULLs pushed on the stack for Python 3.11 function calls
class NullVariable(VariableTracker):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def __repr__(self) -> str:
        return "NullVariable"

    def reconstruct(self, codegen: "PyCodegen") -> None:
        if sys.version_info < (3, 11):
            unimplemented(
                gb_type="cannot reconstruct NullVariable in Python < 3.11",
                context="",
                explanation="Attempted to generate PUSH_NULL instruction in Python < 3.11; "
                "where this instruction does not exist.",
                hints=[
                    *graph_break_hints.DYNAMO_BUG,
                ],
            )
        codegen.append_output(create_instruction("PUSH_NULL"))


class DeletedVariable(VariableTracker):
    """Marker used to implement delattr()"""


class StringFormatVariable(VariableTracker):
    """
    Represents a call to str.format(), we delay calling format until after the graph.
    """

    _nonvar_fields = {"format_string", *VariableTracker._nonvar_fields}

    @classmethod
    def create(
        cls,
        format_string: str,
        sym_args: Sequence[VariableTracker],
        sym_kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        if all(
            x.is_python_constant()
            for x in itertools.chain(sym_args, sym_kwargs.values())
        ):
            return variables.ConstantVariable.create(
                format_string.format(
                    *[v.as_python_constant() for v in sym_args],
                    **{k: v.as_python_constant() for k, v in sym_kwargs.items()},
                )
            )
        return cls(format_string, list(sym_args), dict(sym_kwargs))

    def __init__(
        self,
        format_string: str,
        sym_args: Sequence[VariableTracker],
        sym_kwargs: dict[str, VariableTracker],
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        assert isinstance(format_string, str)
        self.format_string = format_string
        self.sym_args = sym_args
        self.sym_kwargs = sym_kwargs

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.format_string!r}, {self.sym_args!r}, {self.sym_kwargs!r})"

    def reconstruct(self, codegen: "PyCodegen") -> None:
        codegen.add_push_null(
            lambda: codegen.extend_output(
                [
                    codegen.create_load_const(self.format_string),
                    codegen.create_load_attr("format"),
                ]
            ),
            call_function_ex=True,
        )
        codegen(variables.TupleVariable(list(self.sym_args)))
        kwargs = {
            variables.ConstantVariable.create(k): v for k, v in self.sym_kwargs.items()
        }
        codegen(variables.ConstDictVariable(kwargs))
        codegen.extend_output(create_call_function_ex(True, False))


class ObjectVariable(VariableTracker):
    # placeholder for unknown / opaque values
    def __init__(self, value: object, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.value = value

    def python_type(self) -> type[object]:
        return object


class DebuggingVariable(VariableTracker):
    """
    Represents a call to a debugging function like print(), or something
    registered to config.reorderable_logging_functions.
    """

    def __init__(self, value: Any, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.value = value

    @staticmethod
    def is_reorderable_logging_function(
        obj: Any,
    ) -> TypeGuard[types.FunctionType | types.BuiltinFunctionType]:
        return (
            callable(obj)
            and isinstance(obj, (types.FunctionType, types.BuiltinFunctionType))
            and obj in torch._dynamo.config.reorderable_logging_functions
        )

    # type: ignore[override]
    def call_function(
        self,
        tx: "InstructionTranslator",
        args: Sequence[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> None:
        if tx.export:
            # For export cases, we can just make debugging functions no-ops
            return

        if not self.can_reorder_logs(self.value, args, kwargs):
            unimplemented(
                gb_type="attempted to reorder a debugging function that can't actually be reordered",
                context=f"fn: {self.value}, args: {args}, kwargs: {kwargs}",
                explanation="`torch.compile` can only reorder functions where the arguments "
                "are Tensors, constants, or string formatters.",
                hints=[
                    f"Avoid calling the logging function {self.value} with args that are not supported.",
                ],
            )

        tx.debug_locals.append((self, list(args)))

    def reconstruct(self, codegen: "PyCodegen") -> None:
        assert self.source is not None
        return self.source.reconstruct(codegen)

    @staticmethod
    def can_reorder_logs(fn: Any, args: Sequence[Any], kwargs: dict[str, Any]) -> bool:
        """
        Run some additional checks for what sort of function calls can we
        actually reorder.
        """

        allowed_input_types = (
            variables.TensorVariable,
            variables.ConstantVariable,
            StringFormatVariable,
        )

        flat_args = pytree.tree_leaves([args, kwargs])
        for arg in flat_args:
            if not isinstance(arg, allowed_input_types):
                return False

        return True


class IgnoredFunctionVariable(VariableTracker):
    """
    Represents a call to an arbitrary function that should be ignored.
    """

    def __init__(self, value: Any, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.value = value

    def call_function(
        self,
        tx: "InstructionTranslator",
        args: Sequence[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        return variables.CONSTANT_VARIABLE_NONE


class LoggingLoggerVariable(VariableTracker):
    """
    Represents a call to any logging.Logger methods.
    """

    def __init__(self, value: logging.Logger, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.value = value

    def call_method(
        self,
        tx: "InstructionTranslator",
        name: str,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        if tx.export:
            # For export cases, we can just make logging functions no-ops.
            return variables.CONSTANT_VARIABLE_NONE

        method = getattr(self.value, name, None)
        function = getattr(method, "__func__", None)

        # Unified ignore set
        ignore_set = torch._dynamo.config.ignore_logging_functions

        if method in ignore_set or function in ignore_set:
            return variables.CONSTANT_VARIABLE_NONE

        unimplemented(
            gb_type="logging.Logger method not supported for non-export cases",
            context=f"method: {self.value}.{name}, args: {args}, kwargs: {kwargs}",
            explanation="logging.Logger methods are not supported for non-export cases.",
            hints=[
                "Add the logging method to `torch._dynamo.config.ignore_logging_functions`.",
            ],
        )


class ConstantLikeVariable(VariableTracker):
    """self.value is a compile-time constant, but not a literal"""

    try:
        from numpy import (
            dtype as np_dtype,
            floating as np_floating,
            generic as np_generic,
        )
    except ImportError:
        # type: ignore[misc, assignment]
        np_floating = type("invalid_type", (), {})
        # type: ignore[misc, assignment]
        np_dtype = type("invalid_type", (), {})

    def __init__(self, value: Any, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.value = value

    @property
    def _error_prefix(self) -> str:
        """Dynamically compute the prefix from the value's type"""
        t = type(self.value)

        # For builtins (int, str, etc.), just return the name
        if t.__module__ == "builtins":
            return t.__qualname__

        return f"{t.__module__}.{t.__qualname__}"

    def as_python_constant(self) -> Any:
        return self.value

    def call_method(
        self,
        tx: "InstructionTranslator",
        name: str,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # pyrefly: ignore [implicit-any]
        cargs, ckwargs = [], {}
        try:
            # we only support constant propagation for methods
            cargs = [x.as_python_constant() for x in args]
            ckwargs = {k: v.as_python_constant() for k, v in kwargs.items()}
        except NotImplementedError:
            unimplemented(
                gb_type="constant-like method call with non-constant args",
                context=f"{self._error_prefix}.{name}(*{args}, **{kwargs})",
                explanation=f"Attempted to call {self._error_prefix}.{name} with non-constant args.",
                hints=[
                    "Ensure that the args to the method call are constant (int, str, etc.).",
                ],
            )

        result = getattr(self.value, name)(*cargs, **ckwargs)

        if variables.ConstantVariable.is_literal(result):
            return VariableTracker.build(tx, result)
        if isinstance(result, re.Match):
            return ConstantLikeVariable(result)

        unimplemented(
            gb_type="constant-like method call with unsupported return type",
            context=f"{self._error_prefix}.{name}(*{args}, **{kwargs}) returned {result}",
            explanation=f"Attempted to call {self._error_prefix}.{name}, got unsupported return value {result}.",
            hints=[
                *graph_break_hints.SUPPORTABLE,
            ],
        )

    def var_getattr(self, tx: "InstructionTranslator", name: str) -> VariableTracker:
        result = getattr(self.value, name)
        if isinstance(result, self.np_floating):
            result = float(result)
        if isinstance(result, self.np_dtype):
            return NumpyDTypeVariable(result)
        if isinstance(result, type) and issubclass(result, self.np_generic):
            # things like x.dtype.type
            return NumpyVariable(result)
        if variables.ConstantVariable.is_literal(result):
            return VariableTracker.build(tx, result)
        return GetAttrVariable(self, name)


class TorchVersionVariable(ConstantLikeVariable):
    _error_prefix = "torch.__version__"

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("value", torch.__version__)
        assert kwargs["value"] is torch.__version__
        super().__init__(**kwargs)


class NumpyDTypeVariable(ConstantLikeVariable):
    def as_proxy(self) -> str:
        """Similar to how numpy dtype descriptors (e.g. np.float32 ) are handled by NumpyVariable:

        np.dtype() objects are serialized as strings, torch._numpy wrappers will normalize to the torch dtype.
        This also handles unsupported things nicely (i.e. structured arrays and object arrays).
        """
        return self.value.type.__name__


np_constant_collections_map = {
    tnp.finfo: ConstantLikeVariable,
    tnp.iinfo: ConstantLikeVariable,
    tnp.dtype: NumpyDTypeVariable,
}


class RandomClassVariable(VariableTracker):
    """random.Random"""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def call_function(
        self,
        tx: "InstructionTranslator",
        args: Sequence[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> "RandomVariable":
        if len(args) > 1 or kwargs:
            unimplemented(
                gb_type="random.Random() with improper arguments",
                context=f"args: {args}, kwargs: {kwargs}",
                explanation="random.Random() with > 1 arg or with kwargs is not supported.",
                hints=[
                    *graph_break_hints.USER_ERROR,
                ],
            )
        seed = variables.CONSTANT_VARIABLE_NONE if len(args) == 0 else args[0]
        return RandomVariable(
            seed=seed, mutation_type=variables.base.ValueMutationNew()
        )


class RandomVariable(VariableTracker):
    """random.Random()

    Implemented by wrapping a VariableTracker around a random.Random object.
    The supported methods for the random.Random object cannot be overridden.
    Assumes that random objects behave the same given a set seed or state.
    """

    _nonvar_fields = {
        "random",
        *VariableTracker._nonvar_fields,
    }

    _supported_fn_names = {
        "random",
        "randint",
        "randrange",
        "uniform",
    }

    def __init__(
        self,
        rand: random.Random | None = None,
        seed: VariableTracker | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if rand is not None:
            assert self.is_supported_random_obj(rand)
            self.random = random.Random()
            self.random.setstate(rand.getstate())
        else:
            seed = seed.as_python_constant() if seed is not None else None
            self.random = random.Random(seed)

    def python_type(self) -> type[random.Random]:
        return random.Random

    def as_python_constant(self) -> random.Random:
        return self.random

    @staticmethod
    def is_supported_random_obj(val: Random) -> bool:
        if type(val) is not random.Random:
            return False
        for name in itertools.chain(
            RandomVariable._supported_fn_names, ("seed", "getstate", "setstate")
        ):
            if not hasattr(val, name):
                return False
            meth = getattr(val, name)
            if inspect.isbuiltin(meth):
                # e.g. random.Random.random
                if meth != getattr(random.Random, name).__get__(val):
                    return False
            else:
                if getattr(meth, "__func__", None) is not getattr(random.Random, name):
                    return False
        return True

    @staticmethod
    def check_state(state: tuple[int, tuple[int, ...], float | None]) -> None:
        assert type(state) is tuple
        assert type(state[0]) is int
        assert type(state[1]) is tuple
        assert all(type(x) is int for x in state[1])
        assert state[2] is None or type(state[2]) is float

    @staticmethod
    def wrap_state(state: tuple[int, tuple[int, ...], float | None]) -> TupleVariable:
        RandomVariable.check_state(state)
        return variables.TupleVariable(
            [
                variables.ConstantVariable.create(state[0]),
                variables.TupleVariable(
                    [variables.ConstantVariable.create(x) for x in state[1]]
                ),
                variables.ConstantVariable.create(state[2]),
            ]
        )

    @staticmethod
    def unwrap_state(
        state: VariableTracker,
    ) -> tuple[int, tuple[int, ...], float | None]:
        state_obj = state.as_python_constant()
        RandomVariable.check_state(state_obj)
        return state_obj

    def call_method(
        self,
        tx: "InstructionTranslator",
        name: str,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        if name == "seed":
            tx.output.side_effects.mutation(self)
            self.random.seed(
                *[x.as_python_constant() for x in args],
                **{key: val.as_python_constant() for key, val in kwargs.items()},
            )
            return variables.CONSTANT_VARIABLE_NONE
        elif name == "getstate":
            return self.wrap_state(self.random.getstate())
        elif name == "setstate":
            tx.output.side_effects.mutation(self)
            self.random.setstate(self.unwrap_state(args[0]))
            return variables.CONSTANT_VARIABLE_NONE
        elif name in self._supported_fn_names:
            tx.output.side_effects.mutation(self)
            state = self.random.getstate()

            def call_random_meth(*args: Any, **kwargs: Any) -> Any:
                r = random.Random()
                r.setstate(state)
                return getattr(r, name)(*args, **kwargs)

            # self.random state not actually updated by call_random_meth, so update here
            # by calling the method
            getattr(self.random, name)(
                *[x.as_python_constant() for x in args],
                **{k: v.as_python_constant() for k, v in kwargs.items()},
            )

            return call_random_fn(tx, call_random_meth, args, kwargs)
        return super().call_method(tx, name, args, kwargs)

    def reconstruct(self, codegen: "PyCodegen") -> None:
        codegen.add_push_null(
            lambda: codegen.extend_output(
                [
                    codegen.create_load_python_module(random),
                    codegen.create_load_attr("Random"),
                ]
            )
        )
        codegen.call_function(0, False)
        # NOTE using add_push_null may result in NULL being duplicated
        # so defer the push_null to call_function
        codegen.dup_top()
        codegen.load_attr("setstate")
        codegen(self.wrap_state(self.random.getstate()))
        codegen.call_function(1, True)
        codegen.pop_top()


class WeakRefVariable(VariableTracker):
    @staticmethod
    # pyrefly: ignore[bad-param-name-override]
    def build(
        tx: "InstructionTranslator",
        weakref_value: weakref.ReferenceType[Any],
        source: Source | None,
        **options: Any,
    ) -> "WeakRefVariable":
        assert source is not None
        callback = weakref_value.__callback__
        callback_source = source and AttrSource(source, "__callback__")
        callback_vt = VariableTracker.build(tx, callback, callback_source)
        referent = weakref_value()
        source = source and WeakRefCallSource(source)
        referent_vt = VariableTracker.build(tx, referent, source)
        options["source"] = source
        return WeakRefVariable(referent_vt, callback_vt, **options)

    def __init__(
        self, referent_vt: VariableTracker, callback_vt: VariableTracker, **options: Any
    ) -> None:
        super().__init__(**options)
        self.referent_vt = referent_vt
        self.callback_vt = callback_vt

    def call_function(
        self,
        tx: "InstructionTranslator",
        args: Sequence[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        return self.referent_vt

    def reconstruct(self, codegen: "PyCodegen") -> None:
        codegen.add_push_null(lambda: codegen.load_import_from("weakref", "ref"))
        codegen(self.referent_vt)
        codegen(self.callback_vt)
        codegen.extend_output(create_call_function(2, False))

    def is_python_hashable(self) -> bool:
        return self.referent_vt.is_python_hashable()

    def get_python_hash(self) -> int:
        # weakref relies on the referent's hash
        return self.referent_vt.get_python_hash()

    def is_python_equal(self, other: object) -> bool:
        if not isinstance(other, WeakRefVariable):
            return False
        return self.referent_vt.is_python_equal(other.referent_vt)
