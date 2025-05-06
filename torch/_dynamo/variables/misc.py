# mypy: ignore-errors

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
import functools
import inspect
import itertools
import random
import re
import sys
import types
import warnings
from typing import Optional, TYPE_CHECKING

import torch._C
import torch._numpy as tnp
import torch.utils._pytree as pytree

from .. import config, variables
from ..bytecode_transformation import create_call_function, create_instruction
from ..create_parameter_op import do_not_convert_to_tracable_parameter
from ..exc import raise_observed_exception, unimplemented
from ..guards import GuardBuilder, install_guard
from ..mutation_guard import unpatched_nn_module_init
from ..source import (
    AttrSource,
    GenericAttrSource,
    GetItemSource,
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
    set_example_value,
    tuple_methods,
)
from .base import VariableTracker
from .constant import ConstantVariable
from .functions import NestedUserFunctionVariable, UserFunctionVariable
from .user_defined import call_random_fn, is_standard_setattr, UserDefinedObjectVariable


if TYPE_CHECKING:
    from torch._dynamo.symbolic_convert import InstructionTranslator


class NO_SUCH_SUBOBJ:
    pass


class SuperVariable(VariableTracker):
    _nonvar_fields = {
        *VariableTracker._nonvar_fields,
    }

    def __init__(self, typevar, objvar=None, **kwargs) -> None:
        super().__init__(**kwargs)
        # typevar is the fist argument to super(). In the case where no argument
        # is provided to super(), it is the __class__ object where
        # the super() function is being called
        self.typevar = typevar
        # objvar here must be an instance or subtype of typevar.
        # In the case where super() is called without arguments, it is the first argument
        # to the current function where super() is called from (self for regular method,
        # cls for a classmethod)
        self.objvar = objvar

    def reconstruct(self, codegen):
        codegen.add_push_null(lambda: codegen(variables.BuiltinVariable(super)))
        codegen(self.typevar)
        if self.objvar is not None:
            codegen(self.objvar)
            codegen.extend_output(create_call_function(2, False))
        else:
            codegen.extend_output(create_call_function(1, False))

    def _resolved_getattr_and_source(self, tx: "InstructionTranslator", name):
        assert self.objvar, "1-arg super not implemented"
        search_type = self.typevar.as_python_constant()

        # The rest of this function does two things:
        #   - Walk the mro to find where the attribute comes from to be
        #     able to provide accurate source
        #   - Call the getattr to get the object

        # Find the class object, where the function lives.
        # When objvar is "self", use type(self), when objvar is "cls", use it as-is
        type_to_use = self.objvar.python_type()
        type_to_use_source = (
            TypeSource(self.objvar.source) if self.objvar.source else None
        )
        if issubclass(type_to_use, type):
            type_to_use = self.objvar.value
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
                            GetItemSource(
                                AttrSource(type_to_use_source, "__mro__"), index
                            ),
                            name,
                        )
                    return resolved_getattr, source

        unimplemented("Unable to resolve super getattr")

    def var_getattr(self, tx: "InstructionTranslator", name: str) -> "VariableTracker":
        # Check if getattr is a constant. If not, delay the actual work by
        # wrapping the result in GetAttrVariable. Mostly super is called with a
        # method, so most of the work is delayed to call_function.
        #
        # We could have just implemented a const_getattr. However, super is
        # special when it comes to finding sources. Compared to other VTs, super
        # requires the attr name to walk the mro and find the actual source (and
        # not just AttrSource).
        value, source = self._resolved_getattr_and_source(self, name)
        if not variables.ConstantVariable.is_literal(value):
            return GetAttrVariable(self, name)
        if source:
            install_guard(source.make_guard(GuardBuilder.CONSTANT_MATCH))
        return variables.ConstantVariable.create(value, source=source)

    def call_method(
        self,
        tx,
        name,
        args: "list[VariableTracker]",
        kwargs: "dict[str, VariableTracker]",
    ) -> "VariableTracker":
        inner_fn, source = self._resolved_getattr_and_source(self, name)
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
                    return variables.UserFunctionVariable(
                        unpatched_nn_module_init, source=source
                    ).call_function(tx, [self.objvar] + args, kwargs)
            else:
                unimplemented("super() nn.Module.__init__")
        elif (
            self.objvar.source
            and hasattr(inner_fn, "__name__")
            and inner_fn.__name__ == "__new__"
            and variables.UserDefinedClassVariable.is_supported_new_method(inner_fn)
        ):
            user_cls = inner_fn.__self__
            if hasattr(user_cls, "__module__") and user_cls.__module__ == "builtins":
                user_cls_vt = variables.BuiltinVariable(user_cls)
            else:
                user_cls_source = source.member
                user_cls_vt = variables.UserDefinedClassVariable(
                    user_cls, source=user_cls_source
                )
            return user_cls_vt.call_method(tx, "__new__", args, kwargs)
        elif isinstance(inner_fn, staticmethod) and isinstance(
            inner_fn.__func__, types.FunctionType
        ):
            return variables.UserFunctionVariable(
                inner_fn.__func__, source=source
            ).call_function(tx, args, kwargs)
        elif isinstance(inner_fn, classmethod) and isinstance(
            inner_fn.__func__, types.FunctionType
        ):
            return variables.UserMethodVariable(
                inner_fn.__func__, self.objvar, source=source
            ).call_function(tx, args, kwargs)
        elif isinstance(inner_fn, types.FunctionType):
            return variables.UserFunctionVariable(
                inner_fn, source=source
            ).call_function(tx, [self.objvar] + args, kwargs)
        elif isinstance(inner_fn, types.MethodType):
            return variables.UserMethodVariable(
                inner_fn.__func__, self.objvar, source=source
            ).call_function(tx, args, kwargs)
        elif is_standard_setattr(inner_fn) and isinstance(
            self.objvar, UserDefinedObjectVariable
        ):
            return self.objvar.method_setattr_standard(tx, *args, **kwargs)
        elif inner_fn is object.__delattr__:
            attr = args[0]
            try:
                attr = attr.as_python_constant()
            except NotImplementedError:
                unimplemented(f"non-const delattr attr: {attr}")
            if not tx.output.side_effects.is_attribute_mutation(self.objvar):
                unimplemented(f"delattr({self.objvar}, {attr}, ...)")

            tx.output.side_effects.store_attr(
                self.objvar, attr, variables.DeletedVariable()
            )
            return variables.ConstantVariable(None)
        elif (
            isinstance(self.objvar, variables.UserDefinedDictVariable)
            and inner_fn in self.objvar._dict_methods
        ):
            return self.objvar._dict_vt.call_method(tx, name, args, kwargs)
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
            attr_name = args[0].value
            if tx.output.side_effects.has_pending_mutation_of_attr(
                self.objvar, attr_name
            ):
                result = tx.output.side_effects.load_attr(
                    self.objvar, attr_name, deleted_ok=True
                )
                if isinstance(result, variables.DeletedVariable):
                    raise_observed_exception(AttributeError, tx)
                return result

            try:
                # NB - use object.__getattribute__ to prevent running any user code
                attr_value = object.__getattribute__(self.objvar.value, attr_name)
            except AttributeError:
                raise_observed_exception(AttributeError, tx)

            attr_source = None
            if self.objvar.source is not None:
                # setup a object.__getattribute__(self.objvar, name) source
                attr_source = GenericAttrSource(self.objvar.source, attr_name)
            return VariableTracker.build(tx, attr_value, attr_source)

        unimplemented(f"non-function or method super: {inner_fn}")


class ExceptionVariable(VariableTracker):
    # The ExceptionVariable corresponds to the BaseException class in Python
    def __init__(self, exc_type, args, **kwargs) -> None:
        super().__init__(**kwargs)
        self.exc_type = exc_type
        self.args = args
        # When raising a new exception while another exception is already being
        # handled, the new exception's __context__ attribute is automatically
        # set to the handled exception.
        self.__context__ = ConstantVariable(None)
        # Set when user raised an exception from another:
        # raise ... from ...
        self.__cause__ = ConstantVariable(None)
        # Boolean flag that controls whether the __context__ attribute is set
        self.__suppress_context__ = ConstantVariable(False)
        # Contains the call stack where the exception was raised. Dynamo does
        # not track traceback. So, this variable is always set to None
        self.__traceback__ = ConstantVariable(None)

    def set_context(self, context: "ExceptionVariable"):
        self.__context__ = context

    def reconstruct(self, codegen):
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

    def python_type(self):
        return self.exc_type

    def call_setattr(
        self,
        tx: "InstructionTranslator",
        name_var: VariableTracker,
        val: VariableTracker,
    ):
        def raise_error(msg):
            raise_observed_exception(TypeError, tx, args=[ConstantVariable(msg)])

        name = name_var.as_python_constant()
        if name == "__context__":
            self.set_context(val)
        elif name == "__cause__":
            if (isinstance(val, ConstantVariable) and val.value is None) or isinstance(
                val,
                (
                    variables.BuiltinVariable,
                    variables.ExceptionVariable,
                    variables.UserDefinedExceptionClassVariable,
                    variables.UserDefinedExceptionObjectVariable,
                ),
            ):
                self.__cause__ = val
                self.__suppress_context__ = variables.ConstantVariable(True)
            else:
                raise_error("exception cause must be None or derive from BaseException")
        elif name == "__suppress_context__":
            if isinstance(val, ConstantVariable) and val.value in (True, False):
                self.__suppress_context__ = val
            else:
                raise_error("exception cause must be None or derive from BaseException")
        elif name == "__traceback__":
            if isinstance(val, ConstantVariable) and val.value is None:
                self.__traceback__ = val
            else:
                unimplemented(f"setattr(ExceptionVariable, {name_var}, {val})")
        else:
            unimplemented(f"setattr(ExceptionVariable, {name_var}, {val})")
        return variables.ConstantVariable(None)

    def call_method(self, tx, name, args, kwargs):
        if name == "__setattr__":
            return self.call_setattr(tx, *args)
        elif name == "with_traceback":
            [tb] = args
            self.call_setattr(tx, ConstantVariable("__traceback__"), tb)
            return self
        else:
            return super().call_method(tx, name, args, kwargs)

    def var_getattr(self, tx, name):
        if name == "__context__":
            return self.__context__
        elif name == "__cause__":
            return self.__cause__
        elif name == "__suppress_context__":
            return self.__suppress_context__
        elif name == "__traceback__":
            return variables.ConstantVariable(None)
        elif name == "args":
            return variables.ListVariable(self.args, source=self.source)
        return super().var_getattr(tx, name)

    def __str__(self):
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


class ComptimeVariable(VariableTracker):
    """
    This variable is special, it lets you execute arbitrary code at
    Dynamo compile time
    """

    def reconstruct(self, codegen):
        raise NotImplementedError("comptime is special form")

    def var_getattr(self, tx: "InstructionTranslator", name: str) -> "VariableTracker":
        from ..comptime import comptime

        # To support the comptime.print_graph convenience accessors
        from .functions import UserFunctionVariable

        return UserFunctionVariable(
            getattr(comptime, name), source=AttrSource(self.source, name)
        )

    def call_function(
        self,
        tx: "InstructionTranslator",
        args: "list[VariableTracker]",
        kwargs: "dict[str, VariableTracker]",
    ) -> "VariableTracker":
        from ..comptime import ComptimeContext

        # TODO: support an expression form as well

        assert not kwargs
        # Second argument is runtime lambda, ignored
        assert len(args) <= 2
        fn = args[0]
        if isinstance(fn, UserFunctionVariable):
            fn.get_function()(ComptimeContext(tx))
        elif isinstance(fn, NestedUserFunctionVariable):
            # We have to manually bind the freevars ourselves
            code = fn.get_code()
            assert not fn.closure, (
                "comptime function must not have free variables, "
                f"but these variables were free: {code.co_freevars}"
            )
            func = types.FunctionType(
                code,
                fn.f_globals,
                fn.fn_name.as_python_constant(),
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

        return variables.ConstantVariable.create(None)


class CellVariable(VariableTracker):
    # If the cell existed before Dynamo tracing started, this will be the
    # VariableTracker that represents the cell content.
    #
    # Note that all mutation to the cell (i.e., its content) will be buffered in
    # SideEffects, rather than being reflected here. One can think of
    # `CellVariable` as a special case for `UserDefinedObjectVariable`.
    pre_existing_contents: Optional[VariableTracker]

    # This is set when this cell can be referenced via `LOAD/STORE_DEREF` in the
    # root frame via this name (e.g., the name is in `co_cellvars/co_freevars`).
    local_name: Optional[str] = None

    def __init__(
        self, pre_existing_contents: Optional[VariableTracker] = None, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.pre_existing_contents = pre_existing_contents


class NewGlobalVariable(VariableTracker):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)


def produce_trampoline_autograd_apply(fn_cls):
    def trampoline_autograd_apply(*args, **kwargs):
        return fn_cls.apply(*args, **kwargs)

    trampoline_autograd_apply._origin = produce_trampoline_autograd_apply
    return trampoline_autograd_apply


class AutogradFunctionVariable(VariableTracker):
    """represents a torch.autograd.Function subclass"""

    _nonvar_fields = {
        "fn_cls",
        *VariableTracker._nonvar_fields,
    }

    def __init__(self, fn_cls, **kwargs) -> None:
        super().__init__(**kwargs)
        self.fn_cls = fn_cls

    def call_apply(self, tx: "InstructionTranslator", args, kwargs):
        requires_grad = False

        def visit(node):
            nonlocal requires_grad
            if isinstance(node, variables.TensorVariable):
                if node.requires_grad is not False:
                    requires_grad = True
            if isinstance(node, variables.NNModuleVariable):
                if node.is_training(tx):
                    requires_grad = True

        VariableTracker.visit(visit, (args, kwargs))

        if requires_grad and torch.is_grad_enabled():
            if config.capture_autograd_function is False:
                warnings.warn(
                    "The config.capture_autograd_function flag is deprecated, it's now always true."
                )

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

            vjp_fn = self.fn_cls.vjp  # type: ignore[attr-defined]
            if vjp_fn is not torch.autograd.Function.vjp:
                unimplemented("NYI - User defind vjp")

            jvp_fn = self.fn_cls.jvp  # type: ignore[attr-defined]
            if jvp_fn is not torch.autograd.Function.jvp:
                unimplemented("NYI - User defind jvp")

            from .higher_order_ops import AutogradFunctionApplyVariable

            source = self.source
            if source is None:
                source = AttrSource(
                    tx.import_source(self.fn_cls.__module__), self.fn_cls.__name__
                )

            val = AutogradFunctionApplyVariable(
                forward_fn,
                self.fn_cls.backward,
                source,
                source=AttrSource(source, member="apply"),
            ).call_function(tx, args, kwargs)
            # Inside of AutogradFunctionApplyVariable.call_function, we use sourceless variable wrapping
            # the forward function, as we don't want to generate guards for new_forward.__closure__
            # if forward is rewritten by autograd_function_forward_rewritten.
            # But we still need to generate correct guards for the original forward and setup_context
            # functions, so we have to add guards manually.
            if self.source:
                fwd_src = AttrSource(self.source, "forward")
                install_guard(fwd_src.make_guard(GuardBuilder.FUNCTION_MATCH))
                if is_setup_ctx_defined:
                    setup_ctx_src = AttrSource(self.source, "setup_context")
                    install_guard(setup_ctx_src.make_guard(GuardBuilder.FUNCTION_MATCH))

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
            if len(args) - 1 == len(sig._parameters):
                args = args[1:]  # Don't use context
            return variables.UserFunctionVariable(fn, source=source).call_function(
                tx, args, kwargs
            )
        elif isinstance(fn, types.MethodType):
            return variables.UserMethodVariable(
                fn.__func__,
                variables.UserDefinedClassVariable(self.fn_cls),
                source=source,
            ).call_function(tx, args, kwargs)
        else:
            unimplemented(
                f"non-function or method in subclass of torch.autograd.Function: {fn}"
            )

    def call_backward(self, tx: "InstructionTranslator", args, kwargs):
        fn = self.fn_cls.backward
        assert type(args[0].value) is torch._dynamo.external_utils.FakeBackwardCFunction
        assert isinstance(fn, types.FunctionType)

        fn_source = AttrSource(self.source, "backward")
        return variables.UserFunctionVariable(fn, source=fn_source).call_function(
            tx, args, kwargs
        )

    def call_function(self, tx: "InstructionTranslator", args, kwargs):
        return AutogradFunctionVariable(self.fn_cls)

    def call_method(
        self,
        tx,
        name,
        args: "list[VariableTracker]",
        kwargs: "dict[str, VariableTracker]",
    ):
        from ..trace_rules import is_callable_allowed
        from .builder import wrap_fx_proxy

        if name == "apply":
            if is_callable_allowed(self.fn_cls):
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
            from .. import trace_rules

            source = AttrSource(self.source, name) if self.source is not None else None
            try:
                obj = inspect.getattr_static(self.fn_cls, name)
            except AttributeError:
                obj = None

            if isinstance(obj, staticmethod):
                func = obj.__get__(self.fn_cls)
                if source is not None:
                    return (
                        trace_rules.lookup(func)
                        .create_with_source(func, source=source)
                        .call_function(tx, args, kwargs)
                    )
                else:
                    return trace_rules.lookup(func)(func).call_function(
                        tx, args, kwargs
                    )
            elif isinstance(obj, classmethod):
                return variables.UserMethodVariable(
                    obj.__func__, self, source=source
                ).call_function(tx, args, kwargs)
            else:
                unimplemented(f"Unsupported method: {name}")


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
        value,
        value_type=None,
        inference=False,
        proxy=None,
        saved_tensors=None,
        needs_input_grad=None,
        non_differentiable=None,
        **kwargs,
    ) -> None:
        super().__init__(value=value, value_type=value_type, **kwargs)
        self.inference = inference
        self.proxy = proxy
        self.saved_tensors = saved_tensors
        self.needs_input_grad = needs_input_grad
        self.non_differentiable = non_differentiable

    @staticmethod
    def create(tx: "InstructionTranslator", args=None, kwargs=None):
        needs_input_grad = None
        if args and not kwargs:
            needs_input_grad = tuple(
                isinstance(x, variables.TensorVariable) and x.requires_grad
                for x in args
            )
        proxy = tx.output.create_proxy(
            "call_function", torch.autograd.function.FunctionCtx, (), {}
        )
        out = tx.output.side_effects.track_object_new(
            None,
            torch.autograd.function.FunctionCtx,
            functools.partial(
                AutogradFunctionContextVariable,
                inference=True,
                proxy=proxy,
                saved_tensors=SavedTensorBox(),
                needs_input_grad=needs_input_grad,
            ),
            {},
        )
        set_example_value(proxy.node, out.value)

        return out

    def as_proxy(self):
        if self.proxy is None:
            unimplemented("proxy not set")
        return self.proxy

    def call_method(
        self,
        tx,
        name,
        args: "list[VariableTracker]",
        kwargs: "dict[str, VariableTracker]",
    ) -> "VariableTracker":
        if name == "__setattr__":
            return super().call_method(tx, name, args, kwargs)
        elif name == "mark_non_differentiable":
            assert len(kwargs) == 0
            self.non_differentiable = proxy_args_kwargs(args, {})[0]
            return variables.ConstantVariable.create(None)

        if name != "save_for_backward":
            unimplemented(f"autograd.Function context method: {name}")
        if self.saved_tensors is None:
            unimplemented(
                "save_for_backward only supported on a newly constructed FunctionCtx"
            )

        if not self.inference:
            assert self.source and not kwargs
            tx.output.side_effects.track_save_for_backward(self, args)

        # In eager mode, multiple calls to .save_for_backward() will overwrite previous calls.
        if len(self.saved_tensors.tensors) > 0:
            self.saved_tensors.tensors = []
        for arg in args:
            self.saved_tensors.tensors.append(arg)
        return variables.ConstantVariable.create(None)

    def var_getattr(self, tx: "InstructionTranslator", name):
        if name in ["save_for_backward", "mark_non_differentiable"]:
            return LambdaVariable(
                lambda *args, **kwargs: self.call_method(tx, name, args, kwargs)
            )
        if name == "saved_tensors" and self.saved_tensors is not None:
            return variables.TupleVariable(list(self.saved_tensors.tensors))
        if name == "needs_input_grad":
            if self.needs_input_grad is not None:
                return variables.ConstantVariable.create(self.needs_input_grad)
            if self.source:
                source = AttrSource(self.source, "needs_input_grad")
                return VariableTracker.build(tx, self.value.needs_input_grad, source)

        return super().var_getattr(tx, name)


class AutogradEngineVariable(UserDefinedObjectVariable):
    """
    Represents a torch._C._ImperativeEngine instance.
    """

    def __init__(
        self,
        value,
        value_type=None,
        **kwargs,
    ) -> None:
        super().__init__(value=value, value_type=value_type, **kwargs)

    def call_method(
        self,
        tx,
        name,
        args: "list[VariableTracker]",
        kwargs: "dict[str, VariableTracker]",
    ) -> "VariableTracker":
        if name == "queue_callback":
            if torch._dynamo.compiled_autograd.in_compiled_autograd_region:
                assert tx.one_graph, (
                    "queue_callback() is only supported when Compiled Autograd is enabled with fullgraph=True"
                )
                return variables.UserFunctionVariable(
                    torch._dynamo.external_utils.FakeCompiledAutogradEngine.queue_callback,
                    source=self.source,
                ).call_function(
                    tx,
                    (tx.output.side_effects.get_ca_final_callbacks_var(), *args),
                    kwargs,
                )
            else:
                unimplemented(
                    "queue_callback() is only supported when Compiled Autograd is enabled with fullgraph=True"
                )
        else:
            unimplemented(f"torch._C._ImperativeEngine method: {name}")


class LambdaVariable(VariableTracker):
    def __init__(self, fn, **kwargs) -> None:
        super().__init__(**kwargs)
        self.fn = fn

    def call_function(
        self,
        tx: "InstructionTranslator",
        args: "list[VariableTracker]",
        kwargs: "dict[str, VariableTracker]",
    ) -> "VariableTracker":
        return self.fn(*args, **kwargs)


class GetAttrVariable(VariableTracker):
    _nonvar_fields = {
        "name",
        "py_type",
        *VariableTracker._nonvar_fields,
    }

    def __init__(self, obj, name, py_type=None, **kwargs) -> None:
        super().__init__(**kwargs)
        assert isinstance(obj, VariableTracker)
        assert isinstance(name, str)
        self.obj = obj
        self.name = name
        self.py_type = py_type  # In some cases we know the type (ex. tensor methods)

    def python_type(self):
        if self.py_type is not None:
            return self.py_type
        else:
            return super().python_type()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.obj}, {self.name})"

    @staticmethod
    def create_getattr_proxy(base_proxy: torch.fx.Proxy, attr):
        return getattr(base_proxy, attr)

    def as_proxy(self):
        return GetAttrVariable.create_getattr_proxy(self.obj.as_proxy(), self.name)

    def as_python_constant(self):
        constant = self.obj.as_python_constant()
        try:
            return getattr(constant, self.name)
        except AttributeError:
            raise NotImplementedError(f"{self} is not a constant") from None

    def const_getattr(self, tx: "InstructionTranslator", name):
        if not isinstance(self.obj, variables.NNModuleVariable):
            raise NotImplementedError
        step1 = tx.output.get_submodule(self.obj.module_key)
        if self.name not in step1.__dict__:
            raise NotImplementedError
        step2 = inspect.getattr_static(step1, self.name)
        if name not in step2.__dict__:
            raise NotImplementedError
        return inspect.getattr_static(step2, name)

    def reconstruct(self, codegen):
        codegen(self.obj)
        codegen.extend_output(codegen.create_load_attrs(self.name))

    def call_function(
        self,
        tx: "InstructionTranslator",
        args: "list[VariableTracker]",
        kwargs: "dict[str, VariableTracker]",
    ) -> "VariableTracker":
        return self.obj.call_method(tx, self.name, args, kwargs)

    def call_method(
        self,
        tx,
        name,
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
                # redirect to var_getattr on the original obj
                return obj.var_getattr(tx, key)

            # Return the default value for get
            if name == "get":
                if len(args) == 2:
                    return args[1]
                else:
                    return variables.ConstantVariable(None)

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
                return variables.ConstantVariable(True)
            else:
                return variables.ConstantVariable(False)

        elif name == "__setitem__" and self.name == "__dict__" and not kwargs:
            if isinstance(self.obj, variables.UserDefinedObjectVariable):
                # Bypass any custom setattr as we are updating the `__dict__` itself
                return self.obj.method_setattr_standard(tx, args[0], args[1])
            if isinstance(self.obj, variables.NNModuleVariable):
                # This matches how `setattr` is handled for NNModuleVariable
                self.obj.convert_to_unspecialized(tx)

        return super().call_method(tx, name, args, kwargs)


class MethodWrapperVariable(VariableTracker):
    def __init__(self, method_wrapper, **kwargs) -> None:
        super().__init__(**kwargs)
        self.method_wrapper = method_wrapper
        self._builtin_fns = {}

    def call_function(
        self,
        tx: "InstructionTranslator",
        args: "list[VariableTracker]",
        kwargs: "dict[str, VariableTracker]",
    ) -> "VariableTracker":
        if is_tensor_base_attr_getter(self.method_wrapper) and isinstance(
            args[0], variables.TensorVariable
        ):
            assert len(args) == 1 and len(kwargs) == 0

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
                return variables.BuiltinVariable(object).call_method(
                    tx, wrapper_name, [self_obj, *args], kwargs
                )

        super().call_function(tx, args, kwargs)

    def is_python_constant(self):
        return True

    def as_python_constant(self):
        return self.method_wrapper


class GetSetDescriptorVariable(VariableTracker):
    def __init__(self, desc, **kwargs) -> None:
        super().__init__(**kwargs)
        self.desc = desc

    def var_getattr(self, tx: "InstructionTranslator", name):
        if name == "__get__" and self.source:
            source = AttrSource(self.source, "__get__")
            return VariableTracker.build(tx, self.desc.__get__, source)
        else:
            return super().var_getattr(tx, name)

    def is_python_constant(self):
        return True

    def as_python_constant(self):
        return self.desc


class PythonModuleVariable(VariableTracker):
    _nonvar_fields = {
        "value",
        "is_torch",
        *VariableTracker._nonvar_fields,
    }

    def __init__(self, value: types.ModuleType, **kwargs) -> None:
        super().__init__(**kwargs)
        self.value = value
        self.is_torch = self.value is torch or self.value.__name__.startswith("torch.")

    def python_type(self):
        return types.ModuleType

    def as_python_constant(self):
        return self.value

    def __repr__(self) -> str:
        return f"PythonModuleVariable({self.value})"

    def call_obj_hasattr(self, tx: "InstructionTranslator", name):
        result = hasattr(self.value, name)
        return variables.ConstantVariable.create(result)

    def var_getattr(self, tx: "InstructionTranslator", name):
        if tx.output.side_effects.has_pending_mutation_of_attr(self, name):
            return tx.output.side_effects.load_attr(self, name)

        if self.is_torch or name not in self.value.__dict__:
            attr_value = getattr(self.value, name)
        else:
            attr_value = self.value.__dict__[name]

        source = self.source and AttrSource(self.source, name)
        return VariableTracker.build(tx, attr_value, source)


class TypingVariable(VariableTracker):
    def __init__(self, value, **kwargs) -> None:
        super().__init__(**kwargs)
        self.value = value

    def call_method(
        self,
        tx,
        name,
        args: "list[VariableTracker]",
        kwargs: "dict[str, VariableTracker]",
    ) -> "VariableTracker":
        # Create a new typing variable, e.g., `List[int]`
        if name == "__getitem__" and len(args) == 1:
            new_typing = self.value[args[0].as_python_constant()]
            return TypingVariable(new_typing)
        unimplemented("unsupported method call on typing variablel")

    def var_getattr(self, tx: "InstructionTranslator", name: str):
        from .builder import SourcelessBuilder, VariableBuilder

        if name in cmp_name_to_op_mapping:
            return variables.GetAttrVariable(self, name)

        if tx.output.side_effects.has_pending_mutation_of_attr(self, name):
            return tx.side_effects.load_attr(self, name)

        value = getattr(self.value, name)
        if self.source:
            attr_source = AttrSource(self.source, name)
            return VariableBuilder(tx, attr_source)(value)
        else:
            return SourcelessBuilder.create(tx, value)

    def as_python_constant(self):
        return self.value

    def reconstruct(self, codegen: "torch._dynamo.codegen.PyCodegen") -> None:
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


@functools.lru_cache(maxsize=1)
def get_np_to_tnp_map():
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
def get_tnp_to_np_map():
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

    def __init__(self, value, **kwargs) -> None:
        super().__init__(**kwargs)
        self.value = value

    @classmethod
    def can_constant_fold_through(cls, fn):
        mod = fn.__module__.split(".")
        assert len(mod) >= 2 and mod[:2] == ["torch", "_numpy"]
        return fn in cls.constant_fold_functions

    @classmethod
    def get_constant_collection_for_func(cls, fn):
        mod = fn.__module__.split(".")
        assert len(mod) >= 2 and mod[:2] == ["torch", "_numpy"]
        return np_constant_collections_map.get(fn, None)

    def call_function(
        self,
        tx: "InstructionTranslator",
        args: "list[VariableTracker]",
        kwargs: "dict[str, VariableTracker]",
    ) -> "VariableTracker":
        if not config.trace_numpy:
            unimplemented(f"numpy.{self.value}()")

        from ..utils import numpy_to_tensor_wrapper
        from .tensor import NumpyNdarrayVariable

        func = get_np_to_tnp_map().get(self.value)
        if func is None:
            unimplemented(
                f"Can't find numpy function {self.value} in torch._numpy. "
                " Please file an issue to request support for this function."
            )

        # We are dealing with a function that produces a const collection type (np.dtype, np.iinfo/np.finfo)
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
            except NotImplementedError:
                unimplemented(
                    f"{self.value.__name__} with non-const args: {args} {kwargs}"
                )
        else:
            if (
                func.__module__ == "torch._numpy.random"
                and config.use_numpy_random_stream
            ):
                msg = f"delegate '{func.__qualname__}' to NumPy itself via "
                msg += f"confg.use_numpy_random_stream={config.use_numpy_random_stream}"
                unimplemented(msg)

            args, kwargs = NumpyNdarrayVariable.patch_args(func.__name__, args, kwargs)

            if self.can_constant_fold_through(func) and (
                check_unspec_or_constant_args(args, kwargs)
            ):
                # constant fold
                return variables.ConstantVariable.create(
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
        tx,
        name,
        args: "list[VariableTracker]",
        kwargs: "dict[str, VariableTracker]",
    ) -> "VariableTracker":
        unimplemented("numpy")

    def as_python_constant(self):
        return self.value

    def as_proxy(self):
        if config.trace_numpy and isinstance(self.value, type):
            # This handles numpy dtype attributes such as np.float32
            # We return a string as we don't want to serialize non-PyTorch objects in the output FX graph
            # In torch/_numpy we normalize strings to their dtypes when the input is a dtype, as NumPy does
            return self.value.__name__

        return super().as_proxy()


# Used to keep track of NULLs pushed on the stack for Python 3.11 function calls
class NullVariable(VariableTracker):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def __repr__(self) -> str:
        return "NullVariable"

    def reconstruct(self, codegen):
        if sys.version_info < (3, 11):
            unimplemented("cannot reconstruct NullVariable in < Python 3.11")
        codegen.append_output(create_instruction("PUSH_NULL"))


class DeletedVariable(VariableTracker):
    """Marker used to implement delattr()"""


class StringFormatVariable(VariableTracker):
    """
    Represents a call to str.format(), we delay calling format until after the graph.
    """

    _nonvar_fields = {"format_string", *VariableTracker._nonvar_fields}

    @classmethod
    def create(cls, format_string, sym_args, sym_kwargs):
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

    def __init__(self, format_string, sym_args, sym_kwargs, **kwargs) -> None:
        super().__init__(**kwargs)
        assert isinstance(format_string, str)
        self.format_string = format_string
        self.sym_args = sym_args
        self.sym_kwargs = sym_kwargs

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.format_string!r}, {self.sym_args!r}, {self.sym_kwargs!r})"

    def reconstruct(self, codegen):
        codegen.add_push_null(
            lambda: codegen.extend_output(
                [
                    codegen.create_load_const(self.format_string),
                    codegen.create_load_attr("format"),
                ]
            ),
            call_function_ex=True,
        )
        codegen(variables.TupleVariable(self.sym_args))
        kwargs = {
            variables.ConstantVariable.create(k): v for k, v in self.sym_kwargs.items()
        }
        codegen(variables.ConstDictVariable(kwargs))
        codegen.append_output(create_instruction("CALL_FUNCTION_EX", arg=1))


class DebuggingVariable(VariableTracker):
    """
    Represents a call to a debugging function like print(), or something
    registered to config.reorderable_logging_functions.
    """

    def __init__(self, value, **kwargs) -> None:
        super().__init__(**kwargs)
        self.value = value

    @staticmethod
    def is_reorderable_logging_function(obj):
        return (
            callable(obj)
            and isinstance(obj, (types.FunctionType, types.BuiltinFunctionType))
            and obj in torch._dynamo.config.reorderable_logging_functions
        )

    def call_function(self, tx: "InstructionTranslator", args, kwargs):
        if tx.export:
            # For export cases, we can just make debugging functions no-ops
            return

        if not self.can_reorder_logs(self.value, args, kwargs):
            unimplemented(
                f"Reordering debugging function {self.value} "
                f"with inputs {args} {kwargs} is not yet implemented."
            )

        tx.debug_locals.append((self, list(args)))

    def reconstruct(self, codegen):
        return self.source.reconstruct(codegen)

    @staticmethod
    def can_reorder_logs(fn, args, kwargs) -> True:
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


class LoggingLoggerVariable(VariableTracker):
    """
    Represents a call to any of logging.Logger methods
    """

    def __init__(self, value, **kwargs) -> None:
        super().__init__(**kwargs)
        self.value = value

    def call_method(
        self,
        tx,
        name,
        args: "list[VariableTracker]",
        kwargs: "dict[str, VariableTracker]",
    ) -> "VariableTracker":
        if tx.export:
            # For export cases, we can just make debugging functions no-ops
            return
        method = getattr(self.value, name, None)
        function = getattr(method, "__func__", None)
        if {method, function}.intersection(torch._dynamo.config.ignore_logger_methods):
            return variables.ConstantVariable.create(None)
        unimplemented(
            "Logger not supported for non-export cases. "
            "To avoid graph breaks caused by logger in compile-mode, it is recommended to"
            " disable logging by adding logging methods to config.ignore_logger_methods"
        )


class ConstantLikeVariable(VariableTracker):
    """self.value is a compile-time constant, but not a literal"""

    _error_prefix = "ConstantLikeVariable"
    try:
        from numpy import (
            dtype as np_dtype,
            floating as np_floating,
            generic as np_generic,
        )
    except ImportError:
        np_floating = type("invalid_type", (), {})
        np_dtype = type("invalid_type", (), {})

    def __init__(self, value, **kwargs) -> None:
        super().__init__(**kwargs)
        self.value = value

    def as_python_constant(self):
        return self.value

    def call_method(
        self,
        tx,
        name,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        try:
            # we only support constant propagation for methods
            cargs = [x.as_python_constant() for x in args]
            ckwargs = {k: v.as_python_constant() for k, v in kwargs.items()}
        except NotImplementedError:
            unimplemented(f"{self._error_prefix}.{name}(*{args}, **{kwargs})")

        result = getattr(self.value, name)(*cargs, **ckwargs)

        if variables.ConstantVariable.is_literal(result):
            return variables.ConstantVariable.create(result)
        if isinstance(result, re.Match):
            return ConstantRegexMatchVariable(result)

        unimplemented(f"{self._error_prefix}.{name}() -> {result}")

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
            return variables.ConstantVariable.create(result)
        return GetAttrVariable(self, name)


class RegexPatternVariable(ConstantLikeVariable):
    _error_prefix = "re.Pattern"


class ConstantRegexMatchVariable(ConstantLikeVariable):
    _error_prefix = "re.Match"


class TorchVersionVariable(ConstantLikeVariable):
    _error_prefix = "torch.__version__"

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("value", torch.__version__)
        assert kwargs["value"] is torch.__version__
        super().__init__(**kwargs)


class NumpyTypeInfoVariable(ConstantLikeVariable):
    _error_prefix = "np.iinfo/np.finfo"


class NumpyDTypeVariable(ConstantLikeVariable):
    _error_prefix = "np.dtype[...]"

    def as_proxy(self):
        """Similar to how numpy dtype descriptors (e.g. np.float32 ) are handled by NumpyVariable:

        np.dtype() objects are serialized as strings, torch._numpy wrappers will normalize to the torch dtype.
        This also handles unsupported things nicely (i.e. structured arrays and object arrays).
        """
        return self.value.type.__name__


np_constant_collections_map = {
    tnp.finfo: NumpyTypeInfoVariable,
    tnp.iinfo: NumpyTypeInfoVariable,
    tnp.dtype: NumpyDTypeVariable,
}


class RandomClassVariable(VariableTracker):
    """random.Random"""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def call_function(self, tx: "InstructionTranslator", args, kwargs):
        if len(args) > 1:
            unimplemented("random.Random() with > 1 arg")
        elif kwargs:
            unimplemented("random.Random() with kwargs")
        seed = variables.ConstantVariable.create(None) if len(args) == 0 else args[0]
        return RandomVariable(
            seed=seed, mutation_type=variables.base.ValueMutationNew()
        )


class RandomVariable(VariableTracker):
    """random.Random()

    Implemented by wrapping a VariableTracker around a random.Random object.
    The supported methods for the random.Random object cannot be overriden.
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
        rand: Optional[random.Random] = None,
        seed: Optional[VariableTracker] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if rand is not None:
            assert self.is_supported_random_obj(rand)
            self.random = random.Random()
            self.random.setstate(rand.getstate())
        else:
            seed = seed.as_python_constant() if seed is not None else None
            self.random = random.Random(seed)

    def python_type(self):
        return random.Random

    def as_python_constant(self):
        return self.random

    @staticmethod
    def is_supported_random_obj(val):
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
    def check_state(state):
        assert type(state) is tuple
        assert type(state[0]) is int
        assert type(state[1]) is tuple
        assert all(type(x) is int for x in state[1])
        assert state[2] is None or type(state[2]) is float

    @staticmethod
    def wrap_state(state):
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
    def unwrap_state(state):
        state_obj = state.as_python_constant()
        RandomVariable.check_state(state_obj)
        return state_obj

    def call_method(
        self,
        tx,
        name,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        if name == "seed":
            tx.output.side_effects.mutation(self)
            self.random.seed(
                *[x.as_python_constant() for x in args],
                **{key: val.as_python_constant() for key, val in kwargs.items()},
            )
            return variables.ConstantVariable.create(None)
        elif name == "getstate":
            return self.wrap_state(self.random.getstate())
        elif name == "setstate":
            tx.output.side_effects.mutation(self)
            self.random.setstate(self.unwrap_state(args[0]))
            return variables.ConstantVariable.create(None)
        elif name in self._supported_fn_names:
            tx.output.side_effects.mutation(self)
            state = self.random.getstate()

            def call_random_meth(*args, **kwargs):
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

    def reconstruct(self, codegen):
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
    def build(tx, weakref_value, **options):
        source = options.get("source", None)
        referent = weakref_value()
        source = source and WeakRefCallSource(source)
        referent_vt = VariableTracker.build(tx, referent, source)
        options["source"] = source
        return WeakRefVariable(referent_vt, **options)

    def __init__(self, referent_vt, **options):
        super().__init__(**options)
        self.referent_vt = referent_vt

    def call_function(
        self,
        tx: "InstructionTranslator",
        args: "list[VariableTracker]",
        kwargs: "dict[str, VariableTracker]",
    ) -> "VariableTracker":
        return self.referent_vt

    def reconstruct(self, codegen):
        codegen.add_push_null(lambda: codegen.load_import_from("weakref", "ref"))
        codegen(self.referent_vt)
        codegen.extend_output(create_call_function(1, False))
