import inspect
import sys
import types
from typing import Dict, List

import torch._C
from torch._guards import Guard, GuardSource

from .. import variables
from ..bytecode_transformation import create_instruction
from ..exc import unimplemented
from ..guards import GuardBuilder
from ..source import AttrSource
from ..utils import identity, proxy_args_kwargs
from .base import VariableTracker
from .functions import (
    NestedUserFunctionVariable,
    UserFunctionVariable,
    UserMethodVariable,
    WrappedUserFunctionVariable,
    WrappedUserMethodVariable,
)


class SuperVariable(VariableTracker):
    def __init__(self, typevar, objvar=None, specialized=False, **kwargs):
        super().__init__(**kwargs)
        self.typevar = typevar
        self.objvar = objvar
        self.specialized = specialized  # directly get attr from self.typevar if true

    def reconstruct(self, codegen):
        codegen(variables.BuiltinVariable(super))
        codegen(self.typevar)
        if self.objvar is not None:
            codegen(self.objvar)
            return [create_instruction("CALL_FUNCTION", 2)]
        else:
            return [create_instruction("CALL_FUNCTION", 1)]

    def const_getattr(self, tx, name):
        assert self.objvar, "1-arg super not implemented"
        if self.specialized:
            return getattr(self.typevar.as_python_constant(), name)
        search_type = self.typevar.as_python_constant()

        # We default to the python type of the object. However, if this is
        # a `type` or subclass of `type`, then the original object represents
        # the user defined type.
        type_to_use = self.objvar.python_type()
        if issubclass(type_to_use, type):
            type_to_use = self.objvar.value

        # TODO(jansel): there is a small chance this could trigger user code, prevent that
        return getattr(super(search_type, type_to_use), name)

    def call_method(
        self,
        tx,
        name,
        args: "List[VariableTracker]",
        kwargs: "Dict[str, VariableTracker]",
    ) -> "VariableTracker":
        options = VariableTracker.propagate(
            self, args, kwargs.values(), self.objvar, self.typevar
        )
        inner_fn = self.const_getattr(self, name)
        source = None if self.source is None else AttrSource(self.source, name)
        if inner_fn is object.__init__:
            return LambdaVariable(identity, **options)
        elif isinstance(inner_fn, types.FunctionType):
            return variables.UserFunctionVariable(
                inner_fn, source=source, **options
            ).call_function(tx, [self.objvar] + args, kwargs)
        elif isinstance(inner_fn, types.MethodType):
            return variables.UserMethodVariable(
                inner_fn.__func__, self.objvar, source=source, **options
            ).call_function(tx, args, kwargs)
        else:
            unimplemented(f"non-function or method super: {inner_fn}")


class UnknownVariable(VariableTracker):
    """
    It could be anything!
    """


class ComptimeVariable(VariableTracker):
    """
    This variable is special, it lets you execute arbitrary code at
    Dynamo compile time
    """

    def reconstruct(self, codegen):
        raise NotImplementedError("comptime is special form")

    def var_getattr(self, tx, name: str) -> "VariableTracker":
        from ..comptime import comptime

        # To support the comptime.print_graph convenience accessors
        from .functions import UserFunctionVariable

        return UserFunctionVariable(
            getattr(comptime, name), source=AttrSource(self.source, name)
        )

    def call_function(
        self, tx, args: "List[VariableTracker]", kwargs: "Dict[str, VariableTracker]"
    ) -> "VariableTracker":
        from ..comptime import ComptimeContext

        # TODO: support an expression form as well

        assert not kwargs
        assert len(args) == 1
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
                tuple(),
            )
            func(ComptimeContext(tx))
        else:
            raise RuntimeError(f"unsupported argument to comptime: {type(fn)}")

        return variables.ConstantVariable(None)


class ClosureVariable(UnknownVariable):
    def __init__(self, name, **kwargs):
        super().__init__(**kwargs)
        self.name = name

    def reconstruct(self, codegen):
        return [codegen.create_load_closure(self.name)]


class NewCellVariable(VariableTracker):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class NewGlobalVariable(VariableTracker):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class ContextWrappingVariable(VariableTracker):
    def __init__(self, target_values, initial_values=None, **kwargs):
        super().__init__(**kwargs)
        self.target_values = target_values
        self.initial_values = initial_values
        self.recursively_contains = (
            set()
        )  # This var doesn't contain any child vars and doesn't support clone() properly,
        # so don't populate this automatically

    def enter(self, tx):
        self._call_func(tx, self.target_values)
        return variables.ConstantVariable(None, **VariableTracker.propagate(self))

    def exit(self, tx, *args):
        self._call_func(tx, self.initial_values)
        return variables.ConstantVariable(None, **VariableTracker.propagate(self))

    def reconstruct(self, codegen, target_inst=None):
        """
        Generate following Python Bytecode, with a `torch._C._set_grad_enable` call
        Python 3.8
             0 LOAD_GLOBAL              0 (torch)
             2 LOAD_ATTR                1 (_C)
             4 LOAD_METHOD              2 (_set_grad_enable)
             6 LOAD_CONST               1 (False)
             8 CALL_METHOD              1
            10 POP_TOP

            12 SETUP_FINALLY           10 (to 24)

            14 LOAD_GLOBAL              3 (user_inst)
            16 CALL_FUNCTION            0
            18 POP_TOP
            20 POP_BLOCK
            22 BEGIN_FINALLY

            24 LOAD_GLOBAL              0 (torch)
            26 LOAD_ATTR                1 (_C)
            28 LOAD_METHOD              2 (_set_grad_enable)
            30 LOAD_CONST               2 (True)
            32 CALL_METHOD              1
            34 POP_TOP
            36 END_FINALLY
            38 LOAD_CONST               0 (None)
            40 RETURN_VALUE

        Instructions 0-10 and 24-34 call torch._C.set_grad_enable(True/False)

        Python 3.9, 3.10
             0 LOAD_GLOBAL              0 (torch)
             2 LOAD_ATTR                1 (_C)
             4 LOAD_METHOD              2 (_set_grad_enable)
             6 LOAD_CONST               1 (False)
             8 CALL_METHOD              1
            10 POP_TOP

            12 SETUP_FINALLY           22 (to 36)

            14 LOAD_GLOBAL              3 (user_inst)
            16 CALL_FUNCTION            0
            18 POP_TOP
            20 POP_BLOCK

            22 LOAD_GLOBAL              0 (torch)
            24 LOAD_ATTR                1 (_C)
            26 LOAD_METHOD              2 (_set_grad_enable)
            28 LOAD_CONST               2 (True)
            30 CALL_METHOD              1
            32 POP_TOP

            34 JUMP_FORWARD            14 (to 50)

            36 LOAD_GLOBAL              0 (torch)
            38 LOAD_ATTR                1 (_C)
            40 LOAD_METHOD              2 (_set_grad_enable)
            42 LOAD_CONST               2 (True)
            44 CALL_METHOD              1
            46 POP_TOP
            48 RERAISE

            50 LOAD_CONST               0 (None)
            52 RETURN_VALUE

        """
        if self.target_values == self.initial_values:
            return ([], [])

        def set_context_insts(values):
            attr_source = AttrSource(
                codegen.tx.import_source(self.module_name()), self.fn_name()
            )
            load_set_context_enabling_insts = attr_source.reconstruct(codegen)

            if values:
                loads = [codegen.create_load_const(val) for val in values]
            else:
                loads = []

            return [
                *load_set_context_enabling_insts,
                *loads,
                create_instruction("CALL_FUNCTION", len(loads)),
                create_instruction("POP_TOP"),
            ]

        init_block = set_context_insts(self.target_values)
        finally_block = set_context_insts(self.initial_values)
        setup_final_inst = create_instruction("SETUP_FINALLY", target=finally_block[0])
        prologue = init_block + [setup_final_inst]

        # Generate the epilogue - starts with 20 POP_BLOCK and ends at 34 POP_TOP
        if sys.version_info < (3, 9):
            # Generate the prologue that ends with setup_finally
            epilogue = [
                create_instruction("POP_BLOCK"),
                codegen.create_begin_finally(),
                *finally_block,
                create_instruction("END_FINALLY"),
            ]
        else:
            except_block = set_context_insts(self.initial_values)
            epilogue = [
                create_instruction("POP_BLOCK"),
                *except_block,
                create_instruction("JUMP_FORWARD", target=target_inst),
                *finally_block,
                create_instruction("RERAISE"),
            ]

        return (prologue, epilogue)

    def _call_func(self, tx, initial_values):
        raise NotImplementedError("_call_func called on base")

    def module_name(self):
        raise NotImplementedError("module_name called on base")

    def fn_name(self):
        raise NotImplementedError("fn_name called on base")

    def call_function(
        self, tx, args: "List[VariableTracker]", kwargs: "Dict[str, VariableTracker]"
    ) -> "VariableTracker":
        assert len(args) == 1
        if isinstance(args[0], NestedUserFunctionVariable):
            args[0] = UserFunctionVariable(args[0].get_function())
        assert isinstance(args[0], UserMethodVariable) or isinstance(
            args[0], UserFunctionVariable
        )

        if isinstance(args[0], UserMethodVariable):
            return WrappedUserMethodVariable(args[0], self)

        if isinstance(args[0], UserFunctionVariable):
            return WrappedUserFunctionVariable(args[0], self)


class GradModeVariable(ContextWrappingVariable):
    """represents torch.{no_grad,enable_grad,set_grad_mode}()"""

    _guards_singleton = {Guard("", GuardSource.GLOBAL, GuardBuilder.GRAD_MODE)}

    @staticmethod
    def create(tx, target_value, **kwargs):
        var = GradModeVariable(
            target_values=[target_value],
            initial_values=[torch.is_grad_enabled()],
            **kwargs,
        )
        var._call_func(tx, [target_value])
        return var

    def __init__(self, target_values, initial_values=None, **kwargs):
        super().__init__(
            target_values=target_values, initial_values=initial_values, **kwargs
        )
        self.guards = self.guards | self._guards_singleton

    def enter(self, tx):
        return variables.ConstantVariable(None, **VariableTracker.propagate(self))

    def _call_func(self, tx, values):
        assert len(values) == 1
        value = values[0]
        tx.output.create_node(
            "call_function", torch._C._set_grad_enabled, (value,), {}
        ),
        torch._C._set_grad_enabled(value)

    def module_name(self):
        return "torch"

    def fn_name(self):
        return "set_grad_enabled"


class AutocastModeVariable(ContextWrappingVariable):
    @staticmethod
    def create(target_values, kwargs):
        # device_type : str,
        # dtype : Optional[_dtype] = None,
        # enabled : bool = True,
        # cache_enabled : Optional[bool] = None):cache_enabled
        bound_args = inspect.signature(torch.autocast).bind(*target_values, **kwargs)
        bound_args.apply_defaults()
        target_values = []
        kwargs.clear()

        for key in ["device_type", "dtype", "enabled", "cache_enabled"]:
            arg = bound_args.arguments[key]
            if isinstance(arg, VariableTracker):
                target_values.append(bound_args.arguments[key].as_python_constant())
            else:
                target_values.append(bound_args.arguments[key])

        var = AutocastModeVariable(target_values, initial_values=None, **kwargs)
        return var

    def __init__(self, target_values, initial_values=None, **kwargs):
        mode = kwargs.pop("mode", None)
        super().__init__(
            target_values=target_values, initial_values=initial_values, **kwargs
        )
        self.target_values = target_values
        self.mode = mode

    def exit(self, tx, *args):
        self.mode = tx.output.create_node(
            "call_function", exit_functional_autocast, (self.mode,), {}
        )

    def enter(self, tx):
        self.mode = tx.output.create_node(
            "call_function", enter_functional_autocast, (*self.target_values,), {}
        )

    def module_name(self):
        return "torch.amp.autocast_mode"

    def fn_name(self):
        return "autocast"


def enter_functional_autocast(*vals):
    mode = torch.amp.autocast(*vals)
    mode.__enter__()
    return mode


def exit_functional_autocast(mode):
    mode.__exit__(None, None, None)


class NullContextVariable(ContextWrappingVariable):
    """
    This class represents Python contextlib.nullcontext.
    It's used as a placeholder for other context managers that Dynamo doesn't
    support yet, e.g, torch.autograd.profiler.record_function.
    """

    def __init__(self, target_values=None, **kwargs):
        super().__init__(target_values=target_values, **kwargs)

    def enter(self, tx):
        return variables.ConstantVariable(None, **VariableTracker.propagate(self))

    def exit(self, tx, *args):
        return variables.ConstantVariable(None, **VariableTracker.propagate(self))

    def module_name(self):
        return "contextlib"

    def fn_name(self):
        return "nullcontext"


class CUDAStreamContextVariable(ContextWrappingVariable):
    @staticmethod
    def create(tx, target_value, **kwargs):
        from .builder import wrap_fx_proxy_cls

        current_stream = wrap_fx_proxy_cls(
            CUDAStreamVariable,
            tx,
            tx.output.create_proxy(
                "call_function",
                torch.cuda.current_stream,
                (None,),
                {},
            ),
        )
        return CUDAStreamContextVariable(
            target_values=[target_value],
            initial_values=[current_stream],
            **kwargs,
        )

    def __init__(self, target_values, initial_values=None, **kwargs):
        super().__init__(
            target_values=target_values, initial_values=initial_values, **kwargs
        )

    def enter(self, tx):
        tx.output.create_proxy(
            "call_function",
            torch.cuda.set_stream,
            (self.target_values[0].as_proxy(),),
            {},
        )
        torch.cuda.set_stream(self.target_values[0].value)

    def exit(self, tx, *args):
        tx.output.create_proxy(
            "call_function",
            torch.cuda.set_stream,
            (self.initial_values[0].as_proxy(),),
            {},
        )
        torch.cuda.set_stream(self.initial_values[0].value)

    def fn_name(self):
        return "cuda.stream"


class CUDAStreamVariable(VariableTracker):
    def __init__(self, proxy, value, **kwargs):
        if "example_value" in proxy.node.meta:
            assert proxy.node.meta["example_value"] == value
        super().__init__(**kwargs)
        self.proxy = proxy
        self.value = value

    def call_method(
        self,
        tx,
        name,
        args: "List[VariableTracker]",
        kwargs: "Dict[str, VariableTracker]",
    ) -> "VariableTracker":
        unimplemented("cuda stream")

    def as_proxy(self):
        return self.proxy


class WithExitFunctionVariable(VariableTracker):
    def __init__(self, ctx: ContextWrappingVariable, target, **kwargs):
        super().__init__(**kwargs)
        assert isinstance(ctx, ContextWrappingVariable)
        self.ctx = ctx
        self.target = target

    def call_function(
        self, tx, args: "List[VariableTracker]", kwargs: "Dict[str, VariableTracker]"
    ) -> "VariableTracker":
        assert not kwargs
        return self.ctx.exit(tx, *args)

    def reconstruct(self, codegen):
        # Note here we reconstruct the context manager rather than the
        # exit function.  The handler generated by BlockStackEntry
        # will re-enter the context in the resume function.
        output = AttrSource(
            codegen.tx.import_source(self.ctx.module_name()), self.ctx.fn_name()
        ).reconstruct(codegen)

        if codegen.tx.output.partial_convert:
            loads = [codegen.create_load_const(val) for val in self.ctx.target_values]
            output.extend(loads)
            output.extend(
                [
                    create_instruction("CALL_FUNCTION", len(loads)),
                    create_instruction("SETUP_WITH", target=self.target),
                    create_instruction("POP_TOP"),
                ]
            )
        return output


class InspectSignatureVariable(VariableTracker):
    """represents inspect.signature(...)"""

    @staticmethod
    def create(callable, **kwargs):
        if kwargs:
            unimplemented(f"inspect.signature with {kwargs}")
        return InspectSignatureVariable(callable)

    def __init__(self, inspected, **kwargs):
        super().__init__(**kwargs)
        self.inspected = inspected


class AutogradFunctionVariable(VariableTracker):
    """represents a torch.autograd.Function subclass"""

    def __init__(self, fn_cls, **kwargs):
        super().__init__(**kwargs)
        self.fn_cls = fn_cls

    def call_apply(self, tx, args, kwargs):
        requires_grad = False

        def visit(node):
            nonlocal requires_grad
            if isinstance(node, variables.TensorVariable):
                if node.requires_grad is not False:
                    requires_grad = True
            if isinstance(node, variables.NNModuleVariable):
                if node.is_training(tx):
                    requires_grad = True
            return node

        VariableTracker.apply(visit, (args, kwargs))

        if requires_grad and torch.is_grad_enabled():
            # TODO(jansel): handle this in training mode
            unimplemented("autograd.Function with requires_grad")

        args = [BlackHoleVariable()] + list(args)
        options = VariableTracker.propagate(self, args, kwargs.values())
        options["source"] = AttrSource(AttrSource(self.source, "__class__"), "forward")
        fn = self.fn_cls.forward
        if isinstance(fn, types.FunctionType):
            return variables.UserFunctionVariable(fn, **options).call_function(
                tx, args, kwargs
            )
        elif isinstance(fn, types.MethodType):
            return variables.UserMethodVariable(
                fn.__func__, variables.UserDefinedClassVariable(self.fn_cls), **options
            ).call_function(tx, args, kwargs)
        else:
            unimplemented(
                f"non-function or method in subclass of torch.autograd.Function: {fn}"
            )

    def call_function(self, tx, args, kwargs):
        options = VariableTracker.propagate(self, args, kwargs.values())
        return AutogradFunctionVariable(self.fn_cls, source=self.source, **options)


class BlackHoleVariable(VariableTracker):
    """A autograd.function context that just ignores everything (for forward extraction)"""

    def call_method(
        self,
        tx,
        name,
        args: "List[VariableTracker]",
        kwargs: "Dict[str, VariableTracker]",
    ) -> "VariableTracker":
        assert name in ("__setattr__", "save_for_backward"), name
        return variables.ConstantVariable(
            None, **VariableTracker.propagate(self, args, kwargs.values())
        )


class AutogradFunctionContextVariable(VariableTracker):
    """
    A autograd.function context used after graph break in forward.
    Any call method on this context object will be graph break.
    The is different from BlackHoleVariable which is only used in inference mode.
    """

    pass


class LambdaVariable(VariableTracker):
    def __init__(self, fn, **kwargs):
        super().__init__(**kwargs)
        self.fn = fn

    def call_function(
        self, tx, args: "List[VariableTracker]", kwargs: "Dict[str, VariableTracker]"
    ) -> "VariableTracker":
        return self.fn(*args, **kwargs).add_options(self)


class GetAttrVariable(VariableTracker):
    def __init__(self, obj, name, **kwargs):
        super().__init__(**kwargs)
        assert isinstance(obj, VariableTracker)
        assert isinstance(name, str)
        self.obj = obj
        self.name = name

    def __str__(self):
        return f"{self.__class__.__name__}({self.obj}, {self.name})"

    @staticmethod
    def create_getattr_proxy(base_proxy: torch.fx.Proxy, attr):
        return getattr(base_proxy, attr)

    def as_proxy(self):
        return GetAttrVariable.create_getattr_proxy(self.obj.as_proxy(), self.name)

    def const_getattr(self, tx, name):
        if not isinstance(self.obj, variables.NNModuleVariable):
            raise NotImplementedError()
        step1 = tx.output.get_submodule(self.obj.module_key)
        if self.name not in step1.__dict__:
            raise NotImplementedError()
        step2 = inspect.getattr_static(step1, self.name)
        if name not in step2.__dict__:
            raise NotImplementedError()
        return inspect.getattr_static(step2, name)

    def reconstruct(self, codegen):
        codegen(self.obj)
        return codegen.create_load_attrs(self.name)

    def call_function(
        self, tx, args: "List[VariableTracker]", kwargs: "Dict[str, VariableTracker]"
    ) -> "VariableTracker":
        from .builder import wrap_fx_proxy

        # This variable is True when it corresponds to user code such as
        #
        #   super().__torch_function__(...)
        #
        # and the super().__torch_function__ attribute resolves
        # to torch.Tensor.__torch_function__.
        is_original_tensor_torch_function = (
            self.name == "__torch_function__"
            and isinstance(self.obj, SuperVariable)
            # for now, only support one level of inheritance
            and len(self.obj.objvar.value.__mro__) > 1
            and self.obj.objvar.value.__mro__[1] == torch.Tensor
        )
        if is_original_tensor_torch_function:
            # Instead of tracing inside torch.Tensor.__torch_function__,
            # record the `call_function` or `call_method` call into the graph.
            from . import TorchVariable

            original_torch_or_getattr_variable = args[0]
            new_args = args[2].items
            new_kwargs = args[3].items
            options = VariableTracker.propagate(self, new_args, new_kwargs.values())
            # Disable __torch_function__ here to prevent the clone of the
            # example tensor from going into the override.
            with torch._C.DisableTorchFunctionSubclass():
                if isinstance(args[0], TorchVariable):
                    return wrap_fx_proxy(
                        tx=tx,
                        proxy=tx.output.create_proxy(
                            "call_function",
                            original_torch_or_getattr_variable.value,
                            *proxy_args_kwargs(new_args, new_kwargs),
                        ),
                        **options,
                    )
                elif isinstance(args[0], GetAttrVariable):
                    return wrap_fx_proxy(
                        tx=tx,
                        proxy=tx.output.create_proxy(
                            "call_method",
                            original_torch_or_getattr_variable.name,
                            *proxy_args_kwargs(new_args, new_kwargs),
                        ),
                        **options,
                    )
                else:
                    unimplemented(
                        f"GetAttrVariable.call_function original __torch_function__ {args}"
                    )

        if isinstance(self.obj, AutogradFunctionVariable) and self.name == "apply":
            return self.obj.call_apply(tx, args, kwargs).add_options(self)
        # calling parent class‘s non classmethod from child class
        # https://github.com/pytorch/pytorch/issues/90558
        elif (
            isinstance(self.obj, variables.UserDefinedClassVariable)
            and len(args) > 0
            and issubclass(args[0].python_type(), self.obj.value)
        ):
            return SuperVariable(self.obj, args[0], True).call_method(
                tx, self.name, args[1:], kwargs
            )
        return self.obj.call_method(tx, self.name, args, kwargs).add_options(self)

    def call_method(
        self,
        tx,
        name,
        args: "List[VariableTracker]",
        kwargs: "Dict[str, VariableTracker]",
    ) -> "VariableTracker":
        if (
            name == "__len__"
            and isinstance(self.obj, InspectSignatureVariable)
            and self.name == "parameters"
        ):
            return variables.ConstantVariable(
                self.obj.inspected.num_parameters(),
                **VariableTracker.propagate(self, self.obj, self.obj.inspected),
            )
        return super().call_method(tx, name, args, kwargs)


class PythonModuleVariable(VariableTracker):
    def __init__(self, value: types.ModuleType, **kwargs):
        super().__init__(**kwargs)
        self.value = value

    def python_type(self):
        return types.ModuleType


class SkipFilesVariable(VariableTracker):
    def __init__(self, value, **kwargs):
        super().__init__(**kwargs)
        self.value = value

    def python_type(self):
        return type(self.value)

    def as_python_constant(self):
        return self.value

    def call_function(
        self, tx, args: "List[VariableTracker]", kwargs: "Dict[str, VariableTracker]"
    ) -> "VariableTracker":
        if inspect.getattr_static(self.value, "_torchdynamo_disable", False):
            unimplemented(f"call torch._dynamo.disable() wrapped function {self.value}")
        else:
            try:
                path = inspect.getfile(self.value)
            except TypeError:
                path = f"Builtin {self.value.__name__}"
            unimplemented(
                f"call_function {self.value.__qualname__} in skip_files {path}"
            )


class TypingVariable(VariableTracker):
    def __init__(self, value, **kwargs):
        super().__init__(**kwargs)
        self.value = value

    def call_method(
        self,
        tx,
        name,
        args: "List[VariableTracker]",
        kwargs: "Dict[str, VariableTracker]",
    ) -> "VariableTracker":
        if name == "__getitem__" and len(args) == 1:
            return variables.ConstantVariable(
                self.value[args[0].as_python_constant()],
                **VariableTracker.propagate(self, args),
            )
        unimplemented("typing")

    def python_type(self):
        return type(self.value)

    def as_python_constant(self):
        return self.value


class NumpyVariable(VariableTracker):
    """
    Wrapper around `numpy.*` for better error messages.
    """

    def __init__(self, value, **kwargs):
        super().__init__(**kwargs)
        self.value = value

    def call_function(
        self, tx, args: "List[VariableTracker]", kwargs: "Dict[str, VariableTracker]"
    ) -> "VariableTracker":
        unimplemented("numpy")

    def call_method(
        self,
        tx,
        name,
        args: "List[VariableTracker]",
        kwargs: "Dict[str, VariableTracker]",
    ) -> "VariableTracker":
        unimplemented("numpy")

    def python_type(self):
        return type(self.value)

    def as_python_constant(self):
        return self.value
