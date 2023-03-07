import collections
import functools
import inspect
import types
from typing import Dict, List

import torch._C
from torch._guards import Guard, GuardSource

from .. import variables
from ..bytecode_transformation import create_call_function, create_instruction
from ..exc import unimplemented
from ..guards import GuardBuilder
from ..source import AttrSource
from ..utils import check_constant_args, identity, proxy_args_kwargs
from .base import MutableLocal, VariableTracker
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
            return create_call_function(2, True)
        else:
            return create_call_function(1, True)

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

    def reconstruct(self, codegen):
        attr_source = AttrSource(
            codegen.tx.import_source(self.module_name()), self.fn_name()
        )
        return attr_source.reconstruct(codegen)

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
        # CUDA stream generated inside of traced function
        if self.target_values[0].as_proxy() is not None:
            tx.output.create_proxy(
                "call_function",
                torch.cuda.set_stream,
                (self.target_values[0].as_proxy(),),
                {},
            )
        # CUDA stream passed from outside of traced function
        else:
            stream = self.target_values[0].value
            tx.output.create_proxy(
                "call_function",
                torch._C._cuda_setStream,
                (stream.stream_id, stream.device_index, stream.device_type),
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

    def module_name(self):
        return "torch.cuda"

    def fn_name(self):
        return "stream"


class CUDAStreamVariable(VariableTracker):
    def __init__(self, proxy, value, **kwargs):
        if proxy is not None and "example_value" in proxy.node.meta:
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
                    *create_call_function(len(loads), True),
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

    @staticmethod
    @functools.lru_cache(None)
    def fold_through_function_to_wrapper():
        return {
            collections.namedtuple: variables.UserDefinedClassVariable,
        }

    def call_function(
        self, tx, args: "List[VariableTracker]", kwargs: "Dict[str, VariableTracker]"
    ) -> "VariableTracker":
        from .builtin import BuiltinVariable

        options = VariableTracker.propagate(self, args, kwargs.values())

        if inspect.getattr_static(self.value, "_torchdynamo_disable", False):
            unimplemented(f"call torch._dynamo.disable() wrapped function {self.value}")
        # Allowlist a few popular classes(e.g, collections.OrderedDict) calls in skip files.
        elif self.value is collections.OrderedDict and (
            len(args) == 0
            or len(args) == 1
            and BuiltinVariable.is_supported_call_dict_arg(tx, args[0])
        ):
            return BuiltinVariable.call_dict_helper(
                tx,
                collections.OrderedDict,
                None if len(args) == 0 else args[0],
                **options,
            )
        # Fold through the functions(e.g, collections.namedtuple)
        # that inputs & outputs are all python constants
        elif (
            self.value in self.fold_through_function_to_wrapper().keys()
            and check_constant_args(args, kwargs)
        ):
            value = self.value(
                *[x.as_python_constant() for x in args],
                **{k: v.as_python_constant() for k, v in kwargs.items()},
            )
            return self.fold_through_function_to_wrapper().get(self.value)(
                value, mutable_local=MutableLocal(), **options
            )
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


# Used to keep track of NULLs pushed on the stack for Python 3.11 function calls
class NullVariable(VariableTracker):
    def __init__(self, **kwargs):
        super(NullVariable, self).__init__(**kwargs)

    def __str__(self):
        return "NullVariable"

    def reconstruct(self, codegen):
        if sys.version_info < (3, 11):
            unimplemented("cannot reconstruct NullVariable in < Python 3.11")
        return [create_instruction("PUSH_NULL")]
