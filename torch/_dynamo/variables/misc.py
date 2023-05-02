import collections
import functools
import inspect
import itertools
import sys
import types
from typing import Dict, List

import torch._C
from .. import config, variables
from ..bytecode_transformation import create_call_function, create_instruction
from ..exc import unimplemented
from ..source import AttrSource, ODictGetItemSource
from ..utils import (
    check_constant_args,
    HAS_NUMPY_TORCH_INTEROP,
    identity,
    proxy_args_kwargs,
)
from .base import MutableLocal, VariableTracker
from .functions import NestedUserFunctionVariable, UserFunctionVariable
from .user_defined import UserDefinedObjectVariable


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
        elif inner_fn is torch.nn.Module.__init__:
            objvar = self.objvar
            from ..side_effects import AttributeMutationNew

            if (
                isinstance(objvar, variables.UserDefinedObjectVariable)
                and isinstance(objvar.mutable_local, AttributeMutationNew)
                and not (args or kwargs)
            ):
                tx.output.guards.update(options.get("guards", set()))
                tx.output.side_effects.store_attr(
                    objvar, "__call_nn_module_init", variables.ConstantVariable(True)
                )
                return variables.ConstantVariable(None)
            else:
                unimplemented("super() nn.Module.__init__")
        elif isinstance(inner_fn, types.FunctionType):
            return variables.UserFunctionVariable(
                inner_fn, source=source, **options
            ).call_function(tx, [self.objvar] + args, kwargs)
        elif isinstance(inner_fn, types.MethodType):
            return variables.UserMethodVariable(
                inner_fn.__func__, self.objvar, source=source, **options
            ).call_function(tx, args, kwargs)
        elif (
            inner_fn is collections.OrderedDict.__getitem__
            and isinstance(self.objvar, variables.UserDefinedObjectVariable)
            and self.objvar.source
            and len(args) == 1
            and len(kwargs) == 0
            and args[0].is_python_constant()
        ):
            from .builder import VariableBuilder

            key = args[0].as_python_constant()
            return VariableBuilder(tx, ODictGetItemSource(self.objvar.source, key))(
                collections.OrderedDict.__getitem__(self.objvar.value, key)
            )
        else:
            unimplemented(f"non-function or method super: {inner_fn}")


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

        ctx = AutogradFunctionContextVariable.create(tx)
        args = [ctx, *args]

        if requires_grad and torch.is_grad_enabled():
            from .torch import is_fn_safe_to_run, TorchHigherOrderOperator

            def trampoline_autograd_apply(*args, **kwargs):
                return self.fn_cls.apply(*args, **kwargs)

            def trampoline_autograd_fwd(*args, **kwargs):
                return self.fn_cls.forward(*args, **kwargs)

            def trampoline_autograd_bwd(*args, **kwargs):
                return self.fn_cls.backward(*args, **kwargs)

            # Speculate fwd, will raise unimplemented and bubble up if not sound, or will accumulate guards
            # onto tx if sound.
            # TODO(voz): NOTE: This function identity is unguarded, but the odds of someone swapping self.fn_cls from
            # autograd fn to something else is very low.
            higher_order_autograd_fn = TorchHigherOrderOperator(trampoline_autograd_fwd)
            speculated_fwd_result = higher_order_autograd_fn.call_function(
                tx, args, kwargs
            )
            bwd_args = [ctx, speculated_fwd_result]
            # ctx.value.saved_tensors = ctx.value.to_save
            if not is_fn_safe_to_run(
                tx, TorchHigherOrderOperator(trampoline_autograd_bwd), bwd_args
            ):
                unimplemented("Unsafe bwd in autograd.function")

            # If fwd and backward are sound, we want apply in the graph.
            # We do not want forward, because doing so messes with the versioning of tensors for grad in bwd.
            # And we don't want backwards for the obvious reasons.
            args = args[1:]
            return TorchHigherOrderOperator(trampoline_autograd_apply).call_function(
                tx, args, kwargs
            )

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


class SaveSimulatingAutogradFunctionContext(torch.autograd.function.FunctionCtx):
    def __init__(self):
        super().__init__()
        self.saved_tensors = []

    def save_for_backward(self, *tensors: torch.Tensor):
        super().save_for_backward(tensors)
        self.saved_tensors.extend([*tensors])


class AutogradFunctionContextVariable(UserDefinedObjectVariable):
    """
    Tracks an autograd.Function() context using mutation tracking in side_effects.py
    """

    def __init__(self, value, value_type=None, inference=False, **kwargs):
        super().__init__(value=value, value_type=value_type, **kwargs)
        self.inference = inference

    @staticmethod
    def create(tx):
        out = tx.output.side_effects.track_object_new(
            None,
            SaveSimulatingAutogradFunctionContext,
            functools.partial(AutogradFunctionContextVariable, inference=True),
            {},
        )
        out.proxy = tx.output.create_proxy(
            "call_function", SaveSimulatingAutogradFunctionContext, tuple(), {}
        )
        out.proxy.node.meta["example_value"] = out.value
        return out

    def as_proxy(self):
        return self.proxy

    def call_method(
        self,
        tx,
        name,
        args: "List[VariableTracker]",
        kwargs: "Dict[str, VariableTracker]",
    ) -> "VariableTracker":
        if name != "save_for_backward":
            unimplemented(f"autograd.Function context method: {name}")

        if not self.inference:
            assert self.source and not kwargs
            tx.output.side_effects.track_save_for_backward(self, args)

        return variables.ConstantVariable(
            None, **VariableTracker.propagate(self, args, kwargs.values())
        )

    def var_getattr(self, tx, name):
        if name == "save_for_backward":
            return LambdaVariable(
                lambda *args, **kwargs: self.call_method(tx, name, args, kwargs)
            ).add_options(self)
        return super().var_getattr(tx, name)


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
        elif (
            self.value is itertools.product
            and not kwargs
            and all(arg.has_unpack_var_sequence(tx) for arg in args)
        ):
            seqs = [arg.unpack_var_sequence(tx) for arg in args]
            items = []
            for item in itertools.product(*seqs):
                items.append(variables.TupleVariable(list(item), **options))
            return variables.ListIteratorVariable(
                items, mutable_local=MutableLocal(), **options
            )
        elif (
            self.value is functools.wraps
            and not kwargs
            and len(args) == 1
            and args[0].source
        ):

            def wraps(fn):
                if isinstance(fn, variables.NestedUserFunctionVariable):
                    return fn.clone(wraps_source=args[0].source)
                unimplemented(f"functools.wraps({fn})")

            return variables.LambdaVariable(wraps, **options)
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
        if not config.numpy_ndarray_as_tensor or not HAS_NUMPY_TORCH_INTEROP:
            unimplemented(f"numpy.{self.value}()")
        import torch_np

        from .builder import wrap_fx_proxy_cls
        from .tensor import NumpyNdarrayVariable

        options = VariableTracker.propagate([[self]], [args], [list(kwargs.values())])
        # lookup method name in torch_np
        if hasattr(torch_np, self.value.__name__):
            func = getattr(torch_np, self.value.__name__)
            return wrap_fx_proxy_cls(
                target_cls=NumpyNdarrayVariable,
                tx=tx,
                proxy=tx.output.create_proxy(
                    "call_function",
                    func,
                    *proxy_args_kwargs(args, kwargs),
                ),
                example_value=None,
                **options,
            )
        else:
            unimplemented(f"Can't find numpy function {self.value} in torch_np")

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


class DeletedVariable(VariableTracker):
    """Marker used to implement delattr()"""
