from torch.overrides import _get_overloaded_args
from torch.utils._pytree import tree_flatten
from ..exc import unimplemented
from ..source import AttrSource
from .base import VariableTracker
from .constant import ConstantVariable
from .lists import TupleVariable
from .user_defined import UserDefinedClassVariable, UserDefinedObjectVariable


def is_torch_function_user_object(obj):
    return hasattr(obj, "__torch_function__") and hasattr(
        type(obj), "__torch_function__"
    )


class TorchFunctionObjectVariable(UserDefinedObjectVariable):
    pass


class TensorWithTFOverrideVariable(VariableTracker):
    """
    Represents a tensor subclass instance with a __torch_function__ override.
    """

    @staticmethod
    def create(
        tx,
        tensor_variable,
        tensor_variable_source,
        torch_function_fn,
        subclass_type,
        **kwargs,
    ):
        var = TensorWithTFOverrideVariable(
            tensor_variable,
            tensor_variable_source,
            torch_function_fn,
            subclass_type,
            **kwargs,
        )
        # stash the subclass type to rewrap an output tensor if needed
        if var.global_class_name() not in tx.output.global_scope:
            tx.output.install_global(var.global_class_name(), subclass_type)

        return var

    def __init__(
        self,
        tensor_variable,
        tensor_variable_source,
        torch_function_fn,
        subclass_type,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.tensor_variable = tensor_variable
        self.tensor_variable_source = tensor_variable_source
        self.torch_function_fn = torch_function_fn
        self.subclass_type = subclass_type

    def as_proxy(self):
        return self.tensor_variable.as_proxy()

    def python_type(self):
        return self.subclass_type

    def var_getattr(self, tx, name: str) -> VariableTracker:
        return self.tensor_variable.var_getattr(tx, name)

    def torch_function_var(self, tx):
        from .builder import VariableBuilder

        source = AttrSource(
            AttrSource(self.source, "__torch_function__"),
            "__func__",
        )
        return VariableBuilder(tx, source)(self.torch_function_fn)

    def subclass_type_var(self):
        return UserDefinedClassVariable(self.subclass_type)

    def call_method(
        self,
        tx,
        name,
        args: "List[VariableTracker]",
        kwargs: "Dict[str, VariableTracker]",
    ) -> "VariableTracker":
        # This code block implements inlining the __torch_function__ override
        # of `call_method`.
        from . import GetAttrVariable

        options = VariableTracker.propagate(self, args, kwargs.values())
        args = list(args)
        args.insert(0, self)
        func_var = GetAttrVariable(self.tensor_variable, name)

        return TensorWithTFOverrideVariable.inline_torch_function_unwrapped(
            tx,
            func_var,
            self.orig_tensor_variable_source,
            self.subclass_torch_function__func,
            self.subclass_type,
            options,
            args,
            kwargs,
        )

    def global_class_name(self):
        return f"__subclass_{self.subclass_type.__name__}"

    def call_torch_function(self, tx, fn, types, args, kwargs):
        # signature:
        # def __torch_function__(cls, func, types, args=(), kwargs=None):
        tf_args = (
            self.subclass_type_var(),  # cls
            fn,  # func
            types,
            TupleVariable(list(args)),
        )

        return tx.inline_user_function_return(
            self.torch_function_var(tx), tf_args, kwargs
        )


def can_dispatch_torch_function(tx, args, kwargs):
    if tx.output.torch_function_enabled:
        all_args = tree_flatten(args)[0] + tree_flatten(kwargs)[0]
        return any(isinstance(arg, TensorWithTFOverrideVariable) for arg in all_args)
    else:
        return False


def dispatch_torch_function(tx, fn, args, kwargs):
    """Gathers all args that are TensorWithTFOverrideVariable and dispatches based on the ordering in _get_overloaded_args"""

    all_args = args + tree_flatten(kwargs)[0]
    overloaded_args = _get_overloaded_args(
        [arg for arg in all_args if isinstance(arg, TensorWithTFOverrideVariable)],
        lambda x: x.subclass_type,
    )

    for arg in overloaded_args:
        res = arg.call_torch_function(
            tx,
            fn,
            TupleVariable(list({arg.subclass_type_var() for arg in overloaded_args})),
            args,
            kwargs,
        )

        if not (isinstance(res, ConstantVariable) and res.value is NotImplemented):
            return res

    unimplemented(
        f"All __torch_function_overrides__ for call {fn} with args {args} and kwargs {kwargs} returned NotImplemented"
    )
