from torch.overrides import _get_overloaded_args, get_default_nowrap_functions
from torch.utils._pytree import tree_flatten
from ..exc import unimplemented
from ..utils import is_tensor_base_attr_getter
from .base import VariableTracker
from .constant import ConstantVariable
from .lists import TupleVariable

# [Note: __torch_function__] This feature is a prototype and has some rough edges (contact mlazos with issues):
# At a high level, a torch function tensor subclass is represented as a TensorWithTFOverrideVariable, which dispatches
# __torch_function__ on attribute accesses, method calls, and torch API calls.
# The following is not supported:
# - triggering __torch_function__ on tensor subclass non-tensor custom attributes
# - graph breaking on mutating guardable tensor properties within a __torch_function__ context, this can cause
# excessive recompiles in certain degenerate cases
# - Matching the exact eager behavior of *ignoring* __torch_function__ objects in non-tensor argument positions of Torch API calls

# The following is supported:
# - static method impls of __torch_function__ on custom objects; this will trigger on torch API calls with the object as
# any argument
# - triggering __torch_function__ on torch API calls with tensor subclass arguments
# - __torch_function__ calls on base tensor attribute access and method calls for tensor subclass instances
# - matches the dispatch ordering behavior of eager __torch_function__ with subclass/object argumnents in any argument position

# See https://docs.google.com/document/d/1WBxBSvW3NXhRp9ncmtokJloMLCtF4AYNhJaffvHe8Kw/edit#heading=h.vacn73lozd9w
# for more information on the design.

# To enable subclass behavior, add your tensor subclass type to traceable_tensor_subclasses in dynamo/config.py


banned_attrs = [
    fn.__self__.__name__
    for fn in get_default_nowrap_functions()
    if is_tensor_base_attr_getter(fn)
]


def is_torch_function_user_object(obj):
    return hasattr(obj, "__torch_function__")


def call_torch_function(
    tx, torch_function_type, torch_function_var, fn, types, args, kwargs
):
    # signature:
    # def __torch_function__(cls, func, types, args=(), kwargs=None):
    tf_args = (torch_function_type, fn, types, TupleVariable(list(args)))
    return tx.inline_user_function_return(torch_function_var, tf_args, kwargs)


def can_dispatch_torch_function(tx, args, kwargs):
    if tx.output.torch_function_enabled:
        all_args = tree_flatten(args)[0] + tree_flatten(kwargs)[0]
        return any(isinstance(arg, TensorWithTFOverrideVariable) for arg in all_args)
    else:
        return False


def dispatch_torch_function(tx, fn, args, kwargs):
    """Gathers all args that are TensorWithTFOverrideVariable and dispatches based on the ordering in _get_overloaded_args"""

    all_args = tree_flatten(args)[0] + tree_flatten(kwargs)[0]
    overloaded_args = _get_overloaded_args(
        [arg for arg in all_args if isinstance(arg, TensorWithTFOverrideVariable)],
        lambda x: x.subclass_type,
    )

    for arg in overloaded_args:
        res = arg.call_torch_function(
            tx,
            fn,
            TupleVariable([arg.subclass_type_var() for arg in overloaded_args]),
            args,
            kwargs,
        )

        if not (isinstance(res, ConstantVariable) and res.value is NotImplemented):
            return res

    unimplemented(
        f"All __torch_function__ overrides for call {fn} with args {args} and kwargs {kwargs} returned NotImplemented"
    )


class TensorWithTFOverrideVariable(VariableTracker):
    """
    Represents a tensor subclass instance with a __torch_function__ override.
    """

    @staticmethod
    def create(
        tx,
        tensor_variable,
        torch_function_fn,
        subclass_type,
        **kwargs,
    ):
        var = TensorWithTFOverrideVariable(
            tensor_variable,
            torch_function_fn,
            subclass_type,
            **kwargs,
        )
        # stash the subclass type to rewrap an output tensor if needed
        if var.global_mangled_class_name() not in tx.output.global_scope:
            tx.output.install_global(var.global_mangled_class_name(), subclass_type)

        return var

    def __init__(
        self,
        tensor_variable,
        torch_function_fn,
        subclass_type,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.tensor_variable = tensor_variable
        self.torch_function_fn = torch_function_fn
        self.subclass_type = subclass_type

    def as_proxy(self):
        return self.tensor_variable.as_proxy()

    def python_type(self):
        return self.subclass_type

    def subclass_type_var(self):
        from ..source import GlobalSource
        from .user_defined import UserDefinedClassVariable

        return UserDefinedClassVariable(
            self.subclass_type, source=GlobalSource(self.global_mangled_class_name())
        )

    def global_mangled_class_name(self):
        return f"__subclass_{self.subclass_type.__name__}_{id(self.subclass_type)}"

    def call_torch_function(self, tx, fn, types, args, kwargs):
        return call_torch_function(
            tx,
            self.subclass_type_var(),
            self.torch_function_fn,
            fn,
            types,
            args,
            kwargs,
        )

    def call_method(
        self,
        tx,
        name,
        args: "List[VariableTracker]",
        kwargs: "Dict[str, VariableTracker]",
    ) -> "VariableTracker":
        # This code block implements inlining the __torch_function__ override
        # of `call_method`.
        if tx.output.torch_function_enabled:
            import torch
            from .builder import SourcelessBuilder

            # [Note: __torch_function__] Currently we only support methods that are defined on tensor
            # we will graph break in other cases this will need a bigger overhaul of extracting methods/comparing them for equality
            func_var = SourcelessBuilder()(tx, getattr(torch.Tensor, name))
            return dispatch_torch_function(tx, func_var, [self] + args, kwargs)
        else:
            return self.tensor_variable.call_method(tx, name, args, kwargs)
