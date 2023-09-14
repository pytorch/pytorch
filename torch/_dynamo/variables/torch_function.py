from torch.overrides import _get_overloaded_args
from torch.utils._pytree import tree_flatten
from ..exc import unimplemented
from ..source import AttrSource
from .base import VariableTracker
from .constant import ConstantVariable
from .lists import TupleVariable
from .user_defined import UserDefinedClassVariable, UserDefinedObjectVariable

# [Note: __torch_function__] This feature is partially supported with many rough edges (contact mlazos with issues):
# The following is not supported:
# - triggering __torch_function__ on tensor subclass attribute access
# - graph breaking on mutating guardable tensor properties within a __torch_function__ context, this can cause
# excessive recompiles in certain degenerate cases
# - Matching the exact eager behavior of *ignoring* __torch_function__ objects in non-tensor argument positions of Torch API calls

# The following is supported:
# - static method impls of __torch_function__ on custom objects; this will trigger on torch API calls with the object as
# any argument
# - triggering __torch_function__ on torch API calls with tensor subclass arguments
# - matches the dispatch ordering behavior of eager __torch_function__ with subclass/object argumnents in any argument position

# To enable subclass behavior, add your tensor subclass type to traceable_tensor_subclasses in dynamo/config.py


def is_torch_function_user_object(obj):
    return hasattr(obj, "__torch_function__") and hasattr(
        type(obj), "__torch_function__"
    )


def call_torch_function(
    tx, torch_function_type, torch_function_var, fn, types, args, kwargs
):
    # signature:
    # def __torch_function__(cls, func, types, args=(), kwargs=None):
    tf_args = (torch_function_type, fn, types, TupleVariable(list(args)))
    return tx.inline_user_function_return(torch_function_var, tf_args, kwargs)


def build_torch_function_var(tx, fn_value, source):
    from .builder import SourcelessBuilder, VariableBuilder

    if source:
        source = AttrSource(
            AttrSource(source, "__torch_function__"),
            "__func__",
        )
        return VariableBuilder(tx, source)(fn_value)
    else:
        return SourcelessBuilder()(tx, fn_value.__func__)


class TorchFunctionObjectVariable(UserDefinedObjectVariable):
    def __init__(self, *args, subclass_type=None, **kwargs):
        super().__init__(*args, **kwargs)
        if not subclass_type:
            self.subclass_type = self.value_type

    def subclass_type_var(self):
        return UserDefinedClassVariable(self.value_type)

    def call_torch_function(self, tx, fn, types, args, kwargs):
        return call_torch_function(
            tx,
            self.subclass_type_var(),
            build_torch_function_var(
                tx, self.value.__torch_function__.__func__, self.source
            ),
            fn,
            types,
            args,
            kwargs,
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
        if var.global_class_name() not in tx.output.global_scope:
            tx.output.install_global(var.global_class_name(), subclass_type)

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

    def torch_function_var(self, tx):
        from .builder import VariableBuilder

        source = AttrSource(
            AttrSource(self.source, "__torch_function__"),
            "__func__",
        )
        return VariableBuilder(tx, source)(self.torch_function_fn)

    def subclass_type_var(self):
        return UserDefinedClassVariable(self.subclass_type)

    def var_getattr(self, tx, name):
        return self.tensor_variable.var_getattr(tx, name)

    def global_class_name(self):
        return f"__subclass_{self.subclass_type.__name__}"

    def call_torch_function(self, tx, fn, types, args, kwargs):
        return call_torch_function(
            tx,
            self.subclass_type_var(),
            build_torch_function_var(tx, self.torch_function_fn, self.source),
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
            from . import GetAttrVariable

            options = VariableTracker.propagate(self, args, kwargs.values())
            args = list(args)
            func_var = GetAttrVariable(self, name)
            return dispatch_torch_function(tx, func_var, args, kwargs)
        else:
            return self.tensor_variable.call_method(tx, name, args, kwargs)


def can_dispatch_torch_function(tx, args, kwargs):
    if tx.output.torch_function_enabled:
        all_args = tree_flatten(args)[0] + tree_flatten(kwargs)[0]
        return any(
            isinstance(arg, (TensorWithTFOverrideVariable, TorchFunctionObjectVariable))
            for arg in all_args
        )
    else:
        return False


def dispatch_torch_function(tx, fn, args, kwargs):
    """Gathers all args that are TensorWithTFOverrideVariable and dispatches based on the ordering in _get_overloaded_args"""

    all_args = args + tree_flatten(kwargs)[0]

    # FIXME: need a better way to detect methods
    # self should not be in args since it gets handled at the method call
    from .misc import GetAttrVariable

    if isinstance(fn, GetAttrVariable):
        all_args = [fn.obj] + all_args

    overloaded_args = _get_overloaded_args(
        [
            arg
            for arg in all_args
            if isinstance(
                arg, (TensorWithTFOverrideVariable, TorchFunctionObjectVariable)
            )
        ],
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
        f"All __torch_function__ overrides for call {fn} with args {args} and kwargs {kwargs} returned NotImplemented"
    )
