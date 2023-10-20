from torch.overrides import _get_overloaded_args, get_default_nowrap_functions
from torch.utils._pytree import tree_flatten
from ..exc import unimplemented
from ..utils import is_tensor_base_attr_getter
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
    from .tensor import TensorWithTFOverrideVariable

    if tx.output.torch_function_enabled:
        all_args = tree_flatten(args)[0] + tree_flatten(kwargs)[0]
        return any(isinstance(arg, TensorWithTFOverrideVariable) for arg in all_args)
    else:
        return False


def dispatch_torch_function(tx, fn, args, kwargs):
    """Gathers all args that are TensorWithTFOverrideVariable and dispatches based on the ordering in _get_overloaded_args"""
    from .tensor import TensorWithTFOverrideVariable

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
