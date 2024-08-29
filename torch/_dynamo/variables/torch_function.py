# mypy: ignore-errors

import inspect
from typing import Dict, List, TYPE_CHECKING

import torch.utils._pytree as pytree
from torch._guards import Source
from torch.overrides import _get_overloaded_args, get_default_nowrap_functions
from torch.utils._device import DeviceContext

from ..exc import unimplemented
from ..guards import GuardBuilder, install_guard
from ..source import AttrSource, GlobalSource, TorchFunctionModeStackSource, TypeSource
from ..utils import get_safe_global_name, has_torch_function, is_tensor_base_attr_getter
from .base import VariableTracker
from .constant import ConstantVariable
from .ctx_manager import ContextWrappingVariable
from .lists import TupleVariable
from .tensor import TensorSubclassVariable, TensorVariable
from .user_defined import UserDefinedObjectVariable


if TYPE_CHECKING:
    from torch._dynamo.symbolic_convert import InstructionTranslator


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

# Today set default device is placed in the graph and guarded on separately
# so we should not trace through it. In the future we can trace it once
# mode tracing is implemented and not put in the graph, but this is more
# of a BE project and can be evaluated later
IGNORED_MODES = {DeviceContext}


class TorchFunctionModeStackVariable(VariableTracker):
    """Fake VT to use as a dummy object, indicating the presence of torch function mode stack mutation"""

    # singleton value representing the global torch function mode stack
    # singleton (it exists in C++)
    stack_value_singleton = object()

    # offset is used to track if we have inserted/removed a
    # device context which is always placed at the bottom of the stack
    # if a device context is inserted, the graph will run this mutation
    # so when we want to reconstruct any other modes on the stack
    # their indices should be shifted right by 1 (+1)
    # Conversely, if there was a device context on the stack, and the graph
    # mutates the stack to remove that context (set default device to None)
    # each of the indices of other modes should be shifted left by 1 (-1)
    offset = 0

    def __init__(self, source, symbolic_stack):
        self.source = source
        self.symbolic_stack = symbolic_stack

    @classmethod
    def reset(cls):
        cls.offset = 0

    @classmethod
    def register_mutation(cls, tx: "InstructionTranslator"):
        if cls.stack_value_singleton not in tx.output.side_effects:
            var = cls(
                source=Source(), symbolic_stack=tx.symbolic_torch_function_mode_stack
            )
            tx.output.side_effects.track_mutable(cls.stack_value_singleton, var)
            tx.output.side_effects.mutation(var)

    @classmethod
    def register_device_context_insertion(cls, tx: "InstructionTranslator"):
        stack = tx.symbolic_torch_function_mode_stack
        if stack and cls.is_device_context(stack[0]):
            return
        else:
            cls.offset += 1
            tx.symbolic_torch_function_mode_stack.insert(
                0,
                TorchFunctionModeVariable(
                    None, source=TorchFunctionModeStackSource(-cls.offset)
                ),
            )

    @classmethod
    def clear_default_device(cls, tx: "InstructionTranslator"):
        stack = tx.symbolic_torch_function_mode_stack
        if stack and cls.is_device_context(stack[0]):
            stack.popleft()
            cls.offset -= 1

    @staticmethod
    def is_device_context(var):
        return isinstance(var.value, DeviceContext) or var.value is None

    @classmethod
    def get_mode_index(cls, ind):
        return ind + cls.offset


class TorchFunctionModeVariable(ContextWrappingVariable):
    def __init__(self, value, source=None, **kwargs):
        super().__init__(value, **kwargs)
        assert source
        self.value = value
        self.source = source

    def reconstruct(self, codegen):
        # We don't support locally created torch function modes yet
        assert self.source
        self.source.reconstruct(codegen)

    def python_type(self):
        return type(self.value)

    def _call_func(self, tx, values):
        unimplemented("torch function mode context manager is not supported yet")

    def call_torch_function(self, tx: "InstructionTranslator", fn, types, args, kwargs):
        return call_torch_function(
            tx,
            self,
            build_torch_function_fn(tx, self.value, self.source),
            fn,
            types,
            args,
            kwargs,
        )


def _get_all_args(args, kwargs):
    return _flatten_vts(pytree.arg_tree_leaves(*args, **kwargs))


def _flatten_vts(vts):
    from collections import deque

    from .dicts import ConstDictVariable
    from .lazy import LazyVariableTracker
    from .lists import ListVariable

    vts = deque(vts)
    output = []

    while vts:
        vt = vts.pop()
        LazyVariableTracker.realize_all(vt)
        if isinstance(vt, ListVariable):
            vts.extend(vt.items)
        elif isinstance(vt, ConstDictVariable):
            vts.extend(vt.items.values())
        else:
            output.append(vt)

    return output


def _get_subclass_type(var):
    assert isinstance(var, (TensorWithTFOverrideVariable, UserDefinedObjectVariable))
    return var.python_type()


def _get_subclass_type_var(tx: "InstructionTranslator", var):
    assert isinstance(var, (TensorWithTFOverrideVariable, UserDefinedObjectVariable))
    if isinstance(var, TensorWithTFOverrideVariable):
        return var.class_type_var(tx)
    elif isinstance(var, UserDefinedObjectVariable):
        from .builder import SourcelessBuilder, VariableBuilder

        if var.source:
            return VariableBuilder(tx, TypeSource(var.source))(var.python_type())
        else:
            return SourcelessBuilder.create(tx, var.python_type())


def _is_attr_overidden(tx: "InstructionTranslator", var, name):
    import torch

    overridden = False
    try:
        attr_val = inspect.getattr_static(var.python_type(), name)
        overridden |= attr_val != getattr(torch.Tensor, name)
    except AttributeError:
        pass

    return overridden


def call_torch_function(
    tx, torch_function_type, torch_function_var, fn, types, args, kwargs
):
    from .builder import SourcelessBuilder

    # signature:
    # def __torch_function__(cls, func, types, args=(), kwargs=None):
    tf_args = (
        torch_function_type,
        fn,
        types,
        SourcelessBuilder.create(tx, tuple(args)),
        SourcelessBuilder.create(tx, kwargs),
    )
    return tx.inline_user_function_return(torch_function_var, tf_args, {})


def build_torch_function_fn(tx: "InstructionTranslator", value, source):
    from .builder import SourcelessBuilder, VariableBuilder

    if source:
        return VariableBuilder(
            tx,
            AttrSource(AttrSource(source, "__torch_function__"), "__func__"),
        )(value.__torch_function__.__func__)
    else:
        return SourcelessBuilder.create(tx, value.__torch_function__.__func__)


def can_dispatch_torch_function(tx: "InstructionTranslator", args, kwargs):
    has_overridden_args = any(
        has_torch_function(arg) for arg in _get_all_args(args, kwargs)
    )
    return (has_overridden_args and tx.torch_function_subclass_enabled) or (
        tx.torch_function_mode_enabled and tx.in_torch_function_mode()
    )


def dispatch_torch_function(tx: "InstructionTranslator", fn, args, kwargs):
    """Gathers all args that are TensorWithTFOverrideVariable and dispatches based on the ordering in _get_overloaded_args"""

    all_args = _get_all_args(args, kwargs)
    overloaded_args = _get_overloaded_args(
        [arg for arg in all_args if has_torch_function(arg)],
        _get_subclass_type,
    )

    types = TupleVariable([_get_subclass_type_var(tx, arg) for arg in overloaded_args])

    if tx.in_torch_function_mode():
        res = tx.call_torch_function_mode(fn, types, args, kwargs)
        if not (isinstance(res, ConstantVariable) and res.value is NotImplemented):
            return res

    for arg in overloaded_args:
        res = arg.call_torch_function(
            tx,
            fn,
            types,
            args,
            kwargs,
        )

        if not (isinstance(res, ConstantVariable) and res.value is NotImplemented):
            return res

    unimplemented(
        f"All __torch_function__ overrides for call {fn} with args {args} and kwargs {kwargs} returned NotImplemented"
    )


class TensorWithTFOverrideVariable(TensorVariable):
    """
    Represents a tensor subclass instance with a __torch_function__ override.
    """

    def __init__(self, *args, **kwargs) -> None:
        self.torch_function_fn = kwargs.pop("torch_function_fn")
        super().__init__(*args, **kwargs)

    @classmethod
    def from_tensor_var(cls, tx, tensor_var, class_type, torch_function_fn):
        import torch

        kwargs = dict(tensor_var.__dict__)
        assert (
            kwargs.pop("class_type") is torch.Tensor
        ), "invalid class type in TensorWithTFOverrideVariable.from_tensor_var"
        var = cls(torch_function_fn=torch_function_fn, class_type=class_type, **kwargs)
        var.install_global(tx)
        return var

    def install_global(self, tx):
        # stash the subclass type to rewrap an output tensor if needed
        # this is needed because the actual type needs to be available
        # each time the compiled artifact is run and outputs a wrapped tensor.
        if self.global_mangled_class_name(tx) not in tx.output.global_scope:
            # Safe because global_mangled_class_name figures it out
            tx.output.install_global_unsafe(
                self.global_mangled_class_name(tx), self.class_type
            )

    def python_type(self):
        return self.class_type

    def class_type_var(self, tx):
        return TensorSubclassVariable(
            self.class_type, source=GlobalSource(self.global_mangled_class_name(tx))
        )

    def global_mangled_class_name(self, tx):
        return get_safe_global_name(
            tx, f"__subclass_{self.class_type.__name__}", self.class_type
        )

    def var_getattr(self, tx: "InstructionTranslator", name):
        # [Note: __torch_function__] We currently only support attributes that are defined on
        # base tensors, custom attribute accesses will graph break.
        import torch

        from .builder import SourcelessBuilder

        if name in banned_attrs:
            unimplemented(
                f"Accessing {name} on a tensor subclass with a __torch_function__ override is not supported"
            )

        if _is_attr_overidden(tx, self, name):
            unimplemented(
                f"Accessing overridden method/attribute {name} on a tensor"
                " subclass with a __torch_function__ override is not supported"
            )

        if tx.output.torch_function_enabled and hasattr(torch.Tensor, name):
            if self.source:
                install_guard(
                    AttrSource(AttrSource(self.source, "__class__"), name).make_guard(
                        GuardBuilder.FUNCTION_MATCH
                    )
                )
            get_fn = SourcelessBuilder.create(tx, getattr(torch.Tensor, name).__get__)

            return self.call_torch_function(
                tx,
                get_fn,
                TupleVariable([self.class_type_var(tx)]),
                [self],
                {},
            )
        else:
            return super().var_getattr(tx, name)

    def call_torch_function(self, tx: "InstructionTranslator", fn, types, args, kwargs):
        return call_torch_function(
            tx,
            self.class_type_var(tx),
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

            from .builder import SourcelessBuilder, VariableBuilder

            if _is_attr_overidden(tx, self, name):
                unimplemented(
                    f"Calling overridden method {name} on a tensor"
                    " subclass with a __torch_function__ override is not supported"
                )

            # [Note: __torch_function__] Currently we only support methods that are defined on tensor
            # we will graph break in other cases this will need a bigger overhaul of extracting methods/comparing them for equality
            # We've established with the above check that the method is not overridden, so we guard that the method is the same
            # as the impl defined on tensor and retrieve it
            if self.source:
                func_var = VariableBuilder(
                    tx, AttrSource(AttrSource(self.source, "__class__"), name)
                )(inspect.getattr_static(self.python_type(), name))
            else:
                func_var = SourcelessBuilder.create(tx, getattr(torch.Tensor, name))
            return dispatch_torch_function(tx, func_var, [self] + args, kwargs)
        else:
            return super().call_method(tx, name, args, kwargs)
