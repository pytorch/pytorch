"""
This module contains variable tracker classes for handling tensors and tensor-related operations in Dynamo.

The main class is TensorVariable which represents torch.Tensor inputs and intermediate values in the FX graph.
It handles tensor operations, method calls, and maintains metadata about tensor properties like dtype, device, etc.

Other key classes include:
- SymNodeVariable: Represents symbolic scalars (int/float/bool) used for size computation and unspecialized values
- NumpyNdarrayVariable: Handles numpy array interop through torch._numpy
- UnspecializedPythonVariable: Represents unspecialized Python numeric values as 1-element tensors
- TensorSubclassVariable: Handles tensor subclasses with __torch_function__ overrides
- UntypedStorageVariable: Represents tensor storage objects
- DataPtrVariable: Handles tensor data pointer operations

These classes work together to track tensor operations and properties during Dynamo's tracing process.
"""

import functools
import logging
import operator
import textwrap
import traceback
import types
from collections.abc import Iterable, Sequence
from contextlib import nullcontext
from itertools import chain
from types import NoneType
from typing import Any, NoReturn, Optional, TYPE_CHECKING

import sympy

import torch._numpy as tnp
import torch.fx
import torch.random
from torch._dynamo import compiled_autograd
from torch._library.opaque_object import is_opaque_reference_type
from torch._subclasses.meta_utils import is_sparse_any
from torch.fx.experimental.symbolic_shapes import (
    guard_scalar,
    GuardOnDataDependentSymNode,
    has_free_symbols,
    is_symbolic,
    SymTypes,
)
from torch.utils._python_dispatch import is_traceable_wrapper_subclass

from .. import config, graph_break_hints, variables
from .._trace_wrapped_higher_order_op import trace_wrapped
from ..exc import (
    TorchRuntimeError,
    unimplemented,
    UnknownPropertiesDuringBackwardTrace,
    UserError,
    UserErrorType,
)
from ..external_utils import _ApplyBackwardHook, call_hook_from_backward_state
from ..guards import GuardBuilder, install_guard
from ..source import AttrSource
from ..utils import (
    fqn,
    get_custom_getattr,
    get_fake_value,
    get_real_value,
    guard_if_dyn,
    object_has_getattribute,
    product,
    proxy_args_kwargs,
    raise_args_mismatch,
    set_example_value,
    tensortype_to_dtype,
)
from .base import AttributeMutationNew, ValueMutationNew, VariableTracker
from .constant import ConstantVariable
from .lists import ListIteratorVariable, SizeVariable
from .script_object import TorchScriptObjectVariable
from .user_defined import UserDefinedClassVariable


try:
    import numpy as np
except ModuleNotFoundError:
    np = None  # type: ignore[assignment]


if TYPE_CHECKING:
    from torch._dynamo.codegen import PyCodegen
    from torch._dynamo.output_graph import OutputGraph
    from torch._dynamo.symbolic_convert import (
        InstructionTranslator,
        InstructionTranslatorBase,
    )

    from .functions import UserFunctionVariable
    from .torch_function import TensorWithTFOverrideVariable


log = logging.getLogger(__name__)

# Ops that allow tensor <op> tensor
supported_tensor_comparison_ops = {
    ">": operator.gt,
    "<": operator.lt,
    ">=": operator.ge,
    "<=": operator.le,
    "==": operator.eq,
    "!=": operator.ne,
    "is": operator.is_,
    "is not": operator.is_not,
}
# Ops that allow tensor <op> None
supported_const_comparison_ops = {
    "is": operator.is_,
    "is not": operator.is_not,
    "==": operator.eq,
    "!=": operator.ne,
}
supported_comparison_ops = {
    **supported_tensor_comparison_ops,
    **supported_const_comparison_ops,
}
supported_tensor_comparison_op_values = dict.fromkeys(
    supported_tensor_comparison_ops.values()
)
supported_const_comparison_op_values = dict.fromkeys(
    supported_const_comparison_ops.values()
)


def is_bound_tensor_method(value: object) -> bool:
    return bool(
        callable(value)
        and not torch._dynamo.utils.object_has_getattribute(value)
        and hasattr(value, "__self__")
        and isinstance(value.__self__, torch.Tensor)
        and getattr(value.__self__, value.__name__, None)
    )


# instead of using inspect.getattr_static, we directly lookup the appropriate
# dicts. It is necessary to keep the torch._C.TensorBase first in the or
# operation, because the second arg takes priority in or operation when there
# are common keys.
all_tensor_attrs = torch._C.TensorBase.__dict__ | torch.Tensor.__dict__


class TensorVariable(VariableTracker):
    """A torch.Tensor input or an intermediate value in the FX graph"""

    _nonvar_fields = {
        "proxy",
        "dtype",
        "device",
        "layout",
        "ndim",
        "size",
        "stride",
        "requires_grad",
        "is_quantized",
        "is_contiguous",
        "is_nested",
        "is_sparse",
        "class_type",
        "specialized_value",
        "_is_name_set",
        *VariableTracker._nonvar_fields,
    }

    def get_real_value(self) -> torch.Tensor:
        """
        Get the actual value represented by this variable if computation is run
        using the user-provided inputs.
        NOTE: this runs actual tensor computation and may be
        slow and memory-intensive.
        """
        return get_real_value(self.proxy.node, self.proxy.tracer)

    def __init__(
        self,
        proxy: torch.fx.Proxy,
        *,
        dtype: torch.dtype,
        device: torch.device,
        layout: torch.layout,
        ndim: int,
        requires_grad: bool,
        is_nested: bool,
        is_quantized: bool,
        is_sparse: bool,
        class_type: type,
        has_grad_fn: bool,
        _size: tuple[Any, ...] | None = None,
        stride: tuple[Any, ...] | None = None,
        is_contiguous: bool | None = None,
        _is_name_set: bool | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.proxy = proxy
        self.dtype = dtype
        # pyrefly: ignore[read-only]
        self.device = device
        self.layout = layout
        self.ndim = ndim
        self._size = _size  # this is accessed as a property for validation
        self.stride = stride
        self.requires_grad = requires_grad
        self.is_quantized = is_quantized
        self.is_contiguous = is_contiguous
        self.is_nested = is_nested
        self.is_sparse = is_sparse
        self.class_type = class_type
        self.has_grad_fn = has_grad_fn
        if _is_name_set is None:
            # no need to rename inputs
            _is_name_set = self.proxy.node.op == "placeholder"
        self._is_name_set: bool = _is_name_set

    def synchronize_attributes(
        self, tx: "InstructionTranslator", target_cls: type | None = None
    ) -> None:
        from .builder import get_specialized_props, infer_subclass_type

        if target_cls is None:
            target_cls = type(self)

        example_value = self.proxy.node.meta.get("example_value")
        specialized_props = get_specialized_props(
            target_cls, tx, example_value, infer_subclass_type(example_value)
        )
        for k, v in specialized_props.items():
            setattr(self, k, v)

    def debug_repr(self) -> str:
        # TODO: strip off fake tensor from repr here
        return repr(self.proxy.node.meta["example_value"])

    def as_proxy(self) -> torch.fx.Proxy:
        return self.proxy

    def python_type(self) -> type:
        return self.class_type

    def is_tensor(self) -> bool:
        return True

    @staticmethod
    def specialize(value: torch.Tensor) -> dict[str, Any]:
        props: dict[str, Any] = {
            "dtype": value.dtype,
            "device": value.device,
            "layout": value.layout,
            "ndim": int(value.ndim),
            "requires_grad": value.requires_grad,
            "is_nested": value.is_nested,
            "is_quantized": value.is_quantized,
            "is_sparse": value.is_sparse,
            "class_type": type(value),
        }
        try:
            props["has_grad_fn"] = value.grad_fn is not None
        except Exception:
            # Workaround for issues with create_parameter_op in Dynamo. Reading
            # grad_fn should never cause an issue.
            props["has_grad_fn"] = False

        if is_sparse_any(value) and not has_free_symbols(value):
            props["_size"] = tuple(
                int(s) if is_symbolic(s) else s for s in value.size()
            )
        elif not has_free_symbols(value):
            # this is a fully static shape, and the keys on props here inform specialization.
            # We have to cast to int here, because these might get accessed as ConstantVariable, which has
            # a strict no-symint policy. If we got here due to not having free symbols, this is a known constant
            # already. We could remove the discrepancy here, by having ConstantVariable be more permissive for
            # constant backed SymInts, but that assert being strict has led to some good signal in hunting bugs, and
            # I'd like to keep it around for now.
            props["_size"] = tuple(
                # the non is_symbolic case applies to the jagged layout
                # NestedTensor case as singleton ints are not symbolic
                int(s) if is_symbolic(s) else s
                for s in value.size()
            )
            props["stride"] = tuple(value.stride())
            if torch._C._functorch.is_batchedtensor(value):
                # Batched tensors does not support contiguity patterns, so
                # we refrain from computing the `is_contiguous` property
                props["is_contiguous"] = None
            else:
                props["is_contiguous"] = tuple(
                    x
                    for x in torch._prims_common._memory_formats
                    if value.is_contiguous(memory_format=x)
                )
        return props

    def dynamic_getattr(
        self, tx: "InstructionTranslator", name: str
    ) -> VariableTracker:
        fake_val = self.proxy.node.meta["example_value"]
        # For getattrs on tensors without sources,
        # we can do better than the default (creating a GetAttrVariable)
        # if:
        # (1) the tensor is a traceable tensor subclass
        # (2) We are getattr'ing an inner tensor from that subclass
        if not self.source and is_traceable_wrapper_subclass(fake_val):
            attrs, _ctx = fake_val.__tensor_flatten__()
            proxy = getattr(self.as_proxy(), name)
            example_value = getattr(fake_val, name)
            if name in attrs:
                # attrs returned from tensor_flatten are always tensors
                assert isinstance(example_value, torch.Tensor)
                from .builder import wrap_fx_proxy

                return wrap_fx_proxy(tx=tx, proxy=proxy, example_value=example_value)
            elif is_opaque_reference_type(type(example_value)):
                fake_script_obj = torch._library.fake_class_registry.maybe_to_fake_obj(
                    tx.output.fake_mode, example_value
                )
                return TorchScriptObjectVariable.create(proxy, fake_script_obj)
            # any other attributes on the subclass (that are not methods)
            # are assumed to be constant metadata.
            elif not callable(example_value):
                return VariableTracker.build(tx, example_value)

        if not (self.source and self.source.subguards_allowed()):
            raise NotImplementedError

        # For local source, we associate the real value. We use this real value
        # for implementing getattr fallthrough on the variable tracker base class.

        # Note - this scope construction is mirrored in guards
        # A subsequent PR will introduce a util.
        scope = {"L": tx.output.local_scope, "G": tx.output.global_scope}
        try:
            # We raise in case we get a typerror bug w/ SuperSource.
            # SuperSource has bugs in it atm, and can produce code like
            # eval("super(L['mod'].model.model.encoder.embed_positions.forward__class__,
            # L['mod'].model.model.encoder.embed_positions)", scope)
            # Which is incorrect, and violates the invariant that all sources should be eval()-able against the scope.
            _input_associated_real_value = eval(self.source.name, scope)
        except Exception as exc:
            raise NotImplementedError from exc

        if _input_associated_real_value is None:
            raise NotImplementedError

        if object_has_getattribute(_input_associated_real_value):
            raise NotImplementedError

        if get_custom_getattr(_input_associated_real_value):
            raise NotImplementedError

        real_value = getattr(_input_associated_real_value, name)

        attr_source = AttrSource(self.source, name)

        # Typically we'd want to use variable builder here
        # but unfortunately id(real_value.__self__) is not id(<original value>)
        if is_bound_tensor_method(real_value):
            # No need to install the guard because its a bound tensor method
            from .misc import GetAttrVariable

            return GetAttrVariable(
                self, name, source=attr_source, py_type=type(real_value)
            )

        install_guard(attr_source.make_guard(GuardBuilder.HASATTR))
        return VariableTracker.build(tx, real_value, attr_source)

    def method_attr_ndim(self, tx: "InstructionTranslator") -> VariableTracker:
        if self.ndim is not None:
            return ConstantVariable.create(self.ndim)
        else:
            return self.call_method(tx, "dim", [], {})

    def method_attr_dtype(self, tx: "InstructionTranslator") -> VariableTracker | None:
        if self.dtype is not None:
            return ConstantVariable.create(self.dtype)
        return None

    def method_attr_device(self, tx: "InstructionTranslator") -> VariableTracker | None:
        if self.device is not None:
            return ConstantVariable.create(self.device)
        return None

    def method_attr_layout(self, tx: "InstructionTranslator") -> VariableTracker | None:
        if self.layout is not None:
            return ConstantVariable.create(self.layout)
        return None

    def method_attr_is_cuda(
        self, tx: "InstructionTranslator"
    ) -> ConstantVariable | None:
        if self.device is not None:
            return ConstantVariable.create(self.device.type == "cuda")
        return None

    def method_attr_shape(self, tx: "InstructionTranslator") -> VariableTracker:
        if self.valid_size():
            sizes: list[VariableTracker] = [
                variables.ConstantVariable.create(x) for x in self.size
            ]
            return SizeVariable(sizes)
        else:
            return self.call_method(tx, "size", [], {})

    def method_attr_requires_grad(
        self, tx: "InstructionTranslator"
    ) -> ConstantVariable | None:
        if self.requires_grad is not None:
            return ConstantVariable.create(self.requires_grad)
        return None

    def method_attr_is_quantized(
        self, tx: "InstructionTranslator"
    ) -> ConstantVariable | None:
        if self.is_quantized is not None:
            return ConstantVariable.create(self.is_quantized)
        return None

    def method_attr_is_sparse(
        self, tx: "InstructionTranslator"
    ) -> ConstantVariable | None:
        if self.is_sparse is not None:
            return ConstantVariable.create(self.is_sparse)
        return None

    def method_attr_is_nested(
        self, tx: "InstructionTranslator"
    ) -> ConstantVariable | None:
        if self.is_nested is not None:
            return ConstantVariable.create(self.is_nested)
        return None

    def method_attr_retain_grad(self, tx: "InstructionTranslator") -> NoReturn:
        unimplemented(
            gb_type="Tensor.retain_grad() with AOTDispatcher",
            context=f"var_getattr {self} retain_grad",
            explanation="`Tensor.retain_grad()` does not work with AOTDispatcher.",
            hints=[],
        )

    def method_attr_data(self, tx: "InstructionTranslator") -> VariableTracker:
        return variables.TorchInGraphFunctionVariable(
            torch._C._autograd._get_data_attr  # type: ignore[attr-defined]
        ).call_function(tx, [self], {})

    def method_attr_grad_fn(
        self, tx: "InstructionTranslator"
    ) -> ConstantVariable | None:
        if self.has_grad_fn:
            unimplemented(
                gb_type="Tensor with grad_fn()",
                context=f"var_getattr {self} grad_fn",
                explanation="Dynamo does not support tracing tensors with a grad_fn directly.",
                hints=[],
            )
        else:
            return variables.ConstantVariable(None)

    def method_attr__version(self, tx: "InstructionTranslator") -> VariableTracker:
        from ..tensor_version_op import _tensor_version

        return variables.TorchInGraphFunctionVariable(_tensor_version).call_function(
            tx, [self], {}
        )

    def call_obj_hasattr(
        self, tx: "InstructionTranslator", name: str
    ) -> ConstantVariable:
        from . import GetAttrVariable
        from .builtin import BuiltinVariable

        # TODO - This is not a good solution but solves an accuracy issue.
        # Today, var_getattr returns GetAttrVariable for both non-existent
        # attributes and existing attributes. This is a bug and requires more
        # deep dive.
        if name in all_tensor_attrs:
            return ConstantVariable(True)

        try:
            var = BuiltinVariable(getattr).call_function(
                tx, [self, ConstantVariable(name)], {}
            )
            # in the event that TensorVariable returns NotImplemented
            # BuiltinVariable.call_getattr returns GetAttrVariable
            ret_val = not isinstance(var, GetAttrVariable)
        except AttributeError:
            ret_val = False

        if self.source:
            install_guard(
                AttrSource(self.source, name).make_guard(GuardBuilder.HASATTR)
            )

        return ConstantVariable(ret_val)

    def var_getattr(self, tx: "InstructionTranslator", name: str) -> VariableTracker:
        if self.is_strict_mode(tx):
            if name in self._strict_mode_banned_ops():
                unimplemented(
                    gb_type="Strict mode banned op",
                    context=f"var_getattr {self} {name}",
                    explanation=f"Getattr invocation '{name}' in strict mode is not supported.",
                    hints=[
                        f"Remove `{name}` from the list of banned ops by "
                        "setting `torch._dynamo.config._autograd_backward_strict_mode_banned_ops`.",
                    ],
                )
            elif name in self._strict_mode_conditional_banned_ops():
                raise UnknownPropertiesDuringBackwardTrace(
                    f"Unknown property {name} during speculating backward, dynamo will insert contiguous call ahead and speculate it again"  # noqa: B950
                )

        if name == "__class__":
            return UserDefinedClassVariable(self.python_type())

        handler = getattr(self, f"method_attr_{name}", None)
        result = handler(tx) if handler is not None else None

        # Add a guard for type matching, these guards are checked before tensor guards
        # In some cases, a <tensor>.<attr> guard can be evaluated first, and break if
        # <tensor> is later changed to another type
        if (
            result is not None
            and self.source
            and self.source.subguards_allowed()
            and not (
                name not in ("grad", "requires_grad") and result.is_python_constant()
            )
        ):
            install_guard(self.make_guard(GuardBuilder.TYPE_MATCH))
            result.source = AttrSource(self.source, name)

        # It's hard to get inplace view (metadata mutation) on graph input work properly across
        # dynamo/aot/inductor, just fall back.
        if self.source is not None and hasattr(torch.ops.aten, name):
            fn = getattr(torch.ops.aten, name)
            if (
                hasattr(fn, "overloads")
                and hasattr(fn, fn.overloads()[0])
                and torch.Tag.inplace_view in getattr(fn, fn.overloads()[0]).tags
            ):
                # Delay the graph break to the actual call of unsqueeze_/resize_/resize_as_ etc.
                return variables.misc.DelayGraphBreakVariable(
                    source=AttrSource(self.source, name),
                    msg="Getting an inplace view on a graph input is not supported",
                )

        # For attributes (not methods) that were not caught in the special handling above,
        # (e.g. tensor.real), we handle these generically, assuming that the output type is
        # a tensor.
        if result is None and name != "grad":

            def try_generic_attr_handling() -> VariableTracker | None:
                from .builder import wrap_fx_proxy
                from .misc import GetAttrVariable

                static_attr = all_tensor_attrs.get(name, None)
                if static_attr is None:
                    return None

                # Make sure this is an attribute, not a method.
                # type(torch.Tensor.H) should be "getset_descriptor"
                # This is a because of CPython implementation, see THPVariableType:
                # these attributes are implemented under tp_getset, which appear
                # as `getset_descriptor`s, (compared to, say, methods which appear
                # as `method_descriptor`s)
                if type(static_attr) is not types.GetSetDescriptorType:
                    return None

                proxy = GetAttrVariable.create_getattr_proxy(self.as_proxy(), name)
                if self.source is not None:
                    return wrap_fx_proxy(
                        tx=tx, proxy=proxy, source=AttrSource(self.source, name)
                    )
                else:
                    return wrap_fx_proxy(tx=tx, proxy=proxy)

            result = try_generic_attr_handling()

        if result is None:
            result = self.dynamic_getattr(tx, name)

        if result is None:
            raise NotImplementedError
        return result

    def call_id(self, tx: "InstructionTranslator") -> VariableTracker:
        if not self.source:
            unimplemented(
                gb_type="Unsupported call_id() without source",
                context=f"call_id {self}",
                explanation="call_id() not supported for sourceless TensorVariable.",
                hints=[],
            )

        assert self.source
        # For local source, we associate the real value. We use this real value
        scope = {"L": tx.output.local_scope, "G": tx.output.global_scope}
        _input_associated_real_value = None
        try:
            _input_associated_real_value = eval(self.source.name, scope)
        except Exception as exc:
            unimplemented(
                gb_type="Error getting associated real value",
                context=f"call_id {self}",
                explanation="Dynamo encountered an error while trying to "
                "get the associated real value.",
                hints=[],
                from_exc=exc,
            )

        if _input_associated_real_value is None:
            unimplemented(
                gb_type="call_id() without associated real value",
                context=f"call_id {self}",
                explanation="Dynamo could not find an associated real value for the tensor.",
                hints=[],
            )

        install_guard(self.source.make_guard(GuardBuilder.ID_MATCH))
        id_value = id(_input_associated_real_value)
        return ConstantVariable.create(id_value)

    def has_unpack_var_sequence(self, tx: "InstructionTranslator") -> bool:
        return self.ndim > 0

    def unpack_var_sequence(
        self, tx: "InstructionTranslator", idxes: Sequence[int] | None = None
    ) -> list[VariableTracker]:
        from .builder import wrap_fx_proxy_cls
        from .torch_function import TensorWithTFOverrideVariable

        if self.valid_size():
            size_len = len(self.size)
        else:
            size_var = self.call_method(tx, "size", [], {})
            assert isinstance(size_var, SizeVariable)
            size_len = len(size_var.items)
        # Ensure we don't unpack a scalar tensor.
        assert size_len != 0, "Can't unpack scalar tensors."

        if self.valid_size():
            length = self.size[0]
        else:
            dyn_length = self.call_method(tx, "size", [ConstantVariable.create(0)], {})
            # SymNodeVariable for symbolic sizes, ConstantVariable for constants OR values produced through
            # symbolic_shapes, but that end up as int/sympy.Integer
            assert (
                isinstance(dyn_length, SymNodeVariable)
                or dyn_length.is_python_constant()
            )
            if isinstance(dyn_length, SymNodeVariable):
                length = dyn_length.evaluate_expr(tx.output)
            else:
                length = dyn_length.as_python_constant()

        if idxes is None:
            idxes = range(length)  # type: ignore[arg-type]
        else:
            assert len(idxes) == length, (
                f"Can't unpack a tensor of {length} rows into a tuple of {len(idxes)} elements."
            )

        # preserve tensor subclass type when unpacking
        if isinstance(self, TensorWithTFOverrideVariable):
            base_vars = [
                wrap_fx_proxy_cls(
                    target_cls=TensorVariable, tx=tx, proxy=self.as_proxy()[i]
                )
                for i in idxes
            ]
            return [
                TensorWithTFOverrideVariable.from_tensor_var(
                    tx, v, self.class_type, self.source
                )
                for v in base_vars
            ]

        return [
            wrap_fx_proxy_cls(target_cls=type(self), tx=tx, proxy=self.as_proxy()[i])
            for i in idxes
        ]

    def call_tree_map(
        self,
        tx: "InstructionTranslator",
        tree_map_fn: "UserFunctionVariable",
        map_fn: VariableTracker,
        rest: Sequence[VariableTracker],
        tree_map_kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        return map_fn.call_function(tx, [self, *rest], {})

    def valid_size(self) -> bool:
        return self._size is not None

    @property
    def size(self) -> tuple[Any, ...]:
        assert self._size is not None, "accessing None size in TensorVariable"
        return self._size

    def _strict_mode_banned_ops(self) -> list[str]:
        return torch._dynamo.config._autograd_backward_strict_mode_banned_ops

    def _strict_mode_conditional_banned_ops(self) -> list[str]:
        return (
            torch._dynamo.config._autograd_backward_strict_mode_conditional_banned_ops
        )

    def call_method(
        self,
        tx: "InstructionTranslator",
        name: str,
        args: Sequence[VariableTracker],
        kwargs: "dict[str, VariableTracker]",
    ) -> VariableTracker:
        from .builder import SourcelessBuilder, VariableBuilder
        from .torch_function import can_dispatch_torch_function, dispatch_torch_function

        if self.is_strict_mode(tx) and name in self._strict_mode_banned_ops():
            unimplemented(
                gb_type="Illegal method invocation in strict mode",
                context=f"call_method {self} {name} {args} {kwargs}",
                explanation="Dynamo currently does not support this method "
                f"({name}) invocation in strict mode.",
                hints=[],
            )

        # Only override builtin tensor methods
        # The user can manually add override handling
        # with a decorator for other methods (e.g. a dispatch subclass with other methods)
        static_attr = all_tensor_attrs.get(name, None)
        is_base_tensor_method = static_attr is not None

        if (
            can_dispatch_torch_function(tx, tuple([self] + list(args)), kwargs)
            and is_base_tensor_method
        ):
            if self.source:
                func_var = VariableBuilder(
                    tx, AttrSource(AttrSource(self.source, "__class__"), name)
                )(static_attr)
            else:
                func_var = SourcelessBuilder.create(tx, getattr(torch.Tensor, name))

            return dispatch_torch_function(
                tx, func_var, tuple([self] + list(args)), kwargs
            )

        """
        Dispatch to a method-specific handler defined below.  If the
        handler returns None (or doesn't exist) we put the method call
        in the graph.
        """

        # This is seen in inspect signature where we check if the value is a default value
        if name == "__eq__" and isinstance(args[0], UserDefinedClassVariable):
            return variables.ConstantVariable(False)

        # For historical reasons, these ops decompose down to syntactically
        # invalid aten ops because they contain the python keyword `from`, see
        # discussions in #151432 for more details.
        # We graph break for now since this use case is uncommon.
        if name == "random_":
            unimplemented(
                gb_type="Tensor.random_ op",
                context=f"Tensor.{name}({args=}, {kwargs=})",
                explanation="This is currently not supported.",
                hints=[
                    "Use the out-of-place version of this op",
                    *graph_break_hints.SUPPORTABLE,
                ],
            )
        elif name == "uniform_" and "from" in kwargs:
            unimplemented(
                gb_type="Tensor.uniform_ op called with `from` keyword",
                context=f"Tensor.{name}({args=}, {kwargs=})",
                explanation="This is currently not supported.",
                hints=[
                    "Avoid using the `from` keyword.",
                    *graph_break_hints.SUPPORTABLE,
                ],
            )

        try:
            handler_method = getattr(self, f"method_{name}")
        except AttributeError:
            pass
        else:
            try:
                # Realize any LazyVariableTracker in kwargs before calling handler.
                realized_kwargs = {k: v.realize() for k, v in kwargs.items()}
                result = handler_method(tx, *args, **realized_kwargs)
                if result:
                    return result
            except TypeError as e:
                unimplemented(
                    gb_type="Unhandled args for method",
                    context=f"call_method {self} {name} {args} {kwargs}",
                    explanation="Dynamo encountered an error while calling "
                    f"the method `{name}`.",
                    hints=[],
                    from_exc=e,
                )

        from .builder import wrap_fx_proxy

        proxy = tx.output.create_proxy(
            "call_method",
            name,
            *proxy_args_kwargs([self, *args], kwargs),
        )

        # [Note: Inplace ops and VariableTracker metadata]
        # For inplace operations, we need to propagate tensor metadata from the
        # arguments to self. For example:
        #   x.add_(y) where y.requires_grad=True => x.requires_grad becomes True
        # We detect inplace ops by checking if self's fake tensor version changes
        # after wrap_fx_proxy (which runs get_fake_value internally).
        # We only synchronize when there's a tensor argument, since that's when
        # metadata propagation is relevant.
        self_fake = self.proxy.node.meta.get("example_value")
        version_before = self_fake._version if self_fake is not None else None

        result = wrap_fx_proxy(tx, proxy)

        # Check if self was mutated (version increased = inplace op)
        version_after = self_fake._version if self_fake is not None else None
        if (
            version_before is not None
            and version_after is not None
            and version_after > version_before
            and any(arg.is_tensor() for arg in args)
        ):
            self.synchronize_attributes(tx)

        return result

    def method_size(
        self, tx: "InstructionTranslator", *args: Any, **kwargs: Any
    ) -> VariableTracker | None:
        return self._method_size_stride("size", *args, **kwargs)

    def method_stride(
        self, tx: "InstructionTranslator", *args: Any, **kwargs: Any
    ) -> VariableTracker | None:
        return self._method_size_stride("stride", *args, **kwargs)

    def _method_size_stride(
        self, name: str, dim: Any | None = None
    ) -> VariableTracker | None:
        dim = guard_if_dyn(dim)

        def make_const_size_variable(x: Sequence[Any], **options: Any) -> SizeVariable:
            return SizeVariable(
                [ConstantVariable.create(y, **options) for y in x], **options
            )

        RetVariable = (
            make_const_size_variable if name == "size" else ConstantVariable.create
        )

        # Technically, this should not be necessary, but I'm including it
        # for enhanced BC, in case example_value is sometimes not set
        # (it really should always be set though!)
        if name != "size":
            r = getattr(self, name)
        elif name == "size" and self.valid_size():
            r = self.size
        else:
            r = None

        if r is not None:
            if dim is None:
                return RetVariable(r)
            else:
                return ConstantVariable.create(r[dim])

        # It might still be constant!  Consult the fake tensor and see
        if (fake := self.proxy.node.meta.get("example_value")) is not None:
            if dim is None:
                fake_r = getattr(fake, name)()
                if not has_free_symbols(fake_r):
                    # int conversion for safety, in case a SymInt refined
                    # to constant
                    return RetVariable(tuple(int(r) for r in fake_r))
            else:
                fake_r = getattr(fake, name)(dim)
                if not has_free_symbols(fake_r):
                    return ConstantVariable.create(int(fake_r))
        return None

    def method_numel(self, tx: "InstructionTranslator") -> VariableTracker | None:
        if self.valid_size():
            return ConstantVariable.create(product(self.size))

        # It might still be constant!  Consult the fake tensor and see
        if (fake := self.proxy.node.meta.get("example_value")) is not None:
            fake_r = fake.numel()
            if not has_free_symbols(fake_r):
                return ConstantVariable.create(int(fake_r))
        return None

    method_nelement = method_numel

    def method_dim(self, tx: "InstructionTranslator") -> VariableTracker | None:
        if self.ndim is not None:
            return ConstantVariable.create(self.ndim)
        return None

    method_ndimension = method_dim

    def method_is_floating_point(
        self, tx: "InstructionTranslator"
    ) -> ConstantVariable | None:
        if self.dtype is not None:
            return ConstantVariable.create(self.dtype.is_floating_point)
        return None

    def method_is_inference(
        self, tx: "InstructionTranslator"
    ) -> ConstantVariable | None:
        if config.fake_tensor_disable_inference_mode:
            unimplemented(
                gb_type="Encountered tensor.is_inference() during tracing",
                context="",
                explanation="tensor.is_inference() is not supported",
                hints=[
                    *graph_break_hints.FUNDAMENTAL,
                    *graph_break_hints.INFERENCE_MODE,
                ],
            )
        if (fake := self.proxy.node.meta.get("example_value")) is not None:
            return ConstantVariable.create(fake.is_inference())
        return None

    def method_is_complex(self, tx: "InstructionTranslator") -> ConstantVariable | None:
        if self.dtype is not None:
            return ConstantVariable.create(self.dtype.is_complex)
        return None

    def method_is_contiguous(
        self, tx: "InstructionTranslator", memory_format: VariableTracker | None = None
    ) -> ConstantVariable | None:
        memory_format_const = (
            memory_format.as_python_constant()
            if memory_format is not None
            else torch.contiguous_format
        )
        if self.is_contiguous is not None:
            # pyrefly: ignore[not-iterable]
            return ConstantVariable.create(memory_format_const in self.is_contiguous)
        elif (fake := self.proxy.node.meta.get("example_value")) is not None:
            return ConstantVariable.create(
                fake.is_contiguous(memory_format=memory_format_const)
            )
        return None

    def method_type(
        self,
        tx: "InstructionTranslator",
        dtype: Any | None = None,
        non_blocking: bool = False,
        **kwargs: Any,
    ) -> VariableTracker | None:
        if (
            dtype is None
            and self.dtype is not None
            and isinstance(self.device, torch.device)
        ):
            tensortype = next(
                k for k, v in tensortype_to_dtype.items() if self.dtype in v
            )
            if self.device.type == "cpu":
                return ConstantVariable.create(f"torch.{tensortype.__name__}")
            else:
                return ConstantVariable.create(
                    f"torch.{self.device.type}.{tensortype.__name__}"
                )
        elif (
            dtype is not None
            and fqn(type(dtype.as_python_constant())) == "torch.tensortype"
        ):
            # torch.FloatTensor, etc. are all of type "torch.tensortype".
            # torch.fx's tracer fails on these types, because it doesn't support arguments of torch.tensortype type.
            # So, we pass it in as a string (which is also supported, see above implementation for .type() with 0 args)
            tensor_type = dtype.as_python_constant()
            tensor_type_const = ConstantVariable.create(fqn(tensor_type))

            from .builder import wrap_fx_proxy

            if non_blocking:
                kwargs = {"non_blocking": non_blocking, **kwargs}

            return wrap_fx_proxy(
                tx,
                tx.output.create_proxy(
                    "call_method",
                    "type",
                    *proxy_args_kwargs([self, tensor_type_const], kwargs),
                ),
            )
        return None

    def method_as_subclass(
        self, tx: "InstructionTranslator", cls: VariableTracker
    ) -> "TensorWithTFOverrideVariable":
        if isinstance(cls, TensorSubclassVariable) and cls.source:
            from .torch_function import TensorWithTFOverrideVariable

            py_cls = cls.as_python_constant()
            var = TensorWithTFOverrideVariable.from_tensor_var(
                tx, self, py_cls, cls.source
            )
            # See NOTE [Side effect tracking for newly constructed tensor]
            tx.output.side_effects._track_obj(
                object(), var, mutation_type_cls=AttributeMutationNew
            )
            return var
        unimplemented(
            gb_type="Argument of `as_subclass` must be a non-dispatcher-style tensor subclass",
            context=f"{self}.as_subclass({cls})",
            explanation="Currently not supported",
            hints=[
                "Avoid this call or move it outside `torch.compile` regione",
                *graph_break_hints.SUPPORTABLE,
            ],
        )

    def method_get_device(self, tx: "InstructionTranslator") -> VariableTracker | None:
        if isinstance(self.device, torch.device):
            index = self.device.index if self.device.type != "cpu" else -1
            return ConstantVariable.create(index)
        return None

    def method_element_size(self, tx: "InstructionTranslator") -> VariableTracker:
        return ConstantVariable.create(self.dtype.itemsize)

    def method_numpy(
        self, tx: "InstructionTranslator", *, force: VariableTracker | bool = False
    ) -> "NumpyNdarrayVariable":
        if not config.trace_numpy:
            unimplemented(
                gb_type="Tensor.numpy() with trace_numpy=False",
                context=f"call_method {self} numpy",
                explanation="`Tensor.numpy()` was called, but the `trace_numpy` "
                "configuration was manually disabled.",
                hints=[
                    "Set `torch._dynamo.config.trace_numpy = True` to allow "
                    "Dynamo to trace through NumPy.",
                ],
            )
        if not np:
            unimplemented(
                gb_type="Tensor.numpy() without NumPy installed",
                context=f"call_method {self} numpy",
                explanation="`Tensor.numpy()` was called, but the NumPy library "
                "is not available in the current environment.",
                hints=[
                    "Ensure NumPy is installed in your Python environment.",
                ],
            )
        if self.layout != torch.strided:
            raise TypeError(
                f"can't convert {self.layout} layout tensor to numpy. Use Tensor.to_dense() first"
            )

        # We don't check that the tensor is on CPU when force is False, as this
        # allows us to execute NumPy code on CUDA. Same for requires_grad=True
        if force and force.as_python_constant():  # type: ignore[attr-defined]
            # If the user set force=True we try to preserve the semantics (no gradients, move to CPU...)
            t = self.call_method(tx, "detach", [], {})
            proxy = tx.output.create_proxy("call_method", "cpu", (t.as_proxy(),), {})
        else:
            # Hacky way to create a view of self that will be marked as NumpyNdarrayVariable
            proxy = tx.output.create_proxy(
                "call_method", "view_as", *proxy_args_kwargs([self, self], {})
            )
        return NumpyNdarrayVariable.create(tx, proxy)

    def method_tolist(self, tx: "InstructionTranslator") -> VariableTracker:
        from .builder import wrap_fx_proxy

        def tolist(tensor: torch.Tensor, sub_proxy: torch.fx.Proxy) -> Any | list[Any]:
            def wrap(i: Any, sub_proxy: torch.fx.Proxy) -> VariableTracker:
                return wrap_fx_proxy(
                    tx,
                    sub_proxy.item(),
                )

            if tensor.dtype not in [
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
            ]:
                unimplemented(
                    gb_type="Tensor.tolist() with non-integer tensor",
                    context=f"call_method {self} to_list",
                    explanation="Dynamo currently does not support tracing "
                    "`tolist()` on non-integer tensors.",
                    hints=[
                        "Ensure the input tensor to `tolist()` is an integer "
                        "type (e.g., int8, int16, int32, int64)."
                    ],
                )

            if tensor.dim() == 0:
                return wrap(tensor, sub_proxy)

            if tensor.dim() == 1:
                return [wrap(val, sub_proxy[i]) for i, val in enumerate(tensor)]

            return [
                tolist(sub_tensor, sub_proxy=sub_proxy[i])
                for i, sub_tensor in enumerate(tensor)
            ]

        tensor = self.as_proxy().node.meta["example_value"]
        out = tolist(tensor, self.as_proxy())
        return VariableTracker.build(tx, out)

    def _collect_backward_inputs(
        self, vars_iter: Iterable[VariableTracker], error_on_non_leaf: bool = False
    ) -> Optional[list[VariableTracker]]:
        """
        Collect unique leaf tensors from vars_iter for backward.

        Only collects leaf tensors (no grad_fn). Non-leaf tensors are skipped
        (or error if error_on_non_leaf=True) because when auto-detecting inputs,
        we must not stop gradients at non-leafs - they are intermediates, and the
        real leaf tensors (parameters) are further up the autograd graph.

        Deduplicates by proxy.node.
        Returns list of unique leaf tensor variables.
        """
        from ..source import SyntheticLocalSource

        result = []
        seen_nodes: set[torch.fx.Node] = set()
        for var in vars_iter:
            if isinstance(var, TensorVariable) and var.requires_grad:
                # Non-leaf tensors (has_grad_fn=True) must be skipped because:
                # 1. Semantically: they're intermediates, not the leaves we want gradients for
                # 2. Implementation: accumulate_grad polyfill can't handle .grad on non-leafs
                #    (Dynamo creates GetAttrVariable instead of TensorVariable)
                #
                # In-graph created tensors without proper source also can't be handled
                # because subguards_allowed() returns False for SyntheticLocalSource.
                if var.has_grad_fn:
                    if error_on_non_leaf:
                        unimplemented(
                            gb_type="backward() with non-leaf tensor",
                            context=f"backward(inputs=[...]) with non-leaf tensor: {var}",
                            explanation="backward(inputs=[...]) with non-leaf tensors is not yet supported.",
                            hints=[
                                "Only pass leaf tensors (parameters, graph inputs) to backward(inputs=...)",
                            ],
                        )
                elif not var.source or isinstance(var.source, SyntheticLocalSource):
                    if error_on_non_leaf:
                        unimplemented(
                            gb_type="backward() with in-graph created tensor",
                            context=f"backward(inputs=[...]) with in-graph created tensor: {var}",
                            explanation="backward(inputs=[...]) with tensors created inside the "
                            "compiled function is not yet supported.",
                            hints=[
                                "Only pass tensors that are inputs to the compiled function or captured from outside",
                            ],
                        )
                else:
                    node = var.proxy.node
                    if node not in seen_nodes:
                        seen_nodes.add(node)
                        result.append(var)
        return result

    def method_backward(
        self,
        tx: "InstructionTranslator",
        gradient: Optional[VariableTracker] = None,
        retain_graph: Optional[VariableTracker] = None,
        create_graph: Optional[VariableTracker] = None,
        inputs: Optional[VariableTracker] = None,
    ) -> Optional[VariableTracker]:
        """
        Trace tensor.backward() by rewriting as autograd.grad() + accumulate_grad.

        Implementation:
        1. Collect leaf tensors to compute gradients for
        2. Call autograd.grad(loss, inputs) to compute gradients
        3. For each leaf tensor, call accumulate_grad to update .grad

        Non-leaf tensor handling:
        - Auto-detect (inputs=None): Non-leaf tensors are silently skipped.
          This matches eager where only leaves get .grad.
        - User-provided (inputs=[...]): Errors if any non-leaf tensor is found.
          While eager backward(inputs=[non_leaf]) works, Dynamo cannot trace it
          because the accumulate_grad polyfill accesses .grad, and Dynamo creates
          a generic GetAttrVariable for .grad on non-leaf tensors (instead of a
          TensorVariable), which cannot be used in tensor operations.

        TODO: Support non-leaf tensors by fixing .grad access on non-leaf in Dynamo.
        """
        if not config.trace_autograd_ops:
            unimplemented(
                gb_type="Unsupported Tensor.backward() call",
                context=f"call_method {self} backward {gradient} {retain_graph} {create_graph} {inputs}",
                explanation="Dynamo currently does not support tracing `Tensor.backward()` when trace_autograd_ops is off.",
                hints=["Set torch._dynamo.trace_autograd_ops=True"],
            )

        if not self.requires_grad and not self.has_grad_fn:
            raise TorchRuntimeError(
                "tensor does not require grad and does not have a grad_fn"
            )

        # Step 1: Collect leaf tensors to compute gradients for
        #
        # Note: We rely on the autograd.grad handler to validate that the generated
        # autograd.grad call is legal (i.e., doesn't traverse external grad_fns).
        # If the loss depends on leaves we don't know about, the autograd.grad
        # handler will catch it via the external_grad_fns check.
        auto_detect = inputs is None
        if auto_detect:
            # Sources can be either user inputs (params are included here)
            # or parameters that are created in forward.
            all_vars = chain(
                tx.output.leaf_var_creation_order,
                tx.output.input_source_to_var.values(),
            )
            input_vars = self._collect_backward_inputs(all_vars)
            if not input_vars:
                # No leaf tensors found - nothing to accumulate gradients into.
                # This matches eager behavior where backward() is a no-op if there
                # are no leaves requiring grad.
                return ConstantVariable.create(None)
        else:
            provided_vars = (
                inputs.items
                if isinstance(inputs, variables.BaseListVariable)
                else [inputs]
            )
            input_vars = self._collect_backward_inputs(
                provided_vars, error_on_non_leaf=True
            )
            if not input_vars:
                # User explicitly provided inputs but none were valid leaf tensors.
                # This would cause "grad requires non-empty inputs" error at runtime.
                unimplemented(
                    gb_type="backward() with empty inputs",
                    context="backward(inputs=[...]) resulted in no valid leaf tensors",
                    explanation="backward(inputs=[...]) requires at least one valid leaf tensor.",
                    hints=[
                        "Ensure at least one tensor in inputs is a leaf (requires_grad=True, no grad_fn)",
                    ],
                )

        # Build autograd.grad call
        grad_kwargs = {"allow_unused": VariableTracker.build(tx, auto_detect)}
        if retain_graph is not None:
            grad_kwargs["retain_graph"] = retain_graph
        if create_graph is not None:
            grad_kwargs["create_graph"] = create_graph

        inputs_var = VariableTracker.build(tx, input_vars)
        grad_args = [self, inputs_var]
        if gradient is not None:
            grad_args.append(gradient)

        autograd_grad_fn = VariableTracker.build(tx, torch.autograd.grad)
        grads_var = autograd_grad_fn.call_function(tx, grad_args, grad_kwargs)

        # Accumulate gradients for unique leaf tensors under no_grad context
        # to replicate eager autograd engine.
        from .ctx_manager import GradModeVariable

        grad_mode_var = GradModeVariable.create(tx, False, initialized=True)
        grad_mode_var.enter(tx)

        accumulate_grad_fn = VariableTracker.build(
            tx, torch.ops.inductor.accumulate_grad_.default
        )
        assert input_vars is not None
        for idx, input_var in enumerate(input_vars):
            grad_i = grads_var.call_method(
                tx, "__getitem__", [VariableTracker.build(tx, idx)], {}
            )
            accumulate_grad_fn.call_function(tx, [input_var, grad_i], {})

        grad_mode_var.exit(tx)

        return VariableTracker.build(tx, None)

    def method_data_ptr(
        self,
        tx: "InstructionTranslator",
        *args: VariableTracker,
        **kwargs: VariableTracker,
    ) -> "DataPtrVariable":
        return DataPtrVariable(self)

    def method_item(
        self,
        tx: "InstructionTranslator",
        *args: VariableTracker,
        **kwargs: VariableTracker,
    ) -> None:
        # We enable capture_scalar_outputs when full_graph=True by default.
        if not tx.one_graph and not config.capture_scalar_outputs:
            self._warn_capture_scalar_outputs()
            unimplemented(
                gb_type="Unsupported Tensor.item() call with capture_scalar_outputs=False",
                context=f"call_method {self} item {args} {kwargs}",
                explanation="Dynamo does not support tracing `Tensor.item()` "
                "with config.capture_scalar_outputs=False.",
                hints=[
                    "Set `torch._dynamo.config.capture_scalar_outputs = True` "
                    "or `export TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1` "
                    "to include these operations in the captured graph.",
                ],
            )
        return None

    def method___getitem__(
        self,
        tx: "InstructionTranslator",
        *args: VariableTracker,
        **kwargs: VariableTracker,
    ) -> VariableTracker:
        from .builder import wrap_fx_proxy

        if isinstance(args[0], SymNodeVariable):
            # Standard indexing will force specialization due to
            # __index__.  Rewrite as a regular torch op which will
            # trace fine
            fn, args = (  # type: ignore[assignment]
                torch.select,
                [
                    variables.ConstantVariable.create(0),
                    args[0],
                ],
            )
        else:
            fn = operator.getitem

        proxy = tx.output.create_proxy(
            "call_function",
            fn,
            *proxy_args_kwargs([self] + list(args), kwargs),
        )

        return wrap_fx_proxy(tx, proxy)

    @staticmethod
    @functools.cache
    def _warn_capture_scalar_outputs() -> None:
        user_stack = torch._guards.TracingContext.extract_stack()
        user_stack_formatted = "".join(traceback.format_list(user_stack))
        log.warning(
            textwrap.dedent(
                """\
                    Graph break from `Tensor.item()`, consider setting:
                        torch._dynamo.config.capture_scalar_outputs = True
                    or:
                        env TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1
                    to include these operations in the captured graph.

                    Graph break: from user code at:
                    %s
                """
            ),
            user_stack_formatted,
        )

    def method___len__(self, tx: "InstructionTranslator") -> VariableTracker:
        return self.call_method(tx, "size", [ConstantVariable.create(0)], {})

    def method___iter__(self, tx: "InstructionTranslator") -> ListIteratorVariable:
        return ListIteratorVariable(
            self.unpack_var_sequence(tx), mutation_type=ValueMutationNew()
        )

    def method_addcmul_(
        self,
        tx: "InstructionTranslator",
        tensor1: Any,
        tensor2: Any,
        *,
        value: Any | None = None,
    ) -> Any | None:
        if value is not None and config.enable_dynamo_decompositions:
            from .. import polyfills

            return tx.inline_user_function_return(
                VariableTracker.build(tx, polyfills.addcmul_inplace),
                [self, tensor1, tensor2, value],
                {},
            )
        return None

    def method___setitem__(
        self,
        tx: "InstructionTranslator",
        key: VariableTracker,
        value: VariableTracker,
    ) -> VariableTracker:
        proxy = tx.output.create_proxy(
            "call_function",
            operator.setitem,
            *proxy_args_kwargs([self, key, value], {}),
        )

        # See Note [Inplace ops and VariableTracker metadata]
        # __setitem__ is always an inplace operation. We need to run fake execution
        # and propagate metadata if self was mutated.
        # The context managers handle saved tensor hooks and unbacked symbols.
        self_fake = self.proxy.node.meta.get("example_value")
        version_before = self_fake._version if self_fake is not None else None

        with (
            torch._dynamo.utils._disable_saved_tensors_hooks_during_tracing(),
            tx.fake_mode.shape_env.ignore_fresh_unbacked_symbols()
            if tx.fake_mode and tx.fake_mode.shape_env
            else nullcontext(),
        ):
            get_fake_value(proxy.node, tx, allow_non_graph_fake=False)

        # Check if self was mutated (version increased)
        version_after = self_fake._version if self_fake is not None else None
        if (
            version_before is not None
            and version_after is not None
            and version_after > version_before
            and value.is_tensor()
        ):
            self.synchronize_attributes(tx)

        if config.use_graph_deduplication or config.track_nodes_for_deduplication:
            tx.output.region_tracker.add_node_mutation(proxy.node, 0)

        return ConstantVariable.create(None)

    def method_resize_(
        self,
        tx: "InstructionTranslator",
        *args: VariableTracker,
        **kwargs: VariableTracker,
    ) -> NoReturn:
        unimplemented(
            gb_type="Unsupported Tensor.resize_() call",
            context=f"call_method {self} resize_ {args} {kwargs}",
            explanation="Dynamo currently does not support tracing `Tensor.resize_()`.",
            hints=[],
        )

    def method_resize_as_(
        self,
        tx: "InstructionTranslator",
        *args: VariableTracker,
        **kwargs: VariableTracker,
    ) -> NoReturn:
        unimplemented(
            gb_type="Unsupported Tensor.resize_as_() call",
            context=f"call_method {self} resize_as_ {args} {kwargs}",
            explanation="Dynamo currently does not support tracing `Tensor.resize_as_()`.",
            hints=[],
        )

    def method_sparse_resize_(
        self,
        tx: "InstructionTranslator",
        *args: VariableTracker,
        **kwargs: VariableTracker,
    ) -> NoReturn:
        unimplemented(
            gb_type="Unsupported Tensor.sparse_resize_() call",
            context=f"call_method {self} sparse_resize_ {args} {kwargs}",
            explanation="Dynamo currently does not support tracing `Tensor.sparse_resize_()`.",
            hints=[],
        )

    def method_sparse_resize_and_clear_(
        self,
        tx: "InstructionTranslator",
        *args: VariableTracker,
        **kwargs: VariableTracker,
    ) -> NoReturn:
        unimplemented(
            gb_type="Unsupported Tensor.sparse_resize_and_clear_() call",
            context=f"call_method {self} sparse_resize_and_clear_ {args} {kwargs}",
            explanation="Dynamo currently does not support tracing `Tensor.sparse_resize_and_clear_()`.",
            hints=[],
        )

    def method_set_(
        self,
        tx: "InstructionTranslator",
        *args: VariableTracker,
        **kwargs: VariableTracker,
    ) -> None:
        if len(args) > 1:
            # torch.Tensor.set_() has several overloads.
            # aten::set_.source_Tensor(Tensor) gets special handling
            # in AOTAutograd and functionalization, because it is the most common
            # overload and is used by FSDP.
            # graph-breaking on aten::set_source_Tensor_storage_offset for now,
            # unless we find that we need to make it work.
            unimplemented(
                gb_type="Unsupported Tensor.set_() call",
                context=f"call_method {self} set_ {args} {kwargs}",
                explanation="Dynamo currently does not support tracing `Tensor.set_()` "
                "overloads that include more than one argument.",
                hints=[*graph_break_hints.SUPPORTABLE],
            )
        return None

    def method_add_(
        self,
        tx: "InstructionTranslator",
        other: VariableTracker,
        *,
        alpha: VariableTracker | None = None,
    ) -> VariableTracker | None:
        if alpha is not None and config.enable_dynamo_decompositions:
            result = variables.TorchInGraphFunctionVariable(torch.mul).call_function(
                tx, [other, alpha], {}
            )
            return self.call_method(tx, "add_", [result], {})
        return None

    def method_addcdiv_(
        self,
        tx: "InstructionTranslator",
        tensor1: "TensorVariable",
        tensor2: "TensorVariable",
        *,
        value: VariableTracker | None = None,
    ) -> VariableTracker | None:
        if value is not None and config.enable_dynamo_decompositions:
            result = variables.TorchInGraphFunctionVariable(torch.div).call_function(
                tx, [tensor1, tensor2], {}
            )
            result = variables.TorchInGraphFunctionVariable(torch.mul).call_function(
                tx, [result, value], {}
            )
            return self.call_method(tx, "add_", [result], {})
        return None

    def method___contains__(
        self, tx: "InstructionTranslator", arg: VariableTracker
    ) -> VariableTracker:
        # Rewrite __contains__ here so that downstream passes can trace through
        # without dealing with unbacked symbool. Roughly the code we translate is:
        # def __contains__(self, x):
        #     return (x == self).any().item()
        result = variables.TorchInGraphFunctionVariable(torch.eq).call_function(
            tx, [self, arg], {}
        )
        result = variables.TorchInGraphFunctionVariable(torch.any).call_function(
            tx, [result], {}
        )
        return result.call_method(tx, "item", [], {})

    def method_redistribute(
        self,
        tx: "InstructionTranslator",
        *args: VariableTracker,
        **kwargs: VariableTracker,
    ) -> VariableTracker:
        # rewrite non-primitive args/kwargs to be included in the on-the-fly prim function
        # and rewrite args to have only proxyable args, then insert call_function
        args_as_value = [x.as_python_constant() for x in args]
        kwargs_as_value = {k: v.as_python_constant() for k, v in kwargs.items()}

        def redistribute_fn_with_prim_types(x: Any) -> Any:
            return x.redistribute(*args_as_value, **kwargs_as_value)

        # attach the same function name for better debugging
        redistribute_fn_with_prim_types.__name__ = "prim_redistribute"

        from .builder import wrap_fx_proxy

        return wrap_fx_proxy(
            tx=tx,
            proxy=tx.output.create_proxy(
                "call_function",
                redistribute_fn_with_prim_types,
                *proxy_args_kwargs([self], {}),
            ),
        )

    def method_to_local(
        self,
        tx: "InstructionTranslator",
        *args: VariableTracker,
        **kwargs: VariableTracker,
    ) -> VariableTracker:
        # rewrite non-primitive args/kwargs to be included in the on-the-fly prim function
        # and rewrite args to have only proxyable args, then insert call_function

        grad_placements_vt = kwargs.get(
            "grad_placements", ConstantVariable.create(None)
        )
        if isinstance(grad_placements_vt, variables.UserDefinedObjectVariable):
            # grad_placement is a sequence-like structure, iterate over the value
            grad_placements_vt = variables.BuiltinVariable(tuple).call_function(
                tx, [grad_placements_vt], {}
            )

        if kwargs.get("grad_placements") is not None:
            kwargs["grad_placements"] = grad_placements_vt

        args_as_value = [x.as_python_constant() for x in args]
        kwargs_as_value = {k: v.as_python_constant() for k, v in kwargs.items()}

        def to_local_fn_with_prim_types(x: Any) -> Any:
            return x.to_local(*args_as_value, **kwargs_as_value)

        # attach the same function name for better debugging
        to_local_fn_with_prim_types.__name__ = "prim_to_local"

        from .builder import wrap_fx_proxy

        return wrap_fx_proxy(
            tx=tx,
            proxy=tx.output.create_proxy(
                "call_function",
                to_local_fn_with_prim_types,
                *proxy_args_kwargs([self], {}),
            ),
        )

    def method_register_hook(
        self,
        tx: "InstructionTranslator",
        *args: VariableTracker,
        **kwargs: VariableTracker,
    ) -> VariableTracker:
        return self._method_register_hook(tx, "register_hook", *args, **kwargs)

    def method_register_post_accumulate_grad_hook(
        self,
        tx: "InstructionTranslator",
        *args: VariableTracker,
        **kwargs: VariableTracker,
    ) -> VariableTracker:
        return self._method_register_hook(
            tx, "register_post_accumulate_grad_hook", *args, **kwargs
        )

    def _method_register_hook(
        self, tx: "InstructionTranslator", name: str, hook: VariableTracker
    ) -> VariableTracker:
        # Note - do not arbitrarily add hooks here - make sure they match the same contract
        # see [On tensor.register_hook]

        if not self.source:
            # For intermediate tensors (those without a source), we have two approaches:
            # 1. When compiled autograd is enabled: use BackwardState to defer hook execution
            # 2. When compiled autograd is NOT enabled: use a custom autograd function
            if compiled_autograd.compiled_autograd_enabled:
                # Use the BackwardState approach for compiled autograd
                hook_name, bw_state_proxy = tx.output.add_backward_state_hook(hook)

                def _register_hook_trampoline(
                    tensor: torch.Tensor, bw_state: compiled_autograd.BackwardState
                ) -> None:
                    register_hook = getattr(tensor, name)
                    register_hook(
                        functools.partial(
                            trace_wrapped,
                            fn=call_hook_from_backward_state,
                            bw_state=bw_state,
                            hook_name=hook_name,
                        )
                    )
                    # TODO(jansel): returning None here is wrong, it should be
                    # RemovableHandle, but we need some extra work to support
                    # this properly.
                    return None

                from .builder import wrap_fx_proxy

                self_proxy = self.as_proxy()
                self_proxy.node.meta["has_backward_hook"] = True

                return wrap_fx_proxy(
                    tx,
                    tx.output.create_proxy(
                        "call_function",
                        _register_hook_trampoline,
                        (self_proxy, bw_state_proxy),
                        {},
                    ),
                )
            # ----------Handling intermediate tensor custom hooks------
            # Rewrite intermediate tensor hook as custom autograd function
            # Given:
            # glb_list = []
            # glb_dict = {}
            #
            # def fn(x):
            #     y = x * 2
            #     glb_list.append(y)
            #     glb_dict['tensor'] = y
            #     a = glb_list[0] * 3      # Should use output of register_hook
            #     b = glb_dict['tensor'] + 1  # Should use hooked_y
            #     y.register_hook(lambda grad: grad + 1)
            #     return (a + b).sum()
            # We basically want to replace y.register_hook(lambda grad: grad + 1) with
            # custom autograd function where the forward is just identity while backward
            # calls custom hook.
            # The algo works by:
            #    1. When we see a hook, create a node with custom autograd function apply (y')
            #    2. Move the custom autograd node just after definition of the intermediate tensor (y in above),
            #       THEN update references to y with y'.
            # As a result of this algo, above example turns into:
            # def fn(x):
            #     y = x * 2
            #     y_prime = custom_autograd_function.apply()
            #     glb_list.append(y_prime)
            #     glb_dict['tensor'] = y_prime
            #     a = glb_list[0] * 3
            #     b = glb_dict['tensor'] + 1
            #     return (a + b).sum()
            # Get the original tensor's node and save its current users
            tensor_node = self.as_proxy().node

            users_to_replace = list(tensor_node.users.keys())

            # Create the ApplyBackwardHook call
            apply_hook_var = variables.AutogradFunctionVariable(_ApplyBackwardHook)
            result = apply_hook_var.call_apply(tx, [self, hook], {})

            # Get the hooked tensor's node (this is the getitem node)
            tensor_prime_node = result.as_proxy().node

            # DFS to collect all nodes that tensor_prime_node depends on,
            # stopping at tensor_node. These are the nodes we need to move right
            # after tensor_node.
            nodes_to_move: list[torch.fx.Node] = []
            visited: set[torch.fx.Node] = set()

            def collect_deps(node: torch.fx.Node, stop_at: torch.fx.Node) -> None:
                if node in visited or node is stop_at:
                    return
                for arg in node.args:
                    if isinstance(arg, torch.fx.Node):
                        collect_deps(arg, stop_at)
                for kwarg in node.kwargs.values():
                    if isinstance(kwarg, torch.fx.Node):
                        collect_deps(kwarg, stop_at)
                visited.add(node)
                nodes_to_move.append(node)

            collect_deps(tensor_prime_node, tensor_node)

            # Move each node to right after tensor_node using node.append()
            insert_point = tensor_node
            for node in nodes_to_move:
                insert_point.append(node)
                insert_point = node

            # Replace uses of tensor with tensor_prime, but only for the users
            # that existed before we created the hook node
            for user in users_to_replace:
                user.replace_input_with(tensor_node, tensor_prime_node)

            # Update tensor to point to the tensor_prime
            assert isinstance(result, TensorVariable)
            self.proxy = result.as_proxy()
            # TensorVariable doesn't actually store the grad_fn
            # so this is fine.
            self.synchronize_attributes(tx)

            # Return a RemovableHandleVariable for API compatibility
            # can't fall through because side_effects.register_hook
            # require source.
            return variables.RemovableHandleVariable(
                mutation_type=variables.base.ValueMutationNew(),
            )

        handle_variable = variables.RemovableHandleVariable(
            mutation_type=variables.base.ValueMutationNew(),
        )
        tx.output.side_effects.register_hook(self, hook, handle_variable, name)
        return handle_variable

    def method_requires_grad_(
        self, tx: "InstructionTranslator", requires_grad: bool | VariableTracker = True
    ) -> VariableTracker:
        if requires_grad is not True:
            requires_grad = requires_grad.as_python_constant()  # type: ignore[attr-defined]

        if self.as_proxy().node.meta["example_value"].requires_grad != requires_grad:
            unimplemented(
                gb_type="Unsupported Tensor.requires_grad_() call",
                context=f"call_method {self} requires_grad_",
                explanation="Dynamo does not support changes to a Tensor's "
                "`requires_grad` through calling `requires_grad_()`.",
                hints=[],
            )
        else:
            return self

    def method_share_memory_(self) -> NoReturn:
        unimplemented(
            gb_type="Unsupported Tensor.share_memory_() call",
            context=f"call_method {self} share_memory_",
            explanation="Dynamo does not support Tensor.share_memory_() which modifies tensor storage for IPC",
            hints=[
                "Move share_memory_() call outside the compiled region. ",
                *graph_break_hints.SUPPORTABLE,
            ],
        )

    def method_new(
        self,
        tx: "InstructionTranslator",
        *args: VariableTracker,
        **kwargs: VariableTracker,
    ) -> VariableTracker | None:
        # Convert x.new(torch.Size) into x.new_empty(torch.Size),
        # as Tensor.new acts differently with a Size input versus a tuple input.
        if (len(args) == 1 and isinstance(args[0], SizeVariable)) or (
            len(args) >= 1
            and all(
                a.is_python_constant() and isinstance(a.as_python_constant(), int)
                for a in args
            )
        ):
            return self.call_method(tx, "new_empty", args, kwargs)
        return None

    def method_untyped_storage(
        self, tx: "InstructionTranslator"
    ) -> "UntypedStorageVariable":
        return UntypedStorageVariable(
            self, self.as_proxy().node.meta["example_value"].untyped_storage()
        )

    def set_name_hint(self, name: str) -> None:
        if not self._is_name_set:
            self.proxy.node._rename(name)
            self._is_name_set = True
        return None

    def is_python_hashable(self) -> bool:
        # Tensors are hashable if they have an example_value (a fake tensor)
        # Most VT's should have one.
        # It'd be nice if at some point we could assert that they all have one
        return self.as_proxy().node.meta["example_value"] is not None

    def get_python_hash(self) -> int:
        return hash(self.as_proxy().node.meta["example_value"])

    def is_python_equal(self, other: object) -> bool:
        if not isinstance(other, VariableTracker):
            return False
        a = self.as_proxy().node.meta["example_value"]
        b = other.as_proxy().node.meta["example_value"]
        return a is b


class SymNodeVariable(VariableTracker):
    """
    Represents a symbolic scalar, either int, float or bool.  This is most commonly used to
    handle symbolic size computation, e.g., tensor.size(0), but it is also used to
    handle logic like float_tensor.item() or unspecialized float inputs.
    """

    _nonvar_fields = {
        "proxy",
        "sym_num",
        *VariableTracker._nonvar_fields,
    }

    def debug_repr(self) -> str:
        return repr(self.sym_num)

    @classmethod
    def create(
        cls,
        tx: "InstructionTranslatorBase",
        proxy: Any,
        sym_num: Any | None = None,
        **options: Any,
    ) -> "VariableTracker":
        if sym_num is None:
            sym_num = get_fake_value(proxy.node, tx)
        if "example_value" in proxy.node.meta:
            assert proxy.node.meta["example_value"] == sym_num
        set_example_value(proxy.node, sym_num)

        if isinstance(sym_num, (sympy.Integer, int, bool)):
            sym_num = int(sym_num) if isinstance(sym_num, sympy.Integer) else sym_num
            return ConstantVariable.create(sym_num)

        out = SymNodeVariable(proxy, sym_num, **options)
        if proxy.node.op != "placeholder":
            tx.output.current_tracer.record_tensor_or_symint_vt(out)
        return out

    def __init__(self, proxy: Any, sym_num: Any, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.proxy = proxy
        # TODO: Should we allow non SymTypes here?  Today it is allowed
        self.sym_num = sym_num
        self._tensor_var: TensorVariable | None = None

    def python_type(self) -> type:
        if isinstance(self.sym_num, SymTypes):
            return self.sym_num.node.pytype
        else:
            return type(self.sym_num)

    def is_symnode_like(self) -> bool:
        return True

    def as_proxy(self) -> Any:
        return self.proxy

    def as_tensor(self, tx: "InstructionTranslatorBase", dtype: Any) -> TensorVariable:
        if self._tensor_var is None:
            self._tensor_var = VariableTracker.build(
                tx, torch.scalar_tensor
            ).call_function(tx, [self], {"dtype": VariableTracker.build(tx, dtype)})
        return self._tensor_var

    def evaluate_expr(
        self, output_graph: Optional["OutputGraph"] = None
    ) -> bool | int | float:
        try:
            return guard_scalar(self.sym_num)
        except GuardOnDataDependentSymNode as e:
            if torch.fx.experimental._config.no_data_dependent_graph_break:
                raise

            raise UserError(  # noqa: B904
                UserErrorType.ANTI_PATTERN,
                f"Consider annotating your code using torch._check*(). {str(e)}",
                case_name="constrain_as_size_example",
            )

    def call_method(
        self,
        tx: "InstructionTranslator",
        name: str,
        args: Sequence[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        from .builder import wrap_fx_proxy

        return wrap_fx_proxy(
            tx,
            tx.output.create_proxy(
                "call_method",
                name,
                *proxy_args_kwargs([self, *args], kwargs),
            ),
        )

    def is_python_hashable(self) -> bool:
        return True

    def get_python_hash(self) -> int:
        # Essentially convert the SymNode to a constant variable whenever its
        # searched for a dict key.
        return hash(self.evaluate_expr())

    def is_python_equal(self, other: object) -> bool:
        if isinstance(other, SymNodeVariable):
            return self.evaluate_expr() == other.evaluate_expr()
        # could be constant variable as well
        return (
            isinstance(other, VariableTracker)
            and self.evaluate_expr() == other.as_python_constant()
        )


class NumpyNdarrayVariable(TensorVariable):
    """
    Represents a np.ndarray, but backed by torch Tensor via torch._numpy.ndarray.
    Use this for Tensor.numpy() call.
    """

    @staticmethod
    def create(
        tx: "InstructionTranslator", proxy: torch.fx.Proxy, **options: Any
    ) -> "NumpyNdarrayVariable":
        from .builder import wrap_fx_proxy_cls

        return wrap_fx_proxy_cls(
            target_cls=NumpyNdarrayVariable,
            tx=tx,
            proxy=proxy,
            **options,
        )

    def var_getattr(self, tx: "InstructionTranslator", name: str) -> VariableTracker:
        # NB: This INTENTIONALLY does not call super(), because there is
        # no intrinsic reason ndarray properties are related to Tensor
        # properties.  The inheritance here is for implementation sharing.

        from ..utils import numpy_attr_wrapper
        from .builder import wrap_fx_proxy

        result = None

        example_value = self.as_proxy().node.meta["example_value"]
        example_ndarray = tnp.ndarray(example_value)

        def insert_into_graph() -> VariableTracker:
            return wrap_fx_proxy(
                tx,
                tx.output.create_proxy(
                    "call_function", numpy_attr_wrapper, (self.as_proxy(), name), {}
                ),
            )

        if name in ["T", "real", "imag"]:
            proxy = tx.output.create_proxy(
                "call_function",
                numpy_attr_wrapper,
                (self.as_proxy(), name),
                {},
            )
            result = NumpyNdarrayVariable.create(tx, proxy)

        # These are awkward to implement.  The standard playbook for torch._numpy
        # interop is to trace a call into the torch._numpy wrapper which works for
        # Tensor operations.  However, we don't want to do this for calls
        # that don't return Tensors, because in those cases we may not want
        # to trace the attribute access into the graph at all (it is sort
        # of harmless to do so, because AOTAutograd will eliminate them,
        # but it's best not to trace them in to begin with.)  But in any
        # case, tracing these into the graph is like trying to fit a square
        # peg into a round hole; best not to do it.  So instead we
        # painstakingly implement these by hand
        #
        # NB: only ALWAYS specialized attributes can go here; notably,
        # size/shape not allowed!
        elif name in ("ndim", "itemsize"):
            return ConstantVariable.create(getattr(example_ndarray, name))
        elif name in ("shape", "stride"):
            if not has_free_symbols(r := getattr(example_ndarray, name)):
                return ConstantVariable.create(tuple(int(r) for r in r))
            return insert_into_graph()
        elif name == "size":
            if not has_free_symbols(r := example_ndarray.size):
                return ConstantVariable.create(int(r))
            return insert_into_graph()
        elif name in ["base", "flags", "dtype"]:
            unimplemented(
                gb_type="Unsupported ndarray attribute access",
                context=f"var_getattr {self} {name}",
                explanation=f"Dynamo currently does not support tracing `ndarray.{name}`.",
                hints=[],
            )
        elif name == "__version__":
            unimplemented(
                gb_type="Unsupported ndarray.__version__ access",
                context=f"var_getattr {self} {name}",
                explanation=f"Dynamo currently does not support tracing `ndarray.{name}`.",
                hints=[],
            )
        if result is None:
            raise NotImplementedError
        return result

    @staticmethod
    def patch_args(
        name: str, args: Sequence[VariableTracker], kwargs: dict[str, VariableTracker]
    ) -> tuple[Sequence[VariableTracker], dict[str, VariableTracker]]:
        if name == "clip":
            kwargs_rename = {"a_min": "min", "a_max": "max"}
            kwargs = {kwargs_rename.get(k, k): v for k, v in kwargs.items()}
        return args, kwargs

    def call_method(
        self,
        tx: "InstructionTranslator",
        name: str,
        args: Sequence[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        from ..exc import unimplemented
        from ..utils import numpy_method_wrapper

        args, kwargs = self.patch_args(name, args, kwargs)

        if name == "astype":
            from .builtin import BuiltinVariable

            dtype_arg = None
            if "dtype" in kwargs:
                dtype_arg = kwargs["dtype"]
            elif len(args) > 0:
                dtype_arg = args[0]
            is_object_str = dtype_arg is not None and dtype_arg.is_constant_match("O")
            is_object_type = (
                isinstance(dtype_arg, BuiltinVariable) and dtype_arg.fn is object
            )
            if is_object_str or is_object_type:
                unimplemented(
                    gb_type="ndarray.astype(object)",
                    context=f"call_method {self} {name} {args} {kwargs}",
                    explanation=(
                        "`ndarray.astype('O')` or `ndarray.astype(object)` is not supported "
                        "by torch.compile, as there is no equivalent to object type in torch.Tensor. "
                        "This will be executed eagerly."
                    ),
                    hints=[*graph_break_hints.FUNDAMENTAL],
                )
        if name in ["__len__", "size", "tolist", "__iter__"]:
            # delegate back to TensorVariable
            return super().call_method(tx, name, args, kwargs)
        if name in ("tostring", "tobytes", "__delattr__"):
            unimplemented(
                gb_type="Unsupported ndarray method call",
                context=f"call_method {self} {name} {args} {kwargs}",
                explanation=f"`ndarray.{name}()` is not modelled in `torch._numpy`.",
                hints=[],
            )
        proxy = tx.output.create_proxy(
            "call_function",
            numpy_method_wrapper(name),
            *proxy_args_kwargs([self] + list(args), kwargs),
        )
        return NumpyNdarrayVariable.create(tx, proxy)

    def python_type(self) -> type:
        if np is not None:
            return np.ndarray
        else:
            return NoneType


class UnspecializedPythonVariable(TensorVariable):
    """
    This is a 1-element tensor represents unspecialized python float/int.
    """

    _nonvar_fields = {
        "raw_value",
        "need_unwrap",
        *TensorVariable._nonvar_fields,
    }

    def __init__(
        self,
        proxy: torch.fx.Proxy,
        *,
        raw_value: float | int | None = None,
        need_unwrap: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(proxy, **kwargs)
        self.raw_value = raw_value
        self.need_unwrap = need_unwrap

    @classmethod
    def from_tensor_variable(
        cls,
        tensor_variable: TensorVariable,
        raw_value: float | int | None,
        need_unwrap: bool = True,
    ) -> "UnspecializedPythonVariable":
        # Convert a `TensorVariable` instance into an `UnspecializedPythonVariable` instance.
        return UnspecializedPythonVariable(
            **dict(tensor_variable.__dict__),
            raw_value=raw_value,
            need_unwrap=need_unwrap,
        )


class FakeItemVariable(TensorVariable):
    """An unspecialized python variable which prevents access to the underlying raw value.
    This is needed if item is called on a FakeTensor."""

    _nonvar_fields = {
        "need_unwrap",
        *TensorVariable._nonvar_fields,
    }

    def __init__(self, proxy: torch.fx.Proxy, **kwargs: Any) -> None:
        need_unwrap = kwargs.pop("need_unwrap", False)
        super().__init__(proxy, **kwargs)
        self.need_unwrap = need_unwrap

    @classmethod
    def from_tensor_variable(
        cls, tensor_variable: TensorVariable
    ) -> "FakeItemVariable":
        return FakeItemVariable(**dict(tensor_variable.__dict__))


class TensorSubclassVariable(UserDefinedClassVariable):
    def call_function(
        self,
        tx: "InstructionTranslator",
        args: Sequence[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # Handle `Subclass(existing_tensor, ...)` calls.
        from .torch_function import TensorWithTFOverrideVariable

        new_func = self.value.__new__
        if new_func is torch.Tensor.__new__:
            var = None
            if len(args) == 1 and args[0].is_tensor() and len(kwargs) == 0:
                data = args[0]
                # Simulate `torch.Tensor.__new__` as shallow-copying the input
                # tensor data with a new type. TODO polyfill?
                var = TensorWithTFOverrideVariable.from_tensor_var(
                    tx, data, self.value, self.source
                )
            else:
                unimplemented(
                    gb_type="Calling subclass default constructor with more than tensor argument",
                    context=f"{self.value}(args={args}, kwargs={kwargs})",
                    explanation="Currently not supported",
                    hints=[
                        "Avoid this constructor call or move it outside "
                        "`torch.compile` regione",
                        *graph_break_hints.SUPPORTABLE,
                    ],
                )
        else:
            # Let Dynamo trace through custom `__new__`
            var = VariableTracker.build(tx, new_func).call_function(
                tx, [self] + list(args), kwargs
            )
        assert var is not None
        # Let Dynamo trace through custom `__init__`
        init_func = self.value.__init__
        # TODO builder should be able to handle `torch.Tensor.__init__`,
        # which is `object.__init__`, so that we can remove this check.
        if init_func is not torch.Tensor.__init__:
            VariableTracker.build(tx, init_func).call_function(tx, [var], kwargs)

        # See NOTE [Side effect tracking for newly constructed tensor]
        tx.output.side_effects._track_obj(
            object(), var, mutation_type_cls=AttributeMutationNew
        )
        return var

    def as_python_constant(self) -> type:
        return self.value


class UntypedStorageVariable(VariableTracker):
    _nonvar_fields = {
        "example_value",
        *VariableTracker._nonvar_fields,
    }

    def __init__(
        self,
        from_tensor: TensorVariable,
        example_value: torch.UntypedStorage,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.from_tensor = from_tensor
        # Example_value will always have device="meta"
        self.example_value = example_value

    def call_method(
        self,
        tx: "InstructionTranslator",
        name: str,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        if name == "size":
            if args or kwargs:
                raise_args_mismatch(
                    tx,
                    name,
                    "0 args and 0 kwargs",
                    f"{len(args)} args and {len(kwargs)} kwargs",
                )
            result = self.example_value.size()
            if not has_free_symbols(result):
                # avoid creating a node in the graph
                return ConstantVariable.create(int(result))
            else:
                from ..external_utils import untyped_storage_size
                from .builder import wrap_fx_proxy

                return wrap_fx_proxy(
                    tx,
                    tx.output.create_proxy(
                        "call_function",
                        untyped_storage_size,
                        (self.from_tensor.as_proxy(),),
                        {},
                    ),
                )
        if name == "resize_" and len(args) == 1:
            if kwargs:
                raise_args_mismatch(tx, name, "0 kwargs", f"{len(kwargs)} kwargs")
            tx.output.create_proxy(
                "call_function",
                torch.ops.inductor.resize_storage_bytes_,
                (self.from_tensor.as_proxy(), args[0].as_proxy()),
                {},
            )
            return self

        return super().call_method(tx, name, args, kwargs)

    def reconstruct(self, codegen: "PyCodegen") -> None:
        codegen(self.from_tensor)
        codegen.load_method("untyped_storage")
        codegen.call_method(0)


class DataPtrVariable(VariableTracker):
    def __init__(
        self,
        from_tensor: TensorVariable,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.from_tensor = from_tensor

    def reconstruct(self, codegen: "PyCodegen") -> None:
        codegen(self.from_tensor)
        codegen.load_method("data_ptr")
        codegen.call_method(0)
