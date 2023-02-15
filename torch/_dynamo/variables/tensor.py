import inspect
import itertools
import operator
import types
from typing import Dict, List

import torch.fx
import torch.random
from torch.fx.experimental.symbolic_shapes import guard_scalar

from .. import config, variables
from ..exc import unimplemented
from ..guards import GuardBuilder
from ..source import AttrSource

from ..utils import (
    fqn,
    get_fake_value,
    get_real_value,
    HAS_NUMPY,
    np,
    product,
    proxy_args_kwargs,
    tensortype_to_dtype,
)
from .base import VariableTracker
from .constant import ConstantVariable
from .lists import ShapeVariable, SizeVariable

supported_tensor_comparison_ops = {
    ">": operator.gt,
    "<": operator.lt,
    ">=": operator.ge,
    "<=": operator.le,
    "==": operator.eq,
    "!=": operator.ne,
}
supported_const_comparison_ops = {
    "is": operator.is_,
    "is not": operator.is_not,
    "==": operator.eq,
    "!=": operator.ne,
}


class TensorVariable(VariableTracker):
    """A torch.Tensor input or an intermediate value in the FX graph"""

    _nonvar_fields = [
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
    ]

    def get_real_value(self):
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
        dtype=None,
        device=None,
        layout=None,
        ndim=None,
        size=None,
        stride=None,
        requires_grad=None,
        is_quantized=None,
        is_contiguous=None,
        is_sparse=None,
        class_type=torch.Tensor,
        specialized_value=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.proxy = proxy
        self.dtype = dtype
        self.device = device
        self.layout = layout
        self.ndim = ndim
        self.size = size
        self.stride = stride
        self.requires_grad = requires_grad
        self.is_quantized = is_quantized
        self.is_contiguous = is_contiguous
        self.is_sparse = is_sparse
        self.class_type = class_type
        self.specialized_value = specialized_value

    def as_proxy(self):
        return self.proxy

    def python_type(self):
        return self.class_type

    def call_isinstance(self, tensor_type):
        def check_type(ty):
            if ty not in tensortype_to_dtype:
                return issubclass(self.python_type(), ty)

            dtypes = tensortype_to_dtype[ty]
            return self.dtype in dtypes

        if type(tensor_type) is tuple:
            return any([check_type(ty) for ty in tensor_type])
        else:
            return check_type(tensor_type)

    @staticmethod
    def specialize(value: torch.Tensor):
        props = {
            "dtype": value.dtype,
            "device": value.device,
            "layout": value.layout,
            "ndim": int(value.ndim),
            "requires_grad": value.requires_grad,
            "is_quantized": value.is_quantized,
            "is_sparse": value.is_sparse,
            "class_type": type(value),
        }
        if not config.dynamic_shapes:
            props["size"] = tuple(value.size())
            props["stride"] = tuple(value.stride())
            props["is_contiguous"] = tuple(
                [
                    x
                    for x in torch._prims_common._memory_formats
                    if value.is_contiguous(memory_format=x)
                ]
            )
        return props

    def var_getattr(self, tx, name):
        from . import ConstantVariable, TorchVariable

        result = None
        options = VariableTracker.propagate(self)
        if name == "ndim" and self.ndim is not None:
            result = ConstantVariable(self.ndim, **options)
        elif name == "dtype" and self.dtype is not None:
            result = TorchVariable(self.dtype, **options)
        elif name == "device" and self.device is not None:
            result = TorchVariable(self.device, **options)
        elif name == "layout" and self.layout is not None:
            result = TorchVariable(self.layout, **options)
        elif name == "is_cuda" and self.device is not None:
            result = ConstantVariable(self.device.type == "cuda", **options)
        elif name == "shape" and self.size is not None:
            sizes = [variables.ConstantVariable(x) for x in self.size]
            result = ShapeVariable(sizes, **options)
        elif name == "requires_grad" and self.requires_grad is not None:
            result = ConstantVariable(self.requires_grad, **options)
        elif name == "is_quantized" and self.is_quantized is not None:
            result = ConstantVariable(self.is_quantized, **options)
        elif name == "is_sparse" and self.is_sparse is not None:
            result = ConstantVariable(self.is_sparse, **options)
        elif name == "shape" and self.size is None:
            result = self.call_method(tx, "size", [], {})
        elif name == "ndim" and self.ndim is None:
            result = self.call_method(tx, "dim", [], {})
        elif name == "data":
            result = self.call_method(tx, "detach", [], {})
        if name == "__class__":
            return TorchVariable(self.python_type(), **options)

        # Add a guard for type matching, these guards are checked before tensor guards
        # In some cases, a <tensor>.<attr> guard can be evaluated first, and break if
        # <tensor> is later changed to another type
        if result is not None and self.source is not None:
            result = result.add_guard(self.make_guard(GuardBuilder.TYPE_MATCH))

        # For attributes (not methods) that were not caught in the special handling above,
        # (e.g. tensor.real), we handle these generically, assuming that the output type is
        # a tensor.
        if result is None:

            def try_generic_attr_handling():
                from .builder import wrap_fx_proxy
                from .misc import GetAttrVariable

                try:
                    static_attr = inspect.getattr_static(torch.Tensor, name)
                except AttributeError:
                    return None

                # Make sure this is an attribute, not a method.
                # type(torch.Tensor.H) should be "getset_descriptor"
                # This is a because of CPython implementation, see THPVariableType:
                # these attributes are implemented under tp_getset, which appear
                # as `getset_descriptor`s, (compared to, say, methods which appear
                # as `method_descriptor`s)
                if type(static_attr) != types.GetSetDescriptorType:
                    return None

                return wrap_fx_proxy(
                    tx=tx,
                    proxy=GetAttrVariable.create_getattr_proxy(self.as_proxy(), name),
                    **options,
                )

            result = try_generic_attr_handling()

        if result is None:
            raise NotImplementedError()

        return result

    def has_unpack_var_sequence(self, tx):
        return (self.size is not None and len(self.size) > 0) or (
            self.size is None and config.dynamic_shapes
        )

    def unpack_var_sequence(self, tx, idxes=None):
        from .builder import wrap_fx_proxy

        options = VariableTracker.propagate(self)
        if idxes is None:
            if self.size:
                length = self.size[0]
            else:
                dyn_length = self.call_method(tx, "size", [ConstantVariable(0)], {})
                assert isinstance(dyn_length, SymNodeVariable)
                length = dyn_length.evaluate_expr(tx.output)
            idxes = range(length)
        return [wrap_fx_proxy(tx, self.as_proxy()[i], **options) for i in idxes]

    def call_method(
        self,
        tx,
        name,
        args: "List[VariableTracker]",
        kwargs: "Dict[str, VariableTracker]",
    ) -> "VariableTracker":
        from . import ConstantVariable, TorchVariable, TupleVariable
        from .builder import wrap_fx_proxy

        kwargs = dict(kwargs)
        options = VariableTracker.propagate(self, args, kwargs.values())
        if name == "stride" and self.stride is not None:
            constant_result = ConstantVariable(self.stride, **options)
        elif name == "size" and self.size is not None:
            sizes = [variables.ConstantVariable(x) for x in self.size]
            constant_result = SizeVariable(sizes, **options)
        elif name == "size" and self.size is None and config.dynamic_shapes:
            return wrap_fx_proxy(
                tx,
                tx.output.create_proxy(
                    "call_method",
                    name,
                    *proxy_args_kwargs([self] + list(args), kwargs),
                ),
                **options,
            )
        elif name in ("numel", "nelement") and self.size is not None:
            constant_result = ConstantVariable(product(self.size), **options)
        elif name in ("ndimension", "dim") and self.ndim is not None:
            constant_result = ConstantVariable(self.ndim, **options)
        elif name == "is_floating_point" and self.dtype is not None:
            constant_result = ConstantVariable(self.dtype.is_floating_point, **options)
        elif name == "is_contiguous" and self.is_contiguous is not None:
            if "memory_format" in kwargs:
                memory_format = kwargs.pop("memory_format").as_python_constant()
            else:
                memory_format = torch.contiguous_format
            constant_result = ConstantVariable(
                memory_format in self.is_contiguous, **options
            )
        elif (
            name == "type"
            and self.dtype is not None
            and len(args) == 0
            and isinstance(self.device, torch.device)
        ):
            tensortype = [k for k, v in tensortype_to_dtype.items() if self.dtype in v][
                0
            ]
            if self.device.type == "cuda":
                constant_result = ConstantVariable(
                    f"torch.cuda.{tensortype.__name__}", **options
                )
            else:
                constant_result = ConstantVariable(
                    f"torch.{tensortype.__name__}", **options
                )
        elif (
            name == "type"
            and len(args) == 1
            and fqn(type(args[0].as_python_constant())) == "torch.tensortype"
        ):
            # torch.FloatTensor, etc. are all of type "torch.tensortype".
            # torch.fx's tracer fails on these types, because it doesn't support arguments of torch.tensortype type.
            # So, we pass it in as a string (which is also supported, see above implementation for .type() with 0 args)
            tensor_type = args[0].as_python_constant()
            tensor_type_const = ConstantVariable(fqn(tensor_type), **options)
            return wrap_fx_proxy(
                tx,
                tx.output.create_proxy(
                    "call_method",
                    name,
                    *proxy_args_kwargs([self, tensor_type_const], kwargs),
                ),
                **options,
            )
        elif name == "get_device" and isinstance(self.device, torch.device):
            index = self.device.index if self.device.type != "cpu" else -1
            constant_result = ConstantVariable(index, **options)
        else:
            constant_result = None

        if constant_result:
            assert not kwargs, f"Tensor.{name}() unhandled kwargs"
            if len(args) == 1:
                return constant_result.getitem_const(args[0])
            elif args:
                return TupleVariable(
                    [constant_result.getitem_const(a) for a in args], **options
                )
            return constant_result
        elif (
            name == "repeat"
            and not all(
                x.is_python_constant() for x in itertools.chain(args, kwargs.values())
            )
            and not config.dynamic_shapes
        ):
            unimplemented("dynamic Tensor.repeat")
        elif name in ("tolist", "numpy", "backward", "data_ptr"):
            unimplemented(f"Tensor.{name}")
        elif name == "nonzero" and not config.dynamic_shapes:
            unimplemented(f"Tensor.{name}")
        elif name == "item" and not config.capture_scalar_outputs:
            unimplemented(f"Tensor.{name}")
        elif (
            name == "item"
            and config.capture_scalar_outputs
            and not config.dynamic_shapes
        ):
            raise AssertionError(
                "To capture_scalar_outputs, you must also set dynamic_shapes = True"
            )
        elif name == "__len__":
            return self.call_method(tx, "size", [ConstantVariable(0, **options)], {})
        elif name == "__setitem__":
            tx.output.guards.update(options["guards"])
            tx.output.create_proxy(
                "call_function",
                operator.setitem,
                *proxy_args_kwargs([self] + list(args), kwargs),
            )
            return ConstantVariable(None, **options)
        elif name in ("resize_", "resize_as_"):
            if "memory_format" in kwargs:
                memory_format = kwargs["memory_format"].as_python_constant()
            else:
                memory_format = torch.contiguous_format

            if name == "resize_":
                self.size = args[0].as_python_constant()
                self.is_contiguous = (memory_format,)
            else:
                assert isinstance(args[0], TensorVariable)
                if self.size and args[0].size:
                    if (
                        self.size == args[0].size
                        or memory_format is torch.preserve_format
                    ):
                        self.is_contiguous = args[0].is_contiguous
                    else:
                        self.size = args[0].size
                        self.stride = args[0].stride
                        self.ndim = args[0].ndim
                        self.is_contiguous = (memory_format,)

            return wrap_fx_proxy(
                tx,
                tx.output.create_proxy(
                    "call_method",
                    name,
                    *proxy_args_kwargs([self] + list(args), kwargs),
                ),
                **options,
            )
        elif (
            name == "add_" and len(args) == 1 and len(kwargs) == 1 and "alpha" in kwargs
        ):
            result = TorchVariable(torch.mul, **options).call_function(
                tx, args + [kwargs["alpha"]], {}
            )
            return self.call_method(tx, "add_", [result], {})
        elif (
            name == "addcdiv_"
            and len(args) == 2
            and len(kwargs) == 1
            and "value" in kwargs
        ):
            result = TorchVariable(torch.div, **options).call_function(tx, args, {})
            result = TorchVariable(torch.mul, **options).call_function(
                tx, [result, kwargs["value"]], {}
            )
            return self.call_method(tx, "add_", [result], {})
        else:
            # Convert x.new(torch.Size) into x.new_empty(torch.Size),
            # as Tensor.new acts differently with a Size input versus a tuple input.
            if (
                name == "new"
                and len(args) == 1
                and isinstance(args[0], (SizeVariable, ShapeVariable))
                and not config.dynamic_shapes
            ):
                name = "new_empty"
            return wrap_fx_proxy(
                tx,
                tx.output.create_proxy(
                    "call_method",
                    name,
                    *proxy_args_kwargs([self] + list(args), kwargs),
                ),
                **options,
            )


class SymNodeVariable(VariableTracker):
    """
    Represents a symbolic size, e.g., as returned by tensor.size(0)
    """

    @classmethod
    def create(cls, tx, proxy, sym_num, **options):
        if "example_value" in proxy.node.meta:
            assert proxy.node.meta["example_value"] == sym_num
        if sym_num is None:
            sym_num = get_fake_value(proxy.node, tx)
        proxy.node.meta["example_value"] = sym_num
        return SymNodeVariable(proxy, sym_num, **options)

    def __init__(self, proxy, sym_num, **kwargs):
        super().__init__(**kwargs)
        self.proxy = proxy
        self.sym_num = sym_num

    def python_type(self):
        return type(self.sym_num)

    def unpack_var_sequence(self, tx):
        super().unpack_var_sequence(tx)

    def as_proxy(self):
        return self.proxy

    def evaluate_expr(self, output_graph):
        return guard_scalar(self.sym_num)

    def call_method(
        self,
        tx,
        name,
        args: "List[VariableTracker]",
        kwargs: "Dict[str, VariableTracker]",
    ) -> "VariableTracker":
        from .builder import wrap_fx_proxy

        options = VariableTracker.propagate(self, args, kwargs.values())

        return wrap_fx_proxy(
            tx,
            tx.output.create_proxy(
                "call_method",
                name,
                *proxy_args_kwargs([self] + list(args), kwargs),
            ),
            **options,
        )


class TensorWithTFOverrideVariable(VariableTracker):
    """
    Represents a tensor subclass instance with a __torch_function__ override.
    """

    def __init__(
        self,
        tensor_variable,
        orig_tensor_variable_source,
        subclass_torch_function__func,
        subclass_type,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.tensor_variable = tensor_variable
        self.orig_tensor_variable_source = orig_tensor_variable_source
        self.subclass_torch_function__func = subclass_torch_function__func
        self.subclass_type = subclass_type

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
        # insert unwrapped version of self as the first argument
        # TODO: This is wrong!  When you call the internal __torch_function__,
        # you still get the wrapped version of self, and if you call functions
        # inside __torch_function__, they should come back here.  If we unwrap
        # the tensor immediately, that will not happen.
        # See https://github.com/pytorch/torchdynamo/issues/1951
        args = list(args)
        args.insert(0, self.tensor_variable)
        func_var = GetAttrVariable(self.tensor_variable, name)

        unwrapped = TensorWithTFOverrideVariable.inline_torch_function_unwrapped(
            tx,
            func_var,
            self.orig_tensor_variable_source,
            self.subclass_torch_function__func,
            self.subclass_type,
            options,
            args,
            kwargs,
        )

        # TODO(future PR): implement rewrapping conditional on method presence
        # in `torch.overrides.get_default_nowrap_function()`. It's unclear how
        # to do this easily in the current codebase since the resolution of
        # `GetAttrVariable` depends on the type of the underlying object.

        return TensorWithTFOverrideVariable(
            unwrapped,
            self.orig_tensor_variable_source,
            self.subclass_torch_function__func,
            self.subclass_type,
        )

    @staticmethod
    def inline_torch_function_unwrapped(
        tx,
        original_func_var,
        tensor_with_tf_override_source,
        tf_func,
        subclass_type,
        options,
        args,
        kwargs,
    ):
        """
        This function inlines the `__torch_function__` override for `original_func_var`.
        For example, if the user code is

           x1 = torch.sigmoid(x0)

        And `x0` has an override, then:
        * `original_func_var` will be a `VariableTracker` object wrapping `torch.sigmoid`
        * `tensor_with_tf_override_source` will be the `Source` object from
          the original tensor override instance in the beginning of the program
        * `tf_func` will be the custom `__torch_function__` function
        * `subclass_type` will be `type(x0)`

        The caller is expected to properly massage args and kwargs before
        passing them into this function.

        The caller is responsible for wrapping the return value, if needed.
        """
        from . import UserDefinedClassVariable
        from .builder import TupleVariable, VariableBuilder

        source = AttrSource(
            AttrSource(tensor_with_tf_override_source, "__torch_function__"),
            "__func__",
        )
        tf_func_var = VariableBuilder(tx, source)(tf_func)
        type_var = UserDefinedClassVariable(subclass_type, **options)

        # signature:
        # def __torch_function__(cls, func, types, args=(), kwargs=None):
        tf_args = (
            type_var,  # cls
            original_func_var,  # func
            (type_var,),  # types
            TupleVariable(args),  # args
            kwargs,  # kwargs
        )

        # Disable __torch_function__ here to prevent the clone of the
        # example tensor from going into the override.
        with torch._C.DisableTorchFunctionSubclass():
            return tx.inline_user_function_return(tf_func_var, tf_args, {})


class UnspecializedPythonVariable(TensorVariable):
    """
    This is a 1-element tensor represents unspecialized python float/int.
    """

    def __init__(self, proxy: torch.fx.Proxy, **kwargs):
        raw_value = kwargs.pop("raw_value", None)
        if HAS_NUMPY and isinstance(raw_value, np.number):
            raw_values = raw_value.item()
        need_unwrap = kwargs.pop("need_unwrap", True)
        super().__init__(proxy, **kwargs)
        self.raw_value = raw_value
        self.need_unwrap = need_unwrap

    @classmethod
    def from_tensor_variable(cls, tensor_variable, raw_value, need_unwrap=True):
        # Convert a `TensorVariable` instance into an `UnspecializedPythonVariable` instance.
        return UnspecializedPythonVariable(
            **dict(tensor_variable.__dict__),
            raw_value=raw_value,
            need_unwrap=need_unwrap,
        )

    def as_specialized(self, tx):
        for graph_arg in tx.output.graphargs:
            if graph_arg.source is self.source:
                graph_arg.erase()

        for g in self.guards:
            if g.is_volatile:
                g.create_fn = GuardBuilder.CONSTANT_MATCH

        return ConstantVariable(value=self.raw_value, guards=self.guards)


class FakeItemVariable(TensorVariable):
    """An unspecialized python variable which prevents access to the underlying raw value.
    This is needed if item is called on a FakeTensor."""

    def __init__(self, proxy: torch.fx.Proxy, **kwargs):
        need_unwrap = kwargs.pop("need_unwrap", False)
        super().__init__(proxy, **kwargs)
        self.need_unwrap = need_unwrap

    @classmethod
    def from_tensor_variable(cls, tensor_variable):
        return FakeItemVariable(**dict(tensor_variable.__dict__))
