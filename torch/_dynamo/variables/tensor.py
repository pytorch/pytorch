import itertools
import operator
from typing import Dict, List

import torch.fx
import torch.random

from .. import config, variables
from ..exc import unimplemented
from ..guards import GuardBuilder
from ..source import AttrSource

from ..utils import (
    fake_tensors_available,
    get_fake_value,
    get_real_value,
    product,
    proxy_args_kwargs,
    tensortype_to_dtype,
)
from .base import VariableTracker, wrap_fx_proxy
from .constant import ConstantVariable
from .lists import ShapeVariable, SizeVariable


class _missing:
    pass


class TensorVariable(VariableTracker):
    """A torch.Tensor input or an intermediate value in the FX graph"""

    _nonvar_fields = [
        "proxy",
        "dtype",
        "device",
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
        super(TensorVariable, self).__init__(**kwargs)
        self.proxy = proxy
        self.dtype = dtype
        self.device = device
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
            "ndim": int(value.ndim),
            "requires_grad": value.requires_grad,
            "is_quantized": value.is_quantized,
            "is_sparse": value.is_sparse,
            "class_type": type(value),
        }
        if not config.dynamic_shapes:
            props["size"] = tuple(value.size())
            props["stride"] = tuple(value.stride())
            props["is_contiguous"] = value.is_contiguous()
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

        if name == "__class__":
            return TorchVariable(self.python_type(), **options)

        # Add a guard for type matching, these guards are checked before tensor guards
        # In some cases, a <tensor>.<attr> guard can be evaluated first, and break if
        # <tensor> is later changed to another type
        if result is not None and self.source is not None:
            result = result.add_guard(self.make_guard(GuardBuilder.TYPE_MATCH))

        if result is None:
            raise NotImplementedError()

        return result

    def unpack_var_sequence(self, tx):
        options = VariableTracker.propagate(self)
        if self.size:
            return [
                variables.BuiltinVariable(operator.getitem, **options).call_function(
                    tx, [self, variables.ConstantVariable(i)], {}
                )
                for i in range(self.size[0])
            ]

        return super(TensorVariable, self).unpack_var_sequence(tx)

    def call_method(
        self,
        tx,
        name,
        args: "List[VariableTracker]",
        kwargs: "Dict[str, VariableTracker]",
    ) -> "VariableTracker":
        from . import ConstantVariable, TupleVariable

        kwargs = dict(kwargs)
        print("CALLING TENSOR OP", name)
        options = VariableTracker.propagate(self, args, kwargs.values())
        if name == "stride" and self.stride is not None:
            constant_result = ConstantVariable(self.stride, **options)
        elif name == "size" and self.size is not None:
            sizes = [variables.ConstantVariable(x) for x in self.size]
            constant_result = SizeVariable(sizes, **options)
        elif name == "size" and self.size is None and config.dynamic_shapes:
            example_size = self.proxy.node.meta["example_value"].size()
            pure_size_proxy = tx.output.create_proxy(
                "call_method",
                name,
                *proxy_args_kwargs([self], kwargs),
                current_tx=tx,
            )
            if len(args) == 1 and isinstance(args[0], ConstantVariable):
                assert isinstance(args[0].value, int)
                proxy = tx.output.create_proxy(
                    "call_function",
                    operator.getitem,
                    (pure_size_proxy, args[0].as_proxy()),
                    {},
                )
                value = example_size[args[0].value]
                proxy.node.meta["example_value"] = value
                return DynamicShapeVariable.create(tx, proxy, value)
            items = []
            for i, element in enumerate(example_size):
                if isinstance(element, int):
                    items.append(variables.ConstantVariable(element))
                else:
                    proxy = tx.output.create_proxy(
                        "call_function",
                        operator.getitem,
                        (pure_size_proxy, i),
                        {},
                    )
                    proxy.node.meta["example_value"] = element
                    items.append(DynamicShapeVariable.create(tx, proxy, element))

            def extract(item):
                if isinstance(item, ConstantVariable):
                    return item.value
                return item.proxy.node.meta["example_value"]

            pure_size_proxy.node.meta["example_value"] = torch.Size(
                [extract(item) for item in items]
            )
            return SizeVariable(items, pure_size_proxy, **options)
        elif name == "numel" and self.size is not None:
            constant_result = ConstantVariable(product(self.size), **options)
        elif name in ("ndimension", "dim") and self.ndim is not None:
            constant_result = ConstantVariable(self.ndim, **options)
        elif name == "is_floating_point" and self.dtype is not None:
            constant_result = ConstantVariable(self.dtype.is_floating_point, **options)
        elif name == "is_contiguous" and self.is_contiguous is not None:
            if (
                "memory_format" in kwargs
                and kwargs["memory_format"].as_python_constant()
                == torch.contiguous_format
            ):
                kwargs.pop("memory_format")
            constant_result = ConstantVariable(self.is_contiguous, **options)
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
        elif name in ("tolist", "numpy", "backward"):
            unimplemented(f"Tensor.{name}")
        elif name == "nonzero" and not config.dynamic_shapes:
            unimplemented(f"Tensor.{name}")
        elif name == "item":
            if config.capture_scalar_outputs:
                use_fake_tensors = (
                    fake_tensors_available and config.fake_tensor_propagation
                )
                if use_fake_tensors:
                    example_value = get_fake_value(self.proxy.node, tx)
                else:
                    example_value = get_real_value(self.proxy.node, tx.output).item()
                return wrap_fx_proxy(
                    tx,
                    tx.output.create_proxy(
                        "call_method", "item", (self.as_proxy(),), {}, current_tx=tx
                    ),
                    example_value=example_value,
                    **options,
                )
            else:
                unimplemented(f"Tensor.{name}")
        elif name == "__len__":
            if self.size:
                assert not config.dynamic_shapes
                return ConstantVariable(self.size[0], **options)
            else:
                return wrap_fx_proxy(
                    tx,
                    tx.output.create_proxy(
                        "call_function", len, (self.as_proxy(),), {}, current_tx=tx
                    ),
                    **options,
                )
        elif name == "__setitem__":
            tx.output.guards.update(options["guards"])
            tx.output.create_proxy(
                "call_function",
                operator.setitem,
                *proxy_args_kwargs([self] + args, kwargs),
                current_tx=tx,
            )
            return ConstantVariable(None, **options)
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
                    *proxy_args_kwargs([self] + args, kwargs),
                    current_tx=tx,
                ),
                **options,
            )


class DynamicShapeVariable(VariableTracker):
    """
    Represents a symbolic size, e.g., as returned by tensor.size(0)
    """

    @classmethod
    def create(cls, tx, proxy, dyn_shape, **options):
        if "example_value" in proxy.node.meta:
            assert proxy.node.meta["example_value"] == dyn_shape
        if dyn_shape is None:
            dyn_shape = get_fake_value(proxy.node, tx)
        proxy.node.meta["example_value"] = dyn_shape
        return DynamicShapeVariable(proxy, dyn_shape, **options)

    def __init__(self, proxy, dyn_shape, **kwargs):
        super(DynamicShapeVariable, self).__init__(**kwargs)
        self.proxy = proxy
        self.dyn_shape = dyn_shape

    def python_type(self):
        return type(self.dyn_shape)

    def unpack_var_sequence(self, tx):
        super(DynamicShapeVariable, self).unpack_var_sequence(tx)

    def as_proxy(self):
        return self.proxy

    def evaluate_expr(self, output_graph):
        if isinstance(self.dyn_shape, bool):
            # Bool and 0/1 case fallthrough
            return self.dyn_shape
        return output_graph.shape_env.evaluate_expr(self.dyn_shape.get_pyobj().expr)

    def call_method(
        self,
        tx,
        name,
        args: "List[VariableTracker]",
        kwargs: "Dict[str, VariableTracker]",
    ) -> "VariableTracker":
        options = VariableTracker.propagate(self, args, kwargs.values())

        return wrap_fx_proxy(
            tx,
            tx.output.create_proxy(
                "call_method",
                name,
                *proxy_args_kwargs([self] + list(args), kwargs),
                current_tx=tx,
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
        super(TensorWithTFOverrideVariable, self).__init__(**kwargs)
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
        with torch._C.DisableTorchFunction():
            return tx.inline_user_function_return(tf_func_var, tf_args, {})


class UnspecializedNumpyVariable(TensorVariable):
    """
    This is a 1-element tensor represents unspecialized numpy float/int.
    """

    def __init__(self, proxy: torch.fx.Proxy, **kwargs):
        raw_value = kwargs.pop("raw_value", None)
        super(UnspecializedNumpyVariable, self).__init__(proxy, **kwargs)
        self.raw_value = raw_value

    @classmethod
    def from_tensor_variable(cls, tensor_variable, raw_value):
        # Convert a `TensorVariable` instance into an `UnspecializedNumpyVariable` instance.
        return UnspecializedNumpyVariable(
            **dict(tensor_variable.__dict__), raw_value=raw_value
        )

    def as_specialized(self, tx):
        for graph_arg in tx.output.graphargs:
            if graph_arg.source is self.source:
                graph_arg.erase()

        for g in self.guards:
            if g.is_volatile:
                g.create_fn = GuardBuilder.CONSTANT_MATCH

        return ConstantVariable(value=self.raw_value, guards=self.guards)


class UnspecializedPythonVariable(TensorVariable):
    """
    This is a 1-element tensor represents unspecialized python float/int.
    """

    def __init__(self, proxy: torch.fx.Proxy, **kwargs):
        raw_value = kwargs.pop("raw_value", None)
        need_unwrap = kwargs.pop("need_unwrap", True)
        super(UnspecializedPythonVariable, self).__init__(proxy, **kwargs)
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
        super(FakeItemVariable, self).__init__(proxy, **kwargs)
        self.need_unwrap = need_unwrap

    @classmethod
    def from_tensor_variable(cls, tensor_variable):
        return FakeItemVariable(**dict(tensor_variable.__dict__))
