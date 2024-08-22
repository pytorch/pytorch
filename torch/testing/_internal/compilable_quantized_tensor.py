"""
Following is a example for a simple dtype implemented with tensor subclass
it shows
    * the basic structure of a new dtype tensor subclass (__new__, __init__, __tensor_flatten__, __tensor_unflatten__)
    * two types of dispatch that people can overwrite (__torch_function__, __torch_dispatch__)
    * how to abstract away packing format with layout
    * how the tensor subclass composes with torch.compile to get speedup
"""


import functools
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import torch
from torch.utils._python_dispatch import return_and_correct_aliasing


aten = torch.ops.aten


@dataclass(frozen=True)
class LayoutType:
    pass


@dataclass(frozen=True)
class PlainLayoutType(LayoutType):
    pass


###############################
# Base Layout Tensor Subclass #
###############################
class MyDTypeLayout(torch.Tensor):
    """
    Base class for the layout tensor for `MyDTypeTensor`
    """

    def __init__(self, int_data, scale, layout_type):  # type: ignore[no-untyped-def]
        self.int_data = int_data
        self.scale = scale
        self.layout_type = layout_type

    # get the original unpacked Tensors
    def get_plain(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.int_data, self.scale

    def get_layout_type(self) -> LayoutType:
        return self.layout_type

    @classmethod
    def from_plain(
        cls,
        int_data: torch.Tensor,
        scale: torch.Tensor,
        layout_type: LayoutType,
    ) -> torch.Tensor:
        """Construct a layout tensor from plain tensors and a layout_type, which main contain
        extra metadata for packing etc.
        """
        return cls(int_data, scale, layout_type)

    def __repr__(self):  # type: ignore[override, no-untyped-def]
        int_data, scale = self.get_plain()
        layout_type = self.get_layout_type()
        return f"{self.__class__.__name__}(int_data={int_data}, scale={scale}, layout_type={layout_type})"

    __torch_function__ = torch._C._disabled_torch_function_impl


##############################
# Tensor Subclass Definition #
##############################


class MyDTypeTensor(torch.Tensor):
    """We need to define __new__ for constructing a new tensor subclass instance and __init__ for initialize
    the instance. There is no requirement on what the argument list should look like here, only requirement is
    that `__new__` must return a Tensor instance with `torch.Tensor._make_wrapper_subclass(cls, shape, ...)` call
    """

    @staticmethod
    def __new__(  # type: ignore[no-untyped-def]
        cls,
        layout_tensor: MyDTypeLayout,
        shape: torch.Size,
        dtype: Optional[torch.dtype] = None,
    ):
        kwargs = {}
        kwargs["device"] = layout_tensor.device
        kwargs["layout"] = (  # type: ignore[assignment]
            kwargs.get("layout")  # type: ignore[assignment]
            if kwargs.get("layout", False)
            else layout_tensor.layout
        )
        kwargs["dtype"] = dtype  # type: ignore[assignment]
        kwargs["requires_grad"] = False  # type: ignore[assignment]
        return torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)  # type: ignore[attr-defined]

    def __init__(
        self,
        layout_tensor: MyDTypeLayout,
        shape: torch.Size,
        dtype: Optional[torch.dtype] = None,
    ):
        self.layout_tensor = layout_tensor

    """__tensor_flatten__ and __tensor_unflatten__ are used to desugar the tensor into native Tensors/attributes and
    reconstruct the tensor subclass instance from the desugared tensor and attributes, these are required to define
    a Tensor subclass for torch.compile support
    """

    def __tensor_flatten__(self):  # type: ignore[no-untyped-def]
        """
        Given the class, returns the fields of the class as two lists
        The first one contains any tensor fields such as int_data and scale as keys to a dictionary
        The second one contains all other non tensor type fields as values of a list
        """
        return ["layout_tensor"], [self.shape, self.dtype]

    @classmethod
    def __tensor_unflatten__(  # type: ignore[no-untyped-def]
        cls, tensor_data_dict, tensor_attributes, outer_size, outer_stride
    ):
        """
        Given the flattened data from above, returns a class instance
        tensor_data_dict contains the tensor fields of the class as a dictionary
        tensor_attributes contains all other non tensor type fields
        """
        layout_tensor = tensor_data_dict["layout_tensor"]
        shape, dtype = tensor_attributes
        return cls(
            layout_tensor,
            shape if outer_size is None else outer_size,
            dtype=dtype,
        )

    """classmethod that converts from a floating point Tensor (fp32/fp16/bf16) to the current dtype
    """

    @classmethod
    def from_float(
        cls,
        input_float: torch.Tensor,
        layout_type: LayoutType = PlainLayoutType(),
    ) -> torch.Tensor:
        dtype = torch.int16
        scale = input_float.abs().max() / 255
        int_data = (input_float / scale).to(torch.int8)
        layout_tensor = PlainMyDTypeLayout(int_data, scale, PlainLayoutType())
        return cls(layout_tensor, input_float.shape)

    """[Optional] We can overwrite layout property of the Tensor to represent different packing formats
    """

    @property
    def layout_type(self) -> LayoutType:
        return self.layout_tensor.get_layout_type()

    def dequantize(self, output_dtype=None):  # type: ignore[no-untyped-def]
        """We can define a dequantize method to convert the quantized tensor to a floating point tensor"""
        if output_dtype is None:
            output_dtype = torch.get_default_dtype()
        int_data, scale = self.layout_tensor.get_plain()
        return int_data.to(output_dtype) * scale

    def __repr__(self):  # type: ignore[no-untyped-def]
        return (
            f"{self.__class__.__name__}(data={self.dequantize()}, shape={self.shape}, "
            f"device={self.device}, dtype={self.dtype}, requires_grad={self.requires_grad})"
        )

    def _apply_fn_to_data(self, fn):  # type: ignore[no-untyped-def]
        """
        Used for implementing aten ops by applying them only to the relevant tensor atributes
        In this case we only want to call things like to() or view() on the layout tensor
        """
        return self.__class__(
            fn(self.layout_tensor),
            self.shape,
            self.dtype,
        )

    @classmethod
    def implements(cls, aten_ops_or_torch_fns):  # type: ignore[no-untyped-def]
        if not hasattr(cls, "_DISPATCH_TABLE"):
            cls._DISPATCH_TABLE = {}  # type: ignore[attr-defined]

        if not isinstance(aten_ops_or_torch_fns, (list, tuple)):
            aten_ops_or_torch_fns = [aten_ops_or_torch_fns]

        def decorator(func):  # type: ignore[no-untyped-def]
            for op in aten_ops_or_torch_fns:

                @functools.wraps(op)
                def wrapper(f, types, args, kwargs):  # type: ignore[no-untyped-def]
                    return func(f, types, args, kwargs)

                cls._DISPATCH_TABLE[op] = wrapper  # type: ignore[attr-defined]

            return func

        return decorator

    """There are two entry points that we can modify the behavior of a pytorch op: torch_function and torch_dispatch:

    __torch_function__: will be called whenever a torch level function is called on the Tensor object,
    for example: torch.nn.functional.linear, tensor.detach, tensor.reshape, tensor.t etc.

    __torch_dispatch__: will be called in the C++ dispatcher, when an aten operator is called on the Tensor object,
    for example: aten.mm, aten.addmm, aten.detach.default, aten.t.default etc.

    We have some helper functions that can dispatch to the functions registered with MyDTypeTensor.implements,
    but if the default implementation does not work for your use case, please feel free to customize it
    """

    @classmethod
    def __torch_function__(cls, func, types, args, kwargs=None):  # type: ignore[no-untyped-def]
        kwargs = {} if kwargs is None else kwargs

        if hasattr(cls, "_DISPATCH_TABLE") and func in cls._DISPATCH_TABLE:
            return cls._DISPATCH_TABLE[func](func, types, args, kwargs)

        with torch._C.DisableTorchFunctionSubclass():
            return func(*args, **kwargs)

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):  # type: ignore[no-untyped-def]
        kwargs = {} if kwargs is None else kwargs

        if hasattr(cls, "_DISPATCH_TABLE") and func in cls._DISPATCH_TABLE:
            return cls._DISPATCH_TABLE[func](func, types, args, kwargs)

        raise NotImplementedError(
            f"{cls.__name__} dispatch: attempting to run unimplemented operator/function: {func}"
        )


######################################################
# LayoutType and Layout Tensor Subclass Registration #
######################################################


class PlainMyDTypeLayout(MyDTypeLayout):
    def __new__(  # type: ignore[no-untyped-def]
        cls,
        int_data: torch.Tensor,
        scale: torch.Tensor,
        layout_type: LayoutType,
    ):
        kwargs = {}
        kwargs["device"] = int_data.device
        kwargs["layout"] = kwargs.get("layout") if kwargs.get("layout", False) else int_data.layout  # type: ignore[assignment]
        kwargs["dtype"] = int_data.dtype  # type: ignore[assignment]
        kwargs["requires_grad"] = False  # type: ignore[assignment]
        shape = int_data.shape
        return torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)  # type: ignore[attr-defined]

    def __init__(
        self,
        int_data: torch.Tensor,
        scale: torch.Tensor,
        layout_type: LayoutType,
    ):
        self.int_data = int_data
        self.scale = scale
        self.layout_type = layout_type

    def __tensor_flatten__(self):  # type: ignore[no-untyped-def]
        return ["int_data", "scale"], [self.layout_type]

    @classmethod
    def __tensor_unflatten__(cls, tensor_data_dict, tensor_attributes, outer_size, outer_stride):  # type: ignore[no-untyped-def]
        int_data, scale = tensor_data_dict["int_data"], tensor_data_dict["scale"]
        (layout_type,) = tensor_attributes
        return cls(int_data, scale, layout_type)

    @classmethod
    def from_plain(
        cls,
        int_data: torch.Tensor,
        scale: torch.Tensor,
        layout_type: LayoutType,
    ) -> torch.Tensor:
        """Construct a layout tensor from plain tensors and a layout_type, which main contain
        extra metadata for packing etc.
        """
        assert isinstance(layout_type, PlainLayoutType)
        return cls(int_data, scale, layout_type)

    def _apply_fn_to_data(self, fn: Callable) -> torch.Tensor:
        return self.__class__(
            fn(self.int_data),
            fn(self.scale),
            self.layout_type,
        )

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):  # type: ignore[no-untyped-def]
        kwargs = {} if kwargs is None else kwargs

        if func is aten.detach.default:
            return return_and_correct_aliasing(
                func, args, kwargs, args[0]._apply_fn_to_data(torch.detach)
            )

        raise NotImplementedError(
            f"MyDTypeLayout dispatch: attempting to run {func}, this is not supported"
        )


#####################################################
# torch functional and aten operator implementation #
#####################################################

implements = MyDTypeTensor.implements


def _quantized_linear_op(
    input_tensor: torch.Tensor, weight_tensor: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    if isinstance(input_tensor, MyDTypeTensor):
        input_tensor = input_tensor.dequantize()
    if isinstance(weight_tensor, MyDTypeTensor):
        weight_tensor = weight_tensor.dequantize()
    return torch.nn.functional.linear(input_tensor, weight_tensor, bias)


@implements(torch.nn.functional.linear)
def _(func, types, args, kwargs):  # type: ignore[no-untyped-def]
    input_tensor, weight_tensor, bias = (
        args[0],
        args[1],
        args[2] if len(args) > 2 else None,
    )
    # using try/except here so that we can have a general fallback when input_tensor/weight_tensor
    # is not picked up by any of the dispatch paths in `_quantized_linear_op`, this allows us to
    # make the branches easier to understand in `_quantized_linear_op`
    try:
        return _quantized_linear_op(input_tensor, weight_tensor, bias)
    except NotImplementedError:
        if isinstance(input_tensor, MyDTypeTensor):
            input_tensor = input_tensor.dequantize()
        if isinstance(weight_tensor, MyDTypeTensor):
            weight_tensor = weight_tensor.dequantize()
        return torch.nn.functional.linear(input_tensor, weight_tensor, bias)


@implements(aten.detach.default)
def _(func, types, args, kwargs):  # type: ignore[no-untyped-def]
    # `return_and_correct_aliasing` should be used by wrapper tensor ``__torch_dispatch__`` subclasses that would like to
    # work with torch.compile. It ensures that the subclass properly implements the aliasing behavior of every op,
    # which is needed for correctness in AOTAutograd.

    # `_apply_fn_to_data` just applies the function to the tensor data in `args[0]`, `args[0]` is a tensor subclass
    # of `my_dtype`
    return return_and_correct_aliasing(
        func, args, kwargs, args[0]._apply_fn_to_data(torch.detach)
    )


#####################
# Factory functions #
#####################
to_my_dtype = MyDTypeTensor.from_float
