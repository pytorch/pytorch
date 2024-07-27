# mypy: allow-untyped-defs

import torch
from . import ir
from .virtualized import ops


# In a separate file to prevent circular dependency issues
def create_epilogue_with_attr(input_buffer, attr, **kwargs):
    input_loader = input_buffer.make_loader()
    dtype = input_buffer.get_dtype()
    if attr == "relu":

        def inner_fn(index):
            input = input_loader(index)
            zero = ops.constant(0, dtype)
            return ops.maximum(input, zero)

    elif attr == "gelu":
        assert "algorithm" in kwargs
        if kwargs["algorithm"] == "none":

            def inner_fn(index):
                input = input_loader(index)
                if dtype != torch.float:
                    input = ops.to_dtype(input, torch.float)
                half = ops.constant(0.5, torch.float)
                one = ops.constant(1.0, torch.float)
                const = ops.constant(0.7071067811865476, torch.float)
                result = input * half * (ops.erf(input * const) + one)
                if dtype != torch.float:
                    result = ops.to_dtype(result, dtype)
                return result

        else:
            assert kwargs["algorithm"] == "tanh"

            def inner_fn(index):
                input = input_loader(index)
                if dtype != torch.float:
                    input = ops.to_dtype(input, torch.float)
                half = ops.constant(0.5, torch.float)
                one = ops.constant(1.0, torch.float)
                const1 = ops.constant(0.7978845608028654, torch.float)
                const2 = ops.constant(0.044715, torch.float)
                result = (
                    half
                    * input
                    * (
                        one
                        + ops.tanh(const1 * (input + const2 * input * input * input))
                    )
                )
                if dtype != torch.float:
                    result = ops.to_dtype(result, dtype)
                return result

    elif attr == "swish":

        def inner_fn(index):
            input = input_loader(index)
            result = input * ops.sigmoid(input)
            return result

    elif attr == "sigmoid":

        def inner_fn(index):
            return ops.sigmoid(input_loader(index))

    elif attr == "tanh":

        def inner_fn(index):
            return ops.tanh(input_loader(index))

    elif attr == "hardswish" or attr == "hardsigmoid":

        def hardsigmoid_float(input):
            zero = ops.constant(0, torch.float)
            six = ops.constant(6, torch.float)
            three = ops.constant(3, torch.float)
            one_over_six = ops.constant(0.16666666666666666, torch.float)
            max = ops.maximum(input + three, zero)
            min = ops.minimum(max, six)
            return min * one_over_six

        def inner_fn(index):
            input = input_loader(index)
            if dtype != torch.float:
                input = ops.to_dtype(input, torch.float)
            result = hardsigmoid_float(input)
            if attr == "hardswish":
                result = input * result
            if dtype != torch.float:
                result = ops.to_dtype(result, dtype)
            return result

    elif attr == "leaky_relu":
        assert "scalars" in kwargs
        assert len(kwargs["scalars"]) == 1
        negative_slope = kwargs["scalars"][0]

        def inner_fn(index):
            input = input_loader(index)
            if dtype != torch.float:
                input = ops.to_dtype(input, torch.float)
            zero = ops.constant(0, torch.float)
            result = ops.where(
                input > zero, input, input * ops.constant(negative_slope, torch.float)
            )
            if dtype != torch.float:
                result = ops.to_dtype(result, dtype)
            return result

    elif attr == "hardtanh":
        assert "scalars" in kwargs
        assert len(kwargs["scalars"]) == 2
        min_value = kwargs["scalars"][0]
        max_value = kwargs["scalars"][1]

        def inner_fn(index):
            input = input_loader(index)
            if dtype != torch.float:
                input = ops.to_dtype(input, torch.float)
            result = ops.minimum(
                ops.maximum(input, ops.constant(min_value, torch.float)),
                ops.constant(max_value, torch.float),
            )
            if dtype != torch.float:
                result = ops.to_dtype(result, dtype)
            return result

    elif attr in ["add", "sub", "mul"]:
        assert "other" in kwargs
        other = kwargs["other"]
        num_input_dims = len(input_buffer.get_size())
        num_other_dims = len(other.get_size())
        dims_diff = num_input_dims - num_other_dims
        other_loader = other.make_loader()

        def inner_fn(index):
            op = getattr(ops, attr)
            if dims_diff != 0:
                return op(input_loader(index), other_loader(index[dims_diff:]))
            else:
                return op(input_loader(index), other_loader(index))

    elif attr == "bias_add":
        assert "other" in kwargs
        assert "beta" in kwargs
        assert "dtype" in kwargs
        beta = kwargs["beta"]
        other = kwargs["other"]
        dtype = kwargs["dtype"]
        bias_loader = other.make_loader()

        def inner_fn(index):
            bias = bias_loader(index)
            input = input_loader(index)
            if beta != 1:
                result = ops.constant(beta, torch.float) * bias + input
            else:
                result = bias + input
            return result

    else:
        raise ValueError(f"Unsupported epilogue attribute: {attr}")
    return ir.Pointwise(
        device=input_buffer.get_device(),
        dtype=dtype,
        inner_fn=inner_fn,
        ranges=input_buffer.get_size(),
    )
