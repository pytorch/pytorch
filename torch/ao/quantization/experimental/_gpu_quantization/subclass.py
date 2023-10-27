import torch
from torch.ao.quantization.experimental._gpu_quantization.quant_primitives import (
    dequantize_per_channel,
    dynamically_quantize_per_channel,
    quant_int8_dynamic_per_token_linear,
)
from torch.utils._python_dispatch import return_and_correct_aliasing

__all__ = ["DynamicallyQuantizedLinearWeight"]


class DynamicallyQuantizedLinearWeight(torch.Tensor):
    @staticmethod
    def __new__(cls, input_data, q_scales, transposed=False, **kwargs):
        # input data is assumed to be input so that q_axis is the 1th axis
        # also assumes input is non contiguous
        kwargs["device"] = input_data.device
        kwargs["dtype"] = kwargs.get("dtype", torch.int8)
        if input_data is not None:
            kwargs["dtype"] = input_data.dtype
        size = input_data.shape[::-1] if transposed else input_data.shape
        kwargs["layout"] = (
            kwargs.get("layout") if kwargs.get("layout", False) else input_data.layout
        )
        return torch.Tensor._make_wrapper_subclass(cls, size, **kwargs)  # type: ignore[attr-defined]

    def __init__(self, input_data, q_scales, transposed=False):
        self.transposed = transposed
        self.int_data = input_data
        self.q_scales = q_scales

    def __repr__(self):
        return f"DynamicallyQuantizedLinearWeight(shape={self.shape}, data={self.dequantize()})"

    def dequantize(self, dtype=None):
        out = dequantize_per_channel(
            self.int_data.t(), self.q_scales, 0, self.dtype if dtype is None else dtype
        )
        return out if self.transposed else out.t()  # already transposedd for dequantize

    def int_repr(self):
        return self.int_data.t() if self.transposed else self.int_data

    def _detach(self):
        return DynamicallyQuantizedLinearWeight(
            self.int_data, self.q_scales, transposed=self.transposed
        )

    def _transposed(self):
        return DynamicallyQuantizedLinearWeight(
            self.int_data, self.q_scales, transposed=(not self.transposed)
        )

    def __tensor_flatten__(self):
        return ["int_data", "q_scales"], self.transposed

    @staticmethod
    def __tensor_unflatten__(tensor_data, transposed):
        int_data, q_scales = tensor_data["int_data"], tensor_data["q_scales"]
        return DynamicallyQuantizedLinearWeight(
            int_data, q_scales, transposed=transposed
        )

    __torch_function__ = torch._C._disabled_torch_function_impl

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        # two scenarios where we currently fall back to vanilla mm:
        # 1 - when tensor is on CPU: we are missing qmm for CPU, but we should have a CPU implementation
        #     for consistency and to allow people to test
        # 2 - we need to define what happens when we're given non-floats - quantizing long to int8 is probs craxy
        if (
            func in [torch.ops.aten.mm.default, torch.ops.aten.addmm.default]
            and args[0].is_floating_point()
            and args[0].is_cuda
        ):
            if func == torch.ops.aten.addmm.default:
                assert (
                    args[1].shape[-1] == args[2].shape[0]
                ), f"need mat1 shape: {args[1].shape} final dim to match mat2 shape: {args[2].shape} first dim "
                mat1, mat2, scales, bias = (
                    args[1],
                    args[2].int_data,
                    args[2].q_scales,
                    args[0],
                )
            else:
                assert (
                    args[0].shape[-1] == args[1].shape[0]
                ), f"need mat1 shape: {args[0].shape} final dim to match mat2 shape: {args[1].shape} first dim "
                mat1, mat2, scales, bias = (
                    args[0],
                    args[1].int_data,
                    args[1].q_scales,
                    None,
                )
            return quant_int8_dynamic_per_token_linear(
                mat1, mat2, scales, bias, mat1.dtype
            )

        if func is torch.ops.aten.detach.default:
            return return_and_correct_aliasing(func, args, kwargs, args[0]._detach())

        if func is torch.ops.aten.t.default:
            return return_and_correct_aliasing(
                func, args, kwargs, args[0]._transposed()
            )
        breakpoint()
        return NotImplemented

    @classmethod
    def from_float(cls, input_float, qmin=-128, qmax=127, dtype=torch.int8):
        w_int_repr, w_scales, _ = dynamically_quantize_per_channel(
            input_float, qmin, qmax, dtype
        )
        # always store with quantized axis in dim=1 for fast matmul
        return cls(w_int_repr.contiguous().t(), w_scales, transposed=True)
