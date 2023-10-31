"""
Testing out accuracy-only implementation of SmoothQuant
(https://arxiv.org/pdf/2211.10438.pdf)
Note: this is an application of input-weight equalization, with the addition that the
multiplication by scale is fused into the preceding layer, specifically for relevant
parts of transformer blocks.
"""

import torch
import torch.nn.functional as F
from torch.ao.quantization.experimental._gpu_quantization import quant_api

# from fairscale.nn.model_parallel.layers import (
#     ParallelEmbedding,
#     RowParallelLinear,
#     ColumnParallelLinear,
# )
# from fairscale.nn.model_parallel.mappings import (
#     copy_to_model_parallel_region,
#     gather_from_model_parallel_region,
#     reduce_from_model_parallel_region,
#     scatter_to_model_parallel_region,
# )
# from fairscale.nn.model_parallel.initialize import get_model_parallel_group

from .quant_primitives import (
    dynamically_quantize_per_channel,
    quant_int8_dynamic_per_token_linear,
)

__all__ = [
    "get_scale",
    "SmoothFakeDynQuantMixin",
    "SmoothFakeDynamicallyQuantizedLinear",
    "swap_linear_with_smooth_fq_linear",
    "smooth_fq_linear_to_inference",
    "set_smooth_fq_attribute",
]


# def _max_reduce(input_: torch.Tensor) -> torch.Tensor:
#     group = get_model_parallel_group()
#     if torch.distributed.get_world_size(group=group) == 1:
#         return input_
#     # All-reduce.
#     torch.distributed.all_reduce(input_, op=dist.ReduceOp.MAX, group=group)
#     return input_


def get_scale(X_absmax, W_absmax, alpha=0.5):
    """
    Calculate the scale based on abs(max(X)), abs(max(W)) and alpha
    If X is of dimension `b*n*k` and W is dimension `k*m`, the returned
    scale is of dimension `k`.
    Note: X_absmax is calculated outside of this function because we
    need to keep a running version of it during calibration. W_absmax
    is calculated outside of this function for consistency with X_absmax.
    """
    X_pow = torch.pow(X_absmax, alpha)
    W_pow = torch.pow(W_absmax, 1.0 - alpha)
    div = X_pow / W_pow
    return div.reshape(-1)


class SmoothFakeDynQuantMixin(torch.nn.Module):
    def init_smoothquant_variables(self, alpha):
        self.calibrating = True
        self.x_running_abs_max = None
        self.register_buffer("smooth_scale", None)
        self.alpha = alpha
        # debug only
        self.debug_skip_scaling = False
        # self.debug_skip_scaling = True

        # Currently torch._int_mm cuBLAS underlying kernel does not work with
        # non-contiguous weight. However, torch.compil'ing through
        # torch._int_mm leads to triton code which is ~2x faster if the weight
        # is transposed. So, for now we have a debug flag to toggle whether
        # we store the quantized weight transposed, so that we can get correct
        # numerics both in eager mode and after torch.compile.
        # The default is True for cuBLAS / eager mode, set to False for
        # torch.compile.
        # self.store_w_int_repr_t = True
        self.store_w_int_repr_t = False

    def update_x_running_abs_max(self, X):
        # update the running max of incoming activations
        all_dims_except_last = tuple(range(len(X.shape) - 1))
        cur_abs_max = torch.amax(torch.abs(X), dim=all_dims_except_last)
        if self.x_running_abs_max is None:
            self.x_running_abs_max = cur_abs_max
        else:
            self.x_running_abs_max = torch.max(cur_abs_max, self.x_running_abs_max)

    def get_scaled_quantized_w(self):
        # inference
        assert (
            self.smooth_scale is not None
        ), "self.smooth_scale is None, did you turn on inference?"
        W = self.weight

        # scale weight
        # in the future, this can be done ahead of time instead of
        # during inference
        if not self.debug_skip_scaling:
            # TODO(future): do below in `to_inference` instead of here
            W = torch.matmul(
                torch.diag(self.smooth_scale), W.transpose(0, 1)
            ).transpose(0, 1)

        # fake quantize input and weight, and then do matmul in fp32/fp16
        # in the future, this should be replaced with quantized kernels which
        # work on NVIDIA GPUs (such as protoquant's implementation)
        W_dq_dtype = W.dtype
        W_int_repr, W_scales, W_zps = dynamically_quantize_per_channel(
            W, -128, 127, torch.int8
        )
        W_int_repr = W_int_repr.contiguous()
        return W_int_repr, W_scales, W_zps

    def to_inference(self):
        raise NotImplementedError()

    def fold_weight(self):
        # note: _W_zps are zeroes and they are ignored
        # TODO(future PR): set up serialization for this
        W_int_repr, self.W_scales, _W_zps = self.get_scaled_quantized_w()
        # need to store transposed weights to make eager mode matmul
        # op work in cuBlas, or non-transposed to make it fast in torch.compile
        if self.store_w_int_repr_t:
            self.register_buffer("W_int_repr", W_int_repr.transpose(0, 1).contiguous())
        else:
            self.register_buffer("W_int_repr", W_int_repr.contiguous())
        del self.weight

    def set_debug_x_absmax(self):
        """
        Sets `self.x_running_abs_max` to a value which will lead to smooth scale
        of all ones if `alpha=0.5`, to enable performance benchmarking without
        calibration.
        """
        raise NotImplementedError()


class SmoothFakeDynamicallyQuantizedLinear(SmoothFakeDynQuantMixin, torch.nn.Linear):
    """
    This is a replacement for `torch.nn.Linear` which implements fake quantization
    based on Smoothquant scaling.
    """

    def __init__(self, *args, **kwargs):
        alpha = kwargs.pop("alpha")
        super().__init__(*args, **kwargs)
        self.init_smoothquant_variables(alpha)

    def forward(self, X):
        if self.calibrating:
            self.update_x_running_abs_max(X)
            Y = F.linear(X, self.weight, self.bias)
        else:
            if not self.debug_skip_scaling:
                # TODO(future): fuse this into previous layer (LayerNorm,
                # RMSNorm, etc) where appropriate
                X = X / self.smooth_scale
            W_int_repr_t = (
                self.W_int_repr if self.store_w_int_repr_t else self.W_int_repr.t()
            )
            Y = quant_int8_dynamic_per_token_linear(
                X, W_int_repr_t, self.W_scales, self.bias, X.dtype
            )
        return Y

    @classmethod
    def from_float(cls, mod, alpha=0.5):
        """
        Converts a `mod` of class `torch.nn.Linear` to the smooth fake quantized
        version of it.  Note: requires calibration.
        """
        # create the new module with a toy size to ensure initialization is fast
        fake_in_features, fake_out_features = 8, 8
        new_mod = cls(
            fake_in_features, fake_out_features, bias=mod.bias is not None, alpha=alpha
        )
        new_mod.in_features = mod.in_features
        new_mod.out_features = mod.out_features
        new_mod.weight = mod.weight
        new_mod.bias = mod.bias
        # TODO: test when creation is on cuda
        device_to_use = next(mod.parameters()).device
        new_mod.to(device_to_use)
        return new_mod

    def to_inference(self):
        """
        Calculates the smoothquant scale based on calibration
        in preparation for inference
        """
        assert self.x_running_abs_max is not None, "no calibration data found"
        self.calibrating = False
        self.smooth_scale = get_scale(
            self.x_running_abs_max,
            torch.max(torch.abs(self.weight.transpose(0, 1)), dim=1).values,
            alpha=self.alpha,
        )
        self.fold_weight()

    def set_debug_x_absmax(self):
        w_absmax = torch.max(torch.abs(self.weight.transpose(0, 1)), dim=1).values
        self.x_running_abs_max = w_absmax

#
# utils to use the smooth linear on real models
#

source_cls_to_target_cls = {
    torch.nn.Linear: SmoothFakeDynamicallyQuantizedLinear,
}


def swap_linear_with_smooth_fq_linear(
    model, skip_fqn_list=None, cur_fqn="", alpha=0.5
) -> None:
    name_to_child = dict(model.named_children())
    for name, child in name_to_child.items():
        if cur_fqn == "":
            new_fqn = name
        else:
            new_fqn = f"{cur_fqn}.{name}"
        if ((skip_fqn_list is None) or (new_fqn not in skip_fqn_list)) and isinstance(
            child, tuple(source_cls_to_target_cls.keys())
        ):
            target_cls = source_cls_to_target_cls[type(child)]
            new_child = target_cls.from_float(child, alpha=alpha)
            setattr(model, name, new_child)
        else:
            swap_linear_with_smooth_fq_linear(child, skip_fqn_list, new_fqn, alpha)


# code moved, avoid breaking callsites
# TODO clean this up
replace_with_custom_fn_if_matches_filter = (
    quant_api.replace_with_custom_fn_if_matches_filter
)


def smooth_fq_linear_to_inference(model, debug_skip_calibration=False) -> None:
    for _, mod in model.named_modules():
        if isinstance(mod, tuple(source_cls_to_target_cls.values())):
            if debug_skip_calibration:
                mod.set_debug_x_absmax()
            mod.to_inference()


# useful for quickly toggling smoothquant debug settings on all smoothquant
# modules in a model
def set_smooth_fq_attribute(model, attribute_name, new_attribute_val):
    for _, mod in model.named_modules():
        if isinstance(mod, tuple(source_cls_to_target_cls.values())):
            if hasattr(mod, attribute_name):
                setattr(mod, attribute_name, new_attribute_val)
