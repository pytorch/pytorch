import torch
import torch.nn as nn
from torch.nn import functional as F
from functorch.compile import memory_efficient_fusion
from torch.testing._internal.common_utils import TestCase, run_tests
import inspect
from typing import Callable
import unittest

HAS_CUDA = torch.cuda.is_available()


def _num_args(fn: Callable):
    return len(inspect.signature(fn).parameters)


def gelu_bias(bias, y):
    x = bias + y
    return x * 0.5 * (1.0 + torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x)))


def swish(x):
    return x * torch.sigmoid(x)


def mish(x):
    return x.mul(torch.tanh(F.softplus(x)))


def hard_sigmoid(x):
    return (x + 3.0).clamp(min=0.0, max=6.0).div(6.0)


def hard_swish(x):
    return x * (x + 3.0).clamp(min=0.0, max=6.0).div(6.0)


def hard_mish(x):
    return 0.5 * x * (x + 2.0).clamp(min=0.0, max=2.0)


# todo: convert these into tests
# def group_std(x, groups: int = 32, eps: float = 1e-5, flatten: bool = False):
#     B, C, H, W = x.shape
#     x_dtype = x.dtype
#     if flatten:
#         x = x.reshape(B, groups, -1)  # FIXME simpler shape causing TPU / XLA issues
#         std = x.float().var(dim=2, unbiased=False, keepdim=True).add(eps).sqrt().to(x_dtype)
#     else:
#         x = x.reshape(B, groups, C // groups, H, W)
#         std = x.float().var(dim=(2, 3, 4), unbiased=False, keepdim=True).add(eps).sqrt().to(x_dtype)
#     return std.expand(x.shape).reshape(B, C, H, W)

# class EvoNorm2dS0(nn.Module):
#     def __init__(self, num_features, groups=32, group_size=None, apply_act=True, eps=1e-5, **_):
#         super().__init__()
#         self.apply_act = apply_act  # apply activation (non-linearity)
#         if group_size:
#             assert num_features % group_size == 0
#             self.groups = num_features // group_size
#         else:
#             self.groups = groups
#         self.eps = eps
#         self.weight = nn.Parameter(torch.ones(num_features))
#         self.bias = nn.Parameter(torch.zeros(num_features))
#         self.v = nn.Parameter(torch.ones(num_features)) if apply_act else None
#         self.reset_parameters()

#     def reset_parameters(self):
#         nn.init.ones_(self.weight)
#         nn.init.zeros_(self.bias)
#         if self.v is not None:
#             nn.init.ones_(self.v)

#     def forward(self, x):
#         x_dtype = x.dtype
#         v_shape = (1, -1, 1, 1)
#         if self.v is not None:
#             v = self.v.view(v_shape).to(dtype=x_dtype)
#             x = x * (x * v).sigmoid() / group_std(x, self.groups, self.eps)
#         return x * self.weight.view(v_shape).to(dtype=x_dtype) + self.bias.view(v_shape).to(dtype=x_dtype)


# device = "cuda"
# dtype = torch.float

# evo_norm = EvoNorm2dS0(2048)
# evo_norm_inp = [(128, 2048, 8, 8)]




def run_and_compare_activation(self, fn, inps):
    with torch.jit.fuser("fuser1"):
        device = "cuda"
        dtype = torch.float
        if isinstance(fn, nn.Module):
            fn = fn.to(device=device, dtype=dtype)

        ref_args = [torch.randn(shape, device=device, dtype=dtype, requires_grad=True) for shape in inps]
        res_args = [i.clone().detach().requires_grad_(True) for i in ref_args]

        ref = fn(*ref_args)
        ref.sum().backward()

        mem_optimized_fn = memory_efficient_fusion(fn)
        for _ in range(5):
            for i in res_args:
                i.grad = None
            res = mem_optimized_fn(*res_args)
            res.sum().backward()


        self.assertEqual(ref, res)
        for ref_arg, res_arg in zip(ref_args, res_args):
            self.assertEqual(ref_arg.grad, res_arg.grad)



@unittest.skipIf(not torch.cuda.is_available(), "CUDA is unavailable")
class TestMemoryEfficientOpAuthoring(TestCase):
    def test_gelu_bias(self):
        run_and_compare_activation(self, gelu_bias, [(1024,), (1024,)])

    def test_mish(self):
        run_and_compare_activation(self, mish, [(1024,)])

    def test_swish(self):
        run_and_compare_activation(self, swish, [(1024,)])

    def test_hard_sigmoid(self):
        run_and_compare_activation(self, hard_sigmoid, [(1024,)])

    def test_hard_swish(self):
        run_and_compare_activation(self, hard_swish, [(1024,)])

    def test_layer_norm(self):
        def layer_norm(x, weight, bias):
            dim = -1
            eps = 1e-5
            mean = torch.mean(x, dim, keepdim=True)
            centered = x - mean
            var = torch.sum(centered * centered, dim, keepdim=True )/ x.size(-1)
            rvar = 1./torch.sqrt(var+eps)
            normed = (x-mean) * rvar
            return normed * weight + bias

        bs = 10
        ln_size = 16
        layer_norm_inps = [(bs, ln_size), (ln_size,), (ln_size,)]
        run_and_compare_activation(self, layer_norm, layer_norm_inps)


    def test_rmsnorm(self):
        class T5LayerNorm(nn.Module):
            def __init__(self, hidden_size, eps=1e-6):
                """
                Construct a layernorm module in the T5 style No bias and no subtraction of mean.
                """
                super().__init__()
                self.weight = nn.Parameter(torch.ones(hidden_size))
                self.variance_epsilon = eps

            def forward(self, hidden_states):
                # layer norm should always be calculated in float32
                variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
                hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

                # convert into half-precision if necessary
                if self.weight.dtype in [torch.float16, torch.bfloat16]:
                    hidden_states = hidden_states.to(self.weight.dtype)

                return self.weight * hidden_states
        bs = 256
        seq = 256
        hidden = 1024
        t5_norm = T5LayerNorm(hidden)
        t5_norm_inputs = [(bs, seq, hidden)]
        run_and_compare_activation(self, t5_norm, t5_norm_inputs)


    # TODO - Assertion failure
    # def test_hard_mish(self):
    #   for compiler in compilers:
    #     run_and_compare_activation(hard_mish, 1024)


if __name__ == "__main__":
    run_tests()
