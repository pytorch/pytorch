import torch
from torch import Tensor
from pathlib import Path
import importlib.util

import torch

spec = importlib.util.find_spec("torchao")
assert spec is not None
SO_PATH = Path(spec.origin).parent / "_C.abi3.so"
print(SO_PATH)
torch.ops.load_library(SO_PATH)

lib = torch.library.Library("torchao", "FRAGMENT")
lib.define("quant_llm_linear(int EXPONENT, int MANTISSA, Tensor _in_feats, Tensor _weights, Tensor _scales, int splitK) -> Tensor")

def quant_llm_linear(
    EXPONENT: int,
    MANTISSA: int,
    _in_feats: Tensor,
    _weights: Tensor,
    _scales: Tensor,
    splitK: int = 1,
) -> Tensor:
    """
    Quant-LLM linear layer A @ W.T. See https://arxiv.org/abs/2401.14112 for more details.

    Arguments
        EXPONENT: number of exponent bits
        MANTISSA: number of mantissa bits
        _in_feats: input activations in FP16
        _weights: packed Floatx weights
        _scales: scale
        splitK: split K

    Returns
        output of linear layer
    """
    return torch.ops.torchao.quant_llm_linear.default(EXPONENT, MANTISSA, _in_feats, _weights, _scales, splitK)


def register_custom_op(name):
    def decorator(func):
        return torch.library.register_fake(f"{name}")(func)
    return decorator


@register_custom_op("torchao::quant_llm_linear")
def _(
    EXPONENT: int,
    MANTISSA: int,
    _in_feats: Tensor,
    _weights: Tensor,
    _scales: Tensor,
    splitK: int = 1,
) -> Tensor:
    torch._check(_in_feats.dim() == 2, lambda: f"input should be a 2d tensor, got {_in_feats.dim()}D")
    torch._check(_in_feats.dtype in (torch.float16, torch.bfloat16), lambda: f"weight must be FP16 or BF16, got {_in_feats.dtype}")
    torch._check(_weights.dim() == 2, lambda: f"weight should be a 2d tensor, got {_weights.dim()}D")
    torch._check(_weights.dtype is torch.uint8, lambda: f"weight must be UINT8, got {_weights.dtype}")
    torch._check(_scales.dim() == 1, lambda: f"scale should be a 2d tensor, got {_scales.dim()}D")
    torch._check(_scales.dtype in (torch.float16, torch.bfloat16), lambda: f"scale must be FP16 or BF16, got {_scales.dtype}")

    BS, IC = _in_feats.shape
    OC, _ = _weights.shape
    N_BITS = 1 + EXPONENT + MANTISSA
    torch._check(IC // 8 * N_BITS == _weights.shape[1], lambda: "Dimensions mismatched")
    torch._check(OC == _scales.shape[0], lambda: "Dimensions mismatched")

    return _in_feats.new_empty((BS, OC))
