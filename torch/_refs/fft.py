import torch
import torch._prims as prims
import torch._refs as refs
import torch._prims.utils as utils
from torch._prims.utils import (
    TensorLike,
    TensorLikeType,
    ShapeType,
    DimsType,
    DimsSequenceType,
)

from typing import Union
from typing_extensions import Literal
import math

NormType = Union[None, Literal["forward"], Literal["backward"], Literal["ortho"]]


def apply_norm(x: TensorLike, norm: str, signal_numel: int, forward: bool):
    if norm == "ortho":
        return prims.mul(x, 1 / math.sqrt(signal_numel))

    normalize = (not forward and norm == "backward") or (
        forward and (norm is None or norm == "forward")
    )
    if normalize:
        return prims.mul(x, 1 / signal_numel)
    else:
        return x


def promote_type_fft(
    dtype: torch.dtype, require_complex: bool, device: torch.device
) -> torch.dtype:
    if dtype.is_complex:
        return dtype

    # Promote integral to default float type
    if not dtype.is_floating_point:
        dtype = torch.get_default_dtype()

    is_rocm = False  # TODO: How to discern rocm from CUDA?
    if dtype == torch.half and (is_rocm or device.type != "cuda"):
        raise RuntimeError("Unsupported dtype Half")

    if require_complex:
        dtype = utils.corresponding_complex_dtype(dtype)

    return dtype


def promote_tensor_fft(t: TensorLikeType, require_complex: bool = False):
    cur_type = t.dtype
    new_type = promote_type_fft(cur_type, require_complex, t.device)
    return t if cur_type == new_type else t.to(new_type)


# Fixes the shape of x such that x.size(dims[i]) == sizes[i],
# either by zero-padding, or by slicing x starting from 0.
def resize_fft_input(
    x: TensorLikeType, dims: DimsSequenceType, sizes: ShapeType
) -> TensorLikeType:
    assert len(dims) == len(sizes)
    must_copy = False
    x_sizes = x.shape
    pad_amount = [0] * len(x_sizes) * 2
    for i in range(len(dims)):
        if sizes[i] == -1:
            continue

        if x_sizes[dims[i]] < sizes[i]:
            must_copy = True
            pad_idx = len(pad_amount) - 2 * dims[i] - 1
            pad_amount[pad_idx] = sizes[i] - x_sizes[dims[i]]

        if x_sizes[dims[i]] > sizes[i]:
            x = refs.narrow(x, dims[i], 0, sizes[i])

    return refs.constant_pad_nd(x, pad_amount) if must_copy else x

def _fft_r2c(func_name: str, input: TensorLikeType, n: Optional[int],
             int dim, norm: Optional[str], forward: bool, onesided: bool):
    if not input.is_complex:
        raise RuntimeError(f"{func_name} expects a real input tensor, but got {input.dtype}")
    input = promote_tensor_fft(input)

    if n is not None:
        input = resize_fft_input(input, (dim,), (n,))
    else:
        n = input.shape[dim]

    n = n if n is not None else input.shape[dim]

    ret = prims.fft_r2c(input, dim=dim, onesided=onesided)

    return ret if forward else prims.conj(ret)
