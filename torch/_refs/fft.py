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


def apply_norm(x: TensorLike, norm: NormType, signal_numel: int, forward: bool):
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
             int dim, norm: NormType, forward: bool, onesided: bool):
    if not input.is_floating_point:
        raise RuntimeError(f"{func_name} expects a floating point input tensor, but got {input.dtype}")
    input = promote_tensor_fft(input)
    dims = (dim,)

    if n is not None:
        input = resize_fft_input(input, dims, (n,))

    ret = prims.fft_r2c(input, dim=dims, onesided=onesided)
    ret = apply_norm(ret, norm, input.shape[dim], forward)
    return ret if forward else refs.conj(ret)

def _fft_c2c(func_name: str, input: TensorLikeType, n: Optional[int],
             int dim, norm: NormType, forward: bool):
    if not input.dtype.is_complex:
        raise RuntimeError(f"{func_name} expects a complex input tensor, but got {input.dtype}")
    dims = (dim,)

    if n is not None:
        input = resize_fft_input(input, dims, (n,))

    ret = prims.fft_c2c(input, dim=dims, forward=forward)
    return apply_norm(ret, norm, input.shape[dim], forward)

class _ShapeAndDims(NamedTuple):
    shape: List[int]
    dims: List[int]

def _canonicalize_fft_shape_and_dim_args(input: TensorLikeType, shape:
                                         Optional[ShapeType], dim: Optional[DimsType]) -> _ShapeAndDims:
    input_dim = input.ndim
    input_sizes = input.shape

    ret_shape = []
    ret_dims = []

    if dim is not None:
        ret_dims =
