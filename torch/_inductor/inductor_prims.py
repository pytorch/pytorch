import logging

import torch
from torch import _prims
from torch._prims_common import RETURN_TYPE
from .utils import pad_listlike

log = logging.getLogger(__name__)


def make_prim(
    schema,
    impl_aten,
    return_type=_prims.RETURN_TYPE.NEW,
    doc="",
    tags=None,
):
    def meta(*args, **kwargs):
        return _prims.TensorMeta(impl_aten(*args, **kwargs))

    return _prims._make_prim(
        schema=schema,
        return_type=return_type,
        meta=meta,
        impl_aten=impl_aten,
        doc=doc,
        tags=tags,
    )


def eager_force_stride(input_tensor, stride):
    if input_tensor.stride() == stride:
        return input_tensor
    new_tensor = input_tensor.clone().as_strided(
        input_tensor.shape,
        stride,
    )
    new_tensor.copy_(input_tensor)
    return new_tensor


# Custom prims used for handling randomness
seed = make_prim(
    "inductor_seed(Device device) -> Tensor",
    lambda device: torch.randint(2**63 - 1, [], device=device),
    doc="create a fresh seed (one per call) for use with inductor_rand",
    tags=(torch.Tag.nondeterministic_seeded,),
)
seeds = make_prim(
    "inductor_seeds(int count, Device device) -> Tensor",
    lambda count, device: torch.randint(2**63 - 1, [count], device=device),
    doc="Horizontally fusion of many inductor_seed() calls",
    tags=(torch.Tag.nondeterministic_seeded,),
)
lookup_seed = make_prim(
    # if inductor_lookup_seed changes, update partitioners.py
    "inductor_lookup_seed(Tensor seeds, int index) -> Tensor",
    lambda seeds, index: seeds[index],
    doc="Extract a single seed from the result of inductor_seeds()",
)
random = make_prim(
    "inductor_random(SymInt[] size, Tensor seed, str mode) -> Tensor",
    lambda size, seed, mode: getattr(torch, mode)(size, device=seed.device),
    doc="torch.rand()/torch.randn() using backend-specific RNG that can be fused",
)
randint = make_prim(
    "inductor_randint(SymInt low, SymInt high, SymInt[] size, Tensor seed) -> Tensor",
    lambda low, high, size, seed: torch.randint(low, high, size, device=seed.device),
    doc="torch.randint() using backend-specific RNG that can be fused",
)
force_stride_order = make_prim(
    "inductor_force_stride_order(Tensor input, SymInt[] stride) -> Tensor",
    lambda input_tensor, stride: eager_force_stride(input_tensor, stride),
    doc="Force the stride order for input tensor. No-op if the input tensor already has the stride. Do a copy otherwise",
)


def _inductor_bucketize_impl(input, boundaries, *, out_int32=False, right=False):
    return torch.bucketize(input, boundaries, out_int32=out_int32, right=right)


def _inductor_bucketize_meta(input, boundaries, *, out_int32=False, right=False):
    return torch.empty_like(
        input,
        memory_format=torch.preserve_format,
        dtype=(torch.int32 if out_int32 else torch.int64),
    )


_bucketize = _prims._make_prim(
    schema="_inductor_bucketize(Tensor input, Tensor boundaries, *, bool out_int32=False, bool right=False) -> Tensor",
    meta=_inductor_bucketize_meta,
    impl_aten=_inductor_bucketize_impl,
    return_type=RETURN_TYPE.NEW,
    doc="Same as torch.bucketize(), but does not get decomposed.",
)

def _low_mem_maxpool2d_with_indices_meta(*args, **kwargs):
    out, indices = torch.ops.aten.max_pool2d_with_indices.default(*args, **kwargs)
    return (out, indices.to(torch.int8))

def _low_mem_maxpool2d_with_indices_aten(*args, **kwargs):
    out, indices = torch.ops.aten.max_pool2d_with_indices.default(*args, **kwargs)
    assert False

class LowMemoryMaxPool2d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False):
        with torch._C._AutoDispatchBelowAutograd():
            old = torch._C._dispatch_tls_is_dispatch_key_excluded(torch._C.DispatchKey.ADInplaceOrView)
            out, indices_offset = torch.ops.prims._low_memory_maxpool2d_with_indices(x, kernel_size, stride, padding, dilation, ceil_mode)

        if padding == 0:
            padding = [0, 0]
        if dilation == 1:
            dilation = [1, 1]
        if not stride:
            stride = kernel_size
        kernel_size = pad_listlike(kernel_size, 2)
        stride = pad_listlike(stride, 2)
        padding = pad_listlike(padding, 2)
        dilation = pad_listlike(dilation, 2)

        ctx.save_for_backward(x, indices_offset)
        ctx.kernel_size = kernel_size
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.ceil_mode = ceil_mode

        ctx.mark_non_differentiable(indices_offset)

        return (out, indices_offset)

    @staticmethod
    def backward(ctx, grad_output, ind_offset):
        self, indices_offset = ctx.saved_tensors
        device = ind_offset.device
        kernel_size = ctx.kernel_size
        padding = ctx.padding
        stride = ctx.stride

        grad = torch.ops.prims._low_mem_maxpool2d_with_indices_backward(grad_output, self, ctx.kernel_size, ctx.stride, ctx.padding, ctx.dilation, ctx.ceil_mode, indices_offset)
        return (grad, None, None, None)


_low_mem_maxpool2d_with_indices = _prims._make_prim(
    schema="_low_memory_maxpool2d_with_indices(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, int[2] dilation=1, bool ceil_mode=False) -> (Tensor, Tensor)",
    meta=_low_mem_maxpool2d_with_indices_meta,
    impl_aten=_low_mem_maxpool2d_with_indices_aten,
    return_type=RETURN_TYPE.NEW,
    doc="Instead of returning indices, returns indices offsets.",
    autograd_impl=LowMemoryMaxPool2d.apply,
)


def _low_mem_maxpool2d_with_indices_backward_meta(grad_output, self, kernel_size, stride, padding, dilation, ceil_mode, indices):
    indices = indices.to(torch.int64)
    return torch.ops.aten.max_pool2d_with_indices_backward(grad_output, self, kernel_size, stride, padding, dilation, ceil_mode, indices)

def not_implemented(*args, **kwargs):
    assert False


_low_mem_maxpool2d_with_indices_backward = _prims._make_prim(
    schema="_low_mem_maxpool2d_with_indices_backward(Tensor grad_output, Tensor self, int[2] kernel_size, int[2] stride, int[2] padding, int[2] dilation, bool ceil_mode, Tensor indices) -> (Tensor)",
    meta=_low_mem_maxpool2d_with_indices_backward_meta,
    impl_aten=not_implemented,
    return_type=RETURN_TYPE.NEW,
    doc="Low Mem Max Pool 2d indices backward.",
)
