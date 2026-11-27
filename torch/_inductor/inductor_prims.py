# mypy: allow-untyped-defs
from __future__ import annotations

import functools
import logging
import math
import operator
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import _prims, Tensor
from torch._prims_common import (
    compute_elementwise_output_strides,
    make_contiguous_strides_for,
)
from torch._utils import _get_device_index
from torch.nn import functional as F


if TYPE_CHECKING:
    from collections.abc import Sequence


log = logging.getLogger(__name__)


def _blackwell_decompose_k_partial_impl(
    a: Tensor,
    b: Tensor,
    k_split: int,
    config_index: int,
    m_pad: int,
    k_part: int,
) -> Tensor:
    """Reference for the internal aligned decompose-K partial operation."""
    del config_index
    m, k = a.shape
    n = b.shape[1]
    partials = []
    for split in range(k_split):
        start = split * k_part
        end = min(start + k_part, k)
        partial = torch.mm(a[:, start:end], b[start:end], out_dtype=torch.float32)
        partials.append(F.pad(partial, [0, 0, 0, m_pad - m]))
    return torch.stack(partials).view(k_split * m_pad, n)


def _cudnn_batch_norm_statistics_impl(input, running_mean, running_var, momentum, eps):
    dtype = torch.float64 if input.dtype == torch.float64 else torch.float32
    shape = (input.shape[1],)
    mean = (
        torch.zeros(shape, dtype=dtype, device=input.device)
        if running_mean is None
        else running_mean.clone()
    )
    var = (
        torch.ones(shape, dtype=dtype, device=input.device)
        if running_var is None
        else running_var.clone()
    )
    weight, bias = torch.ones_like(mean), torch.zeros_like(mean)
    with torch._C._AutoDispatchBelowAutograd():
        _, save_mean, save_invstd, _ = torch.ops.aten.cudnn_batch_norm.default(
            input, weight, bias, mean, var, True, momentum, eps
        )
    return save_mean, save_invstd, mean, var


def make_prim(
    schema: str,
    impl_aten,
    return_type=_prims.RETURN_TYPE.NEW,
    doc: str = "",
    tags: Sequence[torch.Tag] | None = None,
):
    if return_type is _prims.RETURN_TYPE.INPLACE:
        meta = impl_aten
    elif isinstance(return_type, tuple):

        def meta(*args, **kwargs):
            return tuple(_prims.TensorMeta(o) for o in impl_aten(*args, **kwargs))

    else:

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


cudnn_batch_norm_statistics = make_prim(
    "cudnn_batch_norm_statistics(Tensor input, Tensor? running_mean, Tensor? running_var, float momentum, float eps) -> (Tensor, Tensor, Tensor, Tensor)",
    _cudnn_batch_norm_statistics_impl,
    return_type=(_prims.RETURN_TYPE.NEW,) * 4,
    doc="Compute cuDNN training statistics and updated running statistics without mutating inputs.",
)


def _cudnn_batch_norm_impl(
    input, weight, bias, running_mean, running_var, training, momentum, eps
):
    mean = (
        running_mean.clone() if running_mean is not None else torch.zeros_like(weight)
    )
    var = running_var.clone() if running_var is not None else torch.ones_like(weight)
    with torch._C._AutoDispatchBelowAutograd():
        output, save_mean, invstd, reserve = torch.ops.aten.cudnn_batch_norm.default(
            input,
            weight,
            bias,
            mean if training or running_mean is not None else None,
            var if training or running_var is not None else None,
            training,
            momentum,
            eps,
        )
    return output, save_mean, invstd, reserve, mean, var


def _cudnn_batch_norm_meta(
    input, weight, bias, running_mean, running_var, training, momentum, eps
):
    from torch._prims_common import suggest_memory_format

    output = torch.empty_like(input, memory_format=suggest_memory_format(input))
    stats_shape = (input.shape[1],) if training else (0,)
    mean = torch.empty(stats_shape, dtype=weight.dtype, device=input.device)
    invstd = torch.empty_like(mean)
    reserve_size = torch._C._get_cudnn_batch_norm_reserve_space_size(input, training)  # type: ignore[attr-defined]
    reserve = torch.empty(reserve_size, dtype=torch.uint8, device=input.device)
    new_mean = torch.empty_like(weight if running_mean is None else running_mean)
    new_var = torch.empty_like(weight if running_var is None else running_var)
    return output, mean, invstd, reserve, new_mean, new_var


cudnn_batch_norm = _prims._make_prim(
    schema="inductor_cudnn_batch_norm(Tensor input, Tensor weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)",
    meta=_cudnn_batch_norm_meta,
    impl_aten=_cudnn_batch_norm_impl,
    return_type=(_prims.RETURN_TYPE.NEW,) * 6,
    doc="Functional cuDNN batch norm with its eager forward and backward numerical contracts.",
)


def _cudnn_batch_norm_setup_context(ctx, inputs, output):
    input, weight, bias, mean, var, training, _, eps = inputs
    ctx.training = training
    ctx.eps = eps
    ctx.save_for_backward(
        input,
        weight,
        output[4] if training else mean,
        output[5] if training else var,
        output[1],
        output[2],
        output[3],
    )
    ctx.mark_non_differentiable(*output[1:])


def _cudnn_batch_norm_backward(ctx, grad, *unused):
    input, weight, mean, var, save_mean, invstd, reserve = ctx.saved_tensors
    if ctx.training:
        grads = torch.ops.aten.batch_norm_backward.default(
            grad,
            input,
            weight,
            mean,
            var,
            save_mean,
            invstd,
            True,
            ctx.eps,
            [True, True, True],
            reserve,
        )
    else:
        grads = torch.ops.aten.native_batch_norm_backward.default(
            grad,
            input,
            weight,
            mean,
            var,
            save_mean,
            invstd,
            False,
            ctx.eps,
            list(ctx.needs_input_grad[:3]),
        )
    return (
        *(
            grad if needed else None
            for grad, needed in zip(grads, ctx.needs_input_grad[:3])
        ),
        None,
        None,
        None,
        None,
        None,
    )


torch.library.register_autograd(
    cudnn_batch_norm,
    _cudnn_batch_norm_backward,
    setup_context=_cudnn_batch_norm_setup_context,
)


def _cuda_pointwise_iteration_layout(inputs, contiguous_output):
    shape = torch.broadcast_shapes(*(tensor.shape for tensor in inputs))
    strides = (
        make_contiguous_strides_for(shape)
        if contiguous_output
        else compute_elementwise_output_strides(
            *torch.broadcast_tensors(*inputs), allow_fast_path=False
        )
    )
    return shape, strides


@dataclass(frozen=True)
class _CudaPointwiseChunk:
    shape: tuple[int, ...]
    strides: tuple[tuple[int, ...], ...]
    offsets: tuple[int, ...]


def _cuda_pointwise_chunks(shape, strides):
    """Mirror TensorIterator's byte-offset splits, including singleton coalescing."""
    pending = [(list(shape), [list(s) for s in strides], [0] * len(strides))]
    chunks = []
    while pending:
        size, byte_strides, offsets = pending.pop()
        # TensorIterator coalesces adjacent dimensions after a singleton narrow.
        dim = 1
        while dim < len(size):
            if (
                size[dim - 1] == 1
                or size[dim] == 1
                or all(size[dim - 1] * s[dim - 1] == s[dim] for s in byte_strides)
            ):
                for s in byte_strides:
                    if size[dim - 1] == 1:
                        s[dim - 1] = s[dim]
                    s.pop(dim)
                size[dim - 1] *= size.pop(dim)
            else:
                dim += 1
        count = math.prod(size)
        if count <= 2147483647 and all(
            1 + sum((n - 1) * s for n, s in zip(size, stride)) <= 2147483647
            for stride in byte_strides
        ):
            chunks.append(
                _CudaPointwiseChunk(
                    tuple(size), tuple(map(tuple, byte_strides)), tuple(offsets)
                )
            )
            continue
        # Largest byte extent wins; ties prefer the last iterator dimension.
        split = max(
            range(len(size)),
            key=lambda d: (max((size[d] - 1) * abs(s[d]) for s in byte_strides), d),
        )
        half = size[split] // 2
        for start, length in ((half, size[split] - half), (0, half)):
            child_size = size.copy()
            child_size[split] = length
            pending.append(
                (
                    child_size,
                    [s.copy() for s in byte_strides],
                    [o + s[split] * start for o, s in zip(offsets, byte_strides)],
                )
            )
    return chunks


def _cuda_pointwise_chunk_index(
    physical, chunk, *, floor_div=operator.floordiv, mod=operator.mod
):
    """Decode a physical output byte offset within a possibly gapped chunk."""
    delta = physical - chunk.offsets[0]
    coordinates = []
    for stride in reversed(chunk.strides[0]):
        coordinates.append(floor_div(delta, stride))
        delta = mod(delta, stride)
    coordinates.reverse()
    local, count = 0, 1
    for coord, size in zip(coordinates, chunk.shape):
        local += coord * count
        count *= size
    return coordinates, delta, local, count


def _cuda_pointwise_iterator(inputs, shape, strides):
    from torch._tensor_iterator import TensorIterator

    return TensorIterator(
        outputs=[
            torch.empty_strided(shape, strides, dtype=inputs[0].dtype, device="meta")
        ],
        const_inputs=[
            torch.empty_strided(t.shape, t.stride(), dtype=t.dtype, device="meta")
            for t in inputs
        ],
    )


def _cuda_pointwise_iteration_meta(
    inputs, contiguous_output=True, *, vectorized_tail=False, unroll_prefix=0
):
    if not 0 <= unroll_prefix <= 1024:
        raise ValueError("unroll_prefix must be between zero and 1024")
    shape, strides = _cuda_pointwise_iteration_layout(inputs, contiguous_output)
    return tuple(
        _prims.TensorMeta(
            torch.empty_strided(
                shape, strides, dtype=torch.bool, device=inputs[0].device
            )
        )
        for _ in range(2)
    )


def _cuda_pointwise_iteration_impl(
    inputs, contiguous_output=True, *, vectorized_tail=False, unroll_prefix=0
):
    if vectorized_tail or unroll_prefix or inputs[0].dtype.is_complex:
        return _cuda_pointwise_prefix_impl(
            inputs, contiguous_output, vectorized_tail, unroll_prefix
        )
    expanded = torch.broadcast_tensors(*inputs)
    shape, strides = _cuda_pointwise_iteration_layout(inputs, contiguous_output)
    count = expanded[0].numel()
    if not all(
        all(
            size == 1 or stride == output_stride
            for size, stride, output_stride in zip(shape, tensor.stride(), strides)
        )
        for tensor in expanded
    ):
        mask = torch.empty_strided(
            shape, strides, dtype=torch.bool, device=inputs[0].device
        ).zero_()
        return mask, mask.clone()
    if any(tensor.data_ptr() % 16 for tensor in expanded):
        mask = torch.empty_strided(
            shape, strides, dtype=torch.bool, device=inputs[0].device
        ).fill_(True)
        return mask, ~mask
    index = torch.arange(count, dtype=torch.int64, device=inputs[0].device)
    start = torch.zeros_like(index)
    size = torch.full_like(index, count)
    maximum = count
    while maximum > 268435456:
        active = size > 268435456
        half = size // 2
        right = active & (index - start >= half)
        start = torch.where(right, start + half, start)
        size = torch.where(active, torch.where(right, size - half, half), size)
        maximum = (maximum + 1) // 2
    vectorized = (index - start < size // 1024 * 1024) & (start % 2 == 0)
    return (~vectorized).as_strided(shape, strides), vectorized.as_strided(
        shape, strides
    )


def _cuda_pointwise_prefix_impl(
    inputs, contiguous_output, vectorized_tail, unroll_prefix
):
    shape, strides = _cuda_pointwise_iteration_layout(inputs, contiguous_output)
    iterator = _cuda_pointwise_iterator(inputs, shape, strides)
    itemsize = inputs[0].element_size()
    index = torch.arange(math.prod(shape), dtype=torch.int64, device=inputs[0].device)
    masks = [torch.zeros_like(index, dtype=torch.bool) for _ in range(2)]
    chunks = _cuda_pointwise_chunks(
        tuple(iterator.shape),
        [tuple(iterator.strides(i)) for i in range(iterator.ntensors)],
    )
    for chunk in chunks:
        coordinates, remainder, local, scale = _cuda_pointwise_chunk_index(
            index * itemsize, chunk
        )
        active = (index * itemsize >= chunk.offsets[0]) & (remainder == 0)
        for coord, size in zip(coordinates, chunk.shape):
            active &= coord < size
        contiguous = (
            scale == 1
            or len(chunk.shape) == 1
            and all(s[0] == itemsize for s in chunk.strides)
        )
        aligned = (
            all((t.data_ptr() + o) % 16 == 0 for t, o in zip(inputs, chunk.offsets[1:]))
            and chunk.offsets[0] % 16 == 0
        )
        vectorized = torch.full_like(active, contiguous and aligned and itemsize == 8)
        if not vectorized_tail:
            vectorized &= local < scale // 1024 * 1024
        unrolled = (
            (local % 1024 < unroll_prefix) & ~vectorized
            if unroll_prefix
            else contiguous & ~vectorized
        )
        masks[0] |= active & unrolled
        masks[1] |= active & vectorized
    return tuple(mask.as_strided(shape, strides) for mask in masks)


cuda_pointwise_iteration = _prims._make_prim(
    schema="cuda_pointwise_iteration(Tensor[] inputs, bool contiguous_output=True, *, bool vectorized_tail=False, int unroll_prefix=0) -> (Tensor, Tensor)",
    return_type=_prims.RETURN_TYPE.NEW,
    meta=_cuda_pointwise_iteration_meta,
    impl_aten=_cuda_pointwise_iteration_impl,
    doc="Masks for contiguous unrolled and vectorized iteration of a native CUDA FP64 TensorIterator kernel. The output can be freshly contiguous or preserve the common input layout.",
)


def eager_force_stride(input_tensor: Tensor, stride) -> Tensor:
    if input_tensor.stride() == stride:
        return input_tensor
    new_tensor = input_tensor.clone().as_strided(
        input_tensor.shape,
        stride,
    )
    new_tensor.copy_(input_tensor)
    return new_tensor


masked_sum = _prims._make_prim(
    schema="inductor_masked_sum(Tensor input, Tensor mask) -> Tensor",
    return_type=_prims.RETURN_TYPE.NEW,
    meta=lambda input, mask: _prims.TensorMeta(
        input.new_empty(
            (),
            dtype=input.dtype
            if input.is_floating_point() or input.is_complex()
            else torch.int64,
        )
    ),
    impl_aten=lambda input, mask: input.masked_select(mask).sum(),
    doc="Sum selected values in native compacted reduction order.",
    tags=(torch.Tag.cudagraph_unsafe,),
)


def _masked_sum_setup_context(ctx, inputs, output):
    input, mask = inputs
    ctx.input_shape = input.shape
    ctx.save_for_backward(mask)


def _masked_sum_backward(ctx, grad):
    (mask,) = ctx.saved_tensors
    return torch.where(mask, grad.expand(ctx.input_shape), 0).sum_to_size(
        ctx.input_shape
    ), None


torch.library.register_autograd(
    masked_sum, _masked_sum_backward, setup_context=_masked_sum_setup_context
)


def eager_prepare_softmax(x: Tensor, dim: int) -> tuple[Tensor, Tensor]:
    amax = torch.amax(x, dim, keepdim=True)
    return amax, torch.sum(torch.exp(x - amax), dim, keepdim=True)


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
    doc="Horizontal fusion of many inductor_seed() calls",
    tags=(torch.Tag.nondeterministic_seeded,),
)
lookup_seed = make_prim(
    # if inductor_lookup_seed changes, update partitioners.py
    "inductor_lookup_seed(Tensor seeds, int index) -> Tensor",
    lambda seeds, index: seeds[index].clone(),
    doc="Extract a single seed from the result of inductor_seeds()",
)
# inductor_random() doesn't accept a dtype.
# instead, its lowering always burns in float32, and conversions to a different type
# are explicit in the graph. We therefore need this impl (used during tracing) to hardcode
# the dtype, so it always faithfully produces a float32 tensor during tracing,
# even if the default dtype is set to something else.
random = make_prim(
    "inductor_random(SymInt[] size, Tensor seed, str mode, *, ScalarType? align_dtype=None) -> Tensor",
    lambda size, seed, mode, *, align_dtype=None: getattr(torch, mode)(
        size, device=seed.device, dtype=torch.float32
    ),
    doc="torch.rand()/torch.randn() using backend-specific RNG that can be fused",
)
randint = make_prim(
    "inductor_randint(SymInt low, SymInt high, SymInt[] size, Tensor seed) -> Tensor",
    lambda low, high, size, seed: torch.randint(low, high, size, device=seed.device),
    doc="torch.randint() using backend-specific RNG that can be fused",
)

# Internal plan component for the Blackwell aligned decompose-K prototype.
# Its lowering is a dedicated TMA template over the original rank-2 operands.
blackwell_decompose_k_partial = make_prim(
    "blackwell_decompose_k_partial(Tensor a, Tensor b, int k_split, int config_index, int m_pad, int k_part) -> Tensor",
    _blackwell_decompose_k_partial_impl,
)


def _reserve_rng_state(device: torch.device, used_offset):
    """
    Reserve `used_offset` 32-bit Philox samples on the given CUDA device and
    return (seed, base), where base is in Philox-4x32 units.

    This mirrors how Inductor accounts for Philox consumption so compiled
    dropout kernels can reconstruct eager RNG state.
    """
    dev = device if isinstance(device, torch.device) else torch.device(device)
    if dev.type != "cuda":
        # Only CUDA devices have Philox-based CUDAGenerator. For non-CUDA
        # devices this prim should be dead code and never actually run.
        return 0, 0

    dev_index = _get_device_index(dev, optional=True)
    if dev_index is None:
        dev_index = torch.cuda.current_device()

    gen = torch.cuda.default_generators[dev_index]
    seed_t, off_t, intra_t = torch.ops.inductor_prims.inductor_reserve_rng_state(
        gen, used_offset
    )

    # NOTE: for correctness in eager, intra_t should be 0.
    # Keep everything as tensor math to avoid host sync.
    if intra_t.device.type != off_t.device.type:
        intra = int(intra_t.item())
        base = torch.div(off_t + intra, 4, rounding_mode="floor")
    else:
        base = torch.div(off_t + intra_t, 4, rounding_mode="floor")
    return seed_t, base


def rand_eager_vec(dtype: torch.dtype | None) -> int:
    """Vector width of eager CUDA dropout (fused_dropout_kernel_vec) for an aligned tensor."""
    return 8 if dtype in (torch.float16, torch.bfloat16) else 4


def _rand_eager_offset_impl(offset, device: torch.device) -> Tensor:
    """
    Reserve `offset` 32-bit Philox samples and return a 1-element int64 tensor
    Place-holder: will be replaced by rand_eager_offsets
    In fx_passes/replace_random.py
        fuse_offset_creation_pass()
    """
    return torch.empty(2, dtype=torch.int64, device=device)


def _rand_eager_offsets_impl(offsets, device: torch.device) -> Tensor:
    """
    Batched version of _rand_eager_offset_impl. For each entry in `offsets`,
    reserve that many 32-bit Philox samples and return a 1D int64 tensor
    containing the packed (seed, base) values for each reservation.
    """
    states = [_reserve_rng_state(device, int(off)) for off in offsets]
    seeds = [s for s, _ in states]
    bases = [b for _, b in states]

    def _to_i64(x):
        if isinstance(x, torch.Tensor):
            return x
        return torch.as_tensor(x, device=device, dtype=torch.int64)

    seeds_tensor = torch.stack([_to_i64(x) for x in seeds]).view(-1)
    bases_tensor = torch.stack([_to_i64(x) for x in bases]).view(-1)
    packed = torch.stack([seeds_tensor, bases_tensor], dim=1)
    return packed


def _rand_eager_offsets_meta(offsets, device: torch.device):
    return torch.empty((len(offsets), 2), dtype=torch.int64, device=device)


rand_eager_offset = make_prim(
    "inductor_rand_eager_offset(SymInt offset, Device device) -> Tensor",
    _rand_eager_offset_impl,
    doc=(
        "Reserve `offset` 32-bit Philox samples on `device` and return a "
        "1-element int64 tensor containing packed (seed, base)."
    ),
    tags=(torch.Tag.nondeterministic_seeded,),
)


rand_eager_offsets = _prims._make_prim(
    schema="inductor_rand_eager_offsets(SymInt[] offsets, Device device) -> Tensor",
    return_type=_prims.RETURN_TYPE.NEW,
    meta=_rand_eager_offsets_meta,
    impl_aten=_rand_eager_offsets_impl,
    doc=(
        "Batched version of inductor_rand_eager_offset. For each entry in "
        "`offsets`, reserves that many 64-bit Philox samples and returns "
        "packed (seed, base) values."
    ),
    tags=(torch.Tag.nondeterministic_seeded,),
)


force_stride_order = make_prim(
    "inductor_force_stride_order(Tensor input, SymInt[] stride) -> Tensor",
    eager_force_stride,
    doc="Force the stride order for input tensor. No-op if the input tensor already has the stride. Do a copy otherwise",
)
_unsafe_index_put_ = make_prim(
    "_unsafe_index_put_(Tensor(a!) self, Tensor?[] indices, Tensor values, bool accumulate=False) -> Tensor(a!)",
    lambda self, indices, values, accumulate=False: torch.ops.aten.index_put_(
        self, indices, values, accumulate
    ),
    return_type=_prims.RETURN_TYPE.INPLACE,
    doc="Unsafe index_put_ (doesn't issue device asserts)",
)
fma = make_prim(
    "fma(Tensor a, Tensor b, Tensor c) -> Tensor",
    lambda a, b, c: (a * b) + c,
    doc="Fused multiply add: fma(a, b, c) -> (a * b) + c without rounding after the multiplication",
    tags=(torch.Tag.pointwise,),
)


def _fma_setup_context(ctx, inputs, output):
    a, b, c = inputs
    ctx.a_shape = a.shape
    ctx.b_is_tensor = isinstance(b, Tensor)
    if ctx.b_is_tensor:
        ctx.b_shape = b.shape
        ctx.save_for_backward(a, b)
    else:
        ctx.b = b
        ctx.save_for_backward(a)
    ctx.c_shape = c.shape


def _fma_backward(ctx, grad):
    if ctx.b_is_tensor:
        a, b = ctx.saved_tensors
    else:
        (a,) = ctx.saved_tensors
        b = ctx.b

    grad_a = (grad * b).sum_to_size(ctx.a_shape) if ctx.needs_input_grad[0] else None
    grad_b = (
        (grad * a).sum_to_size(ctx.b_shape)
        if ctx.b_is_tensor and ctx.needs_input_grad[1]
        else None
    )
    grad_c = grad.sum_to_size(ctx.c_shape) if ctx.needs_input_grad[2] else None
    return grad_a, grad_b, grad_c


torch.library.register_autograd(
    fma,
    _fma_backward,
    setup_context=_fma_setup_context,
)


# Register DTensor sharding strategies for inductor prims ops.
# This must happen here (not in _pointwise_ops.py) because the ops
# don't exist until make_prim() is called above.
if torch.distributed.is_available():
    from torch.distributed.tensor._ops._pointwise_ops import register_inductor_prims

    register_inductor_prims()

prepare_softmax_online = make_prim(
    "prepare_softmax_online(Tensor a, int dim) -> (Tensor, Tensor)",
    eager_prepare_softmax,
    return_type=(_prims.RETURN_TYPE.NEW, _prims.RETURN_TYPE.NEW),
    doc="Prepare the softmax by computing the max and sum.",
)


def _flattened_index_to_nd(indices, width):
    import sympy

    from torch.utils._sympy.functions import FloorDiv

    dim = len(width)

    if dim == 1:
        return [indices]
    elif dim >= 2:
        m = functools.reduce(operator.mul, width[1:])
        if isinstance(indices, sympy.Expr) or isinstance(m, sympy.Expr):
            ih = FloorDiv(indices, m)
        else:
            ih = indices // m
        indices_new = indices - (ih * m)
        return [ih, *_flattened_index_to_nd(indices_new, width[1:])]
    else:
        raise ValueError(f"Unknown dim: {dim}")


def _flatten_index(indices, width):
    result = indices[0]
    for d in range(1, len(indices)):
        result = width[d] * result + indices[d]
    return result


def _low_memory_max_pool_with_offsets_aten(
    self,
    kernel_size,
    stride,
    padding,
    dilation,
    ceil_mode,
):
    dim = len(kernel_size)
    if dim == 2:
        vals, indices = torch.ops.aten.max_pool2d_with_indices(
            self, kernel_size, stride, padding, dilation, ceil_mode
        )
    else:
        vals, indices = torch.ops.aten.max_pool3d_with_indices(
            self, kernel_size, stride, padding, dilation, ceil_mode
        )

    idhw = _flattened_index_to_nd(indices, self.shape[-dim:])

    dhw_inc = []

    for d in range(dim):
        bh_shape = [1] * self.ndim
        bh_shape[-dim + d] = -1
        bh = torch.arange(
            indices.shape[-dim + d], dtype=torch.int64, device=self.device
        ).view(bh_shape)
        hbase = bh * stride[d] - padding[d]
        h_inc = (idhw[d] - hbase) // dilation[d]
        dhw_inc.append(h_inc)

    offsets = _flatten_index(dhw_inc, kernel_size)

    return vals, offsets.to(torch.int8)


def _low_memory_max_pool_offsets_to_indices_aten(
    offsets,
    kernel_size,
    input_size,
    stride,
    padding,
    dilation,
):
    dim = len(kernel_size)
    offsets = offsets.to(torch.int64)
    dhw_inc = _flattened_index_to_nd(offsets, kernel_size)

    idhw = []
    for d in range(dim):
        bh_shape = [1] * offsets.ndim
        bh_shape[-dim + d] = -1
        bh = torch.arange(
            offsets.shape[-dim + d], dtype=torch.int64, device=offsets.device
        ).view(bh_shape)
        hbase = bh * stride[d] - padding[d]
        idhw.append(hbase + dhw_inc[d] * dilation[d])

    return _flatten_index(idhw, input_size)


_low_memory_max_pool_with_offsets = make_prim(
    "_low_memory_max_pool_with_offsets(Tensor self, SymInt[] kernel_size, SymInt[] stride,  SymInt[] padding, SymInt[] dilation, bool ceil_mode) -> (Tensor, Tensor)",
    _low_memory_max_pool_with_offsets_aten,
    return_type=(_prims.RETURN_TYPE.NEW, _prims.RETURN_TYPE.NEW),
    doc="Instead of returning indices, returns indices offsets.",
)

_low_memory_max_pool_offsets_to_indices = make_prim(
    "_low_memory_max_pool_offsets_to_indices(Tensor self, SymInt[] kernel_size, SymInt[] input_size, SymInt[] stride, SymInt[] padding, SymInt[] dilation) -> Tensor",
    _low_memory_max_pool_offsets_to_indices_aten,
    doc="Convert small int offsets to regular indices.",
)


def _cvt_e8m0_rceil_aten(inp: Tensor) -> Tensor:
    """
    Convert float to e8m0 format with ceiling rounding and satfinite semantics.

    e8m0 format: 8-bit biased exponent (bias=127), no mantissa.
    For MX format scaling, this extracts the exponent with ceiling rounding.
    Uses satfinite semantics: inf is saturated to 254 (max finite e8m0).
    Accepts float32, float16, or bfloat16 (upcasted to float32 internally).
    """
    if inp.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError(
            f"cvt_e8m0_rceil requires float32, float16, or bfloat16 input, got {inp.dtype}"
        )
    if inp.dtype != torch.float32:
        inp = inp.to(torch.float32)
    inp_bits = inp.view(torch.int32)
    biased_exp = (inp_bits >> 23) & 0xFF
    mantissa = inp_bits & 0x7FFFFF
    needs_round_up = mantissa != 0
    e8m0_biased = biased_exp + needs_round_up.to(torch.int32)
    # satfinite: clamp to max finite e8m0 value (254), not 255 (inf/nan)
    e8m0_biased = torch.clamp(e8m0_biased, 0, 254)
    return e8m0_biased.to(torch.uint8)


cvt_e8m0_rceil = make_prim(
    "inductor_cvt_e8m0_rceil(Tensor input) -> Tensor",
    _cvt_e8m0_rceil_aten,
    doc="Convert float to e8m0 with ceiling rounding. Uses PTX cvt.rp.satfinite.ue8m0x2.f32 on SM100+.",
)
