# mypy: allow-untyped-decorators
# mypy: allow-untyped-defs
import math
import os
import weakref
from functools import lru_cache
from typing import Optional

import torch
from torch._dynamo.utils import warn_once
from torch.utils._triton import has_triton

from ._triton_ops_meta import get_meta


TORCH_SPARSE_BSR_SCATTER_MM_LRU_CACHE_SIZE = int(
    os.getenv("TORCH_SPARSE_BSR_SCATTER_MM_LRU_CACHE_SIZE", 2)
)


def check(cond, msg):
    if not cond:
        raise ValueError(msg)


def check_bsr_layout(f_name, t):
    check(
        t.layout == torch.sparse_bsr,
        f"{f_name}(): only BSR sparse format is supported for the sparse argument.",
    )


def check_device(f_name, t, device):
    check(
        t.device == device and t.device.type == "cuda",
        f"{f_name}(): all inputs are expected to be on the same GPU device.",
    )


def check_mm_compatible_shapes(f_name, lhs, rhs):
    check(
        lhs.dim() >= 2 and rhs.dim() >= 2,
        f"{f_name}(): all inputs involved in the matrix product are expected to be at least 2D, "
        f"but got lhs.dim() == {lhs.dim()} and rhs.dim() == {rhs.dim()}.",
    )

    _m, kl = lhs.shape[-2:]
    kr, _n = rhs.shape[-2:]

    check(
        kl == kr,
        f"{f_name}(): arguments' sizes involved in the matrix product are not compatible for matrix multiplication, "
        f"got lhs.shape[-1] == {kl} which is not equal to rhs.shape[-2] == {kr}.",
    )


def check_dtype(f_name, t, dtype, *additional_dtypes):
    check(
        t.dtype == dtype
        and t.dtype
        in ((torch.half, torch.bfloat16, torch.float) + tuple(*additional_dtypes)),
        f"{f_name}(): all inputs are expected to be of the same dtype "
        f"and one of (half, bfloat16, float32) or {additional_dtypes}, "
        f"but got dtype == {t.dtype}.",
    )


def check_blocksize(f_name, blocksize):
    assert len(blocksize) == 2

    def is_power_of_two(v):
        return not (v & (v - 1))

    def is_compatible_blocksize(b):
        res = True
        for blocksize in b:
            # Triton loads only blocks which are at least 16 and powers of 2.
            res = (blocksize >= 16 and is_power_of_two(blocksize)) and res
        return res

    check(
        is_compatible_blocksize(blocksize),
        f"{f_name}(): sparse inputs' blocksize ({blocksize[0]}, {blocksize[1]}) "
        "should be at least 16 and a power of 2 in each dimension.",
    )


def make_triton_contiguous(t):
    """Return input as a triton-contiguous tensor.

    A triton-contiguous tensor is defined as a tensor that has strides
    with minimal value smaller than or equal to 1.

    While triton kernels support triton-non-contiguous tensors (all
    strides being greater than 1) arguments, a considerable slow-down
    occurs because tensor data is copied element-wise rather than
    chunk-wise. Zero strides is assumed to not have this defect.
    """
    if min(t.stride()) > 1:
        # TODO: investigate if contiguity along other axes than the
        # last one can be beneficial for performance
        return t.contiguous()
    else:
        return t


def broadcast_batch_dims(f_name, *tensors):
    try:
        return torch.broadcast_shapes(*(t.shape[:-2] for t in tensors))
    except Exception:
        check(False, f"{f_name}(): inputs' batch dimensions are not broadcastable!")


def slicer(dim, slice_range, *tensors):
    for t in tensors:
        slices = [slice(None)] * t.dim()
        slices[dim] = slice_range
        yield t[slices]


def multidim_slicer(dims, slices, *tensors):
    for t in tensors:
        s = [slice(None)] * t.dim()
        for d, d_slice in zip(dims, slices, strict=False):
            if d is not None:
                s[d] = d_slice
        yield t[tuple(s)]


def ptr_stride_extractor(*tensors):
    for t in tensors:
        yield t
        yield from t.stride()


def grid_partitioner(full_grid, grid_blocks, tensor_dims_map):
    assert 0 <= len(full_grid) <= 3
    assert 0 <= len(grid_blocks) <= 3

    import itertools

    def generate_grid_points():
        for fg, mg in zip(full_grid, grid_blocks, strict=False):
            yield range(0, fg, mg)

    def generate_sliced_tensors(slices):
        for t, t_dims in tensor_dims_map.items():
            yield next(multidim_slicer(t_dims, slices, t))

    for grid_point in itertools.product(*generate_grid_points()):
        grid = [
            min(fg - gp, mg)
            for fg, gp, mg in zip(full_grid, grid_point, grid_blocks, strict=False)
        ]
        slices = [slice(gp, gp + g) for gp, g in zip(grid_point, grid, strict=False)]
        # grid_points are iterated in a "contiguous" order, i.e.
        # left dimensions traversed slower than right dimensions.
        # This order is reversed for CUDA grids.
        yield grid[::-1], *generate_sliced_tensors(slices)


def launch_kernel(kernel, tensor_dims_map, full_grid, grid_blocks=None):
    # cuda_max_grid = (2 ** 31 - 1, 2 ** 16 - 1, 2 ** 16 - 1)
    cuda_max_grid = (2147483647, 65535, 65535)[::-1]
    if grid_blocks is None:
        grid_blocks = cuda_max_grid
    else:

        def valid_grid_dim(g, mg):
            if g is None:
                return mg
            else:
                # grid must be at least 1 and no greater than mg
                return max(1, min(g, mg))

        grid_blocks = tuple(
            valid_grid_dim(g, mg)
            for g, mg in zip(grid_blocks, cuda_max_grid, strict=False)
        )  # type: ignore[assignment]

    for grid, *sliced_tensors in grid_partitioner(
        full_grid, grid_blocks, tensor_dims_map
    ):
        kernel(grid, *sliced_tensors)


def prepare_inputs(bsr, *dense_tensors):
    # Introduce fake batch dimension if not present for convenience.
    crow_indices = bsr.crow_indices().unsqueeze(0)
    col_indices = bsr.col_indices().unsqueeze(0)
    values = make_triton_contiguous(bsr.values().unsqueeze(0))
    tensors = [make_triton_contiguous(t.unsqueeze(0)) for t in dense_tensors]

    # Compute broadcasted batch dimension
    batch_dims_broadcasted = torch.broadcast_shapes(
        values.shape[:-3], *(t.shape[:-2] for t in tensors)
    )

    # Broadcast batch dimensions and squash.
    # The result can be either a view or a copy.
    def batch_broadcast_and_squash(t, batch_dims, invariant_dims):
        return t.broadcast_to(batch_dims + invariant_dims).flatten(
            0, len(batch_dims) - 1
        )

    crow_indices = batch_broadcast_and_squash(
        crow_indices, batch_dims_broadcasted, (-1,)
    )

    col_indices = batch_broadcast_and_squash(col_indices, batch_dims_broadcasted, (-1,))
    values = batch_broadcast_and_squash(
        values, batch_dims_broadcasted, values.shape[-3:]
    )
    tensors = [
        batch_broadcast_and_squash(t, batch_dims_broadcasted, t.shape[-2:])
        for t in tensors
    ]

    return crow_indices, col_indices, values, *tensors


def broadcast_batch_dims_bsr(f_name, bsr, *tensors):
    batch_shape = broadcast_batch_dims(f_name, bsr, *tensors)

    crow_indices = bsr.crow_indices().broadcast_to(batch_shape + (-1,))
    col_indices = bsr.col_indices().broadcast_to(batch_shape + (-1,))
    values = bsr.values().broadcast_to(batch_shape + bsr.values().shape[-3:])
    size = batch_shape + bsr.shape[-2:]
    return torch.sparse_compressed_tensor(
        crow_indices, col_indices, values, size=size, layout=bsr.layout
    )


# NOTE: this function will ALWAYS create a view
def tile_to_blocksize(t, blocksize):
    *rest, m, n = t.shape
    new_shape = rest + [
        m // blocksize[0],
        blocksize[0],
        n // blocksize[1],
        blocksize[1],
    ]
    # using .view instead of .reshape to ensure that the result is
    # indeed a view:
    return t.view(new_shape).transpose(-3, -2)


def as1Dbatch(tensor):
    """Return tensor as 3D tensor by either prepending new dimensions to
    the tensor shape (when ``tensor.ndim < 3``), or by collapsing
    starting dimensions into the first dimension (when ``tensor.ndim >
    3``).
    """
    while tensor.ndim < 3:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim > 3:
        tensor = tensor.flatten(0, tensor.ndim - 3)
    assert tensor.ndim == 3, tensor.shape
    return tensor


def scatter_mm(blocks, others, indices_data, *, accumulators=None):
    """Scattered matrix multiplication of tensors.

    A scattered matrix multiplication is defined as a series of matrix
    multiplications applied to input tensors according to the input
    and output mappings specified by indices data.

    The following indices data formats are supported for defining a
    scattered matrix multiplication operation (:attr:`indices_data[0]`
    holds the name of the indices data format as specified below):

    - ``"scatter_mm"`` - matrix multiplications scattered in batches
      of tensors.

      If :attr:`blocks` is a :math:`(* \times M \times K) tensor,
      :attr:`others` is a :math:`(* \times K \times N)` tensor,
      :attr:`accumulators` is a :math:`(* \times M \times N)` tensor,
      and :attr:`indices = indices_data['indices']` is a :math:`(*
      \times 3)` tensor, then the operation is equivalent to the
      following code::

        c_offsets, pq = indices_data[1:]
        for r in range(len(c_offsets) - 1):
            for g in range(c_offsets[r], c_offsets[r + 1]):
                p, q = pq[g]
                accumulators[r] += blocks[p] @ others[q]

    - ``"bsr_strided_mm"`` - matrix multiplications scattered in
      batches of tensors and a tensor.

      If :attr:`blocks` is a :math:`(Ms \times Ks) tensor,
      :attr:`others` is a :math:`(* \times K \times N)` tensor,
      :attr:`accumulators` is a :math:`(* \times M \times N)` tensor, then
      the operation is equivalent to the following code::

        c_indices, r_offsets, p_offsets, q_offsets, meta = indices_data[1:]
        for b in range(nbatches):
            for i, r in enumerate(r_offsets):
                r0, r1 = divmod(r, N)
                acc = accumulators[b, r0 : r0 + Ms, r1 : r1 + Ns]
                for g in range(c_indices[i], c_indices[i + 1]):
                    p = p_offsets[g]
                    q0, q1 = divmod(q_offsets[g], N)
                    acc += blocks[p] @ others[b, q0 : q0 + Ks, q1 : q1 + Ns]

      where ``Ns = N // meta['SPLIT_N']``, and ``M`` and ``K`` are
      integer multiples of ``Ms`` and ``Ks``, respectively.

    - ``"bsr_strided_mm_compressed"`` - matrix multiplications
      scattered in batches of tensors and a tensor. A memory and
      processor efficient version of ``"bsr_strided_mm"`` format.  If
      :attr:`blocks` is a :math:`(Ms \times Ks) tensor, :attr:`others`
      is a :math:`(* \times K \times N)` tensor, :attr:`accumulators`
      is a :math:`(* \times M \times N)` tensor, then the operation is
      equivalent to the following code::

        c_indices, r_offsets, q_offsets, meta = indices_data[1:]
        for b in range(nbatches):
            for r in r_offsets:
                m = (r // N) // Ms
                n = (r % N) // Ns
                r0, r1 = divmod(r, N)
                c0, c1 = c_indices[m], c_indices[m + 1]
                acc = accumulators[b, r0 : r0 + Ms, r1 : r1 + Ns]
                for i, p in enumerate(range(c0, c1)):
                    q = q_offsets[n * c1 + (SPLIT_N - n) * c0 + i]
                    q0, q1 = divmod(q, N)
                    acc += blocks[p] @ others[b, q0 : q0 + Ks, q1 : q1 + Ns]

      where ``Ns = N // meta['SPLIT_N']``, and ``M`` and ``K`` are
      integer multiples of ``Ms`` and ``Ks``, respectively.

      Notice that the order of ``r_offsets`` items can be arbitrary;
      this property enables defining swizzle operators via
      rearrangements of ``r_offsets`` items..

    Auxiliary functions are provided for pre-computing
    :attr:`indices_data`. For example,
    :func:`bsr_scatter_mm_indices_data` is used to define indices data
    for matrix multiplication of BSR and strided tensors.

    Parameters
    ----------
    blocks (Tensor): a 3-D tensor of first matrices to be multiplied

    others (Tensor): a tensor of second matrices to be multiplied. If
      ``indices_data[0]=="scatter_mm"``, the tensor is a 1-D batch
      tensor of second input matrices to be multiplied. Otherwise, the
      second input matrices are slices of the :attr:`others` tensor.
    indices_data (tuple): a format data that defines the inputs and
      outputs of scattered matrix multiplications.

    Keyword arguments
    -----------------

    accumulators (Tensor, optional): a tensor of matrix product
      accumulators. If ``indices_data[0]=="scatter_mm"``, the tensor
      is a 1-D batch tensor of output matrices. Otherwise, output
      matrices are slices of the :attr:`accumulators` tensor.
    """
    indices_format = indices_data[0]

    assert blocks.ndim == 3
    _P, Ms, Ks = blocks.shape

    if indices_format == "scatter_mm":
        c_offsets, pq = indices_data[1:]

        assert others.ndim == 3
        _Q, Ks_, Ns = others.shape
        assert Ks == Ks_

        if accumulators is None:
            R = c_offsets.shape[0] - 1
            accumulators = torch.zeros(
                (R, Ms, Ns), dtype=blocks.dtype, device=blocks.device
            )
        else:
            R, Ms_, Ns_ = accumulators.shape
            assert Ms_ == Ms
            assert Ns_ == Ns

        if Ms % 16 or Ks % 16 or Ns % 16 or _scatter_mm2 is None:
            for r in range(c_offsets.shape[0] - 1):
                g0 = c_offsets[r]
                g1 = c_offsets[r + 1]
                for g in range(g0, g1):
                    p, q = pq[g]

                    accumulators[r] += blocks[p] @ others[q]
        else:
            _scatter_mm2(blocks, others, c_offsets, pq, accumulators)
        return accumulators

    elif indices_format == "bsr_strided_mm":
        others_shape = others.shape
        others = as1Dbatch(others)

        B, K, N = others.shape
        assert K % Ks == 0

        c_indices, r_offsets, p_offsets, q_offsets, meta = indices_data[1:]
        SPLIT_N = meta["SPLIT_N"]

        if accumulators is None:
            M = Ms + (r_offsets.max().item() + 1) // N
            accumulators = torch.zeros(
                (*others_shape[:-2], M, N), dtype=blocks.dtype, device=blocks.device
            )
        else:
            M, N_ = accumulators.shape[-2:]
            assert N_ == N

        accumulators_shape = accumulators.shape
        accumulators = as1Dbatch(accumulators)

        Ns = N // SPLIT_N

        if Ms % 16 or Ks % 16 or Ns % 16 or _scatter_mm6 is None:
            accumulators.zero_()
            for b in range(B):
                for r in range(r_offsets.shape[0]):
                    r_ = r_offsets[r].item()
                    g0 = c_indices[r].item()
                    g1 = c_indices[r + 1].item()
                    r0, r1 = divmod(r_, N)
                    acc = accumulators[b, r0 : r0 + Ms, r1 : r1 + Ns]
                    for g in range(g0, g1):
                        p, q = p_offsets[g], q_offsets[g]
                        q0, q1 = divmod(q.item(), N)
                        acc += blocks[p] @ others[b, q0 : q0 + Ks, q1 : q1 + Ns]
        else:
            _scatter_mm6(
                blocks,
                others,
                c_indices,
                r_offsets,
                p_offsets,
                q_offsets,
                meta,
                accumulators,
            )
        return accumulators.view(accumulators_shape)

    elif indices_format == "bsr_strided_mm_compressed":
        others_shape = others.shape
        others = as1Dbatch(others)

        B, K, N = others.shape
        assert K % Ks == 0

        c_indices, r_offsets, q_offsets, meta = indices_data[1:]
        SPLIT_N = meta["SPLIT_N"]

        if accumulators is None:
            M = Ms + (r_offsets.max().item() + 1) // N
            accumulators = torch.zeros(
                (*others_shape[:-2], M, N), dtype=blocks.dtype, device=blocks.device
            )
        else:
            M, N_ = accumulators.shape[-2:]
            assert N_ == N

        accumulators_shape = accumulators.shape
        accumulators = as1Dbatch(accumulators)

        Ns = N // SPLIT_N

        if Ms % 16 or Ks % 16 or Ns % 16 or _scatter_mm6 is None:
            for b in range(B):
                for j in range(len(r_offsets)):
                    r0, r1 = divmod(r_offsets[j].item(), N)
                    m = r0 // Ms
                    n = r1 // Ns
                    c0 = c_indices[m].item()
                    c1 = c_indices[m + 1].item()
                    acc = accumulators[b, r0 : r0 + Ms, r1 : r1 + Ns]
                    for i, p in enumerate(range(c0, c1)):
                        q = q_offsets[n * c1 + (SPLIT_N - n) * c0 + i].item()
                        q0, q1 = divmod(q, N)
                        acc += blocks[p] @ others[b, q0 : q0 + Ks, q1 : q1 + Ns]
        else:
            p_offsets = torch.empty(
                (0,), dtype=q_offsets.dtype, device=q_offsets.device
            )
            _scatter_mm6(
                blocks,
                others,
                c_indices,
                r_offsets,
                p_offsets,
                q_offsets,
                meta,
                accumulators,
            )
        return accumulators.view(accumulators_shape)

    else:
        raise NotImplementedError(indices_format)


def scatter_mm_meta(
    M,
    K,
    N,
    Ms,
    Ks,
    GROUP_SIZE=None,
    TILE_M=None,
    TILE_N=None,
    SPLIT_N=None,
    num_warps=None,
    num_stages=None,
    **extra,
):
    if {TILE_M, TILE_N, SPLIT_N, num_warps, num_stages, GROUP_SIZE} == {None}:
        device_name = torch.cuda.get_device_name()
        meta = get_meta(
            "scatter_mm",
            (M, K, N, Ms, Ks),
            device_name,
            version=(0, torch.float16, 0.5),
        )
        if meta is not None:
            meta.update(**extra)
            return meta
        # The following parameters are optimized for the performance
        # equilibrium points of bsr-dense and dense-dense matrix
        # multiplications when using GPU card NVIDIA GeForce RTX 2060
        # SUPER. For points far from the performance equilibrium
        # points as well as for other GPU cards, the optimal
        # parameters are likely different from what specified below.
        if (M, K, N) == (256,) * 3:
            if (Ms, Ks) == (16, 16):
                SPLIT_N = 1
                TILE_M = 16
                TILE_N = 16
                GROUP_SIZE = 4
                num_stages = 1
                num_warps = 4  # noqa: E225,E231,E702
            elif (Ms, Ks) == (32, 32):
                SPLIT_N = 2
                TILE_M = 32
                TILE_N = 16
                GROUP_SIZE = 4
                num_stages = 1
                num_warps = 4  # noqa: E225,E231,E702
            elif (Ms, Ks) == (64, 64):
                SPLIT_N = 1
                TILE_M = 32
                TILE_N = 32
                GROUP_SIZE = 4
                num_stages = 1
                num_warps = 4  # noqa: E225,E231,E702
            elif (Ms, Ks) == (128, 128):
                SPLIT_N = 1
                TILE_M = 32
                TILE_N = 32
                GROUP_SIZE = 2
                num_stages = 1
                num_warps = 4  # noqa: E225,E231,E702
        elif (M, K, N) == (512,) * 3:
            if (Ms, Ks) == (16, 16):
                SPLIT_N = 8
                TILE_M = 16
                TILE_N = 64
                GROUP_SIZE = 2
                num_stages = 1
                num_warps = 2  # noqa: E225,E231,E702
            elif (Ms, Ks) == (32, 32):
                SPLIT_N = 8
                TILE_M = 32
                TILE_N = 64
                GROUP_SIZE = 4
                num_stages = 1
                num_warps = 2  # noqa: E225,E231,E702
            elif (Ms, Ks) == (64, 64):
                SPLIT_N = 4
                TILE_M = 32
                TILE_N = 128
                GROUP_SIZE = 4
                num_stages = 1
                num_warps = 4  # noqa: E225,E231,E702
            elif (Ms, Ks) == (128, 128):
                SPLIT_N = 8
                TILE_M = 64
                TILE_N = 64
                GROUP_SIZE = 4
                num_stages = 1
                num_warps = 4  # noqa: E225,E231,E702
        elif (M, K, N) == (1024,) * 3:
            if (Ms, Ks) == (16, 16):
                SPLIT_N = 4
                TILE_M = 16
                TILE_N = 128
                GROUP_SIZE = 2
                num_stages = 1
                num_warps = 1  # noqa: E225,E231,E702
            elif (Ms, Ks) == (32, 32):
                SPLIT_N = 8
                TILE_M = 32
                TILE_N = 64
                GROUP_SIZE = 2
                num_stages = 1
                num_warps = 1  # noqa: E225,E231,E702
            elif (Ms, Ks) == (64, 64):
                SPLIT_N = 16
                TILE_M = 64
                TILE_N = 64
                GROUP_SIZE = 4
                num_stages = 1
                num_warps = 2  # noqa: E225,E231,E702
            elif (Ms, Ks) == (128, 128):
                SPLIT_N = 16
                TILE_M = 64
                TILE_N = 64
                GROUP_SIZE = 4
                num_stages = 1
                num_warps = 4  # noqa: E225,E231,E702
            elif (Ms, Ks) == (256, 256):
                SPLIT_N = 16
                TILE_M = 64
                TILE_N = 64
                GROUP_SIZE = 2
                num_stages = 1
                num_warps = 4  # noqa: E225,E231,E702
        elif (M, K, N) == (2048,) * 3:
            if (Ms, Ks) == (16, 16):
                SPLIT_N = 4
                TILE_M = 16
                TILE_N = 128
                GROUP_SIZE = 8
                num_stages = 1
                num_warps = 1  # noqa: E225,E231,E702
            elif (Ms, Ks) == (32, 32):
                SPLIT_N = 4
                TILE_M = 32
                TILE_N = 64
                GROUP_SIZE = 4
                num_stages = 1
                num_warps = 1  # noqa: E225,E231,E702
            elif (Ms, Ks) == (64, 64):
                SPLIT_N = 4
                TILE_M = 64
                TILE_N = 128
                GROUP_SIZE = 4
                num_stages = 1
                num_warps = 4  # noqa: E225,E231,E702
            elif (Ms, Ks) == (128, 128):
                SPLIT_N = 8
                TILE_M = 64
                TILE_N = 64
                GROUP_SIZE = 4
                num_stages = 1
                num_warps = 4  # noqa: E225,E231,E702
            elif (Ms, Ks) == (256, 256):
                SPLIT_N = 4
                TILE_M = 64
                TILE_N = 64
                GROUP_SIZE = 2
                num_stages = 1
                num_warps = 4  # noqa: E225,E231,E702
        elif (M, K, N) == (4096,) * 3:
            if (Ms, Ks) == (16, 16):
                SPLIT_N = 2
                TILE_M = 16
                TILE_N = 256
                GROUP_SIZE = 2
                num_stages = 1
                num_warps = 2  # noqa: E225,E231,E702
            elif (Ms, Ks) == (32, 32):
                SPLIT_N = 2
                TILE_M = 32
                TILE_N = 64
                GROUP_SIZE = 2
                num_stages = 1
                num_warps = 1  # noqa: E225,E231,E702
            elif (Ms, Ks) == (64, 64):
                SPLIT_N = 2
                TILE_M = 64
                TILE_N = 128
                GROUP_SIZE = 2
                num_stages = 1
                num_warps = 4  # noqa: E225,E231,E702

    if SPLIT_N is None:
        # Assume NVIDIA GeForce RTX 2060 SUPER:
        # With the probality of 92% (99.9% when N > 512), the
        # performance will not be worse more than 2% from the
        # performance when using an optimal value.  Otherwise, when N
        # <= 512, using the following heuristics may give upto 15%
        # lower performance.
        SPLIT_N = {
            16: 1,
            32: 2,
            64: 4,
            128: 8,
            256: 16,
            512: 8,
            1024: 16,
            4096: 32,
            8192: 64,
        }.get(N, 16)
        if Ms >= 512 and N >= 2048:
            SPLIT_N = 1
    Ns = N // SPLIT_N
    if TILE_M is None:
        TILE_M = min(64 if Ns < 512 else 32, Ms)
    if TILE_N is None:
        TILE_N = min(64 if Ns < 512 else 32, Ns)
    num_stages = num_stages or 1
    if num_warps is None:
        if min(M, N) > 1024:
            num_warps = {16: 1, 32: 1, 64: 2}.get(Ms, 4)
        elif min(M, N) == 1024:
            num_warps = {16: 1, 32: 1, 64: 2}.get(Ms, 4)
        elif min(M, N) == 256:
            num_warps = {16: 1, 32: 4}.get(Ms, 4)
        else:
            num_warps = {16: 1, 32: 2}.get(Ms, 4)
    GROUP_SIZE = GROUP_SIZE or 4

    assert TILE_M <= Ms, dict(TILE_M=TILE_M, Ms=Ms)
    assert TILE_N <= Ns, dict(TILE_N=TILE_N, Ns=Ns)
    assert Ms <= M, dict(M=M, Ms=Ms)
    assert Ns <= N, dict(N=N, Ns=Ns)
    assert Ks <= K, dict(K=K, Ks=Ks)

    return dict(
        TILE_M=TILE_M,
        TILE_N=TILE_N,
        GROUP_SIZE=GROUP_SIZE,
        num_stages=num_stages,
        num_warps=num_warps,
        SPLIT_N=SPLIT_N,
        **extra,
    )


def bsr_dense_addmm_meta(
    M,
    K,
    N,
    Ms,
    Ks,
    beta,
    alpha,
    SPLIT_N=None,
    GROUP_SIZE_ROW=None,
    num_warps=None,
    num_stages=None,
    sparsity=None,
    dtype=None,
    out_dtype=None,
    _version=0,
    **extra,
):
    # Specifying _version is useful for situations when one wants to
    # discard existing triton kernel tuning results, say, in testing
    # bsr_dense_addmm_meta functionality.
    if dtype is None:
        dtype = torch.float16
    if out_dtype is None:
        out_dtype = dtype
    if sparsity is None:
        sparsity = 0.5
    if {SPLIT_N, num_warps, num_stages, GROUP_SIZE_ROW} == {None}:
        device_name = torch.cuda.get_device_name()
        key = (M, K, N, Ms, Ks, beta == 0, beta == 1, alpha == 1)
        if dtype is out_dtype:
            version_dtype = dtype
        else:
            version_dtype = dtype, out_dtype
        meta = get_meta(
            "bsr_dense_addmm",
            key,
            device_name,
            version=(_version, version_dtype, sparsity),
        )
        if meta is None and sparsity != 0.5:
            meta = get_meta(
                "bsr_dense_addmm",
                key,
                device_name,
                version=(_version, version_dtype, 0.5),
            )
        if meta is None and dtype is not out_dtype:
            meta = get_meta(
                "bsr_dense_addmm", key, device_name, version=(_version, dtype, 0.5)
            )
        if meta is None:
            # find approximate meta such that N % SPLIT_N == 0.
            matching_meta = get_meta(
                "bsr_dense_addmm",
                (*key[:2], "*", *key[3:]),
                device_name,
                version=(_version, version_dtype, 0.5),
            )
            if matching_meta is None and dtype is not out_dtype:
                matching_meta = get_meta(
                    "bsr_dense_addmm",
                    (*key[:2], "*", *key[3:]),
                    device_name,
                    version=(_version, dtype, 0.5),
                )
            for mkey in sorted(matching_meta or {}):
                meta_ = matching_meta[mkey]
                n = mkey[2]
                split_n = meta_["SPLIT_N"]
                c = n // split_n
                if N % c == 0 and n <= N:
                    meta = dict(meta_)
                    meta["SPLIT_N"] = N // c
        if meta is not None:
            meta.update(**extra)
            return meta
        else:
            # see [Computing optimal kernel parameters] in
            # _triton_ops_meta.py for ways to avoid this warning
            # message
            warn_once(
                "bsr_dense_addmm uses non-optimal triton kernel parameters"
                f" for {M=} {K=} {N=} {Ms=}, {Ks=} {beta=} {alpha=} {dtype=} {out_dtype=}"
            )

    SPLIT_N = SPLIT_N or max(N // Ms, 1)
    GROUP_SIZE_ROW = GROUP_SIZE_ROW or 4
    num_stages = num_stages or 1
    num_warps = num_warps or 4
    return dict(
        SPLIT_N=SPLIT_N,
        GROUP_SIZE_ROW=GROUP_SIZE_ROW,
        num_stages=num_stages,
        num_warps=num_warps,
        **extra,
    )


class TensorAsKey:
    """A light-weight wrapper of a tensor that enables storing tensors as
    keys with efficient memory reference based comparison as an
    approximation to data equality based keys.

    Motivation: the hash value of a torch tensor is tensor instance
    based that does not use data equality and makes the usage of
    tensors as keys less useful. For instance, the result of
    ``len({a.crow_indices(), a.crow_indices()})`` is `2`, although,
    the tensor results from `crow_indices` method call are equal, in
    fact, these share the same data storage.
    On the other hand, for efficient caching of tensors we want to
    avoid calling torch.equal that compares tensors item-wise.

    TensorAsKey offers a compromise in that it guarantees key equality
    of tensors that references data in the same storage in the same
    manner and without accessing underlying data. However, this
    approach does not always guarantee correctness. For instance, for
    a complex tensor ``x``, we have ``TensorAsKey(x) ==
    TensorAsKey(x.conj())`` while ``torch.equal(x, x.conj())`` would
    return False.
    """

    def __init__(self, obj):
        def get_tensor_key(obj):
            # Warning: TensorAsKey does not track negative nor
            # conjugate bits of its input object because in the use
            # case of wrapping compressed/plain indices of compressed
            # sparse tensors (that are always integer tensors with
            # non-negative items) these bits are never set. However,
            # when extending the use of TensorAsKey to float or
            # complex tensors, the values of these bits (see is_neg
            # and is_conj methods) must be included in the key as
            # well.
            assert not (obj.dtype.is_floating_point or obj.dtype.is_complex), obj.dtype
            return (
                obj.data_ptr(),
                obj.storage_offset(),
                obj.shape,
                obj.stride(),
                obj.dtype,
            )

        self._obj_ref = weakref.ref(obj)
        if obj.layout is torch.strided:
            self.key = get_tensor_key(obj)
        elif obj.layout in {torch.sparse_csr, torch.sparse_bsr}:
            self.key = (
                get_tensor_key(obj.crow_indices()),
                get_tensor_key(obj.col_indices()),
            )
        elif obj.layout in {torch.sparse_csc, torch.sparse_bsc}:
            self.key = (
                get_tensor_key(obj.ccol_indices()),
                get_tensor_key(obj.row_indices()),
            )
        else:
            raise NotImplementedError(obj.layout)
        self._hash = hash(self.key)

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        if not isinstance(other, TensorAsKey):
            return False
        if self.obj is None or other.obj is None:
            # dead objects always compare unequal unless these are
            # same objects
            return self is other
        return self.key == other.key

    @property
    def obj(self):
        """Return object if alive, otherwise None."""
        return self._obj_ref()


@lru_cache(maxsize=TORCH_SPARSE_BSR_SCATTER_MM_LRU_CACHE_SIZE)
def _bsr_scatter_mm_indices_data(
    indices_format, M, K, N, Ms, Ks, nbatches, SPLIT_N, compressed_sparse_tensor_as_key
):
    bsr = compressed_sparse_tensor_as_key.obj
    assert bsr is not None
    crow_indices, col_indices = bsr.crow_indices(), bsr.col_indices()
    device = crow_indices.device
    indices_dtype = torch.int32

    if indices_format == "bsr_strided_mm_compressed":
        Ns = N // SPLIT_N
        q_offsets_lst = []
        b = torch.arange(SPLIT_N, dtype=indices_dtype, device=device) * Ns
        for m in range(M // Ms):
            r0 = crow_indices[m].item()
            r1 = crow_indices[m + 1].item()
            if r1 == r0:
                continue
            q_offsets_lst.append(
                (col_indices[r0:r1] * (Ks * N)).repeat(SPLIT_N)
                + b.repeat_interleave(r1 - r0)
            )
        q_offsets = torch.cat(q_offsets_lst)
        crow_indices_diff = crow_indices.diff()
        non_zero_row_indices = crow_indices_diff.nonzero()
        a = non_zero_row_indices * (Ms * N)
        r_offsets = (a + b).view(-1)
        c_indices = crow_indices
        # swizzle operation: mm elements with longer sums are computed first:
        nnz_per_row = crow_indices_diff[non_zero_row_indices].repeat_interleave(SPLIT_N)
        nnz_per_row, indices = nnz_per_row.sort(descending=True, stable=True)
        r_offsets = r_offsets[indices]
        return (indices_format, c_indices, r_offsets, q_offsets)

    elif indices_format == "bsr_strided_mm":
        Ns = N // SPLIT_N
        p_offsets_lst = []
        q_offsets_lst = []
        b = torch.arange(SPLIT_N, dtype=indices_dtype, device=device) * Ns
        for m in range(M // Ms):
            r0 = crow_indices[m].item()
            r1 = crow_indices[m + 1].item()
            if r1 == r0:
                continue
            p_offsets_lst.append(
                torch.arange(r0, r1, dtype=indices_dtype, device=device).repeat(SPLIT_N)
            )
            q_offsets_lst.append(
                (col_indices[r0:r1] * (Ks * N)).repeat(SPLIT_N)
                + b.repeat_interleave(r1 - r0)
            )
        q_offsets = torch.cat(q_offsets_lst)
        crow_indices_diff = crow_indices.diff()
        non_zero_row_indices = crow_indices_diff.nonzero()
        a = non_zero_row_indices * (Ms * N)
        r_offsets = (a + b).view(-1)
        c_indices = torch.cat(
            (
                crow_indices[:1],
                torch.cumsum(
                    crow_indices_diff[non_zero_row_indices].repeat_interleave(SPLIT_N),
                    0,
                ),
            )
        )
        p_offsets = torch.cat(p_offsets_lst)
        return (indices_format, c_indices, r_offsets, p_offsets, q_offsets)

    elif indices_format == "scatter_mm":
        Ns = Ms
        c_indices = [0]
        pq_offsets = []
        # todo: eliminate inner for-loops for efficiency
        for b in range(nbatches):
            for m in range(M // Ms):
                r0 = crow_indices[m].item()
                r1 = crow_indices[m + 1].item()
                for n in range(N // Ns):
                    c_indices.append(c_indices[-1] + r1 - r0)
                    for t in range(r1 - r0):
                        p = r0 + t
                        q = (col_indices[p].item() + b * (K // Ks)) * (N // Ns) + n
                        pq_offsets.append([p, q])

        return (
            indices_format,
            torch.tensor(c_indices, dtype=indices_dtype, device=device),
            torch.tensor(pq_offsets, dtype=indices_dtype, device=device),
        )

    else:
        raise ValueError(
            f"Invalid {indices_format=}. Expected bsr_strided_mm_compressed|bsr_strided_mm|scatter_mm"
        )


def bsr_scatter_mm_indices_data(
    bsr, other, indices_format="bsr_strided_mm_compressed", **meta_input
):
    """Computes indices data for :func:`scatter_mm` used in BSR and
    strided tensor matrix multiplication.
    """
    assert bsr.dense_dim() == 0
    assert bsr.ndim == 2  # no batch dims
    blocksize = bsr.values().shape[-2:]
    M, K = bsr.shape
    Ms, Ks = blocksize
    K_, N = other.shape[-2:]
    assert K_ == K
    nbatches = other.shape[:-2].numel()

    meta = scatter_mm_meta(M, K, N, Ms, Ks, **meta_input)
    if "allow_tf32" not in meta_input:
        meta.update(allow_tf32=bsr.dtype in {torch.float16, torch.bfloat16})
    SPLIT_N = meta["SPLIT_N"]
    indices_data = _bsr_scatter_mm_indices_data(
        indices_format, M, K, N, Ms, Ks, nbatches, SPLIT_N, TensorAsKey(bsr)
    )

    if indices_format == "bsr_strided_mm_compressed":
        meta.update(is_compressed=True)
        return indices_data + (meta,)
    elif indices_format == "bsr_strided_mm":
        meta.update(is_compressed=False)
        return indices_data + (meta,)
    else:
        return indices_data


def bsr_scatter_mm(bsr, other, indices_data=None, out=None):
    """BSR @ strided -> strided"""

    assert bsr.ndim == 2
    assert other.ndim >= 2

    Ms, Ks, Ns = bsr.shape[-2], bsr.shape[-1], other.shape[-1]
    blocksize = bsr.values().shape[-2:]

    if indices_data is None:
        indices_data = bsr_scatter_mm_indices_data(
            bsr, other, indices_format="bsr_strided_mm_compressed"
        )

    indices_format = indices_data[0]

    if out is None:
        out = torch.empty(
            (*other.shape[:-2], Ms, Ns), dtype=bsr.dtype, device=bsr.device
        )
    out_shape = out.shape
    out = as1Dbatch(out)

    if bsr._nnz() == 0:
        out.zero_()
    elif indices_format in {"bsr_strided_mm_compressed", "bsr_strided_mm"}:
        out.zero_()
        scatter_mm(bsr.values(), other, indices_data, accumulators=out)
    elif indices_format == "scatter_mm":
        nbatches = other.shape[:-2].numel()
        accumulators = torch.zeros(
            (
                nbatches * Ms // blocksize[0] * Ns // blocksize[0],
                blocksize[0],
                blocksize[0],
            ),
            dtype=bsr.dtype,
            device=bsr.device,
        )
        others = (
            as1Dbatch(other)
            .transpose(-2, -1)
            .view(
                nbatches,
                Ns // blocksize[0],
                blocksize[0],
                Ks // blocksize[1],
                blocksize[1],
            )
            .movedim(
                (3, 1, 4, 2), (1, 2, 3, 4)
            )  # equivalent to .transpose(-3, -2).transpose(-2, -1).transpose(-4, -3)
            .flatten(0, 2)
        )
        scatter_mm(bsr.values(), others, indices_data, accumulators=accumulators)
        out.copy_(
            accumulators.unflatten(
                0, (nbatches, Ms // blocksize[0], Ns // blocksize[0])
            )
            .movedim(
                (1, 2, 3, 4), (3, 1, 4, 2)
            )  # equivalent to .transpose(-4, -3).transpose(-2, -1).transpose(-3, -2)
            .reshape(nbatches, Ns, Ms)
            .transpose(-2, -1)
        )
    else:
        raise NotImplementedError(indices_format)

    return out.view(out_shape)


def _int_bsr_dense_addmm(
    input: torch.Tensor,
    bsr: torch.Tensor,
    dense: torch.Tensor,
    *,
    beta=1,
    alpha=1,
    left_alpha: Optional[torch.Tensor] = None,
    right_alpha: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    skip_checks: bool = False,
    max_grid: Optional[tuple[Optional[int], Optional[int], Optional[int]]] = None,
    meta: Optional[dict] = None,
):
    if out is None and dense.dtype is torch.int8:
        f_name = "_int_bsr_dense_addmm"
        crow_indices = bsr.crow_indices()
        batch_ndim = crow_indices.dim() - 1
        M = bsr.shape[batch_ndim]
        N = dense.shape[-1]
        original_batch_dims_broadcasted = broadcast_batch_dims(f_name, bsr, dense)
        out = torch.empty(
            original_batch_dims_broadcasted + (M, N),
            dtype=torch.int32,
            device=dense.device,
        )
    return bsr_dense_addmm(
        input,
        bsr,
        dense,
        beta=beta,
        alpha=alpha,
        left_alpha=left_alpha,
        right_alpha=right_alpha,
        out=out,
        skip_checks=skip_checks,
        max_grid=max_grid,
        meta=meta,
    )


def bsr_dense_addmm(
    input: torch.Tensor,
    bsr: torch.Tensor,
    dense: torch.Tensor,
    *,
    beta=1,
    alpha=1,
    left_alpha: Optional[torch.Tensor] = None,
    right_alpha: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    skip_checks: bool = False,
    max_grid: Optional[tuple[Optional[int], Optional[int], Optional[int]]] = None,
    meta: Optional[dict] = None,
):
    """Compute

      out = beta * input + left_alpha.reshape(-1, 1) * (alpha * (bsr @ dense)) * right_alpha.reshape(1, -1)

    where left_alpha, right_alpha are (* + 1)-D tensors when
    specified, otherwise, these are treated as tensors filled with
    ones.
    """
    f_name = "bsr_dense_addmm"
    values = bsr.values()
    crow_indices = bsr.crow_indices()
    col_indices = bsr.col_indices()
    batch_ndim = crow_indices.dim() - 1
    M, K = bsr.shape[batch_ndim : batch_ndim + 2]
    blocksize = values.shape[batch_ndim + 1 : batch_ndim + 3]
    N = dense.shape[-1]

    # todo: implement checks

    original_batch_dims_broadcasted = broadcast_batch_dims(f_name, bsr, dense)
    if out is None:
        out = dense.new_empty(original_batch_dims_broadcasted + (M, N))

    if bsr._nnz() == 0 or alpha == 0 or N == 0 or M == 0 or K == 0:
        if beta == 0:
            out.zero_()
        else:
            out.copy_(input)
            if beta != 1:
                out.mul_(beta)
        return out

    left_alpha_is_one = False
    right_alpha_is_one = False
    if left_alpha is None:
        left_alpha_is_one = True
        left_alpha = dense.new_empty(()).expand(
            *original_batch_dims_broadcasted, M, N
        )  # not referenced
    else:
        left_alpha = left_alpha.view(*original_batch_dims_broadcasted, M, 1).expand(
            *original_batch_dims_broadcasted, M, N
        )

    if right_alpha is None:
        right_alpha_is_one = True
        right_alpha = dense.new_empty(()).expand(
            *original_batch_dims_broadcasted, M, N
        )  # not referenced
    else:
        right_alpha = right_alpha.view(*original_batch_dims_broadcasted, 1, N).expand(
            *original_batch_dims_broadcasted, M, N
        )
    assert left_alpha.stride()[-1] == 0
    assert right_alpha.stride()[-2] == 0

    if meta is None:
        sparsity = round(1 - bsr._nnz() * blocksize[0] * blocksize[1] / (M * K), 2)
        meta = bsr_dense_addmm_meta(
            M,
            K,
            N,
            blocksize[0],
            blocksize[1],
            beta,
            alpha,
            sparsity=sparsity,
            dtype=dense.dtype,
            out_dtype=out.dtype,
        )
    out_backup = out

    (
        crow_indices,
        col_indices,
        values,
        input,
        dense,
        left_alpha,
        right_alpha,
        out,
    ) = prepare_inputs(bsr, input, dense, left_alpha, right_alpha, out)

    BM, BK = blocksize
    SPLIT_N = meta.get("SPLIT_N", N // BM)
    BN = N // SPLIT_N

    out_untiled = out
    out = tile_to_blocksize(out, (BM, BN))
    dense = tile_to_blocksize(dense, (BK, BN))
    input = tile_to_blocksize(input, (BM, BN))
    left_alpha = tile_to_blocksize(left_alpha, (BM, BN))
    right_alpha = tile_to_blocksize(right_alpha, (BM, BN))

    # tl.dot supports float16, float32, int32 as accumulator types.
    dot_out_dtype = {
        torch.float16: tl.float32,
        torch.bfloat16: tl.float32,
        torch.float32: tl.float64,
        torch.float64: tl.float64,
        torch.int8: tl.int32,
        torch.int32: tl.int32,
    }[out.dtype]

    n_batches = dense.size(0)
    n_block_rows = crow_indices.size(-1) - 1
    n_block_cols = dense.size(-3)

    full_grid = (n_batches, n_block_cols, n_block_rows)
    if max_grid is not None:
        grid_blocks = tuple(max_grid[:3][::-1]) + (None,) * (3 - len(max_grid[:3]))
    else:
        grid_blocks = None

    tensor_dims_map = {
        values: (0, None, None),
        crow_indices: (0, None, -1),
        col_indices: (0, None, None),
        input: (0, -3, -4),
        dense: (0, -3, None),
        left_alpha: (0, -3, -4),
        right_alpha: (0, -3, -4),
        out: (0, -3, -4),
    }

    assert alpha != 0

    def kernel(grid, *sliced_tensors):
        # pyrefly: ignore [unsupported-operation]
        _bsr_strided_addmm_kernel[grid](
            *ptr_stride_extractor(*sliced_tensors),
            # pyrefly: ignore  # bad-argument-count
            beta,
            alpha,
            # pyrefly: ignore  # bad-keyword-argument, bad-argument-type
            beta_is_one=beta == 1,
            # pyrefly: ignore  # bad-keyword-argument, bad-argument-type
            beta_is_nonzero=beta != 0,
            # pyrefly: ignore  # bad-keyword-argument, bad-argument-type
            alpha_is_one=alpha == 1,
            # pyrefly: ignore  # bad-keyword-argument, bad-argument-type
            left_alpha_is_one=left_alpha_is_one,
            # pyrefly: ignore  # bad-keyword-argument, bad-argument-type
            right_alpha_is_one=right_alpha_is_one,
            # pyrefly: ignore  # bad-keyword-argument, bad-argument-type
            BLOCKSIZE_ROW=BM,
            # pyrefly: ignore  # bad-keyword-argument, bad-argument-type
            BLOCKSIZE_INNER=BK,
            # pyrefly: ignore  # bad-keyword-argument
            BLOCKSIZE_COL=BN,
            # pyrefly: ignore  # bad-keyword-argument
            allow_tf32=dot_out_dtype == tl.float32,
            # pyrefly: ignore  # bad-keyword-argument, bad-argument-type
            acc_dtype=dot_out_dtype,
            **meta,
        )

    launch_kernel(kernel, tensor_dims_map, full_grid, grid_blocks)

    if out.data_ptr() != out_backup.data_ptr():
        # prepare_inputs has made a copy of out, copy its content back
        # to out_backup:
        out_backup.copy_(out_untiled.view(out_backup.shape))

    return out_backup


if has_triton():
    import triton
    import triton.language as tl

    @triton.jit
    def _sampled_addmm_kernel(
        alpha,
        beta,
        IS_BETA_ZERO: tl.constexpr,
        BLOCKSIZE_ROW: tl.constexpr,
        BLOCKSIZE_COL: tl.constexpr,
        k,
        TILE_K: tl.constexpr,
        values_ptr,
        values_batch_stride,
        values_nnz_stride,
        values_row_block_stride,
        values_col_block_stride,
        crow_indices_ptr,
        crow_indices_batch_stride,
        crow_indices_stride,
        col_indices_ptr,
        col_indices_batch_stride,
        col_indices_stride,
        mat1_ptr,
        mat1_batch_stride,
        mat1_tiled_row_stride,
        mat1_tiled_col_stride,
        mat1_row_block_stride,
        mat1_col_block_stride,
        mat2_ptr,
        mat2_batch_stride,
        mat2_tiled_row_stride,
        mat2_tiled_col_stride,
        mat2_row_block_stride,
        mat2_col_block_stride,
        acc_dtype: tl.constexpr,
        allow_tf32: tl.constexpr,
    ):
        batch_pid = tl.program_id(axis=1)
        row_block_pid = tl.program_id(axis=0)

        crow_indices_offset_ptr = (
            crow_indices_ptr
            + crow_indices_batch_stride * batch_pid
            + crow_indices_stride * row_block_pid
        )
        nnz_offset = tl.load(crow_indices_offset_ptr)
        nnz_offset_next = tl.load(crow_indices_offset_ptr + crow_indices_stride)

        # Compute nnz for the row with number row_block_pid.
        # If it is zero, skip the row.
        row_nnz = nnz_offset_next - nnz_offset
        if row_nnz == 0:
            return

        row_block_arange = tl.arange(0, BLOCKSIZE_ROW)
        col_block_arange = tl.arange(0, BLOCKSIZE_COL)

        # Pointers are set to the first block of the current row.
        values_block_ptrs = (
            values_ptr
            + values_batch_stride * batch_pid
            + values_nnz_stride * nnz_offset
            + values_row_block_stride * row_block_arange[:, None]
            + values_col_block_stride * col_block_arange[None, :]
        )

        col_index_nnz_ptr = (
            col_indices_ptr
            + col_indices_batch_stride * batch_pid
            + col_indices_stride * nnz_offset
        )

        # Advance mat1 to the current tiled row, ignore columns.
        mat1_block_ptrs = (
            mat1_ptr
            + mat1_batch_stride * batch_pid
            + mat1_tiled_row_stride * row_block_pid
            + mat1_row_block_stride * row_block_arange[:, None]
        )

        # Advance mat2 in batch and block col dimension.
        mat2_block_ptrs = (
            mat2_ptr
            + mat2_batch_stride * batch_pid
            + mat2_col_block_stride * col_block_arange[None, :]
        )

        k_tile_arange = tl.arange(0, TILE_K)
        for _ in range(row_nnz):
            acc_block = tl.zeros((BLOCKSIZE_ROW, BLOCKSIZE_COL), dtype=acc_dtype)

            # find column block index
            col_block = tl.load(col_index_nnz_ptr)

            for k_tile in range(0, k, TILE_K):
                k_offsets = k_tile + k_tile_arange
                mask_k = k_offsets < k

                mat1_block = tl.load(
                    mat1_block_ptrs + mat1_col_block_stride * k_offsets[None, :],
                    # pyrefly: ignore [index-error]
                    mask=mask_k[None, :],
                    other=0.0,
                )

                mat2_block = tl.load(
                    mat2_block_ptrs
                    + mat2_tiled_col_stride * col_block
                    + mat2_row_block_stride * k_offsets[:, None],
                    # pyrefly: ignore [index-error]
                    mask=mask_k[:, None],
                    other=0.0,
                )

                acc_block += tl.dot(
                    mat1_block, mat2_block, allow_tf32=allow_tf32, out_dtype=acc_dtype
                )

            if IS_BETA_ZERO:
                acc_block *= alpha
            else:
                acc_block = alpha * acc_block + beta * tl.load(values_block_ptrs)

            # write result
            tl.store(values_block_ptrs, acc_block.to(values_ptr.dtype.element_ty))

            # advance val/col_index ptrs to the next block in the row.
            values_block_ptrs += values_nnz_stride
            col_index_nnz_ptr += col_indices_stride

    @triton.jit
    def _bsr_strided_dense_rowspace_kernel(
        # values prologue
        values_ptr,
        values_batch_stride,
        values_nnz_stride,
        values_row_block_stride,
        values_col_block_stride,
        # values epilogue
        # crow_indices prologue
        crow_indices_ptr,
        crow_indices_batch_stride,
        crow_indices_stride,
        # crow_indices epilogue
        # col_indices prologue
        col_indices_ptr,
        col_indices_batch_stride,
        col_indices_stride,
        # col_indices epilogue
        # dense prologue
        dense_ptr,
        dense_batch_stride,
        dense_tiled_row_stride,
        dense_tiled_col_stride,
        dense_row_block_stride,
        dense_col_block_stride,
        # dense epilogue
        # output prologue
        output_ptr,
        output_batch_stride,
        output_tiled_row_stride,
        output_tiled_col_stride,
        output_row_block_stride,
        output_col_block_stride,
        # output epilogue
        #
        # gh-113754: Always keep all constexpr arguments at the end of
        # triton kernel arguments list because with triton 2.1 or
        # earlier non-contiguous outputs will corrupt CUDA state due
        # to a triton bug (fixed in openai/triton#2262).
        BLOCKSIZE_ROW: tl.constexpr,
        BLOCKSIZE_COL: tl.constexpr,
        acc_dtype: tl.constexpr,
        allow_tf32: tl.constexpr,
        GROUP_SIZE_ROW: tl.constexpr,
    ):
        batch_pid = tl.program_id(axis=2)
        row_block_pid = tl.program_id(axis=0)
        col_block_pid = tl.program_id(axis=1)
        n_block_rows = tl.num_programs(axis=0)
        n_block_cols = tl.num_programs(axis=1)

        row_block_pid, col_block_pid = tl.swizzle2d(
            row_block_pid, col_block_pid, n_block_rows, n_block_cols, GROUP_SIZE_ROW
        )

        crow_indices_offset_ptr = (
            crow_indices_ptr
            + crow_indices_batch_stride * batch_pid
            + crow_indices_stride * row_block_pid
        )
        nnz_offset = tl.load(crow_indices_offset_ptr)
        nnz_offset_next = tl.load(crow_indices_offset_ptr + crow_indices_stride)

        # Compute nnz for the row with number row_block_pid.
        # If it is zero, skip the row.
        row_nnz = nnz_offset_next - nnz_offset
        if row_nnz == 0:
            return

        row_block_arange = tl.arange(0, BLOCKSIZE_ROW)
        col_block_arange = tl.arange(0, BLOCKSIZE_COL)

        # Pointers are set to the first block of the current row.
        values_block_ptrs = (
            values_ptr
            + values_batch_stride * batch_pid
            + values_nnz_stride * nnz_offset
            + values_row_block_stride * row_block_arange[:, None]
            + values_col_block_stride * col_block_arange[None, :]
        )

        # NOTE: dense is advanced into all dimensions but the tiled row one.
        # That will be advanced in the loop according to values in col_indices.
        dense_block_ptrs = (
            dense_ptr
            + dense_batch_stride * batch_pid
            + dense_tiled_col_stride * col_block_pid
            + dense_row_block_stride * col_block_arange[:, None]
            + dense_col_block_stride * row_block_arange[None, :]
        )

        # Pointers are set to exact write-to locations
        output_ptrs = (
            output_ptr
            + output_batch_stride * batch_pid
            + output_tiled_row_stride * row_block_pid
            + output_tiled_col_stride * col_block_pid
            + output_row_block_stride * row_block_arange[:, None]
            + output_col_block_stride * row_block_arange[None, :]
        )

        # Set pointer to the first nonzero element in the current row
        col_index_nnz_ptr = (
            col_indices_ptr
            + col_indices_batch_stride * batch_pid
            + col_indices_stride * nnz_offset
        )

        output_acc_block = tl.zeros((BLOCKSIZE_ROW, BLOCKSIZE_COL), dtype=acc_dtype)
        for _ in range(row_nnz):
            values_block = tl.load(values_block_ptrs)

            # find which row of dense needs to get loaded
            # for multiplication with values_block.
            dense_row_idx = tl.load(col_index_nnz_ptr)
            dense_block = tl.load(
                dense_block_ptrs + dense_tiled_row_stride * dense_row_idx
            )

            # do block mm
            output_acc_block += tl.dot(
                values_block, dense_block, allow_tf32=allow_tf32, out_dtype=acc_dtype
            )

            # move val/col_index ptrs to the next block in the row
            values_block_ptrs += values_nnz_stride
            col_index_nnz_ptr += col_indices_stride

        # write back the result
        tl.store(output_ptrs, output_acc_block.to(output_ptr.dtype.element_ty))

    def _run_sampled_addmm_kernel(
        alpha,
        beta,
        is_beta_zero,
        blocksize,
        k,
        tile_k,
        values,
        crow_indices,
        col_indices,
        mat1,
        mat2,
        max_grid,
    ):
        n_batches = values.size(0)
        n_block_rows = crow_indices.size(-1) - 1

        full_grid = (n_batches, n_block_rows)
        if max_grid is not None:
            grid_blocks = tuple(max_grid[:2][::-1]) + (None,) * (2 - len(max_grid[:2]))
        else:
            grid_blocks = None
        tensor_dims_map = {
            values: (0, None),
            crow_indices: (0, -1),
            col_indices: (0, None),
            mat1: (0, -4),
            mat2: (0, None),
        }
        if values.dtype in (torch.half, torch.bfloat16):
            acc_dtype = tl.float32
            allow_tf32 = True
        else:
            acc_dtype = tl.float64
            allow_tf32 = False

        def kernel(grid, *sliced_tensors):
            _sampled_addmm_kernel[grid](
                alpha,
                beta,
                is_beta_zero,
                *blocksize,
                # pyrefly: ignore  # bad-argument-count
                k,
                tile_k,
                *ptr_stride_extractor(*sliced_tensors),
                # pyrefly: ignore  # bad-keyword-argument, bad-argument-type
                acc_dtype=acc_dtype,
                # pyrefly: ignore  # bad-keyword-argument, bad-argument-type
                allow_tf32=allow_tf32,
                # pyrefly: ignore  # unexpected-keyword
                num_stages=1,
                # pyrefly: ignore  # unexpected-keyword
                num_warps=4,
            )

        launch_kernel(kernel, tensor_dims_map, full_grid, grid_blocks)

    def sampled_addmm(
        input: torch.Tensor,
        mat1: torch.Tensor,
        mat2: torch.Tensor,
        *,
        beta=1.0,
        alpha=1.0,
        out: Optional[torch.Tensor] = None,
        skip_checks: bool = False,
        max_grid: Optional[tuple[Optional[int], Optional[int], Optional[int]]] = None,
    ):
        f_name = "sampled_addmm"

        check_bsr_layout(f_name, input)
        input_broadcasted = broadcast_batch_dims_bsr(f_name, input, mat1, mat2)

        if not skip_checks:
            check_device(f_name, mat1, input.device)
            check_device(f_name, mat2, input.device)
            if beta != 0.0 and input.dtype is torch.bool:
                check(
                    False,
                    f"{f_name}(): having beta == {beta} not equal to 0.0 with boolean mask is not allowed.",
                )
            if input.dtype is not torch.bool:
                check_dtype(f_name, mat1, input.dtype)
                check_dtype(f_name, mat2, input.dtype)
            else:
                check_dtype(f_name, mat1, mat2.dtype)
            check_mm_compatible_shapes(f_name, mat1, mat2)
            if out is not None:
                check_bsr_layout(f_name, out)
                check_device(f_name, out, mat1.device)
                check_dtype(f_name, out, input.dtype)
                check(
                    out.shape == input_broadcasted.shape and out._nnz() == input._nnz(),
                    f"{f_name}(): Expects `out` to be of shape {input_broadcasted.shape} "
                    f"and with nnz equal to {input_broadcasted._nnz()} "
                    f"but got out.shape = {out.shape} and out.nnz = {out._nnz()}",
                )

        if out is None:
            out = input_broadcasted.to(mat1.dtype, copy=True)
        else:
            out.copy_(input_broadcasted)

        if out.numel() == 0 or out._nnz() == 0:
            return out

        blocksize = out.values().shape[-2:]
        k = mat1.size(-1)

        # NOTE: (m, 0) @ (0, n) == zeros(m, n)
        if alpha == 0.0 or k == 0:
            out.values().mul_(beta)
            return out

        # prepare inputs by reshaping them to be kernel-compatible
        out_backup = out
        crow_indices, col_indices, values, mat1, mat2 = prepare_inputs(out, mat1, mat2)

        mat1 = tile_to_blocksize(mat1, (blocksize[0], k))
        mat2 = tile_to_blocksize(mat2, (k, blocksize[1]))
        tile_k = max(*blocksize)

        _run_sampled_addmm_kernel(
            alpha,
            beta,
            beta == 0.0,
            blocksize,
            k,
            tile_k,
            values,
            crow_indices,
            col_indices,
            mat1,
            mat2,
            max_grid,
        )

        # If nnz x block strides are not the same in out_backup.values and values,
        # it means that out_backup.values and values are not the views of each other,
        # so we have to copy.
        if out_backup.values().stride()[-3:] != values.stride()[-3:]:
            out_backup.values().copy_(values.reshape(out_backup.values().shape))
        return out_backup

    def bsr_dense_mm(
        bsr: torch.Tensor,
        dense: torch.Tensor,
        *,
        out: Optional[torch.Tensor] = None,
        skip_checks: bool = False,
        max_grid: Optional[tuple[Optional[int], Optional[int], Optional[int]]] = None,
        meta: Optional[dict] = None,
    ):
        f_name = "bsr_dense_mm"
        m, _kl = bsr.shape[-2:]
        if not skip_checks:
            check_bsr_layout(f_name, bsr)
            check_device(f_name, bsr, dense.device)
            check_dtype(f_name, bsr, dense.dtype, (torch.int8,))
            check_mm_compatible_shapes(f_name, bsr, dense)

            n = dense.size(-1)
            row_block, col_block = bsr.values().shape[-2:]
            check_blocksize(f_name, (row_block, col_block))
            check(
                not n % 16,
                f"{f_name}(): dense.size(-1) == {n} should be divisible by 16",
            )
        else:
            _kr, n = dense.shape[-2:]

        original_batch_dims_broadcasted = broadcast_batch_dims(f_name, bsr, dense)

        if out is not None and not skip_checks:
            expected_out_shape = original_batch_dims_broadcasted + (m, n)
            check(
                out.shape == expected_out_shape,
                "bsr_dense_mm(): `out` argument has wrong shape, "
                f"expected {expected_out_shape}, but got {out.shape}.",
            )
            check(
                out.is_contiguous() or out.transpose(-2, -1).is_contiguous(),
                "bsr_dense_mm(): only row-major/col-major `out` arguments are supported, "
                "i.e. (out.is_contiguous() or out.transpose(-2, -1).is_contiguous()) "
                "should be True.",
            )

        # Allocate out
        if out is None:
            out = dense.new_empty(original_batch_dims_broadcasted + (m, n))

        # Short circuit if lhs is zero
        if bsr._nnz() == 0:
            return out.zero_()

        # with beta==0, addmm ignores input content, so we can use out
        # as a placeholder for input because their shapes match:
        return bsr_dense_addmm(out, bsr, dense, alpha=1, beta=0, out=out)

    @triton.jit
    def _bsr_softmax_kernel(
        crow_indices_ptr,
        crow_indices_batch_stride,
        crow_indices_stride,
        values_ptr,
        values_batch_stride,
        values_row_block_stride,
        values_nnz_col_block_stride,
        row_block,
        col_block,
        MAX_ROW_NNZ: tl.constexpr,
        TILE: tl.constexpr,
    ):
        batch_pid = tl.program_id(axis=2)
        row_block_offset_pid = tl.program_id(axis=1)
        row_block_pid = tl.program_id(axis=0)

        crow_indices_offset_ptr = (
            crow_indices_ptr
            + crow_indices_batch_stride * batch_pid
            + crow_indices_stride * row_block_pid
        )
        nnz_offset = tl.load(crow_indices_offset_ptr)
        nnz_offset_next = tl.load(crow_indices_offset_ptr + crow_indices_stride)

        # Compute nnz for the row with number row_block_pid.
        # If it is zero, skip the row.
        row_nnz = nnz_offset_next - nnz_offset
        if row_nnz == 0:
            return

        row_arange = tl.arange(0, TILE)
        mask = row_arange < row_nnz * col_block

        curr_row_values_ptrs = (
            values_ptr
            + values_batch_stride * batch_pid
            + values_row_block_stride * row_block_offset_pid
            + nnz_offset * col_block
        )

        # find max in the row
        row_tile = tl.load(
            curr_row_values_ptrs + row_arange, mask=mask, other=-float("inf")
        ).to(tl.float32)
        max_row_value = tl.max(row_tile, axis=0)
        for _ in range(TILE, MAX_ROW_NNZ, TILE):
            row_arange += TILE
            mask = row_arange < row_nnz * col_block
            row_tile = tl.load(
                curr_row_values_ptrs + row_arange, mask=mask, other=-float("inf")
            ).to(tl.float32)
            curr_max_row_value = tl.max(row_tile, axis=0)
            max_row_value = tl.where(
                max_row_value > curr_max_row_value, max_row_value, curr_max_row_value
            )

        # find denominator for stable softmax
        num = tl.exp(row_tile - max_row_value)
        denom = tl.sum(num, axis=0)
        for _ in range(TILE, MAX_ROW_NNZ, TILE):
            row_arange -= TILE
            mask = row_arange < row_nnz * col_block
            row_tile = tl.load(
                curr_row_values_ptrs + row_arange, mask=mask, other=-float("inf")
            ).to(tl.float32)
            num = tl.exp(row_tile - max_row_value)
            denom += tl.sum(num, axis=0)

        # populate output
        tl.store(
            curr_row_values_ptrs + row_arange,
            (num / denom).to(values_ptr.dtype.element_ty),
            mask=mask,
        )
        for _ in range(TILE, MAX_ROW_NNZ, TILE):
            row_arange += TILE
            mask = row_arange < row_nnz * col_block
            row_tile = tl.load(
                curr_row_values_ptrs + row_arange, mask=mask, other=-float("inf")
            ).to(tl.float32)
            num = tl.exp(row_tile - max_row_value)
            tl.store(
                curr_row_values_ptrs + row_arange,
                (num / denom).to(values_ptr.dtype.element_ty),
                mask=mask,
            )

    def bsr_softmax(input, max_row_nnz=None):
        f_name = "bsr_softmax"

        check_bsr_layout(f_name, input)
        check_dtype(f_name, input, input.dtype)

        if input._nnz() == 0 or input.numel() == 0:
            return input.clone()

        m, n = input.shape[-2:]
        nnz = input._nnz()
        row_block, col_block = input.values().shape[-2:]

        if max_row_nnz is None:
            max_row_nnz = triton.next_power_of_2(n)
        else:
            max_row_nnz = triton.next_power_of_2(max_row_nnz)

        crow_indices = input.crow_indices().unsqueeze(0).flatten(0, -2)
        # reshape values from
        # (b1, ..., bn, nnz, row_block, col_block) to
        # (b1 * ... * bn, row_block, nnz * col_block).
        # This simplifies batch dim manipulation and unlocks
        # the possibility to access all nnzs in any given row.
        if input.values().transpose(-3, -2).is_contiguous():
            # Need to clone to avoid `contiguous` returning a view.
            values = input.values().clone()
        else:
            values = input.values()
        values = (
            values.transpose(-3, -2)
            .contiguous()
            .unsqueeze(0)
            .flatten(0, -4)
            .reshape(-1, row_block, nnz * col_block)
        )
        full_grid = (values.shape[0], row_block, m // row_block)
        grid_blocks = None
        tensor_dims_map = {
            # We span nnz number of blocks, not nnz + 1,
            # hence crow_indices[..., :-1]
            crow_indices[..., :-1]: (0, None, -1),
            values: (0, None, None),
        }

        def kernel(grid, *sliced_tensors):
            _bsr_softmax_kernel[grid](
                *ptr_stride_extractor(*sliced_tensors),
                # pyrefly: ignore  # bad-argument-count
                row_block,
                col_block,
                max_row_nnz,
                # Triton's max numel is bounded by 2 ** 17.
                min(2**17, max_row_nnz),
            )

        launch_kernel(kernel, tensor_dims_map, full_grid, grid_blocks)

        values = (
            values.reshape(-1, row_block, nnz, col_block)
            .transpose(-3, -2)
            .reshape(*input.values().shape)
        )

        return torch.sparse_compressed_tensor(
            input.crow_indices().clone(),
            input.col_indices().clone(),
            values,
            size=input.shape,
            layout=input.layout,
        )

    def _scaled_dot_product_attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        dropout_p: float = 0.0,
        is_causal: bool = False,
        scale: Optional[float] = None,
    ):
        f_name = "_scaled_dot_product_attention"
        check(not is_causal, f"{f_name}(): is_causal == True is not supported.")
        check(attn_mask is not None, f"{f_name}(): attn_mask == None is not supported.")
        assert attn_mask is not None

        check(
            attn_mask.layout == torch.sparse_bsr,
            f"{f_name}(): "
            f"attn_mask.layout must be {torch.sparse_bsr}, but got "
            f"attn_mask.layout == {attn_mask.layout}.",
        )

        check_device(f_name, key, query.device)
        check_device(f_name, value, query.device)
        check_device(f_name, attn_mask, query.device)

        check_dtype(f_name, key, query.dtype)
        check_dtype(f_name, value, query.dtype)
        if attn_mask.dtype is not torch.bool:
            check_dtype(f_name, attn_mask, query.dtype)

        # pyrefly: ignore [not-callable]
        sdpa = sampled_addmm(
            attn_mask, query, key.transpose(-2, -1), beta=0.0, skip_checks=False
        )
        if scale is None and query.size(-1) == 0 or scale == 0.0:
            check(
                False,
                f"{f_name}(): current value of scale == {scale} "
                "results in division by zero.",
            )
        scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
        sdpa.values().mul_(scale_factor)
        # pyrefly: ignore [not-callable]
        sdpa = bsr_softmax(sdpa)
        torch.nn.functional.dropout(sdpa.values(), p=dropout_p, inplace=True)
        # pyrefly: ignore [not-callable]
        sdpa = bsr_dense_mm(sdpa, value)
        return sdpa

    @triton.jit
    def _scatter_mm2_kernel(
        M: tl.constexpr,
        K: tl.constexpr,
        N: tl.constexpr,
        blocks_ptr,
        blocks_stride_P,
        blocks_stride_M,
        blocks_stride_K,
        others_ptr,
        others_stride_Q,
        others_stride_K,
        others_stride_N,
        accumulators_ptr,
        accumulators_stride_R,
        accumulators_stride_M,
        accumulators_stride_N,
        pq_offsets_ptr,
        pq_offsets_stride,
        pq_ptr,
        pq_stride_T,
        pq_stride_1,
        dot_out_dtype: tl.constexpr,
        TILE_M: tl.constexpr,
        TILE_N: tl.constexpr,
        allow_tf32: tl.constexpr,
    ):
        Ms = M // TILE_M

        pid_t = tl.program_id(axis=0)

        pid = tl.program_id(axis=1)
        pid_m = pid // Ms
        pid_n = pid % Ms

        rm = pid_m * TILE_M + tl.arange(0, TILE_M)
        rn = pid_n * TILE_N + tl.arange(0, TILE_N)
        rk = tl.arange(0, K)

        A_ptr = blocks_ptr + (
            rm[:, None] * blocks_stride_M + rk[None, :] * blocks_stride_K
        )
        B_ptr = others_ptr + (
            rk[:, None] * others_stride_K + rn[None, :] * others_stride_N
        )

        g0 = tl.load(pq_offsets_ptr + pid_t * pq_offsets_stride)
        g1 = tl.load(pq_offsets_ptr + (pid_t + 1) * pq_offsets_stride)

        if g0 == g1:
            return

        acc_block = tl.zeros((TILE_M, TILE_N), dtype=dot_out_dtype)

        for i in range(g0, g1):
            p = tl.load(pq_ptr + i * pq_stride_T)
            q = tl.load(pq_ptr + i * pq_stride_T + pq_stride_1)
            A = tl.load(A_ptr + p * blocks_stride_P)
            B = tl.load(B_ptr + q * others_stride_Q)
            acc_block += tl.dot(A, B, out_dtype=dot_out_dtype, allow_tf32=allow_tf32)

        C_ptr = (
            accumulators_ptr
            + pid_t * accumulators_stride_R
            + (
                rm[:, None] * accumulators_stride_M
                + rn[None, :] * accumulators_stride_N
            )
        )
        tl.store(C_ptr, acc_block.to(accumulators_ptr.dtype.element_ty))

    def _scatter_mm2(
        blocks: torch.Tensor,
        others: torch.Tensor,
        pq_offsets: torch.Tensor,
        pq_indices: torch.Tensor,
        accumulators: torch.Tensor,
    ):
        _P, M, K = blocks.shape
        _Q, _, N = others.shape

        meta = dict(
            TILE_M=max(16, M // 4), TILE_N=max(16, N // 4), num_stages=1, num_warps=2
        )

        def grid(META):
            return (
                pq_offsets.shape[0] - 1,
                triton.cdiv(M, META["TILE_M"]) * triton.cdiv(N, META["TILE_N"]),
                1,
            )

        dot_out_dtype = {
            torch.float16: tl.float32,
            torch.bfloat16: tl.float32,
            torch.float32: tl.float64,
            torch.float64: tl.float64,
        }[accumulators.dtype]
        if "allow_tf32" not in meta:
            meta.update(allow_tf32=dot_out_dtype == tl.float32)
        _scatter_mm2_kernel[grid](
            # pyrefly: ignore  # bad-argument-type
            M,
            # pyrefly: ignore  # bad-argument-type
            K,
            # pyrefly: ignore  # bad-argument-type
            N,
            blocks,
            blocks.stride(0),
            blocks.stride(1),
            blocks.stride(2),
            others,
            others.stride(0),
            others.stride(1),
            others.stride(2),
            accumulators,
            accumulators.stride(0),
            accumulators.stride(1),
            accumulators.stride(2),
            pq_offsets,
            pq_offsets.stride(0),
            pq_indices,
            pq_indices.stride(0),
            pq_indices.stride(1),
            # pyrefly: ignore  # bad-argument-type
            dot_out_dtype=dot_out_dtype,
            # pyrefly: ignore  # bad-argument-type
            **meta,
        )

    @triton.jit
    def _scatter_mm6_kernel(
        nbatches,
        Ms,
        Ks: tl.constexpr,
        N,
        blocks_ptr,
        blocks_stride_P,
        blocks_stride_M,
        blocks_stride_K,
        others_ptr,
        others_stride_B,
        others_stride_K,
        others_stride_N,
        accumulators_ptr,
        accumulators_stride_B,
        accumulators_stride_M,
        accumulators_stride_N,
        c_indices_ptr,
        r_offsets_ptr,
        p_offsets_ptr,
        q_offsets_ptr,
        is_compressed: tl.constexpr,
        dot_out_dtype: tl.constexpr,
        SPLIT_N: tl.constexpr,
        TILE_M: tl.constexpr,
        TILE_N: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
        allow_tf32: tl.constexpr,
    ):
        Ns = N // SPLIT_N
        BLOCKS_M = Ms // TILE_M
        BLOCKS_N = Ns // TILE_N

        pid_t_ = tl.program_id(axis=0)
        pid = tl.program_id(axis=1)
        pid_b = pid_t_ % nbatches
        pid_t = pid_t_ // nbatches

        num_pid_in_group = GROUP_SIZE * BLOCKS_N
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE
        group_size_m = min(BLOCKS_M - first_pid_m, GROUP_SIZE)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        rm = pid_m * TILE_M + tl.arange(0, TILE_M)
        rn = pid_n * TILE_N + tl.arange(0, TILE_N)
        rk = tl.arange(0, Ks)
        A_ptr = blocks_ptr + (
            rm[:, None] * blocks_stride_M + rk[None, :] * blocks_stride_K
        )
        B_ptr = (
            others_ptr
            + pid_b * others_stride_B
            + (rk[:, None] * others_stride_K + rn[None, :] * others_stride_N)
        )

        # When is_compressed is True, r is the only variable that
        # depends on pid_t. This property allows sorting r values
        # before calling the kernel. The sorting of r is equivalent to
        # defining swizzle operator outside of the kernel.
        r = tl.load(r_offsets_ptr + pid_t)

        if is_compressed:
            m = (r // N) // Ms
            n = (r % N) // Ns
            r0 = tl.load(c_indices_ptr + m)
            r1 = tl.load(c_indices_ptr + m + 1)
            g0 = n * r1 + (SPLIT_N - n) * r0
            nnz = r1 - r0
        else:
            g0 = tl.load(c_indices_ptr + pid_t)
            g1 = tl.load(c_indices_ptr + pid_t + 1)
            nnz = g1 - g0

        q_ptr = q_offsets_ptr + g0
        acc_block = tl.zeros((TILE_M, TILE_N), dtype=dot_out_dtype)

        if is_compressed:
            A_ptr += r0 * blocks_stride_P  # type: ignore[possibly-undefined]
            for _ in range(nnz):
                q = tl.load(q_ptr)
                B = tl.load(B_ptr + q)
                A = tl.load(A_ptr)
                acc_block += tl.dot(
                    A, B, out_dtype=dot_out_dtype, allow_tf32=allow_tf32
                )
                A_ptr += blocks_stride_P
                q_ptr += 1
        else:
            p_ptr = p_offsets_ptr + g0
            for _ in range(nnz):
                q = tl.load(q_ptr)
                B = tl.load(B_ptr + q)
                p = tl.load(p_ptr)
                A = tl.load(A_ptr + p * blocks_stride_P)
                p_ptr += 1
                q_ptr += 1
                acc_block += tl.dot(
                    A, B, out_dtype=dot_out_dtype, allow_tf32=allow_tf32
                )

        C_ptr = (
            accumulators_ptr
            + r
            + pid_b * accumulators_stride_B
            + (
                rm[:, None] * accumulators_stride_M
                + rn[None, :] * accumulators_stride_N
            )
        )
        tl.store(C_ptr, acc_block.to(accumulators_ptr.dtype.element_ty))

    def _scatter_mm6(
        blocks: torch.Tensor,
        others: torch.Tensor,
        c_indices: torch.Tensor,
        r_offsets: torch.Tensor,
        p_offsets: torch.Tensor,
        q_offsets: torch.Tensor,
        meta: dict,
        accumulators: torch.Tensor,
        force_contiguous: bool = True,
    ):
        SPLIT_N = meta["SPLIT_N"]
        _P, Ms, Ks = blocks.shape
        B, _K, N = others.shape
        B_, _M, N_ = accumulators.shape
        assert N_ == N
        Ns = N // SPLIT_N
        assert B_ == B

        def grid(META):
            return (
                r_offsets.shape[0] * B,
                triton.cdiv(Ms, META["TILE_M"]) * triton.cdiv(Ns, META["TILE_N"]),
            )

        dot_out_dtype = {
            torch.float16: tl.float32,
            torch.bfloat16: tl.float32,
            torch.float32: tl.float64,
            torch.float64: tl.float64,
        }[accumulators.dtype]
        if "allow_tf32" not in meta:
            meta.update(allow_tf32=dot_out_dtype == tl.float32)

        assert c_indices.stride(0) == 1
        assert r_offsets.stride(0) == 1
        assert p_offsets.stride(0) == 1
        assert q_offsets.stride(0) == 1

        # Re non-contiguous tensor arguments. Sometimes triton kernel
        # launches may fail with
        #
        #   RuntimeError: Triton Error [CUDA]: an illegal memory access was encountered
        #
        # that appears to be case when the size of a non-contiguous
        # tensor argument is larger than a certain threshold. Could
        # this be related to shared memory or L1 cache size of a GPU
        # card? In anycase, ensuring that tensor arguments are
        # contiguous seems to avoid the above exception. So, in the
        # following we'll always convert tensor arguments to
        # C-contiguous tensors.

        if force_contiguous:
            blocks = blocks.contiguous()
            others = others.contiguous()
            if not accumulators.is_contiguous():
                accumulators_ = accumulators.contiguous()
            else:
                accumulators_ = accumulators
        else:
            accumulators_ = accumulators

        _scatter_mm6_kernel[grid](
            B,
            Ms,
            # pyrefly: ignore  # bad-argument-type
            Ks,
            N,
            blocks,
            blocks.stride(0),
            blocks.stride(1),
            blocks.stride(2),
            others,
            others.stride(0),
            others.stride(1),
            others.stride(2),
            accumulators_,
            accumulators_.stride(0),
            accumulators_.stride(1),
            accumulators_.stride(2),
            c_indices,
            r_offsets,
            p_offsets,
            q_offsets,
            # pyrefly: ignore  # bad-argument-type
            dot_out_dtype=dot_out_dtype,
            **meta,
        )

        if force_contiguous and not accumulators.is_contiguous():
            accumulators.copy_(accumulators_)

    @triton.jit
    def _bsr_strided_addmm_kernel(
        # values prologue
        values_ptr,
        values_batch_stride,
        values_nnz_stride,
        values_row_block_stride,
        values_col_block_stride,
        # values epilogue
        # crow_indices prologue
        crow_indices_ptr,
        crow_indices_batch_stride,
        crow_indices_stride,
        # crow_indices epilogue
        # col_indices prologue
        col_indices_ptr,
        col_indices_batch_stride,
        col_indices_stride,
        # col_indices epilogue
        # input prologue
        input_ptr,
        input_batch_stride,
        input_tiled_row_stride,
        input_tiled_col_stride,
        input_row_block_stride,
        input_col_block_stride,
        # input epilogue
        # dense prologue
        dense_ptr,
        dense_batch_stride,
        dense_tiled_row_stride,
        dense_tiled_col_stride,
        dense_row_block_stride,
        dense_col_block_stride,
        # dense epilogue
        # left_alpha prologue
        left_alpha_ptr,
        left_alpha_batch_stride,
        left_alpha_tiled_row_stride,
        left_alpha_tiled_col_stride: tl.constexpr,
        left_alpha_row_block_stride,
        left_alpha_col_block_stride: tl.constexpr,
        # left_alpha epilogue
        # right_alpha prologue
        right_alpha_ptr,
        right_alpha_batch_stride,
        right_alpha_tiled_row_stride: tl.constexpr,
        right_alpha_tiled_col_stride,
        right_alpha_row_block_stride: tl.constexpr,
        right_alpha_col_block_stride,
        # right_alpha epilogue
        # output prologue
        output_ptr,
        output_batch_stride,
        output_tiled_row_stride,
        output_tiled_col_stride,
        output_row_block_stride,
        output_col_block_stride,
        # output epilogue
        beta,
        alpha,
        beta_is_one: tl.constexpr,
        beta_is_nonzero: tl.constexpr,
        alpha_is_one: tl.constexpr,
        left_alpha_is_one: tl.constexpr,
        right_alpha_is_one: tl.constexpr,
        BLOCKSIZE_ROW: tl.constexpr,
        BLOCKSIZE_COL: tl.constexpr,
        BLOCKSIZE_INNER: tl.constexpr,
        acc_dtype: tl.constexpr,
        allow_tf32: tl.constexpr,
        GROUP_SIZE_ROW: tl.constexpr,
        SPLIT_N: tl.constexpr,
    ):
        # left/right_alpha tensors are originally (* + 1)-dimensional
        assert left_alpha_tiled_col_stride == 0
        assert left_alpha_col_block_stride == 0
        assert right_alpha_tiled_row_stride == 0
        assert right_alpha_row_block_stride == 0

        batch_pid = tl.program_id(axis=2)
        row_block_pid = tl.program_id(axis=0)
        col_block_pid = tl.program_id(axis=1)
        n_block_rows = tl.num_programs(axis=0)
        n_block_cols = tl.num_programs(axis=1)

        row_block_pid, col_block_pid = tl.swizzle2d(
            row_block_pid, col_block_pid, n_block_rows, n_block_cols, GROUP_SIZE_ROW
        )

        crow_indices_offset_ptr = (
            crow_indices_ptr
            + crow_indices_batch_stride * batch_pid
            + crow_indices_stride * row_block_pid
        )
        nnz_offset = tl.load(crow_indices_offset_ptr)
        nnz_offset_next = tl.load(crow_indices_offset_ptr + crow_indices_stride)

        # Compute nnz for the row with number row_block_pid.
        row_nnz = nnz_offset_next - nnz_offset

        row_block_arange = tl.arange(0, BLOCKSIZE_ROW)
        inner_block_arange = tl.arange(0, BLOCKSIZE_INNER)
        col_block_arange = tl.arange(0, BLOCKSIZE_COL)

        # Pointers are set to the first block of the current row.
        values_block_ptrs = (
            values_ptr
            + values_batch_stride * batch_pid
            + values_nnz_stride * nnz_offset
            + values_row_block_stride * row_block_arange[:, None]
            + values_col_block_stride * inner_block_arange[None, :]
        )

        # NOTE: dense is advanced into all dimensions but the tiled row one.
        # That will be advanced in the loop according to values in col_indices.
        dense_block_ptrs = (
            dense_ptr
            + dense_batch_stride * batch_pid
            + dense_tiled_col_stride * col_block_pid
            + dense_row_block_stride * inner_block_arange[:, None]
            + dense_col_block_stride * col_block_arange[None, :]
        )

        # Pointers are set to exact write-to locations
        output_ptrs = (
            output_ptr
            + output_batch_stride * batch_pid
            + output_tiled_row_stride * row_block_pid
            + output_tiled_col_stride * col_block_pid
            + output_row_block_stride * row_block_arange[:, None]
            + output_col_block_stride * col_block_arange[None, :]
        )

        # Set pointer to the first nonzero element in the current row
        col_index_nnz_ptr = (
            col_indices_ptr
            + col_indices_batch_stride * batch_pid
            + col_indices_stride * nnz_offset
        )

        output_acc_block = tl.zeros((BLOCKSIZE_ROW, BLOCKSIZE_COL), dtype=acc_dtype)

        for _ in range(row_nnz):
            values_block = tl.load(values_block_ptrs)

            # find which row of dense needs to get loaded
            # for multiplication with values_block.
            dense_row_idx = tl.load(col_index_nnz_ptr)
            dense_block = tl.load(
                dense_block_ptrs + dense_tiled_row_stride * dense_row_idx
            )

            # do block mm
            output_acc_block += tl.dot(
                values_block, dense_block, allow_tf32=allow_tf32, out_dtype=acc_dtype
            )

            # move val/col_index ptrs to the next block in the row
            values_block_ptrs += values_nnz_stride
            col_index_nnz_ptr += col_indices_stride

        if not alpha_is_one:
            output_acc_block *= alpha

        if not left_alpha_is_one:
            left_alpha_ptrs = (
                left_alpha_ptr
                + left_alpha_batch_stride * batch_pid
                + left_alpha_tiled_row_stride * row_block_pid
                + left_alpha_tiled_col_stride * col_block_pid
                + left_alpha_row_block_stride * row_block_arange[:, None]
                + left_alpha_col_block_stride * col_block_arange[None, :]
            )
            output_acc_block *= tl.load(left_alpha_ptrs)

        if not right_alpha_is_one:
            right_alpha_ptrs = (
                right_alpha_ptr
                + right_alpha_batch_stride * batch_pid
                + right_alpha_tiled_row_stride * row_block_pid
                + right_alpha_tiled_col_stride * col_block_pid
                + right_alpha_row_block_stride * row_block_arange[:, None]
                + right_alpha_col_block_stride * col_block_arange[None, :]
            )
            output_acc_block *= tl.load(right_alpha_ptrs)

        if beta_is_nonzero:
            input_ptrs = (
                input_ptr
                + input_batch_stride * batch_pid
                + input_tiled_row_stride * row_block_pid
                + input_tiled_col_stride * col_block_pid
                + input_row_block_stride * row_block_arange[:, None]
                + input_col_block_stride * col_block_arange[None, :]
            )
            if beta_is_one:
                output_acc_block += tl.load(input_ptrs)
            else:
                output_acc_block += beta * tl.load(input_ptrs)

        # write back the result
        tl.store(output_ptrs, output_acc_block.to(output_ptr.dtype.element_ty))

else:
    bsr_softmax = None  # type: ignore[assignment]
    bsr_dense_mm = None  # type: ignore[assignment]
    sampled_addmm = None  # type: ignore[assignment]
    _scaled_dot_product_attention = None  # type: ignore[assignment]
    _scatter_mm2 = None  # type: ignore[assignment]
    _scatter_mm6 = None  # type: ignore[assignment]
    _bsr_strided_addmm_kernel = None  # type: ignore[assignment]
