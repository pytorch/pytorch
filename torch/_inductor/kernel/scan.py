import torch
import torch._prims_common as utils
from torch._inductor import ir
from torch._inductor.select_algorithm import (
    autotune_select_algorithm,
    ChoiceCaller,
    ExternKernelChoice,
    TritonTemplate,
)
from torch._inductor.utils import ceildiv, get_dtype_size, sympy_product
from torch.utils._sympy.functions import CeilDiv

# This implements multi-block scans in the same manner as CUB, according to the paper
#
#     Merrill, Duane, and Michael Garland. "Single-pass parallel prefix scan with decoupled look-back."
#     NVIDIA, Tech. Rep. NVR-2016-002 (2016).
#
#     https://research.nvidia.com/publication/2016-03_single-pass-parallel-prefix-scan-decoupled-look-back
#
# The algorithm goes as follows:
# 1. Initailize flags to 0 before launching kernel
# 2. Compute the reduction within a thread block
# 3. Store reduction in a scratch buffer and set block's flag to 1
# 4. Compute an exclusive prefix scan between blocks, via sequential lookback
#    For each block we look at:
#       a) If flag is 2: block prefix is available, combine with our prefix and exit loop
#       b) If flag is 1: block reduction available, combine with our prefix and continue look back
#       c) If flag is 0: no data available, keep polling until flag changes
# 5. Compute block level inclusive scan by combining exclusive scan with our block reduction
# 6. Store inclusive scan in scratch buffer and set our flag to 2
# 7. Combine block level exclusive scan with thread level scan to give final result
#
# This feels sequential since we do sequential lookback, but the decoupling
# means that in practice the prefix "jumps" forward by several blocks at a
# time.
#
# NOTE: In practice, the flag and value are packed into a single element in the
# scratch buffer. This is required to use relaxed atomics for which there is
# far less cache invalidation and thus much better performance.


def cumsum_grid(batch_size, dim_size, meta):
    """
    The CUDA grid size for cumsum template kernel.
    """
    return (ceildiv(dim_size, meta["XBLOCK"]), batch_size, 1)


cumsum_template = TritonTemplate(
    name="cumsum",
    grid=cumsum_grid,
    debug=True,
    source="""
import triton

@triton.jit
def pack_value_flag(value, flag):
    uv = value.to({{DTYPE_VALUE_AS_UINT}}, bitcast=True).to({{DTYPE_PACK}})
    pack = flag.to({{DTYPE_PACK}}) | (uv << {{VALUE_BITWIDTH}})
    return pack.to({{DTYPE_PACK_SIGNED}}, bitcast=True)

@triton.jit
def unpack_value(pack):
    return (pack >> {{VALUE_BITWIDTH}}).to({{DTYPE_VALUE_AS_UINT}}).to({{DTYPE_VALUE}}, bitcast=True)

@triton.jit
def unpack_flag(pack):
    return pack.to({{DTYPE_FLAG}})

{{def_kernel("in_ptr0", "scratch_ptr")}}
    batch_size = tl.num_programs(1)
    num_xblocks = tl.num_programs(0)
    dim_size = {{size("in_ptr0", 1)}}

    yindex = tl.program_id(1)
    xid = tl.program_id(0)
    xindex = xid * XBLOCK + tl.arange(0, XBLOCK)
    xmask = xindex < dim_size
    scratch_base = (scratch_ptr + yindex * num_xblocks).to(tl.pointer_type({{DTYPE_PACK_SIGNED}}))

    reduction_dtype = scratch_ptr.type.scalar.element_ty
    block_data = {{make_load("in_ptr0", ["yindex", "xindex"], "xmask")}}
    masked_data = tl.where(xmask, block_data, 0).to({{DTYPE_VALUE}})
    block_sum = tl.sum(block_data)

    # Make block_sum visible to other blocks
    pack = pack_value_flag(block_sum, tl.full([], 1, {{DTYPE_FLAG}}))
    # tl.atomic_store
    tl.atomic_xchg(scratch_base + xid, pack, sem="relaxed")

    # Calculate exclusive prefix sum via decoupled lookback
    exclusive_prefix = tl.zeros_like(block_sum)
    test_target = xid - 1
    while test_target >= 0:
        # tl.atomic_load
        flag = tl.full([], 0, tl.uint32)
        while flag == 0:
            pack = tl.atomic_add(scratch_base + test_target, 0, sem="relaxed")
            flag = unpack_flag(pack)

        value = unpack_value(pack)
        exclusive_prefix += value

        if flag == 2:
            test_target = -1
        else:
            test_target = test_target - 1

    # Make inclusive block sum visible to other blocks
    pack = pack_value_flag(exclusive_prefix + block_sum, tl.full([], 2, {{DTYPE_FLAG}}))
    # tl.atomic_store
    tl.atomic_xchg(scratch_base + xid, pack, sem="relaxed")

    # Compute final cumsum
    block_scan = tl.cumsum(block_data, 0)
    result = block_scan + exclusive_prefix

    # Inductor generates a suffix
    {{store_output(("yindex", "xindex"), "result", "xmask")}}
""",
)


def view_as_batch_and_dim(x, dim):
    orig_sizes = x.get_size()
    perm = [i for i in range(len(orig_sizes)) if i != dim] + [dim]
    x = ir.PermuteView.create(x, perm)
    return ir.TensorBox(ir.View.create(x, (-1, orig_sizes[dim])))


def view_as_original_shape(x, orig_sizes, dim):
    pre_dim_size = sympy_product(orig_sizes[:dim])
    dim_size = orig_sizes[dim]
    post_dim_size = sympy_product(orig_sizes[dim + 1 :])

    tmp = ir.View.create(x, (pre_dim_size, post_dim_size, dim_size))
    tmp = ir.PermuteView.create(tmp, [0, 2, 1])
    return ir.TensorBox(ir.View.create(tmp, orig_sizes))


def aten_cumsum(x, scratch, flags):
    return x.cumsum(-1)


fallback_aten_cumsum = ExternKernelChoice(aten_cumsum, None)


def split_cumsum(x, dim):
    from torch._inductor.codegen.triton import triton_compute_type
    from torch._inductor.lowering import _full, clone

    assert x.get_device().type == "cuda"

    element_type = x.get_dtype()
    element_size = get_dtype_size(element_type)
    element_bitwidth = 8 * element_size
    assert element_size <= 4, "split_cumsum only supports up to 4 bytes per element"

    torch_pack_dtype = torch.int32 if element_size <= 2 else torch.int64
    triton_pack_dtype = "tl.uint32" if element_size <= 2 else "tl.uint64"

    value_as_uint_dtype = f"tl.uint{element_bitwidth}"
    flag_dtype = value_as_uint_dtype

    kernel_kwargs = {
        "DTYPE_VALUE": triton_compute_type(element_type),
        "DTYPE_VALUE_AS_UINT": value_as_uint_dtype,
        "DTYPE_PACK": triton_pack_dtype,
        "DTYPE_PACK_SIGNED": triton_pack_dtype.replace("uint", "int"),
        "DTYPE_FLAG": value_as_uint_dtype,
        "VALUE_BITWIDTH": str(element_bitwidth),
    }

    orig_sizes = x.get_size()
    if dim < 0:
        dim += len(orig_sizes)
        torch._check(0 < dim < len(orig_sizes), lambda: f"{dim} out of bounds")

    x = clone(view_as_batch_and_dim(x, dim))
    batch_size, dim_size = x.get_size()

    out_layout = ir.FixedLayout(
        device=x.get_device(),
        dtype=x.get_dtype(),
        size=[batch_size, dim_size],
    )
    reduction_dtype = utils.get_computation_dtype(x.get_dtype())
    max_blocks = batch_size * CeilDiv(dim_size, 128)
    scratch = _full(0, size=[max_blocks], dtype=torch_pack_dtype, device=x.get_device())

    x.realize()
    scratch.realize()

    choices: ChoiceCaller = []

    for XBLOCK, num_warps in [
        (512, 4),
        (512, 8),
        (256, 4),
        (128, 4),
        (1024, 4),
        (1024, 8),
    ]:
        cumsum_template.maybe_append_choice(
            choices,
            input_nodes=(x, scratch),
            layout=out_layout,
            XBLOCK=XBLOCK,
            num_warps=num_warps,
            num_stages=1,
            **kernel_kwargs,
        )

    choices.append(fallback_aten_cumsum.bind((x, scratch), out_layout))

    result = autotune_select_algorithm("cumsum", choices, [x, scratch], out_layout)
    return view_as_original_shape(result, orig_sizes, dim)
