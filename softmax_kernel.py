
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

from torch._dynamo.testing import rand_strided
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch
from torch._inductor.runtime.triton_heuristics import grid, split_scan_grid

@triton_heuristics.reduction(
    size_hints=[32768, 65536],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*bf16', 1: '*bf16', 2: 'i32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax_0', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '196b341d951bbdd96de5b2b44b37054c2cba8e89494c8b8a1052a863b8bc7596', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'kernel_num_gb': 6.593445888}
)
@triton.jit
def softmax_kernel(in_ptr0, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 32768
    rnumel = 50304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp3 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (50304*x0)), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
        tmp4 = triton_helpers.maximum(_tmp3, tmp2)
        _tmp3 = tl.where(rmask, tmp4, _tmp3)
    tmp3 = triton_helpers.max2(_tmp3, 1)[:, None]
    _tmp10 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp5 = tl.load(in_ptr0 + (r1 + (50304*x0)), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp6 = tmp5.to(tl.float32)
        tmp7 = tmp6 - tmp3
        tmp8 = tl_math.exp(tmp7)
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(rmask, tmp11, _tmp10)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp12 = tl.load(in_ptr0 + (r1 + (50304*x0)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp13 = tmp12.to(tl.float32)
        tmp14 = tmp13 - tmp3
        tmp15 = tl_math.exp(tmp14)
        tmp16 = tmp15 / tmp10
        tmp17 = tmp16.to(tl.float32)
        tl.store(out_ptr2 + (r1 + (50304*x0)), tmp17, rmask)

@triton_heuristics.reduction(
    size_hints=[32768, 65536],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*bf16', 1: '*bf16', 2: 'i32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax_0', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '196b341d951bbdd96de5b2b44b37054c2cba8e89494c8b8a1052a863b8bc7596', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'kernel_num_gb': 6.593445888}
)
@triton.jit
def online_softmax_kernel(in_ptr0, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 32768
    rnumel = 50304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex

    _tmp3 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    _tmp10 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)

    ones = tl.full([XBLOCK, RBLOCK], 1, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (50304*x0)), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)

        mp_0, mp_1 = triton_helpers.online_softmax_combine(
            _tmp3, _tmp10, tmp1, ones
        )
        _tmp3 = tl.where(rmask, mp_0, _tmp3)
        _tmp10 = tl.where(rmask, mp_1, _tmp10)

    mp3, mp10 = triton_helpers.online_softmax_reduce(_tmp3, _tmp10)
    tmp3 = mp3[:, None]
    tmp10 = mp10[:, None]

    # for roffset in range(rnumel - rnumel % RBLOCK, -RBLOCK, -RBLOCK):
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp12 = tl.load(in_ptr0 + (r1 + (50304*x0)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp13 = tmp12.to(tl.float32)
        tmp14 = tmp13 - tmp3
        tmp15 = tl_math.exp(tmp14)
        tmp16 = tmp15 / tmp10
        tmp17 = tmp16.to(tl.float32)
        tl.store(out_ptr2 + (r1 + (50304*x0)), tmp17, rmask)


def get_args():
    arg_0 = rand_strided((32768, 50304), (50304, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_1 = rand_strided((32768, 50304), (50304, 1), device='cuda:0', dtype=torch.bfloat16)
    return arg_0, arg_1,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        online_softmax_kernel.run(*args, 32768, 50304, grid=grid(32768), stream=stream0)
        # softmax_kernel.run(*args, 32768, 50304, grid=grid(32768), stream=stream0)


if __name__ == '__main__':
    from triton.testing import do_bench

    args = get_args()
    call(args)
    x, actual = args
    expected = torch.softmax(x, dim=-1)
    assert torch.allclose(expected, actual, atol=1e-3, rtol=1e-3)
    ms = do_bench(lambda: call(args), rep=40, fast_flush=True)
    num_gb = 6.593445888
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
