
from ctypes import c_void_p, c_long
import torch
import math
import random
from torch import empty_strided, as_strided, device
from torch._inductor.codecache import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels

aten = torch.ops.aten
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
async_compile = AsyncCompile()

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream

torch.set_float32_matmul_precision('high')

# kernel path: /tmp/tmpom3r99qq/hg/chg5aytsyih6te3cizx4mr7agdpnasx6irfhddmrgmfnc3ydki6e.py
# Original ATen: aten.sum

# aten.sum => sum_13
triton_fused_sum_0 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import reduction
from torch._inductor.utils import instance_descriptor

@reduction(
    size_hints=[32768, 2048],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 3), equal_to_1=())]}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 30522
    rnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp1 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (30522*r1)), rmask & xmask, eviction_policy='evict_last', other=0)
        _tmp1 = tl.where(rmask & xmask, _tmp1 + tmp0, _tmp1)
    tmp1 = tl.sum(_tmp1, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp1, xmask)
''')


# kernel path: /tmp/tmpom3r99qq/gd/cgdykfpn3gw7hjyke5jwxyaajw7guwj36tsxbakmffartqz7o5of.py
# Original ATen: aten.gelu, aten.gelu_backward, aten.native_layer_norm_backward, aten.view

# aten.gelu => add_100, erf_12, mul_162
# aten.gelu_backward => add_104, exp_12, mul_174, mul_175, mul_176, mul_177, mul_178, mul_179
# aten.native_layer_norm_backward => mul_167, mul_168, mul_169, mul_170, mul_171, sub_40, sub_41, sum_14, sum_15
# aten.view => view_234
triton_fused_gelu_gelu_backward_native_layer_norm_backward_view_1 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import persistent_reduction
from torch._inductor.utils import instance_descriptor

@persistent_reduction(
    size_hints=[2048, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask, other=0)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, other=0)
    tmp6 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask, other=0)
    tmp11 = tl.load(in_ptr3 + (x0), None)
    tmp18 = tl.load(in_ptr4 + (r1 + (768*x0)), rmask, other=0)
    tmp2 = tmp0 * tmp1
    tmp4 = tl.where(rmask, tmp2, 0)
    tmp5 = tl.sum(tmp4, 1)[:, None]
    tmp7 = tmp2 * tmp6
    tmp9 = tl.where(rmask, tmp7, 0)
    tmp10 = tl.sum(tmp9, 1)[:, None]
    tmp12 = 768.0
    tmp13 = tmp2 * tmp12
    tmp14 = tmp13 - tmp5
    tmp15 = tmp6 * tmp10
    tmp16 = tmp14 - tmp15
    tmp17 = tmp11 * tmp16
    tmp19 = 0.7071067811865476
    tmp20 = tmp18 * tmp19
    tmp21 = tl.math.erf(tmp20)
    tmp22 = 1.0
    tmp23 = tmp21 + tmp22
    tmp24 = 0.5
    tmp25 = tmp23 * tmp24
    tmp26 = tmp18 * tmp18
    tmp27 = -0.5
    tmp28 = tmp26 * tmp27
    tmp29 = tl.exp(tmp28)
    tmp30 = 0.3989422804014327
    tmp31 = tmp29 * tmp30
    tmp32 = tmp18 * tmp31
    tmp33 = tmp25 + tmp32
    tmp34 = tmp17 * tmp33
    tl.store(in_out_ptr0 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp34, rmask)
''')


# kernel path: /tmp/tmpom3r99qq/an/canncvmg262yzhfpymbvmkofvyhovvfktyomcvl6vo4ob5lrzj53.py
# Original ATen: aten.native_layer_norm_backward

# aten.native_layer_norm_backward => mul_172, sum_16, sum_17
triton_fused_native_layer_norm_backward_2 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import reduction
from torch._inductor.utils import instance_descriptor

@reduction(
    size_hints=[16384, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 12288
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 768
    x1 = (xindex // 768)
    _tmp3 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    x3 = xindex
    _tmp4 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (768*r2) + (98304*x1)), rmask, eviction_policy='evict_last', other=0)
        tmp1 = tl.load(in_ptr1 + (x0 + (768*r2) + (98304*x1)), rmask, eviction_policy='evict_last', other=0)
        tmp2 = tmp0 * tmp1
        _tmp3 = tl.where(rmask, _tmp3 + tmp2, _tmp3)
        _tmp4 = tl.where(rmask, _tmp4 + tmp0, _tmp4)
    tmp3 = tl.sum(_tmp3, 1)[:, None]
    tl.store(out_ptr0 + x3, tmp3, None)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr1 + x3, tmp4, None)
''')


# kernel path: /tmp/tmpom3r99qq/62/c62lgcl4velwlwbtl5thb5crkh4jwwndzdlbvnxiz4tq6mrox4so.py
# Original ATen: aten.native_layer_norm_backward

# aten.native_layer_norm_backward => mul_172, sum_16
triton_fused_native_layer_norm_backward_3 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import persistent_reduction
from torch._inductor.utils import instance_descriptor

@persistent_reduction(
    size_hints=[1024, 16],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 768
    rnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (768*r1)), rmask & xmask, other=0)
    tmp2 = tl.where(rmask & xmask, tmp0, 0)
    tmp3 = tl.sum(tmp2, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp3, xmask)
''')


# kernel path: /tmp/tmpom3r99qq/vv/cvvgvfoerpopezzdwyvbpp4cphwavupplwcvdb5qyx3q2nlgg7u7.py
# Original ATen: aten.sum

# aten.sum => sum_18
triton_fused_sum_4 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import reduction
from torch._inductor.utils import instance_descriptor

@reduction(
    size_hints=[16384, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 12288
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 768
    x1 = (xindex // 768)
    _tmp1 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (768*r2) + (98304*x1)), rmask, eviction_policy='evict_last', other=0)
        _tmp1 = tl.where(rmask, _tmp1 + tmp0, _tmp1)
    tmp1 = tl.sum(_tmp1, 1)[:, None]
    tl.store(out_ptr0 + x3, tmp1, None)
''')


# kernel path: /tmp/tmpom3r99qq/wo/cwoqnp6yglx5t2jqgwtv6k4jsb5rhzx4a4nxplheldjkvxpg7odx.py
# Original ATen: aten._to_copy, aten.gt, aten.mul, aten.native_layer_norm_backward, aten.view, prims.philox_rand_like

# aten._to_copy => convert_element_type_37
# aten.gt => gt_37
# aten.mul => mul_187, mul_188
# aten.native_layer_norm_backward => mul_181, mul_182, mul_183, mul_184, mul_185, sub_43, sub_44, sum_19, sum_20
# aten.view => view_237
# prims.philox_rand_like => philox_rand_like_37
triton_fused__to_copy_gt_mul_native_layer_norm_backward_philox_rand_like_view_5 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import persistent_reduction
from torch._inductor.utils import instance_descriptor

@persistent_reduction(
    size_hints=[2048, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*i64', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask, other=0)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, other=0)
    tmp6 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask, other=0)
    tmp11 = tl.load(in_ptr3 + (x0), None)
    tmp18_load = tl.load(in_ptr4 + (0))
    tmp18 = tl.broadcast_to(tmp18_load, [XBLOCK, RBLOCK])
    tmp2 = tmp0 * tmp1
    tmp4 = tl.where(rmask, tmp2, 0)
    tmp5 = tl.sum(tmp4, 1)[:, None]
    tmp7 = tmp2 * tmp6
    tmp9 = tl.where(rmask, tmp7, 0)
    tmp10 = tl.sum(tmp9, 1)[:, None]
    tmp12 = 768.0
    tmp13 = tmp2 * tmp12
    tmp14 = tmp13 - tmp5
    tmp15 = tmp6 * tmp10
    tmp16 = tmp14 - tmp15
    tmp17 = tmp11 * tmp16
    tmp19 = 188743680 + r1 + (768*x0)
    tmp20 = tl.rand(tmp18, tmp19)
    tmp21 = 0.1
    tmp22 = tmp20 > tmp21
    tmp23 = tmp22.to(tl.float32)
    tmp24 = tmp23 * tmp17
    tmp25 = 1.1111111111111112
    tmp26 = tmp24 * tmp25
    tl.store(out_ptr2 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp17, rmask)
    tl.store(in_out_ptr0 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp26, rmask)
''')


# kernel path: /tmp/tmpom3r99qq/kn/cknl4trelqiue7tkbaqkqaad6iinkne64dwtmd7pds3fptirrcr3.py
# Original ATen: aten.gelu, aten.gelu_backward, aten.view

# aten.gelu => add_96, erf_11, mul_155
# aten.gelu_backward => add_106, exp_13, mul_190, mul_191, mul_192, mul_193, mul_194, mul_195
# aten.view => view_240
triton_fused_gelu_gelu_backward_view_6 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6291456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr0 + (x0), None)
    tmp2 = 0.7071067811865476
    tmp3 = tmp1 * tmp2
    tmp4 = tl.math.erf(tmp3)
    tmp5 = 1.0
    tmp6 = tmp4 + tmp5
    tmp7 = 0.5
    tmp8 = tmp6 * tmp7
    tmp9 = tmp1 * tmp1
    tmp10 = -0.5
    tmp11 = tmp9 * tmp10
    tmp12 = tl.exp(tmp11)
    tmp13 = 0.3989422804014327
    tmp14 = tmp12 * tmp13
    tmp15 = tmp1 * tmp14
    tmp16 = tmp8 + tmp15
    tmp17 = tmp0 * tmp16
    tl.store(in_out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp17, None)
''')


# kernel path: /tmp/tmpom3r99qq/yz/cyzlhlfa3nckxm3tdtrgl4etxqo3ovjpazv7uzqeegda3gxtp5i4.py
# Original ATen: aten.sum

# aten.sum => sum_24
triton_fused_sum_7 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import reduction
from torch._inductor.utils import instance_descriptor

@reduction(
    size_hints=[65536, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 49152
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 3072
    x1 = (xindex // 3072)
    _tmp1 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (3072*r2) + (393216*x1)), rmask, eviction_policy='evict_last', other=0)
        _tmp1 = tl.where(rmask, _tmp1 + tmp0, _tmp1)
    tmp1 = tl.sum(_tmp1, 1)[:, None]
    tl.store(out_ptr0 + x3, tmp1, None)
''')


# kernel path: /tmp/tmpom3r99qq/7i/c7isijyr7hbblqmgese3c2f42n6t4sk6vx4fmh25mrrvft375zve.py
# Original ATen: aten.sum

# aten.sum => sum_24
triton_fused_sum_8 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import persistent_reduction
from torch._inductor.utils import instance_descriptor

@persistent_reduction(
    size_hints=[4096, 16],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3072
    rnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (3072*r1)), rmask & xmask, other=0)
    tmp2 = tl.where(rmask & xmask, tmp0, 0)
    tmp3 = tl.sum(tmp2, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp3, xmask)
''')


# kernel path: /tmp/tmpom3r99qq/aq/caqfczdmtp3h7gyig2pvwxznkuvhppj2rr54elhkkxvc6h2zwiyr.py
# Original ATen: aten._to_copy, aten.add, aten.gt, aten.mul, aten.native_layer_norm_backward, aten.view, prims.philox_rand_like

# aten._to_copy => convert_element_type_38
# aten.add => add_107
# aten.gt => gt_38
# aten.mul => mul_203, mul_204
# aten.native_layer_norm_backward => mul_197, mul_198, mul_199, mul_200, mul_201, sub_46, sub_47, sum_25, sum_26
# aten.view => view_243
# prims.philox_rand_like => philox_rand_like_38
triton_fused__to_copy_add_gt_mul_native_layer_norm_backward_philox_rand_like_view_9 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import persistent_reduction
from torch._inductor.utils import instance_descriptor

@persistent_reduction(
    size_hints=[2048, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*i64', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask, other=0)
    tmp1 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask, other=0)
    tmp3 = tl.load(in_ptr2 + (r1), rmask, other=0)
    tmp8 = tl.load(in_ptr3 + (r1 + (768*x0)), rmask, other=0)
    tmp13 = tl.load(in_ptr4 + (x0), None)
    tmp20_load = tl.load(in_ptr5 + (0))
    tmp20 = tl.broadcast_to(tmp20_load, [XBLOCK, RBLOCK])
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tmp6 = tl.where(rmask, tmp4, 0)
    tmp7 = tl.sum(tmp6, 1)[:, None]
    tmp9 = tmp4 * tmp8
    tmp11 = tl.where(rmask, tmp9, 0)
    tmp12 = tl.sum(tmp11, 1)[:, None]
    tmp14 = 768.0
    tmp15 = tmp4 * tmp14
    tmp16 = tmp15 - tmp7
    tmp17 = tmp8 * tmp12
    tmp18 = tmp16 - tmp17
    tmp19 = tmp13 * tmp18
    tmp21 = 187170816 + r1 + (768*x0)
    tmp22 = tl.rand(tmp20, tmp21)
    tmp23 = 0.1
    tmp24 = tmp22 > tmp23
    tmp25 = tmp24.to(tl.float32)
    tmp26 = tmp25 * tmp19
    tmp27 = 1.1111111111111112
    tmp28 = tmp26 * tmp27
    tl.store(out_ptr2 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp19, rmask)
    tl.store(in_out_ptr0 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp28, rmask)
''')


# kernel path: /tmp/tmpom3r99qq/6w/c6wnmhorbmwqiwvvlpqjym4qz6g3co2dryhntk3bkqb7nqktd3hw.py
# Original ATen: aten.add, aten.native_layer_norm_backward

# aten.add => add_107
# aten.native_layer_norm_backward => mul_202, sum_27, sum_28
triton_fused_add_native_layer_norm_backward_10 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import reduction
from torch._inductor.utils import instance_descriptor

@reduction(
    size_hints=[16384, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 12288
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 768
    x1 = (xindex // 768)
    _tmp5 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    x3 = xindex
    _tmp6 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (768*r2) + (98304*x1)), rmask, eviction_policy='evict_last', other=0)
        tmp1 = tl.load(in_ptr1 + (x0 + (768*r2) + (98304*x1)), rmask, eviction_policy='evict_last', other=0)
        tmp3 = tl.load(in_ptr2 + (x0 + (768*r2) + (98304*x1)), rmask, eviction_policy='evict_last', other=0)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 * tmp3
        _tmp5 = tl.where(rmask, _tmp5 + tmp4, _tmp5)
        _tmp6 = tl.where(rmask, _tmp6 + tmp2, _tmp6)
    tmp5 = tl.sum(_tmp5, 1)[:, None]
    tl.store(out_ptr0 + x3, tmp5, None)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr1 + x3, tmp6, None)
''')


# kernel path: /tmp/tmpom3r99qq/6a/c6adbajr42exxj5thpzs5uupjedquweddhfwdp5l7vqcjatgxk3g.py
# Original ATen: aten.clone

# aten.clone => clone_48
triton_fused_clone_11 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1572864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = (xindex // 64) % 512
    x2 = (xindex // 32768) % 12
    x3 = (xindex // 393216)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*x2) + (768*x1) + (393216*x3)), None)
    tl.store(out_ptr0 + (x4 + tl.zeros([XBLOCK], tl.int32)), tmp0, None)
''')


# kernel path: /tmp/tmpom3r99qq/wy/cwye3abwj6dwbmzhwqhfjrcmnwjdyb443dzuaklin564vbj3qsv7.py
# Original ATen: aten._softmax_backward_data, aten._to_copy, aten.div, aten.gt, aten.mul, prims.philox_rand_like

# aten._softmax_backward_data => mul_207, mul_208, sub_48, sum_30
# aten._to_copy => convert_element_type_39
# aten.div => div_27
# aten.gt => gt_39
# aten.mul => mul_205, mul_206
# prims.philox_rand_like => philox_rand_like_39
triton_fused__softmax_backward_data__to_copy_div_gt_mul_philox_rand_like_12 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import persistent_reduction
from torch._inductor.utils import instance_descriptor

@persistent_reduction(
    size_hints=[32768, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 24576
    rnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0_load = tl.load(in_ptr0 + (0))
    tmp0 = tl.broadcast_to(tmp0_load, [XBLOCK, RBLOCK])
    tmp6 = tl.load(in_ptr1 + (r1 + (512*x0)), rmask, other=0)
    tmp10 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask, other=0)
    tmp1 = 174587904 + r1 + (512*x0)
    tmp2 = tl.rand(tmp0, tmp1)
    tmp3 = 0.1
    tmp4 = tmp2 > tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp7 = tmp5 * tmp6
    tmp8 = 1.1111111111111112
    tmp9 = tmp7 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tl.where(rmask, tmp11, 0)
    tmp14 = tl.sum(tmp13, 1)[:, None]
    tmp15 = tmp10 * tmp14
    tmp16 = tmp11 - tmp15
    tmp17 = 8.0
    tmp18 = tmp16 / tmp17
    tl.store(out_ptr1 + (r1 + (512*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp18, rmask)
''')


# kernel path: /tmp/tmpom3r99qq/zz/czzilo7fojeogv6h3mwwlyolbrvwrxlwls6cqsrz3vb3k6zlntmw.py
# Original ATen: aten.view

# aten.view => view_252
triton_fused_view_13 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1572864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 768
    x1 = (xindex // 768)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + ((64*(x1 % 512)) + (32768*(x0 // 64)) + (393216*(x1 // 512)) + (x0 % 64)), None)
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp0, None)
''')


# kernel path: /tmp/tmpom3r99qq/gh/cghqqdpjbycr4nzkckqyamwx3vb6ebos6fqxcudj6tj52vpnytlj.py
# Original ATen: aten._unsafe_view, aten.clone

# aten._unsafe_view => _unsafe_view_39
# aten.clone => clone_51
triton_fused__unsafe_view_clone_14 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[2048, 1024], tile_hint=TileHint.SQUARE,filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, ynumel, XBLOCK : tl.constexpr, YBLOCK : tl.constexpr):
    xnumel = 2048
    ynumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    x0 = xindex
    y1 = yindex
    tmp0 = tl.load(in_ptr0 + ((512*y1) + (393216*(x0 // 512)) + (x0 % 512)), ymask)
    tl.store(out_ptr0 + (y1 + (768*x0) + tl.zeros([XBLOCK, YBLOCK], tl.int32)), tmp0, ymask)
''')


# kernel path: /tmp/tmpom3r99qq/wi/cwimzs3m7k2kpn53bi5fvbghn4g2exdy4ht55xpfwmmbehcr2n6m.py
# Original ATen: aten._to_copy, aten.add, aten.gt, aten.mul, aten.native_layer_norm_backward, aten.view, prims.philox_rand_like

# aten._to_copy => convert_element_type_40
# aten.add => add_108, add_109, add_110
# aten.gt => gt_40
# aten.mul => mul_216, mul_217
# aten.native_layer_norm_backward => mul_210, mul_211, mul_212, mul_213, mul_214, sub_50, sub_51, sum_34, sum_35
# aten.view => view_261
# prims.philox_rand_like => philox_rand_like_40
triton_fused__to_copy_add_gt_mul_native_layer_norm_backward_philox_rand_like_view_15 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import persistent_reduction
from torch._inductor.utils import instance_descriptor

@persistent_reduction(
    size_hints=[2048, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*i64', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=())]}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask, other=0)
    tmp1 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask, other=0)
    tmp3 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask, other=0)
    tmp5 = tl.load(in_ptr3 + (r1 + (768*x0)), rmask, other=0)
    tmp7 = tl.load(in_ptr4 + (r1), rmask, other=0)
    tmp12 = tl.load(in_ptr5 + (r1 + (768*x0)), rmask, other=0)
    tmp17 = tl.load(in_ptr6 + (x0), None)
    tmp24_load = tl.load(in_ptr7 + (0))
    tmp24 = tl.broadcast_to(tmp24_load, [XBLOCK, RBLOCK])
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 * tmp7
    tmp10 = tl.where(rmask, tmp8, 0)
    tmp11 = tl.sum(tmp10, 1)[:, None]
    tmp13 = tmp8 * tmp12
    tmp15 = tl.where(rmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None]
    tmp18 = 768.0
    tmp19 = tmp8 * tmp18
    tmp20 = tmp19 - tmp11
    tmp21 = tmp12 * tmp16
    tmp22 = tmp20 - tmp21
    tmp23 = tmp17 * tmp22
    tmp25 = 173015040 + r1 + (768*x0)
    tmp26 = tl.rand(tmp24, tmp25)
    tmp27 = 0.1
    tmp28 = tmp26 > tmp27
    tmp29 = tmp28.to(tl.float32)
    tmp30 = tmp29 * tmp23
    tmp31 = 1.1111111111111112
    tmp32 = tmp30 * tmp31
    tl.store(out_ptr3 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp23, rmask)
    tl.store(in_out_ptr0 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp32, rmask)
''')


# kernel path: /tmp/tmpom3r99qq/my/cmyfcuqvxux54l6czr2bksxkh2mkpepeavff7lviarf22bg6mfzu.py
# Original ATen: aten.add, aten.native_layer_norm_backward

# aten.add => add_108, add_109, add_110
# aten.native_layer_norm_backward => mul_215, sum_36, sum_37
triton_fused_add_native_layer_norm_backward_16 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import reduction
from torch._inductor.utils import instance_descriptor

@reduction(
    size_hints=[16384, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 12288
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 768
    x1 = (xindex // 768)
    _tmp9 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    x3 = xindex
    _tmp10 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (768*r2) + (98304*x1)), rmask, eviction_policy='evict_last', other=0)
        tmp1 = tl.load(in_ptr1 + (x0 + (768*r2) + (98304*x1)), rmask, eviction_policy='evict_last', other=0)
        tmp3 = tl.load(in_ptr2 + (x0 + (768*r2) + (98304*x1)), rmask, eviction_policy='evict_last', other=0)
        tmp5 = tl.load(in_ptr3 + (x0 + (768*r2) + (98304*x1)), rmask, eviction_policy='evict_last', other=0)
        tmp7 = tl.load(in_ptr4 + (x0 + (768*r2) + (98304*x1)), rmask, eviction_policy='evict_last', other=0)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 + tmp3
        tmp6 = tmp4 + tmp5
        tmp8 = tmp6 * tmp7
        _tmp9 = tl.where(rmask, _tmp9 + tmp8, _tmp9)
        _tmp10 = tl.where(rmask, _tmp10 + tmp6, _tmp10)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr0 + x3, tmp9, None)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr1 + x3, tmp10, None)
''')


# kernel path: /tmp/tmpom3r99qq/xy/cxymzotyqnw2iciutfnb2d4qmgoypie43p2434z2llc56mlalc5n.py
# Original ATen: aten._to_copy, aten.add, aten.gt, aten.mul, aten.native_layer_norm_backward, aten.view, prims.philox_rand_like

# aten._to_copy => convert_element_type_41
# aten.add => add_113
# aten.gt => gt_41
# aten.mul => mul_232, mul_233
# aten.native_layer_norm_backward => mul_226, mul_227, mul_228, mul_229, mul_230, sub_53, sub_54, sum_40, sum_41
# aten.view => view_267
# prims.philox_rand_like => philox_rand_like_41
triton_fused__to_copy_add_gt_mul_native_layer_norm_backward_philox_rand_like_view_17 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import persistent_reduction
from torch._inductor.utils import instance_descriptor

@persistent_reduction(
    size_hints=[2048, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*i64', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask, other=0)
    tmp1 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask, other=0)
    tmp3 = tl.load(in_ptr2 + (r1), rmask, other=0)
    tmp8 = tl.load(in_ptr3 + (r1 + (768*x0)), rmask, other=0)
    tmp13 = tl.load(in_ptr4 + (x0), None)
    tmp20_load = tl.load(in_ptr5 + (0))
    tmp20 = tl.broadcast_to(tmp20_load, [XBLOCK, RBLOCK])
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tmp6 = tl.where(rmask, tmp4, 0)
    tmp7 = tl.sum(tmp6, 1)[:, None]
    tmp9 = tmp4 * tmp8
    tmp11 = tl.where(rmask, tmp9, 0)
    tmp12 = tl.sum(tmp11, 1)[:, None]
    tmp14 = 768.0
    tmp15 = tmp4 * tmp14
    tmp16 = tmp15 - tmp7
    tmp17 = tmp8 * tmp12
    tmp18 = tmp16 - tmp17
    tmp19 = tmp13 * tmp18
    tmp21 = 171442176 + r1 + (768*x0)
    tmp22 = tl.rand(tmp20, tmp21)
    tmp23 = 0.1
    tmp24 = tmp22 > tmp23
    tmp25 = tmp24.to(tl.float32)
    tmp26 = tmp25 * tmp19
    tmp27 = 1.1111111111111112
    tmp28 = tmp26 * tmp27
    tl.store(out_ptr2 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp19, rmask)
    tl.store(in_out_ptr0 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp28, rmask)
''')


# kernel path: /tmp/tmpom3r99qq/gm/cgmsz23aaz7ighjz657jn5edajefeko2ireyyssw5fo7avbh3kgj.py
# Original ATen: aten._softmax_backward_data, aten._to_copy, aten.div, aten.gt, aten.mul, prims.philox_rand_like

# aten._softmax_backward_data => mul_236, mul_237, sub_55, sum_45
# aten._to_copy => convert_element_type_42
# aten.div => div_30
# aten.gt => gt_42
# aten.mul => mul_234, mul_235
# prims.philox_rand_like => philox_rand_like_42
triton_fused__softmax_backward_data__to_copy_div_gt_mul_philox_rand_like_18 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import persistent_reduction
from torch._inductor.utils import instance_descriptor

@persistent_reduction(
    size_hints=[32768, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 24576
    rnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0_load = tl.load(in_ptr0 + (0))
    tmp0 = tl.broadcast_to(tmp0_load, [XBLOCK, RBLOCK])
    tmp6 = tl.load(in_ptr1 + (r1 + (512*x0)), rmask, other=0)
    tmp10 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask, other=0)
    tmp1 = 158859264 + r1 + (512*x0)
    tmp2 = tl.rand(tmp0, tmp1)
    tmp3 = 0.1
    tmp4 = tmp2 > tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp7 = tmp5 * tmp6
    tmp8 = 1.1111111111111112
    tmp9 = tmp7 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tl.where(rmask, tmp11, 0)
    tmp14 = tl.sum(tmp13, 1)[:, None]
    tmp15 = tmp10 * tmp14
    tmp16 = tmp11 - tmp15
    tmp17 = 8.0
    tmp18 = tmp16 / tmp17
    tl.store(out_ptr1 + (r1 + (512*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp18, rmask)
''')


# kernel path: /tmp/tmpom3r99qq/ru/cru352hmgm76kc4px44cfp7sooar6imxxivl7wpx4ifzbgft6qz4.py
# Original ATen: aten._to_copy, aten.add, aten.gt, aten.mul, aten.native_layer_norm_backward, aten.view, prims.philox_rand_like

# aten._to_copy => convert_element_type_43
# aten.add => add_114, add_115, add_116
# aten.gt => gt_43
# aten.mul => mul_245, mul_246
# aten.native_layer_norm_backward => mul_239, mul_240, mul_241, mul_242, mul_243, sub_57, sub_58, sum_49, sum_50
# aten.view => view_285
# prims.philox_rand_like => philox_rand_like_43
triton_fused__to_copy_add_gt_mul_native_layer_norm_backward_philox_rand_like_view_19 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import persistent_reduction
from torch._inductor.utils import instance_descriptor

@persistent_reduction(
    size_hints=[2048, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*i64', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=())]}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask, other=0)
    tmp1 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask, other=0)
    tmp3 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask, other=0)
    tmp5 = tl.load(in_ptr3 + (r1 + (768*x0)), rmask, other=0)
    tmp7 = tl.load(in_ptr4 + (r1), rmask, other=0)
    tmp12 = tl.load(in_ptr5 + (r1 + (768*x0)), rmask, other=0)
    tmp17 = tl.load(in_ptr6 + (x0), None)
    tmp24_load = tl.load(in_ptr7 + (0))
    tmp24 = tl.broadcast_to(tmp24_load, [XBLOCK, RBLOCK])
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 * tmp7
    tmp10 = tl.where(rmask, tmp8, 0)
    tmp11 = tl.sum(tmp10, 1)[:, None]
    tmp13 = tmp8 * tmp12
    tmp15 = tl.where(rmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None]
    tmp18 = 768.0
    tmp19 = tmp8 * tmp18
    tmp20 = tmp19 - tmp11
    tmp21 = tmp12 * tmp16
    tmp22 = tmp20 - tmp21
    tmp23 = tmp17 * tmp22
    tmp25 = 157286400 + r1 + (768*x0)
    tmp26 = tl.rand(tmp24, tmp25)
    tmp27 = 0.1
    tmp28 = tmp26 > tmp27
    tmp29 = tmp28.to(tl.float32)
    tmp30 = tmp29 * tmp23
    tmp31 = 1.1111111111111112
    tmp32 = tmp30 * tmp31
    tl.store(out_ptr3 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp23, rmask)
    tl.store(in_out_ptr0 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp32, rmask)
''')


# kernel path: /tmp/tmpom3r99qq/ke/ckep7keg47cit3h3khiu5aj6o3bqnhpqiycp2in7fkz6p7jujjcc.py
# Original ATen: aten._to_copy, aten.add, aten.gt, aten.mul, aten.native_layer_norm_backward, aten.view, prims.philox_rand_like

# aten._to_copy => convert_element_type_44
# aten.add => add_119
# aten.gt => gt_44
# aten.mul => mul_261, mul_262
# aten.native_layer_norm_backward => mul_255, mul_256, mul_257, mul_258, mul_259, sub_60, sub_61, sum_55, sum_56
# aten.view => view_291
# prims.philox_rand_like => philox_rand_like_44
triton_fused__to_copy_add_gt_mul_native_layer_norm_backward_philox_rand_like_view_20 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import persistent_reduction
from torch._inductor.utils import instance_descriptor

@persistent_reduction(
    size_hints=[2048, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*i64', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask, other=0)
    tmp1 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask, other=0)
    tmp3 = tl.load(in_ptr2 + (r1), rmask, other=0)
    tmp8 = tl.load(in_ptr3 + (r1 + (768*x0)), rmask, other=0)
    tmp13 = tl.load(in_ptr4 + (x0), None)
    tmp20_load = tl.load(in_ptr5 + (0))
    tmp20 = tl.broadcast_to(tmp20_load, [XBLOCK, RBLOCK])
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tmp6 = tl.where(rmask, tmp4, 0)
    tmp7 = tl.sum(tmp6, 1)[:, None]
    tmp9 = tmp4 * tmp8
    tmp11 = tl.where(rmask, tmp9, 0)
    tmp12 = tl.sum(tmp11, 1)[:, None]
    tmp14 = 768.0
    tmp15 = tmp4 * tmp14
    tmp16 = tmp15 - tmp7
    tmp17 = tmp8 * tmp12
    tmp18 = tmp16 - tmp17
    tmp19 = tmp13 * tmp18
    tmp21 = 155713536 + r1 + (768*x0)
    tmp22 = tl.rand(tmp20, tmp21)
    tmp23 = 0.1
    tmp24 = tmp22 > tmp23
    tmp25 = tmp24.to(tl.float32)
    tmp26 = tmp25 * tmp19
    tmp27 = 1.1111111111111112
    tmp28 = tmp26 * tmp27
    tl.store(out_ptr2 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp19, rmask)
    tl.store(in_out_ptr0 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp28, rmask)
''')


# kernel path: /tmp/tmpom3r99qq/hm/chmkxs3jvpxqgpg2jgtub4cb3xmwmmw262tqs3vzec4may35ulkk.py
# Original ATen: aten._softmax_backward_data, aten._to_copy, aten.div, aten.gt, aten.mul, prims.philox_rand_like

# aten._softmax_backward_data => mul_265, mul_266, sub_62, sum_60
# aten._to_copy => convert_element_type_45
# aten.div => div_33
# aten.gt => gt_45
# aten.mul => mul_263, mul_264
# prims.philox_rand_like => philox_rand_like_45
triton_fused__softmax_backward_data__to_copy_div_gt_mul_philox_rand_like_21 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import persistent_reduction
from torch._inductor.utils import instance_descriptor

@persistent_reduction(
    size_hints=[32768, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 24576
    rnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0_load = tl.load(in_ptr0 + (0))
    tmp0 = tl.broadcast_to(tmp0_load, [XBLOCK, RBLOCK])
    tmp6 = tl.load(in_ptr1 + (r1 + (512*x0)), rmask, other=0)
    tmp10 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask, other=0)
    tmp1 = 143130624 + r1 + (512*x0)
    tmp2 = tl.rand(tmp0, tmp1)
    tmp3 = 0.1
    tmp4 = tmp2 > tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp7 = tmp5 * tmp6
    tmp8 = 1.1111111111111112
    tmp9 = tmp7 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tl.where(rmask, tmp11, 0)
    tmp14 = tl.sum(tmp13, 1)[:, None]
    tmp15 = tmp10 * tmp14
    tmp16 = tmp11 - tmp15
    tmp17 = 8.0
    tmp18 = tmp16 / tmp17
    tl.store(out_ptr1 + (r1 + (512*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp18, rmask)
''')


# kernel path: /tmp/tmpom3r99qq/42/c4274kdn6enk72o6phmgsiux2evxwi7avn36ums733cughqtuq4q.py
# Original ATen: aten._to_copy, aten.add, aten.gt, aten.mul, aten.native_layer_norm_backward, aten.view, prims.philox_rand_like

# aten._to_copy => convert_element_type_46
# aten.add => add_120, add_121, add_122
# aten.gt => gt_46
# aten.mul => mul_274, mul_275
# aten.native_layer_norm_backward => mul_268, mul_269, mul_270, mul_271, mul_272, sub_64, sub_65, sum_64, sum_65
# aten.view => view_309
# prims.philox_rand_like => philox_rand_like_46
triton_fused__to_copy_add_gt_mul_native_layer_norm_backward_philox_rand_like_view_22 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import persistent_reduction
from torch._inductor.utils import instance_descriptor

@persistent_reduction(
    size_hints=[2048, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*i64', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=())]}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask, other=0)
    tmp1 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask, other=0)
    tmp3 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask, other=0)
    tmp5 = tl.load(in_ptr3 + (r1 + (768*x0)), rmask, other=0)
    tmp7 = tl.load(in_ptr4 + (r1), rmask, other=0)
    tmp12 = tl.load(in_ptr5 + (r1 + (768*x0)), rmask, other=0)
    tmp17 = tl.load(in_ptr6 + (x0), None)
    tmp24_load = tl.load(in_ptr7 + (0))
    tmp24 = tl.broadcast_to(tmp24_load, [XBLOCK, RBLOCK])
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 * tmp7
    tmp10 = tl.where(rmask, tmp8, 0)
    tmp11 = tl.sum(tmp10, 1)[:, None]
    tmp13 = tmp8 * tmp12
    tmp15 = tl.where(rmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None]
    tmp18 = 768.0
    tmp19 = tmp8 * tmp18
    tmp20 = tmp19 - tmp11
    tmp21 = tmp12 * tmp16
    tmp22 = tmp20 - tmp21
    tmp23 = tmp17 * tmp22
    tmp25 = 141557760 + r1 + (768*x0)
    tmp26 = tl.rand(tmp24, tmp25)
    tmp27 = 0.1
    tmp28 = tmp26 > tmp27
    tmp29 = tmp28.to(tl.float32)
    tmp30 = tmp29 * tmp23
    tmp31 = 1.1111111111111112
    tmp32 = tmp30 * tmp31
    tl.store(out_ptr3 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp23, rmask)
    tl.store(in_out_ptr0 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp32, rmask)
''')


# kernel path: /tmp/tmpom3r99qq/us/cusw3qpyhwwnosdjgqa7hqhlmfpjsseg5puvrsnxeaz2nwl3umcs.py
# Original ATen: aten._to_copy, aten.add, aten.gt, aten.mul, aten.native_layer_norm_backward, aten.view, prims.philox_rand_like

# aten._to_copy => convert_element_type_47
# aten.add => add_125
# aten.gt => gt_47
# aten.mul => mul_290, mul_291
# aten.native_layer_norm_backward => mul_284, mul_285, mul_286, mul_287, mul_288, sub_67, sub_68, sum_70, sum_71
# aten.view => view_315
# prims.philox_rand_like => philox_rand_like_47
triton_fused__to_copy_add_gt_mul_native_layer_norm_backward_philox_rand_like_view_23 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import persistent_reduction
from torch._inductor.utils import instance_descriptor

@persistent_reduction(
    size_hints=[2048, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*i64', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask, other=0)
    tmp1 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask, other=0)
    tmp3 = tl.load(in_ptr2 + (r1), rmask, other=0)
    tmp8 = tl.load(in_ptr3 + (r1 + (768*x0)), rmask, other=0)
    tmp13 = tl.load(in_ptr4 + (x0), None)
    tmp20_load = tl.load(in_ptr5 + (0))
    tmp20 = tl.broadcast_to(tmp20_load, [XBLOCK, RBLOCK])
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tmp6 = tl.where(rmask, tmp4, 0)
    tmp7 = tl.sum(tmp6, 1)[:, None]
    tmp9 = tmp4 * tmp8
    tmp11 = tl.where(rmask, tmp9, 0)
    tmp12 = tl.sum(tmp11, 1)[:, None]
    tmp14 = 768.0
    tmp15 = tmp4 * tmp14
    tmp16 = tmp15 - tmp7
    tmp17 = tmp8 * tmp12
    tmp18 = tmp16 - tmp17
    tmp19 = tmp13 * tmp18
    tmp21 = 139984896 + r1 + (768*x0)
    tmp22 = tl.rand(tmp20, tmp21)
    tmp23 = 0.1
    tmp24 = tmp22 > tmp23
    tmp25 = tmp24.to(tl.float32)
    tmp26 = tmp25 * tmp19
    tmp27 = 1.1111111111111112
    tmp28 = tmp26 * tmp27
    tl.store(out_ptr2 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp19, rmask)
    tl.store(in_out_ptr0 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp28, rmask)
''')


# kernel path: /tmp/tmpom3r99qq/vp/cvpqhbcwiaczyqta2dxsyylw73draufemc2f5ahjrublqfou62ju.py
# Original ATen: aten._softmax_backward_data, aten._to_copy, aten.div, aten.gt, aten.mul, prims.philox_rand_like

# aten._softmax_backward_data => mul_294, mul_295, sub_69, sum_75
# aten._to_copy => convert_element_type_48
# aten.div => div_36
# aten.gt => gt_48
# aten.mul => mul_292, mul_293
# prims.philox_rand_like => philox_rand_like_48
triton_fused__softmax_backward_data__to_copy_div_gt_mul_philox_rand_like_24 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import persistent_reduction
from torch._inductor.utils import instance_descriptor

@persistent_reduction(
    size_hints=[32768, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 24576
    rnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0_load = tl.load(in_ptr0 + (0))
    tmp0 = tl.broadcast_to(tmp0_load, [XBLOCK, RBLOCK])
    tmp6 = tl.load(in_ptr1 + (r1 + (512*x0)), rmask, other=0)
    tmp10 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask, other=0)
    tmp1 = 127401984 + r1 + (512*x0)
    tmp2 = tl.rand(tmp0, tmp1)
    tmp3 = 0.1
    tmp4 = tmp2 > tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp7 = tmp5 * tmp6
    tmp8 = 1.1111111111111112
    tmp9 = tmp7 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tl.where(rmask, tmp11, 0)
    tmp14 = tl.sum(tmp13, 1)[:, None]
    tmp15 = tmp10 * tmp14
    tmp16 = tmp11 - tmp15
    tmp17 = 8.0
    tmp18 = tmp16 / tmp17
    tl.store(out_ptr1 + (r1 + (512*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp18, rmask)
''')


# kernel path: /tmp/tmpom3r99qq/5h/c5h3ug72bv7qglz6tfhtbvab7u55zzwlbucrcfzkrfyozilerfwf.py
# Original ATen: aten._to_copy, aten.add, aten.gt, aten.mul, aten.native_layer_norm_backward, aten.view, prims.philox_rand_like

# aten._to_copy => convert_element_type_49
# aten.add => add_126, add_127, add_128
# aten.gt => gt_49
# aten.mul => mul_303, mul_304
# aten.native_layer_norm_backward => mul_297, mul_298, mul_299, mul_300, mul_301, sub_71, sub_72, sum_79, sum_80
# aten.view => view_333
# prims.philox_rand_like => philox_rand_like_49
triton_fused__to_copy_add_gt_mul_native_layer_norm_backward_philox_rand_like_view_25 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import persistent_reduction
from torch._inductor.utils import instance_descriptor

@persistent_reduction(
    size_hints=[2048, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*i64', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=())]}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask, other=0)
    tmp1 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask, other=0)
    tmp3 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask, other=0)
    tmp5 = tl.load(in_ptr3 + (r1 + (768*x0)), rmask, other=0)
    tmp7 = tl.load(in_ptr4 + (r1), rmask, other=0)
    tmp12 = tl.load(in_ptr5 + (r1 + (768*x0)), rmask, other=0)
    tmp17 = tl.load(in_ptr6 + (x0), None)
    tmp24_load = tl.load(in_ptr7 + (0))
    tmp24 = tl.broadcast_to(tmp24_load, [XBLOCK, RBLOCK])
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 * tmp7
    tmp10 = tl.where(rmask, tmp8, 0)
    tmp11 = tl.sum(tmp10, 1)[:, None]
    tmp13 = tmp8 * tmp12
    tmp15 = tl.where(rmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None]
    tmp18 = 768.0
    tmp19 = tmp8 * tmp18
    tmp20 = tmp19 - tmp11
    tmp21 = tmp12 * tmp16
    tmp22 = tmp20 - tmp21
    tmp23 = tmp17 * tmp22
    tmp25 = 125829120 + r1 + (768*x0)
    tmp26 = tl.rand(tmp24, tmp25)
    tmp27 = 0.1
    tmp28 = tmp26 > tmp27
    tmp29 = tmp28.to(tl.float32)
    tmp30 = tmp29 * tmp23
    tmp31 = 1.1111111111111112
    tmp32 = tmp30 * tmp31
    tl.store(out_ptr3 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp23, rmask)
    tl.store(in_out_ptr0 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp32, rmask)
''')


# kernel path: /tmp/tmpom3r99qq/bv/cbvy46szjp6vllh3hl5b43l747p5ln4o4wxvvo5sp5eeikwftlrg.py
# Original ATen: aten._to_copy, aten.add, aten.gt, aten.mul, aten.native_layer_norm_backward, aten.view, prims.philox_rand_like

# aten._to_copy => convert_element_type_50
# aten.add => add_131
# aten.gt => gt_50
# aten.mul => mul_319, mul_320
# aten.native_layer_norm_backward => mul_313, mul_314, mul_315, mul_316, mul_317, sub_74, sub_75, sum_85, sum_86
# aten.view => view_339
# prims.philox_rand_like => philox_rand_like_50
triton_fused__to_copy_add_gt_mul_native_layer_norm_backward_philox_rand_like_view_26 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import persistent_reduction
from torch._inductor.utils import instance_descriptor

@persistent_reduction(
    size_hints=[2048, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*i64', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask, other=0)
    tmp1 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask, other=0)
    tmp3 = tl.load(in_ptr2 + (r1), rmask, other=0)
    tmp8 = tl.load(in_ptr3 + (r1 + (768*x0)), rmask, other=0)
    tmp13 = tl.load(in_ptr4 + (x0), None)
    tmp20_load = tl.load(in_ptr5 + (0))
    tmp20 = tl.broadcast_to(tmp20_load, [XBLOCK, RBLOCK])
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tmp6 = tl.where(rmask, tmp4, 0)
    tmp7 = tl.sum(tmp6, 1)[:, None]
    tmp9 = tmp4 * tmp8
    tmp11 = tl.where(rmask, tmp9, 0)
    tmp12 = tl.sum(tmp11, 1)[:, None]
    tmp14 = 768.0
    tmp15 = tmp4 * tmp14
    tmp16 = tmp15 - tmp7
    tmp17 = tmp8 * tmp12
    tmp18 = tmp16 - tmp17
    tmp19 = tmp13 * tmp18
    tmp21 = 124256256 + r1 + (768*x0)
    tmp22 = tl.rand(tmp20, tmp21)
    tmp23 = 0.1
    tmp24 = tmp22 > tmp23
    tmp25 = tmp24.to(tl.float32)
    tmp26 = tmp25 * tmp19
    tmp27 = 1.1111111111111112
    tmp28 = tmp26 * tmp27
    tl.store(out_ptr2 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp19, rmask)
    tl.store(in_out_ptr0 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp28, rmask)
''')


# kernel path: /tmp/tmpom3r99qq/7a/c7awjpzfom3u666ku3ysurst6fsotcdo6ojlfh3iwy6iaypsrzet.py
# Original ATen: aten._softmax_backward_data, aten._to_copy, aten.div, aten.gt, aten.mul, prims.philox_rand_like

# aten._softmax_backward_data => mul_323, mul_324, sub_76, sum_90
# aten._to_copy => convert_element_type_51
# aten.div => div_39
# aten.gt => gt_51
# aten.mul => mul_321, mul_322
# prims.philox_rand_like => philox_rand_like_51
triton_fused__softmax_backward_data__to_copy_div_gt_mul_philox_rand_like_27 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import persistent_reduction
from torch._inductor.utils import instance_descriptor

@persistent_reduction(
    size_hints=[32768, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 24576
    rnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0_load = tl.load(in_ptr0 + (0))
    tmp0 = tl.broadcast_to(tmp0_load, [XBLOCK, RBLOCK])
    tmp6 = tl.load(in_ptr1 + (r1 + (512*x0)), rmask, other=0)
    tmp10 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask, other=0)
    tmp1 = 111673344 + r1 + (512*x0)
    tmp2 = tl.rand(tmp0, tmp1)
    tmp3 = 0.1
    tmp4 = tmp2 > tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp7 = tmp5 * tmp6
    tmp8 = 1.1111111111111112
    tmp9 = tmp7 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tl.where(rmask, tmp11, 0)
    tmp14 = tl.sum(tmp13, 1)[:, None]
    tmp15 = tmp10 * tmp14
    tmp16 = tmp11 - tmp15
    tmp17 = 8.0
    tmp18 = tmp16 / tmp17
    tl.store(out_ptr1 + (r1 + (512*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp18, rmask)
''')


# kernel path: /tmp/tmpom3r99qq/4o/c4o6eglhlz255cuicerjpnelv7qewhinz7oymskbh4scp2nmnkt2.py
# Original ATen: aten._to_copy, aten.add, aten.gt, aten.mul, aten.native_layer_norm_backward, aten.view, prims.philox_rand_like

# aten._to_copy => convert_element_type_52
# aten.add => add_132, add_133, add_134
# aten.gt => gt_52
# aten.mul => mul_332, mul_333
# aten.native_layer_norm_backward => mul_326, mul_327, mul_328, mul_329, mul_330, sub_78, sub_79, sum_94, sum_95
# aten.view => view_357
# prims.philox_rand_like => philox_rand_like_52
triton_fused__to_copy_add_gt_mul_native_layer_norm_backward_philox_rand_like_view_28 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import persistent_reduction
from torch._inductor.utils import instance_descriptor

@persistent_reduction(
    size_hints=[2048, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*i64', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=())]}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask, other=0)
    tmp1 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask, other=0)
    tmp3 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask, other=0)
    tmp5 = tl.load(in_ptr3 + (r1 + (768*x0)), rmask, other=0)
    tmp7 = tl.load(in_ptr4 + (r1), rmask, other=0)
    tmp12 = tl.load(in_ptr5 + (r1 + (768*x0)), rmask, other=0)
    tmp17 = tl.load(in_ptr6 + (x0), None)
    tmp24_load = tl.load(in_ptr7 + (0))
    tmp24 = tl.broadcast_to(tmp24_load, [XBLOCK, RBLOCK])
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 * tmp7
    tmp10 = tl.where(rmask, tmp8, 0)
    tmp11 = tl.sum(tmp10, 1)[:, None]
    tmp13 = tmp8 * tmp12
    tmp15 = tl.where(rmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None]
    tmp18 = 768.0
    tmp19 = tmp8 * tmp18
    tmp20 = tmp19 - tmp11
    tmp21 = tmp12 * tmp16
    tmp22 = tmp20 - tmp21
    tmp23 = tmp17 * tmp22
    tmp25 = 110100480 + r1 + (768*x0)
    tmp26 = tl.rand(tmp24, tmp25)
    tmp27 = 0.1
    tmp28 = tmp26 > tmp27
    tmp29 = tmp28.to(tl.float32)
    tmp30 = tmp29 * tmp23
    tmp31 = 1.1111111111111112
    tmp32 = tmp30 * tmp31
    tl.store(out_ptr3 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp23, rmask)
    tl.store(in_out_ptr0 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp32, rmask)
''')


# kernel path: /tmp/tmpom3r99qq/nh/cnhg6gpqec2uy6rayg2my5ll57z2wyk6zd7ihkfcl4pzuellf7pf.py
# Original ATen: aten._to_copy, aten.add, aten.gt, aten.mul, aten.native_layer_norm_backward, aten.view, prims.philox_rand_like

# aten._to_copy => convert_element_type_53
# aten.add => add_137
# aten.gt => gt_53
# aten.mul => mul_348, mul_349
# aten.native_layer_norm_backward => mul_342, mul_343, mul_344, mul_345, mul_346, sub_81, sub_82, sum_100, sum_101
# aten.view => view_363
# prims.philox_rand_like => philox_rand_like_53
triton_fused__to_copy_add_gt_mul_native_layer_norm_backward_philox_rand_like_view_29 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import persistent_reduction
from torch._inductor.utils import instance_descriptor

@persistent_reduction(
    size_hints=[2048, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*i64', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask, other=0)
    tmp1 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask, other=0)
    tmp3 = tl.load(in_ptr2 + (r1), rmask, other=0)
    tmp8 = tl.load(in_ptr3 + (r1 + (768*x0)), rmask, other=0)
    tmp13 = tl.load(in_ptr4 + (x0), None)
    tmp20_load = tl.load(in_ptr5 + (0))
    tmp20 = tl.broadcast_to(tmp20_load, [XBLOCK, RBLOCK])
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tmp6 = tl.where(rmask, tmp4, 0)
    tmp7 = tl.sum(tmp6, 1)[:, None]
    tmp9 = tmp4 * tmp8
    tmp11 = tl.where(rmask, tmp9, 0)
    tmp12 = tl.sum(tmp11, 1)[:, None]
    tmp14 = 768.0
    tmp15 = tmp4 * tmp14
    tmp16 = tmp15 - tmp7
    tmp17 = tmp8 * tmp12
    tmp18 = tmp16 - tmp17
    tmp19 = tmp13 * tmp18
    tmp21 = 108527616 + r1 + (768*x0)
    tmp22 = tl.rand(tmp20, tmp21)
    tmp23 = 0.1
    tmp24 = tmp22 > tmp23
    tmp25 = tmp24.to(tl.float32)
    tmp26 = tmp25 * tmp19
    tmp27 = 1.1111111111111112
    tmp28 = tmp26 * tmp27
    tl.store(out_ptr2 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp19, rmask)
    tl.store(in_out_ptr0 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp28, rmask)
''')


# kernel path: /tmp/tmpom3r99qq/7b/c7btorewoabudb5od7gw2si6binidnfrwyfisod6wptmozgwq6a7.py
# Original ATen: aten._softmax_backward_data, aten._to_copy, aten.div, aten.gt, aten.mul, prims.philox_rand_like

# aten._softmax_backward_data => mul_352, mul_353, sub_83, sum_105
# aten._to_copy => convert_element_type_54
# aten.div => div_42
# aten.gt => gt_54
# aten.mul => mul_350, mul_351
# prims.philox_rand_like => philox_rand_like_54
triton_fused__softmax_backward_data__to_copy_div_gt_mul_philox_rand_like_30 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import persistent_reduction
from torch._inductor.utils import instance_descriptor

@persistent_reduction(
    size_hints=[32768, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 24576
    rnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0_load = tl.load(in_ptr0 + (0))
    tmp0 = tl.broadcast_to(tmp0_load, [XBLOCK, RBLOCK])
    tmp6 = tl.load(in_ptr1 + (r1 + (512*x0)), rmask, other=0)
    tmp10 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask, other=0)
    tmp1 = 95944704 + r1 + (512*x0)
    tmp2 = tl.rand(tmp0, tmp1)
    tmp3 = 0.1
    tmp4 = tmp2 > tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp7 = tmp5 * tmp6
    tmp8 = 1.1111111111111112
    tmp9 = tmp7 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tl.where(rmask, tmp11, 0)
    tmp14 = tl.sum(tmp13, 1)[:, None]
    tmp15 = tmp10 * tmp14
    tmp16 = tmp11 - tmp15
    tmp17 = 8.0
    tmp18 = tmp16 / tmp17
    tl.store(out_ptr1 + (r1 + (512*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp18, rmask)
''')


# kernel path: /tmp/tmpom3r99qq/sy/csyylnjx5niil2stgxsfudg3ydym7igwrjkkhcvpp5q6dk3x7wew.py
# Original ATen: aten._to_copy, aten.add, aten.gt, aten.mul, aten.native_layer_norm_backward, aten.view, prims.philox_rand_like

# aten._to_copy => convert_element_type_55
# aten.add => add_138, add_139, add_140
# aten.gt => gt_55
# aten.mul => mul_361, mul_362
# aten.native_layer_norm_backward => mul_355, mul_356, mul_357, mul_358, mul_359, sub_85, sub_86, sum_109, sum_110
# aten.view => view_381
# prims.philox_rand_like => philox_rand_like_55
triton_fused__to_copy_add_gt_mul_native_layer_norm_backward_philox_rand_like_view_31 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import persistent_reduction
from torch._inductor.utils import instance_descriptor

@persistent_reduction(
    size_hints=[2048, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*i64', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=())]}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask, other=0)
    tmp1 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask, other=0)
    tmp3 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask, other=0)
    tmp5 = tl.load(in_ptr3 + (r1 + (768*x0)), rmask, other=0)
    tmp7 = tl.load(in_ptr4 + (r1), rmask, other=0)
    tmp12 = tl.load(in_ptr5 + (r1 + (768*x0)), rmask, other=0)
    tmp17 = tl.load(in_ptr6 + (x0), None)
    tmp24_load = tl.load(in_ptr7 + (0))
    tmp24 = tl.broadcast_to(tmp24_load, [XBLOCK, RBLOCK])
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 * tmp7
    tmp10 = tl.where(rmask, tmp8, 0)
    tmp11 = tl.sum(tmp10, 1)[:, None]
    tmp13 = tmp8 * tmp12
    tmp15 = tl.where(rmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None]
    tmp18 = 768.0
    tmp19 = tmp8 * tmp18
    tmp20 = tmp19 - tmp11
    tmp21 = tmp12 * tmp16
    tmp22 = tmp20 - tmp21
    tmp23 = tmp17 * tmp22
    tmp25 = 94371840 + r1 + (768*x0)
    tmp26 = tl.rand(tmp24, tmp25)
    tmp27 = 0.1
    tmp28 = tmp26 > tmp27
    tmp29 = tmp28.to(tl.float32)
    tmp30 = tmp29 * tmp23
    tmp31 = 1.1111111111111112
    tmp32 = tmp30 * tmp31
    tl.store(out_ptr3 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp23, rmask)
    tl.store(in_out_ptr0 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp32, rmask)
''')


# kernel path: /tmp/tmpom3r99qq/7d/c7dhj2owpsh6hsc7jmvwl5r5i654vjq6ipuidkmwsdggkjlc3hpc.py
# Original ATen: aten._to_copy, aten.add, aten.gt, aten.mul, aten.native_layer_norm_backward, aten.view, prims.philox_rand_like

# aten._to_copy => convert_element_type_56
# aten.add => add_143
# aten.gt => gt_56
# aten.mul => mul_377, mul_378
# aten.native_layer_norm_backward => mul_371, mul_372, mul_373, mul_374, mul_375, sub_88, sub_89, sum_115, sum_116
# aten.view => view_387
# prims.philox_rand_like => philox_rand_like_56
triton_fused__to_copy_add_gt_mul_native_layer_norm_backward_philox_rand_like_view_32 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import persistent_reduction
from torch._inductor.utils import instance_descriptor

@persistent_reduction(
    size_hints=[2048, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*i64', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask, other=0)
    tmp1 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask, other=0)
    tmp3 = tl.load(in_ptr2 + (r1), rmask, other=0)
    tmp8 = tl.load(in_ptr3 + (r1 + (768*x0)), rmask, other=0)
    tmp13 = tl.load(in_ptr4 + (x0), None)
    tmp20_load = tl.load(in_ptr5 + (0))
    tmp20 = tl.broadcast_to(tmp20_load, [XBLOCK, RBLOCK])
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tmp6 = tl.where(rmask, tmp4, 0)
    tmp7 = tl.sum(tmp6, 1)[:, None]
    tmp9 = tmp4 * tmp8
    tmp11 = tl.where(rmask, tmp9, 0)
    tmp12 = tl.sum(tmp11, 1)[:, None]
    tmp14 = 768.0
    tmp15 = tmp4 * tmp14
    tmp16 = tmp15 - tmp7
    tmp17 = tmp8 * tmp12
    tmp18 = tmp16 - tmp17
    tmp19 = tmp13 * tmp18
    tmp21 = 92798976 + r1 + (768*x0)
    tmp22 = tl.rand(tmp20, tmp21)
    tmp23 = 0.1
    tmp24 = tmp22 > tmp23
    tmp25 = tmp24.to(tl.float32)
    tmp26 = tmp25 * tmp19
    tmp27 = 1.1111111111111112
    tmp28 = tmp26 * tmp27
    tl.store(out_ptr2 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp19, rmask)
    tl.store(in_out_ptr0 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp28, rmask)
''')


# kernel path: /tmp/tmpom3r99qq/eh/cehqpcao4esc6gp7e5v3k6pj6suzh367boslbqnlg6orsqbrz6iv.py
# Original ATen: aten._softmax_backward_data, aten._to_copy, aten.div, aten.gt, aten.mul, prims.philox_rand_like

# aten._softmax_backward_data => mul_381, mul_382, sub_90, sum_120
# aten._to_copy => convert_element_type_57
# aten.div => div_45
# aten.gt => gt_57
# aten.mul => mul_379, mul_380
# prims.philox_rand_like => philox_rand_like_57
triton_fused__softmax_backward_data__to_copy_div_gt_mul_philox_rand_like_33 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import persistent_reduction
from torch._inductor.utils import instance_descriptor

@persistent_reduction(
    size_hints=[32768, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 24576
    rnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0_load = tl.load(in_ptr0 + (0))
    tmp0 = tl.broadcast_to(tmp0_load, [XBLOCK, RBLOCK])
    tmp6 = tl.load(in_ptr1 + (r1 + (512*x0)), rmask, other=0)
    tmp10 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask, other=0)
    tmp1 = 80216064 + r1 + (512*x0)
    tmp2 = tl.rand(tmp0, tmp1)
    tmp3 = 0.1
    tmp4 = tmp2 > tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp7 = tmp5 * tmp6
    tmp8 = 1.1111111111111112
    tmp9 = tmp7 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tl.where(rmask, tmp11, 0)
    tmp14 = tl.sum(tmp13, 1)[:, None]
    tmp15 = tmp10 * tmp14
    tmp16 = tmp11 - tmp15
    tmp17 = 8.0
    tmp18 = tmp16 / tmp17
    tl.store(out_ptr1 + (r1 + (512*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp18, rmask)
''')


# kernel path: /tmp/tmpom3r99qq/3i/c3i7okkxtlrqb3yfkk6o6ewfzhvcamytilmakmuumwfh4ehs7ahm.py
# Original ATen: aten._to_copy, aten.add, aten.gt, aten.mul, aten.native_layer_norm_backward, aten.view, prims.philox_rand_like

# aten._to_copy => convert_element_type_58
# aten.add => add_144, add_145, add_146
# aten.gt => gt_58
# aten.mul => mul_390, mul_391
# aten.native_layer_norm_backward => mul_384, mul_385, mul_386, mul_387, mul_388, sub_92, sub_93, sum_124, sum_125
# aten.view => view_405
# prims.philox_rand_like => philox_rand_like_58
triton_fused__to_copy_add_gt_mul_native_layer_norm_backward_philox_rand_like_view_34 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import persistent_reduction
from torch._inductor.utils import instance_descriptor

@persistent_reduction(
    size_hints=[2048, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*i64', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=())]}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask, other=0)
    tmp1 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask, other=0)
    tmp3 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask, other=0)
    tmp5 = tl.load(in_ptr3 + (r1 + (768*x0)), rmask, other=0)
    tmp7 = tl.load(in_ptr4 + (r1), rmask, other=0)
    tmp12 = tl.load(in_ptr5 + (r1 + (768*x0)), rmask, other=0)
    tmp17 = tl.load(in_ptr6 + (x0), None)
    tmp24_load = tl.load(in_ptr7 + (0))
    tmp24 = tl.broadcast_to(tmp24_load, [XBLOCK, RBLOCK])
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 * tmp7
    tmp10 = tl.where(rmask, tmp8, 0)
    tmp11 = tl.sum(tmp10, 1)[:, None]
    tmp13 = tmp8 * tmp12
    tmp15 = tl.where(rmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None]
    tmp18 = 768.0
    tmp19 = tmp8 * tmp18
    tmp20 = tmp19 - tmp11
    tmp21 = tmp12 * tmp16
    tmp22 = tmp20 - tmp21
    tmp23 = tmp17 * tmp22
    tmp25 = 78643200 + r1 + (768*x0)
    tmp26 = tl.rand(tmp24, tmp25)
    tmp27 = 0.1
    tmp28 = tmp26 > tmp27
    tmp29 = tmp28.to(tl.float32)
    tmp30 = tmp29 * tmp23
    tmp31 = 1.1111111111111112
    tmp32 = tmp30 * tmp31
    tl.store(out_ptr3 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp23, rmask)
    tl.store(in_out_ptr0 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp32, rmask)
''')


# kernel path: /tmp/tmpom3r99qq/ji/cjih6xxnsprfbbumljykkgvszc3e5bqqrnjuj72dbhpisegc5oqd.py
# Original ATen: aten._to_copy, aten.add, aten.gt, aten.mul, aten.native_layer_norm_backward, aten.view, prims.philox_rand_like

# aten._to_copy => convert_element_type_59
# aten.add => add_149
# aten.gt => gt_59
# aten.mul => mul_406, mul_407
# aten.native_layer_norm_backward => mul_400, mul_401, mul_402, mul_403, mul_404, sub_95, sub_96, sum_130, sum_131
# aten.view => view_411
# prims.philox_rand_like => philox_rand_like_59
triton_fused__to_copy_add_gt_mul_native_layer_norm_backward_philox_rand_like_view_35 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import persistent_reduction
from torch._inductor.utils import instance_descriptor

@persistent_reduction(
    size_hints=[2048, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*i64', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask, other=0)
    tmp1 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask, other=0)
    tmp3 = tl.load(in_ptr2 + (r1), rmask, other=0)
    tmp8 = tl.load(in_ptr3 + (r1 + (768*x0)), rmask, other=0)
    tmp13 = tl.load(in_ptr4 + (x0), None)
    tmp20_load = tl.load(in_ptr5 + (0))
    tmp20 = tl.broadcast_to(tmp20_load, [XBLOCK, RBLOCK])
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tmp6 = tl.where(rmask, tmp4, 0)
    tmp7 = tl.sum(tmp6, 1)[:, None]
    tmp9 = tmp4 * tmp8
    tmp11 = tl.where(rmask, tmp9, 0)
    tmp12 = tl.sum(tmp11, 1)[:, None]
    tmp14 = 768.0
    tmp15 = tmp4 * tmp14
    tmp16 = tmp15 - tmp7
    tmp17 = tmp8 * tmp12
    tmp18 = tmp16 - tmp17
    tmp19 = tmp13 * tmp18
    tmp21 = 77070336 + r1 + (768*x0)
    tmp22 = tl.rand(tmp20, tmp21)
    tmp23 = 0.1
    tmp24 = tmp22 > tmp23
    tmp25 = tmp24.to(tl.float32)
    tmp26 = tmp25 * tmp19
    tmp27 = 1.1111111111111112
    tmp28 = tmp26 * tmp27
    tl.store(out_ptr2 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp19, rmask)
    tl.store(in_out_ptr0 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp28, rmask)
''')


# kernel path: /tmp/tmpom3r99qq/75/c75pqb4ladszubrpryibirvsx6sqbllp6hc2me6r7jxzgiqav2bp.py
# Original ATen: aten._softmax_backward_data, aten._to_copy, aten.div, aten.gt, aten.mul, prims.philox_rand_like

# aten._softmax_backward_data => mul_410, mul_411, sub_97, sum_135
# aten._to_copy => convert_element_type_60
# aten.div => div_48
# aten.gt => gt_60
# aten.mul => mul_408, mul_409
# prims.philox_rand_like => philox_rand_like_60
triton_fused__softmax_backward_data__to_copy_div_gt_mul_philox_rand_like_36 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import persistent_reduction
from torch._inductor.utils import instance_descriptor

@persistent_reduction(
    size_hints=[32768, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 24576
    rnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0_load = tl.load(in_ptr0 + (0))
    tmp0 = tl.broadcast_to(tmp0_load, [XBLOCK, RBLOCK])
    tmp6 = tl.load(in_ptr1 + (r1 + (512*x0)), rmask, other=0)
    tmp10 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask, other=0)
    tmp1 = 64487424 + r1 + (512*x0)
    tmp2 = tl.rand(tmp0, tmp1)
    tmp3 = 0.1
    tmp4 = tmp2 > tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp7 = tmp5 * tmp6
    tmp8 = 1.1111111111111112
    tmp9 = tmp7 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tl.where(rmask, tmp11, 0)
    tmp14 = tl.sum(tmp13, 1)[:, None]
    tmp15 = tmp10 * tmp14
    tmp16 = tmp11 - tmp15
    tmp17 = 8.0
    tmp18 = tmp16 / tmp17
    tl.store(out_ptr1 + (r1 + (512*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp18, rmask)
''')


# kernel path: /tmp/tmpom3r99qq/5g/c5gz3dtozmk2t7zylz3nrisbrumdr6tq57ouu4dgcothztkvl2mg.py
# Original ATen: aten._to_copy, aten.add, aten.gt, aten.mul, aten.native_layer_norm_backward, aten.view, prims.philox_rand_like

# aten._to_copy => convert_element_type_61
# aten.add => add_150, add_151, add_152
# aten.gt => gt_61
# aten.mul => mul_419, mul_420
# aten.native_layer_norm_backward => mul_413, mul_414, mul_415, mul_416, mul_417, sub_100, sub_99, sum_139, sum_140
# aten.view => view_429
# prims.philox_rand_like => philox_rand_like_61
triton_fused__to_copy_add_gt_mul_native_layer_norm_backward_philox_rand_like_view_37 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import persistent_reduction
from torch._inductor.utils import instance_descriptor

@persistent_reduction(
    size_hints=[2048, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*i64', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=())]}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask, other=0)
    tmp1 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask, other=0)
    tmp3 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask, other=0)
    tmp5 = tl.load(in_ptr3 + (r1 + (768*x0)), rmask, other=0)
    tmp7 = tl.load(in_ptr4 + (r1), rmask, other=0)
    tmp12 = tl.load(in_ptr5 + (r1 + (768*x0)), rmask, other=0)
    tmp17 = tl.load(in_ptr6 + (x0), None)
    tmp24_load = tl.load(in_ptr7 + (0))
    tmp24 = tl.broadcast_to(tmp24_load, [XBLOCK, RBLOCK])
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 * tmp7
    tmp10 = tl.where(rmask, tmp8, 0)
    tmp11 = tl.sum(tmp10, 1)[:, None]
    tmp13 = tmp8 * tmp12
    tmp15 = tl.where(rmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None]
    tmp18 = 768.0
    tmp19 = tmp8 * tmp18
    tmp20 = tmp19 - tmp11
    tmp21 = tmp12 * tmp16
    tmp22 = tmp20 - tmp21
    tmp23 = tmp17 * tmp22
    tmp25 = 62914560 + r1 + (768*x0)
    tmp26 = tl.rand(tmp24, tmp25)
    tmp27 = 0.1
    tmp28 = tmp26 > tmp27
    tmp29 = tmp28.to(tl.float32)
    tmp30 = tmp29 * tmp23
    tmp31 = 1.1111111111111112
    tmp32 = tmp30 * tmp31
    tl.store(out_ptr3 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp23, rmask)
    tl.store(in_out_ptr0 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp32, rmask)
''')


# kernel path: /tmp/tmpom3r99qq/sj/csjsy77dmpirzv72q265zqwn4jeicnyljsjkaeqvsw3gqbao36b4.py
# Original ATen: aten._to_copy, aten.add, aten.gt, aten.mul, aten.native_layer_norm_backward, aten.view, prims.philox_rand_like

# aten._to_copy => convert_element_type_62
# aten.add => add_155
# aten.gt => gt_62
# aten.mul => mul_435, mul_436
# aten.native_layer_norm_backward => mul_429, mul_430, mul_431, mul_432, mul_433, sub_102, sub_103, sum_145, sum_146
# aten.view => view_435
# prims.philox_rand_like => philox_rand_like_62
triton_fused__to_copy_add_gt_mul_native_layer_norm_backward_philox_rand_like_view_38 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import persistent_reduction
from torch._inductor.utils import instance_descriptor

@persistent_reduction(
    size_hints=[2048, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*i64', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask, other=0)
    tmp1 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask, other=0)
    tmp3 = tl.load(in_ptr2 + (r1), rmask, other=0)
    tmp8 = tl.load(in_ptr3 + (r1 + (768*x0)), rmask, other=0)
    tmp13 = tl.load(in_ptr4 + (x0), None)
    tmp20_load = tl.load(in_ptr5 + (0))
    tmp20 = tl.broadcast_to(tmp20_load, [XBLOCK, RBLOCK])
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tmp6 = tl.where(rmask, tmp4, 0)
    tmp7 = tl.sum(tmp6, 1)[:, None]
    tmp9 = tmp4 * tmp8
    tmp11 = tl.where(rmask, tmp9, 0)
    tmp12 = tl.sum(tmp11, 1)[:, None]
    tmp14 = 768.0
    tmp15 = tmp4 * tmp14
    tmp16 = tmp15 - tmp7
    tmp17 = tmp8 * tmp12
    tmp18 = tmp16 - tmp17
    tmp19 = tmp13 * tmp18
    tmp21 = 61341696 + r1 + (768*x0)
    tmp22 = tl.rand(tmp20, tmp21)
    tmp23 = 0.1
    tmp24 = tmp22 > tmp23
    tmp25 = tmp24.to(tl.float32)
    tmp26 = tmp25 * tmp19
    tmp27 = 1.1111111111111112
    tmp28 = tmp26 * tmp27
    tl.store(out_ptr2 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp19, rmask)
    tl.store(in_out_ptr0 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp28, rmask)
''')


# kernel path: /tmp/tmpom3r99qq/pg/cpgzqyzrkdrj3qhe22sme4iqv5oqi2otn5w74u5hgix5klxpsrpe.py
# Original ATen: aten._softmax_backward_data, aten._to_copy, aten.div, aten.gt, aten.mul, prims.philox_rand_like

# aten._softmax_backward_data => mul_439, mul_440, sub_104, sum_150
# aten._to_copy => convert_element_type_63
# aten.div => div_51
# aten.gt => gt_63
# aten.mul => mul_437, mul_438
# prims.philox_rand_like => philox_rand_like_63
triton_fused__softmax_backward_data__to_copy_div_gt_mul_philox_rand_like_39 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import persistent_reduction
from torch._inductor.utils import instance_descriptor

@persistent_reduction(
    size_hints=[32768, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 24576
    rnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0_load = tl.load(in_ptr0 + (0))
    tmp0 = tl.broadcast_to(tmp0_load, [XBLOCK, RBLOCK])
    tmp6 = tl.load(in_ptr1 + (r1 + (512*x0)), rmask, other=0)
    tmp10 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask, other=0)
    tmp1 = 48758784 + r1 + (512*x0)
    tmp2 = tl.rand(tmp0, tmp1)
    tmp3 = 0.1
    tmp4 = tmp2 > tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp7 = tmp5 * tmp6
    tmp8 = 1.1111111111111112
    tmp9 = tmp7 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tl.where(rmask, tmp11, 0)
    tmp14 = tl.sum(tmp13, 1)[:, None]
    tmp15 = tmp10 * tmp14
    tmp16 = tmp11 - tmp15
    tmp17 = 8.0
    tmp18 = tmp16 / tmp17
    tl.store(out_ptr1 + (r1 + (512*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp18, rmask)
''')


# kernel path: /tmp/tmpom3r99qq/2e/c2eiz7m2z4n2gon4qwqgapznhh5hfxfmfputkv3ycmzv76vgkiil.py
# Original ATen: aten._to_copy, aten.add, aten.gt, aten.mul, aten.native_layer_norm_backward, aten.view, prims.philox_rand_like

# aten._to_copy => convert_element_type_64
# aten.add => add_156, add_157, add_158
# aten.gt => gt_64
# aten.mul => mul_448, mul_449
# aten.native_layer_norm_backward => mul_442, mul_443, mul_444, mul_445, mul_446, sub_106, sub_107, sum_154, sum_155
# aten.view => view_453
# prims.philox_rand_like => philox_rand_like_64
triton_fused__to_copy_add_gt_mul_native_layer_norm_backward_philox_rand_like_view_40 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import persistent_reduction
from torch._inductor.utils import instance_descriptor

@persistent_reduction(
    size_hints=[2048, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*i64', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=())]}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask, other=0)
    tmp1 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask, other=0)
    tmp3 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask, other=0)
    tmp5 = tl.load(in_ptr3 + (r1 + (768*x0)), rmask, other=0)
    tmp7 = tl.load(in_ptr4 + (r1), rmask, other=0)
    tmp12 = tl.load(in_ptr5 + (r1 + (768*x0)), rmask, other=0)
    tmp17 = tl.load(in_ptr6 + (x0), None)
    tmp24_load = tl.load(in_ptr7 + (0))
    tmp24 = tl.broadcast_to(tmp24_load, [XBLOCK, RBLOCK])
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 * tmp7
    tmp10 = tl.where(rmask, tmp8, 0)
    tmp11 = tl.sum(tmp10, 1)[:, None]
    tmp13 = tmp8 * tmp12
    tmp15 = tl.where(rmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None]
    tmp18 = 768.0
    tmp19 = tmp8 * tmp18
    tmp20 = tmp19 - tmp11
    tmp21 = tmp12 * tmp16
    tmp22 = tmp20 - tmp21
    tmp23 = tmp17 * tmp22
    tmp25 = 47185920 + r1 + (768*x0)
    tmp26 = tl.rand(tmp24, tmp25)
    tmp27 = 0.1
    tmp28 = tmp26 > tmp27
    tmp29 = tmp28.to(tl.float32)
    tmp30 = tmp29 * tmp23
    tmp31 = 1.1111111111111112
    tmp32 = tmp30 * tmp31
    tl.store(out_ptr3 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp23, rmask)
    tl.store(in_out_ptr0 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp32, rmask)
''')


# kernel path: /tmp/tmpom3r99qq/3d/c3dafyfxwz4a2muhgpxr3iahx6iel35hhs3natotifuzn6jdejm6.py
# Original ATen: aten._to_copy, aten.add, aten.gt, aten.mul, aten.native_layer_norm_backward, aten.view, prims.philox_rand_like

# aten._to_copy => convert_element_type_65
# aten.add => add_161
# aten.gt => gt_65
# aten.mul => mul_464, mul_465
# aten.native_layer_norm_backward => mul_458, mul_459, mul_460, mul_461, mul_462, sub_109, sub_110, sum_160, sum_161
# aten.view => view_459
# prims.philox_rand_like => philox_rand_like_65
triton_fused__to_copy_add_gt_mul_native_layer_norm_backward_philox_rand_like_view_41 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import persistent_reduction
from torch._inductor.utils import instance_descriptor

@persistent_reduction(
    size_hints=[2048, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*i64', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask, other=0)
    tmp1 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask, other=0)
    tmp3 = tl.load(in_ptr2 + (r1), rmask, other=0)
    tmp8 = tl.load(in_ptr3 + (r1 + (768*x0)), rmask, other=0)
    tmp13 = tl.load(in_ptr4 + (x0), None)
    tmp20_load = tl.load(in_ptr5 + (0))
    tmp20 = tl.broadcast_to(tmp20_load, [XBLOCK, RBLOCK])
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tmp6 = tl.where(rmask, tmp4, 0)
    tmp7 = tl.sum(tmp6, 1)[:, None]
    tmp9 = tmp4 * tmp8
    tmp11 = tl.where(rmask, tmp9, 0)
    tmp12 = tl.sum(tmp11, 1)[:, None]
    tmp14 = 768.0
    tmp15 = tmp4 * tmp14
    tmp16 = tmp15 - tmp7
    tmp17 = tmp8 * tmp12
    tmp18 = tmp16 - tmp17
    tmp19 = tmp13 * tmp18
    tmp21 = 45613056 + r1 + (768*x0)
    tmp22 = tl.rand(tmp20, tmp21)
    tmp23 = 0.1
    tmp24 = tmp22 > tmp23
    tmp25 = tmp24.to(tl.float32)
    tmp26 = tmp25 * tmp19
    tmp27 = 1.1111111111111112
    tmp28 = tmp26 * tmp27
    tl.store(out_ptr2 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp19, rmask)
    tl.store(in_out_ptr0 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp28, rmask)
''')


# kernel path: /tmp/tmpom3r99qq/xj/cxj3j3ytf4jc63mlkvlh2noopc7zbnp5mb5jr43psrmgbcdlacyw.py
# Original ATen: aten._softmax_backward_data, aten._to_copy, aten.div, aten.gt, aten.mul, prims.philox_rand_like

# aten._softmax_backward_data => mul_468, mul_469, sub_111, sum_165
# aten._to_copy => convert_element_type_66
# aten.div => div_54
# aten.gt => gt_66
# aten.mul => mul_466, mul_467
# prims.philox_rand_like => philox_rand_like_66
triton_fused__softmax_backward_data__to_copy_div_gt_mul_philox_rand_like_42 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import persistent_reduction
from torch._inductor.utils import instance_descriptor

@persistent_reduction(
    size_hints=[32768, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 24576
    rnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0_load = tl.load(in_ptr0 + (0))
    tmp0 = tl.broadcast_to(tmp0_load, [XBLOCK, RBLOCK])
    tmp6 = tl.load(in_ptr1 + (r1 + (512*x0)), rmask, other=0)
    tmp10 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask, other=0)
    tmp1 = 33030144 + r1 + (512*x0)
    tmp2 = tl.rand(tmp0, tmp1)
    tmp3 = 0.1
    tmp4 = tmp2 > tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp7 = tmp5 * tmp6
    tmp8 = 1.1111111111111112
    tmp9 = tmp7 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tl.where(rmask, tmp11, 0)
    tmp14 = tl.sum(tmp13, 1)[:, None]
    tmp15 = tmp10 * tmp14
    tmp16 = tmp11 - tmp15
    tmp17 = 8.0
    tmp18 = tmp16 / tmp17
    tl.store(out_ptr1 + (r1 + (512*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp18, rmask)
''')


# kernel path: /tmp/tmpom3r99qq/zb/czbaec7eiw77dinqiwrtzi5ucyqlzncjgzo4axzl5we5wihkg7sa.py
# Original ATen: aten._to_copy, aten.add, aten.gt, aten.mul, aten.native_layer_norm_backward, aten.view, prims.philox_rand_like

# aten._to_copy => convert_element_type_67
# aten.add => add_162, add_163, add_164
# aten.gt => gt_67
# aten.mul => mul_477, mul_478
# aten.native_layer_norm_backward => mul_471, mul_472, mul_473, mul_474, mul_475, sub_113, sub_114, sum_169, sum_170
# aten.view => view_477
# prims.philox_rand_like => philox_rand_like_67
triton_fused__to_copy_add_gt_mul_native_layer_norm_backward_philox_rand_like_view_43 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import persistent_reduction
from torch._inductor.utils import instance_descriptor

@persistent_reduction(
    size_hints=[2048, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*i64', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=())]}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask, other=0)
    tmp1 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask, other=0)
    tmp3 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask, other=0)
    tmp5 = tl.load(in_ptr3 + (r1 + (768*x0)), rmask, other=0)
    tmp7 = tl.load(in_ptr4 + (r1), rmask, other=0)
    tmp12 = tl.load(in_ptr5 + (r1 + (768*x0)), rmask, other=0)
    tmp17 = tl.load(in_ptr6 + (x0), None)
    tmp24_load = tl.load(in_ptr7 + (0))
    tmp24 = tl.broadcast_to(tmp24_load, [XBLOCK, RBLOCK])
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 * tmp7
    tmp10 = tl.where(rmask, tmp8, 0)
    tmp11 = tl.sum(tmp10, 1)[:, None]
    tmp13 = tmp8 * tmp12
    tmp15 = tl.where(rmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None]
    tmp18 = 768.0
    tmp19 = tmp8 * tmp18
    tmp20 = tmp19 - tmp11
    tmp21 = tmp12 * tmp16
    tmp22 = tmp20 - tmp21
    tmp23 = tmp17 * tmp22
    tmp25 = 31457280 + r1 + (768*x0)
    tmp26 = tl.rand(tmp24, tmp25)
    tmp27 = 0.1
    tmp28 = tmp26 > tmp27
    tmp29 = tmp28.to(tl.float32)
    tmp30 = tmp29 * tmp23
    tmp31 = 1.1111111111111112
    tmp32 = tmp30 * tmp31
    tl.store(out_ptr3 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp23, rmask)
    tl.store(in_out_ptr0 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp32, rmask)
''')


# kernel path: /tmp/tmpom3r99qq/zm/czmnvkhqga27bc3zhyefx2xd7zik3b4zbh662hfor322uqou6vdp.py
# Original ATen: aten._to_copy, aten.add, aten.gt, aten.mul, aten.native_layer_norm_backward, aten.view, prims.philox_rand_like

# aten._to_copy => convert_element_type_68
# aten.add => add_167
# aten.gt => gt_68
# aten.mul => mul_493, mul_494
# aten.native_layer_norm_backward => mul_487, mul_488, mul_489, mul_490, mul_491, sub_116, sub_117, sum_175, sum_176
# aten.view => view_483
# prims.philox_rand_like => philox_rand_like_68
triton_fused__to_copy_add_gt_mul_native_layer_norm_backward_philox_rand_like_view_44 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import persistent_reduction
from torch._inductor.utils import instance_descriptor

@persistent_reduction(
    size_hints=[2048, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*i64', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask, other=0)
    tmp1 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask, other=0)
    tmp3 = tl.load(in_ptr2 + (r1), rmask, other=0)
    tmp8 = tl.load(in_ptr3 + (r1 + (768*x0)), rmask, other=0)
    tmp13 = tl.load(in_ptr4 + (x0), None)
    tmp20_load = tl.load(in_ptr5 + (0))
    tmp20 = tl.broadcast_to(tmp20_load, [XBLOCK, RBLOCK])
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tmp6 = tl.where(rmask, tmp4, 0)
    tmp7 = tl.sum(tmp6, 1)[:, None]
    tmp9 = tmp4 * tmp8
    tmp11 = tl.where(rmask, tmp9, 0)
    tmp12 = tl.sum(tmp11, 1)[:, None]
    tmp14 = 768.0
    tmp15 = tmp4 * tmp14
    tmp16 = tmp15 - tmp7
    tmp17 = tmp8 * tmp12
    tmp18 = tmp16 - tmp17
    tmp19 = tmp13 * tmp18
    tmp21 = 29884416 + r1 + (768*x0)
    tmp22 = tl.rand(tmp20, tmp21)
    tmp23 = 0.1
    tmp24 = tmp22 > tmp23
    tmp25 = tmp24.to(tl.float32)
    tmp26 = tmp25 * tmp19
    tmp27 = 1.1111111111111112
    tmp28 = tmp26 * tmp27
    tl.store(out_ptr2 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp19, rmask)
    tl.store(in_out_ptr0 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp28, rmask)
''')


# kernel path: /tmp/tmpom3r99qq/fc/cfcx7ct6wizxhprsxucjaawahovdmapnfapowmfgofqvx3yxiitv.py
# Original ATen: aten._softmax_backward_data, aten._to_copy, aten.div, aten.gt, aten.mul, prims.philox_rand_like

# aten._softmax_backward_data => mul_497, mul_498, sub_118, sum_180
# aten._to_copy => convert_element_type_69
# aten.div => div_57
# aten.gt => gt_69
# aten.mul => mul_495, mul_496
# prims.philox_rand_like => philox_rand_like_69
triton_fused__softmax_backward_data__to_copy_div_gt_mul_philox_rand_like_45 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import persistent_reduction
from torch._inductor.utils import instance_descriptor

@persistent_reduction(
    size_hints=[32768, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 24576
    rnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0_load = tl.load(in_ptr0 + (0))
    tmp0 = tl.broadcast_to(tmp0_load, [XBLOCK, RBLOCK])
    tmp6 = tl.load(in_ptr1 + (r1 + (512*x0)), rmask, other=0)
    tmp10 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask, other=0)
    tmp1 = 17301504 + r1 + (512*x0)
    tmp2 = tl.rand(tmp0, tmp1)
    tmp3 = 0.1
    tmp4 = tmp2 > tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp7 = tmp5 * tmp6
    tmp8 = 1.1111111111111112
    tmp9 = tmp7 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tl.where(rmask, tmp11, 0)
    tmp14 = tl.sum(tmp13, 1)[:, None]
    tmp15 = tmp10 * tmp14
    tmp16 = tmp11 - tmp15
    tmp17 = 8.0
    tmp18 = tmp16 / tmp17
    tl.store(out_ptr1 + (r1 + (512*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp18, rmask)
''')


# kernel path: /tmp/tmpom3r99qq/6l/c6lnomjbrmumuxnhphbqwlpexxzgmw4f3trhtjq653op6ys7jv2n.py
# Original ATen: aten._to_copy, aten.add, aten.gt, aten.mul, aten.native_layer_norm_backward, aten.view, prims.philox_rand_like

# aten._to_copy => convert_element_type_70
# aten.add => add_168, add_169, add_170
# aten.gt => gt_70
# aten.mul => mul_506, mul_507
# aten.native_layer_norm_backward => mul_500, mul_501, mul_502, mul_503, mul_504, sub_120, sub_121, sum_184, sum_185
# aten.view => view_501
# prims.philox_rand_like => philox_rand_like_70
triton_fused__to_copy_add_gt_mul_native_layer_norm_backward_philox_rand_like_view_46 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import persistent_reduction
from torch._inductor.utils import instance_descriptor

@persistent_reduction(
    size_hints=[2048, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*i64', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=())]}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask, other=0)
    tmp1 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask, other=0)
    tmp3 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask, other=0)
    tmp5 = tl.load(in_ptr3 + (r1 + (768*x0)), rmask, other=0)
    tmp7 = tl.load(in_ptr4 + (r1), rmask, other=0)
    tmp12 = tl.load(in_ptr5 + (r1 + (768*x0)), rmask, other=0)
    tmp17 = tl.load(in_ptr6 + (x0), None)
    tmp24_load = tl.load(in_ptr7 + (0))
    tmp24 = tl.broadcast_to(tmp24_load, [XBLOCK, RBLOCK])
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 * tmp7
    tmp10 = tl.where(rmask, tmp8, 0)
    tmp11 = tl.sum(tmp10, 1)[:, None]
    tmp13 = tmp8 * tmp12
    tmp15 = tl.where(rmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None]
    tmp18 = 768.0
    tmp19 = tmp8 * tmp18
    tmp20 = tmp19 - tmp11
    tmp21 = tmp12 * tmp16
    tmp22 = tmp20 - tmp21
    tmp23 = tmp17 * tmp22
    tmp25 = 15728640 + r1 + (768*x0)
    tmp26 = tl.rand(tmp24, tmp25)
    tmp27 = 0.1
    tmp28 = tmp26 > tmp27
    tmp29 = tmp28.to(tl.float32)
    tmp30 = tmp29 * tmp23
    tmp31 = 1.1111111111111112
    tmp32 = tmp30 * tmp31
    tl.store(out_ptr3 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp23, rmask)
    tl.store(in_out_ptr0 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp32, rmask)
''')


# kernel path: /tmp/tmpom3r99qq/nr/cnrmjuw7yoanhfxdmxo3tgku5bry5p6exwdlxcqpxrgujy4yf3ld.py
# Original ATen: aten._to_copy, aten.add, aten.gt, aten.mul, aten.native_layer_norm_backward, aten.view, prims.philox_rand_like

# aten._to_copy => convert_element_type_71
# aten.add => add_173
# aten.gt => gt_71
# aten.mul => mul_522, mul_523
# aten.native_layer_norm_backward => mul_516, mul_517, mul_518, mul_519, mul_520, sub_123, sub_124, sum_190, sum_191
# aten.view => view_507
# prims.philox_rand_like => philox_rand_like_71
triton_fused__to_copy_add_gt_mul_native_layer_norm_backward_philox_rand_like_view_47 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import persistent_reduction
from torch._inductor.utils import instance_descriptor

@persistent_reduction(
    size_hints=[2048, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*i64', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask, other=0)
    tmp1 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask, other=0)
    tmp3 = tl.load(in_ptr2 + (r1), rmask, other=0)
    tmp8 = tl.load(in_ptr3 + (r1 + (768*x0)), rmask, other=0)
    tmp13 = tl.load(in_ptr4 + (x0), None)
    tmp20_load = tl.load(in_ptr5 + (0))
    tmp20 = tl.broadcast_to(tmp20_load, [XBLOCK, RBLOCK])
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tmp6 = tl.where(rmask, tmp4, 0)
    tmp7 = tl.sum(tmp6, 1)[:, None]
    tmp9 = tmp4 * tmp8
    tmp11 = tl.where(rmask, tmp9, 0)
    tmp12 = tl.sum(tmp11, 1)[:, None]
    tmp14 = 768.0
    tmp15 = tmp4 * tmp14
    tmp16 = tmp15 - tmp7
    tmp17 = tmp8 * tmp12
    tmp18 = tmp16 - tmp17
    tmp19 = tmp13 * tmp18
    tmp21 = 14155776 + r1 + (768*x0)
    tmp22 = tl.rand(tmp20, tmp21)
    tmp23 = 0.1
    tmp24 = tmp22 > tmp23
    tmp25 = tmp24.to(tl.float32)
    tmp26 = tmp25 * tmp19
    tmp27 = 1.1111111111111112
    tmp28 = tmp26 * tmp27
    tl.store(out_ptr2 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp19, rmask)
    tl.store(in_out_ptr0 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp28, rmask)
''')


# kernel path: /tmp/tmpom3r99qq/lv/clvtdng6xhyib4z627hfr66wv33ec22kukfkp3drohgwgrohbwlu.py
# Original ATen: aten._softmax_backward_data, aten._to_copy, aten.div, aten.gt, aten.mul, prims.philox_rand_like

# aten._softmax_backward_data => mul_526, mul_527, sub_125, sum_195
# aten._to_copy => convert_element_type_72
# aten.div => div_60
# aten.gt => gt_72
# aten.mul => mul_524, mul_525
# prims.philox_rand_like => philox_rand_like_72
triton_fused__softmax_backward_data__to_copy_div_gt_mul_philox_rand_like_48 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import persistent_reduction
from torch._inductor.utils import instance_descriptor

@persistent_reduction(
    size_hints=[32768, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 24576
    rnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0_load = tl.load(in_ptr0 + (0))
    tmp0 = tl.broadcast_to(tmp0_load, [XBLOCK, RBLOCK])
    tmp6 = tl.load(in_ptr1 + (r1 + (512*x0)), rmask, other=0)
    tmp10 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask, other=0)
    tmp1 = 1572864 + r1 + (512*x0)
    tmp2 = tl.rand(tmp0, tmp1)
    tmp3 = 0.1
    tmp4 = tmp2 > tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp7 = tmp5 * tmp6
    tmp8 = 1.1111111111111112
    tmp9 = tmp7 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tl.where(rmask, tmp11, 0)
    tmp14 = tl.sum(tmp13, 1)[:, None]
    tmp15 = tmp10 * tmp14
    tmp16 = tmp11 - tmp15
    tmp17 = 8.0
    tmp18 = tmp16 / tmp17
    tl.store(out_ptr1 + (r1 + (512*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp18, rmask)
''')


# kernel path: /tmp/tmpom3r99qq/ay/cayauqntqy2zh3i72uxi4cmhitrsjxfotyzkkj66f7b6hmy5dbdm.py
# Original ATen: aten.embedding_dense_backward

# aten.embedding_dense_backward => full_2, index_put_1, scalar_tensor, where_1
triton_fused_embedding_dense_backward_49 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[2048], filename=__file__, meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp0, xmask)
''')


# kernel path: /tmp/tmpom3r99qq/bl/cblesc54nxyuvskzwzo5aj7ybhpg2ndfuqzbotewmn5pw2afnscd.py
# Original ATen: aten.embedding_dense_backward

# aten.embedding_dense_backward => full_3, index_put_2, scalar_tensor, where_2
triton_fused_embedding_dense_backward_50 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[33554432], filename=__file__, meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 23440896
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp0, xmask)
''')


# kernel path: /tmp/tmpom3r99qq/wf/cwfznrvmq4dyajg3lunidmiyjc2myda6f4qwv3ggek56kdf25v2d.py
# Original ATen: aten._to_copy, aten.add, aten.embedding_dense_backward, aten.gt, aten.mul, aten.native_layer_norm_backward, prims.philox_rand_like

# aten._to_copy => convert_element_type_73
# aten.add => add_174, add_175, add_176
# aten.embedding_dense_backward => full_2, full_3, index_put_1, index_put_2, scalar_tensor, where_1, where_2
# aten.gt => gt_73
# aten.mul => mul_528, mul_529
# aten.native_layer_norm_backward => mul_531, mul_532, mul_533, mul_534, mul_535, sub_127, sub_128, sum_199, sum_200
# prims.philox_rand_like => philox_rand_like_73
triton_fused__to_copy_add_embedding_dense_backward_gt_mul_native_layer_norm_backward_philox_rand_like_51 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import persistent_reduction
from torch._inductor.utils import instance_descriptor

@persistent_reduction(
    size_hints=[2048, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*i64', 9: '*i64', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: 'i32', 14: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0', 'out_ptr3', 'out_ptr4'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14), equal_to_1=())]}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    x2 = xindex % 512
    tmp0_load = tl.load(in_ptr0 + (0))
    tmp0 = tl.broadcast_to(tmp0_load, [XBLOCK, RBLOCK])
    tmp6 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask, other=0)
    tmp7 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask, other=0)
    tmp9 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask, other=0)
    tmp11 = tl.load(in_ptr3 + (r1 + (768*x0)), rmask, other=0)
    tmp16 = tl.load(in_ptr4 + (r1), rmask, other=0)
    tmp21 = tl.load(in_ptr5 + (r1 + (768*x0)), rmask, other=0)
    tmp26 = tl.load(in_ptr6 + (x0), None)
    tmp33 = tl.load(in_ptr7 + (x2), None)
    tmp38 = tl.load(in_ptr8 + (x0), None)
    tmp1 = r1 + (768*x0)
    tmp2 = tl.rand(tmp0, tmp1)
    tmp3 = 0.1
    tmp4 = tmp2 > tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp8 = tmp6 + tmp7
    tmp10 = tmp8 + tmp9
    tmp12 = tmp10 + tmp11
    tmp13 = tmp5 * tmp12
    tmp14 = 1.1111111111111112
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 * tmp16
    tmp19 = tl.where(rmask, tmp17, 0)
    tmp20 = tl.sum(tmp19, 1)[:, None]
    tmp22 = tmp17 * tmp21
    tmp24 = tl.where(rmask, tmp22, 0)
    tmp25 = tl.sum(tmp24, 1)[:, None]
    tmp27 = 768.0
    tmp28 = tmp17 * tmp27
    tmp29 = tmp28 - tmp20
    tmp30 = tmp21 * tmp25
    tmp31 = tmp29 - tmp30
    tmp32 = tmp26 * tmp31
    tmp34 = -1
    tmp35 = tmp33 == tmp34
    tmp36 = 0.0
    tmp37 = tl.where(tmp35, tmp36, tmp32)
    tmp39 = 0
    tmp40 = tmp38 == tmp39
    tmp41 = tl.where(tmp40, tmp36, tmp32)
    tl.store(in_out_ptr0 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp15, rmask)
    tl.store(out_ptr2 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp32, rmask)
    tl.atomic_add(out_ptr3 + (r1 + (768*tmp33) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp37, rmask)
    tl.atomic_add(out_ptr4 + (r1 + (768*tmp38) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp41, rmask)
''')


# kernel path: /tmp/tmpom3r99qq/fe/cfe4troybfwogn6e5d7cefle5qfqtyyfbyhmzaikb5zulkukdptw.py
# Original ATen: aten.embedding_dense_backward, aten.sum

# aten.embedding_dense_backward => full_1, index_put, scalar_tensor, where
# aten.sum => sum_203
triton_fused_embedding_dense_backward_sum_52 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[524288], filename=__file__, meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 393216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp0, None)
''')


# kernel path: /tmp/tmpom3r99qq/bf/cbfejvqpo64ixtlh7qov5fiqn4sxepzoydefjruhpq7bnsb7gnvw.py
# Original ATen: aten.embedding_dense_backward, aten.sum

# aten.embedding_dense_backward => full_1, index_put, scalar_tensor, where
# aten.sum => sum_203
triton_fused_embedding_dense_backward_sum_53 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[524288], filename=__file__, meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 393216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 768)
    x2 = xindex
    x0 = xindex % 768
    tmp0 = tl.load(in_ptr0 + (x1), None)
    tmp4 = tl.load(in_ptr1 + (x2), None)
    tmp5 = tl.load(in_ptr1 + (393216 + x2), None)
    tmp7 = tl.load(in_ptr1 + (786432 + x2), None)
    tmp9 = tl.load(in_ptr1 + (1179648 + x2), None)
    tmp1 = -1
    tmp2 = tmp0 == tmp1
    tmp3 = 0.0
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp10 = tmp8 + tmp9
    tmp11 = tl.where(tmp2, tmp3, tmp10)
    tl.atomic_add(out_ptr0 + (x0 + (768*tmp0) + tl.zeros([XBLOCK], tl.int32)), tmp11, None)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_4, primals_14, primals_20, primals_30, primals_36, primals_46, primals_52, primals_62, primals_68, primals_78, primals_84, primals_94, primals_100, primals_110, primals_116, primals_126, primals_132, primals_142, primals_148, primals_158, primals_164, primals_174, primals_180, primals_190, primals_196, primals_200, primals_206, expand, slice_4, mul_1, philox_seed_like, view, div_1, view_13, mul_9, view_15, addmm_4, view_17, mul_16, view_19, div_3, view_32, mul_22, view_34, addmm_10, view_36, mul_29, view_38, div_5, view_51, mul_35, view_53, addmm_16, view_55, mul_42, view_57, div_7, view_70, mul_48, view_72, addmm_22, view_74, mul_55, view_76, div_9, view_89, mul_61, view_91, addmm_28, view_93, mul_68, view_95, div_11, view_108, mul_74, view_110, addmm_34, view_112, mul_81, view_114, div_13, view_127, mul_87, view_129, addmm_40, view_131, mul_94, view_133, div_15, view_146, mul_100, view_148, addmm_46, view_150, mul_107, view_152, div_17, view_165, mul_113, view_167, addmm_52, view_169, mul_120, view_171, div_19, view_184, mul_126, view_186, addmm_58, view_188, mul_133, view_190, div_21, view_203, mul_139, view_205, addmm_64, view_207, mul_146, view_209, div_23, view_222, mul_152, view_224, addmm_70, view_226, mul_159, view_228, addmm_72, mul_164, view_230, permute_134, div_24, permute_138, div_25, permute_142, permute_146, div_26, permute_150, permute_155, permute_156, permute_157, permute_158, permute_162, permute_167, permute_171, div_28, permute_175, permute_179, div_29, permute_183, permute_188, permute_189, permute_190, permute_191, permute_195, permute_200, permute_204, div_31, permute_208, permute_212, div_32, permute_216, permute_221, permute_222, permute_223, permute_224, permute_228, permute_233, permute_237, div_34, permute_241, permute_245, div_35, permute_249, permute_254, permute_255, permute_256, permute_257, permute_261, permute_266, permute_270, div_37, permute_274, permute_278, div_38, permute_282, permute_287, permute_288, permute_289, permute_290, permute_294, permute_299, permute_303, div_40, permute_307, permute_311, div_41, permute_315, permute_320, permute_321, permute_322, permute_323, permute_327, permute_332, permute_336, div_43, permute_340, permute_344, div_44, permute_348, permute_353, permute_354, permute_355, permute_356, permute_360, permute_365, permute_369, div_46, permute_373, permute_377, div_47, permute_381, permute_386, permute_387, permute_388, permute_389, permute_393, permute_398, permute_402, div_49, permute_406, permute_410, div_50, permute_414, permute_419, permute_420, permute_421, permute_422, permute_426, permute_431, permute_435, div_52, permute_439, permute_443, div_53, permute_447, permute_452, permute_453, permute_454, permute_455, permute_459, permute_464, permute_468, div_55, permute_472, permute_476, div_56, permute_480, permute_485, permute_486, permute_487, permute_488, permute_492, permute_497, permute_501, div_58, permute_505, permute_509, div_59, permute_513, permute_518, permute_519, permute_520, permute_521, permute_525, permute_530, permute_534, div_61, tangents_1 = args
    args.clear()
    start_graph()
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty_strided((2048, 768), (768, 1), device='cuda', dtype=torch.float32)
        extern_kernels.mm(tangents_1, permute_134, out=buf0)
        del permute_134
        buf1 = empty_strided((30522, 768), (768, 1), device='cuda', dtype=torch.float32)
        extern_kernels.mm(as_strided(tangents_1, (30522, 2048), (1, 30522)), view_230, out=buf1)
        del view_230
        buf2 = empty_strided((1, 30522), (30522, 1), device='cuda', dtype=torch.float32)
        stream0 = get_cuda_stream(0)
        other_stream = torch.cuda.Stream()

        triton_fused_sum_0.run(tangents_1, buf2, 30522, 2048, grid=grid(30522), stream=stream0)
        del tangents_1
        buf9 = empty_strided((4, 512, 768), (393216, 768, 1), device='cuda', dtype=torch.float32)
        buf10 = as_strided(buf9, (2048, 768), (768, 1)); del buf9  # reuse
        triton_fused_gelu_gelu_backward_native_layer_norm_backward_view_1.run(buf10, buf0, primals_200, mul_164, div_24, addmm_72, 2048, 768, grid=grid(2048), stream=stream0)
        del addmm_72
        del div_24
        del primals_200
        buf5 = empty_strided((768, 16), (1, 768), device='cuda', dtype=torch.float32)
        buf7 = empty_strided((768, 16), (1, 768), device='cuda', dtype=torch.float32)
        triton_fused_native_layer_norm_backward_2.run(buf0, mul_164, buf5, buf7, 12288, 128, grid=grid(12288), stream=stream0)
        del mul_164
        buf6 = empty_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
        triton_fused_native_layer_norm_backward_3.run(buf5, buf6, 768, 16, grid=grid(768), stream=stream0)
        buf8 = empty_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
        triton_fused_native_layer_norm_backward_3.run(buf7, buf8, 768, 16, grid=grid(768), stream=stream0)
        buf11 = buf0; del buf0  # reuse
        extern_kernels.mm(buf10, permute_138, out=buf11)
        del permute_138
        with torch.cuda.stream(other_stream):
            buf12 = empty_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
            extern_kernels.mm(as_strided(buf10, (768, 2048), (1, 768)), view_228, out=buf12)
        del view_228
        buf13 = as_strided(buf7, (1, 768, 16), (12288, 1, 768)); del buf7  # reuse
        triton_fused_sum_4.run(buf10, buf13, 12288, 128, grid=grid(12288), stream=stream0)
        buf14 = empty_strided((1, 768), (768, 1), device='cuda', dtype=torch.float32)
        triton_fused_native_layer_norm_backward_3.run(buf13, buf14, 768, 16, grid=grid(768), stream=stream0)
        buf17 = as_strided(buf10, (4, 512, 768), (393216, 768, 1)); del buf10  # reuse
        buf22 = empty_strided((4, 512, 768), (393216, 768, 1), device='cuda', dtype=torch.float32)
        buf23 = as_strided(buf22, (2048, 768), (768, 1)); del buf22  # reuse
        triton_fused__to_copy_gt_mul_native_layer_norm_backward_philox_rand_like_view_5.run(buf23, buf11, primals_196, mul_159, div_25, philox_seed_like, buf17, 2048, 768, grid=grid(2048), stream=stream0)
        del div_25
        del primals_196
        buf18 = as_strided(buf13, (768, 16), (1, 768)); del buf13  # reuse
        buf20 = buf5; del buf5  # reuse
        triton_fused_native_layer_norm_backward_2.run(buf11, mul_159, buf18, buf20, 12288, 128, grid=grid(12288), stream=stream0)
        del mul_159
        buf19 = empty_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
        triton_fused_native_layer_norm_backward_3.run(buf18, buf19, 768, 16, grid=grid(768), stream=stream0)
        buf21 = empty_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
        triton_fused_native_layer_norm_backward_3.run(buf20, buf21, 768, 16, grid=grid(768), stream=stream0)
        buf24 = empty_strided((2048, 3072), (3072, 1), device='cuda', dtype=torch.float32)
        extern_kernels.mm(buf23, permute_142, out=buf24)
        del permute_142

        with torch.cuda.stream(other_stream):
            buf25 = empty_strided((768, 3072), (3072, 1), device='cuda', dtype=torch.float32)
            extern_kernels.mm(as_strided(buf23, (768, 2048), (1, 768)), view_226, out=buf25)
            del view_226
        buf26 = as_strided(buf20, (1, 768, 16), (12288, 1, 768)); del buf20  # reuse
        triton_fused_sum_4.run(buf23, buf26, 12288, 128, grid=grid(12288), stream=stream0)
        buf27 = empty_strided((1, 768), (768, 1), device='cuda', dtype=torch.float32)
        triton_fused_native_layer_norm_backward_3.run(buf26, buf27, 768, 16, grid=grid(768), stream=stream0)
        buf28 = as_strided(buf24, (4, 512, 3072), (1572864, 3072, 1)); del buf24  # reuse
        buf29 = as_strided(buf28, (2048, 3072), (3072, 1)); del buf28  # reuse
        triton_fused_gelu_gelu_backward_view_6.run(buf29, addmm_70, 6291456, grid=grid(6291456), stream=stream0)
        del addmm_70
        buf30 = buf23; del buf23  # reuse
        extern_kernels.mm(buf29, permute_146, out=buf30)
        del permute_146
        with torch.cuda.stream(other_stream):
            buf31 = empty_strided((3072, 768), (768, 1), device='cuda', dtype=torch.float32)
            extern_kernels.mm(as_strided(buf29, (3072, 2048), (1, 3072)), view_224, out=buf31)
        del view_224
        buf32 = empty_strided((1, 3072, 16), (49152, 1, 3072), device='cuda', dtype=torch.float32)
        triton_fused_sum_7.run(buf29, buf32, 49152, 128, grid=grid(49152), stream=stream0)
        buf33 = empty_strided((1, 3072), (3072, 1), device='cuda', dtype=torch.float32)
        triton_fused_sum_8.run(buf32, buf33, 3072, 16, grid=grid(3072), stream=stream0)
        buf36 = as_strided(buf11, (4, 512, 768), (393216, 768, 1)); del buf11  # reuse
        buf41 = empty_strided((4, 512, 768), (393216, 768, 1), device='cuda', dtype=torch.float32)
        buf42 = as_strided(buf41, (2048, 768), (768, 1)); del buf41  # reuse
        triton_fused__to_copy_add_gt_mul_native_layer_norm_backward_philox_rand_like_view_9.run(buf42, buf17, buf30, primals_190, mul_152, div_26, philox_seed_like, buf36, 2048, 768, grid=grid(2048), stream=stream0)
        del div_26
        del primals_190
        buf37 = as_strided(buf26, (768, 16), (1, 768)); del buf26  # reuse
        buf39 = buf18; del buf18  # reuse
        triton_fused_add_native_layer_norm_backward_10.run(buf17, buf30, mul_152, buf37, buf39, 12288, 128, grid=grid(12288), stream=stream0)
        del mul_152
        buf38 = empty_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
        triton_fused_native_layer_norm_backward_3.run(buf37, buf38, 768, 16, grid=grid(768), stream=stream0)
        buf40 = empty_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
        triton_fused_native_layer_norm_backward_3.run(buf39, buf40, 768, 16, grid=grid(768), stream=stream0)
        buf43 = buf30; del buf30  # reuse
        extern_kernels.mm(buf42, permute_150, out=buf43)
        del permute_150
        with torch.cuda.stream(other_stream):
            buf44 = empty_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
            extern_kernels.mm(as_strided(buf42, (768, 2048), (1, 768)), view_222, out=buf44)
        del view_222
        buf45 = as_strided(buf39, (1, 768, 16), (12288, 1, 768)); del buf39  # reuse
        triton_fused_sum_4.run(buf42, buf45, 12288, 128, grid=grid(12288), stream=stream0)
        buf46 = empty_strided((1, 768), (768, 1), device='cuda', dtype=torch.float32)
        triton_fused_native_layer_norm_backward_3.run(buf45, buf46, 768, 16, grid=grid(768), stream=stream0)
        buf47 = as_strided(buf42, (4, 12, 512, 64), (393216, 32768, 64, 1)); del buf42  # reuse
        triton_fused_clone_11.run(buf43, buf47, 1572864, grid=grid(1572864), stream=stream0)
        buf48 = as_strided(buf43, (48, 512, 64), (32768, 64, 1)); del buf43  # reuse
        extern_kernels.bmm(permute_155, as_strided(buf47, (48, 512, 64), (32768, 64, 1)), out=buf48)
        del permute_155
        buf49 = empty_strided((48, 512, 512), (262144, 512, 1), device='cuda', dtype=torch.float32)
        extern_kernels.bmm(as_strided(buf47, (48, 512, 64), (32768, 64, 1)), permute_156, out=buf49)
        del permute_156
        buf51 = empty_strided((4, 12, 512, 512), (3145728, 262144, 512, 1), device='cuda', dtype=torch.float32)
        triton_fused__softmax_backward_data__to_copy_div_gt_mul_philox_rand_like_12.run(philox_seed_like, buf49, div_23, buf51, 24576, 512, grid=grid(24576), stream=stream0)
        del div_23
        buf52 = as_strided(buf47, (48, 64, 512), (32768, 512, 1)); del buf47  # reuse
        extern_kernels.bmm(permute_157, as_strided(buf51, (48, 512, 512), (262144, 512, 1)), out=buf52)
        del permute_157
        buf53 = as_strided(buf17, (48, 512, 64), (32768, 64, 1)); del buf17  # reuse
        extern_kernels.bmm(as_strided(buf51, (48, 512, 512), (262144, 512, 1)), permute_158, out=buf53)
        del permute_158
        buf54 = empty_strided((2048, 768), (768, 1), device='cuda', dtype=torch.float32)
        triton_fused_view_13.run(buf48, buf54, 1572864, grid=grid(1572864), stream=stream0)
        buf55 = as_strided(buf48, (2048, 768), (768, 1)); del buf48  # reuse
        extern_kernels.mm(buf54, permute_162, out=buf55)
        del permute_162
        with torch.cuda.stream(other_stream):
            buf56 = empty_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
            extern_kernels.mm(as_strided(buf54, (768, 2048), (1, 768)), view_209, out=buf56)
        buf57 = buf45; del buf45  # reuse
        triton_fused_sum_4.run(buf54, buf57, 12288, 128, grid=grid(12288), stream=stream0)
        buf58 = empty_strided((1, 768), (768, 1), device='cuda', dtype=torch.float32)
        triton_fused_native_layer_norm_backward_3.run(buf57, buf58, 768, 16, grid=grid(768), stream=stream0)
        buf59 = buf54; del buf54  # reuse
        triton_fused__unsafe_view_clone_14.run(buf52, buf59, 2048, 768, grid=grid(2048, 768), stream=stream0)
        buf60 = as_strided(buf52, (2048, 768), (768, 1)); del buf52  # reuse
        extern_kernels.mm(buf59, permute_167, out=buf60)
        del permute_167
        with torch.cuda.stream(other_stream):
            buf61 = empty_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
            extern_kernels.mm(as_strided(buf59, (768, 2048), (1, 768)), view_209, out=buf61)
        buf62 = buf57; del buf57  # reuse
        triton_fused_sum_4.run(buf59, buf62, 12288, 128, grid=grid(12288), stream=stream0)
        buf63 = empty_strided((1, 768), (768, 1), device='cuda', dtype=torch.float32)
        triton_fused_native_layer_norm_backward_3.run(buf62, buf63, 768, 16, grid=grid(768), stream=stream0)
        buf64 = buf59; del buf59  # reuse
        triton_fused_view_13.run(buf53, buf64, 1572864, grid=grid(1572864), stream=stream0)
        buf65 = as_strided(buf53, (2048, 768), (768, 1)); del buf53  # reuse
        extern_kernels.mm(buf64, permute_171, out=buf65)
        del permute_171
        with torch.cuda.stream(other_stream):
            buf66 = empty_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
            extern_kernels.mm(as_strided(buf64, (768, 2048), (1, 768)), view_209, out=buf66)
        del view_209
        buf67 = buf62; del buf62  # reuse
        triton_fused_sum_4.run(buf64, buf67, 12288, 128, grid=grid(12288), stream=stream0)
        buf68 = empty_strided((1, 768), (768, 1), device='cuda', dtype=torch.float32)
        triton_fused_native_layer_norm_backward_3.run(buf67, buf68, 768, 16, grid=grid(768), stream=stream0)
        buf72 = as_strided(buf64, (4, 512, 768), (393216, 768, 1)); del buf64  # reuse
        buf77 = empty_strided((4, 512, 768), (393216, 768, 1), device='cuda', dtype=torch.float32)
        buf78 = as_strided(buf77, (2048, 768), (768, 1)); del buf77  # reuse
        triton_fused__to_copy_add_gt_mul_native_layer_norm_backward_philox_rand_like_view_15.run(buf78, buf36, buf55, buf60, buf65, primals_180, mul_146, div_28, philox_seed_like, buf72, 2048, 768, grid=grid(2048), stream=stream0)
        del div_28
        del primals_180
        buf73 = as_strided(buf67, (768, 16), (1, 768)); del buf67  # reuse
        buf75 = buf37; del buf37  # reuse
        triton_fused_add_native_layer_norm_backward_16.run(buf36, buf55, buf60, buf65, mul_146, buf73, buf75, 12288, 128, grid=grid(12288), stream=stream0)
        # torch.cuda.synchronize()
        return 5

def benchmark_compiled_module():
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_4 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_174 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_180 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_200 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_206 = rand_strided((4, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    expand = rand_strided((4, 512), (0, 1), device='cuda:0', dtype=torch.int64)
    slice_4 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    mul_1 = rand_strided((4, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    philox_seed_like = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    view = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_1 = rand_strided((4, 12, 512, 512), (3145728, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    view_13 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_9 = rand_strided((4, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_15 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_4 = rand_strided((2048, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_17 = rand_strided((2048, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    mul_16 = rand_strided((4, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_19 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_3 = rand_strided((4, 12, 512, 512), (3145728, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    view_32 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_22 = rand_strided((4, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_34 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_10 = rand_strided((2048, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_36 = rand_strided((2048, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    mul_29 = rand_strided((4, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_38 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_5 = rand_strided((4, 12, 512, 512), (3145728, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    view_51 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_35 = rand_strided((4, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_53 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_16 = rand_strided((2048, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_55 = rand_strided((2048, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    mul_42 = rand_strided((4, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_57 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_7 = rand_strided((4, 12, 512, 512), (3145728, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    view_70 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_48 = rand_strided((4, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_72 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_22 = rand_strided((2048, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_74 = rand_strided((2048, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    mul_55 = rand_strided((4, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_76 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_9 = rand_strided((4, 12, 512, 512), (3145728, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    view_89 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_61 = rand_strided((4, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_91 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_28 = rand_strided((2048, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_93 = rand_strided((2048, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    mul_68 = rand_strided((4, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_95 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_11 = rand_strided((4, 12, 512, 512), (3145728, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    view_108 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_74 = rand_strided((4, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_110 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_34 = rand_strided((2048, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_112 = rand_strided((2048, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    mul_81 = rand_strided((4, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_114 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_13 = rand_strided((4, 12, 512, 512), (3145728, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    view_127 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_87 = rand_strided((4, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_129 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_40 = rand_strided((2048, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_131 = rand_strided((2048, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    mul_94 = rand_strided((4, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_133 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_15 = rand_strided((4, 12, 512, 512), (3145728, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    view_146 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_100 = rand_strided((4, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_148 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_46 = rand_strided((2048, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_150 = rand_strided((2048, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    mul_107 = rand_strided((4, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_152 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_17 = rand_strided((4, 12, 512, 512), (3145728, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    view_165 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_113 = rand_strided((4, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_167 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_52 = rand_strided((2048, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_169 = rand_strided((2048, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    mul_120 = rand_strided((4, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_171 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_19 = rand_strided((4, 12, 512, 512), (3145728, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    view_184 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_126 = rand_strided((4, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_186 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_58 = rand_strided((2048, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_188 = rand_strided((2048, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    mul_133 = rand_strided((4, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_190 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_21 = rand_strided((4, 12, 512, 512), (3145728, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    view_203 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_139 = rand_strided((4, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_205 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_64 = rand_strided((2048, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_207 = rand_strided((2048, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    mul_146 = rand_strided((4, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_209 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_23 = rand_strided((4, 12, 512, 512), (3145728, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    view_222 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_152 = rand_strided((4, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_224 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_70 = rand_strided((2048, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_226 = rand_strided((2048, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    mul_159 = rand_strided((4, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_228 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_72 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_164 = rand_strided((4, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_230 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_134 = rand_strided((30522, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_24 = rand_strided((4, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_138 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_25 = rand_strided((4, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_142 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_146 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_26 = rand_strided((4, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_150 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_155 = rand_strided((48, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_156 = rand_strided((48, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_157 = rand_strided((48, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_158 = rand_strided((48, 512, 64), (32768, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_162 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_167 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_171 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_28 = rand_strided((4, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_175 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_179 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_29 = rand_strided((4, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_183 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_188 = rand_strided((48, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_189 = rand_strided((48, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_190 = rand_strided((48, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_191 = rand_strided((48, 512, 64), (32768, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_195 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_200 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_204 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_31 = rand_strided((4, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_208 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_212 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_32 = rand_strided((4, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_216 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_221 = rand_strided((48, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_222 = rand_strided((48, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_223 = rand_strided((48, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_224 = rand_strided((48, 512, 64), (32768, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_228 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_233 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_237 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_34 = rand_strided((4, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_241 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_245 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_35 = rand_strided((4, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_249 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_254 = rand_strided((48, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_255 = rand_strided((48, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_256 = rand_strided((48, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_257 = rand_strided((48, 512, 64), (32768, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_261 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_266 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_270 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_37 = rand_strided((4, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_274 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_278 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_38 = rand_strided((4, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_282 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_287 = rand_strided((48, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_288 = rand_strided((48, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_289 = rand_strided((48, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_290 = rand_strided((48, 512, 64), (32768, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_294 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_299 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_303 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_40 = rand_strided((4, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_307 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_311 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_41 = rand_strided((4, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_315 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_320 = rand_strided((48, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_321 = rand_strided((48, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_322 = rand_strided((48, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_323 = rand_strided((48, 512, 64), (32768, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_327 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_332 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_336 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_43 = rand_strided((4, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_340 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_344 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_44 = rand_strided((4, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_348 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_353 = rand_strided((48, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_354 = rand_strided((48, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_355 = rand_strided((48, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_356 = rand_strided((48, 512, 64), (32768, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_360 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_365 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_369 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_46 = rand_strided((4, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_373 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_377 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_47 = rand_strided((4, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_381 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_386 = rand_strided((48, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_387 = rand_strided((48, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_388 = rand_strided((48, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_389 = rand_strided((48, 512, 64), (32768, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_393 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_398 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_402 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_49 = rand_strided((4, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_406 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_410 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_50 = rand_strided((4, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_414 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_419 = rand_strided((48, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_420 = rand_strided((48, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_421 = rand_strided((48, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_422 = rand_strided((48, 512, 64), (32768, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_426 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_431 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_435 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_52 = rand_strided((4, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_439 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_443 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_53 = rand_strided((4, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_447 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_452 = rand_strided((48, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_453 = rand_strided((48, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_454 = rand_strided((48, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_455 = rand_strided((48, 512, 64), (32768, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_459 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_464 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_468 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_55 = rand_strided((4, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_472 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_476 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_56 = rand_strided((4, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_480 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_485 = rand_strided((48, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_486 = rand_strided((48, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_487 = rand_strided((48, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_488 = rand_strided((48, 512, 64), (32768, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_492 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_497 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_501 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_58 = rand_strided((4, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_505 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_509 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_59 = rand_strided((4, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_513 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_518 = rand_strided((48, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_519 = rand_strided((48, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_520 = rand_strided((48, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_521 = rand_strided((48, 512, 64), (32768, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_525 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_530 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_534 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_61 = rand_strided((4, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    tangents_1 = rand_strided((2048, 30522), (30522, 1), device='cuda:0', dtype=torch.float32)

    from torch.profiler import profile, record_function, ProfilerActivity

    def bench(f, name=None, iters=100, warmup=5, display=True, profile=False):
        import time
        from triton.testing import do_bench

        for _ in range(warmup):
            f()
        if profile:
            with torch.profiler.profile(activities=[ProfilerActivity.CUDA]) as prof:
                f()
            prof.export_chrome_trace(f"{name if name is not None else 'trace'}.json")


        us_per_iter = do_bench(lambda: f())[0]*1000

        if name is None:
            res = us_per_iter
        else:
            res= f"{name}: {us_per_iter:.3f}us"

        if display:
            print(res)
        return res

    fn = lambda: call([primals_4, primals_14, primals_20, primals_30, primals_36, primals_46, primals_52, primals_62, primals_68, primals_78, primals_84, primals_94, primals_100, primals_110, primals_116, primals_126, primals_132, primals_142, primals_148, primals_158, primals_164, primals_174, primals_180, primals_190, primals_196, primals_200, primals_206, expand, slice_4, mul_1, philox_seed_like, view, div_1, view_13, mul_9, view_15, addmm_4, view_17, mul_16, view_19, div_3, view_32, mul_22, view_34, addmm_10, view_36, mul_29, view_38, div_5, view_51, mul_35, view_53, addmm_16, view_55, mul_42, view_57, div_7, view_70, mul_48, view_72, addmm_22, view_74, mul_55, view_76, div_9, view_89, mul_61, view_91, addmm_28, view_93, mul_68, view_95, div_11, view_108, mul_74, view_110, addmm_34, view_112, mul_81, view_114, div_13, view_127, mul_87, view_129, addmm_40, view_131, mul_94, view_133, div_15, view_146, mul_100, view_148, addmm_46, view_150, mul_107, view_152, div_17, view_165, mul_113, view_167, addmm_52, view_169, mul_120, view_171, div_19, view_184, mul_126, view_186, addmm_58, view_188, mul_133, view_190, div_21, view_203, mul_139, view_205, addmm_64, view_207, mul_146, view_209, div_23, view_222, mul_152, view_224, addmm_70, view_226, mul_159, view_228, addmm_72, mul_164, view_230, permute_134, div_24, permute_138, div_25, permute_142, permute_146, div_26, permute_150, permute_155, permute_156, permute_157, permute_158, permute_162, permute_167, permute_171, div_28, permute_175, permute_179, div_29, permute_183, permute_188, permute_189, permute_190, permute_191, permute_195, permute_200, permute_204, div_31, permute_208, permute_212, div_32, permute_216, permute_221, permute_222, permute_223, permute_224, permute_228, permute_233, permute_237, div_34, permute_241, permute_245, div_35, permute_249, permute_254, permute_255, permute_256, permute_257, permute_261, permute_266, permute_270, div_37, permute_274, permute_278, div_38, permute_282, permute_287, permute_288, permute_289, permute_290, permute_294, permute_299, permute_303, div_40, permute_307, permute_311, div_41, permute_315, permute_320, permute_321, permute_322, permute_323, permute_327, permute_332, permute_336, div_43, permute_340, permute_344, div_44, permute_348, permute_353, permute_354, permute_355, permute_356, permute_360, permute_365, permute_369, div_46, permute_373, permute_377, div_47, permute_381, permute_386, permute_387, permute_388, permute_389, permute_393, permute_398, permute_402, div_49, permute_406, permute_410, div_50, permute_414, permute_419, permute_420, permute_421, permute_422, permute_426, permute_431, permute_435, div_52, permute_439, permute_443, div_53, permute_447, permute_452, permute_453, permute_454, permute_455, permute_459, permute_464, permute_468, div_55, permute_472, permute_476, div_56, permute_480, permute_485, permute_486, permute_487, permute_488, permute_492, permute_497, permute_501, div_58, permute_505, permute_509, div_59, permute_513, permute_518, permute_519, permute_520, permute_521, permute_525, permute_530, permute_534, div_61, tangents_1])
    # bench(lambda: fn(), profile=True)
    print_performance(fn)


if __name__ == "__main__":
    import argparse
    from torch._inductor.utils import benchmark_all_kernels

    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-kernels", "-k", action="store_true", help="Whether to benchmark each individual kernels")
    parser.add_argument("--benchmark-all-configs", "-c", action="store_true", help="Whether to benchmark each individual config for a kernel")
    args = parser.parse_args()

    if args.benchmark_kernels:
        benchmark_all_kernels('hf_Bert', args.benchmark_all_configs)
    else:
        benchmark_compiled_module()
