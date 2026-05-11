#pragma once

namespace at::mps {

static const char* SCATTER_OPS_TEMPLATE = R"METAL_SCATTER(
template<typename Y, typename X>
Y cast(const X x);

template<>
{1} cast<{1}, {0}>(const {0} x) {{
 return {2};
}}

template <typename OffsetT>
kernel void scatter_kernel_n(uint linear_index          [[thread_position_in_grid]],
                             constant void * src_       [[buffer(0)]],
                             device void * dst_         [[buffer(1)]],
                             constant uint32_t * size   [[buffer(2)]],
                             constant OffsetT * stride  [[buffer(3)]],
                            constant uint32_t & numel   [[buffer(4)]],
                            constant int32_t & ndim     [[buffer(5)]]) {{
    if (linear_index >= numel) return;

    constant {0} * src = (constant {0} *)src_;
    device {1} * dst = (device {1} *)dst_;

    OffsetT dst_offs = 0;
    auto dst_idx = linear_index;
    for(int dim = ndim - 1; dim >= 0; --dim) {{
      dst_offs += stride[dim] * (dst_idx % size[dim]);
      dst_idx /= size[dim];
    }}

    dst[dst_offs] = cast<{1}>(src[linear_index]);
}}

template [[host_name("scatter_kernel_n_u32")]]
kernel void scatter_kernel_n<uint>(
    uint linear_index          [[thread_position_in_grid]],
    constant void * src_       [[buffer(0)]],
    device void * dst_         [[buffer(1)]],
    constant uint32_t * size   [[buffer(2)]],
    constant uint * stride     [[buffer(3)]],
    constant uint32_t & numel  [[buffer(4)]],
    constant int32_t & ndim    [[buffer(5)]]);

template [[host_name("scatter_kernel_n_u64")]]
kernel void scatter_kernel_n<ulong>(
    uint linear_index          [[thread_position_in_grid]],
    constant void * src_       [[buffer(0)]],
    device void * dst_         [[buffer(1)]],
    constant uint32_t * size   [[buffer(2)]],
    constant ulong * stride    [[buffer(3)]],
    constant uint32_t & numel  [[buffer(4)]],
    constant int32_t & ndim    [[buffer(5)]]);

template <typename PackedT, typename VecT>
kernel void scatter_kernel_4(uint linear_index              [[thread_position_in_grid]],
                             constant void * src_           [[buffer(0)]],
                             device void * dst_             [[buffer(1)]],
                             constant packed_uint4 & size   [[buffer(2)]],
                             constant PackedT & stride      [[buffer(3)]],
                             constant uint32_t & numel      [[buffer(4)]]) {{
    if (linear_index >= numel) return;

    constant {0} * src = (constant {0} *)src_;
    device {1} * dst = (device {1} *)dst_;

    packed_uint4 local_index;
    local_index.x = linear_index / (size[3] * size[2] * size[1]) % size[0];
    local_index.y = linear_index / (size[3] * size[2]) % size[1];
    local_index.z = linear_index / size[3] % size[2];
    local_index.w = linear_index % size[3];

    const PackedT strided_index = VecT(local_index) * stride;
    dst[strided_index.x + strided_index.y + strided_index.z + strided_index.w] = cast<{1}>(src[linear_index]);
}}

template [[host_name("scatter_kernel_4_u32")]]
kernel void scatter_kernel_4<packed_uint4, uint4>(
    uint linear_index              [[thread_position_in_grid]],
    constant void * src_           [[buffer(0)]],
    device void * dst_             [[buffer(1)]],
    constant packed_uint4 & size   [[buffer(2)]],
    constant packed_uint4 & stride [[buffer(3)]],
    constant uint32_t & numel      [[buffer(4)]]);

template [[host_name("scatter_kernel_4_u64")]]
kernel void scatter_kernel_4<packed_ulong4, ulong4>(
    uint linear_index               [[thread_position_in_grid]],
    constant void * src_            [[buffer(0)]],
    device void * dst_              [[buffer(1)]],
    constant packed_uint4 & size    [[buffer(2)]],
    constant packed_ulong4 & stride [[buffer(3)]],
    constant uint32_t & numel       [[buffer(4)]]);

template <typename PackedT, typename VecT>
kernel void scatter_kernel_3(uint linear_index              [[thread_position_in_grid]],
                             constant void * src_           [[buffer(0)]],
                             device void * dst_             [[buffer(1)]],
                             constant packed_uint3 & size   [[buffer(2)]],
                             constant PackedT & stride      [[buffer(3)]],
                             constant uint32_t & numel      [[buffer(4)]]) {{
    if (linear_index >= numel) return;

    constant {0} * src = (constant {0} *)src_;
    device {1} * dst = (device {1} *)dst_;

    packed_uint3 local_index;
    local_index.x = linear_index / (size[2] * size[1]) % size[0];
    local_index.y = linear_index / size[2] % size[1];
    local_index.z = linear_index % size[2];

    const PackedT strided_index = VecT(local_index) * stride;
    dst[strided_index.x + strided_index.y + strided_index.z] = cast<{1}>(src[linear_index]);
}}

template [[host_name("scatter_kernel_3_u32")]]
kernel void scatter_kernel_3<packed_uint3, uint3>(
    uint linear_index              [[thread_position_in_grid]],
    constant void * src_           [[buffer(0)]],
    device void * dst_             [[buffer(1)]],
    constant packed_uint3 & size   [[buffer(2)]],
    constant packed_uint3 & stride [[buffer(3)]],
    constant uint32_t & numel      [[buffer(4)]]);

template [[host_name("scatter_kernel_3_u64")]]
kernel void scatter_kernel_3<packed_ulong3, ulong3>(
    uint linear_index               [[thread_position_in_grid]],
    constant void * src_            [[buffer(0)]],
    device void * dst_              [[buffer(1)]],
    constant packed_uint3 & size    [[buffer(2)]],
    constant packed_ulong3 & stride [[buffer(3)]],
    constant uint32_t & numel       [[buffer(4)]]);

template <typename PackedT, typename VecT>
kernel void scatter_kernel_2(uint linear_index              [[thread_position_in_grid]],
                             constant void * src_           [[buffer(0)]],
                             device void * dst_             [[buffer(1)]],
                             constant packed_uint2 & size   [[buffer(2)]],
                             constant PackedT & stride      [[buffer(3)]],
                             constant uint32_t & numel      [[buffer(4)]]) {{
    if (linear_index >= numel) return;

    constant {0} * src = (constant {0} *)src_;
    device {1} * dst = (device {1} *)dst_;

    packed_uint2 local_index;
    local_index.x = linear_index / size[1] % size[0];
    local_index.y = linear_index % size[1];

    const PackedT strided_index = VecT(local_index) * stride;
    dst[strided_index.x + strided_index.y] = cast<{1}>(src[linear_index]);
}}

template [[host_name("scatter_kernel_2_u32")]]
kernel void scatter_kernel_2<packed_uint2, uint2>(
    uint linear_index              [[thread_position_in_grid]],
    constant void * src_           [[buffer(0)]],
    device void * dst_             [[buffer(1)]],
    constant packed_uint2 & size   [[buffer(2)]],
    constant packed_uint2 & stride [[buffer(3)]],
    constant uint32_t & numel      [[buffer(4)]]);

template [[host_name("scatter_kernel_2_u64")]]
kernel void scatter_kernel_2<packed_ulong2, ulong2>(
    uint linear_index               [[thread_position_in_grid]],
    constant void * src_            [[buffer(0)]],
    device void * dst_              [[buffer(1)]],
    constant packed_uint2 & size    [[buffer(2)]],
    constant packed_ulong2 & stride [[buffer(3)]],
    constant uint32_t & numel       [[buffer(4)]]);

template <typename OffsetT>
kernel void scatter_kernel_1(uint linear_index              [[thread_position_in_grid]],
                             constant void * src_           [[buffer(0)]],
                             device void * dst_             [[buffer(1)]],
                             constant int & size            [[buffer(2)]],
                             constant OffsetT & stride      [[buffer(3)]],
                             constant uint32_t & numel      [[buffer(4)]]) {{
    if (linear_index >= numel) return;

    constant {0} * src = (constant {0} *)src_;
    device {1} * dst = (device {1} *)dst_;

    const uint local_index = linear_index % size;
    const OffsetT strided_index = local_index * stride;
    dst[strided_index] = cast<{1}>(src[linear_index]);
}}

template [[host_name("scatter_kernel_1_u32")]]
kernel void scatter_kernel_1<uint>(
    uint linear_index              [[thread_position_in_grid]],
    constant void * src_           [[buffer(0)]],
    device void * dst_             [[buffer(1)]],
    constant int & size            [[buffer(2)]],
    constant uint & stride         [[buffer(3)]],
    constant uint32_t & numel      [[buffer(4)]]);

template [[host_name("scatter_kernel_1_u64")]]
kernel void scatter_kernel_1<ulong>(
    uint linear_index              [[thread_position_in_grid]],
    constant void * src_           [[buffer(0)]],
    device void * dst_             [[buffer(1)]],
    constant int & size            [[buffer(2)]],
    constant ulong & stride        [[buffer(3)]],
    constant uint32_t & numel      [[buffer(4)]]);
)METAL_SCATTER";

static const char* GATHER_OPS_TEMPLATE = R"METAL_GATHER(
template<typename Y, typename X>
Y cast(const X x);

template<>
{1} cast<{1}, {0}>(const {0} x) {{
 return {2};
}}

template <typename OffsetT>
kernel void gather_kernel_n(uint linear_index           [[thread_position_in_grid]],
                            constant void * src_        [[buffer(0)]],
                            device void * dst_          [[buffer(1)]],
                            constant uint32_t * size    [[buffer(2)]],
                            constant OffsetT * stride   [[buffer(3)]],
                            constant uint32_t & numel   [[buffer(4)]],
                            constant int32_t & ndim     [[buffer(5)]]) {{
    if (linear_index >= numel) return;

    constant {0} * src = (constant {0} *)src_;
    device {1} * dst = (device {1} *)dst_;

    OffsetT src_offs = 0;
    auto src_idx = linear_index;
    for(int dim = ndim - 1; dim >= 0; --dim) {{
      src_offs += stride[dim] * (src_idx % size[dim]);
      src_idx /= size[dim];
    }}

    dst[linear_index] = cast<{1}>(src[src_offs]);
}}

template [[host_name("gather_kernel_n_u32")]]
kernel void gather_kernel_n<uint>(
    uint linear_index           [[thread_position_in_grid]],
    constant void * src_        [[buffer(0)]],
    device void * dst_          [[buffer(1)]],
    constant uint32_t * size    [[buffer(2)]],
    constant uint * stride      [[buffer(3)]],
    constant uint32_t & numel   [[buffer(4)]],
    constant int32_t & ndim     [[buffer(5)]]);

template [[host_name("gather_kernel_n_u64")]]
kernel void gather_kernel_n<ulong>(
    uint linear_index           [[thread_position_in_grid]],
    constant void * src_        [[buffer(0)]],
    device void * dst_          [[buffer(1)]],
    constant uint32_t * size    [[buffer(2)]],
    constant ulong * stride     [[buffer(3)]],
    constant uint32_t & numel   [[buffer(4)]],
    constant int32_t & ndim     [[buffer(5)]]);

template <typename PackedT, typename VecT>
kernel void gather_kernel_4(uint linear_index               [[thread_position_in_grid]],
                            constant void * src_            [[buffer(0)]],
                            device void * dst_              [[buffer(1)]],
                            constant packed_uint4 & size    [[buffer(2)]],
                            constant PackedT & stride       [[buffer(3)]],
                            constant uint32_t & numel       [[buffer(4)]]) {{
    if (linear_index >= numel) return;

    constant {0} * src = (constant {0} *)src_;
    device {1} * dst = (device {1} *)dst_;

    packed_uint4 local_index;
    local_index.x = linear_index / (size[3] * size[2] * size[1]) % size[0];
    local_index.y = linear_index / (size[3] * size[2]) % size[1];
    local_index.z = linear_index / size[3] % size[2];
    local_index.w = linear_index % size[3];

    const PackedT strided_index = VecT(local_index) * stride;
    dst[linear_index] = cast<{1}>(src[strided_index.x + strided_index.y + strided_index.z + strided_index.w]);
}}

template [[host_name("gather_kernel_4_u32")]]
kernel void gather_kernel_4<packed_uint4, uint4>(
    uint linear_index              [[thread_position_in_grid]],
    constant void * src_           [[buffer(0)]],
    device void * dst_             [[buffer(1)]],
    constant packed_uint4 & size   [[buffer(2)]],
    constant packed_uint4 & stride [[buffer(3)]],
    constant uint32_t & numel      [[buffer(4)]]);

template [[host_name("gather_kernel_4_u64")]]
kernel void gather_kernel_4<packed_ulong4, ulong4>(
    uint linear_index               [[thread_position_in_grid]],
    constant void * src_            [[buffer(0)]],
    device void * dst_              [[buffer(1)]],
    constant packed_uint4 & size    [[buffer(2)]],
    constant packed_ulong4 & stride [[buffer(3)]],
    constant uint32_t & numel       [[buffer(4)]]);

template <typename PackedT, typename VecT>
kernel void gather_kernel_3(uint linear_index               [[thread_position_in_grid]],
                            constant void * src_            [[buffer(0)]],
                            device void * dst_              [[buffer(1)]],
                            constant packed_uint3 & size    [[buffer(2)]],
                            constant PackedT & stride       [[buffer(3)]],
                            constant uint32_t & numel       [[buffer(4)]]) {{
    if (linear_index >= numel) return;

    constant {0} * src = (constant {0} *)src_;
    device {1} * dst = (device {1} *)dst_;

    packed_uint3 local_index;
    local_index.x = linear_index / (size[2] * size[1]) % size[0];
    local_index.y = linear_index / size[2] % size[1];
    local_index.z = linear_index % size[2];

    const PackedT strided_index = VecT(local_index) * stride;
    dst[linear_index] = cast<{1}>(src[strided_index.x + strided_index.y + strided_index.z]);
}}

template [[host_name("gather_kernel_3_u32")]]
kernel void gather_kernel_3<packed_uint3, uint3>(
    uint linear_index              [[thread_position_in_grid]],
    constant void * src_           [[buffer(0)]],
    device void * dst_             [[buffer(1)]],
    constant packed_uint3 & size   [[buffer(2)]],
    constant packed_uint3 & stride [[buffer(3)]],
    constant uint32_t & numel      [[buffer(4)]]);

template [[host_name("gather_kernel_3_u64")]]
kernel void gather_kernel_3<packed_ulong3, ulong3>(
    uint linear_index               [[thread_position_in_grid]],
    constant void * src_            [[buffer(0)]],
    device void * dst_              [[buffer(1)]],
    constant packed_uint3 & size    [[buffer(2)]],
    constant packed_ulong3 & stride [[buffer(3)]],
    constant uint32_t & numel       [[buffer(4)]]);

template <typename PackedT, typename VecT>
kernel void gather_kernel_2(uint linear_index               [[thread_position_in_grid]],
                            constant void * src_            [[buffer(0)]],
                            device void * dst_              [[buffer(1)]],
                            constant packed_uint2 & size    [[buffer(2)]],
                            constant PackedT & stride       [[buffer(3)]],
                            constant uint32_t & numel       [[buffer(4)]]) {{
    if (linear_index >= numel) return;

    constant {0} * src = (constant {0} *)src_;
    device {1} * dst = (device {1} *)dst_;

    packed_uint2 local_index;
    local_index.x = linear_index / size[1] % size[0];
    local_index.y = linear_index % size[1];

    const PackedT strided_index = VecT(local_index) * stride;
    dst[linear_index] = cast<{1}>(src[strided_index.x + strided_index.y]);
}}

template [[host_name("gather_kernel_2_u32")]]
kernel void gather_kernel_2<packed_uint2, uint2>(
    uint linear_index              [[thread_position_in_grid]],
    constant void * src_           [[buffer(0)]],
    device void * dst_             [[buffer(1)]],
    constant packed_uint2 & size   [[buffer(2)]],
    constant packed_uint2 & stride [[buffer(3)]],
    constant uint32_t & numel      [[buffer(4)]]);

template [[host_name("gather_kernel_2_u64")]]
kernel void gather_kernel_2<packed_ulong2, ulong2>(
    uint linear_index               [[thread_position_in_grid]],
    constant void * src_            [[buffer(0)]],
    device void * dst_              [[buffer(1)]],
    constant packed_uint2 & size    [[buffer(2)]],
    constant packed_ulong2 & stride [[buffer(3)]],
    constant uint32_t & numel       [[buffer(4)]]);

template <typename OffsetT>
kernel void gather_kernel_1(uint linear_index               [[thread_position_in_grid]],
                            constant void * src_            [[buffer(0)]],
                            device void * dst_              [[buffer(1)]],
                            constant int & size             [[buffer(2)]],
                            constant OffsetT & stride       [[buffer(3)]],
                            constant uint32_t & numel       [[buffer(4)]]) {{
    if (linear_index >= numel) return;

    constant {0} * src = (constant {0} *)src_;
    device {1} * dst = (device {1} *)dst_;

    const uint local_index = linear_index % size;
    const OffsetT strided_index = local_index * stride;
    dst[linear_index] = cast<{1}>(src[strided_index]);
}}

template [[host_name("gather_kernel_1_u32")]]
kernel void gather_kernel_1<uint>(
    uint linear_index              [[thread_position_in_grid]],
    constant void * src_           [[buffer(0)]],
    device void * dst_             [[buffer(1)]],
    constant int & size            [[buffer(2)]],
    constant uint & stride         [[buffer(3)]],
    constant uint32_t & numel      [[buffer(4)]]);

template [[host_name("gather_kernel_1_u64")]]
kernel void gather_kernel_1<ulong>(
    uint linear_index              [[thread_position_in_grid]],
    constant void * src_           [[buffer(0)]],
    device void * dst_             [[buffer(1)]],
    constant int & size            [[buffer(2)]],
    constant ulong & stride        [[buffer(3)]],
    constant uint32_t & numel      [[buffer(4)]]);
)METAL_GATHER";
} // namespace at::mps
