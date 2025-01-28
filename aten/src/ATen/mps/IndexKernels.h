#pragma once

namespace at::mps {

static const char* SCATTER_OPS_TEMPLATE = R"METAL_SCATTER(
template<typename Y, typename X>
Y cast(const X x);

template<>
{1} cast<{1}, {0}>(const {0} x) {{
 return {2};
}}

kernel void scatter_kernel_n(uint linear_index          [[thread_position_in_grid]],
                             constant void * src_       [[buffer(0)]],
                             device void * dst_         [[buffer(1)]],
                             constant uint32_t * size   [[buffer(2)]],
                             constant uint32_t * stride [[buffer(3)]],
                            constant uint32_t & numel   [[buffer(4)]],
                            constant int32_t & ndim     [[buffer(5)]]) {{
    if (linear_index >= numel) return;

    constant {0} * src = (constant {0} *)src_;
    device {1} * dst = (device {1} *)dst_;

    uint64_t dst_offs = 0;
    auto dst_idx = linear_index;
    for(int dim = ndim - 1; dim >= 0; --dim) {{
      dst_offs += stride[dim] * (dst_idx % size[dim]);
      dst_idx /= size[dim];
    }}

    dst[dst_offs] = cast<{1}>(src[linear_index]);
}}

kernel void scatter_kernel_4(uint linear_index              [[thread_position_in_grid]],
                             constant void * src_           [[buffer(0)]],
                             device void * dst_             [[buffer(1)]],
                             constant packed_uint4 & size   [[buffer(2)]],
                             constant packed_uint4 & stride [[buffer(3)]],
                             constant uint32_t & numel      [[buffer(4)]]) {{
    if (linear_index >= numel) return;

    constant {0} * src = (constant {0} *)src_;
    device {1} * dst = (device {1} *)dst_;

    packed_uint4 local_index;
    local_index.x = linear_index / (size[3] * size[2] * size[1]) % size[0];
    local_index.y = linear_index / (size[3] * size[2]) % size[1];
    local_index.z = linear_index / size[3] % size[2];
    local_index.w = linear_index % size[3];

    const packed_uint4 strided_index = local_index * stride;
    dst[strided_index.x + strided_index.y + strided_index.z + strided_index.w] = cast<{1}>(src[linear_index]);
}}

kernel void scatter_kernel_3(uint linear_index              [[thread_position_in_grid]],
                             constant void * src_           [[buffer(0)]],
                             device void * dst_             [[buffer(1)]],
                             constant packed_uint3 & size   [[buffer(2)]],
                             constant packed_uint3 & stride [[buffer(3)]],
                             constant uint32_t & numel      [[buffer(4)]]) {{
    if (linear_index >= numel) return;

    constant {0} * src = (constant {0} *)src_;
    device {1} * dst = (device {1} *)dst_;

    packed_uint3 local_index;
    local_index.x = linear_index / (size[2] * size[1]) % size[0];
    local_index.y = linear_index / size[2] % size[1];
    local_index.z = linear_index % size[2];

    const packed_uint3 strided_index = local_index * stride;
    dst[strided_index.x + strided_index.y + strided_index.z] = cast<{1}>(src[linear_index]);
}}

kernel void scatter_kernel_2(uint linear_index              [[thread_position_in_grid]],
                             constant void * src_           [[buffer(0)]],
                             device void * dst_             [[buffer(1)]],
                             constant packed_uint2 & size   [[buffer(2)]],
                             constant packed_uint2 & stride [[buffer(3)]],
                             constant uint32_t & numel      [[buffer(4)]]) {{
    if (linear_index >= numel) return;

    constant {0} * src = (constant {0} *)src_;
    device {1} * dst = (device {1} *)dst_;

    packed_uint2 local_index;
    local_index.x = linear_index / size[1] % size[0];
    local_index.y = linear_index % size[1];

    const packed_uint2 strided_index = local_index * stride;
    dst[strided_index.x + strided_index.y] = cast<{1}>(src[linear_index]);
}}

kernel void scatter_kernel_1(uint linear_index              [[thread_position_in_grid]],
                             constant void * src_           [[buffer(0)]],
                             device void * dst_             [[buffer(1)]],
                             constant int & size            [[buffer(2)]],
                             constant int & stride          [[buffer(3)]],
                             constant uint32_t & numel      [[buffer(4)]]) {{
    if (linear_index >= numel) return;

    constant {0} * src = (constant {0} *)src_;
    device {1} * dst = (device {1} *)dst_;

    const int local_index = linear_index % size;
    const int strided_index = local_index * stride;
    dst[strided_index] = cast<{1}>(src[linear_index]);
}}
)METAL_SCATTER";

static const char* GATHER_OPS_TEMPLATE = R"METAL_GATHER(
template<typename Y, typename X>
Y cast(const X x);

template<>
{1} cast<{1}, {0}>(const {0} x) {{
 return {2};
}}

kernel void gather_kernel_n(uint linear_index           [[thread_position_in_grid]],
                            constant void * src_        [[buffer(0)]],
                            device void * dst_          [[buffer(1)]],
                            constant uint32_t * size    [[buffer(2)]],
                            constant uint32_t * stride  [[buffer(3)]],
                            constant uint32_t & numel   [[buffer(4)]],
                            constant int32_t & ndim     [[buffer(5)]]) {{
    if (linear_index >= numel) return;

    constant {0} * src = (constant {0} *)src_;
    device {1} * dst = (device {1} *)dst_;

    uint64_t src_offs = 0;
    auto src_idx = linear_index;
    for(int dim = ndim - 1; dim >= 0; --dim) {{
      src_offs += stride[dim] * (src_idx % size[dim]);
      src_idx /= size[dim];
    }}

    dst[linear_index] = cast<{1}>(src[src_offs]);
}}

kernel void gather_kernel_4(uint linear_index               [[thread_position_in_grid]],
                            constant void * src_            [[buffer(0)]],
                            device void * dst_              [[buffer(1)]],
                            constant packed_uint4 & size    [[buffer(2)]],
                            constant packed_uint4 & stride  [[buffer(3)]],
                            constant uint32_t & numel       [[buffer(4)]]) {{
    if (linear_index >= numel) return;

    constant {0} * src = (constant {0} *)src_;
    device {1} * dst = (device {1} *)dst_;

    packed_uint4 local_index;
    local_index.x = linear_index / (size[3] * size[2] * size[1]) % size[0];
    local_index.y = linear_index / (size[3] * size[2]) % size[1];
    local_index.z = linear_index / size[3] % size[2];
    local_index.w = linear_index % size[3];

    const packed_uint4 strided_index = local_index * stride;
    dst[linear_index] = cast<{1}>(src[strided_index.x + strided_index.y + strided_index.z + strided_index.w]);
}}

kernel void gather_kernel_3(uint linear_index               [[thread_position_in_grid]],
                            constant void * src_            [[buffer(0)]],
                            device void * dst_              [[buffer(1)]],
                            constant packed_uint3 & size    [[buffer(2)]],
                            constant packed_uint3 & stride  [[buffer(3)]],
                            constant uint32_t & numel       [[buffer(4)]]) {{
    if (linear_index >= numel) return;

    constant {0} * src = (constant {0} *)src_;
    device {1} * dst = (device {1} *)dst_;

    packed_uint3 local_index;
    local_index.x = linear_index / (size[2] * size[1]) % size[0];
    local_index.y = linear_index / size[2] % size[1];
    local_index.z = linear_index % size[2];

    const packed_uint3 strided_index = local_index * stride;
    dst[linear_index] = cast<{1}>(src[strided_index.x + strided_index.y + strided_index.z]);
}}

kernel void gather_kernel_2(uint linear_index               [[thread_position_in_grid]],
                            constant void * src_            [[buffer(0)]],
                            device void * dst_              [[buffer(1)]],
                            constant packed_uint2 & size    [[buffer(2)]],
                            constant packed_uint2 & stride  [[buffer(3)]],
                            constant uint32_t & numel       [[buffer(4)]]) {{
    if (linear_index >= numel) return;

    constant {0} * src = (constant {0} *)src_;
    device {1} * dst = (device {1} *)dst_;

    packed_uint2 local_index;
    local_index.x = linear_index / size[1] % size[0];
    local_index.y = linear_index % size[1];

    const packed_uint2 strided_index = local_index * stride;
    dst[linear_index] = cast<{1}>(src[strided_index.x + strided_index.y]);
}}

kernel void gather_kernel_1(uint linear_index               [[thread_position_in_grid]],
                            constant void * src_            [[buffer(0)]],
                            device void * dst_              [[buffer(1)]],
                            constant int & size             [[buffer(2)]],
                            constant int & stride           [[buffer(3)]],
                            constant uint32_t & numel       [[buffer(4)]]) {{
    if (linear_index >= numel) return;

    constant {0} * src = (constant {0} *)src_;
    device {1} * dst = (device {1} *)dst_;

    const int local_index = linear_index % size;
    const int strided_index = local_index * stride;
    dst[linear_index] = cast<{1}>(src[strided_index]);
}}
)METAL_GATHER";
} // namespace at::mps
