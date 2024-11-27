#pragma once

namespace at::mps {

static const char* SCATTER_OPS_TEMPLATE = R"METAL_SCATTER(
struct __attribute__ ((packed)) packed_uint5{{
  uint32_t x; uint32_t y; uint32_t z; uint32_t w; uint32_t u;
}};

template<typename Y, typename X>
Y cast(const X x);

template<>
{1} cast<{1}, {0}>(const {0} x) {{
 return {2};
}}

kernel void scatter_kernel_5(uint linear_index              [[thread_position_in_grid]],
                             constant void * src_           [[buffer(0)]],
                             device void * dst_             [[buffer(1)]],
                             constant packed_uint5 & size   [[buffer(2)]],
                             constant packed_uint5 & stride [[buffer(3)]],
                             constant uint32_t & numel      [[buffer(4)]]) {{
    if (linear_index >= numel) return;

    constant {0} * src = (constant {0} *)src_;
    device {1} * dst = (device {1} *)dst_;

    packed_uint5 local_index;
    local_index.x = linear_index / (size.u * size.w * size.z * size.y) % size.x;
    local_index.y = linear_index / (size.u * size.w * size.z) % size.y;
    local_index.z = linear_index / (size.u * size.w) % size.z;
    local_index.w = linear_index / size.u % size.w;
    local_index.u = linear_index % size.u;

    packed_uint5 strided_index;
    strided_index.x = local_index.x * stride.x;
    strided_index.y = local_index.y * stride.y;
    strided_index.z = local_index.z * stride.z;
    strided_index.w = local_index.w * stride.w;
    strided_index.u = local_index.u * stride.u;

    dst[strided_index.x + strided_index.y + strided_index.z + strided_index.w + strided_index.u] = cast<{1}>(src[linear_index]);
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
struct __attribute__ ((packed)) packed_uint5{{
  uint32_t x; uint32_t y; uint32_t z; uint32_t w; uint32_t u;
}};

template<typename Y, typename X>
Y cast(const X x);

template<>
{1} cast<{1}, {0}>(const {0} x) {{
 return {2};
}}

kernel void gather_kernel_5(uint linear_index               [[thread_position_in_grid]],
                            constant void * src_            [[buffer(0)]],
                            device void * dst_              [[buffer(1)]],
                            constant packed_uint5 & size    [[buffer(2)]],
                            constant packed_uint5 & stride  [[buffer(3)]],
                            constant uint32_t & numel       [[buffer(4)]]) {{
    if (linear_index >= numel) return;

    constant {0} * src = (constant {0} *)src_;
    device {1} * dst = (device {1} *)dst_;


    packed_uint5 local_index;
    local_index.x = linear_index / (size.u * size.w * size.z * size.y) % size.x;
    local_index.y = linear_index / (size.u * size.w * size.z) % size.y;
    local_index.z = linear_index / (size.u * size.w) % size.z;
    local_index.w = linear_index / size.u % size.w;
    local_index.u = linear_index % size.u;

    packed_uint5 strided_index;
    strided_index.x = local_index.x * stride.x;
    strided_index.y = local_index.y * stride.y;
    strided_index.z = local_index.z * stride.z;
    strided_index.w = local_index.w * stride.w;
    strided_index.u = local_index.u * stride.u;

    dst[linear_index] = cast<{1}>(src[strided_index.x + strided_index.y + strided_index.z + strided_index.w + strided_index.u]);
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
