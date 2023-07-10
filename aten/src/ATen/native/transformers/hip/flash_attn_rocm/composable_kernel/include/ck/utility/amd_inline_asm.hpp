// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#ifndef CK_AMD_INLINE_ASM_HPP
#define CK_AMD_INLINE_ASM_HPP

#include "data_type.hpp"
#include "c_style_pointer_cast.hpp"

// TODO: deprecate all amd_assembly_outer_product_xxx

namespace ck {

// c0 += inner_product(a, b0)
// c1 += inner_product(a, b1)
__device__ void amd_assembly_outer_product_1x2(float a, float b0, float b1, float& c0, float& c1)
{
    asm volatile("\n \
            v_fmac_f32 %0, %2, %3 \n \
            v_fmac_f32 %1, %2, %4 \n \
            "
                 : "=v"(c0), "=v"(c1)
                 : "v"(a), "v"(b0), "v"(b1), "0"(c0), "1"(c1));
}

// c0 += inner_product(a, b0)
// c1 += inner_product(a, b1)
// c2 += inner_product(a, b2)
// c3 += inner_product(a, b3)
__device__ void amd_assembly_outer_product_1x4(
    float a, float b0, float b1, float b2, float b3, float& c0, float& c1, float& c2, float& c3)
{
    asm volatile("\n \
            v_fmac_f32 %0, %4, %5 \n \
            v_fmac_f32 %1, %4, %6 \n \
            v_fmac_f32 %2, %4, %7 \n \
            v_fmac_f32 %3, %4, %8 \n \
            "
                 : "=v"(c0), "=v"(c1), "=v"(c2), "=v"(c3)
                 : "v"(a), "v"(b0), "v"(b1), "v"(b2), "v"(b3), "0"(c0), "1"(c1), "2"(c2), "3"(c3));
}

// c0 += inner_product(a, b0)
// c1 += inner_product(a, b1)
__device__ void
amd_assembly_outer_product_1x2(half2_t a, half2_t b0, half2_t b1, float& c0, float& c1)
{
    asm volatile("\n \
            v_dot2_f32_f16 %0, %2, %3, %0\n \
            v_dot2_f32_f16 %1, %2, %4, %1\n \
            "
                 : "=v"(c0), "=v"(c1)
                 : "v"(a), "v"(b0), "v"(b1), "0"(c0), "1"(c1));
}

// c0 += inner_product(a, b0)
// c1 += inner_product(a, b1)
__device__ void
amd_assembly_outer_product_1x2(half4_t a, half4_t b0, half4_t b1, float& c0, float& c1)
{
    // TODO remove pointer casting
    const half2_t* p_a_half2  = c_style_pointer_cast<const half2_t*>(&a);
    const half2_t* p_b0_half2 = c_style_pointer_cast<const half2_t*>(&b0);
    const half2_t* p_b1_half2 = c_style_pointer_cast<const half2_t*>(&b1);

    // do dot2 two times
    asm volatile("\n \
            v_dot2_f32_f16 %0, %2, %4, %0\n \
            v_dot2_f32_f16 %1, %2, %6, %1\n \
            v_dot2_f32_f16 %0, %3, %5, %0\n \
            v_dot2_f32_f16 %1, %3, %7, %1\n \
            "
                 : "=v"(c0), "=v"(c1)
                 : "v"(p_a_half2[0]),
                   "v"(p_a_half2[1]),
                   "v"(p_b0_half2[0]),
                   "v"(p_b0_half2[1]),
                   "v"(p_b1_half2[0]),
                   "v"(p_b1_half2[1]),
                   "0"(c0),
                   "1"(c1));
}

// c0 += inner_product(a, b0)
// c1 += inner_product(a, b1)
// c2 += inner_product(a, b2)
// c3 += inner_product(a, b3)
__device__ void amd_assembly_outer_product_1x4(half2_t a,
                                               half2_t b0,
                                               half2_t b1,
                                               half2_t b2,
                                               half2_t b3,
                                               float& c0,
                                               float& c1,
                                               float& c2,
                                               float& c3)
{
    asm volatile("\n \
            v_dot2_f32_f16 %0, %4, %5, %0\n \
            v_dot2_f32_f16 %1, %4, %6, %1\n \
            v_dot2_f32_f16 %2, %4, %7, %2\n \
            v_dot2_f32_f16 %3, %4, %8, %3\n \
            "
                 : "=v"(c0), "=v"(c1), "=v"(c2), "=v"(c3)
                 : "v"(a), "v"(b0), "v"(b1), "v"(b2), "v"(b3), "0"(c0), "1"(c1), "2"(c2), "3"(c3));
}

// c0 += inner_product(a, b0)
// c1 += inner_product(a, b1)
// c2 += inner_product(a, b2)
// c3 += inner_product(a, b3)
__device__ void amd_assembly_outer_product_1x4(half4_t a,
                                               half4_t b0,
                                               half4_t b1,
                                               half4_t b2,
                                               half4_t b3,
                                               float& c0,
                                               float& c1,
                                               float& c2,
                                               float& c3)
{
    // TODO remove pointer casting
    const half2_t* p_a_half2  = c_style_pointer_cast<const half2_t*>(&a);
    const half2_t* p_b0_half2 = c_style_pointer_cast<const half2_t*>(&b0);
    const half2_t* p_b1_half2 = c_style_pointer_cast<const half2_t*>(&b1);
    const half2_t* p_b2_half2 = c_style_pointer_cast<const half2_t*>(&b2);
    const half2_t* p_b3_half2 = c_style_pointer_cast<const half2_t*>(&b3);

    // do dot2 two times
    asm volatile("\n \
            v_dot2_f32_f16 %0, %4, %6,  %0\n \
            v_dot2_f32_f16 %1, %4, %8,  %1\n \
            v_dot2_f32_f16 %2, %4, %10, %2\n \
            v_dot2_f32_f16 %3, %4, %12, %3\n \
            v_dot2_f32_f16 %0, %5, %7,  %0\n \
            v_dot2_f32_f16 %1, %5, %9,  %1\n \
            v_dot2_f32_f16 %2, %5, %11, %2\n \
            v_dot2_f32_f16 %3, %5, %13, %3\n \
            "
                 : "=v"(c0), "=v"(c1), "=v"(c2), "=v"(c3)
                 : "v"(p_a_half2[0]),
                   "v"(p_a_half2[1]),
                   "v"(p_b0_half2[0]),
                   "v"(p_b0_half2[1]),
                   "v"(p_b1_half2[0]),
                   "v"(p_b1_half2[1]),
                   "v"(p_b2_half2[0]),
                   "v"(p_b2_half2[1]),
                   "v"(p_b3_half2[0]),
                   "v"(p_b3_half2[1]),
                   "0"(c0),
                   "1"(c1),
                   "2"(c2),
                   "3"(c3));
}

__device__ void amd_assembly_outer_product_1x4(half8_t a,
                                               half8_t b0,
                                               half8_t b1,
                                               half8_t b2,
                                               half8_t b3,
                                               float& c0,
                                               float& c1,
                                               float& c2,
                                               float& c3)
{

    // TODO remove pointer casting
    const half4_t* p_a_half4  = c_style_pointer_cast<const half4_t*>(&a);
    const half4_t* p_b0_half4 = c_style_pointer_cast<const half4_t*>(&b0);
    const half4_t* p_b1_half4 = c_style_pointer_cast<const half4_t*>(&b1);
    const half4_t* p_b2_half4 = c_style_pointer_cast<const half4_t*>(&b2);
    const half4_t* p_b3_half4 = c_style_pointer_cast<const half4_t*>(&b3);

    amd_assembly_outer_product_1x4(
        p_a_half4[0], p_b0_half4[0], p_b1_half4[0], p_b2_half4[0], p_b3_half4[0], c0, c1, c2, c3);

    amd_assembly_outer_product_1x4(
        p_a_half4[1], p_b0_half4[1], p_b1_half4[1], p_b2_half4[1], p_b3_half4[1], c0, c1, c2, c3);
}

__device__ void amd_assembly_outer_product_1x4(half16_t a,
                                               half16_t b0,
                                               half16_t b1,
                                               half16_t b2,
                                               half16_t b3,
                                               float& c0,
                                               float& c1,
                                               float& c2,
                                               float& c3)
{
    // TODO remove pointer casting
    const half8_t* p_a_half8  = c_style_pointer_cast<const half8_t*>(&a);
    const half8_t* p_b0_half8 = c_style_pointer_cast<const half8_t*>(&b0);
    const half8_t* p_b1_half8 = c_style_pointer_cast<const half8_t*>(&b1);
    const half8_t* p_b2_half8 = c_style_pointer_cast<const half8_t*>(&b2);
    const half8_t* p_b3_half8 = c_style_pointer_cast<const half8_t*>(&b3);

    amd_assembly_outer_product_1x4(
        p_a_half8[0], p_b0_half8[0], p_b1_half8[0], p_b2_half8[0], p_b3_half8[0], c0, c1, c2, c3);

    amd_assembly_outer_product_1x4(
        p_a_half8[1], p_b0_half8[1], p_b1_half8[1], p_b2_half8[1], p_b3_half8[1], c0, c1, c2, c3);
}

// c0 += inner_product(a, b0)
// c1 += inner_product(a, b1)
__device__ void
amd_assembly_outer_product_1x2(int8x4_t a, int8x4_t b0, int8x4_t b1, int32_t& c0, int32_t& c1)
{
#if 1
    asm volatile("\n \
            v_dot4_i32_i8 %0, %2, %3, %0\n \
            v_dot4_i32_i8 %1, %2, %4, %1\n \
            "
                 : "=v"(c0), "=v"(c1)
                 : "v"(bit_cast<int32_t>(a)),
                   "v"(bit_cast<int32_t>(b0)),
                   "v"(bit_cast<int32_t>(b1)),
                   "0"(c0),
                   "1"(c1));
#else
    c0 = __builtin_amdgcn_sdot4(bit_cast<int32_t>(a), bit_cast<int32_t>(b0), c0, false);
    c1 = __builtin_amdgcn_sdot4(bit_cast<int32_t>(a), bit_cast<int32_t>(b1), c1, false);
#endif
}

// c0 += inner_product(a, b0)
// c1 += inner_product(a, b1)
// c2 += inner_product(a, b2)
// c3 += inner_product(a, b3)
__device__ void amd_assembly_outer_product_1x4(int8x4_t a,
                                               int8x4_t b0,
                                               int8x4_t b1,
                                               int8x4_t b2,
                                               int8x4_t b3,
                                               int32_t& c0,
                                               int32_t& c1,
                                               int32_t& c2,
                                               int32_t& c3)
{
#if 1
    asm volatile("\n \
            v_dot4_i32_i8 %0, %4, %5, %0\n \
            v_dot4_i32_i8 %1, %4, %6, %1\n \
            v_dot4_i32_i8 %2, %4, %7, %2\n \
            v_dot4_i32_i8 %3, %4, %8, %3\n \
            "
                 : "=v"(c0), "=v"(c1), "=v"(c2), "=v"(c3)
                 : "v"(bit_cast<int32_t>(a)),
                   "v"(bit_cast<int32_t>(b0)),
                   "v"(bit_cast<int32_t>(b1)),
                   "v"(bit_cast<int32_t>(b2)),
                   "v"(bit_cast<int32_t>(b3)),
                   "0"(c0),
                   "1"(c1),
                   "2"(c2),
                   "3"(c3));
#else
    c0 = __builtin_amdgcn_sdot4(bit_cast<int32_t>(a), bit_cast<int32_t>(b0), c0, false);
    c1 = __builtin_amdgcn_sdot4(bit_cast<int32_t>(a), bit_cast<int32_t>(b1), c1, false);
    c2 = __builtin_amdgcn_sdot4(bit_cast<int32_t>(a), bit_cast<int32_t>(b2), c2, false);
    c3 = __builtin_amdgcn_sdot4(bit_cast<int32_t>(a), bit_cast<int32_t>(b3), c3, false);
#endif
}

__device__ void amd_assembly_outer_product_1x4(int8x8_t a,
                                               int8x8_t b0,
                                               int8x8_t b1,
                                               int8x8_t b2,
                                               int8x8_t b3,
                                               int32_t& c0,
                                               int32_t& c1,
                                               int32_t& c2,
                                               int32_t& c3)
{
    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};

    amd_assembly_outer_product_1x4(vector_type<int8_t, 8>{a}.AsType<int8x4_t>()[I0],
                                   vector_type<int8_t, 8>{b0}.AsType<int8x4_t>()[I0],
                                   vector_type<int8_t, 8>{b1}.AsType<int8x4_t>()[I0],
                                   vector_type<int8_t, 8>{b2}.AsType<int8x4_t>()[I0],
                                   vector_type<int8_t, 8>{b3}.AsType<int8x4_t>()[I0],
                                   c0,
                                   c1,
                                   c2,
                                   c3);

    amd_assembly_outer_product_1x4(vector_type<int8_t, 8>{a}.AsType<int8x4_t>()[I1],
                                   vector_type<int8_t, 8>{b0}.AsType<int8x4_t>()[I1],
                                   vector_type<int8_t, 8>{b1}.AsType<int8x4_t>()[I1],
                                   vector_type<int8_t, 8>{b2}.AsType<int8x4_t>()[I1],
                                   vector_type<int8_t, 8>{b3}.AsType<int8x4_t>()[I1],
                                   c0,
                                   c1,
                                   c2,
                                   c3);
}

__device__ void amd_assembly_outer_product_1x4(int8x16_t a,
                                               int8x16_t b0,
                                               int8x16_t b1,
                                               int8x16_t b2,
                                               int8x16_t b3,
                                               int32_t& c0,
                                               int32_t& c1,
                                               int32_t& c2,
                                               int32_t& c3)

{
    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};
    constexpr auto I3 = Number<3>{};

    amd_assembly_outer_product_1x4(vector_type<int8_t, 16>{a}.AsType<int8x4_t>()[I0],
                                   vector_type<int8_t, 16>{b0}.AsType<int8x4_t>()[I0],
                                   vector_type<int8_t, 16>{b1}.AsType<int8x4_t>()[I0],
                                   vector_type<int8_t, 16>{b2}.AsType<int8x4_t>()[I0],
                                   vector_type<int8_t, 16>{b3}.AsType<int8x4_t>()[I0],
                                   c0,
                                   c1,
                                   c2,
                                   c3);

    amd_assembly_outer_product_1x4(vector_type<int8_t, 16>{a}.AsType<int8x4_t>()[I1],
                                   vector_type<int8_t, 16>{b0}.AsType<int8x4_t>()[I1],
                                   vector_type<int8_t, 16>{b1}.AsType<int8x4_t>()[I1],
                                   vector_type<int8_t, 16>{b2}.AsType<int8x4_t>()[I1],
                                   vector_type<int8_t, 16>{b3}.AsType<int8x4_t>()[I1],
                                   c0,
                                   c1,
                                   c2,
                                   c3);

    amd_assembly_outer_product_1x4(vector_type<int8_t, 16>{a}.AsType<int8x4_t>()[I2],
                                   vector_type<int8_t, 16>{b0}.AsType<int8x4_t>()[I2],
                                   vector_type<int8_t, 16>{b1}.AsType<int8x4_t>()[I2],
                                   vector_type<int8_t, 16>{b2}.AsType<int8x4_t>()[I2],
                                   vector_type<int8_t, 16>{b3}.AsType<int8x4_t>()[I2],
                                   c0,
                                   c1,
                                   c2,
                                   c3);

    amd_assembly_outer_product_1x4(vector_type<int8_t, 16>{a}.AsType<int8x4_t>()[I3],
                                   vector_type<int8_t, 16>{b0}.AsType<int8x4_t>()[I3],
                                   vector_type<int8_t, 16>{b1}.AsType<int8x4_t>()[I3],
                                   vector_type<int8_t, 16>{b2}.AsType<int8x4_t>()[I3],
                                   vector_type<int8_t, 16>{b3}.AsType<int8x4_t>()[I3],
                                   c0,
                                   c1,
                                   c2,
                                   c3);
}

} // namespace ck
#endif
