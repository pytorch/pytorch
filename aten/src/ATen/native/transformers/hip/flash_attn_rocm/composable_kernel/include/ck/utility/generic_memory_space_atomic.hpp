// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once
#include "data_type.hpp"

namespace ck {

// Caution: DO NOT REMOVE
// intentionally have only declaration but no definition to cause compilation failure when trying to
// instantiate this template. The purpose is to make the implementation of atomic_add explicit for
// each datatype.
template <typename X>
__device__ X atomic_add(X* p_dst, const X& x);

template <>
__device__ int32_t atomic_add<int32_t>(int32_t* p_dst, const int32_t& x)
{
    return atomicAdd(p_dst, x);
}

template <>
__device__ uint32_t atomic_add<uint32_t>(uint32_t* p_dst, const uint32_t& x)
{
    return atomicAdd(p_dst, x);
}

template <>
__device__ float atomic_add<float>(float* p_dst, const float& x)
{
    return atomicAdd(p_dst, x);
}

template <>
__device__ double atomic_add<double>(double* p_dst, const double& x)
{
    return atomicAdd(p_dst, x);
}

template <>
__device__ float2_t atomic_add<float2_t>(float2_t* p_dst, const float2_t& x)
{
    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};

    const vector_type<float, 2> vx{x};
    vector_type<float, 2> vy{0};

    vy.template AsType<float>()(I0) =
        atomicAdd(c_style_pointer_cast<float*>(p_dst), vx.template AsType<float>()[I0]);
    vy.template AsType<float>()(I1) =
        atomicAdd(c_style_pointer_cast<float*>(p_dst) + 1, vx.template AsType<float>()[I1]);

    return vy.template AsType<float2_t>()[I0];
}

template <>
__device__ double2_t atomic_add<double2_t>(double2_t* p_dst, const double2_t& x)
{
    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};

    const vector_type<double, 2> vx{x};
    vector_type<double, 2> vy{0};

    vy.template AsType<double>()(I0) =
        atomicAdd(c_style_pointer_cast<double*>(p_dst), vx.template AsType<double>()[I0]);
    vy.template AsType<double>()(I1) =
        atomicAdd(c_style_pointer_cast<double*>(p_dst) + 1, vx.template AsType<double>()[I1]);

    return vy.template AsType<double2_t>()[I0];
}

inline __host__ __device__ half2_t add_fp16x2_t(const half2_t& a, const half2_t& b)
{
    half2_t rtn;
    rtn[0] = a[0] + b[0];
    rtn[1] = a[1] + b[1];
    return rtn;
}

union U32FP162_ADDR
{
    uint32_t* u32_a;
    half2_t* fp162_a;
};

union U32FP162
{
    uint32_t u32;
    half2_t fp162;
};

template <>
__device__ half2_t atomic_add<half2_t>(half2_t* p_dst, const half2_t& x)
{
    U32FP162_ADDR dword_addr;
    U32FP162 cur_v;
    U32FP162 new_;
    uint32_t old_v, new_v;
    dword_addr.fp162_a = p_dst;
    cur_v.u32          = *dword_addr.u32_a;

    do
    {
        old_v      = cur_v.u32;
        new_.fp162 = add_fp16x2_t(cur_v.fp162, x);
        new_v      = new_.u32;
        cur_v.u32  = atomicCAS(dword_addr.u32_a, old_v, new_v);
    } while(cur_v.u32 != old_v);

    return x;
}

// template <>
// __device__ half2_t atomic_add<half2_t>(half2_t* p_dst, const half2_t& x)
// {
//     uint32_t * dword_addr = reinterpret_cast<uint32_t*>(p_dst);
//     uint32_t cur_v = *dword_addr;
//     uint32_t old_v, new_v;

//     do {
//         old_v = cur_v;
//         half2_t new_ = add_fp16x2_t(*reinterpret_cast<half2_t*>(&cur_v), x);
//         new_v = *reinterpret_cast<uint32_t*>(&new_);
//         cur_v = atomicCAS(dword_addr, old_v, new_v);
//     }while(cur_v != old_v);

//     return x;
// }

// union U16BF16 {
//     uint16_t u16;
//     bhalf_t bf16;
// };

// inline __host__ __device__ bhalf_t add_bf16_t(const bhalf_t& a, const bhalf_t& b){
//     U16BF16 xa {.bf16 = a};
//     U16BF16 xb {.bf16 = b};

//     U16BF16 xr;
//     xr.u16 = xa.u16 + xb.u16;
//     return xr.bf16;
// }

inline __host__ __device__ bhalf_t add_bf16_t(const bhalf_t& a, const bhalf_t& b)
{
    return type_convert<bhalf_t>(type_convert<float>(a) + type_convert<float>(b));
}

inline __host__ __device__ bhalf2_t add_bf16x2_t(const bhalf2_t& a, const bhalf2_t& b)
{
    bhalf2_t rtn;
    rtn[0] = add_bf16_t(a[0], b[0]);
    rtn[1] = add_bf16_t(a[1], b[1]);
    return rtn;
}

union U32BF162_ADDR
{
    uint32_t* u32_a;
    bhalf2_t* bf162_a;
};

union U32BF162
{
    uint32_t u32;
    bhalf2_t bf162;
};

template <>
__device__ bhalf2_t atomic_add<bhalf2_t>(bhalf2_t* p_dst, const bhalf2_t& x)
{
    U32BF162_ADDR dword_addr;
    U32BF162 cur_v;
    U32BF162 new_;
    uint32_t old_v, new_v;
    dword_addr.bf162_a = p_dst;
    cur_v.u32          = *dword_addr.u32_a;

    do
    {
        old_v      = cur_v.u32;
        new_.bf162 = add_bf16x2_t(cur_v.bf162, x);
        new_v      = new_.u32;
        cur_v.u32  = atomicCAS(dword_addr.u32_a, old_v, new_v);
    } while(cur_v.u32 != old_v);

    return x;
}

// template <>
// __device__ bhalf2_t atomic_add<bhalf2_t>(bhalf2_t* p_dst, const bhalf2_t& x)
// {
//     uint32_t * dword_addr = reinterpret_cast<uint32_t*>(p_dst);
//     uint32_t cur_v = *dword_addr;
//     uint32_t old_v, new_v;

//     do {
//         old_v = cur_v;
//         bhalf2_t new_ = add_bf16x2_t(*reinterpret_cast<bhalf2_t*>(&cur_v), x);
//         new_v = *reinterpret_cast<uint32_t*>(&new_);
//         cur_v = atomicCAS(dword_addr, old_v, new_v);
//     }while(cur_v != old_v);

//     return x;
// }

// Caution: DO NOT REMOVE
// intentionally have only declaration but no definition to cause compilation failure when trying to
// instantiate this template. The purpose is to make the implementation of atomic_max explicit for
// each datatype.

template <typename X>
__device__ X atomic_max(X* p_dst, const X& x);

template <>
__device__ int32_t atomic_max<int32_t>(int32_t* p_dst, const int32_t& x)
{
    return atomicMax(p_dst, x);
}

template <>
__device__ uint32_t atomic_max<uint32_t>(uint32_t* p_dst, const uint32_t& x)
{
    return atomicMax(p_dst, x);
}

template <>
__device__ float atomic_max<float>(float* p_dst, const float& x)
{
    return atomicMax(p_dst, x);
}

template <>
__device__ double atomic_max<double>(double* p_dst, const double& x)
{
    return atomicMax(p_dst, x);
}

template <>
__device__ float2_t atomic_max<float2_t>(float2_t* p_dst, const float2_t& x)
{
    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};

    const vector_type<float, 2> vx{x};
    vector_type<float, 2> vy{0};

    vy.template AsType<float>()(I0) =
        atomicMax(c_style_pointer_cast<float*>(p_dst), vx.template AsType<float>()[I0]);
    vy.template AsType<float>()(I1) =
        atomicMax(c_style_pointer_cast<float*>(p_dst) + 1, vx.template AsType<float>()[I1]);

    return vy.template AsType<float2_t>()[I0];
}

} // namespace ck
