/*
 * Portions of this file are adapted from DeepEP (https://github.com/deepseek-ai/DeepEP).
 * Copyright (c) 2025 DeepSeek. Licensed under the MIT License.
 * SPDX-License-Identifier: MIT
 */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 * See LICENSE.txt for more license information.
 */

#pragma once

#include "common.hpp"

//==============================================================================
// Macros
//==============================================================================

#define UNROLLED_WARP_COPY(UNROLL_FACTOR, LANE_ID, N, DST, SRC, LD_FUNC, ST_FUNC) \
{ \
    constexpr int kLoopStride = 32 * (UNROLL_FACTOR); \
    typename std::remove_reference<decltype(LD_FUNC((SRC) + 0))>::type unrolled_values[(UNROLL_FACTOR)]; \
    auto __src = (SRC); \
    auto __dst = (DST); \
    for (int __i = (LANE_ID); __i < ((N) / kLoopStride) * kLoopStride; __i += kLoopStride) { \
        _Pragma("unroll") \
        for (int __j = 0; __j < (UNROLL_FACTOR); ++ __j) \
            unrolled_values[__j] = LD_FUNC(__src + __i + __j * 32); \
        _Pragma("unroll") \
        for (int __j = 0; __j < (UNROLL_FACTOR); ++ __j) \
            ST_FUNC(__dst + __i + __j * 32, unrolled_values[__j]); \
    } \
    { \
        int __i = ((N) / kLoopStride) * kLoopStride + (LANE_ID); \
        _Pragma("unroll") \
        for (int __j = 0; __j < (UNROLL_FACTOR); ++ __j) { \
            if (__i + __j * 32 < (N)) { \
                unrolled_values[__j] = LD_FUNC(__src + __i + __j * 32); \
            } \
        } \
        _Pragma("unroll") \
        for (int __j = 0; __j < (UNROLL_FACTOR); ++ __j) { \
            if (__i + __j * 32 < (N)) { \
                ST_FUNC(__dst + __i + __j * 32, unrolled_values[__j]); \
            } \
        } \
    } \
}

namespace nccl_ep {

//==============================================================================
// Type Traits and Helper Types
//==============================================================================

template <int kBytes>
struct VecInt {};
template<> struct VecInt<1>  { using vec_t = int8_t; };
template<> struct VecInt<2>  { using vec_t = int16_t; };
template<> struct VecInt<4>  { using vec_t = int; };
template<> struct VecInt<8>  { using vec_t = int64_t; };
template<> struct VecInt<16> { using vec_t = int4; };

template <typename FuncT>
struct PatternVisitor {
    FuncT func;

    __device__ __host__
    explicit PatternVisitor(FuncT&& func): func(std::forward<FuncT>(func)) {}

    __device__ __host__
    auto operator [](const uint32_t& i) {
        return func(i);
    }
};

//==============================================================================
// Basic Device Utilities
//==============================================================================

__device__ __forceinline__ void trap() {
    asm("trap;");
}

__forceinline__ __device__ int get_lane_id() {
    int lane_id;
    asm("mov.s32 %0, %laneid;" : "=r"(lane_id));
    return lane_id;
}

//==============================================================================
// Math Utilities
//==============================================================================

__device__ __forceinline__ float log2f_approx(const float &x) {
    float ret;
    asm volatile("lg2.approx.f32 %0, %1;" : "=f"(ret) : "f"(x));
    return ret;
}

__device__ __forceinline__ float exp2f_approx(const float &x) {
    float ret;
    asm volatile("ex2.approx.f32 %0, %1;" : "=f"(ret) : "f"(x));
    return ret;
}

__forceinline__ __device__ float fast_pow2(int x) {
    // We can ensure `-126 <= x and x <= 127`
    uint32_t bits_x = (x + 127) << 23;
    return *reinterpret_cast<float*>(&bits_x);
}

__forceinline__ __device__ int fast_log2_ceil(float x) {
    auto bits_x = *reinterpret_cast<uint32_t*>(&x);
    auto exp_x = (bits_x >> 23) & 0xff;
    auto man_bits = bits_x & ((1 << 23) - 1);
    return exp_x - 127 + (man_bits != 0);
}

//==============================================================================
// FP8 Utilities
//==============================================================================

constexpr float kFP8Margin = 1e-4;
constexpr float kFinfoAmaxE4M3 = 448.0f;
constexpr float kFinfoAmaxInvE4M3 = 1 / 448.0f;

__forceinline__ __device__ void calculate_fp8_scales(float amax, float& scale, float& scale_inv, bool round_scale) {
    if (round_scale) {
        auto exp_scale_inv = fast_log2_ceil(amax * kFinfoAmaxInvE4M3);
        scale = fast_pow2(-exp_scale_inv);
        scale_inv = fast_pow2(exp_scale_inv);
    } else {
        scale_inv = amax * kFinfoAmaxInvE4M3;
        scale = kFinfoAmaxE4M3 / amax;
    }
}

template <bool kIsUE8M0, typename out_dtype_t = std::conditional_t<kIsUE8M0, uint8_t, float>>
__forceinline__ __device__ out_dtype_t extract_required_scale_format(float value) {
    if constexpr (kIsUE8M0) {
        return static_cast<uint8_t>((*reinterpret_cast<uint32_t*>(&value)) >> 23);
    } else {
        return value;
    }
}

//==============================================================================
// Memory Fences
//==============================================================================

__device__ __forceinline__ void memory_fence() {
    asm volatile("fence.acq_rel.sys;":: : "memory");
}

__device__ __forceinline__ void memory_fence_gpu() {
    asm volatile("fence.acq_rel.gpu;":: : "memory");
}

__device__ __forceinline__ void memory_fence_cta() {
    asm volatile("fence.acq_rel.cta;":: : "memory");
}

// Drains async-proxy (TMA) stores into the generic memory path. Caller must be
// the thread that issued the cp.async.bulk, after cp_async_bulk_wait_group.
__device__ __forceinline__ void fence_proxy_async() {
    asm volatile("fence.proxy.async;" ::: "memory");
}

//==============================================================================
// Load Operations - Volatile
//==============================================================================

__device__ __forceinline__ int ld_volatile_global(const int *ptr) {
    int ret;
    asm volatile("ld.volatile.global.s32 %0, [%1];" : "=r"(ret) : "l"(ptr));
    return ret;
}

__device__ __forceinline__ float ld_volatile_global(const float *ptr) {
    float ret;
    asm volatile("ld.volatile.global.f32 %0, [%1];" : "=f"(ret) : "l"(ptr));
    return ret;
}

__device__ __forceinline__ int64_t ld_volatile_global(const int64_t *ptr) {
    int64_t ret;
    asm volatile("ld.volatile.global.s64 %0, [%1];" : "=l"(ret) : "l"(ptr));
    return ret;
}

__device__ __forceinline__ int64_t ld_volatile_global(const uint64_t *ptr) {
    int64_t ret;
    asm volatile("ld.volatile.global.u64 %0, [%1];" : "=l"(ret) : "l"(ptr));
    return ret;
}

//==============================================================================
// Load Operations - Relaxed (system scope)
//==============================================================================

__device__ __forceinline__ uint32_t ld_relaxed_sys_global(const uint32_t* ptr) {
    uint32_t ret;
    asm volatile("ld.relaxed.sys.global.u32 %0, [%1];" : "=r"(ret) : "l"(ptr) : "memory");
    return ret;
}

__device__ __forceinline__ uint64_t ld_relaxed_gpu_global(const uint64_t* ptr) {
    uint64_t ret;
    asm volatile("ld.relaxed.gpu.global.b64 %0, [%1];"
                 : "=l"(ret)
                 : "l"(__cvta_generic_to_global(ptr))
                 : "memory");
    return ret;
}

//==============================================================================
// Load Operations - Acquire
//==============================================================================

__device__ __forceinline__ int ld_acquire_sys_global(const int *ptr) {
    int ret;
    asm volatile("ld.acquire.sys.global.s32 %0, [%1];" : "=r"(ret) : "l"(ptr));
    return ret;
}

__device__ __forceinline__ uint64_t ld_acquire_sys_global(const uint64_t *ptr) {
    uint64_t ret;
    asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(ret) : "l"(ptr));
    return ret;
}

__device__ __forceinline__ int ld_acquire_global(const int *ptr) {
    int ret;
    asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(ret) : "l"(ptr));
    return ret;
}

__device__ __forceinline__ int ld_acquire_cta(const int *ptr) {
    int ret;
    asm volatile("ld.acquire.cta.s32 %0, [%1];" : "=r"(ret) : "l"(ptr));
    return ret;
}

//==============================================================================
// Load Operations - Relaxed (No Allocate)
//==============================================================================

__device__ __forceinline__ uint8_t ld_na_relaxed(const uint8_t *ptr) {
    uint16_t ret;
    asm volatile("ld.relaxed.gpu.global.L1::no_allocate.b8 %0, [%1];" : "=h"(ret) : "l"(ptr));
    return static_cast<uint8_t>(ret);
}

__device__ __forceinline__ uint16_t ld_na_relaxed(const uint16_t *ptr) {
    uint16_t ret;
    asm volatile("ld.relaxed.gpu.global.L1::no_allocate.b16 %0, [%1];" : "=h"(ret) : "l"(ptr));
    return ret;
}

__device__ __forceinline__ uint32_t ld_na_relaxed(const uint32_t *ptr) {
    uint32_t ret;
    asm volatile("ld.relaxed.gpu.global.L1::no_allocate.b32 %0, [%1];" : "=r"(ret) : "l"(ptr));
    return ret;
}

__device__ __forceinline__ uint64_t ld_na_relaxed(const uint64_t *ptr) {
    uint64_t ret;
    asm volatile("ld.relaxed.gpu.global.L1::no_allocate.b64 %0, [%1];" : "=l"(ret) : "l"(ptr));
    return ret;
}

//==============================================================================
// Load Operations - Non-Caching (ld.nc)
//==============================================================================

// `ld.global.nc.L1::no_allocate` will be translated into `LDG.E.NA.[width].CONSTANT` in SASS
#ifndef DISABLE_AGGRESSIVE_PTX_INSTRS
#define LD_NC_FUNC "ld.global.nc.L1::no_allocate.L2::256B"
#else
#define LD_NC_FUNC "ld.volatile.global"
#endif

template <typename dtype_t>
__device__ __forceinline__ dtype_t ld_nc_global(const dtype_t *ptr) {
    auto ret = ld_nc_global(reinterpret_cast<const typename VecInt<sizeof(dtype_t)>::vec_t*>(ptr));
    return *reinterpret_cast<dtype_t*>(&ret);
}

template <>
__device__ __forceinline__ uint8_t ld_nc_global(const uint8_t *ptr) {
    uint16_t ret;
    // NOTES: we must use `uint16_t` as inline ASM does not support 8-bit constraint letter (`h` below means unsigned 16-bit)
    asm volatile(LD_NC_FUNC ".u8 %0, [%1];" : "=h"(ret) : "l"(ptr));
    return static_cast<uint8_t>(ret);
}

template <>
__device__ __forceinline__ int ld_nc_global(const int *ptr) {
    int ret;
    asm volatile(LD_NC_FUNC ".s32 %0, [%1];" : "=r"(ret) : "l"(ptr));
    return ret;
}

template <>
__device__ __forceinline__ int64_t ld_nc_global(const int64_t *ptr) {
    int64_t ret;
    asm volatile(LD_NC_FUNC ".s64 %0, [%1];" : "=l"(ret) : "l"(ptr));
    return ret;
}

template <>
__device__ __forceinline__ float ld_nc_global(const float *ptr) {
    float ret;
    asm volatile(LD_NC_FUNC ".f32 %0, [%1];" : "=f"(ret) : "l"(ptr));
    return ret;
}

template <>
__device__ __forceinline__ int2 ld_nc_global(const int2 *ptr) {
    int2 ret;
    asm volatile(LD_NC_FUNC ".v2.s32 {%0, %1}, [%2];" : "=r"(ret.x), "=r"(ret.y) : "l"(ptr));
    return ret;
}

template <>
__device__ __forceinline__ int4 ld_nc_global(const int4 *ptr) {
    int4 ret;
    asm volatile(LD_NC_FUNC ".v4.s32 {%0, %1, %2, %3}, [%4];"
            : "=r"(ret.x), "=r"(ret.y), "=r"(ret.z), "=r"(ret.w) : "l"(ptr));
    return ret;
}

//==============================================================================
// Store Operations - Release
//==============================================================================

__device__ __forceinline__ void st_release_sys_global(const int *ptr, int val) {
    asm volatile("st.release.sys.global.s32 [%0], %1;"::"l"(ptr), "r"(val) : "memory");
}

__device__ __forceinline__ void st_release_cta(const int *ptr, int val) {
    asm volatile("st.release.cta.s32 [%0], %1;"::"l"(ptr), "r"(val) : "memory");
}

//==============================================================================
// Store Operations - Relaxed
//==============================================================================

__device__ __forceinline__ void st_relaxed_sys_global(const int *ptr, int val) {
    asm volatile("st.relaxed.sys.global.s32 [%0], %1;"::"l"(ptr), "r"(val) : "memory");
}

__device__ __forceinline__ void st_relaxed_gpu_global(uint64_t* ptr, uint64_t val) {
    asm volatile("st.relaxed.gpu.global.b64 [%0], %1;"
                 :
                 : "l"(__cvta_generic_to_global(ptr)), "l"(val)
                 : "memory");
}

__device__ __forceinline__ void st_na_relaxed(const uint8_t *ptr, uint8_t val) {
    asm volatile("st.relaxed.gpu.global.L1::no_allocate.b8 [%0], %1;" : : "l"(ptr), "h"(static_cast<uint16_t>(val)));
}

__device__ __forceinline__ void st_na_relaxed(const uint16_t *ptr, uint16_t val) {
    asm volatile("st.relaxed.gpu.global.L1::no_allocate.b16 [%0], %1;" : : "l"(ptr), "h"(val));
}

__device__ __forceinline__ void st_na_relaxed(const uint32_t *ptr, uint32_t val) {
    asm volatile("st.relaxed.gpu.global.L1::no_allocate.b32 [%0], %1;" : : "l"(ptr), "r"(val));
}

__device__ __forceinline__ void st_na_relaxed(const int *ptr, int val) {
    asm volatile("st.relaxed.gpu.global.L1::no_allocate.b32 [%0], %1;" : : "l"(ptr), "r"(val));
}

__device__ __forceinline__ void st_na_relaxed(const int4 *ptr, int4 val) {
    asm volatile("st.relaxed.gpu.global.L1::no_allocate.v4.s32 [%0], {%1, %2, %3, %4};"
            : : "l"(ptr), "r"(val.x), "r"(val.y), "r"(val.z), "r"(val.w));
}

__device__ __forceinline__ void st_na_release(const int *ptr, int val) {
    asm volatile("st.release.gpu.global.L1::no_allocate.b32 [%0], %1;" : : "l"(ptr), "r"(val));
}

__device__ __forceinline__ void st_na_release(const uint32_t *ptr, uint32_t val) {
    asm volatile("st.release.gpu.global.L1::no_allocate.b32 [%0], %1;" : : "l"(ptr), "r"(val));
}

__device__ __forceinline__ void st_na_release(const uint64_t *ptr, uint64_t val) {
    asm volatile("st.release.gpu.global.L1::no_allocate.b64 [%0], %1;" : : "l"(ptr), "l"(val));
}

//==============================================================================
// Store Operations - Non-Allocating (st.na)
//==============================================================================

// `st.global.L1::no_allocate` will be translated into `ST.E.NA.[width]` in SASS
#ifndef DISABLE_AGGRESSIVE_PTX_INSTRS
#define ST_NA_FUNC "st.global.L1::no_allocate"
#else
#define ST_NA_FUNC "st.global"
#endif

template <typename dtype_t>
__device__ __forceinline__ void st_na_global(const dtype_t *ptr, const dtype_t& value) {
    st_na_global(reinterpret_cast<const typename VecInt<sizeof(dtype_t)>::vec_t*>(ptr),
                 *reinterpret_cast<const typename VecInt<sizeof(dtype_t)>::vec_t*>(&value));
}

template <>
__device__ __forceinline__ void st_na_global(const int *ptr, const int& value) {
    asm volatile(ST_NA_FUNC ".s32 [%0], %1;" ::"l"(ptr), "r"(value));
}

template <>
__device__ __forceinline__ void st_na_global(const int64_t *ptr, const int64_t& value) {
    asm volatile(ST_NA_FUNC ".s64 [%0], %1;" ::"l"(ptr), "l"(value));
}

template <>
__device__ __forceinline__ void st_na_global(const float *ptr, const float& value) {
    asm volatile(ST_NA_FUNC ".f32 [%0], %1;" ::"l"(ptr), "f"(value));
}

template <>
__device__ __forceinline__ void st_na_global(const int4 *ptr, const int4& value) {
    asm volatile(ST_NA_FUNC ".v4.s32 [%0], {%1, %2, %3, %4};"
            ::"l"(ptr), "r"(value.x), "r"(value.y), "r"(value.z), "r"(value.w));
}

//==============================================================================
// Atomic Operations
//==============================================================================

__device__ __forceinline__ int atomic_add_release_sys_global(const int* ptr, int value) {
    int ret;
    asm volatile("atom.add.release.sys.global.s32 %0, [%1], %2;" : "=r"(ret) : "l"(ptr), "r"(value));
    return ret;
}

__device__ __forceinline__ int atomic_add_release_global(const int* ptr, int value) {
    int ret;
    asm volatile("atom.add.release.gpu.global.s32 %0, [%1], %2;" : "=r"(ret) : "l"(ptr), "r"(value));
    return ret;
}

__device__ __forceinline__ int atomic_add_acqrel_global(const int* ptr, int value) {
    int ret;
    asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;" : "=r"(ret) : "l"(ptr), "r"(value));
    return ret;
}

// Fire-and-forget release-add at system scope (no return value).
__device__ __forceinline__ void red_add_release_sys_global(uint32_t* ptr, uint32_t value) {
    asm volatile("red.release.sys.global.add.u32 [%0], %1;"
                 :
                 : "l"(__cvta_generic_to_global(ptr)), "r"(value)
                 : "memory");
}

__forceinline__ __device__ int atomic_cas_cta_acquire(int* addr, int x, int y) {
    int ret;
    asm volatile("atom.acquire.cta.shared::cta.cas.b32 %0, [%1], %2, %3;" : "=r"(ret) : "l"(addr), "r"(x), "r"(y) : "memory");
    return ret;
}

__forceinline__ __device__ int atomic_exch_cta_release(int* addr, int x) {
    int ret;
    asm volatile("atom.release.cta.shared::cta.exch.b32 %0, [%1], %2;" : "=r"(ret) : "l"(addr), "r"(x) : "memory");
    return ret;
}

//==============================================================================
// Data Packing and Broadcasting
//==============================================================================

template <typename dtype_a_t, typename dtype_b_t>
__device__ __forceinline__ dtype_b_t pack2(const dtype_a_t& x, const dtype_a_t& y) {
    EP_STATIC_ASSERT(sizeof(dtype_a_t) * 2 == sizeof(dtype_b_t), "Invalid dtypes");
    dtype_b_t packed;
    auto unpacked_ptr = reinterpret_cast<dtype_a_t*>(&packed);
    unpacked_ptr[0] = x, unpacked_ptr[1] = y;
    return packed;
}

template <typename dtype_a_t, typename dtype_b_t>
__device__ __forceinline__ void unpack2(const dtype_b_t& packed, dtype_a_t& x, dtype_a_t& y) {
    EP_STATIC_ASSERT(sizeof(dtype_a_t) * 2 == sizeof(dtype_b_t), "Invalid dtypes");
    auto unpacked_ptr = reinterpret_cast<const dtype_a_t*>(&packed);
    x = unpacked_ptr[0], y = unpacked_ptr[1];
}

template <typename dtype_t>
__device__ __forceinline__ dtype_t broadcast(dtype_t& ptr, int src_lane_idx) {
    EP_STATIC_ASSERT(sizeof(dtype_t) % sizeof(int) == 0, "");
    auto send_int_values = reinterpret_cast<int*>(&ptr);
    int recv_int_values[sizeof(dtype_t) / sizeof(int)];
    #pragma unroll
    for (int i = 0; i < sizeof(dtype_t) / sizeof(int); ++ i)
        recv_int_values[i] = __shfl_sync(0xffffffff, send_int_values[i], src_lane_idx);
    return *reinterpret_cast<dtype_t*>(recv_int_values);
}

//==============================================================================
// Warp-Level Reduction Operations
//==============================================================================

template <typename T> struct ReduceSum { __device__ T operator()(T a, T b) const { return a + b; } };
template <typename T> struct ReduceMax { __device__ T operator()(T a, T b) const { return a > b ? a : b; } };
template <typename T> struct ReduceMin { __device__ T operator()(T a, T b) const { return a < b ? a : b; } };
template <typename T> struct ReduceAnd { __device__ T operator()(T a, T b) const { return a & b; } };
template <typename T> struct ReduceOr  { __device__ T operator()(T a, T b) const { return a | b; } };

template <int kNumLanesPerGroup, bool kIntergroupReduce, typename T, typename Op>
__forceinline__ __device__ T warp_reduce(T value, Op op) {
    EP_STATIC_ASSERT(kNumLanesPerGroup == 32 or kNumLanesPerGroup == 16 or kNumLanesPerGroup == 8 or
                     kNumLanesPerGroup ==  4 or kNumLanesPerGroup == 2  or kNumLanesPerGroup == 1,
                     "Invalid number of lanes");
    constexpr uint32_t mask = 0xffffffff;
    if constexpr (kIntergroupReduce) {
        if constexpr (kNumLanesPerGroup <=  1) value = op(value, __shfl_xor_sync(mask, value,  1));
        if constexpr (kNumLanesPerGroup <=  2) value = op(value, __shfl_xor_sync(mask, value,  2));
        if constexpr (kNumLanesPerGroup <=  4) value = op(value, __shfl_xor_sync(mask, value,  4));
        if constexpr (kNumLanesPerGroup <=  8) value = op(value, __shfl_xor_sync(mask, value,  8));
        if constexpr (kNumLanesPerGroup <= 16) value = op(value, __shfl_xor_sync(mask, value, 16));
    } else {
        if constexpr (kNumLanesPerGroup >= 32) value = op(value, __shfl_xor_sync(mask, value, 16));
        if constexpr (kNumLanesPerGroup >= 16) value = op(value, __shfl_xor_sync(mask, value,  8));
        if constexpr (kNumLanesPerGroup >=  8) value = op(value, __shfl_xor_sync(mask, value,  4));
        if constexpr (kNumLanesPerGroup >=  4) value = op(value, __shfl_xor_sync(mask, value,  2));
        if constexpr (kNumLanesPerGroup >=  2) value = op(value, __shfl_xor_sync(mask, value,  1));
    }
    return value;
}

template <int kNumLanesPerGroup = 32, bool kIntergroupReduce = false, typename T>
__forceinline__ __device__ T warp_reduce_sum(T value) {
    return warp_reduce<kNumLanesPerGroup, kIntergroupReduce, T>(value, ReduceSum<T>{});
}

template <int kNumLanesPerGroup = 32, bool kIntergroupReduce = false, typename T>
__forceinline__ __device__ T warp_reduce_max(T value) {
    return warp_reduce<kNumLanesPerGroup, kIntergroupReduce, T>(value, ReduceMax<T>{});
}

template <int kNumLanesPerGroup = 32, bool kIntergroupReduce = false, typename T>
__forceinline__ __device__ T warp_reduce_min(T value) {
    return warp_reduce<kNumLanesPerGroup, kIntergroupReduce, T>(value, ReduceMin<T>{});
}

template <int kNumLanesPerGroup = 32, bool kIntergroupReduce = false, typename T>
__forceinline__ __device__ T warp_reduce_and(T value) {
    return warp_reduce<kNumLanesPerGroup, kIntergroupReduce, T>(value, ReduceAnd<T>{});
}

template <int kNumLanesPerGroup = 32, bool kIntergroupReduce = false, typename T>
__forceinline__ __device__ T warp_reduce_or(T value) {
    return warp_reduce<kNumLanesPerGroup, kIntergroupReduce, T>(value, ReduceOr<T>{});
}

//==============================================================================
// Block-Level Synchronization
//==============================================================================

__forceinline__ __device__ void acquire_lock(int* mutex) {
    // To make later memory operations valid, we must use `acquire` for memory semantics
    while (atomic_cas_cta_acquire(mutex, 0, 1) != 0);
}

__forceinline__ __device__ void release_lock(int* mutex) {
    // To make previous memory operations visible to other threads, we must use `release` for memory semantics
    atomic_exch_cta_release(mutex, 0);
}

template <int kNumRanks, bool kSyncOnly = false>
__forceinline__ __device__ void
barrier_block(int** barrier_signal_ptrs, int rank) {
    auto thread_id = static_cast<int>(threadIdx.x);

    // For non-sync-only cases, the memory operations by other threads in the block must be visible to the `sys` scope
    if constexpr (not kSyncOnly) {
        memory_fence();
        __syncthreads();
    }

    // Add self-ranks, sub other ranks
    if (thread_id < kNumRanks) {
        atomicAdd_system(barrier_signal_ptrs[rank] + thread_id, FINISHED_SUM_TAG);
        atomicSub_system(barrier_signal_ptrs[thread_id] + rank, FINISHED_SUM_TAG);
    }
    EP_DEVICE_ASSERT(kNumRanks <= blockDim.x);

    // Check timeout
    auto start_time = clock64();
    while (true) {
        auto value = thread_id < kNumRanks ? ld_volatile_global(barrier_signal_ptrs[rank] + thread_id) : 0;
        if (__all_sync(0xffffffff, value <= 0))
            break;

        if (clock64() - start_time > NUM_TIMEOUT_CYCLES and thread_id < kNumRanks) {
            printf("NCCL EP timeout check failed: rank = %d, thread = %d, value = %d)\n", rank, thread_id, value);
            trap();
        }
    }
    __syncthreads();
}

//==============================================================================
// Task Distribution
//==============================================================================

__forceinline__ __device__ void get_channel_task_range(int num_tokens, int num_sms, int sm_id,
                                                       int& token_start_idx, int& token_end_idx) {
    int num_tokens_per_sm = ceil_div(num_tokens, num_sms);
    token_start_idx = min(num_tokens_per_sm * sm_id, num_tokens);
    token_end_idx = min(token_start_idx + num_tokens_per_sm, num_tokens);
}

//==============================================================================
// SM90-Specific Features (TMA, Mbarriers)
//==============================================================================

#ifndef DISABLE_SM90_FEATURES

__device__ __forceinline__ uint32_t elect_one_sync() {
    uint32_t pred = 0;
    asm volatile(
        "{\n"
        ".reg .b32 %%rx;\n"
        ".reg .pred %%px;\n"
        "      elect.sync %%rx|%%px, %1;\n"
        "@%%px mov.s32 %0, 1;\n"
        "}\n"
        : "+r"(pred)
        : "r"(0xffffffff));
    return pred;
}

// Fence operations for async proxy
__device__ __forceinline__ void fence_view_async_shared() {
    asm volatile("fence.proxy.async.shared::cta; \n" :: );
}

__device__ __forceinline__ void fence_barrier_init() {
    asm volatile("fence.mbarrier_init.release.cluster; \n" :: );
}

// Mbarrier operations
__device__ __forceinline__ void mbarrier_init(uint64_t* mbar_ptr, uint32_t arrive_count) {
    auto mbar_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar_ptr));
    asm volatile("mbarrier.init.shared::cta.b64 [%1], %0;" :: "r"(arrive_count), "r"(mbar_int_ptr));
}

__device__ __forceinline__ void mbarrier_inval(uint64_t* mbar_ptr) {
    auto mbar_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar_ptr));
    asm volatile("mbarrier.inval.shared::cta.b64 [%0];" :: "r"(mbar_int_ptr));
}

template <bool kWithMultiStages = false>
__device__ __forceinline__ void mbarrier_wait(uint64_t* mbar_ptr, uint32_t& phase, int stage_idx = 0) {
    auto mbar_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar_ptr));
    const auto& wait = kWithMultiStages ? (phase >> stage_idx) & 1 : phase;
    asm volatile("{\n\t"
                 ".reg .pred       P1; \n\t"
                 "LAB_WAIT: \n\t"
                 "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1, %2; \n\t"
                 "@P1 bra DONE; \n\t"
                 "bra     LAB_WAIT; \n\t"
                 "DONE: \n\t"
                 "}" :: "r"(mbar_int_ptr), "r"(wait), "r"(0x989680));
    phase ^= kWithMultiStages ? (1 << stage_idx) : 1;
}

__device__ __forceinline__ void mbarrier_arrive_and_expect_tx(uint64_t* mbar_ptr, int num_bytes) {
    auto mbar_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar_ptr));
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%1], %0; \n\t" :: "r"(num_bytes), "r"(mbar_int_ptr));
}

__device__ __forceinline__ void mbarrier_arrive(uint64_t* mbar_ptr) {
    auto mbar_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar_ptr));
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0]; \n\t" :: "r"(mbar_int_ptr));
}

// TMA operations
constexpr uint64_t kEvictFirst = 0x12f0000000000000;
constexpr uint64_t kEvictNormal = 0x1000000000000000;

__device__ __forceinline__ void tma_store_fence() {
    asm volatile ("fence.proxy.async.shared::cta;");
}

__device__ __forceinline__ void tma_load_1d(const void* smem_ptr, const void* gmem_ptr, uint64_t* mbar_ptr, int num_bytes,
                                            bool evict_first = true) {
    auto mbar_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar_ptr));
    auto smem_int_ptr  = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    const auto cache_hint = evict_first ? kEvictFirst : kEvictNormal;
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;\n"
                 :: "r"(smem_int_ptr), "l"(gmem_ptr), "r"(num_bytes), "r"(mbar_int_ptr), "l"(cache_hint) : "memory");
}

__device__ __forceinline__ void tma_store_1d(const void* smem_ptr, const void* gmem_ptr, int num_bytes,
                                             bool evict_first = true) {
    auto smem_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    const auto cache_hint = evict_first ? kEvictFirst : kEvictNormal;
    asm volatile("cp.async.bulk.global.shared::cta.bulk_group.L2::cache_hint [%0], [%1], %2, %3;\n"
                 :: "l"(gmem_ptr), "r"(smem_int_ptr), "r"(num_bytes), "l"(cache_hint) : "memory");
    asm volatile("cp.async.bulk.commit_group;");
}

template <int N = 0>
__device__ __forceinline__ void tma_store_wait() {
    asm volatile("cp.async.bulk.wait_group.read %0;" :: "n"(N) : "memory");
}

#endif // DISABLE_SM90_FEATURES

} // namespace nccl_ep
