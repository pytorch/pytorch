// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once
#include "data_type.hpp"

namespace ck {

template <typename T>
union BufferResource
{
    __device__ constexpr BufferResource() : content{} {}

    // 128 bit SGPRs to supply buffer resource in buffer instructions
    // https://rocm-documentation.readthedocs.io/en/latest/GCN_ISA_Manuals/testdocbook.html#vector-memory-buffer-instructions
    int32x4_t content;
    StaticallyIndexedArray<T*, 2> address;
    StaticallyIndexedArray<int32_t, 4> range;
    StaticallyIndexedArray<int32_t, 4> config;
};

template <typename T>
__device__ int32x4_t make_wave_buffer_resource(T* p_wave, index_t element_space_size)
{
    BufferResource<T> wave_buffer_resource;

    // wavewise base address (64 bit)
    wave_buffer_resource.address(Number<0>{}) = const_cast<remove_cv_t<T>*>(p_wave);
    // wavewise range (32 bit)
    wave_buffer_resource.range(Number<2>{}) = element_space_size * sizeof(T);
    // wavewise setting (32 bit)
    wave_buffer_resource.config(Number<3>{}) = CK_BUFFER_RESOURCE_3RD_DWORD;

    return wave_buffer_resource.content;
}

template <typename T>
__device__ int32x4_t make_wave_buffer_resource_with_default_range(T* p_wave)
{
    BufferResource<T> wave_buffer_resource;

    // wavewise base address (64 bit)
    wave_buffer_resource.address(Number<0>{}) = const_cast<remove_cv_t<T>*>(p_wave);
    // wavewise range (32 bit)
    wave_buffer_resource.range(Number<2>{}) = 0xffffffff; // max possible range
    // wavewise setting (32 bit)
    wave_buffer_resource.config(Number<3>{}) = CK_BUFFER_RESOURCE_3RD_DWORD;

    return wave_buffer_resource.content;
}

// buffer load i8
__device__ int8_t
llvm_amdgcn_raw_buffer_load_i8(int32x4_t srsrc,
                               index_t voffset,
                               index_t soffset,
                               index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.load.i8");

__device__ int8x2_t
llvm_amdgcn_raw_buffer_load_i8x2(int32x4_t srsrc,
                                 index_t voffset,
                                 index_t soffset,
                                 index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.load.v2i8");

__device__ int8x4_t
llvm_amdgcn_raw_buffer_load_i8x4(int32x4_t srsrc,
                                 index_t voffset,
                                 index_t soffset,
                                 index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.load.v4i8");

// buffer load i16
__device__ bhalf_t
llvm_amdgcn_raw_buffer_load_i16(int32x4_t srsrc,
                                index_t voffset,
                                index_t soffset,
                                index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.load.i16");

__device__ bhalf2_t
llvm_amdgcn_raw_buffer_load_i16x2(int32x4_t srsrc,
                                  index_t voffset,
                                  index_t soffset,
                                  index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.load.v2i16");

__device__ bhalf4_t
llvm_amdgcn_raw_buffer_load_i16x4(int32x4_t srsrc,
                                  index_t voffset,
                                  index_t soffset,
                                  index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.load.v4i16");

// buffer load i32
__device__ int32_t
llvm_amdgcn_raw_buffer_load_i32(int32x4_t srsrc,
                                index_t voffset,
                                index_t soffset,
                                index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.load.i32");

__device__ int32x2_t
llvm_amdgcn_raw_buffer_load_i32x2(int32x4_t srsrc,
                                  index_t voffset,
                                  index_t soffset,
                                  index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.load.v2i32");

__device__ int32x4_t
llvm_amdgcn_raw_buffer_load_i32x4(int32x4_t srsrc,
                                  index_t voffset,
                                  index_t soffset,
                                  index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.load.v4i32");

// buffer load fp16
__device__ half_t
llvm_amdgcn_raw_buffer_load_fp16(int32x4_t srsrc,
                                 index_t voffset,
                                 index_t soffset,
                                 index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.load.f16");

__device__ half2_t
llvm_amdgcn_raw_buffer_load_fp16x2(int32x4_t srsrc,
                                   index_t voffset,
                                   index_t soffset,
                                   index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.load.v2f16");

__device__ half4_t
llvm_amdgcn_raw_buffer_load_fp16x4(int32x4_t srsrc,
                                   index_t voffset,
                                   index_t soffset,
                                   index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.load.v4f16");

// buffer load fp32
__device__ float
llvm_amdgcn_raw_buffer_load_fp32(int32x4_t srsrc,
                                 index_t voffset,
                                 index_t soffset,
                                 index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.load.f32");

__device__ float2_t
llvm_amdgcn_raw_buffer_load_fp32x2(int32x4_t srsrc,
                                   index_t voffset,
                                   index_t soffset,
                                   index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.load.v2f32");

__device__ float4_t
llvm_amdgcn_raw_buffer_load_fp32x4(int32x4_t srsrc,
                                   index_t voffset,
                                   index_t soffset,
                                   index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.load.v4f32");

// buffer store i8
__device__ void
llvm_amdgcn_raw_buffer_store_i8(int8_t vdata,
                                int32x4_t rsrc,
                                index_t voffset,
                                index_t soffset,
                                index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.store.i8");

__device__ void
llvm_amdgcn_raw_buffer_store_i8x2(int8x2_t vdata,
                                  int32x4_t rsrc,
                                  index_t voffset,
                                  index_t soffset,
                                  index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.store.v2i8");

__device__ void
llvm_amdgcn_raw_buffer_store_i8x4(int8x4_t vdata,
                                  int32x4_t rsrc,
                                  index_t voffset,
                                  index_t soffset,
                                  index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.store.v4i8");

// buffer store i16
__device__ void
llvm_amdgcn_raw_buffer_store_i16(bhalf_t vdata,
                                 int32x4_t rsrc,
                                 index_t voffset,
                                 index_t soffset,
                                 index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.store.i16");

__device__ void
llvm_amdgcn_raw_buffer_store_i16x2(bhalf2_t vdata,
                                   int32x4_t rsrc,
                                   index_t voffset,
                                   index_t soffset,
                                   index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.store.v2i16");

__device__ void
llvm_amdgcn_raw_buffer_store_i16x4(bhalf4_t vdata,
                                   int32x4_t rsrc,
                                   index_t voffset,
                                   index_t soffset,
                                   index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.store.v4i16");

// buffer store i32
__device__ void
llvm_amdgcn_raw_buffer_store_i32(int32_t vdata,
                                 int32x4_t rsrc,
                                 index_t voffset,
                                 index_t soffset,
                                 index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.store.i32");

__device__ void
llvm_amdgcn_raw_buffer_store_i32x2(int32x2_t vdata,
                                   int32x4_t rsrc,
                                   index_t voffset,
                                   index_t soffset,
                                   index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.store.v2i32");

__device__ void
llvm_amdgcn_raw_buffer_store_i32x4(int32x4_t vdata,
                                   int32x4_t rsrc,
                                   index_t voffset,
                                   index_t soffset,
                                   index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.store.v4i32");

// buffer store fp16
__device__ void
llvm_amdgcn_raw_buffer_store_fp16(half_t vdata,
                                  int32x4_t rsrc,
                                  index_t voffset,
                                  index_t soffset,
                                  index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.store.f16");

__device__ void
llvm_amdgcn_raw_buffer_store_fp16x2(half2_t vdata,
                                    int32x4_t rsrc,
                                    index_t voffset,
                                    index_t soffset,
                                    index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.store.v2f16");

__device__ void
llvm_amdgcn_raw_buffer_store_fp16x4(half4_t vdata,
                                    int32x4_t rsrc,
                                    index_t voffset,
                                    index_t soffset,
                                    index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.store.v4f16");

// buffer store fp32
__device__ void
llvm_amdgcn_raw_buffer_store_fp32(float vdata,
                                  int32x4_t rsrc,
                                  index_t voffset,
                                  index_t soffset,
                                  index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.store.f32");

__device__ void
llvm_amdgcn_raw_buffer_store_fp32x2(float2_t vdata,
                                    int32x4_t rsrc,
                                    index_t voffset,
                                    index_t soffset,
                                    index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.store.v2f32");

__device__ void
llvm_amdgcn_raw_buffer_store_fp32x4(float4_t vdata,
                                    int32x4_t rsrc,
                                    index_t voffset,
                                    index_t soffset,
                                    index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.store.v4f32");

// buffer atomic-add fp16
__device__ half2_t llvm_amdgcn_raw_buffer_atomic_add_fp16x2(
    half2_t vdata,
    int32x4_t rsrc,
    index_t voffset,
    index_t soffset,
    index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.atomic.fadd.v2f16");

// buffer atomic-add i32
__device__ int32_t llvm_amdgcn_raw_buffer_atomic_add_i32(
    int32_t vdata,
    int32x4_t rsrc,
    index_t voffset,
    index_t soffset,
    index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.atomic.add.i32");

// buffer atomic-add fp32
__device__ float llvm_amdgcn_raw_buffer_atomic_add_fp32(
    float vdata,
    int32x4_t rsrc,
    index_t voffset,
    index_t soffset,
    index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.atomic.fadd.f32");

// buffer atomic-add fp32
__device__ double
llvm_amdgcn_raw_buffer_atomic_max_fp64(double vdata,
                                       int32x4_t rsrc, // dst_wave_buffer_resource
                                       int voffset,    // dst_thread_addr_offset
                                       int soffset,    // dst_wave_addr_offset
                                       int glc_slc) __asm("llvm.amdgcn.raw.buffer.atomic.fmax.f64");

template <typename T, index_t N>
__device__ typename vector_type<T, N>::type amd_buffer_load_impl(int32x4_t src_wave_buffer_resource,
                                                                 index_t src_thread_addr_offset,
                                                                 index_t src_wave_addr_offset)
{
    static_assert(
        (is_same<T, double>::value && (N == 1 || N == 2 || N == 4)) ||
            (is_same<T, float>::value && (N == 1 || N == 2 || N == 4 || N == 8)) ||
            (is_same<T, half_t>::value && (N == 1 || N == 2 || N == 4 || N == 8)) ||
            (is_same<T, bhalf_t>::value && (N == 1 || N == 2 || N == 4 || N == 8)) ||
            (is_same<T, int32_t>::value && (N == 1 || N == 2 || N == 4 || N == 8)) ||
            (is_same<T, int8_t>::value && (N == 1 || N == 2 || N == 4 || N == 8 || N == 16)),
        "wrong! not implemented");

    if constexpr(is_same<T, double>::value)
    {
        // use fp32 load to mimic fp64 load
        if constexpr(N == 1)
        {
            const float2_t tmp = llvm_amdgcn_raw_buffer_load_fp32x2(
                src_wave_buffer_resource, src_thread_addr_offset, src_wave_addr_offset, 0);

            return bit_cast<double>(tmp);
        }
        else if constexpr(N == 2)
        {
            const float4_t tmp = llvm_amdgcn_raw_buffer_load_fp32x4(
                src_wave_buffer_resource, src_thread_addr_offset, src_wave_addr_offset, 0);

            return bit_cast<double2_t>(tmp);
        }
        else if constexpr(N == 4)
        {
            const float4_t f32_0 = llvm_amdgcn_raw_buffer_load_fp32x4(
                src_wave_buffer_resource, src_thread_addr_offset, src_wave_addr_offset, 0);

            const float4_t f32_1 =
                llvm_amdgcn_raw_buffer_load_fp32x4(src_wave_buffer_resource,
                                                   src_thread_addr_offset,
                                                   src_wave_addr_offset + 4 * sizeof(float),
                                                   0);
            vector_type<double, 4> tmp;

            tmp.AsType<double2_t>()(Number<0>{}) = bit_cast<double2_t>(f32_0);
            tmp.AsType<double2_t>()(Number<1>{}) = bit_cast<double2_t>(f32_1);

            return tmp.AsType<double4_t>()(Number<0>{});
        }
    }
    else if constexpr(is_same<T, float>::value)
    {
        if constexpr(N == 1)
        {
            return llvm_amdgcn_raw_buffer_load_fp32(
                src_wave_buffer_resource, src_thread_addr_offset, src_wave_addr_offset, 0);
        }
        else if constexpr(N == 2)
        {
            return llvm_amdgcn_raw_buffer_load_fp32x2(
                src_wave_buffer_resource, src_thread_addr_offset, src_wave_addr_offset, 0);
        }
        else if constexpr(N == 4)
        {
            return llvm_amdgcn_raw_buffer_load_fp32x4(
                src_wave_buffer_resource, src_thread_addr_offset, src_wave_addr_offset, 0);
        }
        else if constexpr(N == 8)
        {
            vector_type<float, 8> tmp;

            tmp.AsType<float4_t>()(Number<0>{}) = llvm_amdgcn_raw_buffer_load_fp32x4(
                src_wave_buffer_resource, src_thread_addr_offset, src_wave_addr_offset, 0);

            tmp.AsType<float4_t>()(Number<1>{}) =
                llvm_amdgcn_raw_buffer_load_fp32x4(src_wave_buffer_resource,
                                                   src_thread_addr_offset,
                                                   src_wave_addr_offset + 4 * sizeof(float),
                                                   0);

            return tmp.AsType<float8_t>()(Number<0>{});
        }
    }
    else if constexpr(is_same<T, half_t>::value)
    {
        if constexpr(N == 1)
        {
            return llvm_amdgcn_raw_buffer_load_fp16(
                src_wave_buffer_resource, src_thread_addr_offset, src_wave_addr_offset, 0);
        }
        else if constexpr(N == 2)
        {
            return llvm_amdgcn_raw_buffer_load_fp16x2(
                src_wave_buffer_resource, src_thread_addr_offset, src_wave_addr_offset, 0);
        }
        else if constexpr(N == 4)
        {
            return llvm_amdgcn_raw_buffer_load_fp16x4(
                src_wave_buffer_resource, src_thread_addr_offset, src_wave_addr_offset, 0);
        }
        else if constexpr(N == 8)
        {
            // use fp32 load to mimic fp16 load
            float4_t tmp = llvm_amdgcn_raw_buffer_load_fp32x4(
                src_wave_buffer_resource, src_thread_addr_offset, src_wave_addr_offset, 0);

            return bit_cast<half8_t>(tmp);
        }
    }
    else if constexpr(is_same<T, bhalf_t>::value)
    {
        if constexpr(N == 1)
        {
            return llvm_amdgcn_raw_buffer_load_i16(
                src_wave_buffer_resource, src_thread_addr_offset, src_wave_addr_offset, 0);
        }
        else if constexpr(N == 2)
        {
            return llvm_amdgcn_raw_buffer_load_i16x2(
                src_wave_buffer_resource, src_thread_addr_offset, src_wave_addr_offset, 0);
        }
        else if constexpr(N == 4)
        {
            return llvm_amdgcn_raw_buffer_load_i16x4(
                src_wave_buffer_resource, src_thread_addr_offset, src_wave_addr_offset, 0);
        }
        else if constexpr(N == 8)
        {
            int32x4_t tmp = llvm_amdgcn_raw_buffer_load_i32x4(
                src_wave_buffer_resource, src_thread_addr_offset, src_wave_addr_offset, 0);

            return bit_cast<bhalf8_t>(tmp);
        }
    }
    else if constexpr(is_same<T, int32_t>::value)
    {
        if constexpr(N == 1)
        {
            return llvm_amdgcn_raw_buffer_load_i32(
                src_wave_buffer_resource, src_thread_addr_offset, src_wave_addr_offset, 0);
        }
        else if constexpr(N == 2)
        {
            return llvm_amdgcn_raw_buffer_load_i32x2(
                src_wave_buffer_resource, src_thread_addr_offset, src_wave_addr_offset, 0);
        }
        else if constexpr(N == 4)
        {
            return llvm_amdgcn_raw_buffer_load_i32x4(
                src_wave_buffer_resource, src_thread_addr_offset, src_wave_addr_offset, 0);
        }
        else if constexpr(N == 8)
        {
            vector_type<int32_t, 8> tmp;

            tmp.AsType<int32x4_t>()(Number<0>{}) = llvm_amdgcn_raw_buffer_load_i32x4(
                src_wave_buffer_resource, src_thread_addr_offset, src_wave_addr_offset, 0);

            tmp.AsType<int32x4_t>()(Number<1>{}) =
                llvm_amdgcn_raw_buffer_load_i32x4(src_wave_buffer_resource,
                                                  src_thread_addr_offset,
                                                  src_wave_addr_offset + 4 * sizeof(int32_t),
                                                  0);
            return tmp.AsType<int32x8_t>()(Number<0>{});
        }
    }
    else if constexpr(is_same<T, int8_t>::value)
    {
        if constexpr(N == 1)
        {
            return llvm_amdgcn_raw_buffer_load_i8(
                src_wave_buffer_resource, src_thread_addr_offset, src_wave_addr_offset, 0);
        }
        else if constexpr(N == 2)
        {
#if !CK_WORKAROUND_SWDEV_XXXXXX_INT8_BUFFER_LOAD_STORE_ISSUE
            return llvm_amdgcn_raw_buffer_load_i8x2(
                src_wave_buffer_resource, src_thread_addr_offset, src_wave_addr_offset, 0);
#else
            int16_t tmp = llvm_amdgcn_raw_buffer_load_i16(
                src_wave_buffer_resource, src_thread_addr_offset, src_wave_addr_offset, 0);

            return bit_cast<int8x2_t>(tmp);
#endif
        }
        else if constexpr(N == 4)
        {
#if !CK_WORKAROUND_SWDEV_XXXXXX_INT8_BUFFER_LOAD_STORE_ISSUE
            return llvm_amdgcn_raw_buffer_load_i8x4(
                src_wave_buffer_resource, src_thread_addr_offset, src_wave_addr_offset, 0);
#else
            int32_t tmp = llvm_amdgcn_raw_buffer_load_i32(
                src_wave_buffer_resource, src_thread_addr_offset, src_wave_addr_offset, 0);

            return bit_cast<int8x4_t>(tmp);
#endif
        }
        else if constexpr(N == 8)
        {
#if !CK_WORKAROUND_SWDEV_XXXXXX_INT8_BUFFER_LOAD_STORE_ISSUE
            vector_type<int8_t, 8> tmp;

            tmp.AsType<int8x4_t>()(Number<0>{}) = llvm_amdgcn_raw_buffer_load_i8x4(
                src_wave_buffer_resource, src_thread_addr_offset, src_wave_addr_offset, 0);

            tmp.AsType<int8x4_t>()(Number<1>{}) =
                llvm_amdgcn_raw_buffer_load_i8x4(src_wave_buffer_resource,
                                                 src_thread_addr_offset,
                                                 src_wave_addr_offset + 4 * sizeof(int8_t),
                                                 0);

            return tmp.AsType<int8x8_t>()(Number<0>{});
#else
            int32x2_t tmp = llvm_amdgcn_raw_buffer_load_i32x2(
                src_wave_buffer_resource, src_thread_addr_offset, src_wave_addr_offset, 0);

            return bit_cast<int8x8_t>(tmp);
#endif
        }
        else if constexpr(N == 16)
        {
#if !CK_WORKAROUND_SWDEV_XXXXXX_INT8_BUFFER_LOAD_STORE_ISSUE
            vector_type<int8_t, 16> tmp;

            tmp.AsType<int8x4_t>()(Number<0>{}) = llvm_amdgcn_raw_buffer_load_i8x4(
                src_wave_buffer_resource, src_thread_addr_offset, src_wave_addr_offset, 0);

            tmp.AsType<int8x4_t>()(Number<1>{}) =
                llvm_amdgcn_raw_buffer_load_i8x4(src_wave_buffer_resource,
                                                 src_thread_addr_offset,
                                                 src_wave_addr_offset + 4 * sizeof(int8_t),
                                                 0);

            tmp.AsType<int8x4_t>()(Number<2>{}) =
                llvm_amdgcn_raw_buffer_load_i8x4(src_wave_buffer_resource,
                                                 src_thread_addr_offset,
                                                 src_wave_addr_offset + 8 * sizeof(int8_t),
                                                 0);

            tmp.AsType<int8x4_t>()(Number<3>{}) =
                llvm_amdgcn_raw_buffer_load_i8x4(src_wave_buffer_resource,
                                                 src_thread_addr_offset,
                                                 src_wave_addr_offset + 12 * sizeof(int8_t),
                                                 0);

            return tmp.AsType<int8x16_t>()(Number<0>{});
#else
            int32x4_t tmp = llvm_amdgcn_raw_buffer_load_i32x4(
                src_wave_buffer_resource, src_thread_addr_offset, src_wave_addr_offset, 0);

            return bit_cast<int8x16_t>(tmp);
#endif
        }
    }
}

template <typename T, index_t N>
__device__ void amd_buffer_store_impl(const typename vector_type<T, N>::type src_thread_data,
                                      int32x4_t dst_wave_buffer_resource,
                                      index_t dst_thread_addr_offset,
                                      index_t dst_wave_addr_offset)
{
    static_assert(
        (is_same<T, double>::value && (N == 1 || N == 2)) ||
            (is_same<T, float>::value && (N == 1 || N == 2 || N == 4)) ||
            (is_same<T, half_t>::value && (N == 1 || N == 2 || N == 4 || N == 8)) ||
            (is_same<T, bhalf_t>::value && (N == 1 || N == 2 || N == 4 || N == 8)) ||
            (is_same<T, int32_t>::value && (N == 1 || N == 2 || N == 4)) ||
            (is_same<T, int8_t>::value && (N == 1 || N == 2 || N == 4 || N == 8 || N == 16)),
        "wrong! not implemented");

    if constexpr(is_same<T, double>::value)
    {
        // use fp32 store to mimic fp64 store
        if constexpr(N == 1)
        {
            llvm_amdgcn_raw_buffer_store_fp32x2(bit_cast<float2_t>(src_thread_data),
                                                dst_wave_buffer_resource,
                                                dst_thread_addr_offset,
                                                dst_wave_addr_offset,
                                                0);
        }
        else if constexpr(N == 2)
        {
            llvm_amdgcn_raw_buffer_store_fp32x4(bit_cast<float4_t>(src_thread_data),
                                                dst_wave_buffer_resource,
                                                dst_thread_addr_offset,
                                                dst_wave_addr_offset,
                                                0);
        }
    }
    else if constexpr(is_same<T, float>::value)
    {
        if constexpr(N == 1)
        {
            llvm_amdgcn_raw_buffer_store_fp32(src_thread_data,
                                              dst_wave_buffer_resource,
                                              dst_thread_addr_offset,
                                              dst_wave_addr_offset,
                                              0);
        }
        else if constexpr(N == 2)
        {
            llvm_amdgcn_raw_buffer_store_fp32x2(src_thread_data,
                                                dst_wave_buffer_resource,
                                                dst_thread_addr_offset,
                                                dst_wave_addr_offset,
                                                0);
        }
        else if constexpr(N == 4)
        {
            llvm_amdgcn_raw_buffer_store_fp32x4(src_thread_data,
                                                dst_wave_buffer_resource,
                                                dst_thread_addr_offset,
                                                dst_wave_addr_offset,
                                                0);
        }
    }
    else if constexpr(is_same<T, half_t>::value)
    {
        if constexpr(N == 1)
        {
            llvm_amdgcn_raw_buffer_store_fp16(src_thread_data,
                                              dst_wave_buffer_resource,
                                              dst_thread_addr_offset,
                                              dst_wave_addr_offset,
                                              0);
        }
        else if constexpr(N == 2)
        {
            llvm_amdgcn_raw_buffer_store_fp16x2(src_thread_data,
                                                dst_wave_buffer_resource,
                                                dst_thread_addr_offset,
                                                dst_wave_addr_offset,
                                                0);
        }
        else if constexpr(N == 4)
        {
            llvm_amdgcn_raw_buffer_store_fp16x4(src_thread_data,
                                                dst_wave_buffer_resource,
                                                dst_thread_addr_offset,
                                                dst_wave_addr_offset,
                                                0);
        }
        else if constexpr(N == 8)
        {
#if 0
            vector_type<half_t, 8> tmp{src_thread_data};

            llvm_amdgcn_raw_buffer_store_fp16x4(tmp.AsType<half4_t>()[Number<0>{}],
                                                dst_wave_buffer_resource,
                                                dst_thread_addr_offset,
                                                dst_wave_addr_offset,
                                                0);

            llvm_amdgcn_raw_buffer_store_fp16x4(tmp.AsType<half4_t>()[Number<1>{}],
                                                dst_wave_buffer_resource,
                                                dst_thread_addr_offset,
                                                dst_wave_addr_offset + 4 * sizeof(half_t),
                                                0);
#else
            llvm_amdgcn_raw_buffer_store_fp32x4(bit_cast<float4_t>(src_thread_data),
                                                dst_wave_buffer_resource,
                                                dst_thread_addr_offset,
                                                dst_wave_addr_offset,
                                                0);
#endif
        }
    }
    else if constexpr(is_same<T, bhalf_t>::value)
    {
        if constexpr(N == 1)
        {
            llvm_amdgcn_raw_buffer_store_i16(src_thread_data,
                                             dst_wave_buffer_resource,
                                             dst_thread_addr_offset,
                                             dst_wave_addr_offset,
                                             0);
        }
        else if constexpr(N == 2)
        {
            llvm_amdgcn_raw_buffer_store_i16x2(src_thread_data,
                                               dst_wave_buffer_resource,
                                               dst_thread_addr_offset,
                                               dst_wave_addr_offset,
                                               0);
        }
        else if constexpr(N == 4)
        {
            llvm_amdgcn_raw_buffer_store_i16x4(src_thread_data,
                                               dst_wave_buffer_resource,
                                               dst_thread_addr_offset,
                                               dst_wave_addr_offset,
                                               0);
        }
        else if constexpr(N == 8)
        {
            vector_type<bhalf_t, 8> tmp{src_thread_data};

            llvm_amdgcn_raw_buffer_store_i16x4(tmp.AsType<bhalf4_t>()[Number<0>{}],
                                               dst_wave_buffer_resource,
                                               dst_thread_addr_offset,
                                               dst_wave_addr_offset,
                                               0);

            llvm_amdgcn_raw_buffer_store_i16x4(tmp.AsType<bhalf4_t>()[Number<1>{}],
                                               dst_wave_buffer_resource,
                                               dst_thread_addr_offset,
                                               dst_wave_addr_offset + 4 * sizeof(bhalf_t),
                                               0);
        }
    }
    else if constexpr(is_same<T, int32_t>::value)
    {
        if constexpr(N == 1)
        {
            llvm_amdgcn_raw_buffer_store_i32(src_thread_data,
                                             dst_wave_buffer_resource,
                                             dst_thread_addr_offset,
                                             dst_wave_addr_offset,
                                             0);
        }
        else if constexpr(N == 2)
        {
            llvm_amdgcn_raw_buffer_store_i32x2(src_thread_data,
                                               dst_wave_buffer_resource,
                                               dst_thread_addr_offset,
                                               dst_wave_addr_offset,
                                               0);
        }
        else if constexpr(N == 4)
        {
            llvm_amdgcn_raw_buffer_store_i32x4(src_thread_data,
                                               dst_wave_buffer_resource,
                                               dst_thread_addr_offset,
                                               dst_wave_addr_offset,
                                               0);
        }
    }
    else if constexpr(is_same<T, int8_t>::value)
    {
        if constexpr(N == 1)
        {
            llvm_amdgcn_raw_buffer_store_i8(src_thread_data,
                                            dst_wave_buffer_resource,
                                            dst_thread_addr_offset,
                                            dst_wave_addr_offset,
                                            0);
        }
        else if constexpr(N == 2)
        {
#if !CK_WORKAROUND_SWDEV_XXXXXX_INT8_BUFFER_LOAD_STORE_ISSUE
            llvm_amdgcn_raw_buffer_store_i8x2(src_thread_data,
                                              dst_wave_buffer_resource,
                                              dst_thread_addr_offset,
                                              dst_wave_addr_offset,
                                              0);
#else
            llvm_amdgcn_raw_buffer_store_i16(bit_cast<int16_t>(src_thread_data),
                                             dst_wave_buffer_resource,
                                             dst_thread_addr_offset,
                                             dst_wave_addr_offset,
                                             0);
#endif
        }
        else if constexpr(N == 4)
        {
#if !CK_WORKAROUND_SWDEV_XXXXXX_INT8_BUFFER_LOAD_STORE_ISSUE
            llvm_amdgcn_raw_buffer_store_i8x4(src_thread_data,
                                              dst_wave_buffer_resource,
                                              dst_thread_addr_offset,
                                              dst_wave_addr_offset,
                                              0);
#else
            llvm_amdgcn_raw_buffer_store_i32(bit_cast<int32_t>(src_thread_data),
                                             dst_wave_buffer_resource,
                                             dst_thread_addr_offset,
                                             dst_wave_addr_offset,
                                             0);
#endif
        }
        else if constexpr(N == 8)
        {
            llvm_amdgcn_raw_buffer_store_i32x2(bit_cast<int32x2_t>(src_thread_data),
                                               dst_wave_buffer_resource,
                                               dst_thread_addr_offset,
                                               dst_wave_addr_offset,
                                               0);
        }
        else if constexpr(N == 16)
        {
            llvm_amdgcn_raw_buffer_store_i32x4(bit_cast<int32x4_t>(src_thread_data),
                                               dst_wave_buffer_resource,
                                               dst_thread_addr_offset,
                                               dst_wave_addr_offset,
                                               0);
        }
    }
}

template <typename T, index_t N>
__device__ void amd_buffer_atomic_add_impl(const typename vector_type<T, N>::type src_thread_data,
                                           int32x4_t dst_wave_buffer_resource,
                                           index_t dst_thread_addr_offset,
                                           index_t dst_wave_addr_offset)
{
    static_assert((is_same<T, float>::value && (N == 1 || N == 2 || N == 4)) ||
                      (is_same<T, half_t>::value && (N == 2 || N == 4 || N == 8)) ||
                      (is_same<T, int32_t>::value && (N == 1 || N == 2 || N == 4)),
                  "wrong! not implemented");

    if constexpr(is_same<T, float>::value)
    {
        if constexpr(N == 1)
        {
            llvm_amdgcn_raw_buffer_atomic_add_fp32(src_thread_data,
                                                   dst_wave_buffer_resource,
                                                   dst_thread_addr_offset,
                                                   dst_wave_addr_offset,
                                                   0);
        }
        else if constexpr(N == 2)
        {
            vector_type<float, 2> tmp{src_thread_data};

            llvm_amdgcn_raw_buffer_atomic_add_fp32(tmp.AsType<float>()[Number<0>{}],
                                                   dst_wave_buffer_resource,
                                                   dst_thread_addr_offset,
                                                   dst_wave_addr_offset,
                                                   0);

            llvm_amdgcn_raw_buffer_atomic_add_fp32(tmp.AsType<float>()[Number<1>{}],
                                                   dst_wave_buffer_resource,
                                                   dst_thread_addr_offset,
                                                   dst_wave_addr_offset + sizeof(float),
                                                   0);
        }
        else if constexpr(N == 4)
        {
            vector_type<float, 4> tmp{src_thread_data};

            llvm_amdgcn_raw_buffer_atomic_add_fp32(tmp.AsType<float>()[Number<0>{}],
                                                   dst_wave_buffer_resource,
                                                   dst_thread_addr_offset,
                                                   dst_wave_addr_offset,
                                                   0);

            llvm_amdgcn_raw_buffer_atomic_add_fp32(tmp.AsType<float>()[Number<1>{}],
                                                   dst_wave_buffer_resource,
                                                   dst_thread_addr_offset,
                                                   dst_wave_addr_offset + sizeof(float),
                                                   0);

            llvm_amdgcn_raw_buffer_atomic_add_fp32(tmp.AsType<float>()[Number<2>{}],
                                                   dst_wave_buffer_resource,
                                                   dst_thread_addr_offset,
                                                   dst_wave_addr_offset + 2 * sizeof(float),
                                                   0);

            llvm_amdgcn_raw_buffer_atomic_add_fp32(tmp.AsType<float>()[Number<3>{}],
                                                   dst_wave_buffer_resource,
                                                   dst_thread_addr_offset,
                                                   dst_wave_addr_offset + 3 * sizeof(float),
                                                   0);
        }
    }
    else if constexpr(is_same<T, half_t>::value)
    {
        if constexpr(N == 2)
        {
            llvm_amdgcn_raw_buffer_atomic_add_fp16x2(src_thread_data,
                                                     dst_wave_buffer_resource,
                                                     dst_thread_addr_offset,
                                                     dst_wave_addr_offset,
                                                     0);
        }
        else if constexpr(N == 4)
        {
            vector_type<half_t, 4> tmp{src_thread_data};

            static_for<0, 2, 1>{}([&](auto i) {
                llvm_amdgcn_raw_buffer_atomic_add_fp16x2(tmp.AsType<half2_t>()[i],
                                                         dst_wave_buffer_resource,
                                                         dst_thread_addr_offset,
                                                         dst_wave_addr_offset + i * sizeof(half2_t),
                                                         0);
            });
        }
        else if constexpr(N == 8)
        {
            vector_type<half_t, 8> tmp{src_thread_data};

            static_for<0, 4, 1>{}([&](auto i) {
                llvm_amdgcn_raw_buffer_atomic_add_fp16x2(tmp.AsType<half2_t>()[i],
                                                         dst_wave_buffer_resource,
                                                         dst_thread_addr_offset,
                                                         dst_wave_addr_offset + i * sizeof(half2_t),
                                                         0);
            });
        }
    }
    else if constexpr(is_same<T, int32_t>::value)
    {
        if constexpr(N == 1)
        {
            llvm_amdgcn_raw_buffer_atomic_add_i32(src_thread_data,
                                                  dst_wave_buffer_resource,
                                                  dst_thread_addr_offset,
                                                  dst_wave_addr_offset,
                                                  0);
        }
        else if constexpr(N == 2)
        {
            vector_type<int32_t, 2> tmp{src_thread_data};

            llvm_amdgcn_raw_buffer_atomic_add_i32(tmp.AsType<int32_t>()[Number<0>{}],
                                                  dst_wave_buffer_resource,
                                                  dst_thread_addr_offset,
                                                  dst_wave_addr_offset,
                                                  0);

            llvm_amdgcn_raw_buffer_atomic_add_i32(tmp.AsType<int32_t>()[Number<1>{}],
                                                  dst_wave_buffer_resource,
                                                  dst_thread_addr_offset,
                                                  dst_wave_addr_offset + sizeof(int32_t),
                                                  0);
        }
        else if constexpr(N == 4)
        {
            vector_type<int32_t, 4> tmp{src_thread_data};

            llvm_amdgcn_raw_buffer_atomic_add_i32(tmp.AsType<int32_t>()[Number<0>{}],
                                                  dst_wave_buffer_resource,
                                                  dst_thread_addr_offset,
                                                  dst_wave_addr_offset,
                                                  0);

            llvm_amdgcn_raw_buffer_atomic_add_i32(tmp.AsType<int32_t>()[Number<1>{}],
                                                  dst_wave_buffer_resource,
                                                  dst_thread_addr_offset,
                                                  dst_wave_addr_offset + sizeof(int32_t),
                                                  0);

            llvm_amdgcn_raw_buffer_atomic_add_i32(tmp.AsType<int32_t>()[Number<2>{}],
                                                  dst_wave_buffer_resource,
                                                  dst_thread_addr_offset,
                                                  dst_wave_addr_offset + 2 * sizeof(int32_t),
                                                  0);

            llvm_amdgcn_raw_buffer_atomic_add_i32(tmp.AsType<int32_t>()[Number<3>{}],
                                                  dst_wave_buffer_resource,
                                                  dst_thread_addr_offset,
                                                  dst_wave_addr_offset + 3 * sizeof(int32_t),
                                                  0);
        }
    }
}

template <typename T, index_t N>
__device__ void amd_buffer_atomic_max_impl(const typename vector_type<T, N>::type src_thread_data,
                                           int32x4_t dst_wave_buffer_resource,
                                           index_t dst_thread_addr_offset,
                                           index_t dst_wave_addr_offset)
{
    static_assert((is_same<T, double>::value && (N == 1 || N == 2 || N == 4)),
                  "wrong! not implemented");
    if constexpr(is_same<T, double>::value)
    {
        if constexpr(N == 1)
        {
            llvm_amdgcn_raw_buffer_atomic_max_fp64(src_thread_data,
                                                   dst_wave_buffer_resource,
                                                   dst_thread_addr_offset,
                                                   dst_wave_addr_offset,
                                                   0);
        }
        else if constexpr(N == 2)
        {
            vector_type<double, 2> tmp{src_thread_data};

            llvm_amdgcn_raw_buffer_atomic_max_fp64(tmp.AsType<double>()[Number<0>{}],
                                                   dst_wave_buffer_resource,
                                                   dst_thread_addr_offset,
                                                   dst_wave_addr_offset,
                                                   0);

            llvm_amdgcn_raw_buffer_atomic_max_fp64(tmp.AsType<double>()[Number<1>{}],
                                                   dst_wave_buffer_resource,
                                                   dst_thread_addr_offset,
                                                   dst_wave_addr_offset + sizeof(double),
                                                   0);
        }
        else if constexpr(N == 4)
        {
            vector_type<double, 4> tmp{src_thread_data};

            llvm_amdgcn_raw_buffer_atomic_max_fp64(tmp.AsType<double>()[Number<0>{}],
                                                   dst_wave_buffer_resource,
                                                   dst_thread_addr_offset,
                                                   dst_wave_addr_offset,
                                                   0);

            llvm_amdgcn_raw_buffer_atomic_max_fp64(tmp.AsType<double>()[Number<1>{}],
                                                   dst_wave_buffer_resource,
                                                   dst_thread_addr_offset,
                                                   dst_wave_addr_offset + sizeof(double),
                                                   0);

            llvm_amdgcn_raw_buffer_atomic_max_fp64(tmp.AsType<double>()[Number<2>{}],
                                                   dst_wave_buffer_resource,
                                                   dst_thread_addr_offset,
                                                   dst_wave_addr_offset + 2 * sizeof(double),
                                                   0);

            llvm_amdgcn_raw_buffer_atomic_max_fp64(tmp.AsType<double>()[Number<3>{}],
                                                   dst_wave_buffer_resource,
                                                   dst_thread_addr_offset,
                                                   dst_wave_addr_offset + 3 * sizeof(double),
                                                   0);
        }
    }
}

// buffer_load requires:
//   1) p_src_wave must point to global memory space
//   2) p_src_wave must be a wavewise pointer.
// It is user's responsibility to make sure that is true.
template <typename T, index_t N>
__device__ typename vector_type_maker<T, N>::type::type
amd_buffer_load_invalid_element_return_zero(const T* p_src_wave,
                                            index_t src_thread_element_offset,
                                            bool src_thread_element_valid,
                                            index_t src_element_space_size)
{
    const int32x4_t src_wave_buffer_resource =
        make_wave_buffer_resource(p_src_wave, src_element_space_size);

    index_t src_thread_addr_offset = src_thread_element_offset * sizeof(T);

    using vector_t = typename vector_type_maker<T, N>::type::type;
    using scalar_t = typename scalar_type<vector_t>::type;

    constexpr index_t vector_size = scalar_type<vector_t>::vector_size;

#if CK_EXPERIMENTAL_USE_BUFFER_LOAD_OOB_CHECK_OFFSET_TRICK
    uint32_t src_addr_shift = src_thread_element_valid ? 0 : 0x80000000;

    return amd_buffer_load_impl<scalar_t, vector_size>(
        src_wave_buffer_resource, src_addr_shift + src_thread_addr_offset, 0);
#else
    vector_t tmp = amd_buffer_load_impl<scalar_t, vector_size>(
        src_wave_buffer_resource, src_thread_addr_offset, 0);

    return src_thread_element_valid ? tmp : vector_t(0);
#endif
}

// buffer_load requires:
//   1) p_src_wave must point to global memory space
//   2) p_src_wave must be a wavewise pointer.
// It is user's responsibility to make sure that is true.
template <typename T, index_t N>
__device__ typename vector_type_maker<T, N>::type::type
amd_buffer_load_invalid_element_return_customized_value(const T* p_src_wave,
                                                        index_t src_thread_element_offset,
                                                        bool src_thread_element_valid,
                                                        index_t src_element_space_size,
                                                        T customized_value)
{
    const int32x4_t src_wave_buffer_resource =
        make_wave_buffer_resource(p_src_wave, src_element_space_size);

    index_t src_thread_addr_offset = src_thread_element_offset * sizeof(T);

    using vector_t = typename vector_type_maker<T, N>::type::type;
    using scalar_t = typename scalar_type<vector_t>::type;

    constexpr index_t vector_size = scalar_type<vector_t>::vector_size;

    vector_t tmp = amd_buffer_load_impl<scalar_t, vector_size>(
        src_wave_buffer_resource, src_thread_addr_offset, 0);

    return src_thread_element_valid ? tmp : vector_t(customized_value);
}

// buffer_store requires:
//   1) p_dst_wave must point to global memory
//   2) p_dst_wave must be a wavewise pointer.
// It is user's responsibility to make sure that is true.
template <typename T, index_t N>
__device__ void amd_buffer_store(const typename vector_type_maker<T, N>::type::type src_thread_data,
                                 T* p_dst_wave,
                                 const index_t dst_thread_element_offset,
                                 const bool dst_thread_element_valid,
                                 const index_t dst_element_space_size)
{
    const int32x4_t dst_wave_buffer_resource =
        make_wave_buffer_resource(p_dst_wave, dst_element_space_size);

    index_t dst_thread_addr_offset = dst_thread_element_offset * sizeof(T);

    using vector_t                = typename vector_type_maker<T, N>::type::type;
    using scalar_t                = typename scalar_type<vector_t>::type;
    constexpr index_t vector_size = scalar_type<vector_t>::vector_size;

#if CK_EXPERIMENTAL_USE_BUFFER_STORE_OOB_CHECK_OFFSET_TRICK
    uint32_t dst_addr_shift = dst_thread_element_valid ? 0 : 0x80000000;

    amd_buffer_store_impl<scalar_t, vector_size>(
        src_thread_data, dst_wave_buffer_resource, dst_addr_shift + dst_thread_addr_offset, 0);
#else
    if(dst_thread_element_valid)
    {
        amd_buffer_store_impl<scalar_t, vector_size>(
            src_thread_data, dst_wave_buffer_resource, dst_thread_addr_offset, 0);
    }
#endif
}

// buffer_atomic_add requires:
//   1) p_dst_wave must point to global memory
//   2) p_dst_wave must be a wavewise pointer.
// It is user's responsibility to make sure that is true.
template <typename T, index_t N>
__device__ void
amd_buffer_atomic_add(const typename vector_type_maker<T, N>::type::type src_thread_data,
                      T* p_dst_wave,
                      const index_t dst_thread_element_offset,
                      const bool dst_thread_element_valid,
                      const index_t dst_element_space_size)
{
    const int32x4_t dst_wave_buffer_resource =
        make_wave_buffer_resource(p_dst_wave, dst_element_space_size);

    index_t dst_thread_addr_offset = dst_thread_element_offset * sizeof(T);

    using vector_t                = typename vector_type_maker<T, N>::type::type;
    using scalar_t                = typename scalar_type<vector_t>::type;
    constexpr index_t vector_size = scalar_type<vector_t>::vector_size;

#if CK_EXPERIMENTAL_USE_BUFFER_ATOMIC_ADD_OOB_CHECK_OFFSET_TRICK
    uint32_t dst_addr_shift = dst_thread_element_valid ? 0 : 0x80000000;

    amd_buffer_atomic_add_impl<scalar_t, vector_size>(
        src_thread_data, dst_wave_buffer_resource, dst_addr_shift + dst_thread_addr_offset, 0);
#else
    if(dst_thread_element_valid)
    {
        amd_buffer_atomic_add_impl<scalar_t, vector_size>(
            src_thread_data, dst_wave_buffer_resource, dst_thread_addr_offset, 0);
    }
#endif
}

// buffer_atomic_max requires:
//   1) p_dst_wave must point to global memory
//   2) p_dst_wave must be a wavewise pointer.
// It is user's responsibility to make sure that is true.
template <typename T, index_t N>
__device__ void
amd_buffer_atomic_max(const typename vector_type_maker<T, N>::type::type src_thread_data,
                      T* p_dst_wave,
                      const index_t dst_thread_element_offset,
                      const bool dst_thread_element_valid,
                      const index_t dst_element_space_size)
{
    const int32x4_t dst_wave_buffer_resource =
        make_wave_buffer_resource(p_dst_wave, dst_element_space_size);

    index_t dst_thread_addr_offset = dst_thread_element_offset * sizeof(T);

    using vector_t                = typename vector_type_maker<T, N>::type::type;
    using scalar_t                = typename scalar_type<vector_t>::type;
    constexpr index_t vector_size = scalar_type<vector_t>::vector_size;

#if CK_EXPERIMENTAL_USE_BUFFER_ATOMIC_MAX_OOB_CHECK_OFFSET_TRICK
    uint32_t dst_addr_shift = dst_thread_element_valid ? 0 : 0x80000000;

    amd_buffer_atomic_max_impl<scalar_t, vector_size>(
        src_thread_data, dst_wave_buffer_resource, dst_addr_shift + dst_thread_addr_offset, 0);
#else
    if(dst_thread_element_valid)
    {
        amd_buffer_atomic_max_impl<scalar_t, vector_size>(
            src_thread_data, dst_wave_buffer_resource, dst_thread_addr_offset, 0);
    }
#endif
}

} // namespace ck
