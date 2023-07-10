// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/utility/math.hpp"
#include "ck/utility/amd_xdlops.hpp"

namespace ck {

enum struct MfmaInstr
{
    mfma_f32_32x32x1xf32 = 0,
    mfma_f32_16x16x1xf32,
    mfma_f32_4x4x1xf32,
    mfma_f32_32x32x2xf32,
    mfma_f32_16x16x4xf32,
    mfma_f32_32x32x4f16,
    mfma_f32_16x16x4f16,
    mfma_f32_4x4x4f16,
    mfma_f32_32x32x8f16,
    mfma_f32_16x16x16f16,
    mfma_f32_32x32x8bf16_1k,
    mfma_f32_16x16x16bf16_1k,
    mfma_f32_32x32x4bf16,
    mfma_f32_16x16x8bf16,
    mfma_i32_32x32x8i8,
    mfma_i32_16x16x16i8,
    mfma_f64_16x16x4f64
};

template <MfmaInstr instr>
struct mfma_type;

template <>
struct mfma_type<MfmaInstr::mfma_f32_32x32x1xf32>
{
    static constexpr index_t group_size          = 4;
    static constexpr index_t num_groups_per_blk  = 4;
    static constexpr index_t num_regs_per_blk    = 16;
    static constexpr index_t num_threads_per_blk = 32;
    static constexpr index_t wave_size           = 64;
    static constexpr index_t num_input_blks      = 2;
    static constexpr index_t num_output_blks     = 2;
    static constexpr index_t m_per_blk           = 32;
    static constexpr index_t n_per_blk           = 32;
    static constexpr index_t k_per_blk           = 1;
    static constexpr bool is_k_reduction         = false;

    template <index_t MPerXdlops, index_t NPerXdlops, class FloatA, class FloatB, class FloatC>
    __device__ void run(const FloatA& a, const FloatB& b, FloatC& reg_c) const
    {
        intrin_mfma_f32_32x32x1f32<MPerXdlops, NPerXdlops>::Run(a, b, reg_c);
    }
};

template <>
struct mfma_type<MfmaInstr::mfma_f32_32x32x2xf32>
{
    static constexpr index_t group_size          = 4;
    static constexpr index_t num_groups_per_blk  = 4;
    static constexpr index_t num_regs_per_blk    = 16;
    static constexpr index_t num_threads_per_blk = 32;
    static constexpr index_t wave_size           = 64;
    static constexpr index_t num_input_blks      = 2;
    static constexpr index_t num_output_blks     = 1;
    static constexpr index_t m_per_blk           = 32;
    static constexpr index_t n_per_blk           = 32;
    static constexpr index_t k_per_blk           = 1;
    static constexpr bool is_k_reduction         = true;

    template <index_t MPerXdlops, index_t NPerXdlops, class FloatA, class FloatB, class FloatC>
    __device__ void run(const FloatA& a, const FloatB& b, FloatC& reg_c) const
    {
        intrin_mfma_f32_32x32x2f32<MPerXdlops, NPerXdlops>::Run(a, b, reg_c);
    }
};

template <>
struct mfma_type<MfmaInstr::mfma_f32_16x16x4xf32>
{
    static constexpr index_t group_size          = 4;
    static constexpr index_t num_groups_per_blk  = 1;
    static constexpr index_t num_regs_per_blk    = 4;
    static constexpr index_t num_threads_per_blk = 16;
    static constexpr index_t wave_size           = 64;
    static constexpr index_t num_input_blks      = 4;
    static constexpr index_t num_output_blks     = 1;
    static constexpr index_t m_per_blk           = 16;
    static constexpr index_t n_per_blk           = 16;
    static constexpr index_t k_per_blk           = 1;
    static constexpr bool is_k_reduction         = true;

    template <index_t MPerXdlops, index_t NPerXdlops, class FloatA, class FloatB, class FloatC>
    __device__ void run(const FloatA& a, const FloatB& b, FloatC& reg_c) const
    {
        intrin_mfma_f32_16x16x4f32<MPerXdlops, NPerXdlops>::Run(a, b, reg_c);
    }
};

template <>
struct mfma_type<MfmaInstr::mfma_f32_16x16x1xf32>
{
    static constexpr index_t group_size          = 4;
    static constexpr index_t num_groups_per_blk  = 1;
    static constexpr index_t num_regs_per_blk    = 4;
    static constexpr index_t num_threads_per_blk = 16;
    static constexpr index_t wave_size           = 64;
    static constexpr index_t num_input_blks      = 4;
    static constexpr index_t num_output_blks     = 4;
    static constexpr index_t m_per_blk           = 16;
    static constexpr index_t n_per_blk           = 16;
    static constexpr index_t k_per_blk           = 1;
    static constexpr bool is_k_reduction         = false;

    template <index_t MPerXdlops, index_t NPerXdlops, class FloatA, class FloatB, class FloatC>
    __device__ void run(const FloatA& a, const FloatB& b, FloatC& reg_c) const
    {
        intrin_mfma_f32_16x16x1f32<MPerXdlops, NPerXdlops>::Run(a, b, reg_c);
    }
};

// treat 4x4x1 as a single-blk 4x64 mfma
template <>
struct mfma_type<MfmaInstr::mfma_f32_4x4x1xf32>
{
    static constexpr index_t group_size          = 4;
    static constexpr index_t num_groups_per_blk  = 1;
    static constexpr index_t num_regs_per_blk    = 4;
    static constexpr index_t num_threads_per_blk = 64;
    static constexpr index_t wave_size           = 64;
    static constexpr index_t num_input_blks      = 1;
    static constexpr index_t num_output_blks     = 1;
    static constexpr index_t m_per_blk           = 4;
    static constexpr index_t n_per_blk           = 64;
    static constexpr index_t k_per_blk           = 1;
    static constexpr bool is_k_reduction         = false;

    template <index_t MPerXdlops, index_t NPerXdlops, class FloatA, class FloatB, class FloatC>
    __device__ void run(const FloatA& a, const FloatB& b, FloatC& reg_c) const
    {
        intrin_mfma_f32_4x4x1f32<MPerXdlops, NPerXdlops>::Run(a, b, reg_c);
    }
};

template <>
struct mfma_type<MfmaInstr::mfma_f32_32x32x4f16>
{
    static constexpr index_t group_size          = 4;
    static constexpr index_t num_groups_per_blk  = 4;
    static constexpr index_t num_regs_per_blk    = 16;
    static constexpr index_t num_threads_per_blk = 32;
    static constexpr index_t wave_size           = 64;
    static constexpr index_t num_input_blks      = 2;
    static constexpr index_t num_output_blks     = 2;
    static constexpr index_t m_per_blk           = 32;
    static constexpr index_t n_per_blk           = 32;
    static constexpr index_t k_per_blk           = 4;
    static constexpr bool is_k_reduction         = false;

    template <index_t MPerXdlops, index_t NPerXdlops, class FloatA, class FloatB, class FloatC>
    __device__ void run(const FloatA& a, const FloatB& b, FloatC& reg_c) const
    {
        intrin_mfma_f32_32x32x4f16<MPerXdlops, NPerXdlops>::Run(a, b, reg_c);
    }
};

template <>
struct mfma_type<MfmaInstr::mfma_f32_32x32x8f16>
{
    static constexpr index_t group_size          = 4;
    static constexpr index_t num_groups_per_blk  = 4;
    static constexpr index_t num_regs_per_blk    = 16;
    static constexpr index_t num_threads_per_blk = 32;
    static constexpr index_t wave_size           = 64;
    static constexpr index_t num_input_blks      = 2;
    static constexpr index_t num_output_blks     = 1;
    static constexpr index_t m_per_blk           = 32;
    static constexpr index_t n_per_blk           = 32;
    static constexpr index_t k_per_blk           = 4;
    static constexpr bool is_k_reduction         = true;

    template <index_t MPerXdlops, index_t NPerXdlops, class FloatA, class FloatB, class FloatC>
    __device__ void run(const FloatA& a, const FloatB& b, FloatC& reg_c) const
    {
        intrin_mfma_f32_32x32x8f16<MPerXdlops, NPerXdlops>::Run(a, b, reg_c);
    }
};

template <>
struct mfma_type<MfmaInstr::mfma_f32_16x16x16f16>
{
    static constexpr index_t group_size          = 4;
    static constexpr index_t num_groups_per_blk  = 1;
    static constexpr index_t num_regs_per_blk    = 4;
    static constexpr index_t num_threads_per_blk = 16;
    static constexpr index_t wave_size           = 64;
    static constexpr index_t num_input_blks      = 4;
    static constexpr index_t num_output_blks     = 1;
    static constexpr index_t m_per_blk           = 16;
    static constexpr index_t n_per_blk           = 16;
    static constexpr index_t k_per_blk           = 4;
    static constexpr bool is_k_reduction         = true;

    template <index_t MPerXdlops, index_t NPerXdlops, class FloatA, class FloatB, class FloatC>
    __device__ void run(const FloatA& a, const FloatB& b, FloatC& reg_c) const
    {
        intrin_mfma_f32_16x16x16f16<MPerXdlops, NPerXdlops>::Run(a, b, reg_c);
    }
};

template <>
struct mfma_type<MfmaInstr::mfma_f32_16x16x4f16>
{
    static constexpr index_t group_size          = 4;
    static constexpr index_t num_groups_per_blk  = 1;
    static constexpr index_t num_regs_per_blk    = 4;
    static constexpr index_t num_threads_per_blk = 16;
    static constexpr index_t wave_size           = 64;
    static constexpr index_t num_input_blks      = 4;
    static constexpr index_t num_output_blks     = 4;
    static constexpr index_t m_per_blk           = 16;
    static constexpr index_t n_per_blk           = 16;
    static constexpr index_t k_per_blk           = 4;
    static constexpr bool is_k_reduction         = false;

    template <index_t MPerXdlops, index_t NPerXdlops, class FloatA, class FloatB, class FloatC>
    __device__ void run(const FloatA& a, const FloatB& b, FloatC& reg_c) const
    {
        intrin_mfma_f32_16x16x4f16<MPerXdlops, NPerXdlops>::Run(a, b, reg_c);
    }
};

template <>
struct mfma_type<MfmaInstr::mfma_f32_4x4x4f16>
{
    static constexpr index_t group_size          = 4;
    static constexpr index_t num_groups_per_blk  = 1;
    static constexpr index_t num_regs_per_blk    = 4;
    static constexpr index_t num_threads_per_blk = 64;
    static constexpr index_t wave_size           = 64;
    static constexpr index_t num_input_blks      = 1;
    static constexpr index_t num_output_blks     = 1;
    static constexpr index_t m_per_blk           = 4;
    static constexpr index_t n_per_blk           = 64;
    static constexpr index_t k_per_blk           = 4;
    static constexpr bool is_k_reduction         = false;

    template <index_t MPerXdlops, index_t NPerXdlops, class FloatA, class FloatB, class FloatC>
    __device__ void run(const FloatA& a, const FloatB& b, FloatC& reg_c) const
    {
        intrin_mfma_f32_4x4x4f16<MPerXdlops, NPerXdlops>::Run(a, b, reg_c);
    }
};

template <>
struct mfma_type<MfmaInstr::mfma_f32_32x32x8bf16_1k>
{
    static constexpr index_t group_size          = 4;
    static constexpr index_t num_groups_per_blk  = 4;
    static constexpr index_t num_regs_per_blk    = 16;
    static constexpr index_t num_threads_per_blk = 32;
    static constexpr index_t wave_size           = 64;
    static constexpr index_t num_input_blks      = 2;
    static constexpr index_t num_output_blks     = 1;
    static constexpr index_t m_per_blk           = 32;
    static constexpr index_t n_per_blk           = 32;
    static constexpr index_t k_per_blk           = 4;
    static constexpr bool is_k_reduction         = true;

    template <index_t MPerXdlops, index_t NPerXdlops, class FloatA, class FloatB, class FloatC>
    __device__ void run(const FloatA& a, const FloatB& b, FloatC& reg_c) const
    {
        intrin_mfma_f32_32x32x8bf16_1k<MPerXdlops, NPerXdlops>::Run(a, b, reg_c);
    }
};

template <>
struct mfma_type<MfmaInstr::mfma_f32_16x16x16bf16_1k>
{
    static constexpr index_t group_size          = 4;
    static constexpr index_t num_groups_per_blk  = 1;
    static constexpr index_t num_regs_per_blk    = 4;
    static constexpr index_t num_threads_per_blk = 16;
    static constexpr index_t wave_size           = 64;
    static constexpr index_t num_input_blks      = 4;
    static constexpr index_t num_output_blks     = 1;
    static constexpr index_t m_per_blk           = 16;
    static constexpr index_t n_per_blk           = 16;
    static constexpr index_t k_per_blk           = 4;
    static constexpr bool is_k_reduction         = true;

    template <index_t MPerXdlops, index_t NPerXdlops, class FloatA, class FloatB, class FloatC>
    __device__ void run(const FloatA& a, const FloatB& b, FloatC& reg_c) const
    {
        intrin_mfma_f32_16x16x16bf16_1k<MPerXdlops, NPerXdlops>::Run(a, b, reg_c);
    }
};

template <>
struct mfma_type<MfmaInstr::mfma_f32_32x32x4bf16>
{
    static constexpr index_t group_size          = 4;
    static constexpr index_t num_groups_per_blk  = 4;
    static constexpr index_t num_regs_per_blk    = 16;
    static constexpr index_t num_threads_per_blk = 32;
    static constexpr index_t wave_size           = 64;
    static constexpr index_t num_input_blks      = 2;
    static constexpr index_t num_output_blks     = 1;
    static constexpr index_t m_per_blk           = 32;
    static constexpr index_t n_per_blk           = 32;
    static constexpr index_t k_per_blk           = 2;
    static constexpr bool is_k_reduction         = true;

    template <index_t MPerXdlops, index_t NPerXdlops, class FloatA, class FloatB, class FloatC>
    __device__ void run(const FloatA& a, const FloatB& b, FloatC& reg_c) const
    {
        intrin_mfma_f32_32x32x4bf16<MPerXdlops, NPerXdlops>::Run(a, b, reg_c);
    }
};

template <>
struct mfma_type<MfmaInstr::mfma_f32_16x16x8bf16>
{
    static constexpr index_t group_size          = 4;
    static constexpr index_t num_groups_per_blk  = 1;
    static constexpr index_t num_regs_per_blk    = 4;
    static constexpr index_t num_threads_per_blk = 16;
    static constexpr index_t wave_size           = 64;
    static constexpr index_t num_input_blks      = 4;
    static constexpr index_t num_output_blks     = 1;
    static constexpr index_t m_per_blk           = 16;
    static constexpr index_t n_per_blk           = 16;
    static constexpr index_t k_per_blk           = 2;
    static constexpr bool is_k_reduction         = true;

    template <index_t MPerXdlops, index_t NPerXdlops, class FloatA, class FloatB, class FloatC>
    __device__ void run(const FloatA& a, const FloatB& b, FloatC& reg_c) const
    {
        intrin_mfma_f32_16x16x8bf16<MPerXdlops, NPerXdlops>::Run(a, b, reg_c);
    }
};

template <>
struct mfma_type<MfmaInstr::mfma_i32_32x32x8i8>
{
    static constexpr index_t group_size          = 4;
    static constexpr index_t num_groups_per_blk  = 4;
    static constexpr index_t num_regs_per_blk    = 16;
    static constexpr index_t num_threads_per_blk = 32;
    static constexpr index_t wave_size           = 64;
    static constexpr index_t num_input_blks      = 2;
    static constexpr index_t num_output_blks     = 1;
    static constexpr index_t m_per_blk           = 32;
    static constexpr index_t n_per_blk           = 32;
    static constexpr index_t k_per_blk           = 4;
    static constexpr bool is_k_reduction         = true;

    template <index_t MPerXdlops, index_t NPerXdlops, class FloatA, class FloatB, class FloatC>
    __device__ void run(const FloatA& a, const FloatB& b, FloatC& reg_c) const
    {
        intrin_mfma_i32_32x32x8i8<MPerXdlops, NPerXdlops>::Run(a, b, reg_c);
    }
};

template <>
struct mfma_type<MfmaInstr::mfma_i32_16x16x16i8>
{
    static constexpr index_t group_size          = 4;
    static constexpr index_t num_groups_per_blk  = 1;
    static constexpr index_t num_regs_per_blk    = 4;
    static constexpr index_t num_threads_per_blk = 16;
    static constexpr index_t wave_size           = 64;
    static constexpr index_t num_input_blks      = 4;
    static constexpr index_t num_output_blks     = 1;
    static constexpr index_t m_per_blk           = 16;
    static constexpr index_t n_per_blk           = 16;
    static constexpr index_t k_per_blk           = 4;
    static constexpr bool is_k_reduction         = true;

    template <index_t MPerXdlops, index_t NPerXdlops, class FloatA, class FloatB, class FloatC>
    __device__ void run(const FloatA& a, const FloatB& b, FloatC& reg_c) const
    {
        intrin_mfma_i32_16x16x16i8<MPerXdlops, NPerXdlops>::Run(a, b, reg_c);
    }
};

template <>
struct mfma_type<MfmaInstr::mfma_f64_16x16x4f64>
{
    static constexpr index_t group_size          = 1;
    static constexpr index_t num_groups_per_blk  = 4;
    static constexpr index_t num_regs_per_blk    = 4; // group_size * num_groups_per_blk;
    static constexpr index_t num_threads_per_blk = 16;
    static constexpr index_t wave_size           = 64;
    static constexpr index_t num_input_blks      = 4; // wave_size / num_threads_per_blk;
    static constexpr index_t num_output_blks     = 1;
    static constexpr index_t m_per_blk           = 16;
    static constexpr index_t n_per_blk           = 16;
    static constexpr index_t k_per_blk           = 1;
    static constexpr bool is_k_reduction         = true;

    template <index_t MPerXdlops, index_t NPerXdlops, class FloatA, class FloatB, class FloatC>
    __device__ void run(const FloatA& a, const FloatB& b, FloatC& reg_c) const
    {
        intrin_mfma_f64_16x16x4f64<MPerXdlops, NPerXdlops>::Run(a, b, reg_c);
    }
};

template <typename base_type, index_t MPerXdlops, index_t NPerXdlops>
struct MfmaSelector
{
    template <typename base_type_, index_t MPerXdlops_, index_t NPerXdlops_>
    static constexpr auto GetMfma();

    template <>
    static constexpr auto GetMfma<double, 16, 16>()
    {
        return MfmaInstr::mfma_f64_16x16x4f64;
    }

    template <>
    static constexpr auto GetMfma<float, 64, 64>()
    {
        return MfmaInstr::mfma_f32_32x32x1xf32;
    }

    template <>
    static constexpr auto GetMfma<float, 32, 64>()
    {
        return MfmaInstr::mfma_f32_32x32x1xf32;
    }

    template <>
    static constexpr auto GetMfma<float, 16, 64>()
    {
        return MfmaInstr::mfma_f32_16x16x1xf32;
    }

    template <>
    static constexpr auto GetMfma<float, 8, 64>()
    {
        return MfmaInstr::mfma_f32_4x4x1xf32;
    }

    template <>
    static constexpr auto GetMfma<float, 4, 64>()
    {
        return MfmaInstr::mfma_f32_4x4x1xf32;
    }

    template <>
    static constexpr auto GetMfma<float, 32, 32>()
    {
        return MfmaInstr::mfma_f32_32x32x2xf32;
    }

    template <>
    static constexpr auto GetMfma<float, 16, 16>()
    {
        return MfmaInstr::mfma_f32_16x16x4xf32;
    }

    template <>
    static constexpr auto GetMfma<half_t, 64, 64>()
    {
        return MfmaInstr::mfma_f32_32x32x4f16;
    }

    template <>
    static constexpr auto GetMfma<half_t, 32, 64>()
    {
        return MfmaInstr::mfma_f32_32x32x4f16;
    }

    template <>
    static constexpr auto GetMfma<half_t, 32, 32>()
    {
        return MfmaInstr::mfma_f32_32x32x8f16;
    }

    template <>
    static constexpr auto GetMfma<half_t, 16, 16>()
    {
        return MfmaInstr::mfma_f32_16x16x16f16;
    }

    template <>
    static constexpr auto GetMfma<half_t, 16, 64>()
    {
        return MfmaInstr::mfma_f32_16x16x4f16;
    }

    template <>
    static constexpr auto GetMfma<half_t, 8, 64>()
    {
        return MfmaInstr::mfma_f32_4x4x4f16;
    }

    template <>
    static constexpr auto GetMfma<half_t, 4, 64>()
    {
        return MfmaInstr::mfma_f32_4x4x4f16;
    }

    template <>
    static constexpr auto GetMfma<bhalf_t, 32, 32>()
    {
#if defined(CK_USE_AMD_MFMA_BF16_1K_OP)
        return MfmaInstr::mfma_f32_32x32x8bf16_1k;
#else
        return MfmaInstr::mfma_f32_32x32x4bf16;
#endif
    }

    template <>
    static constexpr auto GetMfma<bhalf_t, 16, 16>()
    {
#if defined(CK_USE_AMD_MFMA_BF16_1K_OP)
        return MfmaInstr::mfma_f32_16x16x16bf16_1k;
#else
        return MfmaInstr::mfma_f32_16x16x8bf16;
#endif
    }

    template <>
    static constexpr auto GetMfma<int8_t, 32, 32>()
    {
        return MfmaInstr::mfma_i32_32x32x8i8;
    }

    template <>
    static constexpr auto GetMfma<int8_t, 16, 16>()
    {
        return MfmaInstr::mfma_i32_16x16x16i8;
    }

    static constexpr auto selected_mfma = mfma_type<GetMfma<base_type, MPerXdlops, NPerXdlops>()>{};

    __host__ __device__ constexpr MfmaSelector()
    {
        static_assert(selected_mfma.group_size * selected_mfma.num_groups_per_blk ==
                          selected_mfma.num_regs_per_blk,
                      "wrong! num_regs_per_blk");

        static_assert(selected_mfma.num_threads_per_blk == selected_mfma.n_per_blk,
                      "n_per_blk != num_threads_per_blk");

        static_assert(selected_mfma.num_regs_per_blk * selected_mfma.num_input_blks ==
                          selected_mfma.m_per_blk,
                      "m_per_blk != num_input_blks * num_regs_per_blk");

        static_assert(selected_mfma.num_output_blks == selected_mfma.num_input_blks ||
                          selected_mfma.num_output_blks == 1,
                      "incorrect num_output_blks");

        static_assert(selected_mfma.num_regs_per_blk * selected_mfma.wave_size ==
                          selected_mfma.m_per_blk * selected_mfma.n_per_blk,
                      "num_regs_per_blk incorrect");

        static_assert(selected_mfma.is_k_reduction ||
                          (selected_mfma.num_input_blks == selected_mfma.num_output_blks),
                      "is_k_reduction wrong!");
    }

    static constexpr bool IsABroadcast()
    {
        static_assert(NPerXdlops >= MPerXdlops, "only support ABroadcast");
        return true;
    }

    static constexpr index_t GetKPerXdlops()
    {
        return (selected_mfma.is_k_reduction ? selected_mfma.num_input_blks : 1) *
               selected_mfma.k_per_blk;
    }

    static constexpr index_t GetK1PerXdlops() { return selected_mfma.k_per_blk; }
};

template <typename base_type,
          index_t MPerXdlops,
          index_t NPerXdlops,
          index_t KPack,
          bool TransposeC = false>
struct XdlopsGemm
{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};
    static constexpr auto I4 = Number<4>{};
    static constexpr auto I5 = Number<5>{};

    using CIndex   = MultiIndex<2>;
    using CIndex4D = MultiIndex<4>;

    __device__ static constexpr index_t GetNumBlks() { return mfma_instr.num_output_blks; }

    __device__ static constexpr index_t GetNumXdlops()
    {
        return MPerXdlops * NPerXdlops /
               (mfma_instr.m_per_blk * mfma_instr.n_per_blk * mfma_instr.num_output_blks);
    }

    __host__ __device__ constexpr XdlopsGemm()
    {
        static_assert(NPerXdlops == 4 || NPerXdlops == 8 || NPerXdlops == 16 || NPerXdlops == 32 ||
                          NPerXdlops == 64,
                      "Only support GemmNPerXdlops == 4, 8, 16, 32 or 64 for xdlops");

        static_assert(MPerXdlops == 4 || MPerXdlops == 8 || MPerXdlops == 16 || MPerXdlops == 32 ||
                          MPerXdlops == 64,
                      "Only support GemmMPerXdlops == 4, 8, 16, 32 or 64 for xdlops");

        static_assert(KPack % mfma_instr.k_per_blk == 0, "KPack cannot be divided by k_per_blk");
    }

    // XDL output supporting C = A * B
    // M2_N2 -> M2_M3_M4_N2
    template <typename CDesc_M0_N0_M1_N1_M2_N2>
    __host__ __device__ static constexpr auto
    MakeCDescriptor_M0_N0_M1_N1_M2_M3_M4_N2(const CDesc_M0_N0_M1_N1_M2_N2& c_desc_m0_n0_m1_n1_m2_n2)
    {
        const auto M0 = c_desc_m0_n0_m1_n1_m2_n2.GetLength(I0);
        const auto N0 = c_desc_m0_n0_m1_n1_m2_n2.GetLength(I1);
        const auto M1 = c_desc_m0_n0_m1_n1_m2_n2.GetLength(I2);
        const auto N1 = c_desc_m0_n0_m1_n1_m2_n2.GetLength(I3);

        return transform_tensor_descriptor(
            c_desc_m0_n0_m1_n1_m2_n2,
            make_tuple(make_pass_through_transform(M0),
                       make_pass_through_transform(N0),
                       make_pass_through_transform(M1),
                       make_pass_through_transform(N1),
                       make_unmerge_transform(make_tuple(Number<mfma_instr.num_groups_per_blk>{},
                                                         Number<mfma_instr.num_input_blks>{},
                                                         Number<mfma_instr.group_size>{})),
                       make_pass_through_transform(Number<mfma_instr.num_threads_per_blk>{})),
            make_tuple(Sequence<0>{},
                       Sequence<1>{},
                       Sequence<2>{},
                       Sequence<3>{},
                       Sequence<4>{},
                       Sequence<5>{}),
            make_tuple(Sequence<0>{},
                       Sequence<1>{},
                       Sequence<2>{},
                       Sequence<3>{},
                       Sequence<4, 5, 6>{},
                       Sequence<7>{}));
    }

    // transposed XDL output supporting C' = B' * A'
    // M2_N2 -> M2_N2_N3_N4
    template <typename CDesc_M0_N0_M1_N1_M2_N2>
    __host__ __device__ static constexpr auto
    MakeCDescriptor_M0_N0_M1_N1_M2_N2_N3_N4(const CDesc_M0_N0_M1_N1_M2_N2& c_desc_m0_n0_m1_n1_m2_n2)
    {
        const auto M0 = c_desc_m0_n0_m1_n1_m2_n2.GetLength(I0);
        const auto N0 = c_desc_m0_n0_m1_n1_m2_n2.GetLength(I1);
        const auto M1 = c_desc_m0_n0_m1_n1_m2_n2.GetLength(I2);
        const auto N1 = c_desc_m0_n0_m1_n1_m2_n2.GetLength(I3);

        return transform_tensor_descriptor(
            c_desc_m0_n0_m1_n1_m2_n2,
            make_tuple(make_pass_through_transform(M0),
                       make_pass_through_transform(N0),
                       make_pass_through_transform(M1),
                       make_pass_through_transform(N1),
                       make_pass_through_transform(Number<mfma_instr.num_threads_per_blk>{}),
                       make_unmerge_transform(make_tuple(Number<mfma_instr.num_groups_per_blk>{},
                                                         Number<mfma_instr.num_input_blks>{},
                                                         Number<mfma_instr.group_size>{}))),
            make_tuple(Sequence<0>{},
                       Sequence<1>{},
                       Sequence<2>{},
                       Sequence<3>{},
                       Sequence<4>{},
                       Sequence<5>{}),
            make_tuple(Sequence<0>{},
                       Sequence<1>{},
                       Sequence<2>{},
                       Sequence<3>{},
                       Sequence<4>{},
                       Sequence<5, 6, 7>{}));
    }

    template <typename CDesc_G_M0_N0_M1_N1_M2_N2>
    __host__ __device__ static constexpr auto MakeCDescriptor_G_M0_N0_M1_N1_M2_M3_M4_N2(
        const CDesc_G_M0_N0_M1_N1_M2_N2& c_desc_g_m0_n0_m1_n1_m2_n2)
    {
        const auto G  = c_desc_g_m0_n0_m1_n1_m2_n2.GetLength(I0);
        const auto M0 = c_desc_g_m0_n0_m1_n1_m2_n2.GetLength(I1);
        const auto N0 = c_desc_g_m0_n0_m1_n1_m2_n2.GetLength(I2);
        const auto M1 = c_desc_g_m0_n0_m1_n1_m2_n2.GetLength(I3);
        const auto N1 = c_desc_g_m0_n0_m1_n1_m2_n2.GetLength(I4);

        return transform_tensor_descriptor(
            c_desc_g_m0_n0_m1_n1_m2_n2,
            make_tuple(make_pass_through_transform(G),
                       make_pass_through_transform(M0),
                       make_pass_through_transform(N0),
                       make_pass_through_transform(M1),
                       make_pass_through_transform(N1),
                       make_unmerge_transform(make_tuple(mfma_instr.num_groups_per_blk,
                                                         mfma_instr.num_input_blks,
                                                         mfma_instr.group_size)),
                       make_pass_through_transform(mfma_instr.num_threads_per_blk)),
            make_tuple(Sequence<0>{},
                       Sequence<1>{},
                       Sequence<2>{},
                       Sequence<3>{},
                       Sequence<4>{},
                       Sequence<5>{},
                       Sequence<6>{}),
            make_tuple(Sequence<0>{},
                       Sequence<1>{},
                       Sequence<2>{},
                       Sequence<3>{},
                       Sequence<4>{},
                       Sequence<5, 6, 7>{},
                       Sequence<8>{}));
    }

    __device__ static constexpr index_t GetRegSizePerXdlops()
    {
        return MPerXdlops * NPerXdlops / mfma_instr.wave_size;
    }

    __device__ static constexpr index_t GetWaveSize() { return mfma_instr.wave_size; }

    template <class FloatA, class FloatB, class FloatC>
    __device__ void Run(const FloatA& p_a_wave, const FloatB& p_b_wave, FloatC& p_c_thread) const
    {
        static_assert(is_same<base_type, double>::value || is_same<base_type, float>::value ||
                          is_same<base_type, half_t>::value || is_same<base_type, bhalf_t>::value ||
                          is_same<base_type, int8_t>::value,
                      "base base_type must be double, float, half, bfloat16, and int8_t!");

        static_for<0, KPack / mfma_instr.k_per_blk, 1>{}([&](auto k) {
            if constexpr(!TransposeC)
            {
                mfma_instr.template run<MPerXdlops, NPerXdlops>(
                    p_a_wave[k], p_b_wave[k], p_c_thread);
            }
            else
            {
                mfma_instr.template run<MPerXdlops, NPerXdlops>(
                    p_b_wave[k], p_a_wave[k], p_c_thread);
            }
        });
    }

    __device__ static auto GetLaneId() { return get_thread_local_1d_id() % mfma_instr.wave_size; }

    __device__ static auto GetBlkIdx()
    {
        const auto laneId = GetLaneId();

        constexpr auto threadidx_to_blk_idx_adaptor = make_single_stage_tensor_adaptor(
            make_tuple(make_merge_transform(
                make_tuple(1, mfma_instr.num_input_blks, mfma_instr.num_threads_per_blk))),
            make_tuple(Sequence<0, 1, 2>{}),
            make_tuple(Sequence<0>{}));

        const auto blk_idx =
            threadidx_to_blk_idx_adaptor.CalculateBottomIndex(make_multi_index(laneId));

        const auto blk_id = blk_idx[I1];
        const auto blk_td = blk_idx[I2];

        return make_tuple(blk_id, blk_td);
    }

    __host__ __device__ static auto CalculateAThreadOriginDataIndex()
    {
        const auto laneId  = GetLaneId();
        const auto blk_idx = GetBlkIdx();

        const auto blk_id = blk_idx[I0];
        const auto blk_td = blk_idx[I1];

        if constexpr(mfma_instr.is_k_reduction)
        {
            return make_tuple(blk_id, blk_td);
        }
        else
        {
            return make_tuple(0, laneId);
        }
    }

    __host__ __device__ static auto CalculateBThreadOriginDataIndex()
    {
        const auto laneId  = GetLaneId();
        const auto blk_idx = GetBlkIdx();

        const auto blk_id = blk_idx[I0];
        const auto blk_td = blk_idx[I1];

        if constexpr(mfma_instr.is_k_reduction)
        {
            return make_tuple(blk_id, blk_td);
        }
        else
        {
            return make_tuple(0, laneId);
        }
    }

    __device__ static CIndex GetBeginOfThreadBlk(index_t xdlops_i, index_t blk_i)
    {
        const auto blk_idx = GetBlkIdx();

        const auto blk_id = blk_idx[I0];
        const auto blk_td = blk_idx[I1];

        index_t n_offset = blk_i * mfma_instr.n_per_blk + blk_td;
        index_t m_offset = xdlops_i * mfma_instr.m_per_blk + blk_id * mfma_instr.group_size;

        return TransposeC ? CIndex{n_offset, m_offset} : CIndex{m_offset, n_offset};
    }

    __device__ static CIndex4D GetBeginOfThreadBlk4D(index_t /* xdlops_i */, index_t /* blk_i */)
    {
        const auto blk_idx = GetBlkIdx();

        const auto blk_id = blk_idx[I0];
        const auto blk_td = blk_idx[I1];

        return TransposeC ? CIndex4D{blk_td, I0, blk_id, I0} : CIndex4D{I0, blk_id, I0, blk_td};
    }

    static constexpr auto mfma = MfmaSelector<base_type, MPerXdlops, NPerXdlops>{};

    static constexpr auto mfma_instr = mfma.selected_mfma;

    static constexpr auto KPerXdlops  = mfma.GetKPerXdlops();
    static constexpr auto K1PerXdlops = mfma.GetK1PerXdlops();
    static constexpr auto K0PerXdlops = KPerXdlops / K1PerXdlops;

    __host__ __device__ static constexpr auto GetCM0M1M2NThreadBlkLengths()
    {
        return make_tuple(
            Number<mfma_instr.num_groups_per_blk>{}, I1, Number<mfma_instr.group_size>{}, I1);
    }
};

} // namespace ck
