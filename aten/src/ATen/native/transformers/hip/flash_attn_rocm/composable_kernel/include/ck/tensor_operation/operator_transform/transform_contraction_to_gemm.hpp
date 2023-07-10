// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_operation/gpu/device/matrix_padder.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/tensor_specialization.hpp"

namespace ck {
namespace tensor_operation {

// assume C[G0, G1, ..., M0, M1, M2, ..., N0, N1, N2...]
template <index_t NumDimG,
          index_t NumDimM,
          index_t NumDimN,
          device::TensorSpecialization TensorSpec>
static auto MakeGridDescriptorPair(const std::vector<index_t>& gs_ms_ns_lengths_vec,
                                   const std::vector<index_t>& gs_ms_ns_strides_vec)
{
    if(!(gs_ms_ns_lengths_vec.size() == NumDimG + NumDimM + NumDimN &&
         gs_ms_ns_strides_vec.size() == NumDimG + NumDimM + NumDimN))
    {
        throw std::runtime_error("wrong! dimension must match input lengths");
    }

    const auto to_tuple = [&](auto& vec, auto start, auto end) {
        return generate_tuple([&](auto i) { return vec[start + i]; }, Number<end - start>{});
    };

    const auto gs_ms_ns_lengths =
        to_tuple(gs_ms_ns_lengths_vec, Number<0>{}, Number<NumDimG + NumDimM + NumDimN>{});
    const auto gs_ms_ns_strides =
        to_tuple(gs_ms_ns_strides_vec, Number<0>{}, Number<NumDimG + NumDimM + NumDimN>{});

    // dimension Ids for G0, G1, ...
    constexpr auto gDimIds = typename arithmetic_sequence_gen<0, NumDimG, 1>::type{};

    // dimension Ids for M0, M1, ...
    constexpr auto mDimIds =
        typename arithmetic_sequence_gen<NumDimG, NumDimG + NumDimM, 1>::type{};

    // dimension Ids for N0, N1, ...
    constexpr auto nDimIds =
        typename arithmetic_sequence_gen<NumDimG + NumDimM, NumDimG + NumDimM + NumDimN, 1>::type{};

    // lengths for G0, G1, ...
    const auto gLengths = get_container_subset(gs_ms_ns_lengths, gDimIds);

    // lengths for M0, M1, ...
    const auto mLengths = get_container_subset(gs_ms_ns_lengths, mDimIds);

    // lengths for N0, N1, ...
    const auto nLengths = get_container_subset(gs_ms_ns_lengths, nDimIds);

    if constexpr(TensorSpec == device::TensorSpecialization::Packed)
    {
        auto G = container_reduce(gLengths, math::multiplies{}, Number<1>{});
        auto M = container_reduce(mLengths, math::multiplies{}, Number<1>{});
        auto N = container_reduce(nLengths, math::multiplies{}, Number<1>{});
        const auto grid_desc_g_mraw_nraw = make_naive_tensor_descriptor(
            make_tuple(G, M, N),
            make_tuple(gs_ms_ns_strides[Number<NumDimG - 1>{}],
                       gs_ms_ns_strides[Number<NumDimG + NumDimM - 1>{}],
                       gs_ms_ns_strides[Number<NumDimG + NumDimM + NumDimN - 1>{}]));

        const auto grid_desc_mraw_nraw = make_naive_tensor_descriptor(
            make_tuple(M, N),
            make_tuple(gs_ms_ns_strides[Number<NumDimG + NumDimM - 1>{}],
                       gs_ms_ns_strides[Number<NumDimG + NumDimM + NumDimN - 1>{}]));

        return std::make_pair(grid_desc_g_mraw_nraw, grid_desc_mraw_nraw);
    }
    else
    {
        // naive tensor C[G0, G1, ..., M0, M1, M2, ..., N0, N1, N2...]
        const auto grid_desc_gs_ms_ns =
            make_naive_tensor_descriptor(gs_ms_ns_lengths, gs_ms_ns_strides);

        // transformed tensor C[G = G0 * G1 * ..., MRaw = M0 * M1 * M2 * ... , NRaw = N0 * N1 *
        // N2 * ...]
        // Note: This does not require padding as it only provides G offset calculation. Technically
        // descriptor for only G is needed. Here we opt for backward compatibility purpose to return
        // G_M_N
        const auto grid_desc_g_mraw_nraw =
            transform_tensor_descriptor(grid_desc_gs_ms_ns,
                                        make_tuple(make_merge_transform(gLengths),
                                                   make_merge_transform(mLengths),
                                                   make_merge_transform(nLengths)),
                                        make_tuple(gDimIds, mDimIds, nDimIds),
                                        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));

        const auto c_ms_ns_lengths = to_tuple(
            gs_ms_ns_lengths_vec, Number<NumDimG>{}, Number<NumDimG + NumDimM + NumDimN>{});
        const auto c_ms_ns_strides = to_tuple(
            gs_ms_ns_strides_vec, Number<NumDimG>{}, Number<NumDimG + NumDimM + NumDimN>{});

        // transformed tensor C[MRaw = M0 * M1 * M2 * ... , NRaw = N0 * N1 *
        // N2 * ...]
        const auto grid_desc_ms_ns = make_naive_tensor_descriptor(c_ms_ns_lengths, c_ms_ns_strides);

        const auto grid_desc_mraw_nraw = transform_tensor_descriptor(
            grid_desc_ms_ns,
            make_tuple(make_merge_transform(mLengths), make_merge_transform(nLengths)),
            make_tuple(mDimIds - Number<NumDimG>{}, nDimIds - Number<NumDimG>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}));

        return std::make_pair(grid_desc_g_mraw_nraw, grid_desc_mraw_nraw);
    }
}

template <typename NumDims_G_M_N_K_O, // Sequence<>
          typename PerBlock_M_N_K_O,  // Sequence<>
          device::GemmSpecialization GemmSpec,
          device::TensorSpecialization ASpec,
          device::TensorSpecialization B0Spec,
          device::TensorSpecialization B1Spec,
          device::TensorSpecialization CSpec>
struct TransformBatchedContractionContractionToBatchedGemmGemm
{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};
    static constexpr auto I4 = Number<4>{};

    static constexpr index_t NumDimG = NumDims_G_M_N_K_O::At(I0);
    static constexpr index_t NumDimM = NumDims_G_M_N_K_O::At(I1);
    static constexpr index_t NumDimN = NumDims_G_M_N_K_O::At(I2);
    static constexpr index_t NumDimK = NumDims_G_M_N_K_O::At(I3);
    static constexpr index_t NumDimO = NumDims_G_M_N_K_O::At(I4);

    static constexpr index_t MPerBlock = PerBlock_M_N_K_O::At(I0);
    static constexpr index_t NPerBlock = PerBlock_M_N_K_O::At(I1);
    static constexpr index_t KPerBlock = PerBlock_M_N_K_O::At(I2);
    static constexpr index_t OPerBlock = PerBlock_M_N_K_O::At(I3);

    static constexpr auto matrix_padder =
        device::GemmGemmPadder<GemmSpec, index_t, index_t, index_t, index_t>{
            MPerBlock, NPerBlock, KPerBlock, OPerBlock};

    //
    // A
    //
    static auto MakeAGridDescriptorPair(const std::vector<index_t>& a_gs_ms_ks_lengths_vec,
                                        const std::vector<index_t>& a_gs_ms_ks_strides_vec)
    {
        return MakeGridDescriptorPair<NumDimG, NumDimM, NumDimK, ASpec>(a_gs_ms_ks_lengths_vec,
                                                                        a_gs_ms_ks_strides_vec);
    }

    // TODO: rename to G_MRaw_KRaw
    static auto MakeAGridDescriptor_G_M_K(const std::vector<index_t>& a_gs_ms_ks_lengths_vec,
                                          const std::vector<index_t>& a_gs_ms_ks_strides_vec)
    {
        return MakeAGridDescriptorPair(a_gs_ms_ks_lengths_vec, a_gs_ms_ks_strides_vec).first;
    }
    static auto MakeAGridDescriptor_M_K(const std::vector<index_t>& a_gs_ms_ks_lengths_vec,
                                        const std::vector<index_t>& a_gs_ms_ks_strides_vec)
    {
        return matrix_padder.PadADescriptor_M_K(
            MakeAGridDescriptorPair(a_gs_ms_ks_lengths_vec, a_gs_ms_ks_strides_vec).second);
    }

    template <typename AGridDesc_M_K, typename Number>
    __host__ __device__ static constexpr auto
    MakeAGridDescriptor_AK0_M_AK1(const AGridDesc_M_K& a_grid_desc_m_k, const Number& AK1)
    {
        const auto M = a_grid_desc_m_k.GetLength(I0);
        const auto K = a_grid_desc_m_k.GetLength(I1);

        const auto AK0 = K / AK1;

        return transform_tensor_descriptor(a_grid_desc_m_k,
                                           make_tuple(make_unmerge_transform(make_tuple(AK0, AK1)),
                                                      make_pass_through_transform(M)),
                                           make_tuple(Sequence<1>{}, Sequence<0>{}),
                                           make_tuple(Sequence<0, 2>{}, Sequence<1>{}));
    }

    //
    // B (alias of B0)
    //
    static auto MakeB0GridDescriptorPair(const std::vector<index_t>& b0_gs_ns_ks_lengths_vec,
                                         const std::vector<index_t>& b0_gs_ns_ks_strides_vec)
    {
        return MakeGridDescriptorPair<NumDimG, NumDimN, NumDimK, B0Spec>(b0_gs_ns_ks_lengths_vec,
                                                                         b0_gs_ns_ks_strides_vec);
    }

    // TODO: rename to G_MRaw_NRaw
    static auto MakeB0GridDescriptor_G_N_K(const std::vector<index_t>& b0_gs_ns_ks_lengths_vec,
                                           const std::vector<index_t>& b0_gs_ns_ks_strides_vec)
    {
        return MakeB0GridDescriptorPair(b0_gs_ns_ks_lengths_vec, b0_gs_ns_ks_strides_vec).first;
    }
    static auto MakeB0GridDescriptor_N_K(const std::vector<index_t>& b0_gs_ns_ks_lengths_vec,
                                         const std::vector<index_t>& b0_gs_ns_ks_strides_vec)
    {
        // alias of matrix_padder.PadB0Descriptor_N_K
        return matrix_padder.PadBDescriptor_N_K(
            MakeB0GridDescriptorPair(b0_gs_ns_ks_lengths_vec, b0_gs_ns_ks_strides_vec).second);
    }

    template <typename BGridDesc_N_K, typename Number>
    __host__ __device__ static constexpr auto
    MakeB0GridDescriptor_BK0_N_BK1(const BGridDesc_N_K& b_grid_desc_n_k, const Number& BK1)
    {
        const auto N = b_grid_desc_n_k.GetLength(I0);
        const auto K = b_grid_desc_n_k.GetLength(I1);

        const auto BK0 = K / BK1;

        return transform_tensor_descriptor(b_grid_desc_n_k,
                                           make_tuple(make_unmerge_transform(make_tuple(BK0, BK1)),
                                                      make_pass_through_transform(N)),
                                           make_tuple(Sequence<1>{}, Sequence<0>{}),
                                           make_tuple(Sequence<0, 2>{}, Sequence<1>{}));
    }

    //
    // B1
    //
    static auto MakeB1GridDescriptorPair(const std::vector<index_t>& b1_gs_os_ns_lengths_vec,
                                         const std::vector<index_t>& b1_gs_os_ns_strides_vec)
    {
        return MakeGridDescriptorPair<NumDimG, NumDimO, NumDimN, B1Spec>(b1_gs_os_ns_lengths_vec,
                                                                         b1_gs_os_ns_strides_vec);
    }

    // TODO: rename to G_NRaw_KRaw
    static auto MakeB1GridDescriptor_G_N_K(const std::vector<index_t>& b1_gs_os_ns_lengths_vec,
                                           const std::vector<index_t>& b1_gs_os_ns_strides_vec)
    {
        return MakeB1GridDescriptorPair(b1_gs_os_ns_lengths_vec, b1_gs_os_ns_strides_vec).first;
    }
    static auto MakeB1GridDescriptor_N_K(const std::vector<index_t>& b1_gs_os_ns_lengths_vec,
                                         const std::vector<index_t>& b1_gs_os_ns_strides_vec)
    {
        // alias of matrix_padder.PadB1Descriptor_O_N
        return matrix_padder.PadB1Descriptor_N_K(
            MakeB1GridDescriptorPair(b1_gs_os_ns_lengths_vec, b1_gs_os_ns_strides_vec).second);
    }

    template <typename B1GridDesc_N_K, typename Number>
    __host__ __device__ static constexpr auto
    MakeB1GridDescriptor_BK0_N_BK1(const B1GridDesc_N_K& b1_grid_desc_n_k, const Number& B1K1)
    {
        const auto N = b1_grid_desc_n_k.GetLength(I0);
        const auto K = b1_grid_desc_n_k.GetLength(I1);

        const auto B1K0 = K / B1K1;

        return transform_tensor_descriptor(
            b1_grid_desc_n_k,
            make_tuple(make_unmerge_transform(make_tuple(B1K0, B1K1)),
                       make_pass_through_transform(N)),
            make_tuple(Sequence<1>{}, Sequence<0>{}),
            make_tuple(Sequence<0, 2>{}, Sequence<1>{}));
    }

    //
    // C
    //
    static auto MakeCGridDescriptorPair(const std::vector<index_t>& c_gs_ms_os_lengths_vec,
                                        const std::vector<index_t>& c_gs_ms_os_strides_vec)
    {
        return MakeGridDescriptorPair<NumDimG, NumDimM, NumDimO, CSpec>(c_gs_ms_os_lengths_vec,
                                                                        c_gs_ms_os_strides_vec);
    }

    // TODO: rename to G_MRaw_NRaw
    static auto MakeCGridDescriptor_G_M_N(const std::vector<index_t>& c_gs_ms_os_lengths_vec,
                                          const std::vector<index_t>& c_gs_ms_os_strides_vec)
    {
        return MakeCGridDescriptorPair(c_gs_ms_os_lengths_vec, c_gs_ms_os_strides_vec).first;
    }
    static auto MakeCGridDescriptor_M_N(const std::vector<index_t>& c_gs_ms_os_lengths_vec,
                                        const std::vector<index_t>& c_gs_ms_os_strides_vec)
    {
        return matrix_padder.PadCDescriptor_M_N(
            MakeCGridDescriptorPair(c_gs_ms_os_lengths_vec, c_gs_ms_os_strides_vec).second);
    }
};

} // namespace tensor_operation
} // namespace ck
