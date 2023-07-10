// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>
#include <sstream>

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_batched_contraction_multiple_d.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/tensor_specialization.hpp"
#include "ck/tensor_operation/gpu/device/matrix_padder.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_multiple_d_xdl_cshuffle.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"

namespace ck {

template <typename GridwiseGemm,
          typename FloatAB,
          typename FloatDsPointer,
          typename FloatE,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CDEElementwiseOperation,
          typename AGridDesc_AK0_M_AK1,
          typename BGridDesc_BK0_N_BK1,
          typename DsGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock,
          typename EGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock,
          typename ComputePtrOffsetOfBatch,
          typename Block2ETileMap,
          bool HasMainKBlockLoop>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
    __launch_bounds__(CK_MAX_THREAD_PER_BLOCK, CK_MIN_BLOCK_PER_CU)
#endif
        kernel_contraction_multiple_d_xdl_cshuffle(
            const FloatAB* __restrict__ p_a_grid,
            const FloatAB* __restrict__ p_b_grid,
            FloatDsPointer p_ds_grid,
            FloatE* __restrict__ p_e_grid,
            const index_t batch_count,
            const AElementwiseOperation a_element_op,
            const BElementwiseOperation b_element_op,
            const CDEElementwiseOperation cde_element_op,
            const AGridDesc_AK0_M_AK1 a_grid_desc_ak0_m_ak1,
            const BGridDesc_BK0_N_BK1 b_grid_desc_bk0_n_bk1,
            const DsGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock
                ds_grid_desc_mblock_mperblock_nblock_nperblock,
            const EGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock
                e_grid_desc_mblock_mperblock_nblock_nperblock,
            const ComputePtrOffsetOfBatch compute_ptr_offset_of_batch,
            const Block2ETileMap block_2_etile_map)
{
#if(!defined(__HIP_DEVICE_COMPILE__) || defined(__gfx908__) || defined(__gfx90a__))
    __shared__ char p_shared[GridwiseGemm::GetSharedMemoryNumberOfByte()];

    const index_t num_blocks_per_batch =
        __builtin_amdgcn_readfirstlane(get_grid_size() / batch_count);
    const index_t g_idx = __builtin_amdgcn_readfirstlane(get_block_1d_id() / num_blocks_per_batch);

    const long_index_t a_batch_offset = __builtin_amdgcn_readfirstlane(
        static_cast<long_index_t>(compute_ptr_offset_of_batch.GetAPtrOffset(g_idx)));
    const long_index_t b_batch_offset = __builtin_amdgcn_readfirstlane(
        static_cast<long_index_t>(compute_ptr_offset_of_batch.GetBPtrOffset(g_idx)));
    const long_index_t e_batch_offset = __builtin_amdgcn_readfirstlane(
        static_cast<long_index_t>(compute_ptr_offset_of_batch.GetEPtrOffset(g_idx)));

    const auto ds_batch_offset = compute_ptr_offset_of_batch.GetDsPtrOffset(g_idx);

    FloatDsPointer p_ds_grid_grp;

    static constexpr index_t NumDTensor =
        DsGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock::Size();

    static_for<0, NumDTensor, 1>{}(
        [&](auto i) { p_ds_grid_grp(i) = p_ds_grid[i] + ds_batch_offset[i]; });

    GridwiseGemm::template Run<HasMainKBlockLoop>(p_a_grid + a_batch_offset,
                                                  p_b_grid + b_batch_offset,
                                                  p_ds_grid_grp,
                                                  p_e_grid + e_batch_offset,
                                                  p_shared,
                                                  a_element_op,
                                                  b_element_op,
                                                  cde_element_op,
                                                  a_grid_desc_ak0_m_ak1,
                                                  b_grid_desc_bk0_n_bk1,
                                                  ds_grid_desc_mblock_mperblock_nblock_nperblock,
                                                  e_grid_desc_mblock_mperblock_nblock_nperblock,
                                                  block_2_etile_map);
#else
    ignore = p_a_grid;
    ignore = p_b_grid;
    ignore = p_ds_grid;
    ignore = p_e_grid;
    ignore = batch_count;
    ignore = a_element_op;
    ignore = b_element_op;
    ignore = cde_element_op;
    ignore = a_grid_desc_ak0_m_ak1;
    ignore = b_grid_desc_bk0_n_bk1;
    ignore = ds_grid_desc_mblock_mperblock_nblock_nperblock;
    ignore = e_grid_desc_mblock_mperblock_nblock_nperblock;
    ignore = block_2_etile_map;
    ignore = compute_ptr_offset_of_batch;
#endif
}

} // namespace ck

namespace ck {
namespace tensor_operation {
namespace device {

// Tensor Contraction:
//   input : A
//   input : B
//   input : D0, D1, ...
//   output : E
//   C = a_op(A) * b_op(B)
//   E = cde_op(C, D0, D1, ...)
// Assume:
//   A[G0, G1, ..., M0, M1, M2, ..., K0, K1, K2, ...]
//   B[G0, G1, ..., N0, N1, N2, ..., K0, K1, K2, ...]
//   D[G0, G1, ..., M0, M1, M2, ..., N0, N1, N2, ...]
//   E[G0, G1, ..., M0, M1, M2, ..., N0, N1, N2, ...]

// NOTE: TensorSpecialization::Packed specialized tensor is "packed" in a sense that each inner
// dimension in a dimension group (eg [G0, G1] in Gs, [M0, M1, M2] in Ms, etc.) are contiguous and
// ordered. Not in a sense that the tensor [G0, G1, ..., M0, M1, ..., N0, N1...] can be permuted
// while still being a contiguous, unpadded tensor. In other words, it merely degenerates into
// TensorSpecialization::Default with NumDimG/M/N/K = 1
//
// Detail- Packed tensor satisfies
//   stride_0 = 1
//   stride_i = stride_{i - 1} * extent_{i - 1}
// So tensor
//   [G0, G1, G2, M, N]
// transposed into tensor
//   [G0, G2, G1, M, N]
// with strides
//   [G2 * G1 * M * N, G1 * M * N, M * N, N, 1]
// is again a packed tensor. MakeGridDescriptor() currently just merges dimensions and ignores some
// strides from input tensor extents so finer dimension information is lost. Merging dimensions is
// essentially a degenerated case of TensorSpecialization::Default with NumDimG/M/N/K = 1.
//
// Might need to expose dimension order to the interface to fully support
// TensorSpecialization::Packed in a traditional sense of "packed" tensor
template <index_t NumDimG,
          index_t NumDimM,
          index_t NumDimN,
          index_t NumDimK,
          typename ADataType,
          typename BDataType,
          typename AccDataType,
          typename CShuffleDataType,
          typename DsDataType,
          typename EDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CDEElementwiseOperation,
          GemmSpecialization GemmSpec,
          TensorSpecialization ASpec,
          TensorSpecialization BSpec,
          TensorSpecialization DESpec,
          index_t NumGemmKPrefetchStage,
          index_t BlockSize,
          index_t MPerBlock,
          index_t NPerBlock,
          index_t KPerBlock,
          index_t AK1,
          index_t BK1,
          index_t MPerXDL,
          index_t NPerXDL,
          index_t MXdlPerWave,
          index_t NXdlPerWave,
          typename ABlockTransferThreadClusterLengths_AK0_M_AK1,
          typename ABlockTransferThreadClusterArrangeOrder,
          typename ABlockTransferSrcAccessOrder,
          index_t ABlockTransferSrcVectorDim,
          index_t ABlockTransferSrcScalarPerVector,
          index_t ABlockTransferDstScalarPerVector_AK1,
          bool ABlockLdsExtraM,
          typename BBlockTransferThreadClusterLengths_BK0_N_BK1,
          typename BBlockTransferThreadClusterArrangeOrder,
          typename BBlockTransferSrcAccessOrder,
          index_t BBlockTransferSrcVectorDim,
          index_t BBlockTransferSrcScalarPerVector,
          index_t BBlockTransferDstScalarPerVector_BK1,
          bool BBlockLdsExtraN,
          index_t CShuffleMXdlPerWavePerShuffle,
          index_t CShuffleNXdlPerWavePerShuffle,
          typename CDEBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          index_t CDEBlockTransferScalarPerVector_NPerBlock,
          LoopScheduler LoopSched = make_default_loop_scheduler()>
struct DeviceBatchedContractionMultipleD_Xdl_CShuffle
    : public DeviceBatchedContractionMultipleD<NumDimG,
                                               NumDimM,
                                               NumDimN,
                                               NumDimK,
                                               ADataType,
                                               BDataType,
                                               DsDataType,
                                               EDataType,
                                               AElementwiseOperation,
                                               BElementwiseOperation,
                                               CDEElementwiseOperation>
{
    using DeviceOp = DeviceBatchedContractionMultipleD_Xdl_CShuffle;

    static constexpr index_t NumDTensor = DsDataType::Size();

    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};

    static constexpr auto matrix_padder =
        MatrixPadder<GemmSpec, index_t, index_t, index_t>{MPerBlock, NPerBlock, KPerBlock};

    // Assume: A[G0, G1, ..., M0, M1, M2, ..., K0, K1, K2, ...]
    static auto MakeAGridDescriptor_M_K(const std::vector<index_t>& a_gs_ms_ks_lengths_vec,
                                        const std::vector<index_t>& a_gs_ms_ks_strides_vec)
    {
        assert(a_gs_ms_ks_lengths_vec.size() == NumDimG + NumDimM + NumDimK &&
               a_gs_ms_ks_strides_vec.size() == NumDimG + NumDimM + NumDimK);

        const auto to_tuple = [&](auto& vec, auto start, auto end) {
            return generate_tuple([&](auto i) { return vec[start + i]; }, Number<end - start>{});
        };

        const auto a_ms_ks_lengths = to_tuple(
            a_gs_ms_ks_lengths_vec, Number<NumDimG>{}, Number<NumDimG + NumDimM + NumDimK>{});
        const auto a_ms_ks_strides = to_tuple(
            a_gs_ms_ks_strides_vec, Number<NumDimG>{}, Number<NumDimG + NumDimM + NumDimK>{});

        // dimension Ids for M0, M1, ...
        constexpr auto mDimIds = typename arithmetic_sequence_gen<0, NumDimM, 1>::type{};

        // dimension Ids for K0, K1, ...
        constexpr auto kDimIds =
            typename arithmetic_sequence_gen<NumDimM, NumDimM + NumDimK, 1>::type{};

        // lengths for M0, M1, ...
        const auto mLengths = get_container_subset(a_ms_ks_lengths, mDimIds);

        // lengths for K0, K1, ...
        const auto kLengths = get_container_subset(a_ms_ks_lengths, kDimIds);

        if constexpr(ASpec == TensorSpecialization::Packed)
        {
            auto M = container_reduce(mLengths, math::multiplies{}, Number<1>{});
            auto K = container_reduce(kLengths, math::multiplies{}, Number<1>{});
            const auto a_grid_desc_mraw_kraw = make_naive_tensor_descriptor(
                make_tuple(M, K),
                make_tuple(a_ms_ks_strides[Number<NumDimM - 1>{}],
                           a_ms_ks_strides[Number<NumDimM + NumDimK - 1>{}]));
            return matrix_padder.PadADescriptor_M_K(a_grid_desc_mraw_kraw);
        }
        else
        {
            // naive tensor A[M0, M1, M2, ..., K0, K1, K2...]
            const auto a_grid_desc_ms_ks =
                make_naive_tensor_descriptor(a_ms_ks_lengths, a_ms_ks_strides);

            // transformed tensor A[MRaw = M0 * M1 * M2 * ... , KRaw = K0 * K1 * K2 * ...]
            const auto a_grid_desc_mraw_kraw = transform_tensor_descriptor(
                a_grid_desc_ms_ks,
                make_tuple(make_merge_transform(mLengths), make_merge_transform(kLengths)),
                make_tuple(mDimIds, kDimIds),
                make_tuple(Sequence<0>{}, Sequence<1>{}));

            return matrix_padder.PadADescriptor_M_K(a_grid_desc_mraw_kraw);
        }
    }

    // Assume: B[G0, G1, ..., N0, N1, N2, ..., K0, K1, K2, ...]
    static auto MakeBGridDescriptor_N_K(const std::vector<index_t>& b_gs_ns_ks_lengths_vec,
                                        const std::vector<index_t>& b_gs_ns_ks_strides_vec)
    {
        assert(b_gs_ns_ks_lengths_vec.size() == NumDimG + NumDimN + NumDimK &&
               b_gs_ns_ks_strides_vec.size() == NumDimG + NumDimN + NumDimK);

        const auto to_tuple = [&](auto& vec, auto start, auto end) {
            return generate_tuple([&](auto i) { return vec[start + i]; }, Number<end - start>{});
        };

        const auto b_ns_ks_lengths = to_tuple(
            b_gs_ns_ks_lengths_vec, Number<NumDimG>{}, Number<NumDimG + NumDimN + NumDimK>{});
        const auto b_ns_ks_strides = to_tuple(
            b_gs_ns_ks_strides_vec, Number<NumDimG>{}, Number<NumDimG + NumDimN + NumDimK>{});

        // dimension Ids for N0, N1, ...
        constexpr auto nDimIds = typename arithmetic_sequence_gen<0, NumDimN, 1>::type{};

        // dimension Ids for K0, K1, ...
        constexpr auto kDimIds =
            typename arithmetic_sequence_gen<NumDimN, NumDimN + NumDimK, 1>::type{};

        // lengths for K0, K1, ...
        const auto kLengths = get_container_subset(b_ns_ks_lengths, kDimIds);

        // lengths for N0, N1, ...
        const auto nLengths = get_container_subset(b_ns_ks_lengths, nDimIds);

        if constexpr(BSpec == TensorSpecialization::Packed)
        {
            auto N = container_reduce(nLengths, math::multiplies{}, Number<1>{});
            auto K = container_reduce(kLengths, math::multiplies{}, Number<1>{});
            const auto b_grid_desc_nraw_kraw = make_naive_tensor_descriptor(
                make_tuple(N, K),
                make_tuple(b_ns_ks_strides[Number<NumDimN - 1>{}],
                           b_ns_ks_strides[Number<NumDimN + NumDimK - 1>{}]));
            return matrix_padder.PadBDescriptor_N_K(b_grid_desc_nraw_kraw);
        }
        else
        {
            // naive tensor B[N0, N1, N2, ..., K0, K1, K2, ...]
            const auto b_grid_desc_ns_ks =
                make_naive_tensor_descriptor(b_ns_ks_lengths, b_ns_ks_strides);

            // transformed tensor B[NRaw = N0 * N1 * N2 * ..., KRaw = K0 * K1 * K2 * ...]
            const auto b_grid_desc_nraw_kraw = transform_tensor_descriptor(
                b_grid_desc_ns_ks,
                make_tuple(make_merge_transform(nLengths), make_merge_transform(kLengths)),
                make_tuple(nDimIds, kDimIds),
                make_tuple(Sequence<0>{}, Sequence<1>{}));

            return matrix_padder.PadBDescriptor_N_K(b_grid_desc_nraw_kraw);
        }
    }

    // assume E[G0, G1, ..., M0, M1, M2, ..., N0, N1, N2...]
    static auto MakeEGridDescriptor_M_N(const std::vector<index_t>& e_gs_ms_ns_lengths_vec,
                                        const std::vector<index_t>& e_gs_ms_ns_strides_vec)
    {
        assert(e_gs_ms_ns_lengths_vec.size() == NumDimG + NumDimM + NumDimN &&
               e_gs_ms_ns_strides_vec.size() == NumDimG + NumDimM + NumDimN);

        const auto to_tuple = [&](auto& vec, auto start, auto end) {
            return generate_tuple([&](auto i) { return vec[start + i]; }, Number<end - start>{});
        };

        const auto e_ms_ns_lengths = to_tuple(
            e_gs_ms_ns_lengths_vec, Number<NumDimG>{}, Number<NumDimG + NumDimM + NumDimN>{});
        const auto e_ms_ns_strides = to_tuple(
            e_gs_ms_ns_strides_vec, Number<NumDimG>{}, Number<NumDimG + NumDimM + NumDimN>{});

        // dimension Ids for M0, M1, ...
        constexpr auto mDimIds = typename arithmetic_sequence_gen<0, NumDimM, 1>::type{};

        // dimension Ids for N0, N1, ...
        constexpr auto nDimIds =
            typename arithmetic_sequence_gen<NumDimM, NumDimM + NumDimN, 1>::type{};

        // lengths for M0, M1, ...
        const auto mLengths = get_container_subset(e_ms_ns_lengths, mDimIds);

        // lengths for K0, K1, ...
        const auto nLengths = get_container_subset(e_ms_ns_lengths, nDimIds);

        if constexpr(DESpec == TensorSpecialization::Packed)
        {
            auto M = container_reduce(mLengths, math::multiplies{}, Number<1>{});
            auto N = container_reduce(nLengths, math::multiplies{}, Number<1>{});
            const auto e_grid_desc_mraw_nraw = make_naive_tensor_descriptor(
                make_tuple(M, N),
                make_tuple(e_ms_ns_strides[Number<NumDimM - 1>{}],
                           e_ms_ns_strides[Number<NumDimM + NumDimN - 1>{}]));
            return matrix_padder.PadCDescriptor_M_N(e_grid_desc_mraw_nraw);
        }
        else
        {
            // naive tensor E[M0, M1, M2, ..., N0, N1, N2...]
            const auto e_grid_desc_ms_ns =
                make_naive_tensor_descriptor(e_ms_ns_lengths, e_ms_ns_strides);

            // transformed tensor E[MRaw = M0 * M1 * M2 * ... , NRaw = N0 * N1 * N2 * ...]
            const auto e_grid_desc_mraw_nraw = transform_tensor_descriptor(
                e_grid_desc_ms_ns,
                make_tuple(make_merge_transform(mLengths), make_merge_transform(nLengths)),
                make_tuple(mDimIds, nDimIds),
                make_tuple(Sequence<0>{}, Sequence<1>{}));

            return matrix_padder.PadCDescriptor_M_N(e_grid_desc_mraw_nraw);
        }
    }

    // assume E[G0, G1, ..., M0, M1, M2, ..., N0, N1, N2...]
    static auto MakeEGridDescriptor_G_M_N(const std::vector<index_t>& e_gs_ms_ns_lengths_vec,
                                          const std::vector<index_t>& e_gs_ms_ns_strides_vec)
    {
        assert(e_gs_ms_ns_lengths_vec.size() == NumDimG + NumDimM + NumDimN &&
               e_gs_ms_ns_strides_vec.size() == NumDimG + NumDimM + NumDimN);

        const auto to_tuple = [&](auto& vec, auto start, auto end) {
            return generate_tuple([&](auto i) { return vec[start + i]; }, Number<end - start>{});
        };

        const auto e_gs_ms_ns_lengths =
            to_tuple(e_gs_ms_ns_lengths_vec, Number<0>{}, Number<NumDimG + NumDimM + NumDimN>{});
        const auto e_gs_ms_ns_strides =
            to_tuple(e_gs_ms_ns_strides_vec, Number<0>{}, Number<NumDimG + NumDimM + NumDimN>{});

        // dimension Ids for G0, G1, ...
        constexpr auto gDimIds = typename arithmetic_sequence_gen<0, NumDimG, 1>::type{};

        // dimension Ids for M0, M1, ...
        constexpr auto mDimIds =
            typename arithmetic_sequence_gen<NumDimG, NumDimG + NumDimM, 1>::type{};

        // dimension Ids for N0, N1, ...
        constexpr auto nDimIds = typename arithmetic_sequence_gen<NumDimG + NumDimM,
                                                                  NumDimG + NumDimM + NumDimN,
                                                                  1>::type{};

        // lengths for G0, G1, ...
        const auto gLengths = get_container_subset(e_gs_ms_ns_lengths, gDimIds);

        // lengths for M0, M1, ...
        const auto mLengths = get_container_subset(e_gs_ms_ns_lengths, mDimIds);

        // lengths for K0, K1, ...
        const auto nLengths = get_container_subset(e_gs_ms_ns_lengths, nDimIds);

        if constexpr(DESpec == TensorSpecialization::Packed)
        {
            auto G = container_reduce(gLengths, math::multiplies{}, Number<1>{});
            auto M = container_reduce(mLengths, math::multiplies{}, Number<1>{});
            auto N = container_reduce(nLengths, math::multiplies{}, Number<1>{});
            const auto e_grid_desc_g_mraw_nraw = make_naive_tensor_descriptor(
                make_tuple(G, M, N),
                make_tuple(e_gs_ms_ns_strides[Number<NumDimG - 1>{}],
                           e_gs_ms_ns_strides[Number<NumDimG + NumDimM - 1>{}],
                           e_gs_ms_ns_strides[Number<NumDimG + NumDimM + NumDimN - 1>{}]));
            // return matrix_padder.PadCDescriptor_M_N(e_grid_desc_g_mraw_nraw);
            return e_grid_desc_g_mraw_nraw;
        }
        else
        {
            // naive tensor E[G0, G1, ..., M0, M1, M2, ..., N0, N1, N2...]
            const auto e_grid_desc_gs_ms_ns =
                make_naive_tensor_descriptor(e_gs_ms_ns_lengths, e_gs_ms_ns_strides);

            // transformed tensor E[G = G0 * G1 * ..., MRaw = M0 * M1 * M2 * ... , NRaw = N0 * N1 *
            // N2 * ...]
            const auto e_grid_desc_g_mraw_nraw = transform_tensor_descriptor(
                e_grid_desc_gs_ms_ns,
                make_tuple(make_merge_transform(gLengths),
                           make_merge_transform(mLengths),
                           make_merge_transform(nLengths)),
                make_tuple(gDimIds, mDimIds, nDimIds),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));

            // return matrix_padder.PadCDescriptor_M_N(e_grid_desc_g_mraw_nraw);
            return e_grid_desc_g_mraw_nraw;
        }
    }

    static auto MakeDsGridDescriptor_M_N(
        const std::array<std::vector<index_t>, NumDTensor>& ds_gs_ms_ns_lengths_vec,
        const std::array<std::vector<index_t>, NumDTensor>& ds_gs_ms_ns_strides_vec)
    {
        return generate_tuple(
            [&](auto i) {
                return DeviceOp::MakeEGridDescriptor_M_N(ds_gs_ms_ns_lengths_vec[i],
                                                         ds_gs_ms_ns_strides_vec[i]);
            },
            Number<NumDTensor>{});
    }

    static auto MakeDsGridDescriptor_G_M_N(
        const std::array<std::vector<index_t>, NumDTensor>& ds_gs_ms_ns_lengths_vec,
        const std::array<std::vector<index_t>, NumDTensor>& ds_gs_ms_ns_strides_vec)
    {
        return generate_tuple(
            [&](auto i) {
                return DeviceOp::MakeEGridDescriptor_G_M_N(ds_gs_ms_ns_lengths_vec[i],
                                                           ds_gs_ms_ns_strides_vec[i]);
            },
            Number<NumDTensor>{});
    }

    using AGridDesc_M_K  = decltype(MakeAGridDescriptor_M_K({}, {}));
    using BGridDesc_N_K  = decltype(MakeBGridDescriptor_N_K({}, {}));
    using DsGridDesc_M_N = remove_cvref_t<decltype(MakeDsGridDescriptor_M_N({{}}, {{}}))>;
    using EGridDesc_M_N  = decltype(MakeEGridDescriptor_M_N({}, {}));

    using DsGridDesc_G_M_N = remove_cvref_t<decltype(MakeDsGridDescriptor_G_M_N({}, {}))>;
    using EGridDesc_G_M_N  = decltype(MakeEGridDescriptor_G_M_N({}, {}));

    struct ComputePtrOffsetOfStridedBatch
    {
        ComputePtrOffsetOfStridedBatch(index_t batch_stride_A,
                                       index_t batch_stride_B,
                                       DsGridDesc_G_M_N ds_grid_desc_g_m_n,
                                       EGridDesc_G_M_N e_grid_desc_g_m_n)
            : batch_stride_A_(batch_stride_A),
              batch_stride_B_(batch_stride_B),
              ds_grid_desc_g_m_n_(ds_grid_desc_g_m_n),
              e_grid_desc_g_m_n_(e_grid_desc_g_m_n)
        {
        }

        __host__ __device__ constexpr long_index_t GetAPtrOffset(index_t g_idx) const
        {
            return static_cast<long_index_t>(g_idx) * batch_stride_A_;
        }

        __host__ __device__ constexpr long_index_t GetBPtrOffset(index_t g_idx) const
        {
            return static_cast<long_index_t>(g_idx) * batch_stride_B_;
        }

        __host__ __device__ constexpr auto GetDsPtrOffset(index_t g_idx) const
        {
            std::array<long_index_t, NumDTensor> ds_offset;

            static_for<0, NumDTensor, 1>{}([&](auto i) {
                ds_offset[i] = static_cast<long_index_t>(g_idx) *
                               ds_grid_desc_g_m_n_[i].CalculateOffset(make_multi_index(1, 0, 0));
            });

            return ds_offset;
        }

        __host__ __device__ constexpr long_index_t GetEPtrOffset(index_t g_idx) const
        {
            return static_cast<long_index_t>(g_idx) *
                   e_grid_desc_g_m_n_.CalculateOffset(make_multi_index(1, 0, 0));
        }

        private:
        index_t batch_stride_A_;
        index_t batch_stride_B_;
        DsGridDesc_G_M_N ds_grid_desc_g_m_n_;
        EGridDesc_G_M_N e_grid_desc_g_m_n_;
    };

    // GridwiseGemm
    using GridwiseGemm = GridwiseGemmMultipleD_xdl_cshuffle<
        ADataType, // TODO: distinguish A/B datatype
        AccDataType,
        CShuffleDataType,
        DsDataType,
        EDataType,
        AElementwiseOperation,
        BElementwiseOperation,
        CDEElementwiseOperation,
        InMemoryDataOperationEnum::Set,
        NumGemmKPrefetchStage,
        BlockSize,
        MPerBlock,
        NPerBlock,
        KPerBlock,
        AK1,
        BK1,
        MPerXDL,
        NPerXDL,
        MXdlPerWave,
        NXdlPerWave,
        ABlockTransferThreadClusterLengths_AK0_M_AK1,
        ABlockTransferThreadClusterArrangeOrder,
        ABlockTransferSrcAccessOrder,
        ABlockTransferSrcVectorDim,
        ABlockTransferSrcScalarPerVector,
        ABlockTransferDstScalarPerVector_AK1,
        false,
        ABlockLdsExtraM,
        BBlockTransferThreadClusterLengths_BK0_N_BK1,
        BBlockTransferThreadClusterArrangeOrder,
        BBlockTransferSrcAccessOrder,
        BBlockTransferSrcVectorDim,
        BBlockTransferSrcScalarPerVector,
        BBlockTransferDstScalarPerVector_BK1,
        false,
        BBlockLdsExtraN,
        CShuffleMXdlPerWavePerShuffle,
        CShuffleNXdlPerWavePerShuffle,
        CDEBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
        CDEBlockTransferScalarPerVector_NPerBlock,
        LoopSched>;

    // desc for blockwise copy
    using AGridDesc_AK0_M_AK1                          = remove_cvref_t<decltype(
        GridwiseGemm::MakeDefaultAGridDescriptor_AK0_M_AK1(AGridDesc_M_K{}))>;
    using BGridDesc_BK0_N_BK1                          = remove_cvref_t<decltype(
        GridwiseGemm::MakeDefaultBGridDescriptor_BK0_N_BK1(BGridDesc_N_K{}))>;
    using DsGridDesc_MBlock_MPerBlock_NBlock_NPerBlock = remove_cvref_t<decltype(
        GridwiseGemm::MakeDsGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(DsGridDesc_M_N{}))>;
    using EGridDesc_MBlock_MPerBlock_NBlock_NPerBlock  = remove_cvref_t<decltype(
        GridwiseGemm::MakeEGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(EGridDesc_M_N{}))>;

    // block-to-e-tile map
    using Block2ETileMap =
        remove_cvref_t<decltype(GridwiseGemm::MakeDefaultBlock2ETileMap(EGridDesc_M_N{}))>;

    // Argument
    struct Argument : public BaseArgument
    {
        Argument(const void* p_a_grid,
                 const void* p_b_grid,
                 std::array<const void*, NumDTensor> p_ds_grid,
                 void* p_e_grid,
                 const std::vector<index_t>& a_gs_ms_ns_lengths,
                 const std::vector<index_t>& a_gs_ms_ks_strides,
                 const std::vector<index_t>& b_gs_ns_ks_lengths,
                 const std::vector<index_t>& b_gs_ns_ks_strides,
                 const std::array<std::vector<index_t>, NumDTensor>& ds_gs_ms_ns_lengths,
                 const std::array<std::vector<index_t>, NumDTensor>& ds_gs_ms_ns_strides,
                 const std::vector<index_t>& e_gs_ms_ns_lengths,
                 const std::vector<index_t>& e_gs_ms_ns_strides,
                 AElementwiseOperation a_element_op,
                 BElementwiseOperation b_element_op,
                 CDEElementwiseOperation cde_element_op)
            : p_a_grid_{static_cast<const ADataType*>(p_a_grid)},
              p_b_grid_{static_cast<const BDataType*>(p_b_grid)},
              p_ds_grid_{},
              p_e_grid_{static_cast<EDataType*>(p_e_grid)},
              a_grid_desc_m_k_{
                  DeviceOp::MakeAGridDescriptor_M_K(a_gs_ms_ns_lengths, a_gs_ms_ks_strides)},
              b_grid_desc_n_k_{
                  DeviceOp::MakeBGridDescriptor_N_K(b_gs_ns_ks_lengths, b_gs_ns_ks_strides)},
              ds_grid_desc_m_n_{},
              e_grid_desc_m_n_{
                  DeviceOp::MakeEGridDescriptor_M_N(e_gs_ms_ns_lengths, e_gs_ms_ns_strides)},
              ds_grid_desc_g_m_n_{
                  DeviceOp::MakeDsGridDescriptor_G_M_N(ds_gs_ms_ns_lengths, ds_gs_ms_ns_strides)},
              e_grid_desc_g_m_n_{
                  DeviceOp::MakeEGridDescriptor_G_M_N(e_gs_ms_ns_lengths, e_gs_ms_ns_strides)},
              a_grid_desc_ak0_m_ak1_{
                  GridwiseGemm::MakeDefaultAGridDescriptor_AK0_M_AK1(a_grid_desc_m_k_)},
              b_grid_desc_bk0_n_bk1_{
                  GridwiseGemm::MakeDefaultBGridDescriptor_BK0_N_BK1(b_grid_desc_n_k_)},
              ds_grid_desc_mblock_mperblock_nblock_nperblock_{},
              e_grid_desc_mblock_mperblock_nblock_nperblock_{},
              block_2_etile_map_{GridwiseGemm::MakeDefaultBlock2ETileMap(e_grid_desc_m_n_)},
              a_element_op_{a_element_op},
              b_element_op_{b_element_op},
              cde_element_op_{cde_element_op},
              a_mz_stride_{},
              a_kz_stride_{},
              b_nz_stride_{},
              b_kz_stride_{},
              ds_nz_stride_{},
              e_nz_stride_{},
              a_batch_stride_{a_gs_ms_ks_strides[NumDimG - 1]},
              b_batch_stride_{b_gs_ns_ks_strides[NumDimG - 1]},
              compute_ptr_offset_of_batch_{
                  a_batch_stride_, b_batch_stride_, ds_grid_desc_g_m_n_, e_grid_desc_g_m_n_}
        {
            static_assert(NumDimG > 0 && NumDimM > 0 && NumDimN > 0 && NumDimK > 0, "");

            // populate pointer, batch stride, desc for Ds
            static_for<0, NumDTensor, 1>{}([&](auto i) {
                using DDataType = remove_cvref_t<tuple_element_t<i.value, DsDataType>>;

                // D pointer
                p_ds_grid_(i) = static_cast<const DDataType*>(p_ds_grid[i]);

                // D desc
                ds_grid_desc_m_n_(i) = DeviceOp::MakeEGridDescriptor_M_N(ds_gs_ms_ns_lengths[i],
                                                                         ds_gs_ms_ns_strides[i]);
            });

            // populate desc for Ds/E
            if(GridwiseGemm::CheckValidity(a_grid_desc_m_k_,
                                           b_grid_desc_n_k_,
                                           ds_grid_desc_m_n_,
                                           e_grid_desc_m_n_,
                                           block_2_etile_map_))
            {
                e_grid_desc_mblock_mperblock_nblock_nperblock_ =
                    GridwiseGemm::MakeEGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
                        e_grid_desc_m_n_);

                ds_grid_desc_mblock_mperblock_nblock_nperblock_ =
                    GridwiseGemm::MakeDsGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
                        ds_grid_desc_m_n_);
            }

            // for sanity check of vector memory access
            a_mz_stride_ = a_gs_ms_ks_strides[NumDimG + NumDimM - 1];
            a_kz_stride_ = a_gs_ms_ks_strides[NumDimG + NumDimM + NumDimK - 1];
            b_nz_stride_ = b_gs_ns_ks_strides[NumDimG + NumDimN - 1];
            b_kz_stride_ = b_gs_ns_ks_strides[NumDimG + NumDimN + NumDimK - 1];

            for(index_t i = 0; i < NumDTensor; ++i)
            {
                ds_nz_stride_[i] = ds_gs_ms_ns_strides[i][NumDimG + NumDimM + NumDimN - 1];
            }

            e_nz_stride_ = e_gs_ms_ns_strides[NumDimG + NumDimM + NumDimN - 1];
        }

        void Print() const
        {
            std::cout << "A[M, K]: " << a_grid_desc_m_k_ << std::endl;
            std::cout << "B[N, K]: " << b_grid_desc_n_k_ << std::endl;
            static_for<0, NumDTensor, 1>{}(
                [&](auto i) { std::cout << "Ds[M, N]: " << ds_grid_desc_m_n_[i] << std::endl; });
            std::cout << "E[M, N]: " << e_grid_desc_m_n_ << std::endl;
        }

        //  private:
        // pointers
        const ADataType* p_a_grid_;
        const BDataType* p_b_grid_;
        typename GridwiseGemm::DsGridPointer p_ds_grid_;
        EDataType* p_e_grid_;

        // tensor descriptors for problem definiton
        AGridDesc_M_K a_grid_desc_m_k_;
        BGridDesc_N_K b_grid_desc_n_k_;
        DsGridDesc_M_N ds_grid_desc_m_n_;
        EGridDesc_M_N e_grid_desc_m_n_;

        DsGridDesc_G_M_N ds_grid_desc_g_m_n_;
        EGridDesc_G_M_N e_grid_desc_g_m_n_;

        // tensor descriptors for block/thread-wise copy
        AGridDesc_AK0_M_AK1 a_grid_desc_ak0_m_ak1_;
        BGridDesc_BK0_N_BK1 b_grid_desc_bk0_n_bk1_;
        DsGridDesc_MBlock_MPerBlock_NBlock_NPerBlock
            ds_grid_desc_mblock_mperblock_nblock_nperblock_;
        EGridDesc_MBlock_MPerBlock_NBlock_NPerBlock e_grid_desc_mblock_mperblock_nblock_nperblock_;

        // block-to-e-tile map
        Block2ETileMap block_2_etile_map_;

        // element-wise op
        AElementwiseOperation a_element_op_;
        BElementwiseOperation b_element_op_;
        CDEElementwiseOperation cde_element_op_;

        // Strides for the last M/N/K dimensions of A/B/Ds/E
        //   for sanity check of vector load/store
        index_t a_mz_stride_;
        index_t a_kz_stride_;
        index_t b_nz_stride_;
        index_t b_kz_stride_;
        std::array<index_t, NumDTensor> ds_nz_stride_;
        index_t e_mz_stride_;
        index_t e_nz_stride_;

        index_t a_batch_stride_;
        index_t b_batch_stride_;

        ComputePtrOffsetOfStridedBatch compute_ptr_offset_of_batch_;
    };

    // Invoker
    struct Invoker : public BaseInvoker
    {
        using Argument = DeviceOp::Argument;

        float Run(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
            if(!GridwiseGemm::CheckValidity(arg.a_grid_desc_m_k_,
                                            arg.b_grid_desc_n_k_,
                                            arg.ds_grid_desc_m_n_,
                                            arg.e_grid_desc_m_n_,
                                            arg.block_2_etile_map_))
            {
                throw std::runtime_error(
                    "wrong! GridwiseGemmMultipleD_xdl_cshuffle has invalid setting");
            }

            const index_t G = arg.e_grid_desc_g_m_n_.GetLength(I0);

            const index_t grid_size =
                arg.block_2_etile_map_.CalculateGridSize(arg.e_grid_desc_m_n_) * G;

            const auto K =
                arg.a_grid_desc_ak0_m_ak1_.GetLength(I0) * arg.a_grid_desc_ak0_m_ak1_.GetLength(I2);

            auto launch_kernel = [&](auto has_main_k_block_loop) {
                constexpr bool has_main_loop = has_main_k_block_loop.value;

                const auto kernel = kernel_contraction_multiple_d_xdl_cshuffle<
                    GridwiseGemm,
                    ADataType, // TODO: distiguish A/B datatype
                    typename GridwiseGemm::DsGridPointer,
                    EDataType,
                    AElementwiseOperation,
                    BElementwiseOperation,
                    CDEElementwiseOperation,
                    DeviceOp::AGridDesc_AK0_M_AK1,
                    DeviceOp::BGridDesc_BK0_N_BK1,
                    DeviceOp::DsGridDesc_MBlock_MPerBlock_NBlock_NPerBlock,
                    DeviceOp::EGridDesc_MBlock_MPerBlock_NBlock_NPerBlock,
                    ComputePtrOffsetOfStridedBatch,
                    DeviceOp::Block2ETileMap,
                    has_main_loop>;

                return launch_and_time_kernel(stream_config,
                                              kernel,
                                              dim3(grid_size),
                                              dim3(BlockSize),
                                              0,
                                              arg.p_a_grid_,
                                              arg.p_b_grid_,
                                              arg.p_ds_grid_,
                                              arg.p_e_grid_,
                                              G,
                                              arg.a_element_op_,
                                              arg.b_element_op_,
                                              arg.cde_element_op_,
                                              arg.a_grid_desc_ak0_m_ak1_,
                                              arg.b_grid_desc_bk0_n_bk1_,
                                              arg.ds_grid_desc_mblock_mperblock_nblock_nperblock_,
                                              arg.e_grid_desc_mblock_mperblock_nblock_nperblock_,
                                              arg.compute_ptr_offset_of_batch_,
                                              arg.block_2_etile_map_);
            };

            if(GridwiseGemm::CalculateHasMainKBlockLoop(K))
            {
                return launch_kernel(integral_constant<bool, true>{});
            }
            else
            {
                return launch_kernel(integral_constant<bool, false>{});
            }
        }

        // polymorphic
        float Run(const BaseArgument* p_arg,
                  const StreamConfig& stream_config = StreamConfig{}) override
        {
            return Run(*dynamic_cast<const Argument*>(p_arg), stream_config);
        }
    };

    static bool IsSupportedArgument(const Argument& arg)
    {
        if(!(ck::get_device_name() == "gfx908" || ck::get_device_name() == "gfx90a"))
        {
            return false;
        }

        if(!GridwiseGemm::CheckValidity(arg.a_grid_desc_m_k_,
                                        arg.b_grid_desc_n_k_,
                                        arg.ds_grid_desc_m_n_,
                                        arg.e_grid_desc_m_n_,
                                        arg.block_2_etile_map_))
        {
            return false;
        }

        // check vector access
        static_assert((ABlockTransferSrcVectorDim == 1 || ABlockTransferSrcVectorDim == 2) &&
                          (BBlockTransferSrcVectorDim == 1 || BBlockTransferSrcVectorDim == 2),
                      "wrong!");

        // vector memory access of A: could be on M or AK1 dimension
        if constexpr(ABlockTransferSrcVectorDim == 1)
        {
            if(!(arg.a_mz_stride_ == 1 &&
                 arg.a_grid_desc_ak0_m_ak1_.GetLength(I1) % ABlockTransferSrcScalarPerVector == 0))
            {
                return false;
            }
        }
        else
        {
            if(!(arg.a_kz_stride_ == 1 &&
                 arg.a_grid_desc_ak0_m_ak1_.GetLength(I2) % ABlockTransferSrcScalarPerVector == 0))
            {
                return false;
            }
        }

        // vector memory access of B: could be on N or BK1 dimension
        if constexpr(BBlockTransferSrcVectorDim == 1)
        {
            if(!(arg.b_nz_stride_ == 1 &&
                 arg.b_grid_desc_bk0_n_bk1_.GetLength(I1) % BBlockTransferSrcScalarPerVector == 0))
            {
                return false;
            }
        }
        else
        {
            if(!(arg.b_kz_stride_ == 1 &&
                 arg.b_grid_desc_bk0_n_bk1_.GetLength(I2) % BBlockTransferSrcScalarPerVector == 0))
            {
                return false;
            }
        }

        // vector memory access of Ds: always on NPerBlock dimension
        bool valid_d_access = true;

        static_for<0, NumDTensor, 1>{}([&](auto i) {
            if(!(arg.ds_nz_stride_[i] == 1 &&
                 arg.ds_grid_desc_mblock_mperblock_nblock_nperblock_[i].GetLength(I3) %
                         CDEBlockTransferScalarPerVector_NPerBlock ==
                     0))
            {
                valid_d_access = false;
            }
        });

        if(valid_d_access == false)
        {
            return false;
        }

        // vector memory access of E: always on NPerBlock dimension
        if(!((arg.e_nz_stride_ == 1 &&
              arg.e_grid_desc_mblock_mperblock_nblock_nperblock_.GetLength(I3) %
                      CDEBlockTransferScalarPerVector_NPerBlock ==
                  0) ||
             CDEBlockTransferScalarPerVector_NPerBlock == 1))
        {
            return false;
        }

        return true;
    }

    // polymorphic
    bool IsSupportedArgument(const BaseArgument* p_arg) override
    {
        return IsSupportedArgument(*dynamic_cast<const Argument*>(p_arg));
    }

    static auto
    MakeArgument(const void* p_a,
                 const void* p_b,
                 std::array<const void*, NumDTensor> p_ds,
                 void* p_e,
                 const std::vector<index_t>& a_gs_ms_ns_lengths,
                 const std::vector<index_t>& a_gs_ms_ks_strides,
                 const std::vector<index_t>& b_gs_ns_ks_lengths,
                 const std::vector<index_t>& b_gs_ns_ks_strides,
                 const std::array<std::vector<index_t>, NumDTensor>& ds_gs_ms_ns_lengths,
                 const std::array<std::vector<index_t>, NumDTensor>& ds_gs_ms_ns_strides,
                 const std::vector<index_t>& e_gs_ms_ns_lengths,
                 const std::vector<index_t>& e_gs_ms_ns_strides,
                 AElementwiseOperation a_element_op,
                 BElementwiseOperation b_element_op,
                 CDEElementwiseOperation cde_element_op)
    {
        return Argument{p_a,
                        p_b,
                        p_ds,
                        p_e,
                        a_gs_ms_ns_lengths,
                        a_gs_ms_ks_strides,
                        b_gs_ns_ks_lengths,
                        b_gs_ns_ks_strides,
                        ds_gs_ms_ns_lengths,
                        ds_gs_ms_ns_strides,
                        e_gs_ms_ns_lengths,
                        e_gs_ms_ns_strides,
                        a_element_op,
                        b_element_op,
                        cde_element_op};
    }

    static auto MakeInvoker() { return Invoker{}; }

    // polymorphic
    std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const void* p_a,
                        const void* p_b,
                        std::array<const void*, NumDTensor> p_ds,
                        void* p_e,
                        const std::vector<index_t>& a_gs_ms_ns_lengths,
                        const std::vector<index_t>& a_gs_ms_ks_strides,
                        const std::vector<index_t>& b_gs_ns_ks_lengths,
                        const std::vector<index_t>& b_gs_ns_ks_strides,
                        const std::array<std::vector<index_t>, NumDTensor>& ds_gs_ms_ns_lengths,
                        const std::array<std::vector<index_t>, NumDTensor>& ds_gs_ms_ns_strides,
                        const std::vector<index_t>& e_gs_ms_ns_lengths,
                        const std::vector<index_t>& e_gs_ms_ns_strides,
                        AElementwiseOperation a_element_op,
                        BElementwiseOperation b_element_op,
                        CDEElementwiseOperation cde_element_op) override
    {
        return std::make_unique<Argument>(p_a,
                                          p_b,
                                          p_ds,
                                          p_e,
                                          a_gs_ms_ns_lengths,
                                          a_gs_ms_ks_strides,
                                          b_gs_ns_ks_lengths,
                                          b_gs_ns_ks_strides,
                                          ds_gs_ms_ns_lengths,
                                          ds_gs_ms_ns_strides,
                                          e_gs_ms_ns_lengths,
                                          e_gs_ms_ns_strides,
                                          a_element_op,
                                          b_element_op,
                                          cde_element_op);
    }

    // polymorphic
    std::unique_ptr<BaseInvoker> MakeInvokerPointer() override
    {
        return std::make_unique<Invoker>(Invoker{});
    }

    // polymorphic
    std::string GetTypeString() const override
    {
        auto str = std::stringstream();

        // clang-format off
        str << "DeviceBatchedContractionMultipleD_Xdl_CShuffle"
            << "<"
            << NumDimG << ", "
            << NumDimM << ", "
            << NumDimN << ", "
            << NumDimK << ", "
            << BlockSize << ", "
            << MPerBlock << ", "
            << NPerBlock << ", "
            << KPerBlock << ", "
            << AK1 << ", "
            << BK1 << ", "
            << ABlockTransferSrcVectorDim << ", "
            << BBlockTransferSrcVectorDim
            << ">";
        // clang-format on

        return str.str();
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
