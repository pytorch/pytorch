// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>
#include <sstream>

#include "ck/utility/common_header.hpp"
#include "ck/utility/philox_rand.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/device/device_batched_gemm_softmax_gemm_permute.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/matrix_padder.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_batched_multihead_attention_forward_xdl_cshuffle.hpp"
#include "ck/tensor_operation/operator_transform/transform_contraction_to_gemm.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

template <typename GridwiseGemm,
          typename FloatAB,
          typename FloatC,
          typename ZDataType,
          typename FloatLSE,
          typename GemmAccDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename AccElementwiseOperation,
          typename B1ElementwiseOperation,
          typename CElementwiseOperation,
          typename AGridDesc_AK0_M_AK1,
          typename BGridDesc_BK0_N_BK1,
          typename B1GridDesc_BK0_N_BK1,
          typename CGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock,
          typename ZGridDescriptor_M0_N0_M1_N1_M2_N2_M3_N3_N4_N5,
          typename LSEGridDescriptor_M,
          typename Block2CTileMap,
          typename ComputeBasePtrOfStridedBatch,
          typename C0MatrixMask,
          bool HasMainKBlockLoop,
          bool IsDropout,
          bool IsLseStoring,
          bool Deterministic>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
    __launch_bounds__(CK_MAX_THREAD_PER_BLOCK, CK_MIN_BLOCK_PER_CU)
#endif
        kernel_batched_multiheadattention_forward_xdl_cshuffle(
            const FloatAB* __restrict__ p_a_grid,
            const FloatAB* __restrict__ p_b_grid,
            const FloatAB* __restrict__ p_b1_grid,
            FloatC* __restrict__ p_c_grid,
            ZDataType* __restrict__ p_z_grid,
            FloatLSE* __restrict__ p_lse_grid,
            const AElementwiseOperation a_element_op,
            const BElementwiseOperation b_element_op,
            const AccElementwiseOperation acc_element_op,
            const B1ElementwiseOperation b1_element_op,
            const CElementwiseOperation c_element_op,
            const AGridDesc_AK0_M_AK1 a_grid_desc_ak0_m_ak1,
            const BGridDesc_BK0_N_BK1 b_grid_desc_bk0_n_bk1,
            const B1GridDesc_BK0_N_BK1 b1_grid_desc_bk0_n_bk1,
            const CGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock
                c_grid_desc_mblock_mperblock_nblock_nperblock,
            const ZGridDescriptor_M0_N0_M1_N1_M2_N2_M3_N3_N4_N5
                z_grid_desc_m0_n0_m1_n1_m2_n2_m3_n3_n4_n5,
            const LSEGridDescriptor_M lse_grid_desc_m,
            const Block2CTileMap block_2_ctile_map,
            const index_t batch_count,
            const index_t mblock,
            const ComputeBasePtrOfStridedBatch compute_base_ptr_of_batch,
            const C0MatrixMask c0_matrix_mask,
            const ushort p_dropout_in_16bits,
            const GemmAccDataType p_dropout_rescale,
            const unsigned long long seed,
            const unsigned long long offset)
{
#if(!defined(__HIP_DEVICE_COMPILE__) || defined(__gfx908__) || defined(__gfx90a__))
    __shared__ char p_shared[GridwiseGemm::GetSharedMemoryNumberOfByte()];
    const index_t num_blocks_per_batch =
        __builtin_amdgcn_readfirstlane(get_grid_size() / batch_count);
    const index_t g_idx = __builtin_amdgcn_readfirstlane(get_block_1d_id() / num_blocks_per_batch);

    const long_index_t a_batch_offset = __builtin_amdgcn_readfirstlane(
        static_cast<long_index_t>(compute_base_ptr_of_batch.GetABasePtr(g_idx)));
    const long_index_t b_batch_offset = __builtin_amdgcn_readfirstlane(
        static_cast<long_index_t>(compute_base_ptr_of_batch.GetBBasePtr(g_idx)));
    const long_index_t b1_batch_offset = __builtin_amdgcn_readfirstlane(
        static_cast<long_index_t>(compute_base_ptr_of_batch.GetB1BasePtr(g_idx)));
    const long_index_t c_batch_offset = __builtin_amdgcn_readfirstlane(
        static_cast<long_index_t>(compute_base_ptr_of_batch.GetCBasePtr(g_idx)));
    const long_index_t z_batch_offset = __builtin_amdgcn_readfirstlane(
        static_cast<long_index_t>(compute_base_ptr_of_batch.GetZBasePtr(g_idx)));
    const long_index_t lse_batch_offset = __builtin_amdgcn_readfirstlane(
        static_cast<long_index_t>(compute_base_ptr_of_batch.GetLSEBasePtr(g_idx)));

    const index_t global_thread_id = get_thread_global_1d_id();
    ck::philox ph(seed, global_thread_id, offset);

    if constexpr(Deterministic)
    {
        for(index_t i = 0; i < mblock; i++)
        {
            GridwiseGemm::template Run<HasMainKBlockLoop, IsDropout, IsLseStoring>(
                p_a_grid + a_batch_offset,
                p_b_grid + b_batch_offset,
                p_b1_grid + b1_batch_offset,
                p_c_grid + c_batch_offset,
                nullptr ? nullptr : p_z_grid + z_batch_offset,
                nullptr ? nullptr : p_lse_grid + lse_batch_offset,
                p_shared,
                a_element_op,
                b_element_op,
                acc_element_op,
                b1_element_op,
                c_element_op,
                a_grid_desc_ak0_m_ak1,
                b_grid_desc_bk0_n_bk1,
                b1_grid_desc_bk0_n_bk1,
                c_grid_desc_mblock_mperblock_nblock_nperblock,
                z_grid_desc_m0_n0_m1_n1_m2_n2_m3_n3_n4_n5,
                lse_grid_desc_m,
                block_2_ctile_map,
                c0_matrix_mask,
                p_dropout_in_16bits,
                p_dropout_rescale,
                ph,
                i);
        }
    }
    else
    {
        GridwiseGemm::template Run<HasMainKBlockLoop, IsDropout, IsLseStoring>(
            p_a_grid + a_batch_offset,
            p_b_grid + b_batch_offset,
            p_b1_grid + b1_batch_offset,
            p_c_grid + c_batch_offset,
            nullptr ? nullptr : p_z_grid + z_batch_offset,
            nullptr ? nullptr : p_lse_grid + lse_batch_offset,
            p_shared,
            a_element_op,
            b_element_op,
            acc_element_op,
            b1_element_op,
            c_element_op,
            a_grid_desc_ak0_m_ak1,
            b_grid_desc_bk0_n_bk1,
            b1_grid_desc_bk0_n_bk1,
            c_grid_desc_mblock_mperblock_nblock_nperblock,
            z_grid_desc_m0_n0_m1_n1_m2_n2_m3_n3_n4_n5,
            lse_grid_desc_m,
            block_2_ctile_map,
            c0_matrix_mask,
            p_dropout_in_16bits,
            p_dropout_rescale,
            ph,
            0);
    }
#else
    ignore = p_a_grid;
    ignore = p_b_grid;
    ignore = p_b1_grid;
    ignore = p_c_grid;
    ignore = a_element_op;
    ignore = b_element_op;
    ignore = acc_element_op;
    ignore = b1_element_op;
    ignore = c_element_op;
    ignore = a_grid_desc_ak0_m_ak1;
    ignore = b_grid_desc_bk0_n_bk1;
    ignore = b1_grid_desc_bk0_n_bk1;
    ignore = c_grid_desc_mblock_mperblock_nblock_nperblock;
    ignore = block_2_ctile_map;
    ignore = batch_count;
    ignore = compute_base_ptr_of_batch;
    ignore = c0_matrix_mask;
#endif // end of if (defined(__gfx908__) || defined(__gfx90a__))
}

// Computes C = A * B0 * B1
//              ^^^^^^ (Acc0)
//              ^^^^^^^^^^^ (Acc1)
template <index_t NumDimG,
          index_t NumDimM,
          index_t NumDimN,
          index_t NumDimK,
          index_t NumDimO, // NumDimGemm1N
          typename ADataType,
          typename BDataType,
          typename B1DataType,
          typename CDataType,
          typename GemmDataType,
          typename ZDataType,
          typename LSEDataType,
          typename Acc0BiasDataType,
          typename Acc1BiasDataType,
          typename GemmAccDataType,
          typename CShuffleDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename AccElementwiseOperation,
          typename B1ElementwiseOperation,
          typename CElementwiseOperation,
          GemmSpecialization GemmSpec,
          TensorSpecialization ASpec,
          TensorSpecialization BSpec,
          TensorSpecialization B1Spec,
          TensorSpecialization CSpec,
          index_t NumGemmKPrefetchStage,
          index_t BlockSize,
          index_t MPerBlock,
          index_t NPerBlock, // Gemm0NPerBlock
          index_t KPerBlock, // Gemm0KPerBlock
          index_t Gemm1NPerBlock,
          index_t Gemm1KPerBlock,
          index_t AK1,
          index_t BK1,
          index_t B1K1,
          index_t MPerXDL,
          index_t NPerXDL,
          index_t MXdlPerWave,
          index_t NXdlPerWave,
          index_t Gemm1NXdlPerWave,
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
          typename B1BlockTransferThreadClusterLengths_BK0_N_BK1,
          typename B1BlockTransferThreadClusterArrangeOrder,
          typename B1BlockTransferSrcAccessOrder,
          index_t B1BlockTransferSrcVectorDim,
          index_t B1BlockTransferSrcScalarPerVector,
          index_t B1BlockTransferDstScalarPerVector_BK1,
          bool B1BlockLdsExtraN,
          index_t CShuffleMXdlPerWavePerShuffle,
          index_t CShuffleNXdlPerWavePerShuffle,
          typename CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          index_t CShuffleBlockTransferScalarPerVector_NPerBlock,
          MaskingSpecialization MaskingSpec,
          bool Deterministic,
          LoopScheduler LoopSched = LoopScheduler::Default>
struct DeviceBatchedMultiheadAttentionForward_Xdl_CShuffle
    : public DeviceBatchedMultiheadAttentionForward<NumDimG,
                                                    NumDimM,
                                                    NumDimN,
                                                    NumDimK,
                                                    NumDimO,
                                                    ADataType,
                                                    BDataType,
                                                    B1DataType,
                                                    CDataType,
                                                    ZDataType,
                                                    LSEDataType,
                                                    Acc0BiasDataType,
                                                    Acc1BiasDataType,
                                                    AElementwiseOperation,
                                                    BElementwiseOperation,
                                                    AccElementwiseOperation,
                                                    B1ElementwiseOperation,
                                                    CElementwiseOperation,
                                                    MaskingSpec>
{
    static_assert(NumDimG > 0 && NumDimM > 0 && NumDimN > 0 && NumDimK > 0 && NumDimO > 0,
                  "Number of dimension must be greater than 0");

    static constexpr index_t NumAcc0Bias = Acc0BiasDataType::Size();
    static constexpr index_t NumAcc1Bias = Acc1BiasDataType::Size();

    // TODO ANT: implement bias combination
    static_assert(NumAcc0Bias == 0 && NumAcc0Bias == 0, "Bias addition is unimplemented");

#if 0
    // TODO ANT: use alias
    static constexpr index_t NumDimGemm0M = NumDimM;
    static constexpr index_t NumDimGemm0N = NumDimN;
    static constexpr index_t NumDimGemm0K = NumDimK;
    static constexpr index_t NumDimGemm1M = NumDimM;
    static constexpr index_t NumDimGemm1N = NumDimO;
    static constexpr index_t NumDimGemm1K = NumDimN;
#endif

    using DeviceOp = DeviceBatchedMultiheadAttentionForward_Xdl_CShuffle;

    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};

    using Transform = TransformBatchedContractionContractionToBatchedGemmGemm<
        Sequence<NumDimG, NumDimM, NumDimN, NumDimK, NumDimO>,
        Sequence<MPerBlock, NPerBlock, KPerBlock, Gemm1NPerBlock>,
        GemmSpec,
        ASpec,
        BSpec,
        B1Spec,
        CSpec>;

    static auto MakeAGridDescriptor_AK0_M_AK1(const std::vector<index_t>& a_gs_ms_ks_lengths_vec,
                                              const std::vector<index_t>& a_gs_ms_ks_strides_vec)
    {
        return Transform::MakeAGridDescriptor_AK0_M_AK1(
            Transform::MakeAGridDescriptor_M_K(a_gs_ms_ks_lengths_vec, a_gs_ms_ks_strides_vec),
            Number<AK1>{});
    }

    static auto MakeBGridDescriptor_BK0_N_BK1(const std::vector<index_t>& b_gs_ns_ks_lengths_vec,
                                              const std::vector<index_t>& b_gs_ns_ks_strides_vec)
    {
        return Transform::MakeB0GridDescriptor_BK0_N_BK1(
            Transform::MakeB0GridDescriptor_N_K(b_gs_ns_ks_lengths_vec, b_gs_ns_ks_strides_vec),
            Number<BK1>{});
    }

    static auto
    MakeB1GridDescriptor_BK0_N_BK1(const std::vector<index_t>& b1_gs_gemm1ns_gemm1ks_lengths_vec,
                                   const std::vector<index_t>& b1_gs_gemm1ns_gemm1ks_strides_vec)
    {
        return Transform::MakeB1GridDescriptor_BK0_N_BK1(
            Transform::MakeB1GridDescriptor_N_K(b1_gs_gemm1ns_gemm1ks_lengths_vec,
                                                b1_gs_gemm1ns_gemm1ks_strides_vec),
            Number<B1K1>{});
    }

    static auto MakeZGridDescriptor_M_N(const std::vector<index_t>& z_gs_ms_ns_lengths_vec,
                                        const std::vector<index_t>& z_gs_ms_ns_strides_vec)
    {
        return Transform::MakeCGridDescriptor_M_N(z_gs_ms_ns_lengths_vec, z_gs_ms_ns_strides_vec);
    }

    static auto MakeLSEGridDescriptor_M(index_t MRaw)
    {
        const auto lse_grid_desc_mraw = make_naive_tensor_descriptor_packed(make_tuple(MRaw));

        const auto M    = math::integer_divide_ceil(MRaw, MPerBlock) * MPerBlock;
        const auto MPad = M - MRaw;

        if constexpr(GemmSpec == GemmSpecialization::MPadding ||
                     GemmSpec == GemmSpecialization::MNPadding ||
                     GemmSpec == GemmSpecialization::MKPadding ||
                     GemmSpec == GemmSpecialization::MNKPadding)
        {
            // pad M
            return transform_tensor_descriptor(lse_grid_desc_mraw,
                                               make_tuple(make_right_pad_transform(MRaw, MPad)),
                                               make_tuple(Sequence<0>{}),
                                               make_tuple(Sequence<0>{}));
        }
        else
        {
            // not pad M
            return lse_grid_desc_mraw;
        }
    }

    using AGridDesc_AK0_M_AK1  = decltype(MakeAGridDescriptor_AK0_M_AK1({}, {}));
    using BGridDesc_BK0_N_BK1  = decltype(MakeBGridDescriptor_BK0_N_BK1({}, {}));
    using B1GridDesc_BK0_N_BK1 = decltype(MakeB1GridDescriptor_BK0_N_BK1({}, {}));
    using CGridDesc_M_N        = decltype(Transform::MakeCGridDescriptor_M_N({}, {}));
    using ZGridDesc_M_N        = decltype(MakeZGridDescriptor_M_N({}, {}));
    using LSEGridDesc_M        = decltype(MakeLSEGridDescriptor_M(1));
    using AGridDesc_G_M_K      = decltype(Transform::MakeAGridDescriptor_G_M_K({}, {}));
    using BGridDesc_G_N_K      = decltype(Transform::MakeB0GridDescriptor_G_N_K({}, {}));
    using B1GridDesc_G_N_K     = decltype(Transform::MakeB1GridDescriptor_G_N_K({}, {}));
    using CGridDesc_G_M_N      = decltype(Transform::MakeCGridDescriptor_G_M_N({}, {}));
    using ZGridDesc_G_M_N      = decltype(Transform::MakeCGridDescriptor_G_M_N({}, {}));

    constexpr static auto make_MaskOutPredicate()
    {
        if constexpr(MaskingSpec == MaskingSpecialization::MaskDisabled)
        {
            return MaskDisabledPredicate{};
        }
        else if constexpr(MaskingSpec == MaskingSpecialization::MaskOutUpperTriangle)
        {
            return MaskOutUpperTrianglePredicate{};
        }
    }
    using C0MatrixMask = C0MatrixMask_impl<decltype(make_MaskOutPredicate())>;

    struct ComputeBasePtrOfStridedBatch
    {
        ComputeBasePtrOfStridedBatch(const AGridDesc_G_M_K& a_grid_desc_g_m_k,
                                     const BGridDesc_G_N_K& b_grid_desc_g_n_k,
                                     const B1GridDesc_G_N_K& b1_grid_desc_g_n_k,
                                     const CGridDesc_G_M_N& c_grid_desc_g_m_n,
                                     const ZGridDesc_G_M_N& z_grid_desc_g_m_n,
                                     index_t BatchStrideLSE)
            : a_grid_desc_g_m_k_(a_grid_desc_g_m_k),
              b_grid_desc_g_n_k_(b_grid_desc_g_n_k),
              b1_grid_desc_g_n_k_(b1_grid_desc_g_n_k),
              c_grid_desc_g_m_n_(c_grid_desc_g_m_n),
              z_grid_desc_g_m_n_(z_grid_desc_g_m_n),
              BatchStrideLSE_(BatchStrideLSE)
        {
        }

        __host__ __device__ constexpr long_index_t GetABasePtr(index_t g_idx) const
        {
            return a_grid_desc_g_m_k_.CalculateOffset(make_multi_index(g_idx, 0, 0));
        }

        __host__ __device__ constexpr long_index_t GetBBasePtr(index_t g_idx) const
        {
            return b_grid_desc_g_n_k_.CalculateOffset(make_multi_index(g_idx, 0, 0));
        }

        __host__ __device__ constexpr long_index_t GetB1BasePtr(index_t g_idx) const
        {
            return b1_grid_desc_g_n_k_.CalculateOffset(make_multi_index(g_idx, 0, 0));
        }

        __host__ __device__ constexpr long_index_t GetCBasePtr(index_t g_idx) const
        {
            return c_grid_desc_g_m_n_.CalculateOffset(make_multi_index(g_idx, 0, 0));
        }

        __host__ __device__ constexpr long_index_t GetZBasePtr(index_t g_idx) const
        {
            return z_grid_desc_g_m_n_.CalculateOffset(make_multi_index(g_idx, 0, 0));
        }

        __host__ __device__ constexpr long_index_t GetLSEBasePtr(index_t g_idx) const
        {
            return g_idx * static_cast<long_index_t>(BatchStrideLSE_);
        }

        private:
        AGridDesc_G_M_K a_grid_desc_g_m_k_;
        BGridDesc_G_N_K b_grid_desc_g_n_k_;
        B1GridDesc_G_N_K b1_grid_desc_g_n_k_;
        CGridDesc_G_M_N c_grid_desc_g_m_n_;
        ZGridDesc_G_M_N z_grid_desc_g_m_n_;
        index_t BatchStrideLSE_;
    };

    // GridwiseGemm
    using GridwiseGemm = GridwiseBatchedMultiheadAttentionForward_Xdl_CShuffle<
        ADataType, // TODO: distinguish A/B datatype
        ZDataType,
        GemmDataType,
        GemmAccDataType,
        CShuffleDataType,
        CDataType,
        LSEDataType,
        AElementwiseOperation,
        BElementwiseOperation,
        AccElementwiseOperation,
        B1ElementwiseOperation,
        CElementwiseOperation,
        InMemoryDataOperationEnum::Set,
        AGridDesc_AK0_M_AK1,
        BGridDesc_BK0_N_BK1,
        B1GridDesc_BK0_N_BK1,
        CGridDesc_M_N,
        ZGridDesc_M_N,
        LSEGridDesc_M,
        NumGemmKPrefetchStage,
        BlockSize,
        MPerBlock,
        NPerBlock,
        KPerBlock,
        Gemm1NPerBlock,
        Gemm1KPerBlock,
        AK1,
        BK1,
        B1K1,
        MPerXDL,
        NPerXDL,
        MXdlPerWave,
        NXdlPerWave,
        Gemm1NXdlPerWave,
        ABlockTransferThreadClusterLengths_AK0_M_AK1,
        ABlockTransferThreadClusterArrangeOrder,
        ABlockTransferSrcAccessOrder,
        ABlockTransferSrcVectorDim,
        ABlockTransferSrcScalarPerVector,
        ABlockTransferDstScalarPerVector_AK1,
        true,
        ABlockLdsExtraM,
        BBlockTransferThreadClusterLengths_BK0_N_BK1,
        BBlockTransferThreadClusterArrangeOrder,
        BBlockTransferSrcAccessOrder,
        BBlockTransferSrcVectorDim,
        BBlockTransferSrcScalarPerVector,
        BBlockTransferDstScalarPerVector_BK1,
        true,
        BBlockLdsExtraN,
        B1BlockTransferThreadClusterLengths_BK0_N_BK1,
        B1BlockTransferThreadClusterArrangeOrder,
        B1BlockTransferSrcAccessOrder,
        B1BlockTransferSrcVectorDim,
        B1BlockTransferSrcScalarPerVector,
        B1BlockTransferDstScalarPerVector_BK1,
        false,
        B1BlockLdsExtraN,
        CShuffleMXdlPerWavePerShuffle,
        CShuffleNXdlPerWavePerShuffle,
        CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
        CShuffleBlockTransferScalarPerVector_NPerBlock,
        LoopSched,
        Transform::matrix_padder.PadN,
        MaskingSpec == MaskingSpecialization::MaskOutUpperTriangle,
        Deterministic>;

    // Argument
    // FIXME: constness
    struct Argument : public BaseArgument
    {
        Argument(
            const ADataType* p_a_grid,
            const BDataType* p_b_grid,
            const B1DataType* p_b1_grid,
            CDataType* p_c_grid,
            ZDataType* p_z_grid,
            LSEDataType* p_lse_grid,
            const std::array<void*, NumAcc0Bias> p_acc0_biases,
            const std::array<void*, NumAcc1Bias> p_acc1_biases,
            const std::vector<index_t>& a_gs_ms_ks_lengths,
            const std::vector<index_t>& a_gs_ms_ks_strides,
            const std::vector<index_t>& b_gs_ns_ks_lengths,
            const std::vector<index_t>& b_gs_ns_ks_strides,
            const std::vector<index_t>& b1_gs_gemm1ns_gemm1ks_lengths, // b1_gs_os_ns_lengths
            const std::vector<index_t>& b1_gs_gemm1ns_gemm1ks_strides, // b1_gs_os_ns_strides
            const std::vector<index_t>& c_gs_ms_gemm1ns_lengths,       // c_gs_ms_os_lengths
            const std::vector<index_t>& c_gs_ms_gemm1ns_strides,       // c_gs_ms_os_strides
            const std::vector<index_t>& z_gs_ms_ns_lengths,
            const std::vector<index_t>& z_gs_ms_ns_strides,
            const std::vector<index_t>& lse_gs_ms_lengths,
            const std::array<std::vector<ck::index_t>, NumAcc0Bias> acc0_biases_gs_ms_ns_lengths,
            const std::array<std::vector<ck::index_t>, NumAcc0Bias> acc0_biases_gs_ms_ns_strides,
            const std::array<std::vector<ck::index_t>, NumAcc1Bias>
                acc1_biases_gs_ms_gemm1ns_lengths, // acc1_biases_gs_ms_os_lengths
            const std::array<std::vector<ck::index_t>, NumAcc1Bias>
                acc1_biases_gs_ms_gemm1ns_strides, // acc1_biases_gs_ms_os_strides
            AElementwiseOperation a_element_op,
            BElementwiseOperation b_element_op,
            AccElementwiseOperation acc_element_op,
            B1ElementwiseOperation b1_element_op,
            CElementwiseOperation c_element_op,
            float p_dropout,
            std::tuple<unsigned long long, unsigned long long> seeds)
            : p_a_grid_{p_a_grid},
              p_b_grid_{p_b_grid},
              p_b1_grid_{p_b1_grid},
              p_c_grid_{p_c_grid},
              p_z_grid_{p_z_grid},
              p_lse_grid_{p_lse_grid},
              a_grid_desc_ak0_m_ak1_{
                  DeviceOp::MakeAGridDescriptor_AK0_M_AK1(a_gs_ms_ks_lengths, a_gs_ms_ks_strides)},
              b_grid_desc_bk0_n_bk1_{
                  DeviceOp::MakeBGridDescriptor_BK0_N_BK1(b_gs_ns_ks_lengths, b_gs_ns_ks_strides)},
              b1_grid_desc_bk0_n_bk1_{DeviceOp::MakeB1GridDescriptor_BK0_N_BK1(
                  b1_gs_gemm1ns_gemm1ks_lengths, b1_gs_gemm1ns_gemm1ks_strides)},
              c_grid_desc_m_n_{Transform::MakeCGridDescriptor_M_N(c_gs_ms_gemm1ns_lengths,
                                                                  c_gs_ms_gemm1ns_strides)},
              z_grid_desc_m_n_{MakeZGridDescriptor_M_N(z_gs_ms_ns_lengths, z_gs_ms_ns_strides)},
              lse_grid_desc_m_{DeviceOp::MakeLSEGridDescriptor_M(lse_gs_ms_lengths[NumDimG])},
              a_grid_desc_g_m_k_{
                  Transform::MakeAGridDescriptor_G_M_K(a_gs_ms_ks_lengths, a_gs_ms_ks_strides)},
              b_grid_desc_g_n_k_{
                  Transform::MakeB0GridDescriptor_G_N_K(b_gs_ns_ks_lengths, b_gs_ns_ks_strides)},
              b1_grid_desc_g_n_k_{Transform::MakeB1GridDescriptor_G_N_K(
                  b1_gs_gemm1ns_gemm1ks_lengths, b1_gs_gemm1ns_gemm1ks_strides)},
              c_grid_desc_g_m_n_{Transform::MakeCGridDescriptor_G_M_N(c_gs_ms_gemm1ns_lengths,
                                                                      c_gs_ms_gemm1ns_strides)},
              z_grid_desc_g_m_n_{
                  Transform::MakeCGridDescriptor_G_M_N(z_gs_ms_ns_lengths, z_gs_ms_ns_strides)},
              c_grid_desc_mblock_mperblock_nblock_nperblock_{},
              block_2_ctile_map_{GridwiseGemm::MakeDefaultBlock2CTileMap(c_grid_desc_m_n_)},
              a_element_op_{a_element_op},
              b_element_op_{b_element_op},
              acc_element_op_{acc_element_op},
              b1_element_op_{b1_element_op},
              c_element_op_{c_element_op},
              c0_matrix_mask_{b_grid_desc_g_n_k_.GetLength(I1)},
              raw_lengths_mz_nz_kz_gemm1nz_{a_gs_ms_ks_lengths[NumDimG + NumDimM - 1],
                                            b_gs_ns_ks_lengths[NumDimG + NumDimN - 1],
                                            b_gs_ns_ks_lengths[NumDimG + NumDimN + NumDimK - 1],
                                            b1_gs_gemm1ns_gemm1ks_lengths[NumDimG + NumDimO - 1]},
              a_mz_kz_strides_{a_gs_ms_ks_strides[NumDimG + NumDimM - 1],
                               a_gs_ms_ks_strides[NumDimG + NumDimM + NumDimK - 1]},
              b_nz_kz_strides_{b_gs_ns_ks_strides[NumDimG + NumDimN - 1],
                               b_gs_ns_ks_strides[NumDimG + NumDimN + NumDimK - 1]},
              b1_nz_kz_strides_{b1_gs_gemm1ns_gemm1ks_strides[NumDimG + NumDimO - 1],
                                b1_gs_gemm1ns_gemm1ks_strides[NumDimG + NumDimO + NumDimN - 1]},
              c_mz_gemm1nz_strides_{c_gs_ms_gemm1ns_strides[NumDimG + NumDimM - 1],
                                    c_gs_ms_gemm1ns_strides[NumDimG + NumDimM + NumDimO - 1]},
              batch_count_{c_grid_desc_g_m_n_.GetLength(I0)},
              compute_base_ptr_of_batch_{
                  a_grid_desc_g_m_k_,
                  b_grid_desc_g_n_k_,
                  b1_grid_desc_g_n_k_,
                  c_grid_desc_g_m_n_,
                  z_grid_desc_g_m_n_,
                  type_convert<index_t>(lse_grid_desc_m_.GetElementSpaceSize())}
        {
            // TODO ANT: implement bias addition
            ignore = p_acc0_biases;
            ignore = p_acc1_biases;
            ignore = acc0_biases_gs_ms_ns_lengths;
            ignore = acc0_biases_gs_ms_ns_strides;
            ignore = acc1_biases_gs_ms_gemm1ns_lengths;
            ignore = acc1_biases_gs_ms_gemm1ns_strides;

            if(GridwiseGemm::CheckValidity(a_grid_desc_ak0_m_ak1_,
                                           b_grid_desc_bk0_n_bk1_,
                                           b1_grid_desc_bk0_n_bk1_,
                                           c_grid_desc_m_n_,
                                           block_2_ctile_map_))
            {
                c_grid_desc_mblock_mperblock_nblock_nperblock_ =
                    GridwiseGemm::MakeCGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
                        c_grid_desc_m_n_);
            }

            is_dropout_          = p_dropout > 0.0; //
            p_dropout_           = 1.f - p_dropout;
            p_dropout_in_16bits_ = uint16_t(std::floor(p_dropout_ * 65535.0));
            p_dropout_           = 1.f / p_dropout_;
            p_dropout_rescale_   = type_convert<GemmAccDataType>(p_dropout_);

            seed_   = std::get<0>(seeds);
            offset_ = std::get<1>(seeds);

            z_grid_desc_m0_n0_m1_n1_m2_n2_m3_n3_n4_n5_ =
                GridwiseGemm::MakeCGridDescriptor_M0_N0_M1_N1_M2_N2_M3_N3_N4_N5(z_grid_desc_m_n_);

            if(p_lse_grid == nullptr)
            {
                is_lse_storing_ = false;
            }
        }

        void Print() const
        {
            std::cout << "a_grid_desc_g_m_k_: " << a_grid_desc_g_m_k_.GetLength(I0) << ", "
                      << a_grid_desc_g_m_k_.GetLength(I1) << ", "
                      << a_grid_desc_g_m_k_.GetLength(I2) << '\n';
            std::cout << "b_grid_desc_g_n_k_: " << b_grid_desc_g_n_k_.GetLength(I0) << ", "
                      << b_grid_desc_g_n_k_.GetLength(I1) << ", "
                      << b_grid_desc_g_n_k_.GetLength(I2) << '\n';
            std::cout << "b1_grid_desc_g_n_k_: " << b1_grid_desc_g_n_k_.GetLength(I0) << ", "
                      << b1_grid_desc_g_n_k_.GetLength(I1) << ", "
                      << b1_grid_desc_g_n_k_.GetLength(I2) << '\n';
            std::cout << "c_grid_desc_g_m_n_: " << c_grid_desc_g_m_n_.GetLength(I0) << ", "
                      << c_grid_desc_g_m_n_.GetLength(I1) << ", "
                      << c_grid_desc_g_m_n_.GetLength(I2) << '\n';
        }

        // pointers
        const ADataType* p_a_grid_;
        const BDataType* p_b_grid_;
        const B1DataType* p_b1_grid_;
        CDataType* p_c_grid_;
        ZDataType* p_z_grid_;
        LSEDataType* p_lse_grid_;

        // tensor descriptor
        AGridDesc_AK0_M_AK1 a_grid_desc_ak0_m_ak1_;
        BGridDesc_BK0_N_BK1 b_grid_desc_bk0_n_bk1_;
        B1GridDesc_BK0_N_BK1 b1_grid_desc_bk0_n_bk1_;
        CGridDesc_M_N c_grid_desc_m_n_;
        ZGridDesc_M_N z_grid_desc_m_n_;
        LSEGridDesc_M lse_grid_desc_m_;

        AGridDesc_G_M_K a_grid_desc_g_m_k_;
        BGridDesc_G_N_K b_grid_desc_g_n_k_;
        B1GridDesc_G_N_K b1_grid_desc_g_n_k_;
        CGridDesc_G_M_N c_grid_desc_g_m_n_;
        ZGridDesc_G_M_N z_grid_desc_g_m_n_;
        typename GridwiseGemm::CGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock
            c_grid_desc_mblock_mperblock_nblock_nperblock_;
        typename GridwiseGemm::ZGridDescriptor_M0_N0_M1_N1_M2_N2_M3_N3_N4_N5
            z_grid_desc_m0_n0_m1_n1_m2_n2_m3_n3_n4_n5_;

        // block-to-c-tile map
        typename GridwiseGemm::DefaultBlock2CTileMap block_2_ctile_map_;

        // element-wise op
        AElementwiseOperation a_element_op_;
        BElementwiseOperation b_element_op_;
        AccElementwiseOperation acc_element_op_;
        B1ElementwiseOperation b1_element_op_;
        CElementwiseOperation c_element_op_;

        // check C0 masking and padding
        C0MatrixMask c0_matrix_mask_;

        // For robust IsSupportedArgument() check
        std::vector<index_t> raw_lengths_mz_nz_kz_gemm1nz_;
        std::vector<index_t> a_mz_kz_strides_;
        std::vector<index_t> b_nz_kz_strides_;
        std::vector<index_t> b1_nz_kz_strides_;
        std::vector<index_t> c_mz_gemm1nz_strides_;

        index_t batch_count_;
        ComputeBasePtrOfStridedBatch compute_base_ptr_of_batch_;

        float p_dropout_;
        ushort p_dropout_in_16bits_;
        GemmAccDataType p_dropout_rescale_;
        unsigned long long seed_;
        unsigned long long offset_;
        bool is_dropout_;

        bool is_lse_storing_ = true;
    };

    // Invoker
    struct Invoker : public BaseInvoker
    {
        using Argument = DeviceOp::Argument;

        float Run(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
            if(!DeviceOp::IsSupportedArgument(arg))
            {
                throw std::runtime_error("wrong! unsupported argument");
            }

            const index_t grid_size =
                (Deterministic ? 1
                               : arg.block_2_ctile_map_.CalculateGridSize(arg.c_grid_desc_m_n_)) *
                arg.batch_count_;

            // Gemm0_K
            const auto K =
                arg.a_grid_desc_ak0_m_ak1_.GetLength(I0) * arg.a_grid_desc_ak0_m_ak1_.GetLength(I2);

            float ave_time = 0;

            auto launch_kernel =
                [&](auto has_main_k_block_loop_, auto is_dropout_, auto is_lse_storing_) {
                    const auto kernel = kernel_batched_multiheadattention_forward_xdl_cshuffle<
                        GridwiseGemm,
                        ADataType, // TODO: distiguish A/B datatype
                        CDataType,
                        ZDataType,
                        LSEDataType,
                        GemmAccDataType,
                        AElementwiseOperation,
                        BElementwiseOperation,
                        AccElementwiseOperation,
                        B1ElementwiseOperation,
                        CElementwiseOperation,
                        DeviceOp::AGridDesc_AK0_M_AK1,
                        DeviceOp::BGridDesc_BK0_N_BK1,
                        DeviceOp::B1GridDesc_BK0_N_BK1,
                        typename GridwiseGemm::CGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock,
                        typename GridwiseGemm::ZGridDescriptor_M0_N0_M1_N1_M2_N2_M3_N3_N4_N5,
                        DeviceOp::LSEGridDesc_M,
                        typename GridwiseGemm::DefaultBlock2CTileMap,
                        ComputeBasePtrOfStridedBatch,
                        C0MatrixMask,
                        has_main_k_block_loop_,
                        is_dropout_,
                        is_lse_storing_,
                        Deterministic>;

                    return launch_and_time_kernel(
                        stream_config,
                        kernel,
                        dim3(grid_size),
                        dim3(BlockSize),
                        0,
                        arg.p_a_grid_,
                        arg.p_b_grid_,
                        arg.p_b1_grid_,
                        arg.p_c_grid_,
                        arg.p_z_grid_,
                        arg.p_lse_grid_,
                        arg.a_element_op_,
                        arg.b_element_op_,
                        arg.acc_element_op_,
                        arg.b1_element_op_,
                        arg.c_element_op_,
                        arg.a_grid_desc_ak0_m_ak1_,
                        arg.b_grid_desc_bk0_n_bk1_,
                        arg.b1_grid_desc_bk0_n_bk1_,
                        arg.c_grid_desc_mblock_mperblock_nblock_nperblock_,
                        arg.z_grid_desc_m0_n0_m1_n1_m2_n2_m3_n3_n4_n5_,
                        arg.lse_grid_desc_m_,
                        arg.block_2_ctile_map_,
                        arg.batch_count_,
                        arg.block_2_ctile_map_.CalculateGridSize(arg.c_grid_desc_m_n_),
                        arg.compute_base_ptr_of_batch_,
                        arg.c0_matrix_mask_,
                        arg.p_dropout_in_16bits_,
                        arg.p_dropout_rescale_,
                        arg.seed_,
                        arg.offset_);
                };

            // Gemm1_K is split into Gemm1_K0/K1 where K1 is known at compile time, so we only need
            // to concern Gemm0's loop
            if(GridwiseGemm::CalculateHasMainKBlockLoop(K))
            {
                if(arg.is_dropout_)
                {
                    if(arg.is_lse_storing_)
                    {
                        ave_time = launch_kernel(integral_constant<bool, true>{},
                                                 integral_constant<bool, true>{},
                                                 integral_constant<bool, true>{});
                    }
                    else
                    {
                        ave_time = launch_kernel(integral_constant<bool, true>{},
                                                 integral_constant<bool, true>{},
                                                 integral_constant<bool, false>{});
                    }
                }
                else
                {
                    if(arg.is_lse_storing_)
                    {
                        ave_time = launch_kernel(integral_constant<bool, true>{},
                                                 integral_constant<bool, false>{},
                                                 integral_constant<bool, true>{});
                    }
                    else
                    {
                        ave_time = launch_kernel(integral_constant<bool, true>{},
                                                 integral_constant<bool, false>{},
                                                 integral_constant<bool, false>{});
                    }
                }
            }
            else
            {
                if(arg.is_dropout_)
                {
                    if(arg.is_lse_storing_)
                    {
                        ave_time = launch_kernel(integral_constant<bool, false>{},
                                                 integral_constant<bool, true>{},
                                                 integral_constant<bool, true>{});
                    }
                    else
                    {
                        ave_time = launch_kernel(integral_constant<bool, false>{},
                                                 integral_constant<bool, true>{},
                                                 integral_constant<bool, false>{});
                    }
                }
                else
                {
                    if(arg.is_lse_storing_)
                    {
                        ave_time = launch_kernel(integral_constant<bool, false>{},
                                                 integral_constant<bool, false>{},
                                                 integral_constant<bool, true>{});
                    }
                    else
                    {
                        ave_time = launch_kernel(integral_constant<bool, false>{},
                                                 integral_constant<bool, false>{},
                                                 integral_constant<bool, false>{});
                    }
                }
            }

            return ave_time;
        }

        // polymorphic
        float Run(const BaseArgument* p_arg,
                  const StreamConfig& stream_config = StreamConfig{}) override
        {
            return Run(*dynamic_cast<const Argument*>(p_arg), stream_config);
        }
    };

    static constexpr bool IsValidCompilationParameter()
    {
        // TODO: properly implement this check
        return true;
    }

    static bool IsSupportedArgument(const Argument& arg)
    {
#if DEBUG_LOG
        arg.Print();
#endif

        if(!(ck::get_device_name() == "gfx908" || ck::get_device_name() == "gfx90a"))
        {
            return false;
        }

        // TODO ANT: Check if tensor specialization & strides mismatch

        // Check if C permute dimension matches GEMM + GEMM shape
        const index_t c_g       = arg.c_grid_desc_g_m_n_.GetLength(I0); // unpadded
        const index_t c_m       = arg.c_grid_desc_m_n_.GetLength(I0);
        const index_t c_gemm1n  = arg.c_grid_desc_m_n_.GetLength(I1);
        const index_t a_m       = arg.a_grid_desc_ak0_m_ak1_.GetLength(I1);
        const index_t b1_gemm1n = arg.b1_grid_desc_bk0_n_bk1_.GetLength(I1);

        if(!(c_g == arg.batch_count_ && c_m == a_m && c_gemm1n == b1_gemm1n))
        {
            return false;
        }

        // Note: we need raw lengths since threadwise copy can not handle vector load when part of
        // vector is out of bounds
        // Note: need lowest dim in Ms/Ns/Ks/Os, not merged M/N/K/O
        const auto MzRaw      = arg.raw_lengths_mz_nz_kz_gemm1nz_[0];
        const auto NzRaw      = arg.raw_lengths_mz_nz_kz_gemm1nz_[1];
        const auto KzRaw      = arg.raw_lengths_mz_nz_kz_gemm1nz_[2];
        const auto Gemm1NzRaw = arg.raw_lengths_mz_nz_kz_gemm1nz_[3];

        // Check scalar per vector requirement
        const auto a_extent_lowest  = ABlockTransferSrcVectorDim == 2 ? KzRaw : MzRaw;
        const auto b_extent_lowest  = BBlockTransferSrcVectorDim == 2 ? KzRaw : NzRaw;
        const auto b1_extent_lowest = B1BlockTransferSrcVectorDim == 2 ? NzRaw : Gemm1NzRaw;
        const auto c_extent_lowest  = Gemm1NzRaw;

        if(!(a_extent_lowest % ABlockTransferSrcScalarPerVector == 0 &&
             b_extent_lowest % BBlockTransferSrcScalarPerVector == 0 &&
             b1_extent_lowest % B1BlockTransferSrcScalarPerVector == 0 &&
             c_extent_lowest % CShuffleBlockTransferScalarPerVector_NPerBlock == 0))
        {
            return false;
        }

        // Check vector load/store requirement
        const auto a_stride_lowest =
            ABlockTransferSrcVectorDim == 2 ? arg.a_mz_kz_strides_[1] : arg.a_mz_kz_strides_[0];
        const auto b_stride_lowest =
            BBlockTransferSrcVectorDim == 2 ? arg.b_nz_kz_strides_[1] : arg.b_nz_kz_strides_[0];
        const auto b1_stride_lowest =
            B1BlockTransferSrcVectorDim == 2 ? arg.b1_nz_kz_strides_[1] : arg.b1_nz_kz_strides_[0];
        const auto c_stride_lowest =
            arg.c_mz_gemm1nz_strides_[1]; // cshuffle assumes lowest dim in Gemm1Ns to be contiguous

        if(!(a_stride_lowest == 1 || b_stride_lowest == 1 || b1_stride_lowest == 1 ||
             c_stride_lowest == 1))
        {
            return false;
        }

        return GridwiseGemm::CheckValidity(arg.a_grid_desc_ak0_m_ak1_,
                                           arg.b_grid_desc_bk0_n_bk1_,
                                           arg.b1_grid_desc_bk0_n_bk1_,
                                           arg.c_grid_desc_m_n_,
                                           arg.block_2_ctile_map_);
    }

    // polymorphic
    bool IsSupportedArgument(const BaseArgument* p_arg) override
    {
        return IsSupportedArgument(*dynamic_cast<const Argument*>(p_arg));
    }

    static auto MakeArgument(
        const ADataType* p_a,
        const BDataType* p_b,
        const B1DataType* p_b1,
        CDataType* p_c,
        ZDataType* p_z,
        LSEDataType* p_lse,
        const std::array<void*, NumAcc0Bias> p_acc0_biases,
        const std::array<void*, NumAcc1Bias> p_acc1_biases,
        const std::vector<index_t>& a_gs_ms_ks_lengths,
        const std::vector<index_t>& a_gs_ms_ks_strides,
        const std::vector<index_t>& b_gs_ns_ks_lengths,
        const std::vector<index_t>& b_gs_ns_ks_strides,
        const std::vector<index_t>& b1_gs_gemm1ns_gemm1ks_lengths, // b1_gs_os_ns_lengths
        const std::vector<index_t>& b1_gs_gemm1ns_gemm1ks_strides, // b1_gs_os_ns_strides
        const std::vector<index_t>& c_gs_ms_gemm1ns_lengths,       // c_gs_ms_os_lengths
        const std::vector<index_t>& c_gs_ms_gemm1ns_strides,       // c_gs_ms_os_strides
        const std::vector<index_t>& z_gs_ms_ns_lengths,
        const std::vector<index_t>& z_gs_ms_ns_strides,
        const std::vector<index_t>& lse_gs_ms_lengths,
        const std::array<std::vector<ck::index_t>, NumAcc0Bias> acc0_biases_gs_ms_ns_lengths,
        const std::array<std::vector<ck::index_t>, NumAcc0Bias> acc0_biases_gs_ms_ns_strides,
        const std::array<std::vector<ck::index_t>, NumAcc1Bias>
            acc1_biases_gs_ms_gemm1ns_lengths, // acc1_biases_gs_ms_os_lengths
        const std::array<std::vector<ck::index_t>, NumAcc1Bias>
            acc1_biases_gs_ms_gemm1ns_strides, // acc1_biases_gs_ms_os_strides
        AElementwiseOperation a_element_op,
        BElementwiseOperation b_element_op,
        AccElementwiseOperation acc_element_op,
        B1ElementwiseOperation b1_element_op,
        CElementwiseOperation c_element_op,
        float p_dropout,
        std::tuple<unsigned long long, unsigned long long> seeds)
    {
        return Argument{p_a,
                        p_b,
                        p_b1,
                        p_c,
                        p_z,
                        p_lse,
                        p_acc0_biases,
                        p_acc1_biases,
                        a_gs_ms_ks_lengths,
                        a_gs_ms_ks_strides,
                        b_gs_ns_ks_lengths,
                        b_gs_ns_ks_strides,
                        b1_gs_gemm1ns_gemm1ks_lengths, // b1_gs_os_ns_lengths
                        b1_gs_gemm1ns_gemm1ks_strides, // b1_gs_os_ns_strides
                        c_gs_ms_gemm1ns_lengths,       // c_gs_ms_os_lengths
                        c_gs_ms_gemm1ns_strides,       // c_gs_ms_os_strides
                        z_gs_ms_ns_lengths,
                        z_gs_ms_ns_strides,
                        lse_gs_ms_lengths,
                        acc0_biases_gs_ms_ns_lengths,
                        acc0_biases_gs_ms_ns_strides,
                        acc1_biases_gs_ms_gemm1ns_lengths, // acc1_biases_gs_ms_os_lengths
                        acc1_biases_gs_ms_gemm1ns_strides, // acc1_biases_gs_ms_os_strides
                        a_element_op,
                        b_element_op,
                        acc_element_op,
                        b1_element_op,
                        c_element_op,
                        p_dropout,
                        seeds};
    }

    static auto MakeInvoker() { return Invoker{}; }

    // polymorphic
    // FIXME: constness
    std::unique_ptr<BaseArgument> MakeArgumentPointer(
        const void* p_a,
        const void* p_b,
        const void* p_b1,
        void* p_c,
        void* p_z,
        void* p_lse,
        const std::array<void*, NumAcc0Bias> p_acc0_biases,
        const std::array<void*, NumAcc1Bias> p_acc1_biases,
        const std::vector<index_t>& a_gs_ms_ks_lengths,
        const std::vector<index_t>& a_gs_ms_ks_strides,
        const std::vector<index_t>& b_gs_ns_ks_lengths,
        const std::vector<index_t>& b_gs_ns_ks_strides,
        const std::vector<index_t>& b1_gs_gemm1ns_gemm1ks_lengths, // b1_gs_os_ns_lengths
        const std::vector<index_t>& b1_gs_gemm1ns_gemm1ks_strides, // b1_gs_os_ns_strides
        const std::vector<index_t>& c_gs_ms_gemm1ns_lengths,       // c_gs_ms_os_lengths
        const std::vector<index_t>& c_gs_ms_gemm1ns_strides,       // c_gs_ms_os_strides
        const std::vector<index_t>& z_gs_ms_ns_lengths,
        const std::vector<index_t>& z_gs_ms_ns_strides,
        const std::vector<index_t>& lse_gs_ms_lengths,
        const std::array<std::vector<ck::index_t>, NumAcc0Bias> acc0_biases_gs_ms_ns_lengths,
        const std::array<std::vector<ck::index_t>, NumAcc0Bias> acc0_biases_gs_ms_ns_strides,
        const std::array<std::vector<ck::index_t>, NumAcc1Bias>
            acc1_biases_gs_ms_gemm1ns_lengths, // acc1_biases_gs_ms_os_lengths
        const std::array<std::vector<ck::index_t>, NumAcc1Bias>
            acc1_biases_gs_ms_gemm1ns_strides, // acc1_biases_gs_ms_os_strides
        AElementwiseOperation a_element_op,
        BElementwiseOperation b_element_op,
        AccElementwiseOperation acc_element_op,
        B1ElementwiseOperation b1_element_op,
        CElementwiseOperation c_element_op,
        float p_dropout,
        std::tuple<unsigned long long, unsigned long long> seeds) override
    {
        return std::make_unique<Argument>(static_cast<const ADataType*>(p_a),
                                          static_cast<const BDataType*>(p_b),
                                          static_cast<const B1DataType*>(p_b1),
                                          static_cast<CDataType*>(p_c),
                                          static_cast<ZDataType*>(p_z),
                                          static_cast<LSEDataType*>(p_lse),
                                          p_acc0_biases, // cast in struct Argument
                                          p_acc1_biases, // cast in struct Argument
                                          a_gs_ms_ks_lengths,
                                          a_gs_ms_ks_strides,
                                          b_gs_ns_ks_lengths,
                                          b_gs_ns_ks_strides,
                                          b1_gs_gemm1ns_gemm1ks_lengths, // b1_gs_os_ns_lengths
                                          b1_gs_gemm1ns_gemm1ks_strides, // b1_gs_os_ns_strides
                                          c_gs_ms_gemm1ns_lengths,       // c_gs_ms_os_lengths
                                          c_gs_ms_gemm1ns_strides,       // c_gs_ms_os_strides
                                          z_gs_ms_ns_lengths,
                                          z_gs_ms_ns_strides,
                                          lse_gs_ms_lengths,
                                          acc0_biases_gs_ms_ns_lengths,
                                          acc0_biases_gs_ms_ns_strides,
                                          acc1_biases_gs_ms_gemm1ns_lengths,
                                          acc1_biases_gs_ms_gemm1ns_strides,
                                          a_element_op,
                                          b_element_op,
                                          acc_element_op,
                                          b1_element_op,
                                          c_element_op,
                                          p_dropout,
                                          seeds);
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
        str << "DeviceBatchedMultiheadAttentionForward_Xdl_CShuffle"
            << "<"
            << BlockSize << ", "
            << MPerBlock << ", "
            << NPerBlock << ", "
            << KPerBlock << ", "
            << AK1 << ", "
            << BK1 << ", "
            << MPerBlock << ", "
            << Gemm1NPerBlock << ", "
            << Gemm1KPerBlock << ", "
            << B1K1 << ", "
            << getGemmSpecializationString(GemmSpec) << ", "
            << "ASpec" << getTensorSpecializationString(ASpec) << ", "
            << "B0Spec" << getTensorSpecializationString(BSpec) << ", "
            << "B1Spec" << getTensorSpecializationString(B1Spec) << ", "
            << "CSpec" << getTensorSpecializationString(CSpec) << ", "
            << getMaskingSpecializationString(MaskingSpec) << ">";
        // clang-format on

        return str.str();
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
