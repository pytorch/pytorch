// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>
#include <sstream>

#include "ck/utility/common_header.hpp"
#include "ck/utility/philox_rand.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
// #include "ck/tensor_operation/gpu/device/device_batched_multihead_attention_backward.hpp" // TODO
#include "ck/tensor_operation/gpu/device/device_base.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/masking_specialization.hpp"
#include "ck/tensor_operation/gpu/device/matrix_padder.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_batched_multihead_attention_backward_xdl_cshuffle_pt2.hpp"
#include "ck/tensor_operation/operator_transform/transform_contraction_to_gemm.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"

#include "ck/library/utility/host_tensor.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

template <typename GridwiseGemm,
          typename GroupKernelArg,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename AccElementwiseOperation,
          typename B1ElementwiseOperation,
          typename CElementwiseOperation,
          bool HasMainKBlockLoop,
          bool Deterministic>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
    __launch_bounds__(CK_MAX_THREAD_PER_BLOCK, /*CK_MIN_BLOCK_PER_CU*/ 1)
#endif
        kernel_grouped_multihead_attention_backward_xdl_cshuffle_v2(
            const void CK_CONSTANT_ADDRESS_SPACE* group_kernel_args,
            const index_t group_count,
            const AElementwiseOperation a_element_op,
            const BElementwiseOperation b_element_op,
            const AccElementwiseOperation acc_element_op,
            const B1ElementwiseOperation b1_element_op,
            const CElementwiseOperation c_element_op,
            const float p_dropout,
            const unsigned long long seed,
            const unsigned long long offset)
{
#if(!defined(__HIP_DEVICE_COMPILE__) || defined(__gfx908__) || defined(__gfx90a__))
    __shared__ char p_shared[GridwiseGemm::GetSharedMemoryNumberOfByte()];
    const index_t block_id = get_block_1d_id();
    const auto arg_ptr     = reinterpret_cast<const GroupKernelArg*>(
        cast_pointer_to_generic_address_space(group_kernel_args));

    index_t left     = 0;
    index_t right    = group_count;
    index_t group_id = index_t((left + right) / 2);

    while(
        (!(block_id >= arg_ptr[group_id].block_start_ && block_id < arg_ptr[group_id].block_end_)))
    {
        if(block_id < arg_ptr[group_id].block_start_)
        {
            right = group_id;
        }
        else
        {
            left = group_id;
        }
        group_id = index_t((left + right) / 2);
    }

    // per-group batch offset
    const index_t num_blocks_per_batch = arg_ptr[group_id].num_blocks_per_batch_;
    const index_t g_idx                = __builtin_amdgcn_readfirstlane(
        (block_id - arg_ptr[group_id].block_start_) / (Deterministic ? 1 : num_blocks_per_batch));

    const long_index_t a_batch_offset = __builtin_amdgcn_readfirstlane(
        static_cast<long_index_t>(arg_ptr[group_id].compute_base_ptr_of_batch_.GetABasePtr(g_idx)));
    const long_index_t b_batch_offset = __builtin_amdgcn_readfirstlane(
        static_cast<long_index_t>(arg_ptr[group_id].compute_base_ptr_of_batch_.GetBBasePtr(g_idx)));
    const long_index_t z_batch_offset = __builtin_amdgcn_readfirstlane(
        static_cast<long_index_t>(arg_ptr[group_id].compute_base_ptr_of_batch_.GetZBasePtr(g_idx)));
    const long_index_t b1_batch_offset = __builtin_amdgcn_readfirstlane(static_cast<long_index_t>(
        arg_ptr[group_id].compute_base_ptr_of_batch_.GetB1BasePtr(g_idx)));
    const long_index_t c_batch_offset  = __builtin_amdgcn_readfirstlane(
        static_cast<long_index_t>(arg_ptr[group_id].compute_base_ptr_of_batch_.GetCBasePtr(g_idx)));
    const long_index_t lse_batch_offset = __builtin_amdgcn_readfirstlane(static_cast<long_index_t>(
        arg_ptr[group_id].compute_base_ptr_of_batch_.GetLSEBasePtr(g_idx)));

    const index_t global_thread_id = get_thread_global_1d_id();
    ck::philox ph(seed, global_thread_id, offset);
    auto z_matrix_ptr =
        (arg_ptr[group_id].p_z_grid_ == nullptr ? nullptr
                                                : arg_ptr[group_id].p_z_grid_ + z_batch_offset);

    if constexpr(Deterministic)
    {
        for(index_t i = 0; i < num_blocks_per_batch; i++)
        {
            GridwiseGemm::template Run<HasMainKBlockLoop>(
                arg_ptr[group_id].p_a_grid_ + a_batch_offset,
                arg_ptr[group_id].p_b_grid_ + b_batch_offset,
                z_matrix_ptr,
                arg_ptr[group_id].p_b1_grid_ + b1_batch_offset,
                arg_ptr[group_id].p_c_grid_ + c_batch_offset,
                arg_ptr[group_id].p_lse_grid_ + lse_batch_offset,
                arg_ptr[group_id].p_ygrad_grid_ + c_batch_offset,
                arg_ptr[group_id].p_qgrad_grid_ + a_batch_offset,
                arg_ptr[group_id].p_kgrad_grid_ + b_batch_offset,
                arg_ptr[group_id].p_vgrad_grid_ + b1_batch_offset,
                p_shared,
                a_element_op,
                b_element_op,
                acc_element_op,
                b1_element_op,
                c_element_op,
                arg_ptr[group_id].a_grid_desc_ak0_m_ak1_,
                arg_ptr[group_id].b_grid_desc_bk0_n_bk1_,
                arg_ptr[group_id].c_grid_desc_m0_n0_m1_n1_m2_n2_m3_n3_n4_n5_,
                arg_ptr[group_id].b1_grid_desc_bk0_n_bk1_,
                arg_ptr[group_id].y_grid_desc_mblock_mperblock_oblock_operblock_,
                arg_ptr[group_id].lse_grid_desc_m_,
                arg_ptr[group_id].vgrad_grid_desc_n_o_,
                arg_ptr[group_id].ygrad_grid_desc_m0_o_m1_,
                arg_ptr[group_id].block_2_ctile_map_,
                arg_ptr[group_id].c0_matrix_mask_,
                p_dropout,
                ph,
                i);
        }
    }
    else
    {
        GridwiseGemm::template Run<HasMainKBlockLoop>(
            arg_ptr[group_id].p_a_grid_ + a_batch_offset,
            arg_ptr[group_id].p_b_grid_ + b_batch_offset,
            z_matrix_ptr,
            arg_ptr[group_id].p_b1_grid_ + b1_batch_offset,
            arg_ptr[group_id].p_c_grid_ + c_batch_offset,
            arg_ptr[group_id].p_lse_grid_ + lse_batch_offset,
            arg_ptr[group_id].p_ygrad_grid_ + c_batch_offset,
            arg_ptr[group_id].p_qgrad_grid_ + a_batch_offset,
            arg_ptr[group_id].p_kgrad_grid_ + b_batch_offset,
            arg_ptr[group_id].p_vgrad_grid_ + b1_batch_offset,
            p_shared,
            a_element_op,
            b_element_op,
            acc_element_op,
            b1_element_op,
            c_element_op,
            arg_ptr[group_id].a_grid_desc_ak0_m_ak1_,
            arg_ptr[group_id].b_grid_desc_bk0_n_bk1_,
            arg_ptr[group_id].c_grid_desc_m0_n0_m1_n1_m2_n2_m3_n3_n4_n5_,
            arg_ptr[group_id].b1_grid_desc_bk0_n_bk1_,
            arg_ptr[group_id].y_grid_desc_mblock_mperblock_oblock_operblock_,
            arg_ptr[group_id].lse_grid_desc_m_,
            arg_ptr[group_id].vgrad_grid_desc_n_o_,
            arg_ptr[group_id].ygrad_grid_desc_m0_o_m1_,
            arg_ptr[group_id].block_2_ctile_map_,
            arg_ptr[group_id].c0_matrix_mask_,
            p_dropout,
            ph,
            0);
    }
#else
    ignore = group_kernel_args;
    ignore = group_count;
    ignore = a_element_op;
    ignore = b_element_op;
    ignore = acc_element_op;
    ignore = b1_element_op;
    ignore = c_element_op;
    ignore = p_dropout;
    ignore = seed;
    ignore = offset;
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
          typename InputDataType,
          typename OutputDataType,
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
          index_t Gemm2NXdlPerWave,
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
struct DeviceGroupedMultiheadAttentionBackward_Xdl_CShuffle_V2
    : public BaseOperator // TODO inherit atten bwd op once API stablizes
{
    static_assert(NumDimG > 0 && NumDimM > 0 && NumDimN > 0 && NumDimK > 0 && NumDimO > 0,
                  "Number of dimension must be greater than 0");

    static constexpr index_t NumAcc0Bias = Acc0BiasDataType::Size();
    static constexpr index_t NumAcc1Bias = Acc1BiasDataType::Size();

    // TODO: implement bias combination
    static_assert(NumAcc0Bias == 0 && NumAcc0Bias == 0, "Bias addition is unimplemented");

    using DeviceOp = DeviceGroupedMultiheadAttentionBackward_Xdl_CShuffle_V2;
    struct ProblemDesc
    {
        std::vector<index_t> a_gs_ms_ks_lengths;
        std::vector<index_t> a_gs_ms_ks_strides;

        std::vector<index_t> b_gs_ns_ks_lengths;
        std::vector<index_t> b_gs_ns_ks_strides;

        std::vector<index_t> z_gs_ms_ns_lengths;
        std::vector<index_t> z_gs_ms_ns_strides;

        std::vector<index_t> b1_gs_gemm1ns_gemm1ks_lengths;
        std::vector<index_t> b1_gs_gemm1ns_gemm1ks_strides;

        std::vector<index_t> c_gs_ms_gemm1ns_lengths;
        std::vector<index_t> c_gs_ms_gemm1ns_strides;

        std::vector<index_t> lse_gs_ms_lengths;
        std::vector<index_t> lse_gs_ms_strides;

        std::vector<std::vector<index_t>> acc0_biases_gs_ms_ns_lengths;
        std::vector<std::vector<index_t>> acc0_biases_gs_ms_ns_strides;

        std::vector<std::vector<index_t>> acc1_biases_gs_ms_os_lengths;
        std::vector<std::vector<index_t>> acc1_biases_gs_ms_os_strides;
    };
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};

    static constexpr index_t Q_K1 = 8;
    static constexpr index_t K_K1 = 8;
    static constexpr index_t V_N1 = 2;

    static constexpr index_t Q_M1 = 2;
    static constexpr index_t K_N1 = 2;
    static constexpr index_t V_O1 = 8;
    static constexpr index_t Y_O1 = 8;
    static constexpr index_t Y_M1 = 2;

    static constexpr auto padder = GemmGemmPadder<GemmSpec,
                                                  Number<MPerBlock>,
                                                  Number<NPerBlock>,
                                                  Number<KPerBlock>,
                                                  Number<Gemm1NPerBlock>>{};

    using Transform = TransformBatchedContractionContractionToBatchedGemmGemm<
        Sequence<NumDimG, NumDimM, NumDimN, NumDimK, NumDimO>,
        Sequence<MPerBlock, NPerBlock, KPerBlock, Gemm1NPerBlock>,
        GemmSpec,
        ASpec,
        BSpec,
        B1Spec,
        CSpec>;

    /*
    Descriptors for inputs:

      Q, K, V, Y, dY, per-row softmax stats

    Descriptors for outputs:

      dQ, dK, dV

    */

    // Q in Gemm A position
    static auto MakeAGridDescriptor_AK0_M_AK1(const std::vector<index_t>& a_gs_ms_ks_lengths_vec,
                                              const std::vector<index_t>& a_gs_ms_ks_strides_vec)
    {
        return Transform::MakeAGridDescriptor_AK0_M_AK1(
            Transform::MakeAGridDescriptor_M_K(a_gs_ms_ks_lengths_vec, a_gs_ms_ks_strides_vec),
            Number<AK1>{});
    }

    // K in Gemm B0 position
    static auto MakeBGridDescriptor_BK0_N_BK1(const std::vector<index_t>& b_gs_ns_ks_lengths_vec,
                                              const std::vector<index_t>& b_gs_ns_ks_strides_vec)
    {
        return Transform::MakeB0GridDescriptor_BK0_N_BK1(
            Transform::MakeB0GridDescriptor_N_K(b_gs_ns_ks_lengths_vec, b_gs_ns_ks_strides_vec),
            Number<BK1>{});
    }

    // V in Gemm B1 position
    static auto
    MakeB1GridDescriptor_BK0_N_BK1(const std::vector<index_t>& b1_gs_gemm1ns_gemm1ks_lengths_vec,
                                   const std::vector<index_t>& b1_gs_gemm1ns_gemm1ks_strides_vec)
    {
        return Transform::MakeB1GridDescriptor_BK0_N_BK1(
            Transform::MakeB1GridDescriptor_N_K(b1_gs_gemm1ns_gemm1ks_lengths_vec,
                                                b1_gs_gemm1ns_gemm1ks_strides_vec),
            Number<B1K1>{});
    }

    //
    // dV = P^T * dY
    //

    // VGrad in Gemm C position
    static auto MakeVGradGridDescriptor_N_O(const std::vector<index_t>& v_gs_os_ns_lengths_vec,
                                            const std::vector<index_t>& v_gs_os_ns_strides_vec)
    {
        // v_gs_os_ns -> vgrad_gs_ns_os. O dims last because output is row-major.
        // Here directly rearrange lengths/strides before constructing tensor descriptor to reduce
        // transformation overhead
        // TODO: This will be much easier when inputs are Gs, Ms, Ns, Os. So there's no need to
        // extract subsequence and shuffle them.
        const index_t num_dims = NumDimG + NumDimN + NumDimO;

        // 0, 1, .. NumDimG - 1
        std::vector<index_t> gs_ids(NumDimG);
        std::iota(gs_ids.begin(), gs_ids.end(), 0);

        // NumDimG, NumDimG + 1, ... NumDimG + NumDimO - 1
        std::vector<index_t> os_ids(NumDimO);
        std::iota(os_ids.begin(), os_ids.end(), NumDimG);

        // NumDimG + NumDimO, NumDimG + NumDimO + 1, ... NumDimG + NumDimO + NumDimN - 1
        std::vector<index_t> ns_ids(NumDimN);
        std::iota(ns_ids.begin(), ns_ids.end(), NumDimG + NumDimO);

        std::vector<index_t> ids_old2new;
        ids_old2new.insert(ids_old2new.end(), gs_ids.begin(), gs_ids.end());
        ids_old2new.insert(ids_old2new.end(), ns_ids.begin(), ns_ids.end());
        ids_old2new.insert(ids_old2new.end(), os_ids.begin(), os_ids.end());

        std::vector<index_t> v_gs_ns_os_lengths_vec(num_dims), v_gs_ns_os_strides_vec(num_dims);
        for(int i = 0; i < num_dims; i++)
        {
            index_t id_new            = ids_old2new[i];
            v_gs_ns_os_lengths_vec[i] = v_gs_os_ns_lengths_vec[id_new];
            v_gs_ns_os_strides_vec[i] = v_gs_os_ns_strides_vec[id_new];
        }

        const auto vgrad_desc_nraw_oraw =
            MakeGridDescriptorPair<NumDimG, NumDimN, NumDimO, TensorSpecialization::Default>(
                v_gs_ns_os_lengths_vec, v_gs_ns_os_strides_vec)
                .second;

        return PadTensorDescriptor(vgrad_desc_nraw_oraw,
                                   make_tuple(NPerBlock, Gemm1NPerBlock),
                                   Sequence<padder.PadN, padder.PadO>{});
    }

    template <typename YGridDesc_M_O>
    static auto MakeYGradGridDescriptor_M0_O_M1(const YGridDesc_M_O& ygrad_grid_desc_m_o)
    {
        const auto M = ygrad_grid_desc_m_o.GetLength(I0);
        const auto O = ygrad_grid_desc_m_o.GetLength(I1);

        const auto Y_M0 = M / Y_M1;

        return transform_tensor_descriptor(
            ygrad_grid_desc_m_o,
            make_tuple(make_unmerge_transform(make_tuple(Y_M0, Y_M1)),
                       make_pass_through_transform(O)),
            make_tuple(Sequence<0>{}, Sequence<1>{}),
            make_tuple(Sequence<0, 2>{}, Sequence<1>{}));
    }
    //
    // dS_i_j = P_i_j .* (dP_i_j - dY_i dot Y_i)
    //

    //
    // dQ = alpha * dS * K
    //

    // QGrad in Gemm C position
    static auto MakeQGradGridDescriptor_M_K(const std::vector<index_t>& q_gs_ms_ks_lengths_vec,
                                            const std::vector<index_t>& q_gs_ms_ks_strides_vec)
    {
        return Transform::MakeCGridDescriptor_M_N(q_gs_ms_ks_lengths_vec, q_gs_ms_ks_strides_vec);
    }

    //
    // dK = alpha * dS^T * Q
    //

    // KGrad in Gemm C position
    static auto MakeKGradGridDescriptor_N_K(const std::vector<index_t>& k_gs_ns_ks_lengths_vec,
                                            const std::vector<index_t>& k_gs_ns_ks_strides_vec)
    {
        return Transform::MakeCGridDescriptor_M_N(k_gs_ns_ks_lengths_vec, k_gs_ns_ks_strides_vec);
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
    using YGridDesc_M_O        = decltype(Transform::MakeCGridDescriptor_M_N({}, {}));
    using LSEGridDesc_M        = decltype(MakeLSEGridDescriptor_M(1));
    using AGridDesc_G_M_K      = decltype(Transform::MakeAGridDescriptor_G_M_K({}, {}));
    using BGridDesc_G_N_K      = decltype(Transform::MakeB0GridDescriptor_G_N_K({}, {}));
    using B1GridDesc_G_N_K     = decltype(Transform::MakeB1GridDescriptor_G_N_K({}, {}));
    using CGridDesc_G_M_N      = decltype(Transform::MakeCGridDescriptor_G_M_N({}, {}));
    using ZGridDesc_G_M_N      = decltype(Transform::MakeCGridDescriptor_G_M_N({}, {}));

    using VGradGridDesc_N_O     = decltype(MakeVGradGridDescriptor_N_O({}, {}));
    using YGradGridDesc_M0_O_M1 = decltype(MakeYGradGridDescriptor_M0_O_M1(YGridDesc_M_O{}));
    using ZGridDesc_M_N         = decltype(MakeZGridDescriptor_M_N({}, {}));

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
                                     const ZGridDesc_G_M_N& z_grid_desc_g_m_n,
                                     const B1GridDesc_G_N_K& b1_grid_desc_g_n_k,
                                     const CGridDesc_G_M_N& c_grid_desc_g_m_n,
                                     index_t BatchStrideLSE)
            : a_grid_desc_g_m_k_(a_grid_desc_g_m_k),
              b_grid_desc_g_n_k_(b_grid_desc_g_n_k),
              z_grid_desc_g_m_n_(z_grid_desc_g_m_n),
              b1_grid_desc_g_n_k_(b1_grid_desc_g_n_k),
              c_grid_desc_g_m_n_(c_grid_desc_g_m_n),
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

        __host__ __device__ constexpr long_index_t GetZBasePtr(index_t g_idx) const
        {
            return z_grid_desc_g_m_n_.CalculateOffset(make_multi_index(g_idx, 0, 0));
        }

        __host__ __device__ constexpr long_index_t GetB1BasePtr(index_t g_idx) const
        {
            return b1_grid_desc_g_n_k_.CalculateOffset(make_multi_index(g_idx, 0, 0));
        }

        __host__ __device__ constexpr long_index_t GetCBasePtr(index_t g_idx) const
        {
            return c_grid_desc_g_m_n_.CalculateOffset(make_multi_index(g_idx, 0, 0));
        }

        __host__ __device__ constexpr long_index_t GetLSEBasePtr(index_t g_idx) const
        {
            return g_idx * static_cast<long_index_t>(BatchStrideLSE_);
        }

        private:
        AGridDesc_G_M_K a_grid_desc_g_m_k_;
        BGridDesc_G_N_K b_grid_desc_g_n_k_;
        ZGridDesc_G_M_N z_grid_desc_g_m_n_;
        B1GridDesc_G_N_K b1_grid_desc_g_n_k_;
        CGridDesc_G_M_N c_grid_desc_g_m_n_;
        index_t BatchStrideLSE_;
    };

    // GridwiseGemm
    using GridwiseGemm = GridwiseBatchedMultiheadAttentionBackward_Xdl_CShuffle_V2<
        InputDataType, // TODO: distinguish A/B datatype
        OutputDataType,
        ZDataType,
        GemmDataType,
        GemmAccDataType,
        CShuffleDataType,
        LSEDataType,
        AElementwiseOperation,
        BElementwiseOperation,
        AccElementwiseOperation,
        B1ElementwiseOperation,
        CElementwiseOperation,
        InMemoryDataOperationEnum::Set,
        AGridDesc_AK0_M_AK1,
        BGridDesc_BK0_N_BK1,
        ZGridDesc_M_N,
        B1GridDesc_BK0_N_BK1,
        YGridDesc_M_O,
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
        Gemm2NXdlPerWave,
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

    using Block2CTileMap = OffsettedBlockToCTileMap<typename GridwiseGemm::DefaultBlock2CTileMap>;

    struct GroupKernelArg
    {
        // pointers
        const InputDataType* p_a_grid_;
        const InputDataType* p_b_grid_;
        ZDataType* p_z_grid_;
        const InputDataType* p_b1_grid_;
        const InputDataType* p_c_grid_;
        const LSEDataType* p_lse_grid_;
        const InputDataType* p_ygrad_grid_;
        OutputDataType* p_qgrad_grid_;
        OutputDataType* p_kgrad_grid_;
        OutputDataType* p_vgrad_grid_;

        // tensor descriptors for block/thread-wise copy
        AGridDesc_AK0_M_AK1 a_grid_desc_ak0_m_ak1_;
        BGridDesc_BK0_N_BK1 b_grid_desc_bk0_n_bk1_;
        ZGridDesc_M_N z_grid_desc_m_n_;
        B1GridDesc_BK0_N_BK1 b1_grid_desc_bk0_n_bk1_;
        YGridDesc_M_O y_grid_desc_m_o_;

        typename GridwiseGemm::YGridDescriptor_MBlock_MPerBlock_OBlock_OPerBlock
            y_grid_desc_mblock_mperblock_oblock_operblock_;
        typename GridwiseGemm::ZGridDescriptor_M0_N0_M1_N1_M2_N2_M3_N3_N4_N5
            c_grid_desc_m0_n0_m1_n1_m2_n2_m3_n3_n4_n5_;
        LSEGridDesc_M lse_grid_desc_m_;
        VGradGridDesc_N_O vgrad_grid_desc_n_o_;
        YGradGridDesc_M0_O_M1 ygrad_grid_desc_m0_o_m1_;
        // block-to-c-tile map
        Block2CTileMap block_2_ctile_map_;
        index_t num_blocks_per_batch_;
        ComputeBasePtrOfStridedBatch compute_base_ptr_of_batch_;

        // check C0 masking and padding
        C0MatrixMask c0_matrix_mask_;
        index_t block_start_, block_end_;
    };

    struct GroupDeviceArg
    {
        // lengths for the last dimensions of overall problem for sanity check of vector load/store
        std::vector<index_t> raw_lengths_mz_nz_kz_gemm1nz_;

        // strides for the last dimensions of each tensor for sanity check of vector load/store
        std::vector<index_t> a_mz_kz_strides_;
        std::vector<index_t> b_nz_kz_strides_;
        std::vector<index_t> b1_nz_kz_strides_;
        std::vector<index_t> c_mz_gemm1nz_strides_;

        // for gridwise gemm check
        CGridDesc_G_M_N c_grid_desc_g_m_n_;

        index_t batch_count_;
    };
    // Argument
    struct Argument : public BaseArgument
    {
        Argument(const std::vector<const void*>& p_As,
                 const std::vector<const void*>& p_Bs,
                 const std::vector<void*>& p_Zs,
                 const std::vector<const void*>& p_B1s,
                 const std::vector<const void*>& p_Cs, // for dS
                 const std::vector<const void*>& p_LSEs,
                 const std::vector<const void*>& p_Ygrads,
                 std::vector<void*>& p_Qgrads,
                 std::vector<void*>& p_Kgrads,
                 std::vector<void*>& p_Vgrads,
                 const std::array<void*, NumAcc0Bias>& p_acc0_biases,
                 const std::array<void*, NumAcc1Bias>& p_acc1_biases,
                 const std::vector<ProblemDesc>& problem_desc_vec,
                 AElementwiseOperation a_element_op,
                 BElementwiseOperation b_element_op,
                 AccElementwiseOperation acc_element_op,
                 B1ElementwiseOperation b1_element_op,
                 CElementwiseOperation c_element_op,
                 float p_drop,
                 std::tuple<unsigned long long, unsigned long long> seeds)
            : a_element_op_{a_element_op},
              b_element_op_{b_element_op},
              acc_element_op_{acc_element_op},
              b1_element_op_{b1_element_op},
              c_element_op_{c_element_op},
              p_dropout_{p_drop}
        {
            seed_   = std::get<0>(seeds);
            offset_ = std::get<1>(seeds);

            group_count_ = ck::type_convert<ck::index_t>(problem_desc_vec.size());

            if(!(group_count_ == ck::type_convert<ck::index_t>(p_As.size()) &&
                 group_count_ == ck::type_convert<ck::index_t>(p_Bs.size()) &&
                 group_count_ == ck::type_convert<ck::index_t>(p_Zs.size()) &&
                 group_count_ == ck::type_convert<ck::index_t>(p_B1s.size()) &&
                 group_count_ == ck::type_convert<ck::index_t>(p_Cs.size()) &&
                 group_count_ == ck::type_convert<ck::index_t>(p_Ygrads.size()) &&
                 group_count_ == ck::type_convert<ck::index_t>(p_Qgrads.size()) &&
                 group_count_ == ck::type_convert<ck::index_t>(p_Kgrads.size()) &&
                 group_count_ == ck::type_convert<ck::index_t>(p_Vgrads.size()) &&
                 group_count_ == ck::type_convert<ck::index_t>(p_LSEs.size())))
            {
                throw std::runtime_error("wrong! group_count_ != p_As/b/b1/c.size");
            }

            if(!(p_acc0_biases.size() == p_acc1_biases.size()))
            {
                throw std::runtime_error("wrong! acc0_bias_vec.size != acc1_bias_vec.size");
            }

            grid_size_ = 0;
            for(index_t i = 0; i < group_count_; i++)
            {
                const auto p_a_grid     = static_cast<const InputDataType*>(p_As[i]);
                const auto p_b_grid     = static_cast<const InputDataType*>(p_Bs[i]);
                auto p_z_grid           = static_cast<ZDataType*>(p_Zs[i]);
                const auto p_b1_grid    = static_cast<const InputDataType*>(p_B1s[i]);
                const auto p_c_grid     = static_cast<const InputDataType*>(p_Cs[i]);
                const auto p_lse_grid   = static_cast<const LSEDataType*>(p_LSEs[i]);
                const auto p_ygrad_grid = static_cast<const InputDataType*>(p_Ygrads[i]);
                auto p_qgrad_grid       = static_cast<OutputDataType*>(p_Qgrads[i]);
                auto p_kgrad_grid       = static_cast<OutputDataType*>(p_Kgrads[i]);
                auto p_vgrad_grid       = static_cast<OutputDataType*>(p_Vgrads[i]);

                const auto& problem_desc = problem_desc_vec[i];

                const auto a_grid_desc_ak0_m_ak1 = DeviceOp::MakeAGridDescriptor_AK0_M_AK1(
                    problem_desc.a_gs_ms_ks_lengths, problem_desc.a_gs_ms_ks_strides);
                const auto b_grid_desc_bk0_n_bk1 = DeviceOp::MakeBGridDescriptor_BK0_N_BK1(
                    problem_desc.b_gs_ns_ks_lengths, problem_desc.b_gs_ns_ks_strides);
                const auto z_grid_desc_m_n = DeviceOp::MakeZGridDescriptor_M_N(
                    problem_desc.z_gs_ms_ns_lengths, problem_desc.z_gs_ms_ns_strides);
                const auto b1_grid_desc_bk0_n_bk1 = DeviceOp::MakeB1GridDescriptor_BK0_N_BK1(
                    problem_desc.b1_gs_gemm1ns_gemm1ks_lengths,
                    problem_desc.b1_gs_gemm1ns_gemm1ks_strides);
                const auto y_grid_desc_m_o = Transform::MakeCGridDescriptor_M_N(
                    problem_desc.c_gs_ms_gemm1ns_lengths, problem_desc.c_gs_ms_gemm1ns_strides);

                const auto lse_grid_desc_m =
                    DeviceOp::MakeLSEGridDescriptor_M(problem_desc.lse_gs_ms_lengths[NumDimG]);
                const auto vgrad_grid_desc_n_o = DeviceOp::MakeVGradGridDescriptor_N_O(
                    problem_desc.b1_gs_gemm1ns_gemm1ks_lengths,
                    problem_desc.b1_gs_gemm1ns_gemm1ks_strides);
                const auto ygrad_grid_desc_m0_o_m1 =
                    DeviceOp::MakeYGradGridDescriptor_M0_O_M1(y_grid_desc_m_o);

                const auto a_grid_desc_g_m_k = Transform::MakeAGridDescriptor_G_M_K(
                    problem_desc.a_gs_ms_ks_lengths, problem_desc.a_gs_ms_ks_strides);
                const auto b_grid_desc_g_n_k = Transform::MakeB0GridDescriptor_G_N_K(
                    problem_desc.b_gs_ns_ks_lengths, problem_desc.b_gs_ns_ks_strides);
                const auto z_grid_desc_g_m_n = Transform::MakeCGridDescriptor_G_M_N(
                    problem_desc.z_gs_ms_ns_lengths, problem_desc.z_gs_ms_ns_strides);
                const auto b1_grid_desc_g_n_k = Transform::MakeB1GridDescriptor_G_N_K(
                    problem_desc.b1_gs_gemm1ns_gemm1ks_lengths,
                    problem_desc.b1_gs_gemm1ns_gemm1ks_strides);
                const auto c_grid_desc_g_m_n = Transform::MakeCGridDescriptor_G_M_N(
                    problem_desc.c_gs_ms_gemm1ns_lengths, problem_desc.c_gs_ms_gemm1ns_strides);
                typename GridwiseGemm::YGridDescriptor_MBlock_MPerBlock_OBlock_OPerBlock
                    y_grid_desc_mblock_mperblock_oblock_operblock;
                typename GridwiseGemm::ZGridDescriptor_M0_N0_M1_N1_M2_N2_M3_N3_N4_N5
                    c_grid_desc_m0_n0_m1_n1_m2_n2_m3_n3_n4_n5;
                const index_t BlockStart     = grid_size_;
                const auto block_2_ctile_map = Block2CTileMap(y_grid_desc_m_o, BlockStart);
                if(GridwiseGemm::CheckValidity(a_grid_desc_ak0_m_ak1,
                                               b_grid_desc_bk0_n_bk1,
                                               b1_grid_desc_bk0_n_bk1,
                                               y_grid_desc_m_o,
                                               block_2_ctile_map))
                {
                    y_grid_desc_mblock_mperblock_oblock_operblock =
                        GridwiseGemm::MakeCGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
                            y_grid_desc_m_o);
                }

                c_grid_desc_m0_n0_m1_n1_m2_n2_m3_n3_n4_n5 =
                    GridwiseGemm::MakeCGridDescriptor_M0_N0_M1_N1_M2_N2_M3_N3_N4_N5(
                        z_grid_desc_m_n);

                const index_t batch_count = c_grid_desc_g_m_n.GetLength(I0);
                const index_t grid_size_grp =
                    (Deterministic ? 1 : block_2_ctile_map.CalculateGridSize(y_grid_desc_m_o)) *
                    batch_count;
                const index_t BlockEnd = grid_size_ + grid_size_grp;
                // batch stride
                const auto compute_base_ptr_of_batch = ComputeBasePtrOfStridedBatch(
                    a_grid_desc_g_m_k,
                    b_grid_desc_g_n_k,
                    z_grid_desc_g_m_n,
                    b1_grid_desc_g_n_k,
                    c_grid_desc_g_m_n,
                    type_convert<index_t>(lse_grid_desc_m.GetElementSpaceSize()));

                // C0 mask
                const auto c0_matrix_mask = C0MatrixMask(b_grid_desc_g_n_k.GetLength(I1));

                grid_size_ += grid_size_grp;

                // for each group, make sure acc0_biases_gs_ms_ns_lengths.size() == NumAcc0Bias and
                // so on
                if(!(problem_desc.acc0_biases_gs_ms_ns_lengths.size() == NumAcc0Bias &&
                     problem_desc.acc0_biases_gs_ms_ns_strides.size() == NumAcc0Bias &&
                     problem_desc.acc1_biases_gs_ms_os_lengths.size() == NumAcc1Bias &&
                     problem_desc.acc1_biases_gs_ms_os_strides.size() == NumAcc1Bias))
                {
                    throw std::runtime_error(
                        "wrong! number of biases in function argument does not "
                        "match that in template argument");
                }

                group_kernel_args_.push_back({p_a_grid,
                                              p_b_grid,
                                              p_z_grid,
                                              p_b1_grid,
                                              p_c_grid,
                                              p_lse_grid,
                                              p_ygrad_grid,
                                              p_qgrad_grid,
                                              p_kgrad_grid,
                                              p_vgrad_grid,
                                              a_grid_desc_ak0_m_ak1,
                                              b_grid_desc_bk0_n_bk1,
                                              z_grid_desc_m_n,
                                              b1_grid_desc_bk0_n_bk1,
                                              y_grid_desc_m_o,
                                              y_grid_desc_mblock_mperblock_oblock_operblock,
                                              c_grid_desc_m0_n0_m1_n1_m2_n2_m3_n3_n4_n5,
                                              lse_grid_desc_m,
                                              vgrad_grid_desc_n_o,
                                              ygrad_grid_desc_m0_o_m1,
                                              block_2_ctile_map,
                                              block_2_ctile_map.CalculateGridSize(y_grid_desc_m_o),
                                              compute_base_ptr_of_batch,
                                              c0_matrix_mask,
                                              BlockStart,
                                              BlockEnd});

                group_device_args_.push_back(
                    {{problem_desc.a_gs_ms_ks_lengths[NumDimG + NumDimM - 1],
                      problem_desc.b_gs_ns_ks_lengths[NumDimG + NumDimN - 1],
                      problem_desc.b_gs_ns_ks_lengths[NumDimG + NumDimN + NumDimK - 1],
                      problem_desc.b1_gs_gemm1ns_gemm1ks_lengths[NumDimG + NumDimO - 1]},
                     {problem_desc.a_gs_ms_ks_strides[NumDimG + NumDimM - 1],
                      problem_desc.a_gs_ms_ks_strides[NumDimG + NumDimM + NumDimK - 1]},
                     {problem_desc.b_gs_ns_ks_strides[NumDimG + NumDimN - 1],
                      problem_desc.b_gs_ns_ks_strides[NumDimG + NumDimN + NumDimK - 1]},
                     {problem_desc.b1_gs_gemm1ns_gemm1ks_strides[NumDimG + NumDimO - 1],
                      problem_desc.b1_gs_gemm1ns_gemm1ks_strides[NumDimG + NumDimO + NumDimN - 1]},
                     {problem_desc.c_gs_ms_gemm1ns_strides[NumDimG + NumDimM - 1],
                      problem_desc.c_gs_ms_gemm1ns_strides[NumDimG + NumDimM + NumDimO - 1]},
                     c_grid_desc_g_m_n,
                     batch_count});
            }
            // TODO: implement bias addition
            // ignore = p_acc0_biases;
            // ignore = p_acc1_biases;
            // ignore = acc0_biases_gs_ms_ns_lengths;
            // ignore = acc0_biases_gs_ms_ns_strides;
            // ignore = acc1_biases_gs_ms_gemm1ns_lengths;
            // ignore = acc1_biases_gs_ms_gemm1ns_strides;
        }

        // element-wise op
        AElementwiseOperation a_element_op_;
        BElementwiseOperation b_element_op_;
        AccElementwiseOperation acc_element_op_;
        B1ElementwiseOperation b1_element_op_;
        CElementwiseOperation c_element_op_;

        float p_dropout_;
        unsigned long long seed_;
        unsigned long long offset_;

        index_t grid_size_;
        index_t group_count_;

        std::vector<GroupKernelArg> group_kernel_args_;
        std::vector<GroupDeviceArg> group_device_args_;
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

            bool all_has_main_k_block_loop  = true;
            bool some_has_main_k_block_loop = false;
            for(index_t i = 0; i < arg.group_count_; i++)
            {
                const auto K = arg.group_kernel_args_[i].a_grid_desc_ak0_m_ak1_.GetLength(I0) *
                               arg.group_kernel_args_[i].a_grid_desc_ak0_m_ak1_.GetLength(I2);
                const bool y = GridwiseGemm::CalculateHasMainKBlockLoop(K);
                all_has_main_k_block_loop &= y;
                some_has_main_k_block_loop |= y;
            }

            hipGetErrorString(hipMemcpy(arg.p_workspace_,
                                        arg.group_kernel_args_.data(),
                                        arg.group_kernel_args_.size() * sizeof(GroupKernelArg),
                                        hipMemcpyHostToDevice));

            float ave_time = 0;

            auto launch_kernel = [&](auto has_main_k_block_loop_) {
                const auto kernel = kernel_grouped_multihead_attention_backward_xdl_cshuffle_v2<
                    GridwiseGemm,
                    GroupKernelArg,
                    AElementwiseOperation,
                    BElementwiseOperation,
                    AccElementwiseOperation,
                    B1ElementwiseOperation,
                    CElementwiseOperation,
                    has_main_k_block_loop_,
                    Deterministic>;

                return launch_and_time_kernel(
                    stream_config,
                    kernel,
                    dim3(arg.grid_size_),
                    dim3(BlockSize),
                    0,
                    cast_pointer_to_constant_address_space(arg.p_workspace_),
                    arg.group_count_,
                    arg.a_element_op_,
                    arg.b_element_op_,
                    arg.acc_element_op_,
                    arg.b1_element_op_,
                    arg.c_element_op_,
                    arg.p_dropout_,
                    arg.seed_,
                    arg.offset_);
            };

            // Gemm1_K is split into Gemm1_K0/K1 where K1 is known at compile time, so we only need
            // to concern Gemm0's loop
            if(all_has_main_k_block_loop)
            {
                ave_time = launch_kernel(integral_constant<bool, true>{});
            }
            else if(!some_has_main_k_block_loop)
            {
                ave_time = launch_kernel(integral_constant<bool, false>{});
            }
            else
            {
                throw std::runtime_error("wrong! all gemm problems have to simultaneously meet "
                                         "has_main_k_block_loop or no_main_k_block_loop");
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
        if(!(ck::get_device_name() == "gfx908" || ck::get_device_name() == "gfx90a"))
        {
            return false;
        }

        for(index_t i = 0; i < arg.group_count_; i++)
        {
            // TODO: Check if tensor specialization & strides mismatch
            const auto& kernel_arg = arg.group_kernel_args_[i];
            const auto& device_arg = arg.group_device_args_[i];
            // Check if C permute dimension matches GEMM + GEMM shape
            const index_t c_g       = device_arg.c_grid_desc_g_m_n_.GetLength(I0); // unpadded
            const index_t c_m       = kernel_arg.y_grid_desc_m_o_.GetLength(I0);
            const index_t c_gemm1n  = kernel_arg.y_grid_desc_m_o_.GetLength(I1);
            const index_t a_m       = kernel_arg.a_grid_desc_ak0_m_ak1_.GetLength(I1);
            const index_t b1_gemm1n = kernel_arg.b1_grid_desc_bk0_n_bk1_.GetLength(I1);

            if(!(c_g == device_arg.batch_count_ && c_m == a_m && c_gemm1n == b1_gemm1n))
            {
                return false;
            }

            // Note: we need raw lengths since threadwise copy can not handle vector load when part
            // of vector is out of bounds Note: need lowest dim in Ms/Ns/Ks/Os, not merged M/N/K/O
            const auto MzRaw      = device_arg.raw_lengths_mz_nz_kz_gemm1nz_[0];
            const auto NzRaw      = device_arg.raw_lengths_mz_nz_kz_gemm1nz_[1];
            const auto KzRaw      = device_arg.raw_lengths_mz_nz_kz_gemm1nz_[2];
            const auto Gemm1NzRaw = device_arg.raw_lengths_mz_nz_kz_gemm1nz_[3];

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
            const auto a_stride_lowest = ABlockTransferSrcVectorDim == 2
                                             ? device_arg.a_mz_kz_strides_[1]
                                             : device_arg.a_mz_kz_strides_[0];
            const auto b_stride_lowest = BBlockTransferSrcVectorDim == 2
                                             ? device_arg.b_nz_kz_strides_[1]
                                             : device_arg.b_nz_kz_strides_[0];
            const auto b1_stride_lowest = B1BlockTransferSrcVectorDim == 2
                                              ? device_arg.b1_nz_kz_strides_[1]
                                              : device_arg.b1_nz_kz_strides_[0];
            const auto c_stride_lowest =
                device_arg.c_mz_gemm1nz_strides_[1]; // cshuffle assumes lowest dim in Gemm1Ns to be
                                                     // contiguous

            if(!(a_stride_lowest == 1 || b_stride_lowest == 1 || b1_stride_lowest == 1 ||
                 c_stride_lowest == 1))
            {
                return false;
            }

            if(!GridwiseGemm::CheckValidity(kernel_arg.a_grid_desc_ak0_m_ak1_,
                                            kernel_arg.b_grid_desc_bk0_n_bk1_,
                                            kernel_arg.b1_grid_desc_bk0_n_bk1_,
                                            kernel_arg.y_grid_desc_m_o_,
                                            kernel_arg.block_2_ctile_map_))
            {
                return false;
            }
        }
        return true;
    }

    // polymorphic
    bool IsSupportedArgument(const BaseArgument* p_arg) override
    {
        return IsSupportedArgument(*dynamic_cast<const Argument*>(p_arg));
    }

    size_t GetWorkSpaceSize(const BaseArgument* p_arg) const override
    {
        return dynamic_cast<const Argument*>(p_arg)->group_count_ * sizeof(GroupKernelArg);
    }

    static auto MakeArgument(const std::vector<const void*>& p_As,
                             const std::vector<const void*>& p_Bs,
                             const std::vector<void*>& p_Zs,
                             const std::vector<const void*>& p_B1s,
                             const std::vector<const void*>& p_Cs, // for dS
                             const std::vector<const void*>& p_LSEs,
                             const std::vector<const void*>& p_Ygrads,
                             std::vector<void*>& p_Qgrads,
                             std::vector<void*>& p_Kgrads,
                             std::vector<void*>& p_Vgrads,
                             const std::array<void*, NumAcc0Bias>& p_acc0_biases,
                             const std::array<void*, NumAcc1Bias>& p_acc1_biases,
                             const std::vector<ProblemDesc>& problem_desc_vec,
                             AElementwiseOperation a_element_op,
                             BElementwiseOperation b_element_op,
                             AccElementwiseOperation acc_element_op,
                             B1ElementwiseOperation b1_element_op,
                             CElementwiseOperation c_element_op,
                             float p_drop,
                             std::tuple<unsigned long long, unsigned long long> seeds)
    {
        return Argument{p_As,
                        p_Bs,
                        p_Zs,
                        p_B1s,
                        p_Cs,
                        p_LSEs,
                        p_Ygrads,
                        p_Qgrads,
                        p_Kgrads,
                        p_Vgrads,
                        p_acc0_biases,
                        p_acc1_biases,
                        problem_desc_vec,
                        a_element_op,
                        b_element_op,
                        acc_element_op,
                        b1_element_op,
                        c_element_op,
                        p_drop,
                        seeds};
    }

    static auto MakeInvoker() { return Invoker{}; }

    // polymorphic
    // FIXME: constness
    std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const std::vector<const void*>& p_As,
                        const std::vector<const void*>& p_Bs,
                        const std::vector<void*>& p_Zs,
                        const std::vector<const void*>& p_B1s,
                        const std::vector<const void*>& p_Cs, // for dS
                        const std::vector<const void*>& p_LSEs,
                        const std::vector<const void*>& p_Ygrads,
                        std::vector<void*>& p_Qgrads,
                        std::vector<void*>& p_Kgrads,
                        std::vector<void*>& p_Vgrads,
                        const std::array<void*, NumAcc0Bias>& p_acc0_biases,
                        const std::array<void*, NumAcc1Bias>& p_acc1_biases,
                        const std::vector<ProblemDesc>& problem_desc_vec,
                        AElementwiseOperation a_element_op,
                        BElementwiseOperation b_element_op,
                        AccElementwiseOperation acc_element_op,
                        B1ElementwiseOperation b1_element_op,
                        CElementwiseOperation c_element_op,
                        float p_drop,
                        std::tuple<unsigned long long, unsigned long long> seeds) // override
    {
        return std::make_unique<Argument>(p_As,
                                          p_Bs,
                                          p_Zs,
                                          p_B1s,
                                          p_Cs,
                                          p_LSEs,
                                          p_Ygrads,
                                          p_Qgrads,
                                          p_Kgrads,
                                          p_Vgrads,
                                          p_acc0_biases, // cast in struct Argument
                                          p_acc1_biases, // cast in struct Argument
                                          problem_desc_vec,
                                          a_element_op,
                                          b_element_op,
                                          acc_element_op,
                                          b1_element_op,
                                          c_element_op,
                                          p_drop,
                                          seeds);
    }

    // polymorphic
    std::unique_ptr<BaseInvoker> MakeInvokerPointer() // override
    {
        return std::make_unique<Invoker>(Invoker{});
    }

    // polymorphic
    std::string GetTypeString() const override
    {
        auto str = std::stringstream();

        // clang-format off
        str << "DeviceGroupedMultiheadAttentionBackward_Xdl_CShuffle_V2"
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
