// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>
#include <sstream>

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_gemm_reduce.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_bias_add_reduce_xdl_cshuffle_v1.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

// Note: inter-wave loop scheduler is rolled out to c-shuffle version first. Becuase non c-shuffle
// version currently has compiler issues with register spill which further causes validation
// failures.
template <typename ALayout,
          typename BLayout,
          typename CLayout,
          typename ADataType,
          typename BDataType,
          typename CDataType,
          typename BiasDataType,
          typename D0DataType,
          typename GemmAccDataType,
          typename CShuffleDataType,
          typename ReduceAccDataType,
          typename ReducePtrsGlobal,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation,
          typename D0ElementwiseOperation,
          typename ReduceOperations,
          typename ReduceInElementwiseOperations,
          typename ReduceAccElementwiseOperations,
          typename ReduceGlobalMemoryDataOperation,
          GemmSpecialization GemmSpec,
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
          typename CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          index_t CShuffleBlockTransferScalarPerVector_NPerBlock,
          typename CReduceThreadClusterLengths_MPerBlock_NPerBlock,
          index_t CReduceThreadLds2VGprCopySrcDstScalarPerVector_NPerBlock,
          index_t CReduceThreadVgpr2GlobalCopySrcDstScalarPerVector_MPerBlock,
          LoopScheduler LoopSched = make_default_loop_scheduler()>
struct DeviceGemmBiasAddReduce_Xdl_CShuffle : public DeviceGemmReduce<1, ReduceOperations::Size()>
{
    using DeviceOp = DeviceGemmBiasAddReduce_Xdl_CShuffle;

    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};

    static auto MakeAGridDescriptor_AK0_M_AK1(index_t MRaw, index_t KRaw, index_t StrideA)
    {
        const auto a_grid_desc_mraw_kraw = [&]() {
            if constexpr(is_same_v<tensor_layout::gemm::RowMajor, ALayout>)
            {
                return make_naive_tensor_descriptor(make_tuple(MRaw, KRaw),
                                                    make_tuple(StrideA, I1));
            }
            else if constexpr(is_same_v<tensor_layout::gemm::ColumnMajor, ALayout>)
            {
                return make_naive_tensor_descriptor(make_tuple(MRaw, KRaw),
                                                    make_tuple(I1, StrideA));
            }
        }();

        const auto M = math::integer_divide_ceil(MRaw, MPerBlock) * MPerBlock;
        const auto K = math::integer_divide_ceil(KRaw, KPerBlock) * KPerBlock;

        const auto MPad = M - MRaw;
        const auto KPad = K - KRaw;

        if constexpr(GemmSpec == GemmSpecialization::MKPadding ||
                     GemmSpec == GemmSpecialization::MNKPadding)
        {
            // pad both M and K
            assert(K % AK1 == 0);

            const auto AK0 = K / AK1;

            const auto a_grid_desc_m_k =
                transform_tensor_descriptor(a_grid_desc_mraw_kraw,
                                            make_tuple(make_right_pad_transform(MRaw, MPad),
                                                       make_right_pad_transform(KRaw, KPad)),
                                            make_tuple(Sequence<0>{}, Sequence<1>{}),
                                            make_tuple(Sequence<0>{}, Sequence<1>{}));

            const auto a_grid_desc_ak0_m_ak1 =
                transform_tensor_descriptor(a_grid_desc_m_k,
                                            make_tuple(make_unmerge_transform(make_tuple(AK0, AK1)),
                                                       make_pass_through_transform(M)),
                                            make_tuple(Sequence<1>{}, Sequence<0>{}),
                                            make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

            return a_grid_desc_ak0_m_ak1;
        }
        else if constexpr(GemmSpec == GemmSpecialization::MPadding ||
                          GemmSpec == GemmSpecialization::MNPadding)
        {
            // pad M, but not K
            assert(KRaw % AK1 == 0);

            const auto AK0 = KRaw / AK1;

            const auto a_grid_desc_ak0_m_ak1 =
                transform_tensor_descriptor(a_grid_desc_mraw_kraw,
                                            make_tuple(make_unmerge_transform(make_tuple(AK0, AK1)),
                                                       make_right_pad_transform(MRaw, MPad)),
                                            make_tuple(Sequence<1>{}, Sequence<0>{}),
                                            make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

            return a_grid_desc_ak0_m_ak1;
        }
        else if constexpr(GemmSpec == GemmSpecialization::KPadding ||
                          GemmSpec == GemmSpecialization::NKPadding)
        {
            // pad K, but not M
            assert(K % AK1 == 0);

            const auto AK0 = K / AK1;

            const auto a_grid_desc_m_k = transform_tensor_descriptor(
                a_grid_desc_mraw_kraw,
                make_tuple(make_pass_through_transform(MRaw), make_right_pad_transform(KRaw, KPad)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));

            const auto a_grid_desc_ak0_m_ak1 =
                transform_tensor_descriptor(a_grid_desc_m_k,
                                            make_tuple(make_unmerge_transform(make_tuple(AK0, AK1)),
                                                       make_pass_through_transform(MRaw)),
                                            make_tuple(Sequence<1>{}, Sequence<0>{}),
                                            make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

            return a_grid_desc_ak0_m_ak1;
        }
        else
        {
            // not pad M or K
            assert(KRaw % AK1 == 0);

            const auto AK0 = KRaw / AK1;

            const auto a_grid_desc_ak0_m_ak1 =
                transform_tensor_descriptor(a_grid_desc_mraw_kraw,
                                            make_tuple(make_unmerge_transform(make_tuple(AK0, AK1)),
                                                       make_pass_through_transform(MRaw)),
                                            make_tuple(Sequence<1>{}, Sequence<0>{}),
                                            make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

            return a_grid_desc_ak0_m_ak1;
        }
    }

    static auto MakeBGridDescriptor_BK0_N_BK1(index_t KRaw, index_t NRaw, index_t StrideB)
    {
        const auto b_grid_desc_nraw_kraw = [&]() {
            if constexpr(is_same<tensor_layout::gemm::RowMajor, BLayout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(NRaw, KRaw),
                                                    make_tuple(I1, StrideB));
            }
            else if constexpr(is_same<tensor_layout::gemm::ColumnMajor, BLayout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(NRaw, KRaw),
                                                    make_tuple(StrideB, I1));
            }
        }();

        const auto N = math::integer_divide_ceil(NRaw, NPerBlock) * NPerBlock;
        const auto K = math::integer_divide_ceil(KRaw, KPerBlock) * KPerBlock;

        const auto NPad = N - NRaw;
        const auto KPad = K - KRaw;

        if constexpr(GemmSpec == GemmSpecialization::NKPadding ||
                     GemmSpec == GemmSpecialization::MNKPadding)
        {
            // pad both N and K
            assert(K % BK1 == 0);

            const auto BK0 = K / BK1;

            const auto b_grid_desc_n_k =
                transform_tensor_descriptor(b_grid_desc_nraw_kraw,
                                            make_tuple(make_right_pad_transform(NRaw, NPad),
                                                       make_right_pad_transform(KRaw, KPad)),
                                            make_tuple(Sequence<0>{}, Sequence<1>{}),
                                            make_tuple(Sequence<0>{}, Sequence<1>{}));

            const auto b_grid_desc_bk0_n_bk1 =
                transform_tensor_descriptor(b_grid_desc_n_k,
                                            make_tuple(make_unmerge_transform(make_tuple(BK0, BK1)),
                                                       make_pass_through_transform(N)),
                                            make_tuple(Sequence<1>{}, Sequence<0>{}),
                                            make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

            return b_grid_desc_bk0_n_bk1;
        }
        else if constexpr(GemmSpec == GemmSpecialization::NPadding ||
                          GemmSpec == GemmSpecialization::MNPadding)
        {
            // pad N, but not K
            assert(KRaw % BK1 == 0);

            const auto BK0 = KRaw / BK1;

            const auto b_grid_desc_bk0_n_bk1 =
                transform_tensor_descriptor(b_grid_desc_nraw_kraw,
                                            make_tuple(make_unmerge_transform(make_tuple(BK0, BK1)),
                                                       make_right_pad_transform(NRaw, NPad)),
                                            make_tuple(Sequence<1>{}, Sequence<0>{}),
                                            make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

            return b_grid_desc_bk0_n_bk1;
        }
        else if constexpr(GemmSpec == GemmSpecialization::KPadding ||
                          GemmSpec == GemmSpecialization::MKPadding)
        {
            // pad K, but not N
            assert(K % BK1 == 0);

            const auto BK0 = K / BK1;

            const auto b_grid_desc_n_k = transform_tensor_descriptor(
                b_grid_desc_nraw_kraw,
                make_tuple(make_pass_through_transform(NRaw), make_right_pad_transform(KRaw, KPad)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));

            const auto b_grid_desc_bk0_n_bk1 =
                transform_tensor_descriptor(b_grid_desc_n_k,
                                            make_tuple(make_unmerge_transform(make_tuple(BK0, BK1)),
                                                       make_pass_through_transform(NRaw)),
                                            make_tuple(Sequence<1>{}, Sequence<0>{}),
                                            make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

            return b_grid_desc_bk0_n_bk1;
        }
        else
        {
            // not pad N or K
            assert(KRaw % BK1 == 0);

            const auto BK0 = KRaw / BK1;

            const auto b_grid_desc_bk0_n_bk1 =
                transform_tensor_descriptor(b_grid_desc_nraw_kraw,
                                            make_tuple(make_unmerge_transform(make_tuple(BK0, BK1)),
                                                       make_pass_through_transform(NRaw)),
                                            make_tuple(Sequence<1>{}, Sequence<0>{}),
                                            make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

            return b_grid_desc_bk0_n_bk1;
        }
    }

    static auto MakeCGridDescriptor_M_N(index_t MRaw, index_t NRaw, index_t StrideC)
    {
        const auto c_grid_desc_mraw_nraw = [&]() {
            if constexpr(is_same<tensor_layout::gemm::RowMajor, CLayout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(MRaw, NRaw),
                                                    make_tuple(StrideC, I1));
            }
            else if constexpr(is_same<tensor_layout::gemm::ColumnMajor, CLayout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(MRaw, NRaw),
                                                    make_tuple(I1, StrideC));
            }
        }();

        const auto M = math::integer_divide_ceil(MRaw, MPerBlock) * MPerBlock;
        const auto N = math::integer_divide_ceil(NRaw, NPerBlock) * NPerBlock;

        const auto MPad = M - MRaw;
        const auto NPad = N - NRaw;

        if constexpr(GemmSpec == GemmSpecialization::MNPadding ||
                     GemmSpec == GemmSpecialization::MNKPadding)
        {
            // pad M and N
            return transform_tensor_descriptor(c_grid_desc_mraw_nraw,
                                               make_tuple(make_right_pad_transform(MRaw, MPad),
                                                          make_right_pad_transform(NRaw, NPad)),
                                               make_tuple(Sequence<0>{}, Sequence<1>{}),
                                               make_tuple(Sequence<0>{}, Sequence<1>{}));
        }
        else if constexpr(GemmSpec == GemmSpecialization::MPadding ||
                          GemmSpec == GemmSpecialization::MKPadding)
        {
            // pad M, but not N
            return transform_tensor_descriptor(
                c_grid_desc_mraw_nraw,
                make_tuple(make_right_pad_transform(MRaw, MPad), make_pass_through_transform(NRaw)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));
        }
        else if constexpr(GemmSpec == GemmSpecialization::NPadding ||
                          GemmSpec == GemmSpecialization::NKPadding)
        {
            // pad N, but not M
            return transform_tensor_descriptor(
                c_grid_desc_mraw_nraw,
                make_tuple(make_pass_through_transform(MRaw), make_right_pad_transform(NRaw, NPad)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));
        }
        else
        {
            // not pad M or N
            return c_grid_desc_mraw_nraw;
        }
    }

    // assume D is packed tensor
    static auto MakeReduceGridDescriptor_M(index_t MRaw)
    {
        const auto d_grid_desc_mraw = make_naive_tensor_descriptor_packed(make_tuple(MRaw));

        const auto M    = math::integer_divide_ceil(MRaw, MPerBlock) * MPerBlock;
        const auto MPad = M - MRaw;

        if constexpr(GemmSpec == GemmSpecialization::MPadding ||
                     GemmSpec == GemmSpecialization::MNPadding ||
                     GemmSpec == GemmSpecialization::MKPadding ||
                     GemmSpec == GemmSpecialization::MNKPadding)
        {
            // pad M
            return transform_tensor_descriptor(d_grid_desc_mraw,
                                               make_tuple(make_right_pad_transform(MRaw, MPad)),
                                               make_tuple(Sequence<0>{}),
                                               make_tuple(Sequence<0>{}));
        }
        else
        {
            // not pad M
            return d_grid_desc_mraw;
        }
    }

    using AGridDesc_AK0_M_AK1 = decltype(MakeAGridDescriptor_AK0_M_AK1(1, 1, 1));
    using BGridDesc_BK0_N_BK1 = decltype(MakeBGridDescriptor_BK0_N_BK1(1, 1, 1));
    using CGridDesc_M_N       = decltype(MakeCGridDescriptor_M_N(1, 1, 1));
    using C0GridDesc_M_N      = decltype(MakeCGridDescriptor_M_N(1, 1, 0));
    using C1GridDesc_M_N      = decltype(MakeCGridDescriptor_M_N(1, 1, 1));
    using ReduceGridDesc_M    = decltype(MakeReduceGridDescriptor_M(1));

    // GridwiseGemm
    using GridwiseGemm = GridwiseGemmBiasAddReduce_k0mk1_k0nk1_mn_xdl_cshuffle_v1<
        ADataType, // TODO: distinguish A/B datatype
        GemmAccDataType,
        CShuffleDataType,
        CDataType,
        BiasDataType,
        D0DataType,
        ReduceAccDataType,
        ReducePtrsGlobal,
        AElementwiseOperation,
        BElementwiseOperation,
        CElementwiseOperation,
        D0ElementwiseOperation,
        ReduceOperations,
        ReduceInElementwiseOperations,
        ReduceAccElementwiseOperations,
        InMemoryDataOperationEnum::Set,
        ReduceGlobalMemoryDataOperation,
        AGridDesc_AK0_M_AK1,
        BGridDesc_BK0_N_BK1,
        CGridDesc_M_N,
        C0GridDesc_M_N,
        C1GridDesc_M_N,
        ReduceGridDesc_M,
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
        CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
        CShuffleBlockTransferScalarPerVector_NPerBlock,
        CReduceThreadClusterLengths_MPerBlock_NPerBlock,
        CReduceThreadLds2VGprCopySrcDstScalarPerVector_NPerBlock,
        CReduceThreadVgpr2GlobalCopySrcDstScalarPerVector_MPerBlock,
        LoopSched>;

    // Argument
    struct Argument : public BaseArgument
    {
        Argument(const ADataType* p_a_grid,
                 const BDataType* p_b_grid,
                 CDataType* p_c_grid,
                 const BiasDataType* p_bias_grid,
                 const D0DataType* p_d0_grid,
                 ReducePtrsGlobal p_reduces_grid,
                 index_t MRaw,
                 index_t NRaw,
                 index_t KRaw,
                 index_t StrideA,
                 index_t StrideB,
                 index_t StrideC,
                 index_t StrideC1,
                 AElementwiseOperation a_element_op,
                 BElementwiseOperation b_element_op,
                 CElementwiseOperation c_element_op,
                 D0ElementwiseOperation d0_element_op,
                 ReduceInElementwiseOperations reduce_in_element_ops,
                 ReduceAccElementwiseOperations reduce_out_element_ops)
            : p_a_grid_{p_a_grid},
              p_b_grid_{p_b_grid},
              p_c_grid_{p_c_grid},
              p_bias_grid_{p_bias_grid},
              p_d0_grid_{p_d0_grid},
              p_reduces_grid_{p_reduces_grid},
              a_grid_desc_ak0_m_ak1_{DeviceOp::MakeAGridDescriptor_AK0_M_AK1(MRaw, KRaw, StrideA)},
              b_grid_desc_bk0_n_bk1_{DeviceOp::MakeBGridDescriptor_BK0_N_BK1(KRaw, NRaw, StrideB)},
              c_grid_desc_m_n_{DeviceOp::MakeCGridDescriptor_M_N(MRaw, NRaw, StrideC)},
              c0_grid_desc_m_n_{DeviceOp::MakeCGridDescriptor_M_N(MRaw, NRaw, 0)},
              c1_grid_desc_m_n_{DeviceOp::MakeCGridDescriptor_M_N(MRaw, NRaw, StrideC1)},
              reduce_grid_desc_m_{DeviceOp::MakeReduceGridDescriptor_M(MRaw)},
              c_grid_desc_mblock_mperblock_nblock_nperblock_{},
              c0_grid_desc_mblock_mperblock_nblock_nperblock_{},
              c1_grid_desc_mblock_mperblock_nblock_nperblock_{},
              reduce_grid_desc_mblock_mperblock_{},
              block_2_ctile_map_{GridwiseGemm::MakeDefaultBlock2CTileMap(c_grid_desc_m_n_)},
              a_element_op_{a_element_op},
              b_element_op_{b_element_op},
              c_element_op_{c_element_op},
              d0_element_op_{d0_element_op},
              reduce_in_element_ops_{reduce_in_element_ops},
              reduce_out_element_ops_{reduce_out_element_ops}
        {
            if(GridwiseGemm::CheckValidity(a_grid_desc_ak0_m_ak1_,
                                           b_grid_desc_bk0_n_bk1_,
                                           c_grid_desc_m_n_,
                                           block_2_ctile_map_))
            {
                c_grid_desc_mblock_mperblock_nblock_nperblock_ =
                    GridwiseGemm::MakeCGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
                        c_grid_desc_m_n_);

                c0_grid_desc_mblock_mperblock_nblock_nperblock_ =
                    GridwiseGemm::MakeCGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
                        c0_grid_desc_m_n_);

                c1_grid_desc_mblock_mperblock_nblock_nperblock_ =
                    GridwiseGemm::MakeCGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
                        c1_grid_desc_m_n_);

                reduce_grid_desc_mblock_mperblock_ =
                    GridwiseGemm::MakeReduceGridDescriptor_MBlock_MPerBlock(reduce_grid_desc_m_);
            }
        }

        //  private:
        const ADataType* p_a_grid_;
        const BDataType* p_b_grid_;
        CDataType* p_c_grid_;
        const BiasDataType* p_bias_grid_;
        const D0DataType* p_d0_grid_;
        ReducePtrsGlobal p_reduces_grid_;
        AGridDesc_AK0_M_AK1 a_grid_desc_ak0_m_ak1_;
        BGridDesc_BK0_N_BK1 b_grid_desc_bk0_n_bk1_;
        CGridDesc_M_N c_grid_desc_m_n_;
        C0GridDesc_M_N c0_grid_desc_m_n_;
        C1GridDesc_M_N c1_grid_desc_m_n_;
        ReduceGridDesc_M reduce_grid_desc_m_;
        typename GridwiseGemm::CGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock
            c_grid_desc_mblock_mperblock_nblock_nperblock_;
        typename GridwiseGemm::C0GridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock
            c0_grid_desc_mblock_mperblock_nblock_nperblock_;
        typename GridwiseGemm::C1GridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock
            c1_grid_desc_mblock_mperblock_nblock_nperblock_;
        typename GridwiseGemm::ReduceGridDescriptor_MBlock_MPerBlock
            reduce_grid_desc_mblock_mperblock_;
        typename GridwiseGemm::DefaultBlock2CTileMap block_2_ctile_map_;
        AElementwiseOperation a_element_op_;
        BElementwiseOperation b_element_op_;
        CElementwiseOperation c_element_op_;
        D0ElementwiseOperation d0_element_op_;
        ReduceInElementwiseOperations reduce_in_element_ops_;
        ReduceAccElementwiseOperations reduce_out_element_ops_;
    };

    // Invoker
    struct Invoker : public BaseInvoker
    {
        using Argument = DeviceOp::Argument;

        float Run(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
            if(!GridwiseGemm::CheckValidity(arg.a_grid_desc_ak0_m_ak1_,
                                            arg.b_grid_desc_bk0_n_bk1_,
                                            arg.c_grid_desc_m_n_,
                                            arg.block_2_ctile_map_))
            {
                throw std::runtime_error("wrong! GridwiseGemm has invalid setting");
            }

            const index_t grid_size =
                arg.block_2_ctile_map_.CalculateGridSize(arg.c_grid_desc_m_n_);

            const auto K =
                arg.a_grid_desc_ak0_m_ak1_.GetLength(I0) * arg.a_grid_desc_ak0_m_ak1_.GetLength(I2);

            float elapsed_time = 0.0f;
            if(GridwiseGemm::CalculateHasMainKBlockLoop(K))
            {
                const auto kernel = kernel_gemm_bias_add_reduce_xdl_cshuffle_v1<
                    GridwiseGemm,
                    ADataType, // TODO: distiguish A/B datatype
                    CDataType,
                    BiasDataType,
                    D0DataType,
                    ReducePtrsGlobal,
                    AElementwiseOperation,
                    BElementwiseOperation,
                    CElementwiseOperation,
                    D0ElementwiseOperation,
                    ReduceInElementwiseOperations,
                    ReduceAccElementwiseOperations,
                    DeviceOp::AGridDesc_AK0_M_AK1,
                    DeviceOp::BGridDesc_BK0_N_BK1,
                    typename GridwiseGemm::CGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock,
                    typename GridwiseGemm::C0GridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock,
                    typename GridwiseGemm::C1GridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock,
                    typename GridwiseGemm::ReduceGridDescriptor_MBlock_MPerBlock,
                    typename GridwiseGemm::DefaultBlock2CTileMap,
                    true>;

                elapsed_time =
                    launch_and_time_kernel(stream_config,
                                           kernel,
                                           dim3(grid_size),
                                           dim3(BlockSize),
                                           0,
                                           arg.p_a_grid_,
                                           arg.p_b_grid_,
                                           arg.p_c_grid_,
                                           arg.p_bias_grid_,
                                           arg.p_d0_grid_,
                                           arg.p_reduces_grid_,
                                           arg.a_element_op_,
                                           arg.b_element_op_,
                                           arg.c_element_op_,
                                           arg.d0_element_op_,
                                           arg.reduce_in_element_ops_,
                                           arg.reduce_out_element_ops_,
                                           arg.a_grid_desc_ak0_m_ak1_,
                                           arg.b_grid_desc_bk0_n_bk1_,
                                           arg.c_grid_desc_mblock_mperblock_nblock_nperblock_,
                                           arg.c0_grid_desc_mblock_mperblock_nblock_nperblock_,
                                           arg.c1_grid_desc_mblock_mperblock_nblock_nperblock_,
                                           arg.reduce_grid_desc_mblock_mperblock_,
                                           arg.block_2_ctile_map_);
            }
            else
            {
                const auto kernel = kernel_gemm_bias_add_reduce_xdl_cshuffle_v1<
                    GridwiseGemm,
                    ADataType, // TODO: distiguish A/B datatype
                    CDataType,
                    BiasDataType,
                    D0DataType,
                    ReducePtrsGlobal,
                    AElementwiseOperation,
                    BElementwiseOperation,
                    CElementwiseOperation,
                    D0ElementwiseOperation,
                    ReduceInElementwiseOperations,
                    ReduceAccElementwiseOperations,
                    DeviceOp::AGridDesc_AK0_M_AK1,
                    DeviceOp::BGridDesc_BK0_N_BK1,
                    typename GridwiseGemm::CGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock,
                    typename GridwiseGemm::C0GridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock,
                    typename GridwiseGemm::C1GridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock,
                    typename GridwiseGemm::ReduceGridDescriptor_MBlock_MPerBlock,
                    typename GridwiseGemm::DefaultBlock2CTileMap,
                    false>;

                elapsed_time =
                    launch_and_time_kernel(stream_config,
                                           kernel,
                                           dim3(grid_size),
                                           dim3(BlockSize),
                                           0,
                                           arg.p_a_grid_,
                                           arg.p_b_grid_,
                                           arg.p_c_grid_,
                                           arg.p_bias_grid_,
                                           arg.p_d0_grid_,
                                           arg.p_reduces_grid_,
                                           arg.a_element_op_,
                                           arg.b_element_op_,
                                           arg.c_element_op_,
                                           arg.d0_element_op_,
                                           arg.reduce_in_element_ops_,
                                           arg.reduce_out_element_ops_,
                                           arg.a_grid_desc_ak0_m_ak1_,
                                           arg.b_grid_desc_bk0_n_bk1_,
                                           arg.c_grid_desc_mblock_mperblock_nblock_nperblock_,
                                           arg.c0_grid_desc_mblock_mperblock_nblock_nperblock_,
                                           arg.c1_grid_desc_mblock_mperblock_nblock_nperblock_,
                                           arg.reduce_grid_desc_mblock_mperblock_,
                                           arg.block_2_ctile_map_);
            }

            return elapsed_time;
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
        return GridwiseGemm::CheckValidity(arg.a_grid_desc_ak0_m_ak1_,
                                           arg.b_grid_desc_bk0_n_bk1_,
                                           arg.c_grid_desc_m_n_,
                                           arg.block_2_ctile_map_);
    }

    // polymorphic
    bool IsSupportedArgument(const BaseArgument* p_arg) override
    {
        return IsSupportedArgument(*dynamic_cast<const Argument*>(p_arg));
    }

    static constexpr int NumReduce = ReduceOperations::Size();
    static auto MakeArgument(const void* p_a,
                             const void* p_b,
                             const void* p_bias,
                             std::array<const void*, 1> p_ds,
                             void* p_c,
                             std::array<void*, NumReduce> p_reduces,
                             ck::index_t M,
                             ck::index_t N,
                             ck::index_t K,
                             ck::index_t StrideA,
                             ck::index_t StrideB,
                             ck::index_t StrideC,
                             std::array<ck::index_t, 1> StrideDs,
                             std::array<void*, 3> gemm_element_ops,
                             std::array<void*, 1> d_element_ops,
                             std::array<void*, NumReduce> reduce_in_element_op,
                             std::array<void*, NumReduce> reduce_out_element_op)
    {
        ReducePtrsGlobal reduce_tuple = generate_tuple(
            [&](auto I) {
                auto tmp = ReducePtrsGlobal{}[I];
                using T  = remove_pointer_t<decltype(tmp)>;
                return static_cast<T*>(p_reduces[I]);
            },
            Number<NumReduce>{});

        ReduceInElementwiseOperations reduce_in_element_ops = generate_tuple(
            [&](auto I) {
                auto tmp = ReduceInElementwiseOperations{}[I];
                using T  = remove_pointer_t<decltype(tmp)>;
                return *(static_cast<T*>(reduce_in_element_op[I]));
            },
            Number<NumReduce>{});
        ReduceAccElementwiseOperations reduce_out_element_ops = generate_tuple(
            [&](auto I) {
                auto tmp = ReduceAccElementwiseOperations{}[I];
                using T  = remove_pointer_t<decltype(tmp)>;
                return *(static_cast<T*>(reduce_out_element_op[I]));
            },
            Number<NumReduce>{});

        AElementwiseOperation a_element_op =
            *(static_cast<AElementwiseOperation*>(gemm_element_ops[0]));
        BElementwiseOperation b_element_op =
            *(static_cast<BElementwiseOperation*>(gemm_element_ops[1]));
        CElementwiseOperation c_element_op =
            *(static_cast<CElementwiseOperation*>(gemm_element_ops[2]));
        D0ElementwiseOperation d_element_op =
            *(static_cast<D0ElementwiseOperation*>(d_element_ops[0]));

        return Argument{static_cast<const ADataType*>(p_a),
                        static_cast<const BDataType*>(p_b),
                        static_cast<CDataType*>(p_c),
                        static_cast<const BiasDataType*>(p_bias),
                        static_cast<const D0DataType*>(p_ds[0]),
                        reduce_tuple,
                        M,
                        N,
                        K,
                        StrideA,
                        StrideB,
                        StrideC,
                        StrideDs[0],
                        a_element_op,
                        b_element_op,
                        c_element_op,
                        d_element_op,
                        reduce_in_element_ops,
                        reduce_out_element_ops};
    }

    static auto MakeInvoker() { return Invoker{}; }

    // polymorphic
    std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const void* p_a,
                        const void* p_b,
                        const void* p_bias,
                        std::array<const void*, 1> p_ds,
                        void* p_c,
                        std::array<void*, NumReduce> p_reduces,
                        ck::index_t M,
                        ck::index_t N,
                        ck::index_t K,
                        ck::index_t StrideA,
                        ck::index_t StrideB,
                        ck::index_t StrideC,
                        std::array<ck::index_t, 1> StrideDs,
                        std::array<void*, 3> gemm_element_ops,
                        std::array<void*, 1> d_element_ops,
                        std::array<void*, NumReduce> reduce_in_element_op,
                        std::array<void*, NumReduce> reduce_out_element_op,
                        index_t /* KBatch */ = 1) override
    {
        ReducePtrsGlobal reduce_tuple = generate_tuple(
            [&](auto I) {
                auto tmp = ReducePtrsGlobal{}[I];
                using T  = remove_pointer_t<decltype(tmp)>;
                return static_cast<T*>(p_reduces[I]);
            },
            Number<NumReduce>{});

        ReduceInElementwiseOperations reduce_in_element_ops = generate_tuple(
            [&](auto I) {
                auto tmp = ReduceInElementwiseOperations{}[I];
                using T  = remove_pointer_t<decltype(tmp)>;
                return *(static_cast<T*>(reduce_in_element_op[I]));
            },
            Number<NumReduce>{});
        ReduceAccElementwiseOperations reduce_out_element_ops = generate_tuple(
            [&](auto I) {
                auto tmp = ReduceAccElementwiseOperations{}[I];
                using T  = remove_pointer_t<decltype(tmp)>;
                return *(static_cast<T*>(reduce_out_element_op[I]));
            },
            Number<NumReduce>{});

        AElementwiseOperation a_element_op =
            *(static_cast<AElementwiseOperation*>(gemm_element_ops[0]));
        BElementwiseOperation b_element_op =
            *(static_cast<BElementwiseOperation*>(gemm_element_ops[1]));
        CElementwiseOperation c_element_op =
            *(static_cast<CElementwiseOperation*>(gemm_element_ops[2]));
        D0ElementwiseOperation d_element_op =
            *(static_cast<D0ElementwiseOperation*>(d_element_ops[0]));

        return std::make_unique<Argument>(static_cast<const ADataType*>(p_a),
                                          static_cast<const BDataType*>(p_b),
                                          static_cast<CDataType*>(p_c),
                                          static_cast<const BiasDataType*>(p_bias),
                                          static_cast<const D0DataType*>(p_ds[0]),
                                          reduce_tuple,
                                          M,
                                          N,
                                          K,
                                          StrideA,
                                          StrideB,
                                          StrideC,
                                          StrideDs[0],
                                          a_element_op,
                                          b_element_op,
                                          c_element_op,
                                          d_element_op,
                                          reduce_in_element_ops,
                                          reduce_out_element_ops);
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
        str << "DeviceGemmBiasAddReduce_Xdl_CShuffle"
            << "<"
            << BlockSize << ", "
            << MPerBlock << ", "
            << NPerBlock << ", "
            << KPerBlock << ", "
            << AK1 << ", "
            << BK1
            << ">";
        // clang-format on

        return str.str();
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
