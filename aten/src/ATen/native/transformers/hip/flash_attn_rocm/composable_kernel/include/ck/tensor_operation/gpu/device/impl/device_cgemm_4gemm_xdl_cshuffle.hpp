// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>
#include <sstream>

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_gemm.hpp"
#include "ck/tensor_operation/gpu/device/device_cgemm.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_xdl_cshuffle_v1.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_elementwise_1d.hpp"
#include "ck/tensor_operation/gpu/element/binary_element_wise_operation.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

template <
    typename ALayout,
    typename BLayout,
    typename CLayout,
    typename ADataType,
    typename BDataType,
    typename CDataType,
    typename GemmAccDataType,
    typename CShuffleDataType,
    typename AElementwiseOperation,
    typename BElementwiseOperation,
    typename CElementwiseOperation,
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
    LoopScheduler LoopSched = make_default_loop_scheduler(),
    enable_if_t<
        is_same_v<AElementwiseOperation, ck::tensor_operation::element_wise::PassThrough> &&
            is_same_v<BElementwiseOperation, ck::tensor_operation::element_wise::PassThrough> &&
            is_same_v<CElementwiseOperation, ck::tensor_operation::element_wise::PassThrough>,
        bool> = false>
struct DeviceCGemm_4Gemm_Xdl_CShuffle
    : public DeviceCGemm<AElementwiseOperation, BElementwiseOperation, CElementwiseOperation>
{
    using DeviceOp = DeviceCGemm_4Gemm_Xdl_CShuffle;

    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};

    static constexpr auto MPerThread       = Number<4>{};
    static constexpr auto AScalarPerVector = Number<4>{};
    static constexpr auto BScalarPerVector = Number<4>{};
    static constexpr auto CScalarPerVector = Number<4>{};

    template <typename Desc_M>
    static auto PadDescriptor_M_1d(Desc_M desc_m, index_t gridSize, index_t blockSize)
    {
        const auto M            = desc_m.GetLength(I0);
        const index_t loop_step = gridSize * blockSize * MPerThread;
        const auto pad          = math::integer_least_multiple(M, loop_step) - M;
        const auto desc_m_pad =
            transform_tensor_descriptor(desc_m,
                                        make_tuple(make_right_pad_transform(M, pad)),
                                        make_tuple(Sequence<0>{}),
                                        make_tuple(Sequence<0>{}));
        return desc_m_pad;
    }

    static auto MakeDescriptor_M(const std::vector<index_t>& lengths,
                                 const std::vector<index_t>& strides,
                                 index_t gridSize,
                                 index_t blockSize)
    {
        auto tupleOfShape  = generate_tuple([&](auto I) { return lengths[I]; }, Number<2>{});
        auto tupleOfStride = generate_tuple([&](auto I) { return strides[I]; }, Number<2>{});

        // nd desc - [s0, s1, s2, ...]
        const auto desc   = make_naive_tensor_descriptor(tupleOfShape, tupleOfStride);
        const auto desc_m = transform_tensor_descriptor(
            desc,
            make_tuple(make_merge_transform(tupleOfShape)),
            make_tuple(generate_sequence_v2([&](auto I) { return I; }, Number<2>{})),
            make_tuple(Sequence<0>{}));

        return PadDescriptor_M_1d(desc_m, gridSize, blockSize);
    }

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

    using AGridDesc_AK0_M_AK1 = decltype(MakeAGridDescriptor_AK0_M_AK1(1, 1, 1));
    using BGridDesc_BK0_N_BK1 = decltype(MakeBGridDescriptor_BK0_N_BK1(1, 1, 1));
    using CGridDesc_M_N       = decltype(MakeCGridDescriptor_M_N(1, 1, 1));
    using CGridDesc_M         = decltype(MakeDescriptor_M({1, 1}, {1, 1}, 1, 1));

    // GridwiseGemm
    using GridwiseGemm = GridwiseGemm_k0mk1_k0nk1_mn_xdl_cshuffle_v1<
        ADataType, // TODO: distinguish A/B datatype
        GemmAccDataType,
        CShuffleDataType,
        CDataType,
        AElementwiseOperation,
        BElementwiseOperation,
        CElementwiseOperation,
        InMemoryDataOperationEnum::Set,
        AGridDesc_AK0_M_AK1,
        BGridDesc_BK0_N_BK1,
        CGridDesc_M_N,
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
        LoopSched>;

    // Argument
    struct Argument : public BaseArgument
    {
        Argument(const ADataType* p_a_grid_real,
                 const ADataType* p_a_grid_imag,
                 const BDataType* p_b_grid_real,
                 const BDataType* p_b_grid_imag,
                 CDataType* p_c_grid_real,
                 CDataType* p_c_grid_imag,
                 CDataType* p_workspace,
                 index_t MRaw,
                 index_t NRaw,
                 index_t KRaw,
                 index_t StrideA,
                 index_t StrideB,
                 index_t StrideC,
                 AElementwiseOperation a_element_op,
                 BElementwiseOperation b_element_op,
                 CElementwiseOperation c_element_op)
            : p_a_grid_real_{p_a_grid_real},
              p_a_grid_imag_{p_a_grid_imag},
              p_b_grid_real_{p_b_grid_real},
              p_b_grid_imag_{p_b_grid_imag},
              p_c_grid_real_{p_c_grid_real},
              p_c_grid_imag_{p_c_grid_imag},
              p_aux_grid_{p_workspace},
              a_grid_desc_ak0_m_ak1_{DeviceOp::MakeAGridDescriptor_AK0_M_AK1(MRaw, KRaw, StrideA)},
              b_grid_desc_bk0_n_bk1_{DeviceOp::MakeBGridDescriptor_BK0_N_BK1(KRaw, NRaw, StrideB)},
              c_grid_desc_m_n_{DeviceOp::MakeCGridDescriptor_M_N(MRaw, NRaw, StrideC)},
              c_grid_desc_mblock_mperblock_nblock_nperblock_{},
              block_2_ctile_map_{GridwiseGemm::MakeDefaultBlock2CTileMap(c_grid_desc_m_n_)},
              a_element_op_{a_element_op},
              b_element_op_{b_element_op},
              c_element_op_{c_element_op}
        {
            if(GridwiseGemm::CheckValidity(a_grid_desc_ak0_m_ak1_,
                                           b_grid_desc_bk0_n_bk1_,
                                           c_grid_desc_m_n_,
                                           block_2_ctile_map_))
            {
                c_grid_desc_mblock_mperblock_nblock_nperblock_ =
                    GridwiseGemm::MakeCGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
                        c_grid_desc_m_n_);
            }

            const index_t grid_size = block_2_ctile_map_.CalculateGridSize(c_grid_desc_m_n_);

            if constexpr(is_same<tensor_layout::gemm::RowMajor, CLayout>::value)
            {
                c_grid_desc_m_ =
                    DeviceOp::MakeDescriptor_M({MRaw, NRaw}, {StrideC, I1}, grid_size, BlockSize);
            }
            else if constexpr(is_same<tensor_layout::gemm::ColumnMajor, CLayout>::value)
            {
                c_grid_desc_m_ =
                    DeviceOp::MakeDescriptor_M({MRaw, NRaw}, {I1, StrideC}, grid_size, BlockSize);
            }

            p_aux_2_grid_ = p_workspace + c_grid_desc_m_n_.GetElementSpaceSize();
        }

        //  private:
        const ADataType* p_a_grid_real_;
        const ADataType* p_a_grid_imag_;
        const BDataType* p_b_grid_real_;
        const BDataType* p_b_grid_imag_;
        CDataType* p_c_grid_real_;
        CDataType* p_c_grid_imag_;
        CDataType* p_aux_grid_;
        CDataType* p_aux_2_grid_;
        AGridDesc_AK0_M_AK1 a_grid_desc_ak0_m_ak1_;
        BGridDesc_BK0_N_BK1 b_grid_desc_bk0_n_bk1_;
        CGridDesc_M_N c_grid_desc_m_n_;
        CGridDesc_M c_grid_desc_m_;
        typename GridwiseGemm::CGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock
            c_grid_desc_mblock_mperblock_nblock_nperblock_;
        typename GridwiseGemm::DefaultBlock2CTileMap block_2_ctile_map_;
        AElementwiseOperation a_element_op_;
        BElementwiseOperation b_element_op_;
        CElementwiseOperation c_element_op_;
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

            float ave_time = 0;

            using Add      = ck::tensor_operation::element_wise::Add;
            using Subtract = ck::tensor_operation::element_wise::Subtract;

            using GridwiseBinAdd =
                GridwiseElementwise_1D<Tuple<CGridDesc_M, CGridDesc_M>,
                                       Tuple<CGridDesc_M>,
                                       Tuple<const CDataType*, const CDataType*>,
                                       Tuple<CDataType*>,
                                       Add,
                                       MPerThread,
                                       Sequence<AScalarPerVector, BScalarPerVector>,
                                       Sequence<CScalarPerVector>>;

            using GridwiseBinSubtract =
                GridwiseElementwise_1D<Tuple<CGridDesc_M, CGridDesc_M>,
                                       Tuple<CGridDesc_M>,
                                       Tuple<const CDataType*, const CDataType*>,
                                       Tuple<CDataType*>,
                                       Subtract,
                                       MPerThread,
                                       Sequence<AScalarPerVector, BScalarPerVector>,
                                       Sequence<CScalarPerVector>>;

            const auto add_kernel = kernel_elementwise_1d<GridwiseBinAdd,
                                                          Tuple<CGridDesc_M, CGridDesc_M>,
                                                          Tuple<CGridDesc_M>,
                                                          Tuple<const CDataType*, const CDataType*>,
                                                          Tuple<CDataType*>,
                                                          Add>;

            const auto subtract_kernel =
                kernel_elementwise_1d<GridwiseBinSubtract,
                                      Tuple<CGridDesc_M, CGridDesc_M>,
                                      Tuple<CGridDesc_M>,
                                      Tuple<const CDataType*, const CDataType*>,
                                      Tuple<CDataType*>,
                                      Subtract>;

            if(GridwiseGemm::CalculateHasMainKBlockLoop(K))
            {
                const auto kernel = kernel_gemm_xdl_cshuffle_v1<
                    GridwiseGemm,
                    ADataType, // TODO: distiguish A/B datatype
                    CDataType,
                    AElementwiseOperation,
                    BElementwiseOperation,
                    CElementwiseOperation,
                    DeviceOp::AGridDesc_AK0_M_AK1,
                    DeviceOp::BGridDesc_BK0_N_BK1,
                    typename GridwiseGemm::CGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock,
                    typename GridwiseGemm::DefaultBlock2CTileMap,
                    true>;

                ave_time +=
                    launch_and_time_kernel(stream_config,
                                           kernel,
                                           dim3(grid_size),
                                           dim3(BlockSize),
                                           0,
                                           arg.p_a_grid_real_,
                                           arg.p_b_grid_real_,
                                           arg.p_aux_grid_,
                                           arg.a_element_op_,
                                           arg.b_element_op_,
                                           arg.c_element_op_,
                                           arg.a_grid_desc_ak0_m_ak1_,
                                           arg.b_grid_desc_bk0_n_bk1_,
                                           arg.c_grid_desc_mblock_mperblock_nblock_nperblock_,
                                           arg.block_2_ctile_map_);

                ave_time +=
                    launch_and_time_kernel(stream_config,
                                           kernel,
                                           dim3(grid_size),
                                           dim3(BlockSize),
                                           0,
                                           arg.p_a_grid_imag_,
                                           arg.p_b_grid_imag_,
                                           arg.p_aux_2_grid_,
                                           arg.a_element_op_,
                                           arg.b_element_op_,
                                           arg.c_element_op_,
                                           arg.a_grid_desc_ak0_m_ak1_,
                                           arg.b_grid_desc_bk0_n_bk1_,
                                           arg.c_grid_desc_mblock_mperblock_nblock_nperblock_,
                                           arg.block_2_ctile_map_);

                // c_real = aux - aux_2
                ave_time += launch_and_time_kernel(
                    stream_config,
                    subtract_kernel,
                    dim3(grid_size),
                    dim3(BlockSize),
                    0,
                    make_tuple(arg.c_grid_desc_m_, arg.c_grid_desc_m_),
                    make_tuple(arg.c_grid_desc_m_),
                    make_tuple(const_cast<const CDataType*>(arg.p_aux_grid_),
                               const_cast<const CDataType*>(arg.p_aux_2_grid_)),
                    make_tuple(arg.p_c_grid_real_),
                    Subtract{});

                ave_time +=
                    launch_and_time_kernel(stream_config,
                                           kernel,
                                           dim3(grid_size),
                                           dim3(BlockSize),
                                           0,
                                           arg.p_a_grid_real_,
                                           arg.p_b_grid_imag_,
                                           arg.p_aux_grid_,
                                           arg.a_element_op_,
                                           arg.b_element_op_,
                                           arg.c_element_op_,
                                           arg.a_grid_desc_ak0_m_ak1_,
                                           arg.b_grid_desc_bk0_n_bk1_,
                                           arg.c_grid_desc_mblock_mperblock_nblock_nperblock_,
                                           arg.block_2_ctile_map_);

                ave_time +=
                    launch_and_time_kernel(stream_config,
                                           kernel,
                                           dim3(grid_size),
                                           dim3(BlockSize),
                                           0,
                                           arg.p_a_grid_imag_,
                                           arg.p_b_grid_real_,
                                           arg.p_aux_2_grid_,
                                           arg.a_element_op_,
                                           arg.b_element_op_,
                                           arg.c_element_op_,
                                           arg.a_grid_desc_ak0_m_ak1_,
                                           arg.b_grid_desc_bk0_n_bk1_,
                                           arg.c_grid_desc_mblock_mperblock_nblock_nperblock_,
                                           arg.block_2_ctile_map_);

                // c_imag = aux + aux_2
                ave_time += launch_and_time_kernel(
                    stream_config,
                    add_kernel,
                    dim3(grid_size),
                    dim3(BlockSize),
                    0,
                    make_tuple(arg.c_grid_desc_m_, arg.c_grid_desc_m_),
                    make_tuple(arg.c_grid_desc_m_),
                    make_tuple(const_cast<const CDataType*>(arg.p_aux_grid_),
                               const_cast<const CDataType*>(arg.p_aux_2_grid_)),
                    make_tuple(arg.p_c_grid_imag_),
                    Add{});
            }
            else
            {
                const auto kernel = kernel_gemm_xdl_cshuffle_v1<
                    GridwiseGemm,
                    ADataType, // TODO: distiguish A/B datatype
                    CDataType,
                    AElementwiseOperation,
                    BElementwiseOperation,
                    CElementwiseOperation,
                    DeviceOp::AGridDesc_AK0_M_AK1,
                    DeviceOp::BGridDesc_BK0_N_BK1,
                    typename GridwiseGemm::CGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock,
                    typename GridwiseGemm::DefaultBlock2CTileMap,
                    false>;

                ave_time +=
                    launch_and_time_kernel(stream_config,
                                           kernel,
                                           dim3(grid_size),
                                           dim3(BlockSize),
                                           0,
                                           arg.p_a_grid_real_,
                                           arg.p_b_grid_real_,
                                           arg.p_aux_grid_,
                                           arg.a_element_op_,
                                           arg.b_element_op_,
                                           arg.c_element_op_,
                                           arg.a_grid_desc_ak0_m_ak1_,
                                           arg.b_grid_desc_bk0_n_bk1_,
                                           arg.c_grid_desc_mblock_mperblock_nblock_nperblock_,
                                           arg.block_2_ctile_map_);

                ave_time +=
                    launch_and_time_kernel(stream_config,
                                           kernel,
                                           dim3(grid_size),
                                           dim3(BlockSize),
                                           0,
                                           arg.p_a_grid_imag_,
                                           arg.p_b_grid_imag_,
                                           arg.p_aux_2_grid_,
                                           arg.a_element_op_,
                                           arg.b_element_op_,
                                           arg.c_element_op_,
                                           arg.a_grid_desc_ak0_m_ak1_,
                                           arg.b_grid_desc_bk0_n_bk1_,
                                           arg.c_grid_desc_mblock_mperblock_nblock_nperblock_,
                                           arg.block_2_ctile_map_);

                // c_real = aux - aux_2
                ave_time += launch_and_time_kernel(
                    stream_config,
                    subtract_kernel,
                    dim3(grid_size),
                    dim3(BlockSize),
                    0,
                    make_tuple(arg.c_grid_desc_m_, arg.c_grid_desc_m_),
                    make_tuple(arg.c_grid_desc_m_),
                    make_tuple(const_cast<const CDataType*>(arg.p_aux_grid_),
                               const_cast<const CDataType*>(arg.p_aux_2_grid_)),
                    make_tuple(arg.p_c_grid_real_),
                    Subtract{});

                ave_time +=
                    launch_and_time_kernel(stream_config,
                                           kernel,
                                           dim3(grid_size),
                                           dim3(BlockSize),
                                           0,
                                           arg.p_a_grid_real_,
                                           arg.p_b_grid_imag_,
                                           arg.p_aux_grid_,
                                           arg.a_element_op_,
                                           arg.b_element_op_,
                                           arg.c_element_op_,
                                           arg.a_grid_desc_ak0_m_ak1_,
                                           arg.b_grid_desc_bk0_n_bk1_,
                                           arg.c_grid_desc_mblock_mperblock_nblock_nperblock_,
                                           arg.block_2_ctile_map_);

                ave_time +=
                    launch_and_time_kernel(stream_config,
                                           kernel,
                                           dim3(grid_size),
                                           dim3(BlockSize),
                                           0,
                                           arg.p_a_grid_imag_,
                                           arg.p_b_grid_real_,
                                           arg.p_aux_2_grid_,
                                           arg.a_element_op_,
                                           arg.b_element_op_,
                                           arg.c_element_op_,
                                           arg.a_grid_desc_ak0_m_ak1_,
                                           arg.b_grid_desc_bk0_n_bk1_,
                                           arg.c_grid_desc_mblock_mperblock_nblock_nperblock_,
                                           arg.block_2_ctile_map_);

                // c_imag = aux + aux_2
                ave_time += launch_and_time_kernel(
                    stream_config,
                    add_kernel,
                    dim3(grid_size),
                    dim3(BlockSize),
                    0,
                    make_tuple(arg.c_grid_desc_m_, arg.c_grid_desc_m_),
                    make_tuple(arg.c_grid_desc_m_),
                    make_tuple(const_cast<const CDataType*>(arg.p_aux_grid_),
                               const_cast<const CDataType*>(arg.p_aux_2_grid_)),
                    make_tuple(arg.p_c_grid_imag_),
                    Add{});
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

    static auto MakeArgument(const ADataType* p_a_real,
                             const ADataType* p_a_imag,
                             const BDataType* p_b_real,
                             const BDataType* p_b_imag,
                             CDataType* p_c_real,
                             CDataType* p_c_imag,
                             CDataType* p_workspace,
                             index_t MRaw,
                             index_t NRaw,
                             index_t KRaw,
                             index_t StrideA,
                             index_t StrideB,
                             index_t StrideC,
                             AElementwiseOperation a_element_op,
                             BElementwiseOperation b_element_op,
                             CElementwiseOperation c_element_op)
    {
        return Argument{p_a_real,
                        p_a_imag,
                        p_b_real,
                        p_b_imag,
                        p_c_real,
                        p_c_imag,
                        p_workspace,
                        MRaw,
                        NRaw,
                        KRaw,
                        StrideA,
                        StrideB,
                        StrideC,
                        a_element_op,
                        b_element_op,
                        c_element_op};
    }

    static auto MakeInvoker() { return Invoker{}; }

    // polymorphic
    std::unique_ptr<BaseArgument> MakeArgumentPointer(const void* p_a_real,
                                                      const void* p_a_imag,
                                                      const void* p_b_real,
                                                      const void* p_b_imag,
                                                      void* p_c_real,
                                                      void* p_c_imag,
                                                      void* p_workspace,
                                                      index_t MRaw,
                                                      index_t NRaw,
                                                      index_t KRaw,
                                                      index_t StrideA,
                                                      index_t StrideB,
                                                      index_t StrideC,
                                                      AElementwiseOperation a_element_op,
                                                      BElementwiseOperation b_element_op,
                                                      CElementwiseOperation c_element_op,
                                                      index_t /* KBatch */ = 1) override
    {
        return std::make_unique<Argument>(static_cast<const ADataType*>(p_a_real),
                                          static_cast<const ADataType*>(p_a_imag),
                                          static_cast<const BDataType*>(p_b_real),
                                          static_cast<const BDataType*>(p_b_imag),
                                          static_cast<CDataType*>(p_c_real),
                                          static_cast<CDataType*>(p_c_imag),
                                          static_cast<CDataType*>(p_workspace),
                                          MRaw,
                                          NRaw,
                                          KRaw,
                                          StrideA,
                                          StrideB,
                                          StrideC,
                                          a_element_op,
                                          b_element_op,
                                          c_element_op);
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
        str << "DeviceCGemm_4Gemm_Xdl_CShuffle"
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

    std::size_t GetWorkspaceSize(index_t MRaw,
                                 index_t NRaw,
                                 [[maybe_unused]] index_t KRaw,
                                 [[maybe_unused]] index_t StrideA,
                                 [[maybe_unused]] index_t StrideB,
                                 index_t StrideC) override
    {
        const auto c_grid_desc_m_n = MakeCGridDescriptor_M_N(MRaw, NRaw, StrideC);

        return 2 * sizeof(CDataType) * c_grid_desc_m_n.GetElementSpaceSize();
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
