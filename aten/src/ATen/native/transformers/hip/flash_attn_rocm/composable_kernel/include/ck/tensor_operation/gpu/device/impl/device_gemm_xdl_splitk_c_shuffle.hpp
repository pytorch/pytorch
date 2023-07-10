// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>
#include <sstream>

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_gemm_splitk.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_xdlops_v2r4r2.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

template <typename ADataType,
          typename BDataType,
          typename CDataType,
          typename AccDataType,
          typename ALayout,
          typename BLayout,
          typename CLayout,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation,
          GemmSpecialization GemmSpec,
          ck::index_t BlockSize,
          ck::index_t MPerBlock,
          ck::index_t NPerBlock,
          ck::index_t K0PerBlock,
          ck::index_t K1,
          ck::index_t MPerXDL,
          ck::index_t NPerXDL,
          ck::index_t MXdlPerWave,
          ck::index_t NXdlPerWave,
          typename ABlockTransferThreadClusterLengths_K0_M_K1,
          typename ABlockTransferThreadClusterArrangeOrder,
          typename ABlockTransferSrcAccessOrder,
          ck::index_t ABlockTransferSrcVectorDim,
          ck::index_t ABlockTransferSrcScalarPerVector,
          ck::index_t ABlockTransferDstScalarPerVector_K1,
          bool ABlockLdsAddExtraM,
          typename BBlockTransferThreadClusterLengths_K0_N_K1,
          typename BBlockTransferThreadClusterArrangeOrder,
          typename BBlockTransferSrcAccessOrder,
          ck::index_t BBlockTransferSrcVectorDim,
          ck::index_t BBlockTransferSrcScalarPerVector,
          ck::index_t BBlockTransferDstScalarPerVector_K1,
          bool BBlockLdsAddExtraN,
          index_t CShuffleMRepeatPerShuffle,
          index_t CShuffleNRepeatPerShuffle,
          typename CBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          index_t CBlockTransferScalarPerVector_NWaveNPerXDL>
struct DeviceGemmXdlSplitKCShuffle : public DeviceGemmSplitK<ALayout,
                                                             BLayout,
                                                             CLayout,
                                                             ADataType,
                                                             BDataType,
                                                             CDataType,
                                                             AElementwiseOperation,
                                                             BElementwiseOperation,
                                                             CElementwiseOperation>
{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};

    static constexpr auto K1Number = Number<K1>{};

    static auto
    MakeAGridDescriptor_KBatch_K0_M_K1(index_t M, index_t K, index_t StrideA, int KBatch, int KPad)
    {
        assert(KPad % (K1 * KBatch) == 0);

        const index_t K0 = KPad / (K1 * KBatch);

        const auto a_grid_desc_m_k = [&]() {
            if constexpr(is_same<tensor_layout::gemm::RowMajor, ALayout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(M, K), make_tuple(StrideA, I1));
            }
            else if constexpr(is_same<tensor_layout::gemm::ColumnMajor, ALayout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(M, K), make_tuple(I1, StrideA));
            }
        }();

        const auto a_grid_desc_m_kpad = transform_tensor_descriptor(
            a_grid_desc_m_k,
            make_tuple(make_pass_through_transform(M), make_right_pad_transform(K, KPad - K)),
            make_tuple(Sequence<0>{}, Sequence<1>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}));

        if constexpr(GemmSpec == GemmSpecialization::MNPadding)
        {
            const auto PadM = (MPerBlock - M % MPerBlock) % MPerBlock;
            return transform_tensor_descriptor(
                a_grid_desc_m_kpad,
                make_tuple(make_unmerge_transform(make_tuple(KBatch, K0, K1Number)),
                           make_right_pad_transform(M, PadM)),
                make_tuple(Sequence<1>{}, Sequence<0>{}),
                make_tuple(Sequence<0, 1, 3>{}, Sequence<2>{}));
        }
        else
        {
            return transform_tensor_descriptor(
                a_grid_desc_m_kpad,
                make_tuple(make_unmerge_transform(make_tuple(KBatch, K0, K1Number)),
                           make_pass_through_transform(M)),
                make_tuple(Sequence<1>{}, Sequence<0>{}),
                make_tuple(Sequence<0, 1, 3>{}, Sequence<2>{}));
        }
    }

    static auto
    MakeBGridDescriptor_KBatch_K0_N_K1(index_t K, index_t N, index_t StrideB, int KBatch, int KPad)
    {
        assert(KPad % (K1 * KBatch) == 0);

        const index_t K0 = KPad / (K1 * KBatch);

        const auto b_grid_desc_k_n = [&]() {
            if constexpr(is_same<tensor_layout::gemm::RowMajor, BLayout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(K, N), make_tuple(StrideB, I1));
            }
            else if constexpr(is_same<tensor_layout::gemm::ColumnMajor, BLayout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(K, N), make_tuple(I1, StrideB));
            }
        }();

        const auto b_grid_desc_kpad_n = transform_tensor_descriptor(
            b_grid_desc_k_n,
            make_tuple(make_right_pad_transform(K, KPad - K), make_pass_through_transform(N)),
            make_tuple(Sequence<0>{}, Sequence<1>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}));

        if constexpr(GemmSpec == GemmSpecialization::MNPadding)
        {
            const auto PadN = (NPerBlock - N % NPerBlock) % NPerBlock;
            return transform_tensor_descriptor(
                b_grid_desc_kpad_n,
                make_tuple(make_unmerge_transform(make_tuple(KBatch, K0, K1Number)),
                           make_right_pad_transform(N, PadN)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0, 1, 3>{}, Sequence<2>{}));
        }
        else
        {
            return transform_tensor_descriptor(
                b_grid_desc_kpad_n,
                make_tuple(make_unmerge_transform(make_tuple(KBatch, K0, K1Number)),
                           make_pass_through_transform(N)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0, 1, 3>{}, Sequence<2>{}));
        }
    }

    static auto MakeCGridDescriptor_M_N(index_t M, index_t N, index_t StrideC)
    {
        const auto c_grid_desc_m_n = [&]() {
            if constexpr(is_same<tensor_layout::gemm::RowMajor, CLayout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(M, N), make_tuple(StrideC, I1));
            }
            else if constexpr(is_same<tensor_layout::gemm::ColumnMajor, CLayout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(M, N), make_tuple(I1, StrideC));
            }
        }();

        if constexpr(GemmSpec == GemmSpecialization::MNPadding)
        {
            const auto PadM = (MPerBlock - M % MPerBlock) % MPerBlock;
            const auto PadN = (NPerBlock - N % NPerBlock) % NPerBlock;

            return transform_tensor_descriptor(
                c_grid_desc_m_n,
                make_tuple(make_right_pad_transform(M, PadM), make_right_pad_transform(N, PadN)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));
        }
        else
        {

            return transform_tensor_descriptor(
                c_grid_desc_m_n,
                make_tuple(make_pass_through_transform(M), make_pass_through_transform(N)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));
        }
    }

    static auto GetKPad(index_t K, index_t KBatch)
    {
        const index_t K0   = math::integer_divide_ceil(K, K1 * K0PerBlock * KBatch) * K0PerBlock;
        const index_t KPad = KBatch * K0 * K1;
        return KPad;
    }

    using AGridDesc_K0_M_K1 = decltype(MakeAGridDescriptor_KBatch_K0_M_K1(1, 1, 1, 1, 1));
    using BGridDesc_K0_N_K1 = decltype(MakeBGridDescriptor_KBatch_K0_N_K1(1, 1, 1, 1, 1));
    using CGridDesc_M_N     = decltype(MakeCGridDescriptor_M_N(1, 1, 1));

    // GridwiseGemm
    using GridwiseGemm = GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2<
        BlockSize,
        ADataType, // TODO: distinguish A/B datatype
        AccDataType,
        CDataType,
        InMemoryDataOperationEnum::Set,
        AGridDesc_K0_M_K1,
        BGridDesc_K0_N_K1,
        CGridDesc_M_N,
        AElementwiseOperation,
        BElementwiseOperation,
        CElementwiseOperation,
        MPerBlock,
        NPerBlock,
        K0PerBlock,
        MPerXDL,
        NPerXDL,
        K1,
        MXdlPerWave,
        NXdlPerWave,
        ABlockTransferThreadClusterLengths_K0_M_K1,
        ABlockTransferThreadClusterArrangeOrder,
        ABlockTransferSrcAccessOrder,
        ABlockTransferSrcVectorDim,
        ABlockTransferSrcScalarPerVector,
        ABlockTransferDstScalarPerVector_K1,
        false, // AThreadTransferSrcResetCoordinateAfterRun,
        ABlockLdsAddExtraM,
        BBlockTransferThreadClusterLengths_K0_N_K1,
        BBlockTransferThreadClusterArrangeOrder,
        BBlockTransferSrcAccessOrder,
        BBlockTransferSrcVectorDim,
        BBlockTransferSrcScalarPerVector,
        BBlockTransferDstScalarPerVector_K1,
        false, // BThreadTransferSrcResetCoordinateAfterRun,
        BBlockLdsAddExtraN,
        CShuffleMRepeatPerShuffle,
        CShuffleNRepeatPerShuffle,
        CBlockTransferScalarPerVector_NWaveNPerXDL,
        CBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock>;

    // GridwiseGemm
    using GridwiseGemmAtomicAdd = GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2<
        BlockSize,
        ADataType, // TODO: distinguish A/B datatype
        AccDataType,
        CDataType,
        InMemoryDataOperationEnum::AtomicAdd,
        AGridDesc_K0_M_K1,
        BGridDesc_K0_N_K1,
        CGridDesc_M_N,
        AElementwiseOperation,
        BElementwiseOperation,
        CElementwiseOperation,
        MPerBlock,
        NPerBlock,
        K0PerBlock,
        MPerXDL,
        NPerXDL,
        K1,
        MXdlPerWave,
        NXdlPerWave,
        ABlockTransferThreadClusterLengths_K0_M_K1,
        ABlockTransferThreadClusterArrangeOrder,
        ABlockTransferSrcAccessOrder,
        ABlockTransferSrcVectorDim,
        ABlockTransferSrcScalarPerVector,
        ABlockTransferDstScalarPerVector_K1,
        false, // AThreadTransferSrcResetCoordinateAfterRun,
        ABlockLdsAddExtraM,
        BBlockTransferThreadClusterLengths_K0_N_K1,
        BBlockTransferThreadClusterArrangeOrder,
        BBlockTransferSrcAccessOrder,
        BBlockTransferSrcVectorDim,
        BBlockTransferSrcScalarPerVector,
        BBlockTransferDstScalarPerVector_K1,
        false, // BThreadTransferSrcResetCoordinateAfterRun,
        BBlockLdsAddExtraN,
        CShuffleMRepeatPerShuffle,
        CShuffleNRepeatPerShuffle,
        CBlockTransferScalarPerVector_NWaveNPerXDL,
        CBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock>;

    using CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock =
        decltype(GridwiseGemm::MakeCGridDesc_MBlock_MPerBlock_NBlock_NPerBlock(CGridDesc_M_N{}));

    using Block2CTileMap = typename GridwiseGemm::CBlockClusterAdaptor;

    // Argument
    struct Argument : public BaseArgument
    {
        Argument(const ADataType* p_a_grid,
                 const BDataType* p_b_grid,
                 CDataType* p_c_grid,
                 index_t M,
                 index_t N,
                 index_t K,
                 index_t StrideA,
                 index_t StrideB,
                 index_t StrideC,
                 index_t M01,
                 index_t N01,
                 AElementwiseOperation a_element_op,
                 BElementwiseOperation b_element_op,
                 CElementwiseOperation c_element_op,
                 index_t k_batch)
            : p_a_grid_{p_a_grid},
              p_b_grid_{p_b_grid},
              p_c_grid_{p_c_grid},
              a_grid_desc_kbatch_k0_m_k1_{},
              b_grid_desc_kbatch_k0_n_k1_{},
              c_grid_desc_m_n_{},
              c_grid_desc_mblock_mperblock_nblock_nperblock_{},
              block_2_ctile_map_{},
              M01_{M01},
              N01_{N01},
              a_element_op_{a_element_op},
              b_element_op_{b_element_op},
              c_element_op_{c_element_op},
              k_batch_{k_batch}
        {
            int KPad = DeviceGemmXdlSplitKCShuffle::GetKPad(K, k_batch_);

            a_grid_desc_kbatch_k0_m_k1_ =
                DeviceGemmXdlSplitKCShuffle::MakeAGridDescriptor_KBatch_K0_M_K1(
                    M, K, StrideA, k_batch_, KPad);
            b_grid_desc_kbatch_k0_n_k1_ =
                DeviceGemmXdlSplitKCShuffle::MakeBGridDescriptor_KBatch_K0_N_K1(
                    K, N, StrideB, k_batch_, KPad);
            c_grid_desc_m_n_ = DeviceGemmXdlSplitKCShuffle::MakeCGridDescriptor_M_N(M, N, StrideC);

            block_2_ctile_map_ =
                GridwiseGemm::MakeCBlockClusterAdaptor(c_grid_desc_m_n_, M01, N01, k_batch_);

            if(GridwiseGemm::CheckValidity(a_grid_desc_kbatch_k0_m_k1_,
                                           b_grid_desc_kbatch_k0_n_k1_,
                                           c_grid_desc_m_n_,
                                           block_2_ctile_map_))
            {
                c_grid_desc_mblock_mperblock_nblock_nperblock_ =
                    GridwiseGemm::MakeCGridDesc_MBlock_MPerBlock_NBlock_NPerBlock(c_grid_desc_m_n_);
            }
        }

        //  private:
        const ADataType* p_a_grid_;
        const BDataType* p_b_grid_;
        CDataType* p_c_grid_;
        AGridDesc_K0_M_K1 a_grid_desc_kbatch_k0_m_k1_;
        BGridDesc_K0_N_K1 b_grid_desc_kbatch_k0_n_k1_;
        CGridDesc_M_N c_grid_desc_m_n_;
        CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock c_grid_desc_mblock_mperblock_nblock_nperblock_;
        Block2CTileMap block_2_ctile_map_;
        index_t M01_;
        index_t N01_;
        AElementwiseOperation a_element_op_;
        BElementwiseOperation b_element_op_;
        CElementwiseOperation c_element_op_;
        index_t k_batch_;
    };

    // Invoker
    struct Invoker : public BaseInvoker
    {
        using Argument = DeviceGemmXdlSplitKCShuffle::Argument;

        void Print(const Argument& arg)
        {
            std::cout << "arg.a_grid_desc_kbatch_k0_m_k1_{"
                      << arg.a_grid_desc_kbatch_k0_m_k1_.GetLength(I0) << ", "
                      << arg.a_grid_desc_kbatch_k0_m_k1_.GetLength(I1) << ", "
                      << arg.a_grid_desc_kbatch_k0_m_k1_.GetLength(I2) << ", "
                      << arg.a_grid_desc_kbatch_k0_m_k1_.GetLength(I3) << "}" << std::endl;

            std::cout << "arg.b_grid_desc_kbatch_k0_n_k1_{"
                      << arg.b_grid_desc_kbatch_k0_n_k1_.GetLength(I0) << ", "
                      << arg.b_grid_desc_kbatch_k0_n_k1_.GetLength(I1) << ", "
                      << arg.b_grid_desc_kbatch_k0_n_k1_.GetLength(I2) << ", "
                      << arg.b_grid_desc_kbatch_k0_n_k1_.GetLength(I3) << "}" << std::endl;

            std::cout << "arg.c_grid_desc_m_n_{ " << arg.c_grid_desc_m_n_.GetLength(I0) << ", "
                      << arg.c_grid_desc_m_n_.GetLength(I1) << "}" << std::endl;
        }

        float Run(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
            if(stream_config.log_level_ > 0)
            {
                Print(arg);
            }

            const auto kbatch = arg.a_grid_desc_kbatch_k0_m_k1_.GetLength(I0);

            if(!GridwiseGemm::CheckValidity(arg.a_grid_desc_kbatch_k0_m_k1_,
                                            arg.b_grid_desc_kbatch_k0_n_k1_,
                                            arg.c_grid_desc_m_n_,
                                            arg.block_2_ctile_map_))
            {
                throw std::runtime_error(
                    "wrong! GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2 has invalid setting");
            }

            const index_t grid_size =
                arg.block_2_ctile_map_.CalculateGridSize(arg.c_grid_desc_m_n_);

            const auto K0 = arg.a_grid_desc_kbatch_k0_m_k1_.GetLength(I1);

            const bool has_main_k0_block_loop = GridwiseGemm::CalculateHasMainK0BlockLoop(K0);

            float ave_time = 0;

            const auto Run = [&](const auto& kernel) {
                hipGetErrorString(hipMemset(
                    arg.p_c_grid_,
                    0,
                    arg.c_grid_desc_mblock_mperblock_nblock_nperblock_.GetElementSpaceSize() *
                        sizeof(CDataType)));

                ave_time =
                    launch_and_time_kernel(stream_config,
                                           kernel,
                                           dim3(grid_size),
                                           dim3(BlockSize),
                                           0,
                                           arg.p_a_grid_,
                                           arg.p_b_grid_,
                                           arg.p_c_grid_,
                                           arg.a_grid_desc_kbatch_k0_m_k1_,
                                           arg.b_grid_desc_kbatch_k0_n_k1_,
                                           arg.c_grid_desc_mblock_mperblock_nblock_nperblock_,
                                           arg.a_element_op_,
                                           arg.b_element_op_,
                                           arg.c_element_op_,
                                           arg.block_2_ctile_map_);
            };

            if(has_main_k0_block_loop)
            {
                if(kbatch == 1)
                {
                    const auto kernel = kernel_gemm_xdlops_v2r4r2<
                        GridwiseGemm,
                        ADataType, // TODO: distiguish A/B datatype
                        CDataType,
                        remove_reference_t<DeviceGemmXdlSplitKCShuffle::AGridDesc_K0_M_K1>,
                        remove_reference_t<DeviceGemmXdlSplitKCShuffle::BGridDesc_K0_N_K1>,
                        remove_reference_t<DeviceGemmXdlSplitKCShuffle::
                                               CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock>,
                        AElementwiseOperation,
                        BElementwiseOperation,
                        CElementwiseOperation,
                        remove_reference_t<DeviceGemmXdlSplitKCShuffle::Block2CTileMap>,
                        true>;

                    Run(kernel);
                }
                else
                {
                    const auto kernel = kernel_gemm_xdlops_v2r4r2<
                        GridwiseGemmAtomicAdd,
                        ADataType, // TODO: distiguish A/B datatype
                        CDataType,
                        remove_reference_t<DeviceGemmXdlSplitKCShuffle::AGridDesc_K0_M_K1>,
                        remove_reference_t<DeviceGemmXdlSplitKCShuffle::BGridDesc_K0_N_K1>,
                        remove_reference_t<DeviceGemmXdlSplitKCShuffle::
                                               CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock>,
                        AElementwiseOperation,
                        BElementwiseOperation,
                        CElementwiseOperation,
                        remove_reference_t<DeviceGemmXdlSplitKCShuffle::Block2CTileMap>,
                        true>;

                    Run(kernel);
                }
            }
            else
            {
                if(kbatch == 1)
                {
                    const auto kernel = kernel_gemm_xdlops_v2r4r2<
                        GridwiseGemm,
                        ADataType, // TODO: distiguish A/B datatype
                        CDataType,
                        remove_reference_t<DeviceGemmXdlSplitKCShuffle::AGridDesc_K0_M_K1>,
                        remove_reference_t<DeviceGemmXdlSplitKCShuffle::BGridDesc_K0_N_K1>,
                        remove_reference_t<DeviceGemmXdlSplitKCShuffle::
                                               CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock>,
                        AElementwiseOperation,
                        BElementwiseOperation,
                        CElementwiseOperation,
                        remove_reference_t<DeviceGemmXdlSplitKCShuffle::Block2CTileMap>,
                        false>;

                    Run(kernel);
                }
                else
                {
                    const auto kernel = kernel_gemm_xdlops_v2r4r2<
                        GridwiseGemmAtomicAdd,
                        ADataType, // TODO: distiguish A/B datatype
                        CDataType,
                        remove_reference_t<DeviceGemmXdlSplitKCShuffle::AGridDesc_K0_M_K1>,
                        remove_reference_t<DeviceGemmXdlSplitKCShuffle::BGridDesc_K0_N_K1>,
                        remove_reference_t<DeviceGemmXdlSplitKCShuffle::
                                               CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock>,
                        AElementwiseOperation,
                        BElementwiseOperation,
                        CElementwiseOperation,
                        remove_reference_t<DeviceGemmXdlSplitKCShuffle::Block2CTileMap>,
                        false>;

                    Run(kernel);
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
        return GridwiseGemm::CheckValidity(arg.a_grid_desc_kbatch_k0_m_k1_,
                                           arg.b_grid_desc_kbatch_k0_n_k1_,
                                           arg.c_grid_desc_m_n_,
                                           arg.block_2_ctile_map_);
    }

    // polymorphic
    bool IsSupportedArgument(const BaseArgument* p_arg) override
    {
        return IsSupportedArgument(*dynamic_cast<const Argument*>(p_arg));
    }

    static auto MakeArgument(const ADataType* p_a,
                             const BDataType* p_b,
                             CDataType* p_c,
                             index_t M,
                             index_t N,
                             index_t K,
                             index_t StrideA,
                             index_t StrideB,
                             index_t StrideC,
                             AElementwiseOperation a_element_op,
                             BElementwiseOperation b_element_op,
                             CElementwiseOperation c_element_op,
                             index_t KBatch)
    {
        return Argument{p_a,
                        p_b,
                        p_c,
                        M,
                        N,
                        K,
                        StrideA,
                        StrideB,
                        StrideC,
                        1,
                        1,
                        a_element_op,
                        b_element_op,
                        c_element_op,
                        KBatch};
    }

    static auto MakeInvoker() { return Invoker{}; }

    // polymorphic
    std::unique_ptr<BaseArgument> MakeArgumentPointer(const void* p_a,
                                                      const void* p_b,
                                                      void* p_c,
                                                      index_t M,
                                                      index_t N,
                                                      index_t K,
                                                      index_t StrideA,
                                                      index_t StrideB,
                                                      index_t StrideC,
                                                      AElementwiseOperation a_element_op,
                                                      BElementwiseOperation b_element_op,
                                                      CElementwiseOperation c_element_op,
                                                      ck::index_t KBatch = 1) override
    {
        return std::make_unique<Argument>(static_cast<const ADataType*>(p_a),
                                          static_cast<const BDataType*>(p_b),
                                          static_cast<CDataType*>(p_c),
                                          M,
                                          N,
                                          K,
                                          StrideA,
                                          StrideB,
                                          StrideC,
                                          1,
                                          1,
                                          a_element_op,
                                          b_element_op,
                                          c_element_op,
                                          KBatch);
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
        str << "DeviceGemmXdlSplitKCShuffle"
            << "<"
            << BlockSize << ", "
            << MPerBlock << ", "
            << NPerBlock << ", "
            << K0PerBlock
            << ">";
        // clang-format on

        return str.str();
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
