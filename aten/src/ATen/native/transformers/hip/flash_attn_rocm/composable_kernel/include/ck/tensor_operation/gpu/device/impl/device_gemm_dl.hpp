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
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_dl_v1r3.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

template <
    typename ADataType,
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
    index_t BlockSize,
    index_t MPerBlock,
    index_t NPerBlock,
    index_t K0PerBlock,
    index_t K1,
    index_t M1PerThread,
    index_t N1PerThread,
    index_t KPerThread,
    typename M1N1ThreadClusterM1Xs,
    typename M1N1ThreadClusterN1Xs,
    typename ABlockTransferThreadSliceLengths_K0_M0_M1_K1,
    typename ABlockTransferThreadClusterLengths_K0_M0_M1_K1,
    typename ABlockTransferThreadClusterArrangeOrder,
    typename ABlockTransferSrcAccessOrder,
    typename ABlockTransferSrcVectorTensorLengths_K0_M0_M1_K1,
    typename ABlockTransferSrcVectorTensorContiguousDimOrder,
    typename ABlockTransferDstVectorTensorLengths_K0_M0_M1_K1,
    typename BBlockTransferThreadSliceLengths_K0_N0_N1_K1,
    typename BBlockTransferThreadClusterLengths_K0_N0_N1_K1,
    typename BBlockTransferThreadClusterArrangeOrder,
    typename BBlockTransferSrcAccessOrder,
    typename BBlockTransferSrcVectorTensorLengths_K0_N0_N1_K1,
    typename BBlockTransferSrcVectorTensorContiguousDimOrder,
    typename BBlockTransferDstVectorTensorLengths_K0_N0_N1_K1,
    typename CThreadTransferSrcDstAccessOrder,
    index_t CThreadTransferSrcDstVectorDim,
    index_t CThreadTransferDstScalarPerVector,
    enable_if_t<
        is_same_v<AElementwiseOperation, ck::tensor_operation::element_wise::PassThrough> &&
            is_same_v<BElementwiseOperation, ck::tensor_operation::element_wise::PassThrough> &&
            is_same_v<CElementwiseOperation, ck::tensor_operation::element_wise::PassThrough>,
        bool> = false>
struct DeviceGemmDl : public DeviceGemm<ALayout,
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
    static constexpr auto I4 = Number<4>{};
    static constexpr auto I5 = Number<5>{};

    static constexpr auto K1Number = Number<K1>{};

    static auto MakeAGridDescriptor_K0_M_K1(index_t M, index_t K, index_t StrideA)
    {
        assert(K % K1 == 0);

        const index_t K0 = K / K1;

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

        if constexpr(GemmSpec == GemmSpecialization::MNPadding)
        {
            const auto PadM = (MPerBlock - M % MPerBlock) % MPerBlock;

            return transform_tensor_descriptor(
                a_grid_desc_m_k,
                make_tuple(make_unmerge_transform(make_tuple(K0, K1Number)),
                           make_right_pad_transform(M, PadM)),
                make_tuple(Sequence<1>{}, Sequence<0>{}),
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}));
        }
        else
        {
            return transform_tensor_descriptor(
                a_grid_desc_m_k,
                make_tuple(make_unmerge_transform(make_tuple(K0, K1Number)),
                           make_pass_through_transform(M)),
                make_tuple(Sequence<1>{}, Sequence<0>{}),
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}));
        }
    }

    static auto MakeBGridDescriptor_K0_N_K1(index_t K, index_t N, index_t StrideB)
    {
        assert(K % K1 == 0);

        const index_t K0 = K / K1;

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

        if constexpr(GemmSpec == GemmSpecialization::MNPadding)
        {
            const auto PadN = (NPerBlock - N % NPerBlock) % NPerBlock;

            return transform_tensor_descriptor(
                b_grid_desc_k_n,
                make_tuple(make_unmerge_transform(make_tuple(K0, K1Number)),
                           make_right_pad_transform(N, PadN)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}));
        }
        else
        {
            return transform_tensor_descriptor(
                b_grid_desc_k_n,
                make_tuple(make_unmerge_transform(make_tuple(K0, K1Number)),
                           make_pass_through_transform(N)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}));
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

    using AGridDesc_K0_M_K1 = decltype(MakeAGridDescriptor_K0_M_K1(1, 1, 1));
    using BGridDesc_K0_N_K1 = decltype(MakeBGridDescriptor_K0_N_K1(1, 1, 1));
    using CGridDesc_M_N     = decltype(MakeCGridDescriptor_M_N(1, 1, 1));

    // GridwiseGemm
    using GridwiseGemm =
        GridwiseGemmDl_km_kn_mn_v1r3<BlockSize,
                                     ADataType,
                                     AccDataType,
                                     CDataType,
                                     InMemoryDataOperationEnum::Set,
                                     AGridDesc_K0_M_K1,
                                     BGridDesc_K0_N_K1,
                                     CGridDesc_M_N,
                                     MPerBlock,
                                     NPerBlock,
                                     K0PerBlock,
                                     K1,
                                     M1PerThread,
                                     N1PerThread,
                                     KPerThread,
                                     M1N1ThreadClusterM1Xs,
                                     M1N1ThreadClusterN1Xs,
                                     ABlockTransferThreadSliceLengths_K0_M0_M1_K1,
                                     ABlockTransferThreadClusterLengths_K0_M0_M1_K1,
                                     ABlockTransferThreadClusterArrangeOrder,
                                     ABlockTransferSrcAccessOrder,
                                     ABlockTransferSrcVectorTensorLengths_K0_M0_M1_K1,
                                     ABlockTransferSrcVectorTensorContiguousDimOrder,
                                     ABlockTransferDstVectorTensorLengths_K0_M0_M1_K1,
                                     BBlockTransferThreadSliceLengths_K0_N0_N1_K1,
                                     BBlockTransferThreadClusterLengths_K0_N0_N1_K1,
                                     BBlockTransferThreadClusterArrangeOrder,
                                     BBlockTransferSrcAccessOrder,
                                     BBlockTransferSrcVectorTensorLengths_K0_N0_N1_K1,
                                     BBlockTransferSrcVectorTensorContiguousDimOrder,
                                     BBlockTransferDstVectorTensorLengths_K0_N0_N1_K1,
                                     CThreadTransferSrcDstAccessOrder,
                                     CThreadTransferSrcDstVectorDim,
                                     CThreadTransferDstScalarPerVector>;

    using AGridDesc_K0_M0_M1_K1 =
        decltype(GridwiseGemm::MakeAGridDescriptor_K0_M0_M1_K1(AGridDesc_K0_M_K1{}));
    using BGridDesc_K0_N0_N1_K1 =
        decltype(GridwiseGemm::MakeBGridDescriptor_K0_N0_N1_K1(BGridDesc_K0_N_K1{}));
    using CGridDesc_M0_M10_M11_N0_N10_N11 =
        decltype(GridwiseGemm::MakeCGridDescriptor_M0_M10_M11_N0_N10_N11(CGridDesc_M_N{}));
    using DefaultBlock2CTileMap =
        decltype(GridwiseGemm::MakeDefaultBlock2CTileMap(CGridDesc_M_N{}));

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
                 CElementwiseOperation c_element_op)
            : p_a_grid_{p_a_grid},
              p_b_grid_{p_b_grid},
              p_c_grid_{p_c_grid},
              a_grid_desc_k0_m0_m1_k1_{},
              b_grid_desc_k0_n0_n1_k1_{},
              c_grid_desc_m0_m10_m11_n0_n10_n11_{},
              block_2_ctile_map_{},
              M01_{M01},
              N01_{N01},
              a_element_op_{a_element_op},
              b_element_op_{b_element_op},
              c_element_op_{c_element_op}
        {
            a_grid_desc_k0_m_k1_ = DeviceGemmDl::MakeAGridDescriptor_K0_M_K1(M, K, StrideA);
            b_grid_desc_k0_n_k1_ = DeviceGemmDl::MakeBGridDescriptor_K0_N_K1(K, N, StrideB);
            c_grid_desc_m_n_     = DeviceGemmDl::MakeCGridDescriptor_M_N(M, N, StrideC);

            if(GridwiseGemm::CheckValidity(
                   a_grid_desc_k0_m_k1_, b_grid_desc_k0_n_k1_, c_grid_desc_m_n_))
            {
                a_grid_desc_k0_m0_m1_k1_ =
                    GridwiseGemm::MakeAGridDescriptor_K0_M0_M1_K1(a_grid_desc_k0_m_k1_);
                b_grid_desc_k0_n0_n1_k1_ =
                    GridwiseGemm::MakeBGridDescriptor_K0_N0_N1_K1(b_grid_desc_k0_n_k1_);
                c_grid_desc_m0_m10_m11_n0_n10_n11_ =
                    GridwiseGemm::MakeCGridDescriptor_M0_M10_M11_N0_N10_N11(c_grid_desc_m_n_);

                block_2_ctile_map_ = GridwiseGemm::MakeDefaultBlock2CTileMap(c_grid_desc_m_n_);
            }
        }

        //  private:
        const ADataType* p_a_grid_;
        const BDataType* p_b_grid_;
        CDataType* p_c_grid_;

        AGridDesc_K0_M_K1 a_grid_desc_k0_m_k1_;
        BGridDesc_K0_N_K1 b_grid_desc_k0_n_k1_;
        CGridDesc_M_N c_grid_desc_m_n_;

        AGridDesc_K0_M0_M1_K1 a_grid_desc_k0_m0_m1_k1_;
        BGridDesc_K0_N0_N1_K1 b_grid_desc_k0_n0_n1_k1_;
        CGridDesc_M0_M10_M11_N0_N10_N11 c_grid_desc_m0_m10_m11_n0_n10_n11_;

        DefaultBlock2CTileMap block_2_ctile_map_;

        // TODO: unused, but may be useful in future.
        index_t M01_;
        index_t N01_;

        // TODO: unused since gridwise_gemm_dl_v1r3 does NOT support prologue for the time being.
        AElementwiseOperation a_element_op_;
        BElementwiseOperation b_element_op_;
        CElementwiseOperation c_element_op_;
    };

    // Invoker
    struct Invoker : public BaseInvoker
    {
        using Argument = DeviceGemmDl::Argument;

        float Run(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
#if DEBUG_LOG
            {
                std::cout << "arg.a_grid_desc_k0_m0_m1_k1_{"
                          << arg.a_grid_desc_k0_m_k1_.GetLength(I0) << ", "
                          << arg.a_grid_desc_k0_m_k1_.GetLength(I1) << ", "
                          << arg.a_grid_desc_k0_m_k1_.GetLength(I2) << "}" << std::endl;

                std::cout << "arg.b_grid_desc_k0_n0_n1_k1_{"
                          << arg.b_grid_desc_k0_n_k1_.GetLength(I0) << ", "
                          << arg.b_grid_desc_k0_n_k1_.GetLength(I1) << ", "
                          << arg.b_grid_desc_k0_n_k1_.GetLength(I2) << "}" << std::endl;

                std::cout << "arg.c_grid_desc_m_n_{ " << arg.c_grid_desc_m_n_.GetLength(I0) << ", "
                          << arg.c_grid_desc_m_n_.GetLength(I1) << "}" << std::endl;
            }
#endif

            if(!GridwiseGemm::CheckValidity(
                   arg.a_grid_desc_k0_m_k1_, arg.b_grid_desc_k0_n_k1_, arg.c_grid_desc_m_n_))
            {
                throw std::runtime_error(
                    "wrong! GridwiseGemm_k0mk1_k0nk1_mn_xdl_v2r3 has invalid setting");
            }

            const index_t grid_size = GridwiseGemm::CalculateGridSize(
                arg.c_grid_desc_m_n_.GetLength(I0), arg.c_grid_desc_m_n_.GetLength(I1));

            const auto K0                    = arg.a_grid_desc_k0_m0_m1_k1_.GetLength(I0);
            const bool has_main_k_block_loop = GridwiseGemm::CalculateHasMainKBlockLoop(K0);
            const bool has_double_tail_k_block_loop =
                GridwiseGemm::CalculateHasDoubleTailKBlockLoop(K0);

            float ave_time = 0;

            if(has_main_k_block_loop && has_double_tail_k_block_loop)
            {
                const auto kernel =
                    kernel_gemm_dl_v1r3<GridwiseGemm,
                                        ADataType,
                                        CDataType,
                                        remove_reference_t<AGridDesc_K0_M0_M1_K1>,
                                        remove_reference_t<BGridDesc_K0_N0_N1_K1>,
                                        remove_reference_t<CGridDesc_M0_M10_M11_N0_N10_N11>,
                                        remove_reference_t<DefaultBlock2CTileMap>,
                                        true,
                                        true>;

                ave_time = launch_and_time_kernel(stream_config,
                                                  kernel,
                                                  dim3(grid_size),
                                                  dim3(BlockSize),
                                                  0,
                                                  arg.p_a_grid_,
                                                  arg.p_b_grid_,
                                                  arg.p_c_grid_,
                                                  arg.a_grid_desc_k0_m0_m1_k1_,
                                                  arg.b_grid_desc_k0_n0_n1_k1_,
                                                  arg.c_grid_desc_m0_m10_m11_n0_n10_n11_,
                                                  arg.block_2_ctile_map_);
            }
            else if(has_main_k_block_loop && !has_double_tail_k_block_loop)
            {
                const auto kernel =
                    kernel_gemm_dl_v1r3<GridwiseGemm,
                                        ADataType,
                                        CDataType,
                                        remove_reference_t<AGridDesc_K0_M0_M1_K1>,
                                        remove_reference_t<BGridDesc_K0_N0_N1_K1>,
                                        remove_reference_t<CGridDesc_M0_M10_M11_N0_N10_N11>,
                                        remove_reference_t<DefaultBlock2CTileMap>,
                                        true,
                                        false>;

                ave_time = launch_and_time_kernel(stream_config,
                                                  kernel,
                                                  dim3(grid_size),
                                                  dim3(BlockSize),
                                                  0,
                                                  arg.p_a_grid_,
                                                  arg.p_b_grid_,
                                                  arg.p_c_grid_,
                                                  arg.a_grid_desc_k0_m0_m1_k1_,
                                                  arg.b_grid_desc_k0_n0_n1_k1_,
                                                  arg.c_grid_desc_m0_m10_m11_n0_n10_n11_,
                                                  arg.block_2_ctile_map_);
            }
            else if(!has_main_k_block_loop && has_double_tail_k_block_loop)
            {
                const auto kernel =
                    kernel_gemm_dl_v1r3<GridwiseGemm,
                                        ADataType,
                                        CDataType,
                                        remove_reference_t<AGridDesc_K0_M0_M1_K1>,
                                        remove_reference_t<BGridDesc_K0_N0_N1_K1>,
                                        remove_reference_t<CGridDesc_M0_M10_M11_N0_N10_N11>,
                                        remove_reference_t<DefaultBlock2CTileMap>,
                                        false,
                                        true>;

                ave_time = launch_and_time_kernel(stream_config,
                                                  kernel,
                                                  dim3(grid_size),
                                                  dim3(BlockSize),
                                                  0,
                                                  arg.p_a_grid_,
                                                  arg.p_b_grid_,
                                                  arg.p_c_grid_,
                                                  arg.a_grid_desc_k0_m0_m1_k1_,
                                                  arg.b_grid_desc_k0_n0_n1_k1_,
                                                  arg.c_grid_desc_m0_m10_m11_n0_n10_n11_,
                                                  arg.block_2_ctile_map_);
            }
            else
            {
                const auto kernel =
                    kernel_gemm_dl_v1r3<GridwiseGemm,
                                        ADataType,
                                        CDataType,
                                        remove_reference_t<AGridDesc_K0_M0_M1_K1>,
                                        remove_reference_t<BGridDesc_K0_N0_N1_K1>,
                                        remove_reference_t<CGridDesc_M0_M10_M11_N0_N10_N11>,
                                        remove_reference_t<DefaultBlock2CTileMap>,
                                        false,
                                        false>;

                ave_time = launch_and_time_kernel(stream_config,
                                                  kernel,
                                                  dim3(grid_size),
                                                  dim3(BlockSize),
                                                  0,
                                                  arg.p_a_grid_,
                                                  arg.p_b_grid_,
                                                  arg.p_c_grid_,
                                                  arg.a_grid_desc_k0_m0_m1_k1_,
                                                  arg.b_grid_desc_k0_n0_n1_k1_,
                                                  arg.c_grid_desc_m0_m10_m11_n0_n10_n11_,
                                                  arg.block_2_ctile_map_);
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
        if(ck::get_device_name() == "gfx906" || ck::get_device_name() == "gfx1030")
        {
            return GridwiseGemm::CheckValidity(
                arg.a_grid_desc_k0_m_k1_, arg.b_grid_desc_k0_n_k1_, arg.c_grid_desc_m_n_);
        }
        else
        {
            return false;
        }
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
                             CElementwiseOperation c_element_op)
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
                        c_element_op};
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
                                                      CElementwiseOperation c_element_op) override
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
        str << "DeviceGemmDl"
            << "<"
            << BlockSize << ", "
            << MPerBlock << ", "
            << NPerBlock << ", "
            << K0PerBlock << ", "
            << K1 << ", "
            << M1PerThread << ", "
            << N1PerThread << ", "
            << KPerThread
            << ">";
        // clang-format on

        return str.str();
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
