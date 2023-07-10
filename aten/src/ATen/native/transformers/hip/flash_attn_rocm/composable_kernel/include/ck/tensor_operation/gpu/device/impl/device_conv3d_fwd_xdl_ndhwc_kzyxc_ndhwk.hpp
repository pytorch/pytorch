// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#ifndef DEVICE_CONV3D_FWD_XDL_HPP
#define DEVICE_CONV3D_FWD_XDL_HPP

#include <iostream>
#include <memory>
#include <sstream>
#include "device.hpp"
#include "device_conv_fwd.hpp"
#include "common_header.hpp"
#include "tensor_layout.hpp"
#include "convolution_forward_specialization.hpp"
#include "tensor_descriptor.hpp"
#include "tensor_descriptor_helper.hpp"
#include "transform_forward_convolution3d_into_gemm_v4r4r4_ndhwc_kzyxc_ndhwk.hpp"
#include "gridwise_gemm_xdlops_v2r3.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

/*
 * \see \link impl/device_batched_gemm_xdl.hpp kernel_batched_gemm_xdlops_v2r3() \endlink.
 */
template <typename GridwiseGemm,
          typename FloatAB,
          typename FloatC,
          typename AGridDesc_K0_M_K1,
          typename BGridDesc_K0_N_K1,
          typename CGridDesc_M0_N0_M1_N1_M2_M3_M4_N2,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation,
          typename Block2CTileMap,
          bool HasMainKBlockLoop>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
    __launch_bounds__(CK_MAX_THREAD_PER_BLOCK, CK_MIN_BLOCK_PER_CU)
#endif
        kernel_gemm_xdlops_v2r3_for_conv3d(
            const FloatAB* __restrict__ p_a_grid,
            const FloatAB* __restrict__ p_b_grid,
            FloatC* __restrict__ p_c_grid,
            const index_t num_batches,
            const index_t a_batch_stride,
            const index_t b_batch_stride,
            const index_t c_batch_stride,
            const AGridDesc_K0_M_K1 a_grid_desc_k0_m_k1,
            const BGridDesc_K0_N_K1 b_grid_desc_k0_n_k1,
            const CGridDesc_M0_N0_M1_N1_M2_M3_M4_N2 c_grid_desc_m0_n0_m1_n1_m2_m3_m4_n2,
            const AElementwiseOperation a_element_op,
            const BElementwiseOperation b_element_op,
            const CElementwiseOperation c_element_op,
            const Block2CTileMap block_2_ctile_map)
{
#if(!defined(__HIP_DEVICE_COMPILE__) || defined(__gfx908__) || defined(__gfx90a__))
    const index_t num_blocks_per_batch =
        __builtin_amdgcn_readfirstlane(get_grid_size() / num_batches);
    const index_t g_idx = __builtin_amdgcn_readfirstlane(get_block_1d_id() / num_blocks_per_batch);

    const long_index_t a_batch_offset =
        __builtin_amdgcn_readfirstlane(static_cast<long_index_t>(a_batch_stride) * g_idx);
    const long_index_t b_batch_offset =
        __builtin_amdgcn_readfirstlane(static_cast<long_index_t>(b_batch_stride) * g_idx);
    const long_index_t c_batch_offset =
        __builtin_amdgcn_readfirstlane(static_cast<long_index_t>(c_batch_stride) * g_idx);

    __shared__ char p_shared[GridwiseGemm::GetSharedMemoryNumberOfByte()];

    GridwiseGemm::template Run<HasMainKBlockLoop>(p_a_grid + a_batch_offset,
                                                  p_b_grid + b_batch_offset,
                                                  p_c_grid + c_batch_offset,
                                                  p_shared,
                                                  a_grid_desc_k0_m_k1,
                                                  b_grid_desc_k0_n_k1,
                                                  c_grid_desc_m0_n0_m1_n1_m2_m3_m4_n2,
                                                  a_element_op,
                                                  b_element_op,
                                                  c_element_op,
                                                  block_2_ctile_map);

#else
    ignore = p_a_grid;
    ignore = p_b_grid;
    ignore = p_c_grid;
    ignore = num_batches;
    ignore = a_batch_stride;
    ignore = b_batch_stride;
    ignore = c_batch_stride;
    ignore = a_grid_desc_k0_m_k1;
    ignore = b_grid_desc_k0_n_k1;
    ignore = c_grid_desc_m0_n0_m1_n1_m2_m3_m4_n2;
    ignore = a_element_op;
    ignore = b_element_op;
    ignore = c_element_op;
    ignore = block_2_ctile_map;
#endif // end of if (defined(__gfx908__) || defined(__gfx90a__))
}

// specialization for #D conv: in[n, di, hi, wi, c] * wei[k, z, y, x, c] = out[n, do, ho, wo, k]
template <typename InDataType,
          typename WeiDataType, // WeiDataType must be the same as InDataType
          typename OutDataType,
          typename AccDataType,
          typename InElementwiseOperation,
          typename WeiElementwiseOperation,
          typename OutElementwiseOperation,
          ConvolutionForwardSpecialization ConvForwardSpecialization,
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
          ck::index_t CThreadTransferSrcDstVectorDim,
          ck::index_t CThreadTransferDstScalarPerVector>
struct DeviceConv3dFwdXdl_Input_N_Di_Hi_Wi_C_Weight_K_Z_Y_X_C_Output_N_Do_Ho_Wo_K
    : public DeviceConvFwd<InElementwiseOperation, WeiElementwiseOperation, OutElementwiseOperation>

{
    using DeviceOp = DeviceConv3dFwdXdl_Input_N_Di_Hi_Wi_C_Weight_K_Z_Y_X_C_Output_N_Do_Ho_Wo_K;

    using ADataType = InDataType;
    using BDataType = WeiDataType;
    using CDataType = OutDataType;
    // TODO make A/B datatype different
    using ABDataType = InDataType;

    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};

    /*
     * \brief Split the number of batches, \p N, into N = B * N1, such that the memory
     * space of input and output tensors stays with the value range of index_t, and each subbatch
     * can be dealed with GridwiseGemm.
     */
    static index_t GetMaxAllowableSubBatchSize(const index_t N,
                                               const index_t K,
                                               const index_t C,
                                               std::vector<ck::index_t> input_spatial_lengths,
                                               std::vector<ck::index_t> output_spatial_lengths)
    {
        const index_t Di = input_spatial_lengths[0];
        const index_t Hi = input_spatial_lengths[1];
        const index_t Wi = input_spatial_lengths[2];

        const index_t Do = output_spatial_lengths[0];
        const index_t Ho = output_spatial_lengths[1];
        const index_t Wo = output_spatial_lengths[2];

        // N1 should satisfy that
        //   1) N % N1 = 0;
        //   2) N1 * (Do * Ho * Wo * K) < (2^31 - 1)
        //   3) N1 * (Di * Hi * Wi * C) < (2^31 - 1)
        //
        // Do NOT confuse (B, N1) in this function with (B, N1) in gridewise GEMM.
        auto N1 = N + 1;

        const auto stride =
            math::max(long_index_t(Do) * Ho * Wo * K, long_index_t(Di) * Hi * Wi * C);
        const index_t max_stride = NumericLimits<index_t>::Max();

        for(index_t n0 = 1; n0 <= N; ++n0)
        {
            index_t n1 = N / n0;
            if(n0 * n1 == N && long_index_t(n1) * long_index_t(stride) < max_stride)
            {
                N1 = n1;
                break;
            }
        }

        const auto B = N / N1;
        if(B * N1 != N)
        {
            throw std::runtime_error(__func__ +
                                     std::string(": failed to find num_subbatches for conv3d.\n"));
        }

        return N1;
    }

    static auto
    MakeABCGridDescriptor_A_K0_M_K1_B_K0_N_K1_C_M_N(const index_t N,
                                                    const index_t K,
                                                    const index_t C,
                                                    std::vector<ck::index_t> input_spatial_lengths,
                                                    std::vector<ck::index_t> filter_spatial_lengths,
                                                    std::vector<ck::index_t> output_spatial_lengths,
                                                    std::vector<ck::index_t> conv_filter_strides,
                                                    std::vector<ck::index_t> conv_filter_dilations,
                                                    std::vector<ck::index_t> input_left_pads,
                                                    std::vector<ck::index_t> input_right_pads)
    {
        assert(input_spatial_lengths.size() > 2);
        assert(filter_spatial_lengths.size() > 2);
        assert(conv_filter_strides.size() > 2);
        assert(conv_filter_dilations.size() > 2);
        assert(input_left_pads.size() > 2);
        assert(input_right_pads.size() > 2);

        const index_t Di = input_spatial_lengths[0];
        const index_t Hi = input_spatial_lengths[1];
        const index_t Wi = input_spatial_lengths[2];
        const index_t Z  = filter_spatial_lengths[0];
        const index_t Y  = filter_spatial_lengths[1];
        const index_t X  = filter_spatial_lengths[2];

        const index_t Do = output_spatial_lengths[0];
        const index_t Ho = output_spatial_lengths[1];
        const index_t Wo = output_spatial_lengths[2];

        static_assert(ConvForwardSpecialization == ConvolutionForwardSpecialization::Default,
                      "Wrong! This specialization not implemented!");

        const auto in_desc_n_di_hi_wi_c =
            make_naive_tensor_descriptor_packed(make_tuple(N, Di, Hi, Wi, C));
        const auto wei_desc_k_z_y_x_c =
            make_naive_tensor_descriptor_packed(make_tuple(K, Z, Y, X, C));
        const auto out_desc_n_do_ho_wo_k =
            make_naive_tensor_descriptor_packed(make_tuple(N, Do, Ho, Wo, K));

        const auto descs = transform_forward_convolution3d_into_gemm_v4r4r4_ndhwc_kzyxc_ndhwk_pad(
            in_desc_n_di_hi_wi_c,
            wei_desc_k_z_y_x_c,
            out_desc_n_do_ho_wo_k,
            make_tuple(conv_filter_strides[0], conv_filter_strides[1], conv_filter_strides[2]),
            make_tuple(
                conv_filter_dilations[0], conv_filter_dilations[1], conv_filter_dilations[2]),
            make_tuple(input_left_pads[0], input_left_pads[1], input_left_pads[2]),
            make_tuple(input_right_pads[0], input_right_pads[1], input_right_pads[2]),
            Number<K1>{});

        return descs;
    }

    using ABCGridDescs = remove_cvref_t<decltype(MakeABCGridDescriptor_A_K0_M_K1_B_K0_N_K1_C_M_N(
        1, 1, 1, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}))>;

    using AGridDesc_K0_M_K1 = remove_cvref_t<decltype(ABCGridDescs{}[I0])>;
    using BGridDesc_K0_N_K1 = remove_cvref_t<decltype(ABCGridDescs{}[I1])>;
    using CGridDesc_M_N     = remove_cvref_t<decltype(ABCGridDescs{}[I2])>;

    using GridwiseGemm = GridwiseGemm_k0mk1_k0nk1_mn_xdlops_v2r3<
        BlockSize,
        InDataType,
        AccDataType,
        OutDataType,
        InMemoryDataOperationEnum::Set,
        AGridDesc_K0_M_K1,
        BGridDesc_K0_N_K1,
        CGridDesc_M_N,
        InElementwiseOperation,
        WeiElementwiseOperation,
        OutElementwiseOperation,
        MPerBlock,
        NPerBlock,
        K0PerBlock,
        MPerXDL,
        NPerXDL,
        K1,
        MXdlPerWave,
        NXdlPerWave,
        ABlockTransferThreadClusterLengths_K0_M_K1,
        Sequence<1, 0, 2>, // ABlockTransferThreadClusterArrangeOrder,
        Sequence<1, 0, 2>, // ABlockTransferSrcAccessOrder,
        2,
        ABlockTransferSrcScalarPerVector,
        ABlockTransferDstScalarPerVector_K1,
        false, // AThreadTransferSrcResetCoordinateAfterRun,
        ABlockLdsAddExtraM,
        BBlockTransferThreadClusterLengths_K0_N_K1,
        Sequence<1, 0, 2>, // ABlockTransferThreadClusterArrangeOrder,
        Sequence<1, 0, 2>, // ABlockTransferSrcAccessOrder,
        2,
        BBlockTransferSrcScalarPerVector,
        BBlockTransferDstScalarPerVector_K1,
        false, // BThreadTransferSrcResetCoordinateAfterRun,
        BBlockLdsAddExtraN,
        Sequence<2, 3, 0, 1, 7, 5, 4, 6>,
        7,
        CThreadTransferDstScalarPerVector>;

    using CGridDesc_M0_N0_M1_N1_M2_M3_M4_N2 =
        decltype(GridwiseGemm::MakeCGridDescriptor_M0_N0_M1_N1_M2_M3_M4_N2(CGridDesc_M_N{}));
    using Block2CTileMap = typename GridwiseGemm::DefaultBlock2CTileMap;

    // Argument
    struct Argument : public BaseArgument
    {
        Argument(const InDataType* p_in,
                 const WeiDataType* p_wei,
                 OutDataType* p_out,
                 const index_t N,
                 const index_t K,
                 const index_t C,
                 std::vector<ck::index_t> input_spatial_lengths,
                 std::vector<ck::index_t> filter_spatial_lengths,
                 std::vector<ck::index_t> output_spatial_lengths,
                 std::vector<ck::index_t> conv_filter_strides,
                 std::vector<ck::index_t> conv_filter_dilations,
                 std::vector<ck::index_t> input_left_pads,
                 std::vector<ck::index_t> input_right_pads,
                 index_t M01,
                 index_t N01,
                 InElementwiseOperation in_element_op,
                 WeiElementwiseOperation wei_element_op,
                 OutElementwiseOperation out_element_op)
            : p_a_grid_{p_in},
              p_b_grid_{p_wei},
              p_c_grid_{p_out},
              M01_{M01},
              N01_{N01},
              in_element_op_{in_element_op},
              wei_element_op_{wei_element_op},
              out_element_op_{out_element_op}
        {
            const index_t subbatch_size =
                GetMaxAllowableSubBatchSize(N, K, C, input_spatial_lengths, output_spatial_lengths);
            num_subbatches_ = N / subbatch_size;

            const auto descs =
                MakeABCGridDescriptor_A_K0_M_K1_B_K0_N_K1_C_M_N(subbatch_size,
                                                                K,
                                                                C,
                                                                input_spatial_lengths,
                                                                filter_spatial_lengths,
                                                                output_spatial_lengths,
                                                                conv_filter_strides,
                                                                conv_filter_dilations,
                                                                input_left_pads,
                                                                input_right_pads);

            a_grid_desc_k0_m_k1_ = descs[I0];
            b_grid_desc_k0_n_k1_ = descs[I1];
            c_grid_desc_m_n_     = descs[I2];

            block_2_ctile_map_ =
                GridwiseGemm::MakeDefaultBlock2CTileMap(c_grid_desc_m_n_, M01, N01);

            a_batch_stride_ = a_grid_desc_k0_m_k1_.GetElementSpaceSize();
            b_batch_stride_ = 0;
            c_batch_stride_ = c_grid_desc_m_n_.GetElementSpaceSize();

            if(GridwiseGemm::CheckValidity(a_grid_desc_k0_m_k1_,
                                           b_grid_desc_k0_n_k1_,
                                           c_grid_desc_m_n_,
                                           block_2_ctile_map_))
            {
                c_grid_desc_m0_n0_m1_n1_m2_m3_m4_n2_ =
                    GridwiseGemm::MakeCGridDescriptor_M0_N0_M1_N1_M2_M3_M4_N2(c_grid_desc_m_n_);
            }
        }

        //  private:
        const InDataType* p_a_grid_;
        const WeiDataType* p_b_grid_;
        OutDataType* p_c_grid_;
        index_t num_subbatches_;
        index_t a_batch_stride_;
        index_t b_batch_stride_;
        index_t c_batch_stride_;
        AGridDesc_K0_M_K1 a_grid_desc_k0_m_k1_;
        BGridDesc_K0_N_K1 b_grid_desc_k0_n_k1_;
        CGridDesc_M_N c_grid_desc_m_n_;
        CGridDesc_M0_N0_M1_N1_M2_M3_M4_N2 c_grid_desc_m0_n0_m1_n1_m2_m3_m4_n2_;
        Block2CTileMap block_2_ctile_map_;
        index_t M01_;
        index_t N01_;
        InElementwiseOperation in_element_op_;
        WeiElementwiseOperation wei_element_op_;
        OutElementwiseOperation out_element_op_;
    };

    // Invoker
    struct Invoker : public BaseInvoker
    {
        using Argument = DeviceOp::Argument;

        float Run(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
#if DEBUG_LOG
            {
                std::cout << "num_batches_of_GEMM = " << arg.num_subbatches_ << std::endl;
                std::cout << "a_grid_desc_k0_m_k1{" << arg.a_grid_desc_k0_m_k1_.GetLength(I0)
                          << ", " << arg.a_grid_desc_k0_m_k1_.GetLength(I1) << ", "
                          << arg.a_grid_desc_k0_m_k1_.GetLength(I2) << "}" << std::endl;

                std::cout << "b_grid_desc_k0_n_k1{" << arg.b_grid_desc_k0_n_k1_.GetLength(I0)
                          << ", " << arg.b_grid_desc_k0_n_k1_.GetLength(I1) << ", "
                          << arg.b_grid_desc_k0_n_k1_.GetLength(I2) << "}" << std::endl;

                std::cout << "c_grid_desc_m_n{ " << arg.c_grid_desc_m_n_.GetLength(I0) << ", "
                          << arg.c_grid_desc_m_n_.GetLength(I1) << "}" << std::endl;
            }
#endif

            if(!GridwiseGemm::CheckValidity(arg.a_grid_desc_k0_m_k1_,
                                            arg.b_grid_desc_k0_n_k1_,
                                            arg.c_grid_desc_m_n_,
                                            arg.block_2_ctile_map_))
            {
                throw std::runtime_error(
                    "wrong! GridwiseGemm_k0mk1_k0nk1_mn_xdlops_v2r3 has invalid setting");
            }

            const index_t grid_size =
                arg.block_2_ctile_map_.CalculateGridSize(arg.c_grid_desc_m_n_) *
                arg.num_subbatches_;

            const auto K0 = arg.a_grid_desc_k0_m_k1_.GetLength(I0);

            const bool has_main_k0_block_loop = GridwiseGemm::CalculateHasMainK0BlockLoop(K0);

            float ave_time = 0;
            if(has_main_k0_block_loop)
            {
                const auto kernel = kernel_gemm_xdlops_v2r3_for_conv3d<
                    GridwiseGemm,
                    InDataType,
                    OutDataType,
                    remove_reference_t<AGridDesc_K0_M_K1>,
                    remove_reference_t<BGridDesc_K0_N_K1>,
                    remove_reference_t<CGridDesc_M0_N0_M1_N1_M2_M3_M4_N2>,
                    InElementwiseOperation,
                    WeiElementwiseOperation,
                    OutElementwiseOperation,
                    remove_reference_t<Block2CTileMap>,
                    true>;
                ave_time = launch_and_time_kernel(stream_config,
                                                  kernel,
                                                  dim3(grid_size),
                                                  dim3(BlockSize),
                                                  0,
                                                  arg.p_a_grid_,
                                                  arg.p_b_grid_,
                                                  arg.p_c_grid_,
                                                  arg.num_subbatches_,
                                                  arg.a_batch_stride_,
                                                  arg.b_batch_stride_,
                                                  arg.c_batch_stride_,
                                                  arg.a_grid_desc_k0_m_k1_,
                                                  arg.b_grid_desc_k0_n_k1_,
                                                  arg.c_grid_desc_m0_n0_m1_n1_m2_m3_m4_n2_,
                                                  arg.in_element_op_,
                                                  arg.wei_element_op_,
                                                  arg.out_element_op_,
                                                  arg.block_2_ctile_map_);
            }
            else
            {
                const auto kernel = kernel_gemm_xdlops_v2r3_for_conv3d<
                    GridwiseGemm,
                    InDataType,
                    OutDataType,
                    remove_reference_t<AGridDesc_K0_M_K1>,
                    remove_reference_t<BGridDesc_K0_N_K1>,
                    remove_reference_t<CGridDesc_M0_N0_M1_N1_M2_M3_M4_N2>,
                    InElementwiseOperation,
                    WeiElementwiseOperation,
                    OutElementwiseOperation,
                    remove_reference_t<Block2CTileMap>,
                    false>;

                ave_time = launch_and_time_kernel(stream_config,
                                                  kernel,
                                                  dim3(grid_size),
                                                  dim3(BlockSize),
                                                  0,
                                                  arg.p_a_grid_,
                                                  arg.p_b_grid_,
                                                  arg.p_c_grid_,
                                                  arg.num_subbatches_,
                                                  arg.a_batch_stride_,
                                                  arg.b_batch_stride_,
                                                  arg.c_batch_stride_,
                                                  arg.a_grid_desc_k0_m_k1_,
                                                  arg.b_grid_desc_k0_n_k1_,
                                                  arg.c_grid_desc_m0_n0_m1_n1_m2_m3_m4_n2_,
                                                  arg.in_element_op_,
                                                  arg.wei_element_op_,
                                                  arg.out_element_op_,
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
        return GridwiseGemm::CheckValidity(arg.a_grid_desc_k0_m_k1_,
                                           arg.b_grid_desc_k0_n_k1_,
                                           arg.c_grid_desc_m_n_,
                                           arg.block_2_ctile_map_);
    }

    // polymorphic
    bool IsSupportedArgument(const BaseArgument* p_arg) override
    {
        return IsSupportedArgument(*dynamic_cast<const Argument*>(p_arg));
    }

    static auto MakeArgument(const InDataType* p_in,
                             const WeiDataType* p_wei,
                             OutDataType* p_out,
                             const index_t N,
                             const index_t K,
                             const index_t C,
                             std::vector<ck::index_t> input_spatial_lengths,
                             std::vector<ck::index_t> filter_spatial_lengths,
                             std::vector<ck::index_t> output_spatial_lengths,
                             std::vector<ck::index_t> conv_filter_strides,
                             std::vector<ck::index_t> conv_filter_dilations,
                             std::vector<ck::index_t> input_left_pads,
                             std::vector<ck::index_t> input_right_pads,
                             InElementwiseOperation in_element_op,
                             WeiElementwiseOperation wei_element_op,
                             OutElementwiseOperation out_element_op)
    {
        return Argument{p_in,
                        p_wei,
                        p_out,
                        N,
                        K,
                        C,
                        input_spatial_lengths,
                        filter_spatial_lengths,
                        output_spatial_lengths,
                        conv_filter_strides,
                        conv_filter_dilations,
                        input_left_pads,
                        input_right_pads,
                        1,
                        1,
                        in_element_op,
                        wei_element_op,
                        out_element_op};
    }

    static auto MakeInvoker() { return Invoker{}; }

    // polymorphic
    std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const void* p_in,
                        const void* p_wei,
                        void* p_out,
                        const index_t N,
                        const index_t K,
                        const index_t C,
                        std::vector<ck::index_t> input_spatial_lengths,
                        std::vector<ck::index_t> filter_spatial_lengths,
                        std::vector<ck::index_t> output_spatial_lengths,
                        std::vector<ck::index_t> conv_filter_strides,
                        std::vector<ck::index_t> conv_filter_dilations,
                        std::vector<ck::index_t> input_left_pads,
                        std::vector<ck::index_t> input_right_pads,
                        InElementwiseOperation in_element_op,
                        WeiElementwiseOperation wei_element_op,
                        OutElementwiseOperation out_element_op) override

    {
        return std::make_unique<Argument>(static_cast<const InDataType*>(p_in),
                                          static_cast<const WeiDataType*>(p_wei),
                                          static_cast<OutDataType*>(p_out),
                                          N,
                                          K,
                                          C,
                                          input_spatial_lengths,
                                          filter_spatial_lengths,
                                          output_spatial_lengths,
                                          conv_filter_strides,
                                          conv_filter_dilations,
                                          input_left_pads,
                                          input_right_pads,
                                          1,
                                          1,
                                          in_element_op,
                                          wei_element_op,
                                          out_element_op);
    }

    // polymorphic
    std::unique_ptr<BaseInvoker> MakeInvokerPointer() override
    {
        return std::make_unique<Invoker>(Invoker{});
    }

    std::string GetTypeString() const override
    {
        auto str = std::stringstream();

        // clang-format off
        str << "DeviceConv3dFwdXdl_Input_N_Di_Hi_Wi_C_Weight_K_Z_Y_X_C_Output_N_Do_Ho_Wo_K"
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
#endif
