// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include "convnd_bwd_data_common.hpp"

#include "ck/tensor_operation/gpu/device/impl/device_convnd_bwd_data_nwc_kxc_nwk_xdl.hpp"

using InDataType  = ck::half_t;
using WeiDataType = ck::half_t;
using OutDataType = ck::half_t;
using AccDataType = float;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using InElementOp  = ck::tensor_operation::element_wise::PassThrough;
using WeiElementOp = ck::tensor_operation::element_wise::PassThrough;
using OutElementOp = ck::tensor_operation::element_wise::PassThrough;

static constexpr auto ConvBwdDefault =
    ck::tensor_operation::device::ConvolutionBackwardDataSpecialization::Default;

template <ck::index_t NDimSpatial>
using DeviceConvNdBwdDataInstance = ck::tensor_operation::device::DeviceConvNdBwdDataNwcKxcNwk_Xdl<
    NDimSpatial,    // NDimSpatial
    InDataType,     // InDataType
    WeiDataType,    // WeiDataType
    OutDataType,    // OutDataType
    AccDataType,    // AccDataType
    InElementOp,    // InElementwiseOperation
    WeiElementOp,   // WeiElementwiseOperation
    OutElementOp,   // OutElementwiseOperation
    ConvBwdDefault, // ConvolutionBackwardDataSpecialization
    256,            // BlockSize
    128,            // MPerBlock
    128,            // NPerBlock
    4,              // K0PerBlock
    8,              // K1
    32,             // MPerXdl
    32,             // NPerXdl
    2,              // MXdlPerWave
    2,              // NXdlPerWave
    S<4, 64, 1>,    // ABlockTransferThreadClusterLengths_K0_M_K1
    S<1, 0, 2>,     // ABlockTransferThreadClusterArrangeOrder
    S<1, 0, 2>,     // ABlockTransferSrcAccessOrder
    2,              // ABlockTransferSrcVectorDim
    8,              // ABlockTransferSrcScalarPerVector
    8,              // ABlockTransferDstScalarPerVector_K1
    true,           // ABlockLdsAddExtraM
    S<4, 64, 1>,    // BBlockTransferThreadClusterLengths_K0_N_K1
    S<2, 0, 1>,     // BBlockTransferThreadClusterArrangeOrder
    S<0, 2, 1>,     // BBlockTransferSrcAccessOrder
    1,              // BBlockTransferSrcVectorDim
    2,              // BBlockTransferSrcScalarPerVector
    8,              // BBlockTransferDstScalarPerVector_K1
    true,           // BBlockLdsAddExtraN
    7,
    1>; // GemmCThreadTransferDstScalarPerVector

int main(int argc, char* argv[])
{
    namespace ctc = ck::tensor_layout::convolution;

    print_helper_msg();

    bool do_verification = true;
    int init_method      = 1;
    bool time_kernel     = false;

    ck::utils::conv::ConvParam conv_param{
        2, 1, 128, 256, 256, {3, 3}, {71, 71}, {2, 2}, {1, 1}, {1, 1}, {1, 1}};

    if(argc == 1)
    {
        // use default
    }
    else if(argc == 4)
    {
        do_verification = std::stoi(argv[1]);
        init_method     = std::stoi(argv[2]);
        time_kernel     = std::stoi(argv[3]);
    }
    else
    {
        do_verification                   = std::stoi(argv[1]);
        init_method                       = std::stoi(argv[2]);
        time_kernel                       = std::stoi(argv[3]);
        const ck::index_t num_dim_spatial = std::stoi(argv[4]);

        conv_param = ck::utils::conv::parse_conv_param(num_dim_spatial, 5, argv);
    }

    const auto in_element_op  = InElementOp{};
    const auto wei_element_op = WeiElementOp{};
    const auto out_element_op = OutElementOp{};

    if(conv_param.num_dim_spatial_ == 1)
    {
        using InLayout  = ctc::GNWC;
        using WeiLayout = ctc::GKXC;
        using OutLayout = ctc::GNWK;

        const auto in_g_n_c_wis_desc =
            ck::utils::conv::make_input_host_tensor_descriptor_g_n_c_wis_packed<InLayout>(
                conv_param);

        const auto wei_g_k_c_xs_desc =
            ck::utils::conv::make_weight_host_tensor_descriptor_g_k_c_xs_packed<WeiLayout>(
                conv_param);

        const auto out_g_n_k_wos_desc =
            ck::utils::conv::make_output_host_tensor_descriptor_g_n_k_wos_packed<OutLayout>(
                conv_param);

        return run_conv_bwd_data<1,
                                 InDataType,
                                 WeiDataType,
                                 OutDataType,
                                 InElementOp,
                                 WeiElementOp,
                                 OutElementOp,
                                 DeviceConvNdBwdDataInstance<1>>(do_verification,
                                                                 init_method,
                                                                 time_kernel,
                                                                 conv_param,
                                                                 in_g_n_c_wis_desc,
                                                                 wei_g_k_c_xs_desc,
                                                                 out_g_n_k_wos_desc,
                                                                 in_element_op,
                                                                 wei_element_op,
                                                                 out_element_op);
    }
    else if(conv_param.num_dim_spatial_ == 2)
    {
        using InLayout  = ctc::GNHWC;
        using WeiLayout = ctc::GKYXC;
        using OutLayout = ctc::GNHWK;

        const auto in_g_n_c_wis_desc =
            ck::utils::conv::make_input_host_tensor_descriptor_g_n_c_wis_packed<InLayout>(
                conv_param);

        const auto wei_g_k_c_xs_desc =
            ck::utils::conv::make_weight_host_tensor_descriptor_g_k_c_xs_packed<WeiLayout>(
                conv_param);

        const auto out_g_n_k_wos_desc =
            ck::utils::conv::make_output_host_tensor_descriptor_g_n_k_wos_packed<OutLayout>(
                conv_param);

        return run_conv_bwd_data<2,
                                 InDataType,
                                 WeiDataType,
                                 OutDataType,
                                 InElementOp,
                                 WeiElementOp,
                                 OutElementOp,
                                 DeviceConvNdBwdDataInstance<2>>(do_verification,
                                                                 init_method,
                                                                 time_kernel,
                                                                 conv_param,
                                                                 in_g_n_c_wis_desc,
                                                                 wei_g_k_c_xs_desc,
                                                                 out_g_n_k_wos_desc,
                                                                 in_element_op,
                                                                 wei_element_op,
                                                                 out_element_op);
    }
    else if(conv_param.num_dim_spatial_ == 3)
    {
        using InLayout  = ctc::GNDHWC;
        using WeiLayout = ctc::GKZYXC;
        using OutLayout = ctc::GNDHWK;

        const auto in_g_n_c_wis_desc =
            ck::utils::conv::make_input_host_tensor_descriptor_g_n_c_wis_packed<InLayout>(
                conv_param);

        const auto wei_g_k_c_xs_desc =
            ck::utils::conv::make_weight_host_tensor_descriptor_g_k_c_xs_packed<WeiLayout>(
                conv_param);

        const auto out_g_n_k_wos_desc =
            ck::utils::conv::make_output_host_tensor_descriptor_g_n_k_wos_packed<OutLayout>(
                conv_param);

        return run_conv_bwd_data<3,
                                 InDataType,
                                 WeiDataType,
                                 OutDataType,
                                 InElementOp,
                                 WeiElementOp,
                                 OutElementOp,
                                 DeviceConvNdBwdDataInstance<3>>(do_verification,
                                                                 init_method,
                                                                 time_kernel,
                                                                 conv_param,
                                                                 in_g_n_c_wis_desc,
                                                                 wei_g_k_c_xs_desc,
                                                                 out_g_n_k_wos_desc,
                                                                 in_element_op,
                                                                 wei_element_op,
                                                                 out_element_op);
    }

    return 0;
}
