// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

bool run_convnd_fwd_example(int argc, char* argv[])
{
    print_helper_msg();

    bool do_verification = true;
    int init_method      = 1;
    bool time_kernel     = false;

    ck::utils::conv::ConvParam conv_param{
        2, 1, 128, 256, 192, {3, 3}, {71, 71}, {2, 2}, {1, 1}, {1, 1}, {1, 1}};

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

    const auto run = [&](auto ndim_spatial, auto in_layout, auto wei_layout, auto out_layout) {
        constexpr ck::index_t ndim_spatial_value = ndim_spatial.value;

        using InLayout  = decltype(in_layout);
        using WeiLayout = decltype(wei_layout);
        using OutLayout = decltype(out_layout);

        const auto in_g_n_c_wis_desc =
            ck::utils::conv::make_input_host_tensor_descriptor_g_n_c_wis_packed<InLayout>(
                conv_param);

        const auto wei_g_k_c_xs_desc =
            ck::utils::conv::make_weight_host_tensor_descriptor_g_k_c_xs_packed<WeiLayout>(
                conv_param);

        const auto out_g_n_k_wos_desc =
            ck::utils::conv::make_output_host_tensor_descriptor_g_n_k_wos_packed<OutLayout>(
                conv_param);

        return run_grouped_conv_fwd<
            ndim_spatial_value,
            InDataType,
            WeiDataType,
            OutDataType,
            InElementOp,
            WeiElementOp,
            OutElementOp,
            DeviceGroupedConvNDFwdInstance<ndim_spatial_value, InLayout, WeiLayout, OutLayout>>(
            do_verification,
            init_method,
            time_kernel,
            conv_param,
            in_g_n_c_wis_desc,
            wei_g_k_c_xs_desc,
            out_g_n_k_wos_desc,
            in_element_op,
            wei_element_op,
            out_element_op);
    };

    namespace ctc = ck::tensor_layout::convolution;

    if(conv_param.num_dim_spatial_ == 1)
    {
        return run(ck::Number<1>{}, ctc::GNWC{}, ctc::GKXC{}, ctc::GNWK{});
    }
    else if(conv_param.num_dim_spatial_ == 2)
    {
        return run(ck::Number<2>{}, ctc::GNHWC{}, ctc::GKYXC{}, ctc::GNHWK{});
    }
    else if(conv_param.num_dim_spatial_ == 3)
    {
        return run(ck::Number<3>{}, ctc::GNDHWC{}, ctc::GKZYXC{}, ctc::GNDHWK{});
    }

    return true;
}
