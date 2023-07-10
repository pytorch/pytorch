// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

bool run_conv_bwd_data_bias_relu(const ExecutionConfig& config,
                                 const ck::utils::conv::ConvParam& conv_params,
                                 const HostTensorDescriptor& out_g_n_k_wos_desc,
                                 const HostTensorDescriptor& wei_g_k_c_xs_desc,
                                 const HostTensorDescriptor& bias_g_n_c_wis_desc,
                                 const HostTensorDescriptor& in_g_n_c_wis_desc,
                                 const OutElementOp& out_element_op,
                                 const WeiElementOp& wei_element_op,
                                 const InElementOp& in_element_op)
{
    Tensor<OutDataType> out(out_g_n_k_wos_desc);
    Tensor<WeiDataType> wei(wei_g_k_c_xs_desc);
    Tensor<BiasDataType> bias(bias_g_n_c_wis_desc);
    Tensor<InDataType> in_host(in_g_n_c_wis_desc);
    Tensor<InDataType> in_device(in_g_n_c_wis_desc);

    std::cout << "out: " << out.mDesc << std::endl;
    std::cout << "wei: " << wei.mDesc << std::endl;
    std::cout << "bias: " << bias.mDesc << std::endl;
    std::cout << "in: " << in_host.mDesc << std::endl;

    switch(config.init_method)
    {
    case 0: break;
    case 1:
        out.GenerateTensorValue(GeneratorTensor_2<OutDataType>{-5, 5});
        wei.GenerateTensorValue(GeneratorTensor_2<WeiDataType>{-5, 5});
        bias.GenerateTensorValue(GeneratorTensor_2<BiasDataType>{-5, 5});
        break;
    default:
        out.GenerateTensorValue(GeneratorTensor_3<OutDataType>{0.0, 1.0});
        wei.GenerateTensorValue(GeneratorTensor_3<WeiDataType>{-0.5, 0.5});
        bias.GenerateTensorValue(GeneratorTensor_3<BiasDataType>{0.0, 1.0});
    }

    DeviceMem out_device_buf(sizeof(OutDataType) * out.mDesc.GetElementSpaceSize());
    DeviceMem wei_device_buf(sizeof(WeiDataType) * wei.mDesc.GetElementSpaceSize());
    DeviceMem bias_device_buf(sizeof(BiasDataType) * bias.mDesc.GetElementSpaceSize());
    DeviceMem in_device_buf(sizeof(InDataType) * in_device.mDesc.GetElementSpaceSize());

    out_device_buf.ToDevice(out.mData.data());
    wei_device_buf.ToDevice(wei.mData.data());
    bias_device_buf.ToDevice(bias.mData.data());

    // reset input to zero
    in_device_buf.SetZero();

    std::array<ck::index_t, NDimSpatial + 3> a_g_n_k_wos_lengths{};
    std::array<ck::index_t, NDimSpatial + 3> a_g_n_k_wos_strides{};
    std::array<ck::index_t, NDimSpatial + 3> b_g_k_c_xs_lengths{};
    std::array<ck::index_t, NDimSpatial + 3> b_g_k_c_xs_strides{};
    std::array<ck::index_t, NDimSpatial + 3> d0_g_n_c_wis_lengths{};
    std::array<ck::index_t, NDimSpatial + 3> d0_g_n_c_wis_strides{};
    std::array<ck::index_t, NDimSpatial + 3> e_g_n_c_wis_lengths{};
    std::array<ck::index_t, NDimSpatial + 3> e_g_n_c_wis_strides{};
    std::array<ck::index_t, NDimSpatial> conv_filter_strides{};
    std::array<ck::index_t, NDimSpatial> conv_filter_dilations{};
    std::array<ck::index_t, NDimSpatial> input_left_pads{};
    std::array<ck::index_t, NDimSpatial> input_right_pads{};

    auto copy = [](const auto& x, auto& y) { ck::ranges::copy(x, y.begin()); };

    copy(out_g_n_k_wos_desc.GetLengths(), a_g_n_k_wos_lengths);
    copy(out_g_n_k_wos_desc.GetStrides(), a_g_n_k_wos_strides);
    copy(wei_g_k_c_xs_desc.GetLengths(), b_g_k_c_xs_lengths);
    copy(wei_g_k_c_xs_desc.GetStrides(), b_g_k_c_xs_strides);
    copy(bias_g_n_c_wis_desc.GetLengths(), d0_g_n_c_wis_lengths);
    copy(bias_g_n_c_wis_desc.GetStrides(), d0_g_n_c_wis_strides);
    copy(in_g_n_c_wis_desc.GetLengths(), e_g_n_c_wis_lengths);
    copy(in_g_n_c_wis_desc.GetStrides(), e_g_n_c_wis_strides);
    copy(conv_params.conv_filter_strides_, conv_filter_strides);
    copy(conv_params.conv_filter_dilations_, conv_filter_dilations);
    copy(conv_params.input_left_pads_, input_left_pads);
    copy(conv_params.input_right_pads_, input_right_pads);

    static_assert(std::is_default_constructible_v<DeviceConvInstance>);

    // do conv
    auto conv     = DeviceConvInstance{};
    auto invoker  = conv.MakeInvoker();
    auto argument = conv.MakeArgument(
        out_device_buf.GetDeviceBuffer(),
        wei_device_buf.GetDeviceBuffer(),
        std::array<const void*, 1>{bias_device_buf.GetDeviceBuffer()},
        in_device_buf.GetDeviceBuffer(),
        a_g_n_k_wos_lengths,
        a_g_n_k_wos_strides,
        b_g_k_c_xs_lengths,
        b_g_k_c_xs_strides,
        std::array<std::array<ck::index_t, NDimSpatial + 3>, 1>{d0_g_n_c_wis_lengths},
        std::array<std::array<ck::index_t, NDimSpatial + 3>, 1>{d0_g_n_c_wis_strides},
        e_g_n_c_wis_lengths,
        e_g_n_c_wis_strides,
        conv_filter_strides,
        conv_filter_dilations,
        input_left_pads,
        input_right_pads,
        out_element_op,
        wei_element_op,
        in_element_op);

    if(!conv.IsSupportedArgument(argument))
    {
        std::cerr << "wrong! device_conv with the specified compilation parameters does "
                     "not support this Conv problem"
                  << std::endl;

        return false;
    }

    float ave_time = invoker.Run(argument, StreamConfig{nullptr, config.time_kernel});

    std::size_t flop      = conv_params.GetFlops();
    std::size_t num_btype = conv_params.GetByte<InDataType, WeiDataType, OutDataType>();

    float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

    float gb_per_sec = num_btype / 1.E6 / ave_time;

    std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s"
              << std::endl;

    if(config.do_verification)
    {
        // c doesn't physically exist, any layout is fine
        Tensor<float> c_host(in_g_n_c_wis_desc);

        auto ref_conv = ck::tensor_operation::host::ReferenceConvBwdData<NDimSpatial,
                                                                         float,
                                                                         WeiDataType,
                                                                         OutDataType,
                                                                         PassThrough,
                                                                         WeiElementOp,
                                                                         OutElementOp>();

        auto ref_invoker = ref_conv.MakeInvoker();

        auto ref_argument = ref_conv.MakeArgument(c_host,
                                                  wei,
                                                  out,
                                                  conv_params.conv_filter_strides_,
                                                  conv_params.conv_filter_dilations_,
                                                  conv_params.input_left_pads_,
                                                  conv_params.input_right_pads_,
                                                  PassThrough{},
                                                  wei_element_op,
                                                  out_element_op);

        ref_invoker.Run(ref_argument);

        // TODO: implement elementwise operation for host
        in_host.ForEach(
            [&](auto&, auto idx) { in_element_op(in_host(idx), c_host(idx), bias(idx)); });

        in_device_buf.FromDevice(in_device.mData.data());

        return ck::utils::check_err(in_device, in_host);
    }

    return true;
}

int run_grouped_conv_bwd_data_bias_relu_example(int argc, char* argv[])
{
    namespace ctc = ck::tensor_layout::convolution;

    ExecutionConfig config;
    ck::utils::conv::ConvParam conv_params = DefaultConvParams;

    if(!parse_cmd_args(argc, argv, config, conv_params))
    {
        return EXIT_FAILURE;
    }

    const auto in_element_op  = InElementOp{};
    const auto wei_element_op = WeiElementOp{};
    const auto out_element_op = OutElementOp{};

    if(conv_params.num_dim_spatial_ != NDimSpatial)
    {
        std::cerr << "unsupported # of spatials dimensions" << std::endl;
        return EXIT_FAILURE;
    }

    // output image: GNHWK
    const auto out_g_n_k_wos_desc =
        ck::utils::conv::make_output_host_tensor_descriptor_g_n_k_wos_packed<OutLayout>(
            conv_params);

    // weight: GKYXC
    const auto wei_g_k_c_xs_desc =
        ck::utils::conv::make_weight_host_tensor_descriptor_g_k_c_xs_packed<WeiLayout>(conv_params);

    // input image bias: G_C
    const auto bias_g_n_c_wis_desc = HostTensorDescriptor({conv_params.G_,
                                                           conv_params.N_,
                                                           conv_params.C_,
                                                           conv_params.input_spatial_lengths_[0],
                                                           conv_params.input_spatial_lengths_[1]},
                                                          {
                                                              conv_params.C_, // g
                                                              0,              // n
                                                              1,              // c
                                                              0,              // hi
                                                              0               // wi
                                                          });

    // input image: GNHWC
    const auto in_g_n_c_wis_desc =
        ck::utils::conv::make_input_host_tensor_descriptor_g_n_c_wis_packed<InLayout>(conv_params);

    return !run_conv_bwd_data_bias_relu(config,
                                        conv_params,
                                        out_g_n_k_wos_desc,
                                        wei_g_k_c_xs_desc,
                                        bias_g_n_c_wis_desc,
                                        in_g_n_c_wis_desc,
                                        wei_element_op,
                                        out_element_op,
                                        in_element_op);
}
