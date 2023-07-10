// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

template <ck::index_t NDimSpatial,
          typename In0DataType,
          typename Wei0DataType,
          typename Out0DataType,
          typename Wei1DataType,
          typename Out1DataType,
          typename In0ElementOp,
          typename Wei0ElementOp,
          typename Out0ElementOp,
          typename Wei1ElementOp,
          typename Out1ElementOp,
          typename DeviceOpInstance>
bool run_grouped_conv_conv_fwd(bool do_verification,
                               int init_method,
                               bool time_kernel,
                               const ck::utils::conv::ConvParam& conv0_param,
                               const ck::utils::conv::ConvParam& conv1_param,
                               const HostTensorDescriptor& in0_g_n_c_wis_desc,
                               const HostTensorDescriptor& wei0_g_k_c_xs_desc,
                               const HostTensorDescriptor& out0_g_n_k_wos_desc,
                               const HostTensorDescriptor& wei1_g_k_c_xs_desc,
                               const HostTensorDescriptor& out1_g_n_k_wos_desc,
                               const In0ElementOp& in0_element_op,
                               const Wei0ElementOp& wei0_element_op,
                               const Wei1ElementOp& wei1_element_op,
                               const Out0ElementOp& out0_element_op,
                               const Out1ElementOp& out1_element_op)
{
    Tensor<In0DataType> in0(in0_g_n_c_wis_desc);
    Tensor<Wei0DataType> wei0(wei0_g_k_c_xs_desc);
    Tensor<Wei1DataType> wei1(wei1_g_k_c_xs_desc);
    Tensor<Out1DataType> out1_host(out1_g_n_k_wos_desc);
    Tensor<Out1DataType> out1_device(out1_g_n_k_wos_desc);

    std::cout << "in0: " << in0.mDesc << std::endl;
    std::cout << "wei0: " << wei0.mDesc << std::endl;
    std::cout << "wei1: " << wei1.mDesc << std::endl;
    std::cout << "out1: " << out1_host.mDesc << std::endl;

    switch(init_method)
    {
    case 0: break;
    case 1:
        in0.GenerateTensorValue(GeneratorTensor_2<In0DataType>{-5, 5});
        wei0.GenerateTensorValue(GeneratorTensor_2<Wei0DataType>{-5, 5});
        wei1.GenerateTensorValue(GeneratorTensor_2<Wei1DataType>{-5, 5});
        break;
    default:
        in0.GenerateTensorValue(GeneratorTensor_3<In0DataType>{0.0, 1.0});
        wei0.GenerateTensorValue(GeneratorTensor_3<Wei0DataType>{-0.5, 0.5});
        wei1.GenerateTensorValue(GeneratorTensor_3<Wei1DataType>{-0.5, 0.5});
    }

#ifdef BUILD_INT4_EXAMPLE
    DeviceMem in0_device_buf(sizeof(KernelIn0DataType) * in0.mDesc.GetElementSpaceSize());
    DeviceMem wei0_device_buf(sizeof(KernelWei0DataType) * wei0.mDesc.GetElementSpaceSize());
    DeviceMem wei1_device_buf(sizeof(KernelWei1DataType) * wei1.mDesc.GetElementSpaceSize());
    DeviceMem out1_device_buf(sizeof(KernelOut1DataType) * out1_device.mDesc.GetElementSpaceSize());

    const Tensor<KernelIn0DataType> in0_converted(in0);
    const Tensor<KernelWei0DataType> wei0_converted(wei0);
    const Tensor<KernelWei1DataType> wei1_converted(wei1);

    in0_device_buf.ToDevice(in0_converted.mData.data());
    wei0_device_buf.ToDevice(wei0_converted.mData.data());
    wei1_device_buf.ToDevice(wei1_converted.mData.data());
#else
    DeviceMem in0_device_buf(sizeof(In0DataType) * in0.mDesc.GetElementSpaceSize());
    DeviceMem wei0_device_buf(sizeof(Wei0DataType) * wei0.mDesc.GetElementSpaceSize());
    DeviceMem wei1_device_buf(sizeof(Wei1DataType) * wei1.mDesc.GetElementSpaceSize());
    DeviceMem out1_device_buf(sizeof(Out1DataType) * out1_device.mDesc.GetElementSpaceSize());

    in0_device_buf.ToDevice(in0.mData.data());
    wei0_device_buf.ToDevice(wei0.mData.data());
    wei1_device_buf.ToDevice(wei1.mData.data());
#endif

    std::array<ck::index_t, NDimSpatial + 3> a0_g_n_c_wis_lengths{};
    std::array<ck::index_t, NDimSpatial + 3> a0_g_n_c_wis_strides{};
    std::array<ck::index_t, NDimSpatial + 3> b0_g_k_c_xs_lengths{};
    std::array<ck::index_t, NDimSpatial + 3> b0_g_k_c_xs_strides{};
    std::array<ck::index_t, NDimSpatial + 3> b1_g_k_c_xs_lengths{};
    std::array<ck::index_t, NDimSpatial + 3> b1_g_k_c_xs_strides{};
    std::array<ck::index_t, NDimSpatial + 3> e1_g_n_k_wos_lengths{};
    std::array<ck::index_t, NDimSpatial + 3> e1_g_n_k_wos_strides{};
    std::array<ck::index_t, NDimSpatial> conv0_filter_strides{};
    std::array<ck::index_t, NDimSpatial> conv0_filter_dilations{};
    std::array<ck::index_t, NDimSpatial> input0_left_pads{};
    std::array<ck::index_t, NDimSpatial> input0_right_pads{};
    std::array<ck::index_t, NDimSpatial> conv1_filter_strides{};
    std::array<ck::index_t, NDimSpatial> conv1_filter_dilations{};
    std::array<ck::index_t, NDimSpatial> input1_left_pads{};
    std::array<ck::index_t, NDimSpatial> input1_right_pads{};

    auto copy = [](const auto& x, auto& y) { ck::ranges::copy(x, y.begin()); };

    copy(in0_g_n_c_wis_desc.GetLengths(), a0_g_n_c_wis_lengths);
    copy(in0_g_n_c_wis_desc.GetStrides(), a0_g_n_c_wis_strides);
    copy(wei0_g_k_c_xs_desc.GetLengths(), b0_g_k_c_xs_lengths);
    copy(wei0_g_k_c_xs_desc.GetStrides(), b0_g_k_c_xs_strides);
    copy(wei1_g_k_c_xs_desc.GetLengths(), b1_g_k_c_xs_lengths);
    copy(wei1_g_k_c_xs_desc.GetStrides(), b1_g_k_c_xs_strides);
    copy(out1_g_n_k_wos_desc.GetLengths(), e1_g_n_k_wos_lengths);
    copy(out1_g_n_k_wos_desc.GetStrides(), e1_g_n_k_wos_strides);
    copy(conv0_param.conv_filter_strides_, conv0_filter_strides);
    copy(conv0_param.conv_filter_dilations_, conv0_filter_dilations);
    copy(conv0_param.input_left_pads_, input0_left_pads);
    copy(conv0_param.input_right_pads_, input0_right_pads);
    copy(conv1_param.conv_filter_strides_, conv1_filter_strides);
    copy(conv1_param.conv_filter_dilations_, conv1_filter_dilations);
    copy(conv1_param.input_left_pads_, input1_left_pads);
    copy(conv1_param.input_right_pads_, input1_right_pads);

    // do Conv using GEMM, only works for 1x1 conv for now
    const ck::index_t gemm_batch = a0_g_n_c_wis_lengths[0];

    const ck::index_t gemm0_m_length =
        e1_g_n_k_wos_lengths[1] *
        ck::accumulate_n<ck::index_t>(
            e1_g_n_k_wos_lengths.begin() + 3, NDimSpatial, 1, std::multiplies<>{});

    const ck::index_t gemm0_n_length = b0_g_k_c_xs_lengths[1];

    const ck::index_t gemm0_k_length = ck::accumulate_n<ck::index_t>(
        b0_g_k_c_xs_lengths.begin() + 2, NDimSpatial + 1, 1, std::multiplies<>{});

    const ck::index_t gemm1_n_length = b1_g_k_c_xs_lengths[1];

    //
    const ck::index_t a0_stride = a0_g_n_c_wis_strides[2 + NDimSpatial];
    const ck::index_t b0_stride = b0_g_k_c_xs_strides[2 + NDimSpatial];
    const ck::index_t b1_stride = b1_g_k_c_xs_strides[2 + NDimSpatial];
    const ck::index_t e1_stride = e1_g_n_k_wos_strides[2 + NDimSpatial];

    //
    const ck::index_t a0_batch_stride = a0_g_n_c_wis_strides[0];
    const ck::index_t b0_batch_stride = b0_g_k_c_xs_strides[0];
    const ck::index_t b1_batch_stride = b1_g_k_c_xs_strides[0];
    const ck::index_t e1_batch_stride = e1_g_n_k_wos_strides[0];

    auto device_op = DeviceOpInstance{};
    auto invoker   = device_op.MakeInvoker();
    auto argument  = device_op.MakeArgument(
#ifdef BUILD_INT4_EXAMPLE
        static_cast<KernelIn0DataType*>(in0_device_buf.GetDeviceBuffer()),
        static_cast<KernelWei0DataType*>(wei0_device_buf.GetDeviceBuffer()),
        static_cast<KernelWei1DataType*>(wei1_device_buf.GetDeviceBuffer()),
        static_cast<KernelOut1DataType*>(out1_device_buf.GetDeviceBuffer()),
#else
        static_cast<In0DataType*>(in0_device_buf.GetDeviceBuffer()),
        static_cast<Wei0DataType*>(wei0_device_buf.GetDeviceBuffer()),
        static_cast<Wei1DataType*>(wei1_device_buf.GetDeviceBuffer()),
        static_cast<Out1DataType*>(out1_device_buf.GetDeviceBuffer()),
#endif
        gemm0_m_length,
        gemm0_n_length,
        gemm0_k_length,
        gemm1_n_length,
        gemm_batch,
        a0_stride,
        b0_stride,
        b1_stride,
        e1_stride,
        a0_batch_stride,
        b0_batch_stride,
        b1_batch_stride,
        e1_batch_stride,
        in0_element_op,
        wei0_element_op,
        out0_element_op,
        wei1_element_op,
        out1_element_op);

    if(!device_op.IsSupportedArgument(argument))
    {
        throw std::runtime_error(
            "wrong! device_conv with the specified compilation parameters does "
            "not support this Conv problem");
    }

    float avg_time = invoker.Run(argument, StreamConfig{nullptr, time_kernel});

    std::size_t flop      = conv0_param.GetFlops() + conv1_param.GetFlops();
    std::size_t num_btype = conv0_param.template GetInputByte<In0DataType>() +
                            conv0_param.template GetWeightByte<Wei0DataType>() +
                            conv1_param.template GetWeightByte<Wei1DataType>() +
                            conv1_param.template GetOutputByte<Out1DataType>();

    float tflops     = static_cast<float>(flop) / 1.E9 / avg_time;
    float gb_per_sec = num_btype / 1.E6 / avg_time;
    std::cout << "Perf: " << avg_time << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s, "
              << device_op.GetTypeString() << std::endl;

    if(do_verification)
    {
        using PassThrough = ck::tensor_operation::element_wise::PassThrough;

        Tensor<Out0DataType> out0_host(out0_g_n_k_wos_desc);

        auto ref_conv0 = ck::tensor_operation::host::ReferenceConvFwd<NDimSpatial,
                                                                      In0DataType,
                                                                      Wei0DataType,
                                                                      Out0DataType,
                                                                      In0ElementOp,
                                                                      Wei0ElementOp,
                                                                      Out0ElementOp>();

        auto ref_conv1 = ck::tensor_operation::host::ReferenceConvFwd<NDimSpatial,
                                                                      Out0DataType,
                                                                      Wei1DataType,
                                                                      Out1DataType,
                                                                      PassThrough,
                                                                      Wei1ElementOp,
                                                                      Out1ElementOp>();

        auto ref_conv0_invoker = ref_conv0.MakeInvoker();
        auto ref_conv1_invoker = ref_conv1.MakeInvoker();

        auto ref_conv0_argument = ref_conv0.MakeArgument(in0,
                                                         wei0,
                                                         out0_host,
                                                         conv0_param.conv_filter_strides_,
                                                         conv0_param.conv_filter_dilations_,
                                                         conv0_param.input_left_pads_,
                                                         conv0_param.input_right_pads_,
                                                         in0_element_op,
                                                         wei0_element_op,
                                                         out0_element_op);

        auto ref_conv1_argument = ref_conv1.MakeArgument(out0_host,
                                                         wei1,
                                                         out1_host,
                                                         conv1_param.conv_filter_strides_,
                                                         conv1_param.conv_filter_dilations_,
                                                         conv1_param.input_left_pads_,
                                                         conv1_param.input_right_pads_,
                                                         out0_element_op,
                                                         wei1_element_op,
                                                         out1_element_op);

        ref_conv0_invoker.Run(ref_conv0_argument);
        ref_conv1_invoker.Run(ref_conv1_argument);

#ifdef BUILD_INT4_EXAMPLE
        Tensor<KernelOut1DataType> out1_device_converted(out1_host.mDesc);

        out1_device_buf.FromDevice(out1_device_converted.mData.data());

        out1_device = out1_device_converted.CopyAsType<Out1DataType>();
#else
        out1_device_buf.FromDevice(out1_device.mData.data());
#endif

        return ck::utils::check_err(
            out1_device, out1_host, "Error: incorrect results!", 1e-5f, 1e-4f);
    }

    return true;
}

bool run_grouped_conv_conv_fwd_example(int argc, char* argv[])
{
    bool do_verification = true;
    int init_method      = 1;
    bool time_kernel     = false;

    ck::utils::conv::ConvParam conv0_param{
        2, 1, 128, 512, 128, {1, 1}, {28, 28}, {1, 1}, {1, 1}, {0, 0}, {0, 0}};

    ck::utils::conv::ConvParam conv1_param{
        2, 1, 128, 128, 512, {1, 1}, {28, 28}, {1, 1}, {1, 1}, {0, 0}, {0, 0}};

    if(argc == 1)
    {
        // use default case
    }
    else if(argc == 4)
    {
        do_verification = std::stoi(argv[1]);
        init_method     = std::stoi(argv[2]);
        time_kernel     = std::stoi(argv[3]);
    }
    else
    {
        printf("arg1: verification (0=no, 1=yes)\n");
        printf("arg2: initialization (0=no init, 1=integer value, 2=decimal value)\n");
        printf("arg3: time kernel (0=no, 1=yes)\n");
        exit(0);
    }

    const auto in0_element_op  = In0ElementOp{};
    const auto wei0_element_op = Wei0ElementOp{};
    const auto wei1_element_op = Wei1ElementOp{};
    const auto out0_element_op = Out0ElementOp{};
    const auto out1_element_op = Out1ElementOp{};

    const auto run = [&](auto ndim_spatial,
                         auto in0_layout,
                         auto wei0_layout,
                         auto wei1_layout,
                         auto out1_layout) {
        constexpr ck::index_t ndim_spatial_value = ndim_spatial.value;

        using In0Layout  = decltype(in0_layout);
        using Wei0Layout = decltype(wei0_layout);
        using Wei1Layout = decltype(wei1_layout);
        using Out1Layout = decltype(out1_layout);

        const auto in0_g_n_c_wis_desc =
            ck::utils::conv::make_input_host_tensor_descriptor_g_n_c_wis_packed<In0Layout>(
                conv0_param);

        const auto wei0_g_k_c_xs_desc =
            ck::utils::conv::make_weight_host_tensor_descriptor_g_k_c_xs_packed<Wei0Layout>(
                conv0_param);

        // out0 doesn't physical exist, any layout for host verification is OK
        const auto out0_g_n_k_wos_desc =
            ck::utils::conv::make_output_host_tensor_descriptor_g_n_k_wos_packed<Out1Layout>(
                conv0_param);

        const auto wei1_g_k_c_xs_desc =
            ck::utils::conv::make_weight_host_tensor_descriptor_g_k_c_xs_packed<Wei1Layout>(
                conv1_param);

        const auto out1_g_n_k_wos_desc =
            ck::utils::conv::make_output_host_tensor_descriptor_g_n_k_wos_packed<Out1Layout>(
                conv1_param);

        return run_grouped_conv_conv_fwd<ndim_spatial_value,
                                         In0DataType,
                                         Wei0DataType,
                                         Out0DataType,
                                         Wei1DataType,
                                         Out1DataType,
                                         In0ElementOp,
                                         Wei0ElementOp,
                                         Out0ElementOp,
                                         Wei1ElementOp,
                                         Out1ElementOp,
                                         DeviceBatchedGemmGemmInstance>(do_verification,
                                                                        init_method,
                                                                        time_kernel,
                                                                        conv0_param,
                                                                        conv1_param,
                                                                        in0_g_n_c_wis_desc,
                                                                        wei0_g_k_c_xs_desc,
                                                                        out0_g_n_k_wos_desc,
                                                                        wei1_g_k_c_xs_desc,
                                                                        out1_g_n_k_wos_desc,
                                                                        in0_element_op,
                                                                        wei0_element_op,
                                                                        wei1_element_op,
                                                                        out0_element_op,
                                                                        out1_element_op);
    };

    namespace ctc = ck::tensor_layout::convolution;

    if(conv0_param.num_dim_spatial_ == 1)
    {
        return run(ck::Number<1>{}, ctc::GNWC{}, ctc::GKXC{}, ctc::GKXC{}, ctc::GNWK{});
    }
    else if(conv0_param.num_dim_spatial_ == 2)
    {
        return run(ck::Number<2>{}, ctc::GNHWC{}, ctc::GKYXC{}, ctc::GKYXC{}, ctc::GNHWK{});
    }
    else if(conv0_param.num_dim_spatial_ == 3)
    {
        return run(ck::Number<3>{}, ctc::GNDHWC{}, ctc::GKZYXC{}, ctc::GKZYXC{}, ctc::GNDHWK{});
    }

    return true;
}
