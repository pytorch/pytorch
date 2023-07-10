// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <cstdlib>
#include <iostream>
#include <numeric>
#include <type_traits>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/utility/algorithm.hpp"
#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/convolution_parameter.hpp"
#include "ck/library/utility/convolution_host_tensor_descriptor_helper.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_conv_fwd.hpp"

void print_helper_msg()
{
    std::cout << "arg1: verification (0=no, 1=yes)\n"
              << "arg2: initialization (0=no init, 1=integer value, 2=decimal value)\n"
              << "arg3: time kernel (0=no, 1=yes)\n"
              << ck::utils::conv::get_conv_param_parser_helper_msg() << std::endl;
}

template <ck::index_t NDimSpatial,
          typename InDataType,
          typename WeiDataType,
          typename DsDataType,
          typename OutDataType,
          typename InElementOp,
          typename WeiElementOp,
          typename OutElementOp,
          typename DeviceConvNDFwdInstance>
bool run_grouped_conv_fwd_dl(bool do_verification,
                             int init_method,
                             bool time_kernel,
                             const ck::utils::conv::ConvParam& conv_param,
                             const HostTensorDescriptor& in_g_n_c_wis_desc,
                             const HostTensorDescriptor& wei_g_k_c_xs_desc,
                             const HostTensorDescriptor& out_g_n_k_wos_desc,
                             const InElementOp& in_element_op,
                             const WeiElementOp& wei_element_op,
                             const OutElementOp& out_element_op)
{
    using DDataType = ck::remove_cvref_t<ck::tuple_element_t<0, DsDataType>>;
    Tensor<InDataType> in(in_g_n_c_wis_desc);
    Tensor<WeiDataType> wei(wei_g_k_c_xs_desc);
    Tensor<DDataType> bias(out_g_n_k_wos_desc);
    Tensor<OutDataType> out_host(out_g_n_k_wos_desc);
    Tensor<OutDataType> out_device(out_g_n_k_wos_desc);

    std::cout << "in: " << in.mDesc << std::endl;
    std::cout << "wei: " << wei.mDesc << std::endl;
    std::cout << "out: " << out_host.mDesc << std::endl;

    switch(init_method)
    {
    case 0: break;
    case 1:
        in.GenerateTensorValue(GeneratorTensor_2<InDataType>{-2, 3});
        wei.GenerateTensorValue(GeneratorTensor_2<WeiDataType>{-2, 3});
        bias.GenerateTensorValue(GeneratorTensor_2<DDataType>{-2, 3});
        break;
    case 2:
        in.GenerateTensorValue(GeneratorTensor_3<InDataType>{0.0, 1.0});
        wei.GenerateTensorValue(GeneratorTensor_3<WeiDataType>{-0.5, 0.5});
        bias.GenerateTensorValue(GeneratorTensor_3<DDataType>{-0.5, 0.5});
        break;
    default:
        in.GenerateTensorValue(GeneratorTensor_1<InDataType>{1});
        wei.GenerateTensorValue(GeneratorTensor_1<WeiDataType>{-1});
        bias.GenerateTensorValue(GeneratorTensor_1<DDataType>{1});
    }

    DeviceMem in_device_buf(sizeof(InDataType) * in.mDesc.GetElementSpaceSize());
    DeviceMem wei_device_buf(sizeof(WeiDataType) * wei.mDesc.GetElementSpaceSize());
    DeviceMem bias_device_buf(sizeof(DDataType) * bias.mDesc.GetElementSpaceSize());
    DeviceMem out_device_buf(sizeof(OutDataType) * out_device.mDesc.GetElementSpaceSize());

    in_device_buf.ToDevice(in.mData.data());
    wei_device_buf.ToDevice(wei.mData.data());
    bias_device_buf.ToDevice(bias.mData.data());

    std::array<ck::index_t, NDimSpatial + 3> a_g_n_c_wis_lengths{};
    std::array<ck::index_t, NDimSpatial + 3> a_g_n_c_wis_strides{};
    std::array<ck::index_t, NDimSpatial + 3> b_g_k_c_xs_lengths{};
    std::array<ck::index_t, NDimSpatial + 3> b_g_k_c_xs_strides{};
    std::array<ck::index_t, NDimSpatial + 3> d_g_n_k_wos_lengths{};
    std::array<ck::index_t, NDimSpatial + 3> d_g_n_k_wos_strides{};
    std::array<ck::index_t, NDimSpatial + 3> e_g_n_k_wos_lengths{};
    std::array<ck::index_t, NDimSpatial + 3> e_g_n_k_wos_strides{};
    std::array<ck::index_t, NDimSpatial> conv_filter_strides{};
    std::array<ck::index_t, NDimSpatial> conv_filter_dilations{};
    std::array<ck::index_t, NDimSpatial> input_left_pads{};
    std::array<ck::index_t, NDimSpatial> input_right_pads{};

    auto copy = [](auto& x, auto& y) { ck::ranges::copy(x, y.begin()); };

    copy(in_g_n_c_wis_desc.GetLengths(), a_g_n_c_wis_lengths);
    copy(in_g_n_c_wis_desc.GetStrides(), a_g_n_c_wis_strides);
    copy(wei_g_k_c_xs_desc.GetLengths(), b_g_k_c_xs_lengths);
    copy(wei_g_k_c_xs_desc.GetStrides(), b_g_k_c_xs_strides);
    copy(out_g_n_k_wos_desc.GetLengths(), d_g_n_k_wos_lengths);
    copy(out_g_n_k_wos_desc.GetStrides(), d_g_n_k_wos_strides);
    copy(out_g_n_k_wos_desc.GetLengths(), e_g_n_k_wos_lengths);
    copy(out_g_n_k_wos_desc.GetStrides(), e_g_n_k_wos_strides);
    copy(conv_param.conv_filter_strides_, conv_filter_strides);
    copy(conv_param.conv_filter_dilations_, conv_filter_dilations);
    copy(conv_param.input_left_pads_, input_left_pads);
    copy(conv_param.input_right_pads_, input_right_pads);

    // do Conv
    auto conv     = DeviceConvNDFwdInstance{};
    auto invoker  = conv.MakeInvoker();
    auto argument = conv.MakeArgument(
        in_device_buf.GetDeviceBuffer(),
        wei_device_buf.GetDeviceBuffer(),
        std::array<const void*, 1>{bias_device_buf.GetDeviceBuffer()},
        out_device_buf.GetDeviceBuffer(),
        a_g_n_c_wis_lengths,
        a_g_n_c_wis_strides,
        b_g_k_c_xs_lengths,
        b_g_k_c_xs_strides,
        std::array<std::array<ck::index_t, NDimSpatial + 3>, 1>{{d_g_n_k_wos_lengths}},
        std::array<std::array<ck::index_t, NDimSpatial + 3>, 1>{{d_g_n_k_wos_strides}},
        e_g_n_k_wos_lengths,
        e_g_n_k_wos_strides,
        conv_filter_strides,
        conv_filter_dilations,
        input_left_pads,
        input_right_pads,
        in_element_op,
        wei_element_op,
        out_element_op);

    if(!conv.IsSupportedArgument(argument))
    {
        std::cout << "wrong! device_conv with the specified compilation parameters does not "
                     "support this Conv problem"
                  << std::endl;
        return true;
    }

    float avg_time = invoker.Run(argument, StreamConfig{nullptr, time_kernel});

    std::size_t flop      = conv_param.GetFlops();
    std::size_t num_btype = conv_param.GetByte<InDataType, WeiDataType, OutDataType>();

    float tflops     = static_cast<float>(flop) / 1.E9 / avg_time;
    float gb_per_sec = num_btype / 1.E6 / avg_time;
    std::cout << "Perf: " << avg_time << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s, "
              << conv.GetTypeString() << std::endl;

    if(do_verification)
    {
        auto ref_conv = ck::tensor_operation::host::ReferenceConvFwd<
            NDimSpatial,
            InDataType,
            WeiDataType,
            OutDataType,
            InElementOp,
            WeiElementOp,
            ck::tensor_operation::element_wise::PassThrough>();

        auto ref_invoker = ref_conv.MakeInvoker();
        auto ref_argument =
            ref_conv.MakeArgument(in,
                                  wei,
                                  out_host,
                                  conv_param.conv_filter_strides_,
                                  conv_param.conv_filter_dilations_,
                                  conv_param.input_left_pads_,
                                  conv_param.input_right_pads_,
                                  in_element_op,
                                  wei_element_op,
                                  ck::tensor_operation::element_wise::PassThrough{});

        ref_invoker.Run(ref_argument);

        // cde_elementwise
        out_host.ForEach(
            [&](auto&, auto idx) { out_element_op(out_host(idx), out_host(idx), bias(idx)); });

        out_device_buf.FromDevice(out_device.mData.data());

        return ck::utils::check_err(
            out_device.mData, out_host.mData, "Error: incorrect results!", 1e-5f, 1e-4f);
    }

    return true;
}
