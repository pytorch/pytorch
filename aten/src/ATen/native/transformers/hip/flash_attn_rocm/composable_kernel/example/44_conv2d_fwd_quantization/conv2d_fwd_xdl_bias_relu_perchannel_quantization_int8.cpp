// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_fwd_multiple_d_xdl_cshuffle.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/utility/algorithm.hpp"
#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/literals.hpp"
#include "ck/library/utility/convolution_parameter.hpp"
#include "ck/library/utility/convolution_host_tensor_descriptor_helper.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_conv_fwd.hpp"

using InDataType           = int8_t;
using WeiDataType          = int8_t;
using BiasDataType         = int32_t;
using RequantScaleDataType = float;
using AccDataType          = int32_t;
using CShuffleDataType     = int32_t;
using OutDataType          = int8_t;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using PassThrough  = ck::tensor_operation::element_wise::PassThrough;
using InElementOp  = PassThrough;
using WeiElementOp = PassThrough;
using ActivationOp = ck::tensor_operation::element_wise::Relu;
using OutElementOp = ck::tensor_operation::element_wise::Add_Activation_Mul2_Clamp<ActivationOp>;

static constexpr auto ConvSpec =
    ck::tensor_operation::device::ConvolutionForwardSpecialization::Default;

static constexpr auto GemmSpec = ck::tensor_operation::device::GemmSpecialization::MNKPadding;

template <ck::index_t NDimSpatial,
          typename InLayout,
          typename WeiLayout,
          typename BiasLayout,
          typename RequantScaleLayout,
          typename OutLayout>
using DeviceGroupedConvNDFwdInstance =
    ck::tensor_operation::device::DeviceGroupedConvFwdMultipleD_Xdl_CShuffle<
        NDimSpatial,
        InLayout,
        WeiLayout,
        ck::Tuple<BiasLayout, RequantScaleLayout>,
        OutLayout,
        InDataType,
        WeiDataType,
        AccDataType,
        CShuffleDataType,
        ck::Tuple<BiasDataType, RequantScaleDataType>,
        OutDataType,
        InElementOp,
        WeiElementOp,
        OutElementOp,
        ConvSpec,    // ConvForwardSpecialization
        GemmSpec,    // GemmSpecialization
        1,           //
        256,         // BlockSize
        128,         // MPerBlock
        256,         // NPerBlock
        64,          // KPerBlock
        16,          // AK1
        16,          // BK1
        32,          // MPerXdl
        32,          // NPerXdl
        2,           // MXdlPerWave
        4,           // NXdlPerWave
        S<4, 64, 1>, // ABlockTransferThreadClusterLengths_AK0_M_AK1
        S<1, 0, 2>,  // ABlockTransferThreadClusterArrangeOrder
        S<1, 0, 2>,  // ABlockTransferSrcAccessOrder
        2,           // ABlockTransferSrcVectorDim
        16,          // ABlockTransferSrcScalarPerVector
        16,          // ABlockTransferDstScalarPerVector_AK1
        1,           // ABlockLdsExtraM
        S<4, 64, 1>, // BBlockTransferThreadClusterLengths_BK0_N_BK1
        S<1, 0, 2>,  // BBlockTransferThreadClusterArrangeOrder
        S<1, 0, 2>,  // BBlockTransferSrcAccessOrder
        2,           // BBlockTransferSrcVectorDim
        16,          // BBlockTransferSrcScalarPerVector
        16,          // BBlockTransferDstScalarPerVector_BK1
        1,           // BBlockLdsExtraN
        1,
        1,
        S<1, 64, 1, 4>,
        8>;

template <ck::index_t NDimSpatial,
          typename InDataType,
          typename WeiDataType,
          typename OutDataType,
          typename InElementOp,
          typename WeiElementOp,
          typename OutElementOp,
          typename DeviceConvNDFwdInstance>
bool run_grouped_conv_fwd(bool do_verification,
                          bool time_kernel,
                          const ck::utils::conv::ConvParam& conv_param,
                          const HostTensorDescriptor& in_g_n_c_wis_desc,
                          const HostTensorDescriptor& wei_g_k_c_xs_desc,
                          const HostTensorDescriptor& bias_g_k_desc,
                          const HostTensorDescriptor& requant_scale_g_k_desc,
                          const HostTensorDescriptor& out_g_n_k_wos_desc,
                          const InElementOp& in_element_op,
                          const WeiElementOp& wei_element_op,
                          const OutElementOp& out_element_op)
{
    Tensor<InDataType> in(in_g_n_c_wis_desc);
    Tensor<WeiDataType> wei(wei_g_k_c_xs_desc);
    Tensor<BiasDataType> bias(bias_g_k_desc);
    Tensor<RequantScaleDataType> requant_scale(requant_scale_g_k_desc);
    Tensor<OutDataType> out_host(out_g_n_k_wos_desc);
    Tensor<OutDataType> out_device(out_g_n_k_wos_desc);

    std::cout << "in: " << in.mDesc << std::endl;
    std::cout << "wei: " << wei.mDesc << std::endl;
    std::cout << "bias: " << bias.mDesc << std::endl;
    std::cout << "requant_scale: " << requant_scale.mDesc << std::endl;
    std::cout << "out: " << out_host.mDesc << std::endl;

    in.GenerateTensorValue(GeneratorTensor_2<InDataType>{-128, 127});
    wei.GenerateTensorValue(GeneratorTensor_2<WeiDataType>{-128, 127});
    bias.GenerateTensorValue(GeneratorTensor_2<BiasDataType>{-128, 127});
    requant_scale.GenerateTensorValue(GeneratorTensor_2<RequantScaleDataType>{0, 1});

    DeviceMem in_device_buf(sizeof(InDataType) * in.mDesc.GetElementSpaceSize());
    DeviceMem wei_device_buf(sizeof(WeiDataType) * wei.mDesc.GetElementSpaceSize());
    DeviceMem bias_device_buf(sizeof(BiasDataType) * bias.mDesc.GetElementSpaceSize());
    DeviceMem requant_scale_device_buf(sizeof(RequantScaleDataType) *
                                       requant_scale.mDesc.GetElementSpaceSize());
    DeviceMem out_device_buf(sizeof(OutDataType) * out_device.mDesc.GetElementSpaceSize());

    in_device_buf.ToDevice(in.mData.data());
    wei_device_buf.ToDevice(wei.mData.data());
    bias_device_buf.ToDevice(bias.mData.data());
    requant_scale_device_buf.ToDevice(requant_scale.mData.data());

    std::array<ck::index_t, NDimSpatial + 3> a_g_n_c_wis_lengths{};
    std::array<ck::index_t, NDimSpatial + 3> a_g_n_c_wis_strides{};
    std::array<ck::index_t, NDimSpatial + 3> b_g_k_c_xs_lengths{};
    std::array<ck::index_t, NDimSpatial + 3> b_g_k_c_xs_strides{};
    std::array<ck::index_t, NDimSpatial + 3> d0_g_n_k_wos_lengths{};
    std::array<ck::index_t, NDimSpatial + 3> d0_g_n_k_wos_strides{};
    std::array<ck::index_t, NDimSpatial + 3> d1_g_n_k_wos_lengths{};
    std::array<ck::index_t, NDimSpatial + 3> d1_g_n_k_wos_strides{};
    std::array<ck::index_t, NDimSpatial + 3> e_g_n_k_wos_lengths{};
    std::array<ck::index_t, NDimSpatial + 3> e_g_n_k_wos_strides{};
    std::array<ck::index_t, NDimSpatial> conv_filter_strides{};
    std::array<ck::index_t, NDimSpatial> conv_filter_dilations{};
    std::array<ck::index_t, NDimSpatial> input_left_pads{};
    std::array<ck::index_t, NDimSpatial> input_right_pads{};

    auto copy = [](const auto& x, auto& y) { ck::ranges::copy(x, y.begin()); };

    copy(in_g_n_c_wis_desc.GetLengths(), a_g_n_c_wis_lengths);
    copy(in_g_n_c_wis_desc.GetStrides(), a_g_n_c_wis_strides);
    copy(wei_g_k_c_xs_desc.GetLengths(), b_g_k_c_xs_lengths);
    copy(wei_g_k_c_xs_desc.GetStrides(), b_g_k_c_xs_strides);
    copy(bias_g_k_desc.GetLengths(), d0_g_n_k_wos_lengths);
    copy(bias_g_k_desc.GetStrides(), d0_g_n_k_wos_strides);
    copy(requant_scale_g_k_desc.GetLengths(), d1_g_n_k_wos_lengths);
    copy(requant_scale_g_k_desc.GetStrides(), d1_g_n_k_wos_strides);
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
        {bias_device_buf.GetDeviceBuffer(), requant_scale_device_buf.GetDeviceBuffer()},
        out_device_buf.GetDeviceBuffer(),
        a_g_n_c_wis_lengths,
        a_g_n_c_wis_strides,
        b_g_k_c_xs_lengths,
        b_g_k_c_xs_strides,
        {d0_g_n_k_wos_lengths, d1_g_n_k_wos_lengths},
        {d0_g_n_k_wos_strides, d1_g_n_k_wos_strides},
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
        throw std::runtime_error(
            "wrong! device_conv with the specified compilation parameters does "
            "not support this Conv problem");
    }

    float avg_time = invoker.Run(argument, StreamConfig{nullptr, time_kernel});

    std::size_t flop      = conv_param.GetFlops();
    std::size_t num_btype = conv_param.GetByte<InDataType, WeiDataType, OutDataType>();

    float tflops     = static_cast<float>(flop) / 1.E9 / avg_time;
    float gb_per_sec = num_btype / 1.E6 / avg_time;
    std::cout << "Perf: " << avg_time << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s, "
              << conv.GetTypeString() << std::endl;

    bool pass = true;

    if(do_verification)
    {
        Tensor<CShuffleDataType> c_host(out_g_n_k_wos_desc);

        auto ref_conv = ck::tensor_operation::host::ReferenceConvFwd<NDimSpatial,
                                                                     InDataType,
                                                                     WeiDataType,
                                                                     CShuffleDataType,
                                                                     InElementOp,
                                                                     WeiElementOp,
                                                                     PassThrough>();

        auto ref_invoker  = ref_conv.MakeInvoker();
        auto ref_argument = ref_conv.MakeArgument(in,
                                                  wei,
                                                  c_host,
                                                  conv_param.conv_filter_strides_,
                                                  conv_param.conv_filter_dilations_,
                                                  conv_param.input_left_pads_,
                                                  conv_param.input_right_pads_,
                                                  in_element_op,
                                                  wei_element_op,
                                                  PassThrough{});

        ref_invoker.Run(ref_argument);

        // TODO: implement elementwise operation for host
        out_host.ForEach([&](auto&, auto idx) {
            out_element_op(out_host(idx), c_host(idx), bias(idx), requant_scale(idx));
        });

        out_device_buf.FromDevice(out_device.mData.data());

        pass &=
            ck::utils::check_err(out_device, out_host, "Error: incorrect results!", 1e-5f, 1e-4f);
    }

    return (pass ? 0 : 1);
}

int main()
{
    bool do_verification           = true;
    bool time_kernel               = true;
    const ck::index_t ndim_spatial = 2;

    ck::utils::conv::ConvParam conv_param{
        ndim_spatial, // n_dim
        1,            // group
        4,            // batch
        64,           // output channels
        32,           // input chanels
        {3, 3},       // weight HW
        {71, 71},     // x HW
        {2, 2},       // strides
        {1, 1},       // dilations
        {1, 1},       // left_pads
        {1, 1}        // right_pads
    };

    const auto in_element_op  = InElementOp{};
    const auto wei_element_op = WeiElementOp{};
    const auto out_element_op = OutElementOp{ActivationOp{}};

    using InLayout           = ck::tensor_layout::convolution::GNHWC;
    using WeiLayout          = ck::tensor_layout::convolution::GKYXC;
    using BiasLayout         = ck::tensor_layout::convolution::G_K;
    using RequantScaleLayout = ck::tensor_layout::convolution::G_K;
    using OutLayout          = ck::tensor_layout::convolution::GNHWK;

    const auto in_g_n_c_wis_desc =
        ck::utils::conv::make_input_host_tensor_descriptor_g_n_c_wis_packed<InLayout>(conv_param);

    const auto wei_g_k_c_xs_desc =
        ck::utils::conv::make_weight_host_tensor_descriptor_g_k_c_xs_packed<WeiLayout>(conv_param);

    // TODO - make_bias_host_tensor_descriptor_g_n_k_wos_packed()
    const auto bias_g_k_desc = HostTensorDescriptor({conv_param.G_,
                                                     conv_param.N_,
                                                     conv_param.K_,
                                                     conv_param.output_spatial_lengths_[0],
                                                     conv_param.output_spatial_lengths_[1]},
                                                    {
                                                        conv_param.K_, // g
                                                        0,             // n
                                                        1,             // k
                                                        0,             // ho
                                                        0              // wo
                                                    });

    const auto requant_scale_g_k_desc = bias_g_k_desc;

    const auto out_g_n_k_wos_desc =
        ck::utils::conv::make_output_host_tensor_descriptor_g_n_k_wos_packed<OutLayout>(conv_param);

    std::cout << out_g_n_k_wos_desc << std::endl;

    using deviceOp = DeviceGroupedConvNDFwdInstance<ndim_spatial,
                                                    InLayout,
                                                    WeiLayout,
                                                    BiasLayout,
                                                    RequantScaleLayout,
                                                    OutLayout>;

    return run_grouped_conv_fwd<ndim_spatial,
                                InDataType,
                                WeiDataType,
                                OutDataType,
                                InElementOp,
                                WeiElementOp,
                                OutElementOp,
                                deviceOp>(do_verification,
                                          time_kernel,
                                          conv_param,
                                          in_g_n_c_wis_desc,
                                          wei_g_k_c_xs_desc,
                                          bias_g_k_desc,
                                          requant_scale_g_k_desc,
                                          out_g_n_k_wos_desc,
                                          in_element_op,
                                          wei_element_op,
                                          out_element_op);
}
