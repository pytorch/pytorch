// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_conv_fwd_bias_activation_add.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/literals.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_conv_fwd_bias_activation_add.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

using DeviceConvFwdBiasReluAddPtr =
    DeviceConvFwdBiasActivationAddPtr<ck::tensor_operation::element_wise::PassThrough,
                                      ck::tensor_operation::element_wise::PassThrough,
                                      ck::tensor_operation::element_wise::AddReluAdd>;

void add_device_conv2d_fwd_xdl_c_shuffle_bias_relu_add_nhwc_kyxc_nhwk_f16_instances(
    std::vector<DeviceConvFwdBiasReluAddPtr>&);

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck

namespace ck {
namespace profiler {

template <int NDimSpatial,
          typename InDataType,
          typename WeiDataType,
          typename OutDataType,
          typename InLayout,
          typename WeiLayout,
          typename OutLayout>
void profile_conv_fwd_bias_relu_add_impl(int do_verification,
                                         int init_method,
                                         bool do_log,
                                         bool time_kernel,
                                         ck::index_t N,
                                         ck::index_t K,
                                         ck::index_t C,
                                         std::vector<ck::index_t> input_spatial_lengths,
                                         std::vector<ck::index_t> filter_spatial_lengths,
                                         std::vector<ck::index_t> output_spatial_lengths,
                                         std::vector<ck::index_t> conv_filter_strides,
                                         std::vector<ck::index_t> conv_filter_dilations,
                                         std::vector<ck::index_t> input_left_pads,
                                         std::vector<ck::index_t> input_right_pads)
{
    const ck::index_t Y = filter_spatial_lengths[0];
    const ck::index_t X = filter_spatial_lengths[1];

    const ck::index_t Hi = input_spatial_lengths[0];
    const ck::index_t Wi = input_spatial_lengths[1];

    const ck::index_t Ho = output_spatial_lengths[0];
    const ck::index_t Wo = output_spatial_lengths[1];

    auto f_host_tensor_descriptor =
        [](std::size_t N_, std::size_t C_, std::size_t H, std::size_t W, auto layout) {
            using namespace ck::literals;

            if constexpr(is_same<decltype(layout), ck::tensor_layout::convolution::NCHW>::value ||
                         is_same<decltype(layout), ck::tensor_layout::convolution::KCYX>::value ||
                         is_same<decltype(layout), ck::tensor_layout::convolution::NKHW>::value)
            {
                return HostTensorDescriptor({N_, C_, H, W}, {C_ * H * W, H * W, W, 1_uz});
            }
            else if constexpr(is_same<decltype(layout), tensor_layout::convolution::NHWC>::value ||
                              is_same<decltype(layout), tensor_layout::convolution::KYXC>::value ||
                              is_same<decltype(layout), tensor_layout::convolution::NHWK>::value)
            {
                return HostTensorDescriptor({N_, C_, H, W}, {C_ * H * W, 1_uz, W * C_, C_});
            }
        };

    Tensor<InDataType> in_n_c_hi_wi(f_host_tensor_descriptor(N, C, Hi, Wi, InLayout{}));
    Tensor<WeiDataType> wei_k_c_y_x(f_host_tensor_descriptor(K, C, Y, X, WeiLayout{}));
    Tensor<OutDataType> out_n_k_ho_wo_host_result(
        f_host_tensor_descriptor(N, K, Ho, Wo, OutLayout{}));
    Tensor<OutDataType> out_n_k_ho_wo_device_result(
        f_host_tensor_descriptor(N, K, Ho, Wo, OutLayout{}));

    // bias: assume contiguous 1d vector
    Tensor<OutDataType> bias_k({K});

    // residual: assume same layout as output tensor
    Tensor<OutDataType> resi_n_k_ho_wo(f_host_tensor_descriptor(N, K, Ho, Wo, OutLayout{}));

    std::cout << "in_n_c_hi_wi: " << in_n_c_hi_wi.mDesc << std::endl;
    std::cout << "wei_k_c_y_x: " << wei_k_c_y_x.mDesc << std::endl;
    std::cout << "out_n_k_ho_wo: " << out_n_k_ho_wo_host_result.mDesc << std::endl;
    std::cout << "bias_k: " << bias_k.mDesc << std::endl;
    std::cout << "resi_n_k_ho_wo: " << resi_n_k_ho_wo.mDesc << std::endl;

    switch(init_method)
    {
    case 0: break;
    case 1:
        in_n_c_hi_wi.GenerateTensorValue(GeneratorTensor_2<InDataType>{-5, 5});
        wei_k_c_y_x.GenerateTensorValue(GeneratorTensor_2<WeiDataType>{-5, 5});
        bias_k.GenerateTensorValue(GeneratorTensor_2<OutDataType>{-5, 5});
        resi_n_k_ho_wo.GenerateTensorValue(GeneratorTensor_2<OutDataType>{-5, 5});
        break;
    default:
        in_n_c_hi_wi.GenerateTensorValue(GeneratorTensor_3<InDataType>{0.0, 1.0});
        wei_k_c_y_x.GenerateTensorValue(GeneratorTensor_3<WeiDataType>{-0.5, 0.5});
        bias_k.GenerateTensorValue(GeneratorTensor_3<OutDataType>{0.0, 1.0});
        resi_n_k_ho_wo.GenerateTensorValue(GeneratorTensor_3<OutDataType>{0.0, 1.0});
    }

    using InElementOp  = ck::tensor_operation::element_wise::PassThrough;
    using WeiElementOp = ck::tensor_operation::element_wise::PassThrough;
    using OutElementOp = ck::tensor_operation::element_wise::AddReluAdd;

    const auto in_element_op  = InElementOp{};
    const auto wei_element_op = WeiElementOp{};
    const auto out_element_op = OutElementOp{};

    if(do_verification)
    {
        using ReferenceConvFwdInstance =
            ck::tensor_operation::host::ReferenceConvFwd_Bias_Activation_Add<InDataType,
                                                                             WeiDataType,
                                                                             OutDataType,
                                                                             InElementOp,
                                                                             WeiElementOp,
                                                                             OutElementOp>;

        auto ref_conv    = ReferenceConvFwdInstance{};
        auto ref_invoker = ref_conv.MakeInvoker();

        auto ref_argument = ref_conv.MakeArgument(in_n_c_hi_wi,
                                                  wei_k_c_y_x,
                                                  out_n_k_ho_wo_host_result,
                                                  bias_k,
                                                  resi_n_k_ho_wo,
                                                  conv_filter_strides,
                                                  conv_filter_dilations,
                                                  input_left_pads,
                                                  input_right_pads,
                                                  in_element_op,
                                                  wei_element_op,
                                                  out_element_op);

        ref_invoker.Run(ref_argument);
    }

    DeviceMem in_device_buf(sizeof(InDataType) * in_n_c_hi_wi.mDesc.GetElementSpaceSize());
    DeviceMem wei_device_buf(sizeof(WeiDataType) * wei_k_c_y_x.mDesc.GetElementSpaceSize());
    DeviceMem out_device_buf(sizeof(OutDataType) *
                             out_n_k_ho_wo_device_result.mDesc.GetElementSpaceSize());
    DeviceMem bias_device_buf(sizeof(OutDataType) * bias_k.mDesc.GetElementSpaceSize());
    DeviceMem resi_device_buf(sizeof(OutDataType) * resi_n_k_ho_wo.mDesc.GetElementSpaceSize());

    in_device_buf.ToDevice(in_n_c_hi_wi.mData.data());
    wei_device_buf.ToDevice(wei_k_c_y_x.mData.data());
    bias_device_buf.ToDevice(bias_k.mData.data());
    resi_device_buf.ToDevice(resi_n_k_ho_wo.mData.data());

    using DeviceConvFwdBiasReluAddPtr = ck::tensor_operation::device::
        DeviceConvFwdBiasActivationAddPtr<InElementOp, WeiElementOp, OutElementOp>;

    // add device operator instances
    std::vector<DeviceConvFwdBiasReluAddPtr> op_ptrs;

    if constexpr(ck::is_same_v<ck::remove_cv_t<InDataType>, ck::half_t> &&
                 ck::is_same_v<ck::remove_cv_t<WeiDataType>, ck::half_t> &&
                 ck::is_same_v<ck::remove_cv_t<OutDataType>, ck::half_t>)
    {
        ck::tensor_operation::device::instance::
            add_device_conv2d_fwd_xdl_c_shuffle_bias_relu_add_nhwc_kyxc_nhwk_f16_instances(op_ptrs);
    }

    if(op_ptrs.size() <= 0)
    {
        throw std::runtime_error("wrong! no device Conv instance found");
    }

    std::string best_conv_name;
    float best_ave_time   = 0;
    float best_tflops     = 0;
    float best_gb_per_sec = 0;

    // profile device Conv instances
    for(auto& op_ptr : op_ptrs)
    {
        auto argument_ptr = op_ptr->MakeArgumentPointer(
            static_cast<const InDataType*>(in_device_buf.GetDeviceBuffer()),
            static_cast<const WeiDataType*>(wei_device_buf.GetDeviceBuffer()),
            static_cast<OutDataType*>(out_device_buf.GetDeviceBuffer()),
            static_cast<const OutDataType*>(bias_device_buf.GetDeviceBuffer()),
            static_cast<const OutDataType*>(resi_device_buf.GetDeviceBuffer()),
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
            in_element_op,
            wei_element_op,
            out_element_op);

        auto invoker_ptr = op_ptr->MakeInvokerPointer();

        if(op_ptr->IsSupportedArgument(argument_ptr.get()))
        {
            std::string conv_name = op_ptr->GetTypeString();

            float ave_time =
                invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, time_kernel});

            std::size_t flop = std::size_t(2) * N * K * Ho * Wo * C * Y * X;

            std::size_t num_btype =
                sizeof(InDataType) * (N * C * Hi * Wi) + sizeof(WeiDataType) * (K * C * Y * X) +
                sizeof(OutDataType) * (N * K * Ho * Wo) + sizeof(OutDataType) * (K) +
                sizeof(OutDataType) * (N * K * Ho * Wo);

            float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

            float gb_per_sec = num_btype / 1.E6 / ave_time;

            std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec
                      << " GB/s, " << conv_name << std::endl;

            if(tflops > best_tflops)
            {
                best_conv_name  = conv_name;
                best_tflops     = tflops;
                best_ave_time   = ave_time;
                best_gb_per_sec = gb_per_sec;
            }

            if(do_verification)
            {
                out_device_buf.FromDevice(out_n_k_ho_wo_device_result.mData.data());

                ck::utils::check_err(out_n_k_ho_wo_device_result, out_n_k_ho_wo_host_result);

                if(do_log)
                {
                    LogRangeAsType<float>(std::cout << "in : ", in_n_c_hi_wi.mData, ",")
                        << std::endl;
                    LogRangeAsType<float>(std::cout << "wei: ", wei_k_c_y_x.mData, ",")
                        << std::endl;
                    LogRangeAsType<float>(
                        std::cout << "out_host  : ", out_n_k_ho_wo_host_result.mData, ",")
                        << std::endl;
                    LogRangeAsType<float>(
                        std::cout << "out_device: ", out_n_k_ho_wo_device_result.mData, ",")
                        << std::endl;
                }
            }
        }
    }

    std::cout << "Best Perf: " << best_ave_time << " ms, " << best_tflops << " TFlops, "
              << best_gb_per_sec << " GB/s, " << best_conv_name << std::endl;
}

} // namespace profiler
} // namespace ck
