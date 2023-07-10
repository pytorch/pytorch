// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>

#include "ck/ck.hpp"
#include "ck/utility/reduction_enums.hpp"
#include "ck/utility/reduction_functions_accumulate.hpp"
#include "ck/tensor_operation/gpu/device/reduction_operator_mapping.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_pool2d_fwd_nhwc_nhwc.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/literals.hpp"

template <typename InDataType,
          typename OutDataType,
          typename AccDataType,
          typename IndexDataType,
          ck::ReduceTensorOp ReduceOpId,
          bool PropagateNan,
          bool OutputIndex>
static void pool_host_verify(const Tensor<InDataType>& in,
                             Tensor<OutDataType>& out,
                             Tensor<IndexDataType>& out_indices,
                             const std::array<ck::index_t, 2>& window_spatial_lengths,
                             const std::array<ck::index_t, 2>& window_strides,
                             const std::array<ck::index_t, 2>& in_left_pads,
                             const std::array<ck::index_t, 2>& /*in_right_pads*/)
{
    const int32_t reduceLength = window_spatial_lengths[0] * window_spatial_lengths[1];

    using ReduceOperation = typename ck::reduce_binary_operator<ReduceOpId>::opType;

    auto elementwise_ops =
        ck::reduce_unary_operator<ReduceOpId, true, true>::GetElementwiseOperator(reduceLength);

    auto in_elementwise_op  = std::get<0>(elementwise_ops);
    auto acc_elementwise_op = std::get<1>(elementwise_ops);

    if constexpr(!OutputIndex)
    {
        using Accumulation =
            ck::detail::AccumulateWithNanCheck<PropagateNan, ReduceOperation, AccDataType>;

        auto f_nchw = [&](auto n, auto c, auto ho, auto wo) {
            auto accuVal = ReduceOperation::template GetIdentityValue<AccDataType>();

            for(ck::index_t y = 0; y < window_spatial_lengths[0]; ++y)
            {
                ck::index_t hi = ho * window_strides[0] + y - in_left_pads[0];
                for(ck::index_t x = 0; x < window_spatial_lengths[1]; ++x)
                {
                    ck::index_t wi = wo * window_strides[1] + x - in_left_pads[1];
                    if(hi >= 0 && hi < static_cast<ck::index_t>(in.mDesc.GetLengths()[2]) &&
                       wi >= 0 && wi < static_cast<ck::index_t>(in.mDesc.GetLengths()[3]))
                    {
                        AccDataType currVal = static_cast<AccDataType>(in(n, c, hi, wi));

                        in_elementwise_op(currVal, currVal);

                        Accumulation::Calculate(accuVal, currVal);
                    }
                }
            }

            acc_elementwise_op(accuVal, accuVal);

            out(n, c, ho, wo) = accuVal;
        };

        make_ParallelTensorFunctor(f_nchw,
                                   out.mDesc.GetLengths()[0],
                                   out.mDesc.GetLengths()[1],
                                   out.mDesc.GetLengths()[2],
                                   out.mDesc.GetLengths()[3])(std::thread::hardware_concurrency());
    }
    else
    {
        using Accumulation = ck::detail::AccumulateWithIndexAndNanCheck<PropagateNan,
                                                                        ReduceOperation,
                                                                        AccDataType,
                                                                        IndexDataType>;
        auto f_nchw        = [&](auto n, auto c, auto ho, auto wo) {
            auto accuVal            = ReduceOperation::template GetIdentityValue<AccDataType>();
            IndexDataType accuIndex = 0;

            for(ck::index_t y = 0; y < window_spatial_lengths[0]; ++y)
            {
                ck::index_t hi = ho * window_strides[0] + y - in_left_pads[0];
                for(ck::index_t x = 0; x < window_spatial_lengths[1]; ++x)
                {
                    ck::index_t wi = wo * window_strides[1] + x - in_left_pads[1];
                    if(hi >= 0 && hi < in.mDesc.GetLengths()[2] && wi >= 0 &&
                       wi < in.mDesc.GetLengths()[3])
                    {
                        AccDataType currVal     = static_cast<AccDataType>(in(n, c, hi, wi));
                        IndexDataType currIndex = y * window_spatial_lengths[1] + x;

                        in_elementwise_op(currVal, currVal);

                        Accumulation::Calculate(accuVal, currVal, accuIndex, currIndex);
                    }
                }
            }

            acc_elementwise_op(accuVal, accuVal);

            out(n, c, ho, wo)         = accuVal;
            out_indices(n, c, ho, wo) = accuIndex;
        };

        make_ParallelTensorFunctor(f_nchw,
                                   out.mDesc.GetLengths()[0],
                                   out.mDesc.GetLengths()[1],
                                   out.mDesc.GetLengths()[2],
                                   out.mDesc.GetLengths()[3])(std::thread::hardware_concurrency());
    };
}

template <typename InDataType,
          typename OutDataType,
          typename AccDataType,
          typename IndexDataType,
          typename InLayout,
          typename OutLayout,
          ck::ReduceTensorOp ReduceOpId,
          bool PropagateNan,
          bool OutputIndex>
bool pool_test(bool do_verification,
               int init_method,
               bool time_kernel,
               ck::index_t N,
               ck::index_t C,
               ck::index_t Y,
               ck::index_t X,
               ck::index_t Hi,
               ck::index_t Wi,
               ck::index_t window_stride_h,
               ck::index_t window_stride_w,
               ck::index_t in_left_pad_h,
               ck::index_t in_left_pad_w,
               ck::index_t in_right_pad_h,
               ck::index_t in_right_pad_w)
{
    using DevicePoolFwdInstance =
        ck::tensor_operation::device::DevicePool2dFwd_Input_N_Hi_Wi_C_Output_N_Ho_Wo_C<
            InDataType,  // InDataType
            OutDataType, // OutDataType
            AccDataType, // AccDataType
            ReduceOpId,
            OutputIndex,
            64, // BlockSize
            64, // ReduceMThreadClusterSize
            1,  // ReduceKThreadClusterSize
            4,  // ReduceMThreadSliceSize
            1,  // ReduceKThreadSliceSize
            4>; // InSrcOutDstVectorSize

    const ck::index_t Ho = (Hi + in_left_pad_h + in_right_pad_h - Y) / window_stride_h + 1;
    const ck::index_t Wo = (Wi + in_left_pad_w + in_right_pad_w - X) / window_stride_w + 1;

    const std::array<ck::index_t, 2> window_spatial_lengths{{Y, X}};
    const std::array<ck::index_t, 2> window_strides{{window_stride_h, window_stride_w}};
    const std::array<ck::index_t, 2> input_left_pads{{in_left_pad_h, in_left_pad_w}};
    const std::array<ck::index_t, 2> input_right_pads{{in_right_pad_h, in_right_pad_w}};

    // tensor layout
    auto f_host_tensor_descriptor =
        [](std::size_t N_, std::size_t C_, std::size_t H, std::size_t W, auto layout) {
            using namespace ck::literals;

            if constexpr(ck::is_same<decltype(layout), ck::tensor_layout::convolution::NCHW>::value)
            {
                return HostTensorDescriptor({N_, C_, H, W}, {C_ * H * W, H * W, W, 1_uz});
            }
            else if constexpr(ck::is_same<decltype(layout),
                                          ck::tensor_layout::convolution::NHWC>::value)
            {
                return HostTensorDescriptor({N_, C_, H, W}, {C_ * H * W, 1_uz, W * C_, C_});
            }
        };

    Tensor<InDataType> in_n_c_hi_wi(f_host_tensor_descriptor(N, C, Hi, Wi, InLayout{}));
    Tensor<OutDataType> out_n_c_ho_wo_host(f_host_tensor_descriptor(N, C, Ho, Wo, OutLayout{}));
    Tensor<IndexDataType> out_indices_n_c_ho_wo_host(
        f_host_tensor_descriptor(N, C, Ho, Wo, OutLayout{}));
    Tensor<OutDataType> out_n_c_ho_wo_device(f_host_tensor_descriptor(N, C, Ho, Wo, OutLayout{}));
    Tensor<IndexDataType> out_indices_n_c_ho_wo_device(
        f_host_tensor_descriptor(N, C, Ho, Wo, OutLayout{}));

    std::cout << "in_n_c_hi_wi: " << in_n_c_hi_wi.mDesc << std::endl;
    std::cout << "out_n_c_ho_wo: " << out_n_c_ho_wo_host.mDesc << std::endl;

    switch(init_method)
    {
    case 0: break;
    case 1: in_n_c_hi_wi.GenerateTensorValue(GeneratorTensor_1<InDataType>{1}); break;
    case 2: in_n_c_hi_wi.GenerateTensorValue(GeneratorTensor_2<InDataType>{-5, 5}); break;
    default: in_n_c_hi_wi.GenerateTensorValue(GeneratorTensor_3<InDataType>{-5.0, 5.0});
    }

    DeviceMem in_device_buf(sizeof(InDataType) * in_n_c_hi_wi.mDesc.GetElementSpaceSize());
    DeviceMem out_device_buf(sizeof(OutDataType) *
                             out_n_c_ho_wo_device.mDesc.GetElementSpaceSize());
    DeviceMem out_indices_device_buf(sizeof(IndexDataType) *
                                     out_indices_n_c_ho_wo_device.mDesc.GetElementSpaceSize());

    in_device_buf.ToDevice(in_n_c_hi_wi.mData.data());

    auto pool         = DevicePoolFwdInstance{};
    auto invoker_ptr  = pool.MakeInvokerPointer();
    auto argument_ptr = pool.MakeArgumentPointer(
        static_cast<InDataType*>(in_device_buf.GetDeviceBuffer()),
        static_cast<OutDataType*>(out_device_buf.GetDeviceBuffer()),
        static_cast<IndexDataType*>(out_indices_device_buf.GetDeviceBuffer()),
        N,
        C,
        std::array<ck::index_t, 2>{{Hi, Wi}},
        std::array<ck::index_t, 2>{{Y, X}},
        std::array<ck::index_t, 2>{{Ho, Wo}},
        window_strides,
        input_left_pads,
        input_right_pads);

    if(!pool.IsSupportedArgument(argument_ptr.get()))
    {
        throw std::runtime_error("wrong! device_op with the specified compilation parameters does "
                                 "not support this problem");
    }

    float ave_time = invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, time_kernel});

    std::size_t flop = std::size_t(2) * N * C * Ho * Wo * Y * X;

    std::size_t num_btype =
        sizeof(InDataType) * (N * C * Hi * Wi) + sizeof(OutDataType) * (N * C * Ho * Wo);

    float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

    float gb_per_sec = num_btype / 1.E6 / ave_time;

    std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s"
              << std::endl;

    bool pass = true;

    if(do_verification)
    {
        pool_host_verify<InDataType,
                         OutDataType,
                         AccDataType,
                         IndexDataType,
                         ReduceOpId,
                         PropagateNan,
                         OutputIndex>(in_n_c_hi_wi,
                                      out_n_c_ho_wo_host,
                                      out_indices_n_c_ho_wo_host,
                                      window_spatial_lengths,
                                      window_strides,
                                      input_left_pads,
                                      input_right_pads);

        out_device_buf.FromDevice(out_n_c_ho_wo_device.mData.data());

        pass = pass && ck::utils::check_err(out_n_c_ho_wo_device, out_n_c_ho_wo_host);

        if constexpr(OutputIndex)
        {
            out_indices_device_buf.FromDevice(out_indices_n_c_ho_wo_device.mData.data());

            pass = pass &&
                   ck::utils::check_err(out_indices_n_c_ho_wo_device, out_indices_n_c_ho_wo_host);
        };
    }

    return (pass);
};
