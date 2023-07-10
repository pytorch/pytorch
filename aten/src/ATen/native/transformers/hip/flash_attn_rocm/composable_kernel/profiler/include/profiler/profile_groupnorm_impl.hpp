// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iomanip>

#include "ck/ck.hpp"

#include "ck/library/tensor_operation_instance/gpu/normalization.hpp"

#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_groupnorm.hpp"

namespace ck {
namespace profiler {

template <typename XDataType,
          typename GammaDataType,
          typename BetaDataType,
          typename AccDataType,
          typename YDataType>
bool profile_groupnorm_impl(int do_verification,
                            int init_method,
                            bool do_log,
                            bool time_kernel,
                            std::vector<index_t> length)
{
    using PassThrough = ck::tensor_operation::element_wise::PassThrough;

    if(length.size() != 5)
        return false;

    index_t G = length[3];
    index_t C = length[4];

    std::vector<index_t> reduce_dim      = {1, 2, 4};
    std::vector<index_t> gammaBetaLength = {G, C};
    std::vector<index_t> gammaBetaStride = {0, 0, 0, C, 1};

    Tensor<XDataType> x(length);
    Tensor<GammaDataType> gamma(gammaBetaLength);
    Tensor<BetaDataType> beta(gammaBetaLength);
    Tensor<YDataType> y(length);
    Tensor<YDataType> host_y(length);

    switch(init_method)
    {
    case 0:
        x.GenerateTensorValue(GeneratorTensor_1<XDataType>{});
        gamma.GenerateTensorValue(GeneratorTensor_1<GammaDataType>{});
        beta.GenerateTensorValue(GeneratorTensor_1<BetaDataType>{});
        break;
    case 1:
        x.GenerateTensorValue(GeneratorTensor_2<XDataType>{-5, 5});
        gamma.GenerateTensorValue(GeneratorTensor_2<GammaDataType>{-5, 5});
        beta.GenerateTensorValue(GeneratorTensor_2<BetaDataType>{-5, 5});
        break;
    default:
        x.GenerateTensorValue(GeneratorTensor_3<XDataType>{0, 1});
        gamma.GenerateTensorValue(GeneratorTensor_3<GammaDataType>{-0.5, 0.5});
        beta.GenerateTensorValue(GeneratorTensor_3<BetaDataType>{-0.5, 0.5});
    }

    DeviceMem x_dev(sizeof(XDataType) * x.mDesc.GetElementSpaceSize());
    DeviceMem gamma_dev(sizeof(GammaDataType) * gamma.mDesc.GetElementSpaceSize());
    DeviceMem beta_dev(sizeof(BetaDataType) * beta.mDesc.GetElementSpaceSize());
    DeviceMem y_dev(sizeof(YDataType) * y.mDesc.GetElementSpaceSize());

    x_dev.ToDevice(x.mData.data());
    gamma_dev.ToDevice(gamma.mData.data());
    beta_dev.ToDevice(beta.mData.data());

    // add device normalization instances
    using DeviceOp = ck::tensor_operation::device::DeviceNormalization<XDataType,
                                                                       GammaDataType,
                                                                       BetaDataType,
                                                                       AccDataType,
                                                                       YDataType,
                                                                       PassThrough,
                                                                       5,
                                                                       3>;

    // get device op instances
    const auto instance_ptrs =
        ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
            DeviceOp>::GetInstances();

    std::cout << "found " << instance_ptrs.size() << " instances" << std::endl;

    std::string best_instance_name;
    float best_avg_time   = std::numeric_limits<float>::max();
    float best_gb_per_sec = 0;

    if(do_verification)
    {
        using ReferenceInstance = ck::tensor_operation::host::ReferenceGroupnorm<XDataType,
                                                                                 GammaDataType,
                                                                                 BetaDataType,
                                                                                 YDataType,
                                                                                 AccDataType,
                                                                                 PassThrough>;

        ReferenceInstance ref;
        auto ref_argument = ref.MakeArgument(x, gamma, beta, host_y, PassThrough{}, length, 1e-6);
        auto ref_invoker  = ref.MakeInvoker();
        ref_invoker.Run(ref_argument);
    }

    int num_kernel = 0;

    for(auto& inst_ptr : instance_ptrs)
    {
        auto argument_ptr = inst_ptr->MakeArgumentPointer(
            length,
            std::vector<ck::index_t>{x.mDesc.GetStrides().begin(), x.mDesc.GetStrides().end()},
            gammaBetaStride,
            gammaBetaStride,
            std::vector<ck::index_t>{y.mDesc.GetStrides().begin(), y.mDesc.GetStrides().end()},
            reduce_dim,
            1e-6,
            x_dev.GetDeviceBuffer(),
            gamma_dev.GetDeviceBuffer(),
            beta_dev.GetDeviceBuffer(),
            y_dev.GetDeviceBuffer(),
            nullptr,
            nullptr,
            PassThrough{});

        if(inst_ptr->IsSupportedArgument(argument_ptr.get()))
        {
            ++num_kernel;
        }
        else
        {
            continue;
        }

        auto invoker_ptr = inst_ptr->MakeInvokerPointer();

        float avg_time = invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, time_kernel});

        std::size_t num_bytes = x.mDesc.GetElementSize() * sizeof(XDataType) +
                                gamma.mDesc.GetElementSize() * sizeof(GammaDataType) +
                                beta.mDesc.GetElementSize() * sizeof(BetaDataType) +
                                y.mDesc.GetElementSize() * sizeof(YDataType);

        float gb_per_sec = num_bytes / 1.E6 / avg_time;

        if(time_kernel)
            std::cout << "Perf: " << std::setw(10) << avg_time << " ms, " << gb_per_sec << " GB/s, "
                      << inst_ptr->GetTypeString() << std::endl;

        if(avg_time < best_avg_time)
        {
            best_instance_name = inst_ptr->GetTypeString();
            best_avg_time      = avg_time;
            best_gb_per_sec    = gb_per_sec;
        }

        if(do_verification)
        {
            y_dev.FromDevice(y.mData.data());

            bool pass = ck::utils::check_err(y, host_y, "Error: Incorrect results", 1e-3, 1e-3);

            if(do_log)
            {
                LogRangeAsType<float>(std::cout << "x  : ", x.mData, ",") << std::endl;
                LogRangeAsType<float>(std::cout << "host_y  : ", host_y.mData, ",") << std::endl;
                LogRangeAsType<float>(std::cout << "y  : ", y.mData, ",") << std::endl;
            }

            if(!pass)
            {
                std::cout << inst_ptr->GetTypeString() << " failed verification: ";
                LogRange(std::cout << "lengths = [", length, ", ") << "]." << std::endl;
                return false;
            }
            else
            {
                if(time_kernel)
                    std::cout << "pass" << std::endl;
            }
        }
    }

    if(time_kernel)
    {
        LogRange(std::cout << "length = ", length, ",") << ", ";
        std::cout << "num_kernel = " << num_kernel << ", best perf = " << best_avg_time << " ms, "
                  << best_gb_per_sec << " GB/s, " << best_instance_name << std::endl;
    }

    if(num_kernel == 0)
    {
        std::cout << "Error: No kernel is applicable" << std::endl;
        return false;
    }

    return true;
}

} // namespace profiler
} // namespace ck
