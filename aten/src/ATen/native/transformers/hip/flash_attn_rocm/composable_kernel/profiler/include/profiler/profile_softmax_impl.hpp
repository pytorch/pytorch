// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "ck/ck.hpp"
#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/fill.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_softmax.hpp"
#include "ck/library/tensor_operation_instance/gpu/softmax.hpp"
#include "ck/tensor_operation/gpu/device/device_softmax.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/utility/data_type.hpp"

namespace ck {
namespace profiler {

enum struct SoftmaxDataType
{
    F32_F32, // in, out
    F16_F16,
    BF16_BF16,
    INT8_INT8,
};

// clang-format off
template <typename SoftmaxDataType> std::string type_to_string();
template <> std::string type_to_string<float>()   { return "f32"; }
template <> std::string type_to_string<half_t>()  { return "f16"; }
template <> std::string type_to_string<bhalf_t>() { return "bf16"; }
template <> std::string type_to_string<int8_t>()  { return "int8"; }
template <> std::string type_to_string<int32_t>() { return "int32"; }
// clang-format on

template <typename InDataType, typename AccDataType, typename OutDataType, index_t Rank>
bool profile_softmax_impl(int do_verification,
                          int init_method,
                          bool do_log,
                          bool time_kernel,
                          std::vector<index_t> in_length,
                          std::vector<index_t> in_strides,
                          std::vector<index_t> reduce_dims,
                          AccDataType alpha,
                          AccDataType beta)
{
    if(Rank != in_length.size())
    {
        throw std::runtime_error("Input tensor rank is different from template argument Rank!");
    }

    Tensor<InDataType> in = in_strides.empty() ? Tensor<InDataType>(in_length)
                                               : Tensor<InDataType>(in_length, in_strides);
    Tensor<OutDataType> out(in.mDesc);
    Tensor<OutDataType> prior_out(in.mDesc);

    switch(init_method)
    {
    case 0: break;
    case 1:
        ck::utils::FillUniformDistributionIntegerValue<InDataType>{-5.f, 5.f}(in.begin(), in.end());
        ck::utils::FillUniformDistributionIntegerValue<OutDataType>{-5.f, 5.f}(prior_out.begin(),
                                                                               prior_out.end());
        break;
    default:
        ck::utils::FillUniformDistribution<InDataType>{0.0f, 1.0f}(in);
        ck::utils::FillUniformDistribution<OutDataType>{-0.5f, 0.5f}(prior_out);
    }

    Tensor<OutDataType> out_ref(prior_out);

    if(do_verification)
    {
        using ReferenceSoftmax =
            tensor_operation::host::ReferenceSoftmax<InDataType, OutDataType, AccDataType>;
        ReferenceSoftmax{}.MakeInvoker().Run({in, out_ref, alpha, beta, reduce_dims});
    }

    DeviceMem in_dev(in.GetElementSpaceSizeInBytes());
    DeviceMem out_dev(out.GetElementSpaceSizeInBytes());
    in_dev.ToDevice(in.data());

    std::vector<index_t> in_tensor_lengths(in.GetLengths().begin(), in.GetLengths().end());
    std::vector<index_t> in_tensor_strides(in.GetStrides().begin(), in.GetStrides().end());

    // add device softmax instances
    using PassThrough = ck::tensor_operation::element_wise::PassThrough;
    using DeviceOp    = tensor_operation::device::
        DeviceSoftmax<InDataType, AccDataType, OutDataType, PassThrough, PassThrough, Rank>;

    // get device op instances
    const auto instances = tensor_operation::device::instance::DeviceOperationInstanceFactory<
        DeviceOp>::GetInstances();
    std::cout << "found " << instances.size() << " instances" << std::endl;

    if(instances.size() <= 0)
    {
        throw std::runtime_error("wrong! no device normalization instance found");
    }

    std::string best_instance_name;
    float best_avg_time   = std::numeric_limits<float>::max();
    float best_gb_per_sec = 0;
    std::vector<bool> instance_pass;

    for(auto& inst_ptr : instances)
    {
        // Is this user's responsibility to check if problem mismatches kernel instance (ie. rank 3
        // problem to rank 4 kernel) other than invoking IsSupportedArgument()?
        if(!(inst_ptr->GetNumReduceDim() == static_cast<index_t>(reduce_dims.size())))
        {
            continue;
        }

        auto argument_ptr = inst_ptr->MakeArgumentPointer(in_tensor_lengths,
                                                          in_tensor_strides,
                                                          reduce_dims,
                                                          &alpha,
                                                          &beta,
                                                          in_dev.GetDeviceBuffer(),
                                                          out_dev.GetDeviceBuffer(),
                                                          PassThrough{},
                                                          PassThrough{});

        if(!inst_ptr->IsSupportedArgument(argument_ptr.get()))
        {
            std::cout << inst_ptr->GetTypeString() << " skipped due to unsupported argument: ";
            LogRange(std::cout << "input lengths = [", in_length, ", ")
                << "], "
                << "scaler = [" << alpha << ", " << beta << "]";
            LogRange(std::cout << ", reduce dims = [", reduce_dims, ", ") << "]." << std::endl;
            instance_pass.push_back(true);
            continue;
        }

        out_dev.ToDevice(prior_out.data());
        auto invoker_ptr = inst_ptr->MakeInvokerPointer();
        float avg_time   = invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, time_kernel});

        if(time_kernel)
        {
            std::size_t num_bytes =
                in.GetElementSize() * sizeof(InDataType) +
                (beta == 0.0f ? 1 : 2) * out.GetElementSize() * sizeof(OutDataType);
            float gb_per_sec = num_bytes / 1.E6 / avg_time;

            std::cout << "Perf: " << std::setw(10) << avg_time << " ms, " << gb_per_sec << " GB/s, "
                      << inst_ptr->GetTypeString() << std::endl;

            if(avg_time < best_avg_time)
            {
                best_instance_name = inst_ptr->GetTypeString();
                best_avg_time      = avg_time;
                best_gb_per_sec    = gb_per_sec;
            }
        }

        if(do_verification)
        {
            out_dev.FromDevice(out.data());
            bool pass = true;
            if(std::is_same<InDataType, int8_t>::value)
            {
                pass = pass && ck::utils::check_err(
                                   out.mData, out_ref.mData, "Error: Incorrect results!", 0, 1);
                if(do_log)
                {
                    LogRangeAsType<int>(std::cout << "in  : ", in.mData, ",") << std::endl;
                    LogRangeAsType<int>(std::cout << "out_ref  : ", out_ref.mData, ",")
                        << std::endl;
                    LogRangeAsType<int>(std::cout << "out  : ", out.mData, ",") << std::endl;
                }
            }
            else
            {
                pass = pass && ck::utils::check_err(out.mData, out_ref.mData);
                if(do_log)
                {
                    LogRangeAsType<float>(std::cout << "in  : ", in.mData, ",") << std::endl;
                    LogRangeAsType<float>(std::cout << "out_ref  : ", out_ref.mData, ",")
                        << std::endl;
                    LogRangeAsType<float>(std::cout << "out  : ", out.mData, ",") << std::endl;
                }
            }

            if(!pass)
            {
                std::cout << inst_ptr->GetTypeString() << " failed verification: ";
                LogRange(std::cout << "input lengths = [", in_length, ", ")
                    << "], "
                    << "scaler = [" << alpha << ", " << beta << "]." << std::endl;
            }
            instance_pass.push_back(pass);
        }
    }
    if(time_kernel)
    {
        std::cout << "Best Perf for datatype = " << type_to_string<InDataType>() << "_"
                  << type_to_string<OutDataType>() << ", ";
        LogRange(std::cout << "length = ", in_tensor_lengths, ",") << ", ";
        LogRange(std::cout << "stride = ", in_tensor_strides, ",") << ", ";
        LogRange(std::cout << "reduce dims ", reduce_dims, ",") << ", ";
        std::cout << "alpha = " << alpha << ", "
                  << "beta = " << beta << ", " << best_avg_time << " ms, " << best_gb_per_sec
                  << " GB/s, " << best_instance_name << std::endl;
    }
    return std::all_of(
        std::begin(instance_pass), std::end(instance_pass), [](bool p) { return p; });
}

} // namespace profiler
} // namespace ck
