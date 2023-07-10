// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iomanip>
#include <stdexcept>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/tensor_operation_instance/gpu/batchnorm_backward.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_batchnorm_backward.hpp"

namespace ck {
namespace profiler {

template <typename XDataType,
          typename DxDataType,
          typename DyDataType,
          typename AccDataType,
          typename ScaleDataType,
          typename DscaleDbiasDataType,
          typename MeanVarDataType,
          index_t Rank,
          index_t NumBatchNormReduceDim>
bool profile_batchnorm_backward_impl(bool do_verification,
                                     int init_method,
                                     bool do_dumpout,
                                     bool time_kernel,
                                     const std::vector<size_t> inOutLengths,
                                     const std::vector<int> reduceDims,
                                     bool haveSavedMeanInvVar,
                                     double epsilon)
{
    if(inOutLengths.size() != Rank || reduceDims.size() != NumBatchNormReduceDim)
    {
        throw std::runtime_error("Invalid tensor lengths or number of reduce dimensions!");
    };

    std::vector<size_t> scaleBiasMeanVarLengths;

    // used for calculating the effective transferred bytes by each operation
    size_t total_length;
    size_t invariant_length = 1;

    total_length =
        std::accumulate(inOutLengths.begin(), inOutLengths.end(), 1, std::multiplies<size_t>{});

    if(std::any_of(reduceDims.begin(), reduceDims.end(), [](int d) { return d < 0 || d >= Rank; }))
        throw std::runtime_error("Invalid reduce dimensions!");

    for(int dim = 0; dim < Rank; dim++)
    {
        if(std::none_of(reduceDims.begin(), reduceDims.end(), [&](int d) { return dim == d; }))
        {
            scaleBiasMeanVarLengths.push_back(inOutLengths[dim]);
            invariant_length *= inOutLengths[dim];
        };
    }

    // input data of the batchnorm backward algorithm
    Tensor<XDataType> x(inOutLengths);
    Tensor<DyDataType> dy(inOutLengths);
    Tensor<ScaleDataType> bnScale(scaleBiasMeanVarLengths);

    Tensor<MeanVarDataType> savedMean(scaleBiasMeanVarLengths);
    Tensor<MeanVarDataType> savedInvVar(scaleBiasMeanVarLengths);
    // savedVariance is only used for initializing savedInvVar
    Tensor<MeanVarDataType> savedVariance(scaleBiasMeanVarLengths);

    // output data of the batchnorm backward algorithm
    Tensor<DxDataType> dx_ref(inOutLengths);
    Tensor<DxDataType> dx(inOutLengths);

    Tensor<DscaleDbiasDataType> dscale(scaleBiasMeanVarLengths);
    Tensor<DscaleDbiasDataType> dbias(scaleBiasMeanVarLengths);

    Tensor<DscaleDbiasDataType> dscale_ref(scaleBiasMeanVarLengths);
    Tensor<DscaleDbiasDataType> dbias_ref(scaleBiasMeanVarLengths);

    auto inOutStrides            = x.mDesc.GetStrides();
    auto scaleBiasMeanVarStrides = bnScale.mDesc.GetStrides();

    std::size_t num_thread = std::thread::hardware_concurrency();

    if(haveSavedMeanInvVar)
    {
        const float x_mean       = 0.0f;
        const float x_stddev     = 1.0f;
        const float noise_stddev = 0.0001f;

        // input data in normal distribution
        x.GenerateTensorValue(GeneratorTensor_4<XDataType>{x_mean, x_stddev}, num_thread);

        // initialize the savedMean to be values with tiny variation to the mean of the x values
        savedMean.GenerateTensorValue(GeneratorTensor_4<MeanVarDataType>{x_mean, noise_stddev},
                                      num_thread);

        // initialize the variance to be values with tiny variation to the variance of the x values
        savedVariance.GenerateTensorValue(
            GeneratorTensor_4<MeanVarDataType>{x_stddev * x_stddev, noise_stddev}, num_thread);

        auto it_src       = savedVariance.mData.begin();
        auto it_dst       = savedInvVar.mData.begin();
        float tmp_epsilon = std::numeric_limits<float>::epsilon();

        while(it_src != savedVariance.mData.end())
        {
            *it_dst = type_convert<AccDataType>(
                1.0f / std::sqrtf(type_convert<float>(*it_src) + tmp_epsilon));

            it_src++;
            it_dst++;
        };
    }
    else
    {
        const float x_mean   = 0.0f;
        const float x_stddev = 1.0f;

        // input data in normal distribution
        x.GenerateTensorValue(GeneratorTensor_4<XDataType>{x_mean, x_stddev}, num_thread);
    };

    if(do_verification)
    {
        switch(init_method)
        {
        case 0:
            dy.GenerateTensorValue(GeneratorTensor_0<DyDataType>{}, num_thread);
            bnScale.GenerateTensorValue(GeneratorTensor_0<ScaleDataType>{}, num_thread);
            break;
        case 1:
            dy.GenerateTensorValue(GeneratorTensor_1<DyDataType>{1}, num_thread);
            bnScale.GenerateTensorValue(GeneratorTensor_1<ScaleDataType>{1}, num_thread);
            break;
        case 2:
            dy.GenerateTensorValue(GeneratorTensor_2<DyDataType>{-2, 2}, num_thread);
            bnScale.GenerateTensorValue(GeneratorTensor_2<ScaleDataType>{-5, 5}, num_thread);
            break;
        default:
            dy.GenerateTensorValue(GeneratorTensor_3<DyDataType>{-0.2f, 0.2f}, num_thread);
            bnScale.GenerateTensorValue(GeneratorTensor_3<ScaleDataType>{-0.5f, 0.5f}, num_thread);
        }
    };

    // input data of the batchnorm backward algorithm
    DeviceMem x_dev(sizeof(XDataType) * x.mDesc.GetElementSpaceSize());
    DeviceMem dy_dev(sizeof(DyDataType) * dy.mDesc.GetElementSpaceSize());

    DeviceMem bnScale_dev(sizeof(ScaleDataType) * bnScale.mDesc.GetElementSpaceSize());

    DeviceMem savedMean_dev(sizeof(MeanVarDataType) * savedMean.mDesc.GetElementSpaceSize());
    DeviceMem savedInvVar_dev(sizeof(MeanVarDataType) * savedInvVar.mDesc.GetElementSpaceSize());

    // output data of the batchnorm backward algorithm
    DeviceMem dx_dev(sizeof(DxDataType) * dx.mDesc.GetElementSpaceSize());

    DeviceMem dscale_dev(sizeof(DscaleDbiasDataType) * dscale.mDesc.GetElementSpaceSize());
    DeviceMem dbias_dev(sizeof(DscaleDbiasDataType) * dbias.mDesc.GetElementSpaceSize());

    x_dev.ToDevice(x.mData.data());
    dy_dev.ToDevice(dy.mData.data());
    bnScale_dev.ToDevice(bnScale.mData.data());

    if(haveSavedMeanInvVar)
    {
        savedMean_dev.ToDevice(savedMean.mData.data());
        savedInvVar_dev.ToDevice(savedInvVar.mData.data());
    };

    std::array<index_t, Rank> arrInOutLengths;
    std::array<index_t, Rank> arrInOutStrides;
    std::array<index_t, Rank - NumBatchNormReduceDim> arrScaleBiasMeanVarLengths;
    std::array<index_t, Rank - NumBatchNormReduceDim> arrScaleBiasMeanVarStrides;
    std::array<int, NumBatchNormReduceDim> arrReduceDims;

    std::copy(inOutLengths.begin(), inOutLengths.end(), arrInOutLengths.begin());
    std::copy(inOutStrides.begin(), inOutStrides.end(), arrInOutStrides.begin());
    std::copy(scaleBiasMeanVarLengths.begin(),
              scaleBiasMeanVarLengths.end(),
              arrScaleBiasMeanVarLengths.begin());
    std::copy(scaleBiasMeanVarStrides.begin(),
              scaleBiasMeanVarStrides.end(),
              arrScaleBiasMeanVarStrides.begin());

    std::copy(reduceDims.begin(), reduceDims.end(), arrReduceDims.begin());

    using PassThroughOp = ck::tensor_operation::element_wise::PassThrough;

    // add device batchnorm-backward instances
    using DeviceOp = ck::tensor_operation::device::DeviceBatchNormBwd<XDataType,
                                                                      DxDataType,
                                                                      DxDataType,
                                                                      AccDataType,
                                                                      ScaleDataType,
                                                                      DscaleDbiasDataType,
                                                                      MeanVarDataType,
                                                                      PassThroughOp,
                                                                      Rank,
                                                                      NumBatchNormReduceDim>;

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
        using ReferenceBatchNormBwdInstance =
            ck::tensor_operation::host::ReferenceBatchNormBwd<XDataType,
                                                              DxDataType,
                                                              DyDataType,
                                                              AccDataType,
                                                              ScaleDataType,
                                                              DscaleDbiasDataType,
                                                              MeanVarDataType,
                                                              PassThroughOp,
                                                              Rank,
                                                              NumBatchNormReduceDim>;

        auto batchNormBwd_ref = ReferenceBatchNormBwdInstance{};

        auto argument_ptr_ref = batchNormBwd_ref.MakeArgumentPointer(
            arrInOutLengths,
            arrInOutStrides,
            arrInOutStrides,
            arrInOutStrides,
            arrReduceDims,
            arrScaleBiasMeanVarLengths,
            arrScaleBiasMeanVarStrides,
            arrScaleBiasMeanVarStrides,
            arrScaleBiasMeanVarStrides,
            x.mData.data(),
            dy.mData.data(),
            bnScale.mData.data(),
            haveSavedMeanInvVar ? savedMean.mData.data() : nullptr,
            haveSavedMeanInvVar ? savedInvVar.mData.data() : nullptr,
            epsilon,
            PassThroughOp{},
            dx_ref.mData.data(),
            dscale_ref.mData.data(),
            dbias_ref.mData.data());

        if(!batchNormBwd_ref.IsSupportedArgument(argument_ptr_ref.get()))
        {
            std::cout << "The runtime parameters not supported by the reference instance, exiting!"
                      << std::endl;
            return (false);
        };

        auto invoker_ptr_ref = batchNormBwd_ref.MakeInvokerPointer();

        (void)invoker_ptr_ref->Run(argument_ptr_ref.get());
    }

    int num_kernel = 0;
    bool pass      = true;

    for(auto& inst_ptr : instance_ptrs)
    {
        auto argument_ptr = inst_ptr->MakeArgumentPointer(
            arrInOutLengths,
            arrInOutStrides,
            arrInOutStrides,
            arrInOutStrides,
            arrReduceDims,
            arrScaleBiasMeanVarLengths,
            arrScaleBiasMeanVarStrides,
            arrScaleBiasMeanVarStrides,
            arrScaleBiasMeanVarStrides,
            x_dev.GetDeviceBuffer(),
            dy_dev.GetDeviceBuffer(),
            bnScale_dev.GetDeviceBuffer(),
            haveSavedMeanInvVar ? savedMean_dev.GetDeviceBuffer() : nullptr,
            haveSavedMeanInvVar ? savedInvVar_dev.GetDeviceBuffer() : nullptr,
            epsilon,
            PassThroughOp{},
            dx_dev.GetDeviceBuffer(),
            dscale_dev.GetDeviceBuffer(),
            dbias_dev.GetDeviceBuffer());

        if(inst_ptr->IsSupportedArgument(argument_ptr.get()))
        {
            num_kernel++;
        }
        else
        {
            if(time_kernel)
            {
                std::cout << inst_ptr->GetTypeString()
                          << " skipped due to unsupported argument: " << std::endl;
            }

            continue;
        };

        size_t workspace_sz = inst_ptr->GetWorkSpaceSize(argument_ptr.get());

        DeviceMem workspace_dev(workspace_sz);

        inst_ptr->SetWorkSpacePointer(argument_ptr.get(), workspace_dev.GetDeviceBuffer());

        auto invoker_ptr = inst_ptr->MakeInvokerPointer();

        float avg_time = invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, time_kernel});

        size_t num_bytes = 0;

        // inputing of x, dy, scale, outputing of dx, dscale, dbias
        num_bytes += total_length * (sizeof(XDataType) + sizeof(DyDataType) + sizeof(DxDataType)) +
                     invariant_length * sizeof(DscaleDbiasDataType) * 2;

        // inputting of savedMean, savedInvVariance
        if(haveSavedMeanInvVar)
            num_bytes += invariant_length * sizeof(MeanVarDataType) * 2;

        float gb_per_sec = num_bytes / 1.E6 / avg_time;

        if(time_kernel)
            std::cout << "Perf: " << avg_time << " ms, " << gb_per_sec << " GB/s, "
                      << inst_ptr->GetTypeString() << std::endl;

        if(avg_time < best_avg_time)
        {
            best_instance_name = inst_ptr->GetTypeString();
            best_avg_time      = avg_time;
            best_gb_per_sec    = gb_per_sec;
        }

        if(do_verification)
        {
            using ck::utils::check_err;
            bool single_pass = true;

            dx_dev.FromDevice(dx.mData.data());
            dscale_dev.FromDevice(dscale.data());
            dbias_dev.FromDevice(dbias.data());

            // clang-format off
            single_pass = single_pass && ck::utils::check_err(dx.mData, dx_ref.mData, "dx result:", 5e-4, 5e-4);
            single_pass = single_pass && ck::utils::check_err(dscale.mData, dscale_ref.mData, "dScale result:", 3e-3, 3e-3);
            single_pass = single_pass && ck::utils::check_err(dbias.mData, dbias_ref.mData, "dBias result:", 3e-3, 3e-3);
            // clang-format on

            pass = pass && single_pass;
        };

        if(do_dumpout)
        {
            using ck::host_common::dumpBufferToFile;

            // clang-format off
            dumpBufferToFile("dump_x.bin", x.mData.data(), x.mDesc.GetElementSize());
            dumpBufferToFile("dump_dy.bin", dy.mData.data(), dy.mDesc.GetElementSize());
            dumpBufferToFile("dump_dx.bin", dx.mData.data(), dx.mDesc.GetElementSize());
            dumpBufferToFile("dump_dx_ref.bin", dx_ref.mData.data(), dx_ref.mDesc.GetElementSize());
            dumpBufferToFile("dump_dscale.bin", dscale.mData.data(), dscale.mDesc.GetElementSize());
            dumpBufferToFile("dump_dscale_ref.bin", dscale_ref.mData.data(), dscale_ref.mDesc.GetElementSize());
            // clang-format off
        };
    }

    if(time_kernel)
    {
        std::cout << "best perf = " << best_avg_time << " ms, " << best_gb_per_sec << " GB/s, "
                  << best_instance_name << std::endl;
    }

    if(num_kernel == 0)
    {
        std::cout << "Error: No kernel is applicable" << std::endl;
        return false;
    }

    return pass;
}

} // namespace profiler
} // namespace ck
