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
#include "ck/library/tensor_operation_instance/gpu/batchnorm_forward.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_batchnorm_forward.hpp"

namespace ck {
namespace profiler {

template <typename XDataType,
          typename YDataType,
          typename AccDataType,
          typename ScaleDataType,
          typename BiasDataType,
          typename MeanVarDataType,
          index_t Rank,
          index_t NumBatchNormReduceDim>
bool profile_batchnorm_forward_impl(int do_verification,
                                    int init_method,
                                    bool do_dumpout,
                                    bool time_kernel,
                                    const std::vector<size_t> inOutLengths,
                                    const std::vector<int> reduceDims,
                                    bool updateMovingAverage,
                                    bool saveMeanAndInvVariance,
                                    double averageFactor,
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

    // input data of the batchnorm forward algorithm
    Tensor<XDataType> x(inOutLengths);
    Tensor<ScaleDataType> bnScale(scaleBiasMeanVarLengths);
    Tensor<BiasDataType> bnBias(scaleBiasMeanVarLengths);

    // output data of the batchnorm forward algorithm
    Tensor<YDataType> y_ref(inOutLengths);
    Tensor<YDataType> y(inOutLengths);

    Tensor<MeanVarDataType> resultSaveMean_ref(scaleBiasMeanVarLengths);
    Tensor<MeanVarDataType> resultSaveInvVariance_ref(scaleBiasMeanVarLengths);

    Tensor<MeanVarDataType> resultRunningMean_ref(scaleBiasMeanVarLengths);
    Tensor<MeanVarDataType> resultRunningVariance_ref(scaleBiasMeanVarLengths);

    auto inOutStrides            = x.mDesc.GetStrides();
    auto scaleBiasMeanVarStrides = bnScale.mDesc.GetStrides();

    std::size_t num_thread = std::thread::hardware_concurrency();

    if(updateMovingAverage)
    {
        const float x_mean       = 0.0f;
        const float x_stddev     = 1.0f;
        const float noise_stddev = 0.04f;

        // input data in normal distribution
        x.GenerateTensorValue(GeneratorTensor_4<XDataType>{x_mean, x_stddev}, num_thread);

        // initialize the runningMean to be values with tiny variation to the mean of the x
        // values
        resultRunningMean_ref.GenerateTensorValue(
            GeneratorTensor_4<MeanVarDataType>{x_mean, noise_stddev}, num_thread);

        // initialize the runningVariance to be values with tiny variation to the variance of
        // the x values
        resultRunningVariance_ref.GenerateTensorValue(
            GeneratorTensor_4<MeanVarDataType>{x_stddev * x_stddev, noise_stddev}, num_thread);
    }
    else
    {
        if constexpr(ck::is_same_v<XDataType, int8_t>)
            x.GenerateTensorValue(GeneratorTensor_2<XDataType>{-5, 5}, num_thread);
        else
            x.GenerateTensorValue(GeneratorTensor_3<XDataType>{-1.0f, 1.0f}, num_thread);
    };

    if(do_verification)
    {
        switch(init_method)
        {
        case 0:
            bnScale.GenerateTensorValue(GeneratorTensor_0<ScaleDataType>{}, num_thread);
            bnBias.GenerateTensorValue(GeneratorTensor_0<BiasDataType>{}, num_thread);
            break;
        case 1:
            bnScale.GenerateTensorValue(GeneratorTensor_1<ScaleDataType>{1}, num_thread);
            bnBias.GenerateTensorValue(GeneratorTensor_1<BiasDataType>{0}, num_thread);
            break;
        case 2:
            bnScale.GenerateTensorValue(GeneratorTensor_2<ScaleDataType>{-5, 5}, num_thread);
            bnBias.GenerateTensorValue(GeneratorTensor_2<BiasDataType>{-5, 5}, num_thread);
            break;
        default:
            bnScale.GenerateTensorValue(GeneratorTensor_3<ScaleDataType>{-1.0f, 1.0f}, num_thread);
            bnBias.GenerateTensorValue(GeneratorTensor_3<BiasDataType>{-1.0f, 1.0f}, num_thread);
        }
    };

    // these buffers are usually provided by the user application
    DeviceMem x_dev(sizeof(XDataType) * x.mDesc.GetElementSpaceSize());
    DeviceMem y_dev(sizeof(XDataType) * y.mDesc.GetElementSpaceSize());
    DeviceMem bnScale_dev(sizeof(ScaleDataType) * bnScale.mDesc.GetElementSpaceSize());
    DeviceMem bnBias_dev(sizeof(BiasDataType) * bnBias.mDesc.GetElementSpaceSize());

    // mean_dev or resultSaveMean_dev
    DeviceMem resultSaveMean_dev(sizeof(MeanVarDataType) *
                                 resultSaveMean_ref.mDesc.GetElementSpaceSize());
    // meansquare_dev or resultSaveInvVariance_dev
    DeviceMem resultSaveInvVariance_dev(sizeof(MeanVarDataType) *
                                        resultSaveInvVariance_ref.mDesc.GetElementSpaceSize());
    // resultRunningMean_dev
    DeviceMem resultRunningMean_dev(sizeof(MeanVarDataType) *
                                    resultRunningMean_ref.mDesc.GetElementSpaceSize());
    // resultRunningVariance_dev
    DeviceMem resultRunningVariance_dev(sizeof(MeanVarDataType) *
                                        resultRunningVariance_ref.mDesc.GetElementSpaceSize());

    x_dev.ToDevice(x.mData.data());
    bnScale_dev.ToDevice(bnScale.mData.data());
    bnBias_dev.ToDevice(bnBias.mData.data());

    if(updateMovingAverage)
    {
        resultRunningMean_dev.ToDevice(resultRunningMean_ref.mData.data());
        resultRunningVariance_dev.ToDevice(resultRunningVariance_ref.mData.data());
    };

    // used for storing the device result for verification when updateMovingAverage is enabled
    Tensor<MeanVarDataType> resultRunningMean(scaleBiasMeanVarLengths);
    Tensor<MeanVarDataType> resultRunningVariance(scaleBiasMeanVarLengths);

    // used for storing the device result for verification when saveMeanAndInvVariance is enabled
    Tensor<MeanVarDataType> resultSaveMean(scaleBiasMeanVarLengths);
    Tensor<MeanVarDataType> resultSaveInvVariance(scaleBiasMeanVarLengths);

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

    // add device batchnorm-forward instances
    using DeviceOp = ck::tensor_operation::device::DeviceBatchNormFwd<XDataType,
                                                                      YDataType,
                                                                      AccDataType,
                                                                      ScaleDataType,
                                                                      BiasDataType,
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
        using ReferenceBatchNormFwdInstance =
            ck::tensor_operation::host::ReferenceBatchNormFwd<XDataType,
                                                              YDataType,
                                                              AccDataType,
                                                              ScaleDataType,
                                                              BiasDataType,
                                                              MeanVarDataType,
                                                              PassThroughOp,
                                                              Rank,
                                                              NumBatchNormReduceDim>;

        auto batchNormFwd_ref = ReferenceBatchNormFwdInstance{};

        auto argument_ptr_ref = batchNormFwd_ref.MakeArgumentPointer(
            arrInOutLengths,
            arrInOutStrides,
            arrInOutStrides,
            arrReduceDims,
            arrScaleBiasMeanVarLengths,
            arrScaleBiasMeanVarStrides,
            arrScaleBiasMeanVarStrides,
            arrScaleBiasMeanVarStrides,
            x.mData.data(),
            bnScale.mData.data(),
            bnBias.mData.data(),
            epsilon,
            PassThroughOp{},
            y_ref.mData.data(),
            saveMeanAndInvVariance ? resultSaveMean_ref.mData.data() : nullptr,
            saveMeanAndInvVariance ? resultSaveInvVariance_ref.mData.data() : nullptr,
            averageFactor,
            updateMovingAverage ? resultRunningMean_ref.mData.data() : nullptr,
            updateMovingAverage ? resultRunningVariance_ref.mData.data() : nullptr);

        if(!batchNormFwd_ref.IsSupportedArgument(argument_ptr_ref.get()))
        {
            std::cout << "The runtime parameters not supported by the reference instance, exiting!"
                      << std::endl;
            return (false);
        };

        auto invoker_ptr_ref = batchNormFwd_ref.MakeInvokerPointer();

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
            arrReduceDims,
            arrScaleBiasMeanVarLengths,
            arrScaleBiasMeanVarStrides,
            arrScaleBiasMeanVarStrides,
            arrScaleBiasMeanVarStrides,
            x_dev.GetDeviceBuffer(),
            bnScale_dev.GetDeviceBuffer(),
            bnBias_dev.GetDeviceBuffer(),
            epsilon,
            PassThroughOp{},
            y_dev.GetDeviceBuffer(),
            saveMeanAndInvVariance ? resultSaveMean_dev.GetDeviceBuffer() : nullptr,
            saveMeanAndInvVariance ? resultSaveInvVariance_dev.GetDeviceBuffer() : nullptr,
            averageFactor,
            updateMovingAverage ? resultRunningMean_dev.GetDeviceBuffer() : nullptr,
            updateMovingAverage ? resultRunningVariance_dev.GetDeviceBuffer() : nullptr);

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

        // inputing of x, scale, bias, outputing of y
        num_bytes += total_length * (sizeof(XDataType) + sizeof(YDataType)) +
                     invariant_length * (sizeof(ScaleDataType) + sizeof(BiasDataType));

        // outputing of mean, inv-variance
        num_bytes += saveMeanAndInvVariance ? invariant_length * sizeof(MeanVarDataType) * 2 : 0;

        // updating of moving mean, variance
        num_bytes += updateMovingAverage ? invariant_length * sizeof(MeanVarDataType) * 4 : 0;

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
            bool single_pass;

            y_dev.FromDevice(y.mData.data());

            if constexpr(ck::is_same_v<YDataType, ck::bhalf_t>)
                single_pass = check_err(y.mData, y_ref.mData, "y results", 1e-2, 1e-2);
            else
                single_pass = check_err(y.mData, y_ref.mData, "y results", 4e-3, 4e-3);

            if(updateMovingAverage)
            {
                resultRunningMean_dev.FromDevice(resultRunningMean.mData.data());
                resultRunningVariance_dev.FromDevice(resultRunningVariance.mData.data());

                // clang-format off
                single_pass = single_pass && check_err(resultRunningMean.mData, resultRunningMean_ref.mData, "average mean results", 1.5e-5, 1.5e-5);
                single_pass = single_pass && check_err(resultRunningVariance.mData, resultRunningVariance_ref.mData, "average variance results", 1e-5, 1e-5);
                // clang-format on
            };

            if(saveMeanAndInvVariance)
            {
                resultSaveMean_dev.FromDevice(resultSaveMean.mData.data());
                resultSaveInvVariance_dev.FromDevice(resultSaveInvVariance.mData.data());

                // clang-format off
                single_pass = single_pass && check_err(resultSaveMean.mData, resultSaveMean_ref.mData, "mean results", 3e-5, 3e-5);
                single_pass = single_pass && check_err(resultSaveInvVariance.mData, resultSaveInvVariance_ref.mData, "inv-variance results", 7e-5, 7e-5);
                // clang-format on
            };

            pass = pass && single_pass;
        };

        if(do_dumpout)
        {
            using ck::host_common::dumpBufferToFile;

            // clang-format off
            dumpBufferToFile("dump_x.bin", x.mData.data(), x.mDesc.GetElementSize());
            dumpBufferToFile("dump_y.bin", y.mData.data(), y.mDesc.GetElementSize());
            dumpBufferToFile("dump_y_ref.bin", y_ref.mData.data(), y_ref.mDesc.GetElementSize());
            // clang-format off

            if(saveMeanAndInvVariance)
            {
                // clang-format off
                dumpBufferToFile("dump_mean.bin", resultSaveMean.mData.data(), resultSaveMean.mDesc.GetElementSize());
                dumpBufferToFile("dump_mean_ref.bin", resultSaveMean_ref.mData.data(), resultSaveMean_ref.mDesc.GetElementSize()); 
                dumpBufferToFile("dump_invvar.bin", resultSaveInvVariance.mData.data(), resultSaveInvVariance.mDesc.GetElementSize());
                dumpBufferToFile("dump_invvar_ref.bin", resultSaveInvVariance_ref.mData.data(), resultSaveInvVariance_ref.mDesc.GetElementSize());
                // clang-format on
            };
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
