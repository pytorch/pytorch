// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <iomanip>
#include <vector>
#include <iostream>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_normalization.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/tensor_operation_instance/gpu/normalization.hpp"

using XDataType     = ck::half_t;
using GammaDataType = ck::half_t;
using BetaDataType  = ck::half_t;
using YDataType     = ck::half_t;
using AccDataType   = float;
using PassThrough   = ck::tensor_operation::element_wise::PassThrough;

constexpr int Rank         = 2;
constexpr int NumReduceDim = 1;

struct SimpleDeviceMem
{
    SimpleDeviceMem() = delete;

    SimpleDeviceMem(std::size_t mem_size) : p_mem_{}
    {
        (void)hipMalloc(static_cast<void**>(&p_mem_), mem_size);
    }

    void* GetDeviceBuffer() { return p_mem_; }

    ~SimpleDeviceMem() { (void)hipFree(p_mem_); }

    void* p_mem_;
};

int main(int argc, char* argv[])
{
    ck::index_t M      = 1024;
    ck::index_t N      = 1024;
    ck::index_t Stride = 1024;

    auto xy_size = (M - 1) * Stride + N;

    SimpleDeviceMem x_device_buf(sizeof(XDataType) * xy_size);
    SimpleDeviceMem gamma_device_buf(sizeof(GammaDataType) * N);
    SimpleDeviceMem beta_device_buf(sizeof(BetaDataType) * N);
    SimpleDeviceMem y_device_buf(sizeof(YDataType) * xy_size);

    using DeviceOp = ck::tensor_operation::device::DeviceNormalization<XDataType,
                                                                       GammaDataType,
                                                                       BetaDataType,
                                                                       AccDataType,
                                                                       YDataType,
                                                                       PassThrough,
                                                                       Rank,
                                                                       NumReduceDim>;

    // get device op instances
    const auto op_ptrs = ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
        DeviceOp>::GetInstances();

    std::cout << "found " << op_ptrs.size() << " instances" << std::endl;

    std::string best_op_name;
    bool found            = false;
    int best_op_id        = -1;
    float best_ave_time   = std::numeric_limits<float>::max();
    float best_gb_per_sec = 0;

    // profile device operation instances
    std::cout << "Run all instances and do timing" << std::endl;

    for(int i = 0; i < op_ptrs.size(); ++i)
    {
        auto& op_ptr = op_ptrs[i];

        auto argument_ptr = op_ptr->MakeArgumentPointer({M, N},      // lengths
                                                        {Stride, 1}, // xStrides
                                                        {0, 1},      // gammaStrides
                                                        {0, 1},      // betaStrides
                                                        {Stride, 1}, // yStrides
                                                        {1},         // reduceDims
                                                        1e-4,
                                                        x_device_buf.GetDeviceBuffer(),
                                                        gamma_device_buf.GetDeviceBuffer(),
                                                        beta_device_buf.GetDeviceBuffer(),
                                                        y_device_buf.GetDeviceBuffer(),
                                                        nullptr,
                                                        nullptr,
                                                        PassThrough{});

        auto invoker_ptr = op_ptr->MakeInvokerPointer();

        std::string op_name = op_ptr->GetTypeString();

        if(op_ptr->IsSupportedArgument(argument_ptr.get()))
        {
            float ave_time = invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, true});

            std::size_t num_byte = sizeof(XDataType) * M * N + sizeof(GammaDataType) * N +
                                   sizeof(BetaDataType) * N + sizeof(YDataType) * M * N;

            float gb_per_sec = num_byte / 1.E6 / ave_time;

            std::cout << "Perf: " << std::setw(10) << ave_time << " ms, " << gb_per_sec << " GB/s, "
                      << op_name << std::endl;

            if(ave_time < best_ave_time)
            {
                found           = true;
                best_op_id      = i;
                best_op_name    = op_name;
                best_ave_time   = ave_time;
                best_gb_per_sec = gb_per_sec;
            }
        }
        else
        {
            std::cout << op_name << " does not support this problem" << std::endl;
        }
    }

    std::cout << "Best Perf: " << best_ave_time << " ms, " << best_gb_per_sec << " GB/s, "
              << best_op_name << std::endl;

    // run the best intance
    {
        auto& op_ptr = op_ptrs[best_op_id];
        std::cout << "Run the best instance without timing: " << op_ptr->GetTypeString()
                  << std::endl;

        auto argument_ptr = op_ptr->MakeArgumentPointer({M, N},      // lengths
                                                        {Stride, 1}, // xStrides
                                                        {1},         // gammaStrides
                                                        {1},         // betaStrides
                                                        {Stride, 1}, // yStrides
                                                        {1},         // reduceDims
                                                        1e-4,
                                                        x_device_buf.GetDeviceBuffer(),
                                                        gamma_device_buf.GetDeviceBuffer(),
                                                        beta_device_buf.GetDeviceBuffer(),
                                                        y_device_buf.GetDeviceBuffer(),
                                                        nullptr,
                                                        nullptr,
                                                        PassThrough{});

        auto invoker_ptr = op_ptr->MakeInvokerPointer();

        if(op_ptr->IsSupportedArgument(argument_ptr.get()))
        {
            invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, false});
        }

        std::cout << "Done" << std::endl;
    }

    return 0;
}
