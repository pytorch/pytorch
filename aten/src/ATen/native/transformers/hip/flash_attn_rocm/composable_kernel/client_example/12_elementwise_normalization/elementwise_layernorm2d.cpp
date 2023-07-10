// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <iomanip>
#include <vector>
#include <iostream>

#include "ck/ck.hpp"
#include "ck/utility/reduction_enums.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_elementwise_normalization_impl.hpp"
#include "ck/tensor_operation/gpu/device/reduction_operator_mapping.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/tensor_operation_instance/gpu/elementwise_normalization.hpp"

using ADataType             = ck::half_t; // Input 1
using BDataType             = ck::half_t; // Input 2
using XDataType             = ck::half_t;
using GammaDataType         = ck::half_t;
using BetaDataType          = ck::half_t;
using YDataType             = ck::half_t;
using AccDataType           = float;
using XElementwiseOperation = ck::tensor_operation::element_wise::Add;
using YElementwiseOperation = ck::tensor_operation::element_wise::PassThrough;

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

int main()
{
    bool time_kernel = true;

    ck::index_t M      = 48 * 256;
    ck::index_t N      = 1024;
    ck::index_t Stride = N;

    auto mn_size = (M - 1) * Stride + N;

    SimpleDeviceMem a_dev_buf(sizeof(ADataType) * mn_size);
    SimpleDeviceMem b_dev_buf(sizeof(BDataType) * mn_size);
    SimpleDeviceMem gamma_dev_buf(sizeof(GammaDataType) * N);
    SimpleDeviceMem beta_dev_buf(sizeof(BetaDataType) * N);
    SimpleDeviceMem y_dev_buf(sizeof(YDataType) * mn_size);

    std::array<const void*, 2> ab_input               = {a_dev_buf.GetDeviceBuffer(),
                                           b_dev_buf.GetDeviceBuffer()};
    std::vector<ck::index_t> abStride                 = {Stride, 1};
    std::array<std::vector<ck::index_t>, 2> abStrides = {abStride, abStride};

    using DeviceOp = ck::tensor_operation::device::DeviceElementwiseNormalization<
        ck::Tuple<ADataType, BDataType>,
        GammaDataType,
        BetaDataType,
        AccDataType,
        YDataType,
        XElementwiseOperation,
        YElementwiseOperation,
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

        auto argument_ptr = op_ptr->MakeArgumentPointer({M, N}, // lengths
                                                        abStrides,
                                                        {0, 1},      // gammaStrides
                                                        {0, 1},      // betaStrides
                                                        {Stride, 1}, // yStrides
                                                        {1},         // reduceDims
                                                        1e-4,
                                                        ab_input,
                                                        gamma_dev_buf.GetDeviceBuffer(),
                                                        beta_dev_buf.GetDeviceBuffer(),
                                                        y_dev_buf.GetDeviceBuffer(),
                                                        XElementwiseOperation{},
                                                        YElementwiseOperation{});

        auto invoker_ptr = op_ptr->MakeInvokerPointer();

        std::string op_name = op_ptr->GetTypeString();

        if(op_ptr->IsSupportedArgument(argument_ptr.get()))
        {
            float ave_time = invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, true});

            std::size_t num_byte = sizeof(ADataType) * M * N + sizeof(BDataType) * M * N +
                                   sizeof(GammaDataType) * N + sizeof(BetaDataType) * N +
                                   sizeof(YDataType) * M * N;

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

        auto argument_ptr = op_ptr->MakeArgumentPointer({M, N}, // lengths
                                                        abStrides,
                                                        {1},         // gammaStrides
                                                        {1},         // betaStrides
                                                        {Stride, 1}, // yStrides
                                                        {1},         // reduceDims
                                                        1e-4,
                                                        ab_input,
                                                        gamma_dev_buf.GetDeviceBuffer(),
                                                        beta_dev_buf.GetDeviceBuffer(),
                                                        y_dev_buf.GetDeviceBuffer(),
                                                        XElementwiseOperation{},
                                                        YElementwiseOperation{});

        auto invoker_ptr = op_ptr->MakeInvokerPointer();

        if(op_ptr->IsSupportedArgument(argument_ptr.get()))
        {
            invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, false});
        }

        std::cout << "Done" << std::endl;
    }

    return 0;
}
