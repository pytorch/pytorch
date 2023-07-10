// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <functional>
#include <numeric>
#include <iomanip>
#include <iostream>
#include <vector>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_softmax.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/tensor_operation_instance/gpu/softmax.hpp"

using InDataType  = ck::half_t;
using OutDataType = ck::half_t;
using AccDataType = float;
using PassThrough = ck::tensor_operation::element_wise::PassThrough;

constexpr int Rank         = 4;
constexpr int NumReduceDim = 2;

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
    std::vector<ck::index_t> in_lengths{2, 8, 128, 1024};
    std::vector<ck::index_t> in_strides{8 * 128 * 1024, 128 * 1024, 1024, 1};
    std::vector<ck::index_t> reduce_dims{2, 3};

    ck::index_t num_elements =
        std::accumulate(in_lengths.begin(), in_lengths.end(), 1, std::multiplies<ck::index_t>());

    AccDataType alpha{2.0f};
    AccDataType beta{2.0f};

    SimpleDeviceMem in(sizeof(InDataType) * num_elements);
    SimpleDeviceMem out(sizeof(OutDataType) * num_elements);

    using DeviceOp = ck::tensor_operation::device::
        DeviceSoftmax<InDataType, AccDataType, OutDataType, PassThrough, PassThrough, Rank>;
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

        if(op_ptr->GetRank() != Rank || op_ptr->GetNumReduceDim() != NumReduceDim)
        {
            continue;
        }

        auto argument_ptr   = op_ptr->MakeArgumentPointer(in_lengths,
                                                        in_strides,
                                                        reduce_dims,
                                                        &alpha,
                                                        &beta,
                                                        in.GetDeviceBuffer(),
                                                        out.GetDeviceBuffer(),
                                                        PassThrough{},
                                                        PassThrough{});
        auto invoker_ptr    = op_ptr->MakeInvokerPointer();
        std::string op_name = op_ptr->GetTypeString();

        if(op_ptr->IsSupportedArgument(argument_ptr.get()))
        {
            float ave_time = invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, true});

            std::size_t num_bytes = num_elements * sizeof(InDataType) +
                                    (beta == 0.0f ? 1 : 2) * num_elements * sizeof(OutDataType);

            float gb_per_sec = num_bytes / 1.E6 / ave_time;

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
        auto argument_ptr = op_ptr->MakeArgumentPointer(in_lengths,
                                                        in_strides,
                                                        reduce_dims,
                                                        &alpha,
                                                        &beta,
                                                        in.GetDeviceBuffer(),
                                                        out.GetDeviceBuffer(),
                                                        PassThrough{},
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