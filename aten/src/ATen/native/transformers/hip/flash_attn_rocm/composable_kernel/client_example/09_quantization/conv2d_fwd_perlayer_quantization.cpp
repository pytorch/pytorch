// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <iomanip>
#include <iostream>
#include <vector>

#include "ck/ck.hpp"
#include "ck/library/tensor_operation_instance/gpu/quantization/grouped_convolution_forward_perlayer_quantization.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_conv_fwd.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

using InDataType  = int8_t;
using WeiDataType = int8_t;
using OutDataType = int8_t;

using InLayout     = ck::tensor_layout::convolution::GNHWC;
using WeiLayout    = ck::tensor_layout::convolution::GKYXC;
using OutLayout    = ck::tensor_layout::convolution::GNHWK;
using PassThrough  = ck::tensor_operation::element_wise::PassThrough;
using ActivationOp = PassThrough;
using OutElementOp = ck::tensor_operation::element_wise::Activation_Mul_Clamp<ActivationOp>;

static constexpr ck::index_t NumDimSpatial = 2;
static constexpr ck::index_t G             = 1;
static constexpr ck::index_t N             = 4;
static constexpr ck::index_t K             = 64;
static constexpr ck::index_t C             = 32;
static constexpr ck::index_t Y             = 3;
static constexpr ck::index_t X             = 3;
static constexpr ck::index_t Hi            = 71;
static constexpr ck::index_t Wi            = 71;
static constexpr ck::index_t Ho            = 36;
static constexpr ck::index_t Wo            = 36;

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
    std::array<ck::index_t, 5> in_lengths{G, N, C, Hi, Wi};
    std::array<ck::index_t, 5> in_strides{N * Hi * Wi * C, Hi * Wi * C, 1, Wi * C, C};
    std::array<ck::index_t, 5> weight_lengths{G, K, C, Y, X};
    std::array<ck::index_t, 5> weight_strides{K * Y * X * C, Y * X * C, 1, X * C, C};
    std::array<ck::index_t, 5> out_lengths{G, N, C, Ho, Wo};
    std::array<ck::index_t, 5> out_strides{N * Ho * Wo * C, Ho * Wo * C, 1, Wo * C, C};
    std::array<ck::index_t, 2> in_left_pad{1, 1};
    std::array<ck::index_t, 2> in_right_pad{1, 1};
    std::array<ck::index_t, 2> conv_strides{2, 2};
    std::array<ck::index_t, 2> conv_dilations{1, 1};

    SimpleDeviceMem in(sizeof(InDataType) * N * Hi * Wi * C);
    SimpleDeviceMem wei(sizeof(WeiDataType) * K * Y * X * C);
    SimpleDeviceMem out(sizeof(OutDataType) * N * Ho * Wo * K);

    using DeviceOp = ck::tensor_operation::device::DeviceGroupedConvFwdMultipleD<NumDimSpatial,
                                                                                 InLayout,
                                                                                 WeiLayout,
                                                                                 ck::Tuple<>,
                                                                                 OutLayout,
                                                                                 InDataType,
                                                                                 WeiDataType,
                                                                                 ck::Tuple<>,
                                                                                 OutDataType,
                                                                                 PassThrough,
                                                                                 PassThrough,
                                                                                 OutElementOp>;
    // get device op instances
    const auto op_ptrs = ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
        DeviceOp>::GetInstances();

    std::cout << "found " << op_ptrs.size() << " instances" << std::endl;

    std::string best_op_name;
    int best_op_id        = -1;
    float best_avg_time   = std::numeric_limits<float>::max();
    float best_gb_per_sec = 0;
    float best_tflops     = 0;

    // profile device operation instances
    std::cout << "Run all instances and do timing" << std::endl;

    for(int i = 0; i < op_ptrs.size(); ++i)
    {
        auto& op_ptr      = op_ptrs[i];
        auto argument_ptr = op_ptr->MakeArgumentPointer(in.GetDeviceBuffer(),
                                                        wei.GetDeviceBuffer(),
                                                        {},
                                                        out.GetDeviceBuffer(),
                                                        in_lengths,
                                                        in_strides,
                                                        weight_lengths,
                                                        weight_strides,
                                                        {},
                                                        {},
                                                        out_lengths,
                                                        out_strides,
                                                        conv_strides,
                                                        conv_dilations,
                                                        in_left_pad,
                                                        in_right_pad,
                                                        PassThrough{},
                                                        PassThrough{},
                                                        OutElementOp{0.5f, ActivationOp{}});

        auto invoker_ptr    = op_ptr->MakeInvokerPointer();
        std::string op_name = op_ptr->GetTypeString();

        if(op_ptr->IsSupportedArgument(argument_ptr.get()))
        {
            float avg_time = invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, true});

            std::size_t flop      = G * 2 * N * K * C * Ho * Wo * Y * X;
            std::size_t num_bytes = G * sizeof(InDataType) * N * Hi * Wi * C +
                                    G * sizeof(WeiDataType) * K * Y * X * C +
                                    G * sizeof(OutDataType) * N * Ho * Wo * K;

            float tflops     = static_cast<float>(flop) / 1.E9 / avg_time;
            float gb_per_sec = num_bytes / 1.E6 / avg_time;

            std::cout << "Perf: " << std::setw(10) << avg_time << " ms, " << tflops << " TFlops, "
                      << gb_per_sec << " GB/s, " << op_name << std::endl;

            if(tflops > best_tflops)
            {
                best_op_id      = i;
                best_op_name    = op_name;
                best_avg_time   = avg_time;
                best_gb_per_sec = gb_per_sec;
                best_tflops     = tflops;
            }
        }
        else
        {
            std::cout << op_name << " does not support this problem" << std::endl;
        }
    }

    std::cout << "Best Perf: " << std::setw(10) << best_avg_time << " ms, " << best_tflops
              << " TFlops, " << best_gb_per_sec << " GB/s, " << best_op_name << std::endl;

    // run the best intance
    {
        auto& op_ptr = op_ptrs[best_op_id];
        std::cout << "Run the best instance without timing: " << op_ptr->GetTypeString()
                  << std::endl;
        auto argument_ptr = op_ptr->MakeArgumentPointer(in.GetDeviceBuffer(),
                                                        wei.GetDeviceBuffer(),
                                                        {},
                                                        out.GetDeviceBuffer(),
                                                        in_lengths,
                                                        in_strides,
                                                        weight_lengths,
                                                        weight_strides,
                                                        {},
                                                        {},
                                                        out_lengths,
                                                        out_strides,
                                                        conv_strides,
                                                        conv_dilations,
                                                        in_left_pad,
                                                        in_right_pad,
                                                        PassThrough{},
                                                        PassThrough{},
                                                        OutElementOp{0.5f, ActivationOp{}});

        auto invoker_ptr = op_ptr->MakeInvokerPointer();

        if(op_ptr->IsSupportedArgument(argument_ptr.get()))
        {
            invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, false});
        }

        std::cout << "Done" << std::endl;
    }

    return 0;
}