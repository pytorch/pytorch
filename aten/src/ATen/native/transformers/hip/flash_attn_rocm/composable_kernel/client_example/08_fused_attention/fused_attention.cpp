// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <iostream>
#include <vector>

#include "ck/ck.hpp"
#include "ck/library/tensor_operation_instance/gpu/batched_gemm_softmax_gemm_permute.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_batched_gemm_softmax_gemm_permute.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

using AElementOp    = ck::tensor_operation::element_wise::PassThrough;
using B0ElementOp   = ck::tensor_operation::element_wise::PassThrough;
using Acc0ElementOp = ck::tensor_operation::element_wise::Scale;
using B1ElementOp   = ck::tensor_operation::element_wise::PassThrough;
using CElementOp    = ck::tensor_operation::element_wise::PassThrough;

constexpr static auto MaskingSpec =
    ck::tensor_operation::device::MaskingSpecialization::MaskDisabled;

using ADataType   = ck::half_t;
using B0DataType  = ck::half_t;
using B1DataType  = ck::half_t;
using CDataType   = ck::half_t;
using AccDataType = float;

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
    int G0 = 48;
    int G1 = 16;
    int M  = 1024;
    int N  = 1024;
    int K  = 64;
    int O  = 64;

    // A layout [G0, M, G1, K]
    std::vector<ck::index_t> a_gs_ms_ks_lengths{G0, G1, M, K};
    std::vector<ck::index_t> a_gs_ms_ks_strides{M * G1 * K, K, G1 * K, 1};

    // B0 layout [G0, N, G1, K]
    std::vector<ck::index_t> b0_gs_ns_ks_lengths{G0, G1, N, K};
    std::vector<ck::index_t> b0_gs_ns_ks_strides{N * G1 * K, K, G1 * K, 1};

    // B1 layout [G0, N, G1, O]
    std::vector<ck::index_t> b1_gs_os_ns_lengths{G0, G1, O, N};
    std::vector<ck::index_t> b1_gs_os_ns_strides{N * G1 * O, O, 1, G1 * O};

    // C layout [G0, M, G1, O]
    std::vector<ck::index_t> c_gs_ms_os_lengths{G0, G1, M, O};
    std::vector<ck::index_t> c_gs_ms_os_strides{M * G1 * O, O, G1 * O, 1};

    SimpleDeviceMem a_device_buf(sizeof(ADataType) * G0 * G1 * M * K);
    SimpleDeviceMem b0_device_buf(sizeof(B0DataType) * G0 * G1 * N * K);
    SimpleDeviceMem b1_device_buf(sizeof(B1DataType) * G0 * G1 * O * N);
    SimpleDeviceMem c_device_buf(sizeof(CDataType) * G0 * G1 * M * O);

    using DeviceOp =
        ck::tensor_operation::device::DeviceBatchedGemmSoftmaxGemmPermute<2,
                                                                          1,
                                                                          1,
                                                                          1,
                                                                          1,
                                                                          ADataType,
                                                                          B0DataType,
                                                                          B1DataType,
                                                                          CDataType,
                                                                          ck::Tuple<>,
                                                                          ck::Tuple<>,
                                                                          AElementOp,
                                                                          B0ElementOp,
                                                                          Acc0ElementOp,
                                                                          B1ElementOp,
                                                                          CElementOp,
                                                                          MaskingSpec>;

    // get device op instances
    const auto op_ptrs = ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
        DeviceOp>::GetInstances();

    std::cout << "found " << op_ptrs.size() << " instances" << std::endl;

    std::string best_op_name;
    int best_op_id        = -1;
    float best_ave_time   = 0;
    float best_tflops     = 0;
    float best_gb_per_sec = 0;

    // profile device op instances
    std::cout << "Run all instances and do timing" << std::endl;

    for(int i = 0; i < op_ptrs.size(); ++i)
    {
        auto& op_ptr      = op_ptrs[i];
        auto argument_ptr = op_ptr->MakeArgumentPointer(a_device_buf.GetDeviceBuffer(),
                                                        b0_device_buf.GetDeviceBuffer(),
                                                        b1_device_buf.GetDeviceBuffer(),
                                                        c_device_buf.GetDeviceBuffer(),
                                                        {}, // p_acc0_biases
                                                        {}, // p_acc1_biases
                                                        a_gs_ms_ks_lengths,
                                                        a_gs_ms_ks_strides,
                                                        b0_gs_ns_ks_lengths,
                                                        b0_gs_ns_ks_strides,
                                                        b1_gs_os_ns_lengths,
                                                        b1_gs_os_ns_strides,
                                                        c_gs_ms_os_lengths,
                                                        c_gs_ms_os_strides,
                                                        {}, // acc0_biases_gs_ms_ns_lengths
                                                        {}, // acc0_biases_gs_ms_ns_strides
                                                        {}, // acc1_biases_gs_ms_os_lengths
                                                        {}, // acc1_biases_gs_ms_os_strides
                                                        AElementOp{},
                                                        B0ElementOp{},
                                                        Acc0ElementOp{1 / sqrtf(K)},
                                                        B1ElementOp{},
                                                        CElementOp{});

        auto invoker_ptr    = op_ptr->MakeInvokerPointer();
        std::string op_name = op_ptr->GetTypeString();

        if(op_ptr->IsSupportedArgument(argument_ptr.get()))
        {

            float ave_time = invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, true});

            std::size_t flop      = (size_t(M) * N * K * 2 + size_t(M) * N * O * 2) * G0 * G1;
            std::size_t num_btype = (sizeof(ADataType) * M * K + sizeof(B0DataType) * K * N +
                                     sizeof(B1DataType) * N * O + sizeof(CDataType) * M * O) *
                                    G0 * G1;

            float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

            float gb_per_sec = num_btype / 1.E6 / ave_time;

            std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec
                      << " GB/s, " << op_name << std::endl;

            if(tflops > best_tflops)
            {
                best_op_id      = i;
                best_op_name    = op_name;
                best_tflops     = tflops;
                best_ave_time   = ave_time;
                best_gb_per_sec = gb_per_sec;
            }
        }
        else
        {
            std::cout << op_name << " does not support this problem" << std::endl;
        }
    }

    std::cout << "Best Perf: " << best_ave_time << " ms, " << best_tflops << " TFlops, "
              << best_gb_per_sec << " GB/s, " << best_op_name << std::endl;

    // run the best instance
    {
        auto& op_ptr = op_ptrs[best_op_id];
        std::cout << "Run the best instance without timing: " << op_ptr->GetTypeString()
                  << std::endl;
        auto argument_ptr = op_ptr->MakeArgumentPointer(a_device_buf.GetDeviceBuffer(),
                                                        b0_device_buf.GetDeviceBuffer(),
                                                        b1_device_buf.GetDeviceBuffer(),
                                                        c_device_buf.GetDeviceBuffer(),
                                                        {}, // p_acc0_biases
                                                        {}, // p_acc1_biases
                                                        a_gs_ms_ks_lengths,
                                                        a_gs_ms_ks_strides,
                                                        b0_gs_ns_ks_lengths,
                                                        b0_gs_ns_ks_strides,
                                                        b1_gs_os_ns_lengths,
                                                        b1_gs_os_ns_strides,
                                                        c_gs_ms_os_lengths,
                                                        c_gs_ms_os_strides,
                                                        {}, // acc0_biases_gs_ms_ns_lengths
                                                        {}, // acc0_biases_gs_ms_ns_strides
                                                        {}, // acc1_biases_gs_ms_os_lengths
                                                        {}, // acc1_biases_gs_ms_os_strides
                                                        AElementOp{},
                                                        B0ElementOp{},
                                                        Acc0ElementOp{1 / sqrtf(K)},
                                                        B1ElementOp{},
                                                        CElementOp{});

        auto invoker_ptr = op_ptr->MakeInvokerPointer();

        if(op_ptr->IsSupportedArgument(argument_ptr.get()))
        {
            invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, false});
        }

        std::cout << "Done" << std::endl;
    }

    return 0;
}
