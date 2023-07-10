// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <iomanip>
#include <vector>
#include <iostream>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_gemm_multiple_d.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/tensor_operation_instance/gpu/gemm_add_fastgelu.hpp"

using F16 = ck::half_t;
using F32 = float;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;
using AddFastGelu = ck::tensor_operation::element_wise::AddFastGelu;

using AElementOp   = PassThrough;
using BElementOp   = PassThrough;
using CDEElementOp = AddFastGelu;

using ADataType  = F16;
using BDataType  = F16;
using D0DataType = F16;
using EDataType  = F16;

using ALayout  = Row;
using BLayout  = Col;
using D0Layout = Row;
using ELayout  = Row;

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
    // GEMM shape
    ck::index_t M = 3840;
    ck::index_t N = 4096;
    ck::index_t K = 4096;

    ck::index_t StrideA  = 4096;
    ck::index_t StrideB  = 4096;
    ck::index_t StrideD0 = 0;
    ck::index_t StrideE  = 4096;

    if(argc == 1)
    {
        // use default case
    }
    else if(argc == 8)
    {
        M = std::stoi(argv[1]);
        N = std::stoi(argv[2]);
        K = std::stoi(argv[3]);

        StrideA  = std::stoi(argv[4]);
        StrideB  = std::stoi(argv[5]);
        StrideD0 = std::stoi(argv[6]);
        StrideE  = std::stoi(argv[8]);
    }
    else
    {
        printf("arg1 to 7: M, N, K, StrideA, StrideB, StrideD0, StrideE\n");
        exit(0);
    }

    auto f_matrix_space_size =
        [](std::size_t nRow, std::size_t nCol, std::size_t stride, auto layout) {
            using Layout = decltype(layout);

            if(std::is_same<Layout, ck::tensor_layout::gemm::RowMajor>::value)
            {
                return (nRow - 1) * stride + nCol;
            }
            else
            {
                return (nCol - 1) * stride + nRow;
            }
        };

    SimpleDeviceMem a_device_buf(sizeof(ADataType) * f_matrix_space_size(M, K, StrideA, ALayout{}));
    SimpleDeviceMem b_device_buf(sizeof(BDataType) * f_matrix_space_size(K, N, StrideB, BLayout{}));
    SimpleDeviceMem d0_m_n_device_buf(sizeof(D0DataType) *
                                      f_matrix_space_size(M, N, StrideD0, D0Layout{}));
    SimpleDeviceMem e_device_buf(sizeof(EDataType) * f_matrix_space_size(M, N, StrideE, ELayout{}));

    using DeviceOp = ck::tensor_operation::device::DeviceGemmMultipleD<
        ALayout,
        BLayout,
        ck::Tuple<D0Layout>,
        ELayout,
        ADataType,
        BDataType,
        ck::Tuple<D0DataType>,
        EDataType,
        ck::tensor_operation::element_wise::PassThrough,
        ck::tensor_operation::element_wise::PassThrough,
        ck::tensor_operation::element_wise::AddFastGelu>;

    // get device op instances
    const auto op_ptrs = ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
        DeviceOp>::GetInstances();

    std::cout << "found " << op_ptrs.size() << " instances" << std::endl;

    const auto a_element_op   = AElementOp{};
    const auto b_element_op   = BElementOp{};
    const auto cde_element_op = CDEElementOp{};

    std::string best_op_name;
    bool found            = false;
    int best_op_id        = -1;
    float best_ave_time   = 0;
    float best_tflops     = 0;
    float best_gb_per_sec = 0;

    // profile device operation instances
    std::cout << "Run all instances and do timing" << std::endl;

    for(int i = 0; i < op_ptrs.size(); ++i)
    {
        auto& op_ptr = op_ptrs[i];

        auto argument_ptr = op_ptr->MakeArgumentPointer(
            a_device_buf.GetDeviceBuffer(),
            b_device_buf.GetDeviceBuffer(),
            std::array<const void*, 1>{d0_m_n_device_buf.GetDeviceBuffer()},
            e_device_buf.GetDeviceBuffer(),
            M,
            N,
            K,
            StrideA,
            StrideB,
            std::array<ck::index_t, 1>{StrideD0},
            StrideE,
            a_element_op,
            b_element_op,
            cde_element_op);

        auto invoker_ptr = op_ptr->MakeInvokerPointer();

        std::string op_name = op_ptr->GetTypeString();

        if(op_ptr->IsSupportedArgument(argument_ptr.get()))
        {
            float ave_time = invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, true});

            std::size_t flop = std::size_t(2) * M * N * K;

            std::size_t num_btype =
                sizeof(ADataType) * M * K + sizeof(BDataType) * K * N + sizeof(EDataType) * M * N;

            float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

            float gb_per_sec = num_btype / 1.E6 / ave_time;

            std::cout << "Perf: " << std::setw(10) << ave_time << " ms, " << tflops << " TFlops, "
                      << gb_per_sec << " GB/s, " << op_name << std::endl;

            if(tflops > best_tflops)
            {
                found           = true;
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

    // run the best intance
    {
        auto& op_ptr = op_ptrs[best_op_id];

        std::cout << "Run the best instance without timing: " << op_ptr->GetTypeString()
                  << std::endl;

        auto argument_ptr = op_ptr->MakeArgumentPointer(
            a_device_buf.GetDeviceBuffer(),
            b_device_buf.GetDeviceBuffer(),
            std::array<const void*, 1>{d0_m_n_device_buf.GetDeviceBuffer()},
            e_device_buf.GetDeviceBuffer(),
            M,
            N,
            K,
            StrideA,
            StrideB,
            std::array<ck::index_t, 1>{StrideD0},
            StrideE,
            a_element_op,
            b_element_op,
            cde_element_op);

        auto invoker_ptr = op_ptr->MakeInvokerPointer();

        if(op_ptr->IsSupportedArgument(argument_ptr.get()))
        {
            invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, false});
        }

        std::cout << "Done" << std::endl;
    }

    return 0;
}
