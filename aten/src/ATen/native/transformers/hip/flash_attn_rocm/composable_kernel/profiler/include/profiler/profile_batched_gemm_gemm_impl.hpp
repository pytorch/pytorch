// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <memory>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_batched_gemm_gemm.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/tensor_operation_instance/gpu/batched_gemm_gemm.hpp"

#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/literals.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_batched_gemm.hpp"

namespace ck {
namespace profiler {

template <typename ADataType,
          typename B0DataType,
          typename B1DataType,
          typename CDataType,
          typename ALayout,
          typename B0Layout,
          typename B1Layout,
          typename CLayout>
bool profile_batched_gemm_gemm_impl(bool do_verification,
                                    int init_method,
                                    bool do_log,
                                    bool time_kernel,
                                    int M,
                                    int N,
                                    int K,
                                    int O,
                                    int BatchCount    = 1,
                                    int StrideA       = -1,
                                    int StrideB0      = -1,
                                    int StrideB1      = -1,
                                    int StrideC       = -1,
                                    int BatchStrideA  = -1,
                                    int BatchStrideB0 = -1,
                                    int BatchStrideB1 = -1,
                                    int BatchStrideC  = -1)

{

    using Row           = tensor_layout::gemm::RowMajor;
    using Col           = tensor_layout::gemm::ColumnMajor;
    using PassThrough   = tensor_operation::element_wise::PassThrough;
    using AElementOp    = PassThrough;
    using B0ElementOp   = PassThrough;
    using B1ElementOp   = PassThrough;
    using Acc0ElementOp = PassThrough;
    using CElementOp    = PassThrough;
    using AccDataType   = float;

    // Ref Gemm0
    using ReferenceGemm0Instance = tensor_operation::host::ReferenceBatchedGemm<ADataType,
                                                                                B0DataType,
                                                                                ADataType,
                                                                                AccDataType,
                                                                                AElementOp,
                                                                                B0ElementOp,
                                                                                CElementOp>;

    // Ref Gemm
    using ReferenceGemm1Instance = tensor_operation::host::ReferenceBatchedGemm<ADataType,
                                                                                B1DataType,
                                                                                CDataType,
                                                                                AccDataType,
                                                                                AElementOp,
                                                                                B1ElementOp,
                                                                                CElementOp>;

    bool pass = true;

    const int DefaultStrideA  = ck::is_same_v<ALayout, Row> ? K : M;
    const int DefaultStrideB0 = ck::is_same_v<B0Layout, Row> ? N : K;
    const int DefaultStrideB1 = ck::is_same_v<B1Layout, Row> ? O : N;
    const int DefaultStrideC  = ck::is_same_v<CLayout, Row> ? O : M;

    StrideA  = (StrideA < 0) ? DefaultStrideA : StrideA;
    StrideB0 = (StrideB0 < 0) ? DefaultStrideB0 : StrideB0;
    StrideB1 = (StrideB1 < 0) ? DefaultStrideB1 : StrideB1;
    StrideC  = (StrideC < 0) ? DefaultStrideC : StrideC;

    const int DefaultBatchStrideA  = (ck::is_same_v<ALayout, Col> ? K : M) * StrideA;
    const int DefaultBatchStrideB0 = (ck::is_same_v<B0Layout, Col> ? N : K) * StrideB0;
    const int DefaultBatchStrideB1 = (ck::is_same_v<B1Layout, Col> ? O : N) * StrideB1;
    const int DefaultBatchStrideC  = (ck::is_same_v<CLayout, Col> ? O : M) * StrideC;

    BatchStrideA  = BatchStrideA < 0 ? DefaultBatchStrideA : BatchStrideA;
    BatchStrideB0 = BatchStrideB0 < 0 ? DefaultBatchStrideB0 : BatchStrideB0;
    BatchStrideB1 = BatchStrideB1 < 0 ? DefaultBatchStrideB1 : BatchStrideB1;
    BatchStrideC  = BatchStrideC < 0 ? DefaultBatchStrideC : BatchStrideC;

    auto f_host_tensor_descriptor = [](std::size_t batch_count,
                                       std::size_t row,
                                       std::size_t col,
                                       std::size_t stride,
                                       std::size_t batch_stride,
                                       auto layout) {
        using namespace ck::literals;

        if(std::is_same<decltype(layout), Row>::value)
        {
            return HostTensorDescriptor({batch_count, row, col}, {batch_stride, stride, 1_uz});
        }
        else
        {
            return HostTensorDescriptor({batch_count, row, col}, {batch_stride, 1_uz, stride});
        }
    };

    // C_m_o = A_m_k * B0_k_n * B1_n_o
    Tensor<ADataType> a_g_m_k(
        f_host_tensor_descriptor(BatchCount, M, K, StrideA, BatchStrideA, ALayout{}));
    Tensor<B0DataType> b0_g_k_n(
        f_host_tensor_descriptor(BatchCount, K, N, StrideB0, BatchStrideB0, B0Layout{}));
    Tensor<B1DataType> b1_g_n_o(
        f_host_tensor_descriptor(BatchCount, N, O, StrideB1, BatchStrideB1, B1Layout{}));
    Tensor<CDataType> c_g_m_o_host_result(
        f_host_tensor_descriptor(BatchCount, M, O, StrideC, BatchStrideC, CLayout{}));
    Tensor<CDataType> c_g_m_o_device_result(
        f_host_tensor_descriptor(BatchCount, M, O, StrideC, BatchStrideC, CLayout{}));
    // Host verification: Output of Gemm0 is input A of Gemm1
    Tensor<ADataType> acc0_g_m_n(f_host_tensor_descriptor(BatchCount, M, N, N, M * N, Row{}));

    std::cout << "a_g_m_k: " << a_g_m_k.mDesc << std::endl;
    std::cout << "b0_g_k_n: " << b0_g_k_n.mDesc << std::endl;
    std::cout << "b1_g_n_o: " << b1_g_n_o.mDesc << std::endl;
    std::cout << "c_g_m_o: " << c_g_m_o_host_result.mDesc << std::endl;

    switch(init_method)
    {
    case 0: break;
    case 1:
        a_g_m_k.GenerateTensorValue(GeneratorTensor_2<ADataType>{-2, 3});
        b0_g_k_n.GenerateTensorValue(GeneratorTensor_2<B0DataType>{-2, 3});
        b1_g_n_o.GenerateTensorValue(GeneratorTensor_2<B1DataType>{-2, 3});
        break;
    case 2:
        a_g_m_k.GenerateTensorValue(GeneratorTensor_3<ADataType>{0.0, 1.0});
        b0_g_k_n.GenerateTensorValue(GeneratorTensor_3<B0DataType>{0.0, 1.0});
        b1_g_n_o.GenerateTensorValue(GeneratorTensor_3<B1DataType>{-0.5, 0.5});
        break;
    case 3:
        a_g_m_k.GenerateTensorValue(GeneratorTensor_2<ADataType>{-2, 2});
        b0_g_k_n.GenerateTensorValue(GeneratorTensor_Diagonal<B0DataType>{});
        b1_g_n_o.GenerateTensorValue(GeneratorTensor_Diagonal<B1DataType>{});
        break;
    default:
        a_g_m_k.GenerateTensorValue(GeneratorTensor_1<ADataType>{1});
        b0_g_k_n.GenerateTensorValue(GeneratorTensor_Sequential<1>{});
        b1_g_n_o.GenerateTensorValue(GeneratorTensor_Diagonal<B1DataType>{});
    }

    DeviceMem a_g_m_k_device_buf(sizeof(ADataType) * a_g_m_k.mDesc.GetElementSize());
    DeviceMem b0_g_k_n_device_buf(sizeof(B0DataType) * b0_g_k_n.mDesc.GetElementSize());
    DeviceMem b1_g_n_o_device_buf(sizeof(B1DataType) * b1_g_n_o.mDesc.GetElementSize());
    DeviceMem c_g_m_o_device_buf(sizeof(CDataType) * c_g_m_o_device_result.mDesc.GetElementSize());

    a_g_m_k_device_buf.ToDevice(a_g_m_k.mData.data());
    b0_g_k_n_device_buf.ToDevice(b0_g_k_n.mData.data());
    b1_g_n_o_device_buf.ToDevice(b1_g_n_o.mData.data());

    auto a_element_op    = AElementOp{};
    auto b0_element_op   = B0ElementOp{};
    auto acc0_element_op = Acc0ElementOp{};
    auto b1_element_op   = B1ElementOp{};
    auto c_element_op    = CElementOp{};

    using DeviceOp = tensor_operation::device::DeviceBatchedGemmGemm<ALayout,
                                                                     B0Layout,
                                                                     B1Layout,
                                                                     CLayout,
                                                                     ADataType,
                                                                     B0DataType,
                                                                     B1DataType,
                                                                     CDataType,
                                                                     AElementOp,
                                                                     B0ElementOp,
                                                                     Acc0ElementOp,
                                                                     B1ElementOp,
                                                                     CElementOp>;

    // get device op instances
    const auto op_ptrs = tensor_operation::device::instance::DeviceOperationInstanceFactory<
        DeviceOp>::GetInstances();

    std::cout << "found " << op_ptrs.size() << " instances" << std::endl;

    // early fail when no instances are found
    if(op_ptrs.size() == 0)
    {
        return false;
    }

    if(do_verification)
    {
        auto ref_gemm0          = ReferenceGemm0Instance{};
        auto ref_gemm0_invoker  = ref_gemm0.MakeInvoker();
        auto ref_gemm0_argument = ref_gemm0.MakeArgument(
            a_g_m_k, b0_g_k_n, acc0_g_m_n, a_element_op, b0_element_op, PassThrough{});

        ref_gemm0_invoker.Run(ref_gemm0_argument);

        auto ref_gemm1          = ReferenceGemm1Instance{};
        auto ref_gemm1_invoker  = ref_gemm1.MakeInvoker();
        auto ref_gemm1_argument = ref_gemm1.MakeArgument(
            acc0_g_m_n, b1_g_n_o, c_g_m_o_host_result, PassThrough{}, b1_element_op, c_element_op);

        ref_gemm1_invoker.Run(ref_gemm1_argument);
    }

    std::string best_op_name;
    float best_ave_time   = 0;
    float best_tflops     = 0;
    float best_gb_per_sec = 0;

    // profile device op instances
    for(auto& op_ptr : op_ptrs)
    {
        auto argument_ptr = op_ptr->MakeArgumentPointer(
            static_cast<ADataType*>(a_g_m_k_device_buf.GetDeviceBuffer()),
            static_cast<B0DataType*>(b0_g_k_n_device_buf.GetDeviceBuffer()),
            static_cast<B1DataType*>(b1_g_n_o_device_buf.GetDeviceBuffer()),
            static_cast<CDataType*>(c_g_m_o_device_buf.GetDeviceBuffer()),
            M,
            N,
            K,
            O,
            BatchCount,
            StrideA,
            StrideB0,
            StrideB1,
            StrideC,
            BatchStrideA,
            BatchStrideB0,
            BatchStrideB1,
            BatchStrideC,
            a_element_op,
            b0_element_op,
            acc0_element_op,
            b1_element_op,
            c_element_op);

        auto invoker_ptr = op_ptr->MakeInvokerPointer();

        if(op_ptr->IsSupportedArgument(argument_ptr.get()))
        {
            std::string op_name = op_ptr->GetTypeString();

            float ave_time =
                invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, time_kernel});

            std::size_t flop      = (size_t(M) * N * K * 2 + size_t(M) * N * O * 2) * BatchCount;
            std::size_t num_btype = (sizeof(ADataType) * M * K + sizeof(B0DataType) * K * N +
                                     sizeof(B1DataType) * N * O + sizeof(CDataType) * M * O) *
                                    BatchCount;

            float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

            float gb_per_sec = num_btype / 1.E6 / ave_time;

            std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec
                      << " GB/s, " << op_name << std::endl;

            if(tflops > best_tflops)
            {
                best_op_name    = op_name;
                best_tflops     = tflops;
                best_ave_time   = ave_time;
                best_gb_per_sec = gb_per_sec;
            }

            if(do_verification)
            {
                c_g_m_o_device_buf.FromDevice(c_g_m_o_device_result.mData.data());

                pass = pass & ck::utils::check_err(c_g_m_o_device_result, c_g_m_o_host_result);

                if(do_log)
                {
                    LogRangeAsType<float>(std::cout << "a_g_m_k: ", a_g_m_k.mData, ",")
                        << std::endl;
                    LogRangeAsType<float>(std::cout << "b0_g_k_n : ", b0_g_k_n.mData, ",")
                        << std::endl;
                    LogRangeAsType<float>(std::cout << "b1_g_n_o : ", b1_g_n_o.mData, ",")
                        << std::endl;
                    LogRangeAsType<float>(
                        std::cout << "c_g_m_o_host_result : ", c_g_m_o_host_result.mData, ",")
                        << std::endl;
                    LogRangeAsType<float>(
                        std::cout << "c_g_m_o_device_result : ", c_g_m_o_device_result.mData, ",")
                        << std::endl;
                }
            }
        }
        else
        {
            std::cout << op_ptr->GetTypeString() << " does not support this problem" << std::endl;
        }
    }

    std::cout << "Best Perf: " << best_ave_time << " ms, " << best_tflops << " TFlops, "
              << best_gb_per_sec << " GB/s, " << best_op_name << std::endl;

    return pass;
}

} // namespace profiler
} // namespace ck
