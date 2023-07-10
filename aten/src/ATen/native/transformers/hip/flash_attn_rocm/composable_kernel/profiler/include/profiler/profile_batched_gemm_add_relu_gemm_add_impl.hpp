// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <memory>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/library/tensor_operation_instance/gpu/batched_gemm_add_relu_gemm_add.hpp"

#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/literals.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_batched_gemm.hpp"

namespace ck {
namespace profiler {

template <typename A0Layout,
          typename B0Layout,
          typename D0sLayout,
          typename B1Layout,
          typename D1sLayout,
          typename E1Layout,
          typename A0DataType,
          typename B0DataType,
          typename D0sDataType,
          typename B1DataType,
          typename D1sDataType,
          typename E1DataType>
bool profile_batched_gemm_add_relu_gemm_add_impl(bool do_verification,
                                                 int init_method,
                                                 bool do_log,
                                                 bool time_kernel,
                                                 int M,
                                                 int N,
                                                 int K,
                                                 int O,
                                                 int BatchCount    = 1,
                                                 int StrideA0      = -1,
                                                 int StrideB0      = -1,
                                                 int StrideD0      = -1,
                                                 int StrideB1      = -1,
                                                 int StrideD1      = -1,
                                                 int StrideE1      = -1,
                                                 int BatchStrideA0 = -1,
                                                 int BatchStrideB0 = -1,
                                                 int BatchStrideD0 = -1,
                                                 int BatchStrideB1 = -1,
                                                 int BatchStrideD1 = -1,
                                                 int BatchStrideE1 = -1)

{
    using Row = tensor_layout::gemm::RowMajor;
    using Col = tensor_layout::gemm::ColumnMajor;

    using PassThrough = tensor_operation::element_wise::PassThrough;

    using A0ElementOp   = PassThrough;
    using B0ElementOp   = PassThrough;
    using CDE0ElementOp = ck::tensor_operation::element_wise::AddRelu;
    using B1ElementOp   = PassThrough;
    using CDE1ElementOp = ck::tensor_operation::element_wise::Add;

    using D0DataType = remove_cvref_t<tuple_element_t<0, D0sDataType>>;

    using D0Layout   = remove_cvref_t<tuple_element_t<0, D0sLayout>>;
    using D1DataType = remove_cvref_t<tuple_element_t<0, D1sDataType>>;
    using D1Layout   = remove_cvref_t<tuple_element_t<0, D1sLayout>>;

    // for reference
    using RefAcc0DataType = float;
    using RefAcc1DataType = float;

    bool pass = true;

    const int DefaultStrideA0 = ck::is_same_v<A0Layout, Row> ? K : M;
    const int DefaultStrideB0 = ck::is_same_v<B0Layout, Row> ? N : K;
    const int DefaultStrideD0 = ck::is_same_v<D0Layout, Row> ? N : M;
    const int DefaultStrideB1 = ck::is_same_v<B1Layout, Row> ? O : N;
    const int DefaultStrideD1 = ck::is_same_v<D1Layout, Row> ? O : M;
    const int DefaultStrideE1 = ck::is_same_v<E1Layout, Row> ? O : M;

    StrideA0 = (StrideA0 < 0) ? DefaultStrideA0 : StrideA0;
    StrideB0 = (StrideB0 < 0) ? DefaultStrideB0 : StrideB0;
    StrideD0 = (StrideD0 < 0) ? DefaultStrideD0 : StrideD0;
    StrideB1 = (StrideB1 < 0) ? DefaultStrideB1 : StrideB1;
    StrideD1 = (StrideD1 < 0) ? DefaultStrideD1 : StrideD1;
    StrideE1 = (StrideE1 < 0) ? DefaultStrideE1 : StrideE1;

    const int DefaultBatchStrideA0 = (ck::is_same_v<A0Layout, Col> ? K : M) * StrideA0;
    const int DefaultBatchStrideB0 = (ck::is_same_v<B0Layout, Col> ? N : K) * StrideB0;
    const int DefaultBatchStrideD0 = (ck::is_same_v<D0Layout, Col> ? N : M) * StrideD0;
    const int DefaultBatchStrideB1 = (ck::is_same_v<B1Layout, Col> ? O : N) * StrideB1;
    const int DefaultBatchStrideD1 = (ck::is_same_v<D1Layout, Col> ? O : M) * StrideD1;
    const int DefaultBatchStrideE1 = (ck::is_same_v<E1Layout, Col> ? O : M) * StrideE1;

    BatchStrideA0 = BatchStrideA0 < 0 ? DefaultBatchStrideA0 : BatchStrideA0;
    BatchStrideB0 = BatchStrideB0 < 0 ? DefaultBatchStrideB0 : BatchStrideB0;
    BatchStrideD0 = BatchStrideD0 < 0 ? DefaultBatchStrideD0 : BatchStrideD0;
    BatchStrideB1 = BatchStrideB1 < 0 ? DefaultBatchStrideB1 : BatchStrideB1;
    BatchStrideD1 = BatchStrideD1 < 0 ? DefaultBatchStrideD1 : BatchStrideD1;
    BatchStrideE1 = BatchStrideE1 < 0 ? DefaultBatchStrideE1 : BatchStrideE1;

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

    // E_m_o = A_m_k * B0_k_n * B1_n_o
    Tensor<A0DataType> a0_g_m_k(
        f_host_tensor_descriptor(BatchCount, M, K, StrideA0, BatchStrideA0, A0Layout{}));
    Tensor<B0DataType> b0_g_k_n(
        f_host_tensor_descriptor(BatchCount, K, N, StrideB0, BatchStrideB0, B0Layout{}));
    Tensor<D0DataType> d0_g_m_n(
        f_host_tensor_descriptor(BatchCount, M, N, StrideD0, BatchStrideD0, D0Layout{}));
    Tensor<B1DataType> b1_g_n_o(
        f_host_tensor_descriptor(BatchCount, N, O, StrideB1, BatchStrideB1, B1Layout{}));
    Tensor<D1DataType> d1_g_m_o(
        f_host_tensor_descriptor(BatchCount, M, O, StrideD1, BatchStrideD1, D1Layout{}));
    Tensor<E1DataType> e1_g_m_o_host_result(
        f_host_tensor_descriptor(BatchCount, M, O, StrideE1, BatchStrideE1, E1Layout{}));
    Tensor<E1DataType> e1_g_m_o_device_result(
        f_host_tensor_descriptor(BatchCount, M, O, StrideE1, BatchStrideE1, E1Layout{}));

    // Host verification: Output of Gemm0 is input A of Gemm1
    Tensor<RefAcc0DataType> c0_g_m_n(f_host_tensor_descriptor(BatchCount, M, N, N, M * N, Row{}));
    Tensor<RefAcc0DataType> e0_g_m_n(f_host_tensor_descriptor(BatchCount, M, N, N, M * N, Row{}));
    Tensor<RefAcc1DataType> c1_g_m_o(f_host_tensor_descriptor(BatchCount, M, O, O, M * O, Row{}));

    std::cout << "a0_g_m_k: " << a0_g_m_k.mDesc << std::endl;
    std::cout << "b0_g_k_n: " << b0_g_k_n.mDesc << std::endl;
    std::cout << "d0_g_m_n: " << d0_g_m_n.mDesc << std::endl;
    std::cout << "b1_g_n_o: " << b1_g_n_o.mDesc << std::endl;
    std::cout << "d1_g_m_o: " << d1_g_m_o.mDesc << std::endl;
    std::cout << "e1_g_m_o: " << e1_g_m_o_host_result.mDesc << std::endl;

    switch(init_method)
    {
    case 0: break;
    case 1:
        a0_g_m_k.GenerateTensorValue(GeneratorTensor_2<A0DataType>{-2, 3});
        b0_g_k_n.GenerateTensorValue(GeneratorTensor_2<B0DataType>{-2, 3});
        d0_g_m_n.GenerateTensorValue(GeneratorTensor_2<D0DataType>{-2, 3});
        b1_g_n_o.GenerateTensorValue(GeneratorTensor_2<B1DataType>{-2, 3});
        d1_g_m_o.GenerateTensorValue(GeneratorTensor_2<D1DataType>{-2, 3});
        break;
    default:
        a0_g_m_k.GenerateTensorValue(GeneratorTensor_3<A0DataType>{0.0, 1.0});
        b0_g_k_n.GenerateTensorValue(GeneratorTensor_3<B0DataType>{-0.5, 0.5});
        d0_g_m_n.GenerateTensorValue(GeneratorTensor_3<D0DataType>{0.0, 1.0});
        b1_g_n_o.GenerateTensorValue(GeneratorTensor_3<B1DataType>{-0.5, 0.5});
        d1_g_m_o.GenerateTensorValue(GeneratorTensor_3<D1DataType>{0.0, 1.0});
    }

    DeviceMem a0_g_m_k_device_buf(sizeof(A0DataType) * a0_g_m_k.mDesc.GetElementSize());
    DeviceMem b0_g_k_n_device_buf(sizeof(B0DataType) * b0_g_k_n.mDesc.GetElementSize());
    DeviceMem d0_g_m_n_device_buf(sizeof(D0DataType) * d0_g_m_n.mDesc.GetElementSpaceSize());
    DeviceMem b1_g_n_o_device_buf(sizeof(B1DataType) * b1_g_n_o.mDesc.GetElementSize());
    DeviceMem d1_g_m_o_device_buf(sizeof(D1DataType) * d1_g_m_o.mDesc.GetElementSpaceSize());
    DeviceMem e1_g_m_o_device_buf(sizeof(E1DataType) *
                                  e1_g_m_o_device_result.mDesc.GetElementSize());

    a0_g_m_k_device_buf.ToDevice(a0_g_m_k.mData.data());
    b0_g_k_n_device_buf.ToDevice(b0_g_k_n.mData.data());
    d0_g_m_n_device_buf.ToDevice(d0_g_m_n.mData.data());
    b1_g_n_o_device_buf.ToDevice(b1_g_n_o.mData.data());
    d1_g_m_o_device_buf.ToDevice(d1_g_m_o.mData.data());

    auto a0_element_op   = A0ElementOp{};
    auto b0_element_op   = B0ElementOp{};
    auto cde0_element_op = CDE0ElementOp{};
    auto b1_element_op   = B1ElementOp{};
    auto cde1_element_op = CDE1ElementOp{};

    using DeviceOp =
        tensor_operation::device::DeviceBatchedGemmMultipleDGemmMultipleD<A0Layout,
                                                                          B0Layout,
                                                                          D0sLayout,
                                                                          B1Layout,
                                                                          D1sLayout,
                                                                          E1Layout,
                                                                          A0DataType,
                                                                          B0DataType,
                                                                          D0sDataType,
                                                                          B1DataType,
                                                                          D1sDataType,
                                                                          E1DataType,
                                                                          A0ElementOp,
                                                                          B0ElementOp,
                                                                          CDE0ElementOp,
                                                                          B1ElementOp,
                                                                          CDE1ElementOp>;

    // get device op instances
    const auto op_ptrs = tensor_operation::device::instance::DeviceOperationInstanceFactory<
        DeviceOp>::GetInstances();

    std::cout << "found " << op_ptrs.size() << " instances" << std::endl;

    if(do_verification)
    {
        // Ref Gemm0
        using ReferenceGemm0Instance = tensor_operation::host::ReferenceBatchedGemm<A0DataType,
                                                                                    B0DataType,
                                                                                    RefAcc0DataType,
                                                                                    RefAcc0DataType,
                                                                                    A0ElementOp,
                                                                                    B0ElementOp,
                                                                                    PassThrough>;

        // Ref Gemm1
        using ReferenceGemm1Instance = tensor_operation::host::ReferenceBatchedGemm<RefAcc0DataType,
                                                                                    B1DataType,
                                                                                    RefAcc1DataType,
                                                                                    RefAcc1DataType,
                                                                                    PassThrough,
                                                                                    B1ElementOp,
                                                                                    PassThrough>;

        auto ref_gemm0          = ReferenceGemm0Instance{};
        auto ref_gemm0_invoker  = ref_gemm0.MakeInvoker();
        auto ref_gemm0_argument = ref_gemm0.MakeArgument(
            a0_g_m_k, b0_g_k_n, c0_g_m_n, a0_element_op, b0_element_op, PassThrough{});

        ref_gemm0_invoker.Run(ref_gemm0_argument);

        // cde0_elementwise
        e0_g_m_n.ForEach(
            [&](auto&, auto idx) { cde0_element_op(e0_g_m_n(idx), c0_g_m_n(idx), d0_g_m_n(idx)); });

        auto ref_gemm1          = ReferenceGemm1Instance{};
        auto ref_gemm1_invoker  = ref_gemm1.MakeInvoker();
        auto ref_gemm1_argument = ref_gemm1.MakeArgument(
            e0_g_m_n, b1_g_n_o, c1_g_m_o, PassThrough{}, b1_element_op, PassThrough{});

        ref_gemm1_invoker.Run(ref_gemm1_argument);

        // cde1_elementwise
        e1_g_m_o_host_result.ForEach([&](auto&, auto idx) {
            cde1_element_op(e1_g_m_o_host_result(idx), c1_g_m_o(idx), d1_g_m_o(idx));
        });
    }

    std::string best_op_name;
    float best_ave_time   = 0;
    float best_tflops     = 0;
    float best_gb_per_sec = 0;

    // profile device op instances
    for(auto& op_ptr : op_ptrs)
    {
        auto argument_ptr = op_ptr->MakeArgumentPointer(
            static_cast<A0DataType*>(a0_g_m_k_device_buf.GetDeviceBuffer()),
            static_cast<B0DataType*>(b0_g_k_n_device_buf.GetDeviceBuffer()),
            std::array<const void*, 1>{d0_g_m_n_device_buf.GetDeviceBuffer()},
            static_cast<B1DataType*>(b1_g_n_o_device_buf.GetDeviceBuffer()),
            std::array<const void*, 1>{d1_g_m_o_device_buf.GetDeviceBuffer()},
            static_cast<E1DataType*>(e1_g_m_o_device_buf.GetDeviceBuffer()),
            M,
            N,
            K,
            O,
            BatchCount,
            StrideA0,
            StrideB0,
            std::array<ck::index_t, 1>{StrideD0},
            StrideB1,
            std::array<ck::index_t, 1>{StrideD1},
            StrideE1,
            BatchStrideA0,
            BatchStrideB0,
            std::array<ck::index_t, 1>{BatchStrideD0},
            BatchStrideB1,
            std::array<ck::index_t, 1>{BatchStrideD1},
            BatchStrideE1,
            a0_element_op,
            b0_element_op,
            cde0_element_op,
            b1_element_op,
            cde1_element_op);

        auto invoker_ptr = op_ptr->MakeInvokerPointer();

        if(op_ptr->IsSupportedArgument(argument_ptr.get()))
        {
            std::string op_name = op_ptr->GetTypeString();

            float ave_time =
                invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, time_kernel});

            std::size_t flop = (size_t(M) * N * K * 2 + size_t(M) * N * O * 2) * BatchCount;
            std::size_t num_btype =
                (sizeof(A0DataType) * M * K + sizeof(B0DataType) * K * N + sizeof(D0DataType) * N +
                 sizeof(B1DataType) * N * O + sizeof(E1DataType) * M * O + sizeof(D1DataType) * O) *
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
                e1_g_m_o_device_buf.FromDevice(e1_g_m_o_device_result.mData.data());

                pass = pass & ck::utils::check_err(e1_g_m_o_device_result, e1_g_m_o_host_result);

                if(do_log)
                {
                    LogRangeAsType<float>(
                        std::cout << "e1_g_m_o_host_result : ", e1_g_m_o_host_result.mData, ",")
                        << std::endl;
                    LogRangeAsType<float>(
                        std::cout << "e1_g_m_o_device_result : ", e1_g_m_o_device_result.mData, ",")
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
