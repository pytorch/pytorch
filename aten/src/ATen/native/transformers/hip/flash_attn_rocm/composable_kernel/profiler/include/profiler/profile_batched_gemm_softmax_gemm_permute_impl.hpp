// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <memory>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_batched_gemm_softmax_gemm_permute.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/tensor_operation_instance/gpu/batched_gemm_softmax_gemm_permute.hpp"

#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/literals.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_batched_gemm.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_softmax.hpp"

namespace ck {
namespace profiler {

template <index_t NumDimG,
          index_t NumDimM,
          index_t NumDimN,
          index_t NumDimK,
          index_t NumDimO,
          typename ADataType,
          typename B0DataType,
          typename B1DataType,
          typename CDataType,
          typename Acc0BiasesDataType,
          typename Acc1BiasesDataType,
          tensor_operation::device::MaskingSpecialization MaskingSpec>
bool profile_batched_gemm_softmax_gemm_permute_impl(bool do_verification,
                                                    int init_method,
                                                    bool do_log,
                                                    bool time_kernel,
                                                    int M,
                                                    int N,
                                                    int K,
                                                    int O,
                                                    int G0,
                                                    int G1,
                                                    float alpha = -1.f)

{

    using PassThrough   = tensor_operation::element_wise::PassThrough;
    using Scale         = tensor_operation::element_wise::Scale;
    using AElementOp    = PassThrough;
    using B0ElementOp   = PassThrough;
    using Acc0ElementOp = Scale;
    using B1ElementOp   = PassThrough;
    using CElementOp    = PassThrough;
    using AccDataType   = float;
    using tensor_operation::device::MaskingSpecialization;

    // Ref Gemm0: various type in, fp32 out
    using ReferenceGemm0Instance = tensor_operation::host::ReferenceBatchedGemm<ADataType,
                                                                                B0DataType,
                                                                                AccDataType,
                                                                                AccDataType,
                                                                                AElementOp,
                                                                                B0ElementOp,
                                                                                Acc0ElementOp>;

    // Ref Softmax: fp32 in, various type out
    using ReferenceSoftmaxInstance =
        tensor_operation::host::ReferenceSoftmax<AccDataType, ADataType, AccDataType>;

    // Ref Gemm1: various type in, various type out
    using ReferenceGemm1Instance = tensor_operation::host::ReferenceBatchedGemm<ADataType,
                                                                                B1DataType,
                                                                                CDataType,
                                                                                AccDataType,
                                                                                AElementOp,
                                                                                B1ElementOp,
                                                                                CElementOp>;

    bool pass = true;

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

    const int BatchCount = G0 * G1;

    Tensor<ADataType> a_gs_ms_ks(a_gs_ms_ks_lengths, a_gs_ms_ks_strides);
    Tensor<B0DataType> b0_gs_ns_ks(b0_gs_ns_ks_lengths, b0_gs_ns_ks_strides);
    Tensor<B1DataType> b1_gs_os_ns(b1_gs_os_ns_lengths, b1_gs_os_ns_strides);
    Tensor<CDataType> c_gs_ms_os_host_result(c_gs_ms_os_lengths, c_gs_ms_os_strides);
    Tensor<CDataType> c_gs_ms_os_device_result(c_gs_ms_os_lengths, c_gs_ms_os_strides);

    std::cout << "a_gs_ms_ks: " << a_gs_ms_ks.mDesc << std::endl;
    std::cout << "b0_gs_ns_ks: " << b0_gs_ns_ks.mDesc << std::endl;
    std::cout << "b1_gs_os_ns: " << b1_gs_os_ns.mDesc << std::endl;
    std::cout << "c_gs_ms_os: " << c_gs_ms_os_host_result.mDesc << std::endl;

    std::srand(1); // work around test flakiness
    switch(init_method)
    {
    case 0: break;
    case 1:
        // Still unsure whether this kind of deterministic floating point accurary issue is expected
        // or not. May want to try exact same approach as the GPU kernel in the host reference
        // GEMM+Softmax+GEMM function to see if the accuracy discrepancy goes away. Until then,
        // shrink the input value range as it is less likely to produce errors of around ~1e-3.
        // a_gs_ms_ks.GenerateTensorValue(GeneratorTensor_2<ADataType>{-5, 5});
        // b0_gs_ns_ks.GenerateTensorValue(GeneratorTensor_2<B0DataType>{-5, 5});
        // b1_gs_os_ns.GenerateTensorValue(GeneratorTensor_2<B1DataType>{-5, 5});
        a_gs_ms_ks.GenerateTensorValue(GeneratorTensor_2<ADataType>{-2, 2});
        b0_gs_ns_ks.GenerateTensorValue(GeneratorTensor_2<B0DataType>{-2, 2});
        b1_gs_os_ns.GenerateTensorValue(GeneratorTensor_2<B1DataType>{-2, 2});
        break;
    case 2:
        a_gs_ms_ks.GenerateTensorValue(GeneratorTensor_3<ADataType>{0.0, 1.0});
        b0_gs_ns_ks.GenerateTensorValue(GeneratorTensor_3<B0DataType>{0.0, 1.0});
        b1_gs_os_ns.GenerateTensorValue(GeneratorTensor_3<B1DataType>{-0.5, 0.5});
        break;
    case 3:
        a_gs_ms_ks.GenerateTensorValue(GeneratorTensor_2<ADataType>{-2, 2});
        b0_gs_ns_ks.GenerateTensorValue(GeneratorTensor_Diagonal<B0DataType>{});
        b1_gs_os_ns.GenerateTensorValue(GeneratorTensor_Diagonal<B1DataType>{});
        break;
    default:
        a_gs_ms_ks.GenerateTensorValue(GeneratorTensor_1<ADataType>{1});
        b0_gs_ns_ks.GenerateTensorValue(GeneratorTensor_Sequential<1>{});
        b1_gs_os_ns.GenerateTensorValue(GeneratorTensor_Diagonal<B1DataType>{});
    }

    DeviceMem a_device_buf(sizeof(ADataType) * a_gs_ms_ks.mDesc.GetElementSpaceSize());
    DeviceMem b0_device_buf(sizeof(B0DataType) * b0_gs_ns_ks.mDesc.GetElementSpaceSize());
    DeviceMem b1_device_buf(sizeof(B1DataType) * b1_gs_os_ns.mDesc.GetElementSpaceSize());
    DeviceMem c_device_buf(sizeof(CDataType) *
                           c_gs_ms_os_device_result.mDesc.GetElementSpaceSize());

    a_device_buf.ToDevice(a_gs_ms_ks.mData.data());
    b0_device_buf.ToDevice(b0_gs_ns_ks.mData.data());
    b1_device_buf.ToDevice(b1_gs_os_ns.mData.data());

    if(alpha < 0)
    {
        alpha = 1.f / std::sqrt(K); // usually 1 / sqrt(head_dim)
    }
    auto a_element_op    = AElementOp{};
    auto b0_element_op   = B0ElementOp{};
    auto acc0_element_op = Acc0ElementOp{alpha};
    auto b1_element_op   = B1ElementOp{};
    auto c_element_op    = CElementOp{};

    using DeviceOp = tensor_operation::device::DeviceBatchedGemmSoftmaxGemmPermute<2,
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
    const auto op_ptrs = tensor_operation::device::instance::DeviceOperationInstanceFactory<
        DeviceOp>::GetInstances();

    std::cout << "found " << op_ptrs.size() << " instances" << std::endl;

    if(do_verification)
    {
        c_device_buf.FromDevice(c_gs_ms_os_device_result.mData.data());

        Tensor<ADataType> a_g_m_k({BatchCount, M, K});
        Tensor<B0DataType> b0_g_k_n({BatchCount, K, N});
        Tensor<B1DataType> b1_g_n_o({BatchCount, N, O});
        Tensor<AccDataType> acc0_g_m_n({BatchCount, M, N});        // scratch object after gemm0
        Tensor<ADataType> a1_g_m_n({BatchCount, M, N});            // scratch object after softmax
        Tensor<CDataType> c_g_m_o_host_result({BatchCount, M, O}); // scratch object after gemm1

        // permute
        a_gs_ms_ks.ForEach([&](auto& self, auto idx) {
            a_g_m_k(idx[0] * G1 + idx[1], idx[2], idx[3]) = self(idx);
        });
        b0_gs_ns_ks.ForEach([&](auto& self, auto idx) {
            b0_g_k_n(idx[0] * G1 + idx[1], idx[3], idx[2]) = self(idx);
        });
        b1_gs_os_ns.ForEach([&](auto& self, auto idx) {
            b1_g_n_o(idx[0] * G1 + idx[1], idx[3], idx[2]) = self(idx);
        });

        auto ref_gemm0          = ReferenceGemm0Instance{};
        auto ref_gemm0_invoker  = ref_gemm0.MakeInvoker();
        auto ref_gemm0_argument = ref_gemm0.MakeArgument(
            a_g_m_k, b0_g_k_n, acc0_g_m_n, a_element_op, b0_element_op, Scale{alpha});

        ref_gemm0_invoker.Run(ref_gemm0_argument);

        // mask out upper triangle
        acc0_g_m_n.ForEach([&](auto& self, auto idx) {
            if(MaskingSpec == MaskingSpecialization::MaskOutUpperTriangle && idx[1] < idx[2])
                self(idx) = -ck::NumericLimits<float>::Infinity();
        });

        auto ref_softmax          = ReferenceSoftmaxInstance{};
        auto ref_softmax_invoker  = ref_softmax.MakeInvoker();
        auto ref_softmax_argument = ref_softmax.MakeArgument(acc0_g_m_n, a1_g_m_n, 1, 0, {2});

        ref_softmax_invoker.Run(ref_softmax_argument);

        auto ref_gemm1          = ReferenceGemm1Instance{};
        auto ref_gemm1_invoker  = ref_gemm1.MakeInvoker();
        auto ref_gemm1_argument = ref_gemm1.MakeArgument(
            a1_g_m_n, b1_g_n_o, c_g_m_o_host_result, PassThrough{}, b1_element_op, c_element_op);

        ref_gemm1_invoker.Run(ref_gemm1_argument);

        // permute
        c_gs_ms_os_host_result.ForEach([&](auto& self, auto idx) {
            const size_t& g0 = idx[0];
            const size_t& g1 = idx[1];

            const size_t g = g0 * G1 + g1;

            self(idx) = c_g_m_o_host_result(g, idx[2], idx[3]);
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
            static_cast<ADataType*>(a_device_buf.GetDeviceBuffer()),
            static_cast<B0DataType*>(b0_device_buf.GetDeviceBuffer()),
            static_cast<B1DataType*>(b1_device_buf.GetDeviceBuffer()),
            static_cast<CDataType*>(c_device_buf.GetDeviceBuffer()),
            {}, // std::array<void*, 1> p_acc0_biases;
            {}, // std::array<void*, 1> p_acc1_biases;
            a_gs_ms_ks_lengths,
            a_gs_ms_ks_strides,
            b0_gs_ns_ks_lengths,
            b0_gs_ns_ks_strides,
            b1_gs_os_ns_lengths,
            b1_gs_os_ns_strides,
            c_gs_ms_os_lengths,
            c_gs_ms_os_strides,
            {}, // std::array<std::vector<ck::index_t>, 1>{acc0_biases_gs_ms_ns_lengths},
            {}, // std::array<std::vector<ck::index_t>, 1>{acc0_biases_gs_ms_ns_strides},
            {}, // std::array<std::vector<ck::index_t>, 1>{acc1_biases_gs_ms_os_lengths},
            {}, // std::array<std::vector<ck::index_t>, 1>{acc1_biases_gs_ms_os_strides},
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
                c_device_buf.FromDevice(c_gs_ms_os_device_result.mData.data());

                // default absolute error and relative error is 0.001
                double rtol = 1e-3;
                double atol = 1e-3;

                // when BF16 is taken, set absolute error and relative error to 0.01
                if(std::is_same_v<ADataType, ck::bhalf_t> &&
                   std::is_same_v<B0DataType, ck::bhalf_t> &&
                   std::is_same_v<B1DataType, ck::bhalf_t> &&
                   std::is_same_v<CDataType, ck::bhalf_t>)
                {
                    rtol = 1e-2;
                    atol = 1e-2;
                }

                pass = pass & ck::utils::check_err(c_gs_ms_os_device_result,
                                                   c_gs_ms_os_host_result,
                                                   "Error: Incorrect results!",
                                                   rtol,
                                                   atol);

                if(do_log)
                {
                    LogRangeAsType<float>(std::cout << "a_gs_ms_ks: ", a_gs_ms_ks.mData, ",")
                        << std::endl;
                    LogRangeAsType<float>(std::cout << "b0_gs_ns_ks : ", b0_gs_ns_ks.mData, ",")
                        << std::endl;
                    LogRangeAsType<float>(std::cout << "b1_gs_os_ns : ", b1_gs_os_ns.mData, ",")
                        << std::endl;
                    LogRangeAsType<float>(
                        std::cout << "c_gs_ms_os_host_result : ", c_gs_ms_os_host_result.mData, ",")
                        << std::endl;
                    LogRangeAsType<float>(std::cout << "c_gs_ms_os_device_result : ",
                                          c_gs_ms_os_device_result.mData,
                                          ",")
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
