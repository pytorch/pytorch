// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/ck.hpp"
#include "ck/utility/reduction_operator.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_gemm_reduce.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/convolution_parameter.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/literals.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_batched_gemm.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

using F32                 = float;
using F16                 = ck::half_t;
using ReducePtrsGlobal    = ck::Tuple<F32*, F32*>;
using Identity            = ck::tensor_operation::element_wise::PassThrough;
using Square              = ck::tensor_operation::element_wise::UnarySquare;
using ReduceInElementOps  = ck::Tuple<Identity, Square>;
using ReduceOutElementOps = ck::Tuple<Identity, Identity>;

using DeviceGemmReduceNoOpPtr =
    ck::tensor_operation::device::DeviceGemmReducePtr<0, ReducePtrsGlobal::Size()>;

void add_device_batched_gemm_reduce_xdl_cshuffle_f16_f16_f16_f32_f32_gmk_gkn_gmn_instances(
    std::vector<DeviceGemmReduceNoOpPtr>&);

void add_device_batched_gemm_reduce_xdl_cshuffle_f16_f16_f16_f32_f32_gmk_gnk_gmn_instances(
    std::vector<DeviceGemmReduceNoOpPtr>&);

void add_device_batched_gemm_reduce_xdl_cshuffle_f16_f16_f16_f32_f32_gkm_gkn_gmn_instances(
    std::vector<DeviceGemmReduceNoOpPtr>&);

void add_device_batched_gemm_reduce_xdl_cshuffle_f16_f16_f16_f32_f32_gkm_gnk_gmn_instances(
    std::vector<DeviceGemmReduceNoOpPtr>&);

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck

namespace ck {
namespace profiler {

template <typename ADataType,
          typename BDataType,
          typename CDataType,
          typename ReduceDataType,
          typename ALayout,
          typename BLayout,
          typename CLayout>
bool profile_batched_gemm_reduce_impl(int do_verification,
                                      int init_method,
                                      bool do_log,
                                      bool time_kernel,
                                      int M,
                                      int N,
                                      int K,
                                      int StrideA,
                                      int StrideB,
                                      int StrideC,
                                      int BatchCount)
{
    bool pass = true;

    auto f_host_tensor_descriptor = [](std::size_t batch_count,
                                       std::size_t row,
                                       std::size_t col,
                                       std::size_t stride,
                                       auto layout) {
        using namespace ck::literals;

        if(std::is_same<decltype(layout), ck::tensor_layout::gemm::RowMajor>::value)
        {
            return HostTensorDescriptor({batch_count, row, col}, {row * stride, stride, 1_uz});
        }
        else
        {
            return HostTensorDescriptor({batch_count, row, col}, {col * stride, 1_uz, stride});
        }
    };

    Tensor<ADataType> a_g_m_k(f_host_tensor_descriptor(BatchCount, M, K, StrideA, ALayout{}));
    Tensor<BDataType> b_g_k_n(f_host_tensor_descriptor(BatchCount, K, N, StrideB, BLayout{}));

    Tensor<CDataType> c_g_m_n_host_result(
        f_host_tensor_descriptor(BatchCount, M, N, StrideC, CLayout{}));
    Tensor<ReduceDataType> d0_g_m_host_result({BatchCount, M});
    Tensor<ReduceDataType> d1_g_m_host_result({BatchCount, M});

    Tensor<CDataType> c_g_m_n_device_result(
        f_host_tensor_descriptor(BatchCount, M, N, StrideC, CLayout{}));
    Tensor<ReduceDataType> d0_g_m_device_result({BatchCount, M});
    Tensor<ReduceDataType> d1_g_m_device_result({BatchCount, M});

    std::cout << "a_g_m_k: " << a_g_m_k.mDesc << std::endl;
    std::cout << "b_g_k_n: " << b_g_k_n.mDesc << std::endl;
    std::cout << "c_g_m_n: " << c_g_m_n_host_result.mDesc << std::endl;
    std::cout << "d0_g_m: " << d0_g_m_host_result.mDesc << std::endl;
    std::cout << "d1_g_m: " << d1_g_m_host_result.mDesc << std::endl;

    std::size_t num_thread = std::thread::hardware_concurrency();
    switch(init_method)
    {
    case 0: break;
    case 1:
        std::srand(0);
        a_g_m_k.GenerateTensorValue(GeneratorTensor_2<ADataType>{-5, 5}, num_thread);
        b_g_k_n.GenerateTensorValue(GeneratorTensor_2<BDataType>{-5, 5}, num_thread);
        break;
    default:
        std::srand(0);
        a_g_m_k.GenerateTensorValue(GeneratorTensor_3<ADataType>{0.0, 1.0}, num_thread);
        b_g_k_n.GenerateTensorValue(GeneratorTensor_3<BDataType>{-0.5, 0.5}, num_thread);
    }

    using AElementOp            = ck::tensor_operation::element_wise::PassThrough;
    using BElementOp            = ck::tensor_operation::element_wise::PassThrough;
    using CElementOp            = ck::tensor_operation::element_wise::PassThrough;
    using ReduceOp0             = ck::reduce::Add;
    using ReduceOp1             = ck::reduce::Add;
    using UnaryIdenticElementOp = ck::tensor_operation::element_wise::PassThrough;
    using UnarySquareElementOp  = ck::tensor_operation::element_wise::UnarySquare;

    auto a_element_op                     = AElementOp{};
    auto b_element_op                     = BElementOp{};
    auto c_element_op                     = CElementOp{};
    std::array<void*, 3> gemm_element_ops = {&a_element_op, &b_element_op, &c_element_op};

    const auto reduce0_op = ReduceOp0{};
    const auto reduce1_op = ReduceOp1{};

    auto passthrough                            = UnaryIdenticElementOp{};
    auto square                                 = UnarySquareElementOp{};
    std::array<void*, 2> reduce_in_element_ops  = {&passthrough, &square};
    std::array<void*, 2> reduce_out_element_ops = {&passthrough, &passthrough};

    if(do_verification)
    {
        using ReferenceBatchedGemmInstance =
            ck::tensor_operation::host::ReferenceBatchedGemm<ADataType,
                                                             BDataType,
                                                             CDataType,
                                                             float,
                                                             AElementOp,
                                                             BElementOp,
                                                             CElementOp>;

        using ReduceAccDataType = ReduceDataType;

        auto ref_batched_gemm = ReferenceBatchedGemmInstance{};
        auto ref_invoker      = ref_batched_gemm.MakeInvoker();

        auto ref_argument = ref_batched_gemm.MakeArgument(
            a_g_m_k, b_g_k_n, c_g_m_n_host_result, a_element_op, b_element_op, c_element_op);

        ref_invoker.Run(ref_argument);

        for(int batch = 0; batch < BatchCount; ++batch)
        {
            for(int m = 0; m < M; ++m)
            {
                auto reduce0_acc = reduce0_op.GetIdentityValue<ReduceAccDataType>();
                auto reduce1_acc = reduce1_op.GetIdentityValue<ReduceAccDataType>();

                for(int n = 0; n < N; ++n)
                {
                    ReduceAccDataType d0_val =
                        ck::type_convert<ReduceAccDataType>(c_g_m_n_host_result(batch, m, n));
                    ReduceAccDataType d1_val;

                    square(d1_val, d0_val);
                    reduce0_op(reduce0_acc, d0_val);
                    reduce1_op(reduce1_acc, d1_val);
                }

                d0_g_m_host_result(batch, m) = ck::type_convert<ReduceDataType>(reduce0_acc);
                d1_g_m_host_result(batch, m) = ck::type_convert<ReduceDataType>(reduce1_acc);
            }
        }
    }

    DeviceMem a_device_buf(sizeof(ADataType) * a_g_m_k.mDesc.GetElementSpaceSize());
    DeviceMem b_device_buf(sizeof(BDataType) * b_g_k_n.mDesc.GetElementSpaceSize());
    DeviceMem c_device_buf(sizeof(CDataType) * c_g_m_n_device_result.mDesc.GetElementSpaceSize());
    DeviceMem reduce0_device_buf(sizeof(ReduceDataType) *
                                 d0_g_m_device_result.mDesc.GetElementSpaceSize());
    DeviceMem reduce1_device_buf(sizeof(ReduceDataType) *
                                 d1_g_m_device_result.mDesc.GetElementSpaceSize());

    std::array<void*, 2> p_reduces = {reduce0_device_buf.GetDeviceBuffer(),
                                      reduce1_device_buf.GetDeviceBuffer()};

    a_device_buf.ToDevice(a_g_m_k.mData.data());
    b_device_buf.ToDevice(b_g_k_n.mData.data());

    // add device GEMM instances
    std::vector<ck::tensor_operation::device::instance::DeviceGemmReduceNoOpPtr> gemm_ptrs;

    if constexpr(is_same<ADataType, half_t>::value && is_same<BDataType, half_t>::value &&
                 is_same<CDataType, half_t>::value)
    {
        if constexpr(is_same<ALayout, tensor_layout::gemm::RowMajor>::value &&
                     is_same<BLayout, tensor_layout::gemm::RowMajor>::value &&
                     is_same<CLayout, tensor_layout::gemm::RowMajor>::value)
        {
            ck::tensor_operation::device::instance::
                add_device_batched_gemm_reduce_xdl_cshuffle_f16_f16_f16_f32_f32_gmk_gkn_gmn_instances(
                    gemm_ptrs);
        }
        else if constexpr(is_same<ALayout, tensor_layout::gemm::RowMajor>::value &&
                          is_same<BLayout, tensor_layout::gemm::ColumnMajor>::value &&
                          is_same<CLayout, tensor_layout::gemm::RowMajor>::value)
        {
            ck::tensor_operation::device::instance::
                add_device_batched_gemm_reduce_xdl_cshuffle_f16_f16_f16_f32_f32_gmk_gnk_gmn_instances(
                    gemm_ptrs);
        }
        else if constexpr(is_same<ALayout, tensor_layout::gemm::ColumnMajor>::value &&
                          is_same<BLayout, tensor_layout::gemm::RowMajor>::value &&
                          is_same<CLayout, tensor_layout::gemm::RowMajor>::value)
        {
            ck::tensor_operation::device::instance::
                add_device_batched_gemm_reduce_xdl_cshuffle_f16_f16_f16_f32_f32_gkm_gkn_gmn_instances(
                    gemm_ptrs);
        }
        else if constexpr(is_same<ALayout, tensor_layout::gemm::ColumnMajor>::value &&
                          is_same<BLayout, tensor_layout::gemm::ColumnMajor>::value &&
                          is_same<CLayout, tensor_layout::gemm::RowMajor>::value)
        {
            ck::tensor_operation::device::instance::
                add_device_batched_gemm_reduce_xdl_cshuffle_f16_f16_f16_f32_f32_gkm_gnk_gmn_instances(
                    gemm_ptrs);
        }
    }

    if(gemm_ptrs.size() <= 0)
    {
        throw std::runtime_error("wrong! no device GEMM instance found");
    }

    std::string best_gemm_name;
    float best_ave_time   = 0;
    float best_tflops     = 0;
    float best_gb_per_sec = 0;

    // profile device GEMM instances
    for(auto& gemm_ptr : gemm_ptrs)
    {
        auto argument_ptr = gemm_ptr->MakeArgumentPointer(a_device_buf.GetDeviceBuffer(),
                                                          b_device_buf.GetDeviceBuffer(),
                                                          nullptr,
                                                          {},
                                                          c_device_buf.GetDeviceBuffer(),
                                                          p_reduces,
                                                          M,
                                                          N,
                                                          K,
                                                          StrideA,
                                                          StrideB,
                                                          StrideC,
                                                          {},
                                                          gemm_element_ops,
                                                          {},
                                                          reduce_in_element_ops,
                                                          reduce_out_element_ops,
                                                          BatchCount);

        auto invoker_ptr = gemm_ptr->MakeInvokerPointer();

        if(gemm_ptr->IsSupportedArgument(argument_ptr.get()))
        {
            // init DO, D1 to 0
            reduce0_device_buf.SetZero();
            reduce1_device_buf.SetZero();

            float ave_time =
                invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, time_kernel});

            std::string gemm_name = gemm_ptr->GetTypeString();

            std::size_t flop      = std::size_t(2) * BatchCount * M * N * K;
            std::size_t num_btype = sizeof(ADataType) * BatchCount * M * K +
                                    sizeof(BDataType) * BatchCount * K * N +
                                    sizeof(CDataType) * BatchCount * M * N;

            float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

            float gb_per_sec = num_btype / 1.E6 / ave_time;

            std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec
                      << " GB/s, " << gemm_name << std::endl;

            if(tflops > best_tflops)
            {
                best_gemm_name  = gemm_name;
                best_tflops     = tflops;
                best_ave_time   = ave_time;
                best_gb_per_sec = gb_per_sec;
            }

            if(do_verification)
            {
                c_device_buf.FromDevice(c_g_m_n_device_result.mData.data());
                reduce0_device_buf.FromDevice(d0_g_m_device_result.mData.data());
                reduce1_device_buf.FromDevice(d1_g_m_device_result.mData.data());

                bool c_error  = ck::utils::check_err(c_g_m_n_device_result, c_g_m_n_host_result);
                bool d0_error = ck::utils::check_err(d0_g_m_device_result, d0_g_m_host_result);
                bool d1_error = ck::utils::check_err(d1_g_m_device_result, d1_g_m_host_result);

                pass = pass && (c_error == true);
                pass = pass && (d0_error == true);
                pass = pass && (d1_error == true);

                if(do_log)
                {
                    LogRangeAsType<float>(std::cout << "a : ", a_g_m_k.mData, ",") << std::endl;
                    LogRangeAsType<float>(std::cout << "b: ", b_g_k_n.mData, ",") << std::endl;
                    LogRangeAsType<float>(std::cout << "c_host: ", c_g_m_n_host_result.mData, ",")
                        << std::endl;
                    LogRangeAsType<float>(
                        std::cout << "c_device: ", c_g_m_n_device_result.mData, ",")
                        << std::endl;
                    LogRangeAsType<float>(std::cout << "d0_host: ", d0_g_m_host_result.mData, ",")
                        << std::endl;
                    LogRangeAsType<float>(
                        std::cout << "d0_device: ", d0_g_m_device_result.mData, ",")
                        << std::endl;
                    LogRangeAsType<float>(std::cout << "d1_host: ", d1_g_m_host_result.mData, ",")
                        << std::endl;
                    LogRangeAsType<float>(
                        std::cout << "d1_device: ", d1_g_m_device_result.mData, ",")
                        << std::endl;
                }
            }
        }
        else
        {
            std::cout << "does not support this GEMM problem" << std::endl;
        }
    }

    std::cout << "Best Perf: " << best_ave_time << " ms, " << best_tflops << " TFlops, "
              << best_gb_per_sec << " GB/s, " << best_gemm_name << std::endl;

    return pass;
}

} // namespace profiler
} // namespace ck
