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
#include "ck/library/reference_tensor_operation/cpu/reference_gemm.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

using F32                 = float;
using F16                 = ck::half_t;
using ReducePtrsGlobal    = ck::Tuple<F32*, F32*>;
using Div                 = ck::tensor_operation::element_wise::UnaryDivide;
using Identity            = ck::tensor_operation::element_wise::PassThrough;
using Square              = ck::tensor_operation::element_wise::UnarySquare;
using ReduceInElementOps  = ck::Tuple<Identity, Square>;
using ReduceOutElementOps = ck::Tuple<Div, Div>;

using DeviceGemmReduceNoOpPtr =
    ck::tensor_operation::device::DeviceGemmReducePtr<0, ReducePtrsGlobal::Size()>;

void add_device_gemm_reduce_xdl_cshuffle_f16_f16_f16_f32_f32_mk_kn_mn_instances(
    std::vector<DeviceGemmReduceNoOpPtr>&);

void add_device_gemm_reduce_xdl_cshuffle_f16_f16_f16_f32_f32_mk_nk_mn_instances(
    std::vector<DeviceGemmReduceNoOpPtr>&);

void add_device_gemm_reduce_xdl_cshuffle_f16_f16_f16_f32_f32_km_kn_mn_instances(
    std::vector<DeviceGemmReduceNoOpPtr>&);

void add_device_gemm_reduce_xdl_cshuffle_f16_f16_f16_f32_f32_km_nk_mn_instances(
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
bool profile_gemm_reduce_impl(int do_verification,
                              int init_method,
                              bool do_log,
                              bool time_kernel,
                              int M,
                              int N,
                              int K,
                              int StrideA,
                              int StrideB,
                              int StrideC)
{
    bool pass = true;

    auto f_host_tensor_descriptor =
        [](std::size_t row, std::size_t col, std::size_t stride, auto layout) {
            using namespace ck::literals;

            if(is_same<decltype(layout), tensor_layout::gemm::RowMajor>::value)
            {
                return HostTensorDescriptor({row, col}, {stride, 1_uz});
            }
            else
            {
                return HostTensorDescriptor({row, col}, {1_uz, stride});
            }
        };

    Tensor<ADataType> a_m_k(f_host_tensor_descriptor(M, K, StrideA, ALayout{}));
    Tensor<BDataType> b_k_n(f_host_tensor_descriptor(K, N, StrideB, BLayout{}));

    Tensor<CDataType> c_m_n_host_result(f_host_tensor_descriptor(M, N, StrideC, CLayout{}));
    Tensor<ReduceDataType> reduce0_m_host_result({M});
    Tensor<ReduceDataType> reduce1_m_host_result({M});

    Tensor<CDataType> c_m_n_device_result(f_host_tensor_descriptor(M, N, StrideC, CLayout{}));
    Tensor<ReduceDataType> reduce0_m_device_result({M});
    Tensor<ReduceDataType> reduce1_m_device_result({M});

    std::cout << "a_m_k: " << a_m_k.mDesc << std::endl;
    std::cout << "b_k_n: " << b_k_n.mDesc << std::endl;
    std::cout << "c_m_n: " << c_m_n_host_result.mDesc << std::endl;
    std::cout << "reduce0_m: " << reduce0_m_host_result.mDesc << std::endl;
    std::cout << "reduce1_m: " << reduce1_m_host_result.mDesc << std::endl;

    std::size_t num_thread = 1;
    switch(init_method)
    {
    case 0: break;
    case 1:
        std::srand(0);
        a_m_k.GenerateTensorValue(GeneratorTensor_2<ADataType>{-5, 5}, num_thread);
        b_k_n.GenerateTensorValue(GeneratorTensor_2<BDataType>{-5, 5}, num_thread);
        break;
    default:
        std::srand(0);
        a_m_k.GenerateTensorValue(GeneratorTensor_3<ADataType>{0.0, 1.0}, num_thread);
        b_k_n.GenerateTensorValue(GeneratorTensor_3<BDataType>{-0.5, 0.5}, num_thread);
    }

    using AElementOp            = ck::tensor_operation::element_wise::PassThrough;
    using BElementOp            = ck::tensor_operation::element_wise::PassThrough;
    using CElementOp            = ck::tensor_operation::element_wise::PassThrough;
    using ReduceOp0             = ck::reduce::Add;
    using ReduceOp1             = ck::reduce::Add;
    using UnaryIdenticElementOp = ck::tensor_operation::element_wise::PassThrough;
    using UnarySquareElementOp  = ck::tensor_operation::element_wise::UnarySquare;
    using UnaryDivElementOp     = ck::tensor_operation::element_wise::UnaryDivide;

    auto a_element_op                     = AElementOp{};
    auto b_element_op                     = BElementOp{};
    auto c_element_op                     = CElementOp{};
    std::array<void*, 3> gemm_element_ops = {&a_element_op, &b_element_op, &c_element_op};

    const auto reduce0_op = ReduceOp0{};
    const auto reduce1_op = ReduceOp1{};

    auto passthrough                            = UnaryIdenticElementOp{};
    auto square                                 = UnarySquareElementOp{};
    auto div                                    = UnaryDivElementOp{N};
    std::array<void*, 2> reduce_in_element_ops  = {&passthrough, &square};
    std::array<void*, 2> reduce_out_element_ops = {&div, &div};

    if(do_verification)
    {
        using ReferenceGemmInstance = ck::tensor_operation::host::ReferenceGemm<ADataType,
                                                                                BDataType,
                                                                                CDataType,
                                                                                ReduceDataType,
                                                                                AElementOp,
                                                                                BElementOp,
                                                                                CElementOp>;

        using ReduceAccDataType = ReduceDataType;

        auto ref_gemm    = ReferenceGemmInstance{};
        auto ref_invoker = ref_gemm.MakeInvoker();

        auto ref_argument = ref_gemm.MakeArgument(
            a_m_k, b_k_n, c_m_n_host_result, a_element_op, b_element_op, c_element_op);

        ref_invoker.Run(ref_argument);

        for(int m = 0; m < M; ++m)
        {
            auto reduce0_acc = reduce0_op.GetIdentityValue<ReduceAccDataType>();
            auto reduce1_acc = reduce1_op.GetIdentityValue<ReduceAccDataType>();

            for(int n = 0; n < N; ++n)
            {
                ReduceAccDataType d0_val =
                    ck::type_convert<ReduceAccDataType>(c_m_n_host_result(m, n));
                ReduceAccDataType d1_val;

                square(d1_val, d0_val);
                reduce0_op(reduce0_acc, d0_val);
                reduce1_op(reduce1_acc, d1_val);
            }

            div(reduce0_acc, reduce0_acc);
            div(reduce1_acc, reduce1_acc);
            reduce0_m_host_result(m) = ck::type_convert<ReduceDataType>(reduce0_acc);
            reduce1_m_host_result(m) = ck::type_convert<ReduceDataType>(reduce1_acc);
        }
    }

    DeviceMem a_device_buf(sizeof(ADataType) * a_m_k.mDesc.GetElementSpaceSize());
    DeviceMem b_device_buf(sizeof(BDataType) * b_k_n.mDesc.GetElementSpaceSize());
    DeviceMem c_device_buf(sizeof(CDataType) * c_m_n_device_result.mDesc.GetElementSpaceSize());
    DeviceMem reduce0_device_buf(sizeof(ReduceDataType) *
                                 reduce0_m_device_result.mDesc.GetElementSpaceSize());
    DeviceMem reduce1_device_buf(sizeof(ReduceDataType) *
                                 reduce1_m_device_result.mDesc.GetElementSpaceSize());

    std::array<void*, 2> p_reduces = {reduce0_device_buf.GetDeviceBuffer(),
                                      reduce1_device_buf.GetDeviceBuffer()};

    a_device_buf.ToDevice(a_m_k.mData.data());
    b_device_buf.ToDevice(b_k_n.mData.data());

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
                add_device_gemm_reduce_xdl_cshuffle_f16_f16_f16_f32_f32_mk_kn_mn_instances(
                    gemm_ptrs);
        }
        else if constexpr(is_same<ALayout, tensor_layout::gemm::RowMajor>::value &&
                          is_same<BLayout, tensor_layout::gemm::ColumnMajor>::value &&
                          is_same<CLayout, tensor_layout::gemm::RowMajor>::value)
        {
            ck::tensor_operation::device::instance::
                add_device_gemm_reduce_xdl_cshuffle_f16_f16_f16_f32_f32_mk_nk_mn_instances(
                    gemm_ptrs);
        }
        else if constexpr(is_same<ALayout, tensor_layout::gemm::ColumnMajor>::value &&
                          is_same<BLayout, tensor_layout::gemm::RowMajor>::value &&
                          is_same<CLayout, tensor_layout::gemm::RowMajor>::value)
        {
            ck::tensor_operation::device::instance::
                add_device_gemm_reduce_xdl_cshuffle_f16_f16_f16_f32_f32_km_kn_mn_instances(
                    gemm_ptrs);
        }
        else if constexpr(is_same<ALayout, tensor_layout::gemm::ColumnMajor>::value &&
                          is_same<BLayout, tensor_layout::gemm::ColumnMajor>::value &&
                          is_same<CLayout, tensor_layout::gemm::RowMajor>::value)
        {
            ck::tensor_operation::device::instance::
                add_device_gemm_reduce_xdl_cshuffle_f16_f16_f16_f32_f32_km_nk_mn_instances(
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
                                                          reduce_out_element_ops);

        auto invoker_ptr = gemm_ptr->MakeInvokerPointer();

        if(gemm_ptr->IsSupportedArgument(argument_ptr.get()))
        {
            // init DO, D1 to 0
            reduce0_device_buf.SetZero();
            reduce1_device_buf.SetZero();

            float ave_time =
                invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, time_kernel});

            std::string gemm_name = gemm_ptr->GetTypeString();

            std::size_t flop = std::size_t(2) * M * N * K;

            std::size_t num_btype = sizeof(ADataType) * M * K + sizeof(BDataType) * K * N +
                                    sizeof(CDataType) * M * N + sizeof(CDataType) * N;

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
                c_device_buf.FromDevice(c_m_n_device_result.mData.data());
                reduce0_device_buf.FromDevice(reduce0_m_device_result.mData.data());
                reduce1_device_buf.FromDevice(reduce1_m_device_result.mData.data());

                ck::utils::check_err(c_m_n_device_result, c_m_n_host_result);
                ck::utils::check_err(reduce0_m_device_result, reduce0_m_host_result);
                ck::utils::check_err(reduce1_m_device_result, reduce1_m_host_result);

                if(do_log)
                {
                    LogRangeAsType<float>(std::cout << "a : ", a_m_k.mData, ",") << std::endl;
                    LogRangeAsType<float>(std::cout << "b: ", b_k_n.mData, ",") << std::endl;
                    LogRangeAsType<float>(std::cout << "c_host: ", c_m_n_host_result.mData, ",")
                        << std::endl;
                    LogRangeAsType<float>(std::cout << "c_device: ", c_m_n_device_result.mData, ",")
                        << std::endl;
                    LogRangeAsType<float>(
                        std::cout << "d0_host: ", reduce0_m_host_result.mData, ",")
                        << std::endl;
                    LogRangeAsType<float>(
                        std::cout << "d0_device: ", reduce0_m_device_result.mData, ",")
                        << std::endl;
                    LogRangeAsType<float>(
                        std::cout << "d1_host: ", reduce1_m_host_result.mData, ",")
                        << std::endl;
                    LogRangeAsType<float>(
                        std::cout << "d1_device: ", reduce1_m_device_result.mData, ",")
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
