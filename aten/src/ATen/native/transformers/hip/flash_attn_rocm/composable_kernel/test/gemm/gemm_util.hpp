// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/literals.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_gemm.hpp"

namespace ck {
namespace gemm_util {

struct GemmParams
{
    ck::index_t M = 1024;
    ck::index_t N = 1024;
    ck::index_t K = 1024;

    ck::index_t StrideA = 1024;
    ck::index_t StrideB = 1024;
    ck::index_t StrideC = 1024;
};

template <typename GemmInstance,
          typename ADataType,
          typename BDataType,
          typename CDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation>
void RunHostGEMM(const Tensor<ADataType>& A,
                 const Tensor<BDataType>& B,
                 Tensor<CDataType>& C,
                 AElementwiseOperation a_element_op,
                 BElementwiseOperation b_element_op,
                 CElementwiseOperation c_element_op)
{
    auto ref_gemm    = GemmInstance{};
    auto ref_invoker = ref_gemm.MakeInvoker();

    auto ref_argument = ref_gemm.MakeArgument(A, B, C, a_element_op, b_element_op, c_element_op);

    ref_invoker.Run(ref_argument);
}

template <typename DeviceGemmPtr_,
          typename ADataType,
          typename BDataType,
          typename CDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation>
bool RunDeviceGEMM(DeviceGemmPtr_& gemmPtr,
                   const ck::gemm_util::GemmParams& params,
                   const Tensor<ADataType>& A,
                   const Tensor<BDataType>& B,
                   Tensor<CDataType>& C,
                   AElementwiseOperation a_element_op,
                   BElementwiseOperation b_element_op,
                   CElementwiseOperation c_element_op,
                   bool time_kernel)
{
    DeviceMem a_m_k_device_buf(sizeof(ADataType) * A.mDesc.GetElementSpaceSize());
    DeviceMem b_k_n_device_buf(sizeof(BDataType) * B.mDesc.GetElementSpaceSize());
    DeviceMem c_m_n_device_buf(sizeof(CDataType) * C.mDesc.GetElementSpaceSize());

    auto invoker_ptr = gemmPtr->MakeInvokerPointer();
    auto argument_ptr =
        gemmPtr->MakeArgumentPointer(static_cast<ADataType*>(a_m_k_device_buf.GetDeviceBuffer()),
                                     static_cast<BDataType*>(b_k_n_device_buf.GetDeviceBuffer()),
                                     static_cast<CDataType*>(c_m_n_device_buf.GetDeviceBuffer()),
                                     params.M,
                                     params.N,
                                     params.K,
                                     params.StrideA,
                                     params.StrideB,
                                     params.StrideC,
                                     a_element_op,
                                     b_element_op,
                                     c_element_op);

    if(gemmPtr->IsSupportedArgument(argument_ptr.get()))
    {
        a_m_k_device_buf.ToDevice(A.mData.data());
        b_k_n_device_buf.ToDevice(B.mData.data());
        float ave_time = invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, time_kernel});

        std::size_t flop      = std::size_t(2) * params.M * params.N * params.K;
        std::size_t num_btype = sizeof(ADataType) * params.M * params.K +
                                sizeof(BDataType) * params.K * params.N +
                                sizeof(CDataType) * params.M * params.N;

        float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

        float gb_per_sec = num_btype / 1.E6 / ave_time;

        std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec
                  << " GB/s, " << std::endl;

        c_m_n_device_buf.FromDevice(C.mData.data());

        return true;
    }
    else
    {
        std::cout << "device_gemm with the specified compilation parameters does "
                     "not support this GEMM problem"
                  << std::endl;

        return false;
    }
}

template <typename AccDataType>
struct TestGemm
{
    template <typename ADataType,
              typename BDataType,
              typename CDataType,
              typename ALayout,
              typename BLayout,
              typename CLayout>
    auto PrepareGemmTensor(const ck::gemm_util::GemmParams& params)
    {
        auto f_host_tensor_descriptor =
            [](std::size_t row, std::size_t col, std::size_t stride, auto layout) {
                using namespace ck::literals;

                if(std::is_same<decltype(layout), ck::tensor_layout::gemm::RowMajor>::value)
                {
                    return HostTensorDescriptor({row, col}, {stride, 1_uz});
                }
                else
                {
                    return HostTensorDescriptor({row, col}, {1_uz, stride});
                }
            };

        Tensor<ADataType> a_m_k(
            f_host_tensor_descriptor(params.M, params.K, params.StrideA, ALayout{}));
        Tensor<BDataType> b_k_n(
            f_host_tensor_descriptor(params.K, params.N, params.StrideB, BLayout{}));
        Tensor<CDataType> c_m_n_host_result(
            f_host_tensor_descriptor(params.M, params.N, params.StrideC, CLayout{}));
        Tensor<CDataType> c_m_n_device_result(
            f_host_tensor_descriptor(params.M, params.N, params.StrideC, CLayout{}));

        auto f_generate_tensor_value = [](auto& tensor, auto type) {
            using dataType = decltype(type);

            tensor.GenerateTensorValue(GeneratorTensor_2<dataType>{-5, 5});
        };

        f_generate_tensor_value(a_m_k, ADataType{});
        f_generate_tensor_value(b_k_n, BDataType{});

        std::cout << "a_m_k: " << a_m_k.mDesc << std::endl;
        std::cout << "b_k_n: " << b_k_n.mDesc << std::endl;
        std::cout << "c_m_n: " << c_m_n_host_result.mDesc << std::endl;

        return std::make_tuple(a_m_k, b_k_n, c_m_n_host_result, c_m_n_device_result);
    }

    template <template <class...> class DeviceGemmPtr_,
              typename ALayout,
              typename BLayout,
              typename CLayout,
              typename ADataType,
              typename BDataType,
              typename CDataType,
              typename AElementwiseOperation,
              typename BElementwiseOperation,
              typename CElementwiseOperation>
    auto operator()(DeviceGemmPtr_<ALayout,
                                   BLayout,
                                   CLayout,
                                   ADataType,
                                   BDataType,
                                   CDataType,
                                   AElementwiseOperation,
                                   BElementwiseOperation,
                                   CElementwiseOperation>* gemmPtr,
                    const GemmParams& params = GemmParams{},
                    bool do_verification     = true,
                    bool time_kernel         = false)
    {
        std::cout << "ALayout = " << ALayout{}.name << ", BLayout = " << BLayout{}.name
                  << ", CLayout = " << CLayout{}.name << std::endl;
        std::cout << gemmPtr->GetTypeString() << std::endl;

        auto host_tensors =
            PrepareGemmTensor<ADataType, BDataType, CDataType, ALayout, BLayout, CLayout>(params);

        const Tensor<ADataType>& a  = std::get<0>(host_tensors);
        const Tensor<BDataType>& b  = std::get<1>(host_tensors);
        Tensor<CDataType>& c_host   = std::get<2>(host_tensors);
        Tensor<CDataType>& c_device = std::get<3>(host_tensors);

        auto a_element_op = AElementwiseOperation{};
        auto b_element_op = BElementwiseOperation{};
        auto c_element_op = CElementwiseOperation{};

        using ReferenceGemmInstance =
            ck::tensor_operation::host::ReferenceGemm<ADataType,
                                                      BDataType,
                                                      CDataType,
                                                      AccDataType,
                                                      AElementwiseOperation,
                                                      BElementwiseOperation,
                                                      CElementwiseOperation>;

        if(do_verification)
        {
            ck::gemm_util::RunHostGEMM<ReferenceGemmInstance>(
                a, b, c_host, a_element_op, b_element_op, c_element_op);
        }

        // Act
        bool is_supported = ck::gemm_util::RunDeviceGEMM(
            gemmPtr, params, a, b, c_device, a_element_op, b_element_op, c_element_op, time_kernel);

        if(is_supported && do_verification)
        {
            // Assert
            bool res = false;
            if(std::is_same<CDataType, float>::value)
            {
                res = ck::utils::check_err(c_device, c_host);
                std::cout << (res ? "SUCCESS" : "FAILURE") << std::endl;
            }
            else if(std::is_same<CDataType, ck::half_t>::value)
            {
                res = ck::utils::check_err(c_device, c_host);
                std::cout << (res ? "SUCCESS" : "FAILURE") << std::endl;
            }
            else if(std::is_same<CDataType, ck::bhalf_t>::value)
            {
                res = ck::utils::check_err(c_device, c_host);
                std::cout << (res ? "SUCCESS" : "FAILURE") << std::endl;
            }
            else if(std::is_same<CDataType, int8_t>::value)
            {
                res = ck::utils::check_err(c_device, c_host);
                std::cout << (res ? "SUCCESS" : "FAILURE") << std::endl;
            }
            else if(std::is_same<CDataType, double>::value)
            {
                res = ck::utils::check_err(c_device, c_host);
                std::cout << (res ? "SUCCESS" : "FAILURE") << std::endl;
            }

            return res;
        }
        else
        {
            return true;
        }
    }
};

} // namespace gemm_util
} // namespace ck
