// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <numeric>
#include <initializer_list>
#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/stream_config.hpp"
#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/literals.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using F16   = ck::half_t;
using F32   = float;
using BF16  = ck::bhalf_t;
using INT8  = std::int8_t;
using INT32 = std::int32_t;
#ifdef CK_EXPERIMENTAL_BIT_INT_EXTENSION_INT4
using INT4 = ck::int4_t;
#endif

template <typename ADataType,
          typename BDataType,
          typename CDataType,
          typename ALayout,
          typename BLayout,
          typename CLayout,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation,
          typename DeviceCGemmInstance,
          typename ReferenceCGemmInstance,
          typename KernelADataType = ADataType,
          typename KernelBDataType = BDataType,
          typename KernelCDataType = CDataType>
bool run_cgemm_xdl(ck::index_t M,
                   ck::index_t N,
                   ck::index_t K,
                   ck::index_t StrideA,
                   ck::index_t StrideB,
                   ck::index_t StrideC,
                   bool do_verification,
                   int init_method,
                   bool time_kernel)
{
#ifdef CK_EXPERIMENTAL_BIT_INT_EXTENSION_INT4
    static_assert(sizeof(ck::int4_t) == sizeof(int8_t),
                  "sizeof ck::int4_t and int8_t is different!");
    static_assert(sizeof(ADataType) == sizeof(KernelADataType),
                  "sizeof ADataType and KernelADataType is different!");
    static_assert(sizeof(BDataType) == sizeof(KernelBDataType),
                  "sizeof BDataType and KernelBDataType is different!");
    static_assert(sizeof(CDataType) == sizeof(KernelCDataType),
                  "sizeof CDataType and KernelCDataType is different!");
#endif

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

    Tensor<ADataType> a_m_k_real(f_host_tensor_descriptor(M, K, StrideA, ALayout{}));
    Tensor<ADataType> a_m_k_imag(f_host_tensor_descriptor(M, K, StrideA, ALayout{}));
    Tensor<BDataType> b_k_n_real(f_host_tensor_descriptor(K, N, StrideB, BLayout{}));
    Tensor<BDataType> b_k_n_imag(f_host_tensor_descriptor(K, N, StrideB, BLayout{}));
    Tensor<KernelCDataType> c_m_n_real_device_result(
        f_host_tensor_descriptor(M, N, StrideC, CLayout{}));
    Tensor<KernelCDataType> c_m_n_imag_device_result(
        f_host_tensor_descriptor(M, N, StrideC, CLayout{}));

    std::cout << "a_m_k_real: " << a_m_k_real.mDesc << std::endl;
    std::cout << "a_m_k_imag: " << a_m_k_imag.mDesc << std::endl;
    std::cout << "b_k_n_real: " << b_k_n_real.mDesc << std::endl;
    std::cout << "b_k_n_imag: " << b_k_n_imag.mDesc << std::endl;
    std::cout << "c_m_n_real: " << c_m_n_real_device_result.mDesc << std::endl;
    std::cout << "c_m_n_imag: " << c_m_n_imag_device_result.mDesc << std::endl;

    switch(init_method)
    {
    case 0: break;
    case 1:
        a_m_k_real.GenerateTensorValue(GeneratorTensor_2<ADataType>{-2, 2});
        a_m_k_imag.GenerateTensorValue(GeneratorTensor_2<ADataType>{-2, 2});
        b_k_n_real.GenerateTensorValue(GeneratorTensor_2<BDataType>{-2, 2});
        b_k_n_imag.GenerateTensorValue(GeneratorTensor_2<BDataType>{-2, 2});
        break;
    default:
        a_m_k_real.GenerateTensorValue(GeneratorTensor_3<ADataType>{-0.5, 0.5});
        a_m_k_imag.GenerateTensorValue(GeneratorTensor_3<ADataType>{-0.5, 0.5});
        b_k_n_real.GenerateTensorValue(GeneratorTensor_3<BDataType>{-0.5, 0.5});
        b_k_n_imag.GenerateTensorValue(GeneratorTensor_3<BDataType>{-0.5, 0.5});
    }

    auto cgemm = DeviceCGemmInstance{};

    DeviceMem a_m_k_real_device_buf(sizeof(KernelADataType) *
                                    a_m_k_real.mDesc.GetElementSpaceSize());
    DeviceMem a_m_k_imag_device_buf(sizeof(KernelADataType) *
                                    a_m_k_imag.mDesc.GetElementSpaceSize());
    DeviceMem b_k_n_real_device_buf(sizeof(KernelBDataType) *
                                    b_k_n_real.mDesc.GetElementSpaceSize());
    DeviceMem b_k_n_imag_device_buf(sizeof(KernelBDataType) *
                                    b_k_n_imag.mDesc.GetElementSpaceSize());
    DeviceMem c_m_n_real_device_buf(sizeof(KernelCDataType) *
                                    c_m_n_real_device_result.mDesc.GetElementSpaceSize());
    DeviceMem c_m_n_imag_device_buf(sizeof(KernelCDataType) *
                                    c_m_n_imag_device_result.mDesc.GetElementSpaceSize());
    DeviceMem workspace_device_buf(cgemm.GetWorkspaceSize(M, N, K, StrideA, StrideB, StrideC));

#ifdef CK_EXPERIMENTAL_BIT_INT_EXTENSION_INT4
    if constexpr(std::is_same_v<ADataType, ck::int4_t>)
    {
        Tensor<KernelADataType> a_m_k_real_converted(a_m_k_real);
        Tensor<KernelADataType> a_m_k_imag_converted(a_m_k_imag);
        Tensor<KernelBDataType> b_k_n_real_converted(b_k_n_real);
        Tensor<KernelBDataType> b_k_n_imag_converted(b_k_n_imag);

        a_m_k_real_device_buf.ToDevice(a_m_k_real_converted.mData.data());
        a_m_k_imag_device_buf.ToDevice(a_m_k_imag_converted.mData.data());
        b_k_n_real_device_buf.ToDevice(b_k_n_real_converted.mData.data());
        b_k_n_imag_device_buf.ToDevice(b_k_n_imag_converted.mData.data());
    }
    else
#endif // CK_EXPERIMENTAL_BIT_INT_EXTENSION_INT4
    {
        a_m_k_real_device_buf.ToDevice(a_m_k_real.mData.data());
        a_m_k_imag_device_buf.ToDevice(a_m_k_imag.mData.data());
        b_k_n_real_device_buf.ToDevice(b_k_n_real.mData.data());
        b_k_n_imag_device_buf.ToDevice(b_k_n_imag.mData.data());
    }

    auto a_element_op = AElementwiseOperation{};
    auto b_element_op = BElementwiseOperation{};
    auto c_element_op = CElementwiseOperation{};

    // do GEMM
    auto invoker = cgemm.MakeInvoker();
    auto argument =
        cgemm.MakeArgument(static_cast<KernelADataType*>(a_m_k_real_device_buf.GetDeviceBuffer()),
                           static_cast<KernelADataType*>(a_m_k_imag_device_buf.GetDeviceBuffer()),
                           static_cast<KernelBDataType*>(b_k_n_real_device_buf.GetDeviceBuffer()),
                           static_cast<KernelBDataType*>(b_k_n_imag_device_buf.GetDeviceBuffer()),
                           static_cast<KernelCDataType*>(c_m_n_real_device_buf.GetDeviceBuffer()),
                           static_cast<KernelCDataType*>(c_m_n_imag_device_buf.GetDeviceBuffer()),
                           static_cast<KernelCDataType*>(workspace_device_buf.GetDeviceBuffer()),
                           M,
                           N,
                           K,
                           StrideA,
                           StrideB,
                           StrideC,
                           a_element_op,
                           b_element_op,
                           c_element_op);

    if(!cgemm.IsSupportedArgument(argument))
    {
        throw std::runtime_error(
            "wrong! device_cgemm with the specified compilation parameters does "
            "not support this CGEMM problem");
    }

    float ave_time = invoker.Run(argument, StreamConfig{nullptr, time_kernel});

    std::size_t flop = std::size_t(8) * M * N * K;
    std::size_t num_btype =
        std::size_t(2) *
        (sizeof(ADataType) * M * K + sizeof(BDataType) * K * N + sizeof(CDataType) * M * N);

    float tflops     = static_cast<float>(flop) / 1.E9 / ave_time;
    float gb_per_sec = num_btype / 1.E6 / ave_time;

    std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s, "
              << cgemm.GetTypeString() << std::endl;

    if(do_verification)
    {
        Tensor<CDataType> c_m_n_real_host_result(
            f_host_tensor_descriptor(M, N, StrideC, CLayout{}));
        Tensor<CDataType> c_m_n_imag_host_result(
            f_host_tensor_descriptor(M, N, StrideC, CLayout{}));

        auto ref_cgemm    = ReferenceCGemmInstance{};
        auto ref_invoker  = ref_cgemm.MakeInvoker();
        auto ref_argument = ref_cgemm.MakeArgument(a_m_k_real,
                                                   a_m_k_imag,
                                                   b_k_n_real,
                                                   b_k_n_imag,
                                                   c_m_n_real_host_result,
                                                   c_m_n_imag_host_result,
                                                   a_element_op,
                                                   b_element_op,
                                                   c_element_op);

        ref_invoker.Run(ref_argument);

        c_m_n_real_device_buf.FromDevice(c_m_n_real_device_result.mData.data());
        c_m_n_imag_device_buf.FromDevice(c_m_n_imag_device_result.mData.data());

        bool result = true;
#ifdef CK_EXPERIMENTAL_BIT_INT_EXTENSION_INT4
        if constexpr(std::is_same_v<ADataType, ck::int4_t>)
        {
            const Tensor<CDataType> c_m_n_real_device_result_converted(c_m_n_real_device_result);
            const Tensor<CDataType> c_m_n_imag_device_result_converted(c_m_n_imag_device_result);

            result = ck::utils::check_err(c_m_n_real_device_result_converted,
                                          c_m_n_real_host_result,
                                          "Verification error: incorrect results in real part!",
                                          1e-2f,
                                          1e-1f);
            result = result && ck::utils::check_err(
                                   c_m_n_imag_device_result_converted,
                                   c_m_n_imag_host_result,
                                   "Verification error: incorrect results in imaginary part!",
                                   1e-2f,
                                   1e-1f);
        }
        else
#endif // CK_EXPERIMENTAL_BIT_INT_EXTENSION_INT4
        {
            result = ck::utils::check_err(c_m_n_real_device_result,
                                          c_m_n_real_host_result,
                                          "Verification error: incorrect results in real part!",
                                          1e-2f,
                                          1e-1f);
            result = result && ck::utils::check_err(
                                   c_m_n_imag_device_result,
                                   c_m_n_imag_host_result,
                                   "Verification error: incorrect results in imaginary part!",
                                   1e-2f,
                                   1e-1f);
        }

        return result;
    }
    return true;
}
