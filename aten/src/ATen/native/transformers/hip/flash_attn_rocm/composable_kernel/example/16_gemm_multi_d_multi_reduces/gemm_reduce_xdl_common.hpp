// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <numeric>
#include <initializer_list>
#include <cstdlib>

#include <iostream>

#include "ck/ck.hpp"
#include "ck/host_utility/io.hpp"
#include "ck/stream_config.hpp"
#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/fill.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/literals.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;
using F16         = ck::half_t;
using BF16        = ck::bhalf_t;
using F32         = float;
using F64         = double;
#ifdef CK_EXPERIMENTAL_BIT_INT_EXTENSION_INT4
using INT4 = ck::int4_t;
#endif
using INT8  = std::int8_t;
using INT32 = std::int32_t;

template <typename ADataType, typename BDataType, typename EDataType, typename R0DataType>
void DumpGemmReduceMaxPerf(float ave_time, int M, int N, int K)
{
    using namespace ck::literals;

    std::size_t flop          = 2_uz * M * N * K;
    std::size_t gemm_num_byte = sizeof(ADataType) * M * K + sizeof(BDataType) * K * N +
                                sizeof(EDataType) * M * N + sizeof(R0DataType) * M;

    float tflops          = static_cast<float>(flop) / 1.E9 / ave_time;
    float gemm_gb_per_sec = gemm_num_byte / 1.E6 / ave_time;

    std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gemm_gb_per_sec
              << " GB/s, " << std::endl;
}

template <typename ADataType,
          typename BDataType,
          typename EDataType,
          typename R0DataType,
          typename R1DataType>
void DumpGemmReduceMeanSquareMeanPerf(float ave_time, int M, int N, int K)
{
    using namespace ck::literals;

    std::size_t flop          = 2_uz * M * N * K + M * (3_uz * N + 2_uz);
    std::size_t gemm_num_byte = sizeof(ADataType) * M * K + sizeof(BDataType) * K * N +
                                sizeof(EDataType) * M * N + sizeof(R0DataType) * M +
                                sizeof(R1DataType) * M;

    float tflops          = static_cast<float>(flop) / 1.E9 / ave_time;
    float gemm_gb_per_sec = gemm_num_byte / 1.E6 / ave_time;

    std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gemm_gb_per_sec
              << " GB/s, " << std::endl;
}

template <typename ADataType,
          typename BDataType,
          typename EDataType,
          typename R0DataType,
          typename ALayout,
          typename BLayout,
          typename ELayout,
          typename AElementOp,
          typename BElementOp,
          typename CDEElementOp,
          typename QsElementOp,
          typename RsElementOp,
          typename RsThreadReduceOp,
          typename ReduceAccDataType,
          typename DeviceOpInstance,
          typename ReferenceGemmInstance,
          typename ADataKernelType = ADataType,
          typename BDataKernelType = BDataType,
          typename EDataKernelType = EDataType>
auto run_gemm_reduce_max_xdl(ck::index_t M,
                             ck::index_t N,
                             ck::index_t K,
                             ck::index_t StrideA,
                             ck::index_t StrideB,
                             ck::index_t StrideE,
                             bool do_verification,
                             int init_method,
                             bool time_kernel)
{
#ifdef CK_EXPERIMENTAL_BIT_INT_EXTENSION_INT4
    static_assert(sizeof(ck::int4_t) == sizeof(int8_t));
    static_assert(sizeof(ADataType) == sizeof(ADataKernelType));
    static_assert(sizeof(BDataType) == sizeof(BDataKernelType));
    static_assert(sizeof(EDataType) == sizeof(EDataKernelType));
#endif
    using namespace ck::literals;

    auto f_host_tensor_descriptor1d = [](std::size_t len, std::size_t stride) {
        return HostTensorDescriptor({len}, {stride});
    };

    auto f_host_tensor_descriptor2d =
        [](std::size_t row, std::size_t col, std::size_t stride, auto layout) {
            if(std::is_same<decltype(layout), ck::tensor_layout::gemm::RowMajor>::value)
            {
                return HostTensorDescriptor({row, col}, {stride, 1_uz});
            }
            else
            {
                return HostTensorDescriptor({row, col}, {1_uz, stride});
            }
        };

    Tensor<ADataType> a_m_k(f_host_tensor_descriptor2d(M, K, StrideA, ALayout{}));
    Tensor<BDataType> b_k_n(f_host_tensor_descriptor2d(K, N, StrideB, BLayout{}));
    Tensor<EDataKernelType> e_m_n(f_host_tensor_descriptor2d(M, N, StrideE, ELayout{}));
    Tensor<R0DataType> r0_m(f_host_tensor_descriptor1d(M, 1));

    switch(init_method)
    {
    case 0: break;
    case 1:
        ck::utils::FillUniformDistributionIntegerValue<ADataType>{-5.f, 5.f}(a_m_k);
        ck::utils::FillUniformDistributionIntegerValue<BDataType>{-5.f, 5.f}(b_k_n);
        break;
    default:
        ck::utils::FillUniformDistribution<ADataType>{-1.f, 1.f}(a_m_k);
        ck::utils::FillUniformDistribution<BDataType>{-1.f, 1.f}(b_k_n);
        break;
    }

    DeviceMem a_device_buf(sizeof(ADataKernelType) * a_m_k.mDesc.GetElementSpaceSize());
    DeviceMem b_device_buf(sizeof(BDataKernelType) * b_k_n.mDesc.GetElementSpaceSize());
    DeviceMem e_device_buf(sizeof(EDataKernelType) * e_m_n.mDesc.GetElementSpaceSize());
    DeviceMem r0_device_buf(sizeof(R0DataType) * r0_m.mDesc.GetElementSpaceSize());

#ifdef CK_EXPERIMENTAL_BIT_INT_EXTENSION_INT4
    if constexpr(std::is_same_v<ADataType, ck::int4_t>)
    {
        Tensor<ADataKernelType> a_m_k_converted = a_m_k.template CopyAsType<ADataKernelType>();
        Tensor<BDataKernelType> b_k_n_converted = b_k_n.template CopyAsType<BDataKernelType>();

        a_device_buf.ToDevice(a_m_k_converted.mData.data());
        b_device_buf.ToDevice(b_k_n_converted.mData.data());
    }
    else
#endif // CK_EXPERIMENTAL_BIT_INT_EXTENSION_INT4
    {
        a_device_buf.ToDevice(a_m_k.mData.data());
        b_device_buf.ToDevice(b_k_n.mData.data());
    }

    auto a_element_op   = AElementOp{};
    auto b_element_op   = BElementOp{};
    auto cde_element_op = CDEElementOp{};
    auto qs_element_op  = QsElementOp{};
    auto rs_element_op  = RsElementOp{};

    // Prepare GEMM, max
    auto device_op = DeviceOpInstance{};
    auto invoker   = device_op.MakeInvoker();
    auto argument  = device_op.MakeArgument(a_device_buf.GetDeviceBuffer(),
                                           b_device_buf.GetDeviceBuffer(),
                                           {},
                                           e_device_buf.GetDeviceBuffer(),
                                           {r0_device_buf.GetDeviceBuffer()},
                                           M,
                                           N,
                                           K,
                                           StrideA,
                                           StrideB,
                                           {},
                                           StrideE,
                                           a_element_op,
                                           b_element_op,
                                           cde_element_op,
                                           qs_element_op,
                                           rs_element_op);

    if(!device_op.IsSupportedArgument(argument))
    {
        throw std::runtime_error("wrong! this device_op instance does not support this problem");
    }

    // [CAUTION]: launch_and_time_kernel will not initialize D.
    // If we evaluate kernel multiple time but without initialize D. Verification will fail
    r0_device_buf.SetValue(ck::NumericLimits<R0DataType>::Lowest());

    invoker.Run(argument, StreamConfig{nullptr, false});

    bool pass = true;

    if(do_verification)
    {
        auto I0 = ck::Number<0>{};

        Tensor<ReduceAccDataType> e_m_n_host(e_m_n.mDesc);
        Tensor<R0DataType> r0_m_host(r0_m.mDesc);

        auto ref_gemm    = ReferenceGemmInstance{};
        auto ref_invoker = ref_gemm.MakeInvoker();

        auto ref_argument = ref_gemm.MakeArgument(
            a_m_k, b_k_n, e_m_n_host, a_element_op, b_element_op, cde_element_op);

        ref_invoker.Run(ref_argument);

        auto reduce0_op = RsThreadReduceOp{}[I0];

        for(int m = 0; m < M; ++m)
        {
            auto reduce0_acc = reduce0_op.template GetIdentityValue<ReduceAccDataType>();

            for(int n = 0; n < N; ++n)
            {
                auto e_val = e_m_n_host(m, n);
                reduce0_op(reduce0_acc, e_val);
            };

            r0_m_host(m) = ck::type_convert<R0DataType>(reduce0_acc);
        }

        e_device_buf.FromDevice(e_m_n.mData.data());
        Tensor<EDataType> e_m_n_host_converted(e_m_n_host);

#ifdef CK_EXPERIMENTAL_BIT_INT_EXTENSION_INT4
        if constexpr(std::is_same_v<ADataType, ck::int4_t>)
        {
            Tensor<EDataType> e_m_n_device_converted(e_m_n);
            pass = ck::utils::check_err(e_m_n_device_converted,
                                        e_m_n_host_converted,
                                        "Error: Incorrect results c",
                                        1e-2,
                                        1e-2);
        }
        else
#endif // CK_EXPERIMENTAL_BIT_INT_EXTENSION_INT4
        {
            pass = ck::utils::check_err(
                e_m_n, e_m_n_host_converted, "Error: Incorrect results c", 1e-2, 1e-2);
        }

        r0_device_buf.FromDevice(r0_m.mData.data());
        pass &= ck::utils::check_err(r0_m, r0_m_host, "Error: Incorrect results d0", 1e-2, 1e-2);

        if(pass)
        {
            std::cout << "Success!" << std::endl;
        }
    }

    if(time_kernel)
    {
        float ave_time = invoker.Run(argument, StreamConfig{nullptr, time_kernel});
        DumpGemmReduceMaxPerf<ADataType, BDataType, EDataType, R0DataType>(ave_time, M, N, K);
    }

    return pass ? 0 : 1;
}

template <typename ADataType,
          typename BDataType,
          typename EDataType,
          typename R0DataType,
          typename R1DataType,
          typename ALayout,
          typename BLayout,
          typename ELayout,
          typename AElementOp,
          typename BElementOp,
          typename CDEElementOp,
          typename QsElementOp,
          typename RsElementOp,
          typename RsThreadReduceOp,
          typename ReduceAccDataType,
          typename DeviceOpInstance,
          typename ReferenceGemmInstance,
          typename ADataKernelType = ADataType,
          typename BDataKernelType = BDataType,
          typename EDataKernelType = EDataType>
bool run_gemm_reduce_mean_meansquare_xdl(ck::index_t M,
                                         ck::index_t N,
                                         ck::index_t K,
                                         ck::index_t StrideA,
                                         ck::index_t StrideB,
                                         ck::index_t StrideE,
                                         bool do_verification,
                                         int init_method,
                                         bool time_kernel)
{
#ifdef CK_EXPERIMENTAL_BIT_INT_EXTENSION_INT4
    static_assert(sizeof(ck::int4_t) == sizeof(int8_t));
    static_assert(sizeof(ADataType) == sizeof(ADataKernelType));
    static_assert(sizeof(BDataType) == sizeof(BDataKernelType));
    static_assert(sizeof(EDataType) == sizeof(EDataKernelType));
#endif
    using namespace ck::literals;

    auto f_host_tensor_descriptor1d = [](std::size_t len, std::size_t stride) {
        return HostTensorDescriptor({len}, {stride});
    };

    auto f_host_tensor_descriptor2d =
        [](std::size_t row, std::size_t col, std::size_t stride, auto layout) {
            if(std::is_same<decltype(layout), ck::tensor_layout::gemm::RowMajor>::value)
            {
                return HostTensorDescriptor({row, col}, {stride, 1_uz});
            }
            else
            {
                return HostTensorDescriptor({row, col}, {1_uz, stride});
            }
        };

    Tensor<ADataType> a_m_k(f_host_tensor_descriptor2d(M, K, StrideA, ALayout{}));
    Tensor<BDataType> b_k_n(f_host_tensor_descriptor2d(K, N, StrideB, BLayout{}));
    Tensor<EDataKernelType> e_m_n(f_host_tensor_descriptor2d(M, N, StrideE, ELayout{}));
    Tensor<R0DataType> r0_m(f_host_tensor_descriptor1d(M, 1));
    Tensor<R1DataType> r1_m(f_host_tensor_descriptor1d(M, 1));

    switch(init_method)
    {
    case 0: break;
    case 1:
        ck::utils::FillUniformDistributionIntegerValue<ADataType>{-5.f, 5.f}(a_m_k);
        ck::utils::FillUniformDistributionIntegerValue<BDataType>{-5.f, 5.f}(b_k_n);
        break;
    default:
        ck::utils::FillUniformDistribution<ADataType>{-1.f, 1.f}(a_m_k);
        ck::utils::FillUniformDistribution<BDataType>{-1.f, 1.f}(b_k_n);
        break;
    }

    DeviceMem a_device_buf(sizeof(ADataKernelType) * a_m_k.mDesc.GetElementSpaceSize());
    DeviceMem b_device_buf(sizeof(BDataKernelType) * b_k_n.mDesc.GetElementSpaceSize());
    DeviceMem e_device_buf(sizeof(EDataKernelType) * e_m_n.mDesc.GetElementSpaceSize());
    DeviceMem r0_device_buf(sizeof(R0DataType) * r0_m.mDesc.GetElementSpaceSize());
    DeviceMem r1_device_buf(sizeof(R1DataType) * r1_m.mDesc.GetElementSpaceSize());

#ifdef CK_EXPERIMENTAL_BIT_INT_EXTENSION_INT4
    if constexpr(std::is_same_v<ADataType, ck::int4_t>)
    {
        Tensor<ADataKernelType> a_m_k_converted = a_m_k.template CopyAsType<ADataKernelType>();
        Tensor<BDataKernelType> b_k_n_converted = b_k_n.template CopyAsType<BDataKernelType>();

        a_device_buf.ToDevice(a_m_k_converted.mData.data());
        b_device_buf.ToDevice(b_k_n_converted.mData.data());
    }
    else
#endif // CK_EXPERIMENTAL_BIT_INT_EXTENSION_INT4
    {
        a_device_buf.ToDevice(a_m_k.mData.data());
        b_device_buf.ToDevice(b_k_n.mData.data());
    }

    auto a_element_op   = AElementOp{};
    auto b_element_op   = BElementOp{};
    auto cde_element_op = CDEElementOp{};
    auto qs_element_op  = QsElementOp{};
    auto rs_element_op  = RsElementOp{N, N};

    // Prepare GEMM, mean, mean_square
    auto device_op = DeviceOpInstance{};
    auto invoker   = device_op.MakeInvoker();
    auto argument =
        device_op.MakeArgument(a_device_buf.GetDeviceBuffer(),
                               b_device_buf.GetDeviceBuffer(),
                               {},
                               e_device_buf.GetDeviceBuffer(),
                               {r0_device_buf.GetDeviceBuffer(), r1_device_buf.GetDeviceBuffer()},
                               M,
                               N,
                               K,
                               StrideA,
                               StrideB,
                               {},
                               StrideE,
                               a_element_op,
                               b_element_op,
                               cde_element_op,
                               qs_element_op,
                               rs_element_op);

    if(!device_op.IsSupportedArgument(argument))
    {
        throw std::runtime_error("wrong! this device_op instance does not support this problem");
    }

    // init reducetion buffer to 0
    r0_device_buf.SetZero();
    r1_device_buf.SetZero();

    invoker.Run(argument, StreamConfig{nullptr, false});

    bool pass = true;

    if(do_verification)
    {
        auto I0 = ck::Number<0>{};
        auto I1 = ck::Number<1>{};

        Tensor<ReduceAccDataType> e_m_n_host(e_m_n.mDesc);
        Tensor<R0DataType> r0_m_host(r0_m.mDesc);
        Tensor<R1DataType> r1_m_host(r1_m.mDesc);

        auto ref_gemm    = ReferenceGemmInstance{};
        auto ref_invoker = ref_gemm.MakeInvoker();

        auto ref_argument = ref_gemm.MakeArgument(
            a_m_k, b_k_n, e_m_n_host, a_element_op, b_element_op, PassThrough{});

        ref_invoker.Run(ref_argument);

        auto reduce0_op = RsThreadReduceOp{}[I0];
        auto reduce1_op = RsThreadReduceOp{}[I1];

        for(int m = 0; m < M; ++m)
        {
            auto reduce0_acc = reduce0_op.template GetIdentityValue<ReduceAccDataType>();
            auto reduce1_acc = reduce1_op.template GetIdentityValue<ReduceAccDataType>();

            for(int n = 0; n < N; ++n)
            {
                ReduceAccDataType square_e_val;
                auto e_val = ck::type_convert<ReduceAccDataType>(e_m_n_host(m, n));
                qs_element_op[I1](square_e_val, e_val);

                reduce0_op(reduce0_acc, e_val);
                reduce1_op(reduce1_acc, square_e_val);
            }

            rs_element_op[I0](reduce0_acc, reduce0_acc);
            rs_element_op[I1](reduce1_acc, reduce1_acc);
            r0_m_host(m) = ck::type_convert<R0DataType>(reduce0_acc);
            r1_m_host(m) = ck::type_convert<R1DataType>(reduce1_acc);
        }
        e_device_buf.FromDevice(e_m_n.mData.data());
        Tensor<EDataType> e_m_n_host_converted(e_m_n_host);

#ifdef CK_EXPERIMENTAL_BIT_INT_EXTENSION_INT4
        if constexpr(std::is_same_v<ADataType, ck::int4_t>)
        {
            Tensor<EDataType> e_m_n_device_converted(e_m_n);
            pass = ck::utils::check_err(e_m_n_device_converted,
                                        e_m_n_host_converted,
                                        "Error: Incorrect results c",
                                        1e-2,
                                        1e-2);
        }
        else
#endif // CK_EXPERIMENTAL_BIT_INT_EXTENSION_INT4
        {
            pass = ck::utils::check_err(
                e_m_n, e_m_n_host_converted, "Error: Incorrect results c", 1e-2, 1e-2);
        }

        r0_device_buf.FromDevice(r0_m.mData.data());
        r1_device_buf.FromDevice(r1_m.mData.data());

        pass &= ck::utils::check_err(r0_m, r0_m_host, "Error: Incorrect results d0", 1e-2, 1e-2);
        pass &= ck::utils::check_err(r1_m, r1_m_host, "Error: Incorrect results d1", 1e-2, 1e-2);

        if(pass)
        {
            std::cout << "Success!" << std::endl;
        }
    }

    if(time_kernel)
    {
        float ave_time = invoker.Run(argument, StreamConfig{nullptr, time_kernel});
        DumpGemmReduceMeanSquareMeanPerf<ADataType, BDataType, EDataType, R0DataType, R1DataType>(
            ave_time, M, N, K);
    }

    return pass;
}
