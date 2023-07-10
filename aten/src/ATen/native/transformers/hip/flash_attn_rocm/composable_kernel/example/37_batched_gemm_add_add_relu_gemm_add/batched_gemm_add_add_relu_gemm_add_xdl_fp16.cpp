// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

/*
Computes C_m_o = Relu(A0[m, k] * B0[n, k] + D00[m, n] + D01[mn]) * B1[n, o] + D1[m, o]
*/

#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_batched_gemm_multiple_d_gemm_multiple_d_xdl_cshuffle.hpp"
#include "ck/tensor_operation/gpu/element/binary_element_wise_operation.hpp"

#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/literals.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_batched_gemm.hpp"

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using F16 = ck::half_t;
using F32 = float;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

using A0DataType        = F16;
using B0DataType        = F16;
using Acc0DataType      = F32;
using D00DataType       = F16;
using D01DataType       = F16;
using B1DataType        = F16;
using Acc1DataType      = F32;
using C1ShuffleDataType = F32;
using D1DataType        = F16;
using E1DataType        = F16;

using A0Layout  = Row;
using B0Layout  = Col;
using D00Layout = Row;
using D01Layout = Row;
using B1Layout  = Row;
using D1Layout  = Row;
using E1Layout  = Row;

// E = Relu(C + D0 + D1)
struct AddAddRelu
{
    __host__ __device__ void
    operator()(ck::half_t& e, const ck::half_t& c, const ck::half_t& d0, const ck::half_t& d1) const
    {
        const ck::half_t x = c + d0 + d1;

        ck::tensor_operation::element_wise::Relu{}.template operator()<ck::half_t>(e, x);
    }
    __host__ __device__ void
    operator()(float& e, const float& c, const ck::half_t& d0, const ck::half_t& d1) const
    {
        const float x = c + (d0 + d1);

        ck::tensor_operation::element_wise::Relu{}.template operator()<float>(e, x);
    }
};

// E = Gelu(C + D0 + D1)
struct AddAddGelu
{
    __host__ __device__ void
    operator()(ck::half_t& e, const ck::half_t& c, const ck::half_t& d0, const ck::half_t& d1) const
    {
        const ck::half_t x = c + d0 + d1;

        ck::tensor_operation::element_wise::Gelu{}.template operator()<ck::half_t, ck::half_t>(e,
                                                                                               x);
    }

    __host__ __device__ void
    operator()(float& e, const float& c, const ck::half_t& d0, const ck::half_t& d1) const
    {
        const float x = c + (d0 + d1);

        ck::tensor_operation::element_wise::Gelu{}.template operator()<float, float>(e, x);
    }
};

// E = FastGelu(C + D0 + D1)
struct AddAddFastGelu
{
    __host__ __device__ void
    operator()(float& e, const float& c, const ck::half_t& d0, const ck::half_t& d1) const
    {
        const float x = c + (d0 + d1);

        ck::tensor_operation::element_wise::FastGelu{}.template operator()<float, float>(e, x);
    }
};

using A0ElementOp   = PassThrough;
using B0ElementOp   = PassThrough;
using CDE0ElementOp = AddAddRelu;
using A1ElementOp   = PassThrough;
using B1ElementOp   = PassThrough;
using CDE1ElementOp = ck::tensor_operation::element_wise::Add;

static constexpr bool PadGemm0M = false;
static constexpr bool PadGemm0N = false;
static constexpr bool PadGemm0K = false;
static constexpr bool PadGemm1N = false;
static constexpr bool PadGemm1K = false;

using DeviceGemmInstance =
    ck::tensor_operation::device::DeviceBatchedGemmMultipleDGemmMultipleD_Xdl_CShuffle<
        A0Layout,
        B0Layout,
        ck::Tuple<D00Layout, D01Layout>,
        B1Layout,
        ck::Tuple<D1Layout>,
        E1Layout,
        A0DataType,
        B0DataType,
        Acc0DataType,
        ck::Tuple<D00DataType, D01DataType>,
        B1DataType,
        Acc1DataType,
        C1ShuffleDataType,
        ck::Tuple<D1DataType>,
        E1DataType,
        A0ElementOp,
        B0ElementOp,
        CDE0ElementOp,
        B1ElementOp,
        CDE1ElementOp,
        PadGemm0M,
        PadGemm0N,
        PadGemm0K,
        PadGemm1N,
        PadGemm1K,
        1,
        256,
        128,         // MPerBlock
        128,         // NPerBlock
        32,          // KPerBlock
        128,         // Gemm1NPerBlock
        32,          // Gemm1KPerBlock
        8,           // AK1
        8,           // BK1
        2,           // B1K1
        32,          // MPerXDL
        32,          // NPerXDL
        1,           // MXdlPerWave
        4,           // NXdlPerWave
        4,           // Gemm1NXdlPerWave
        S<4, 64, 1>, // ABlockTransfer
        S<1, 0, 2>,
        S<1, 0, 2>,
        2,
        8,
        8,
        true,
        S<4, 64, 1>, // BBlockTransfer
        S<1, 0, 2>,
        S<1, 0, 2>,
        2,
        8,
        8,
        true,
        S<8, 32, 1>, // B1BlockTransfer
        S<0, 2, 1>,
        S<0, 2, 1>,
        1,
        4,
        2,
        false,
        1,              // CShuffleMXdlPerWavePerShuffle
        2,              // CShuffleNXdlPerWavePerShuffle
        S<1, 32, 1, 8>, // CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock
        8>;             // CShuffleBlockTransferScalarPerVector_NPerBlock

int main(int argc, char* argv[])
{
    bool do_verification = true;
    int init_method      = 1;
    bool time_kernel     = false;

    // GEMM shape
    ck::index_t M              = 1024;
    ck::index_t N              = 1024;
    ck::index_t K              = 64;
    ck::index_t O              = 128;
    ck::index_t BatchCount     = 4;
    ck::index_t StrideA0       = -1;
    ck::index_t StrideB0       = -1;
    ck::index_t StrideD00      = -1;
    ck::index_t StrideD01      = -1;
    ck::index_t StrideB1       = -1;
    ck::index_t StrideD1       = -1;
    ck::index_t StrideE1       = -1;
    ck::index_t BatchStrideA0  = -1;
    ck::index_t BatchStrideB0  = -1;
    ck::index_t BatchStrideD00 = -1;
    ck::index_t BatchStrideD01 = -1;
    ck::index_t BatchStrideB1  = -1;
    ck::index_t BatchStrideD1  = -1;
    ck::index_t BatchStrideE1  = -1;

    if(argc == 1)
    {
        // use default case
    }
    else if(argc == 4)
    {
        do_verification = std::stoi(argv[1]);
        init_method     = std::stoi(argv[2]);
        time_kernel     = std::stoi(argv[3]);
    }
    else if(argc == 9)
    {
        do_verification = std::stoi(argv[1]);
        init_method     = std::stoi(argv[2]);
        time_kernel     = std::stoi(argv[3]);

        M = std::stoi(argv[4]);
        N = std::stoi(argv[5]);
        K = std::stoi(argv[6]);
        O = std::stoi(argv[7]);

        BatchCount = std::stoi(argv[8]);
    }
    else if(argc == 23)
    {
        do_verification = std::stoi(argv[1]);
        init_method     = std::stoi(argv[2]);
        time_kernel     = std::stoi(argv[3]);

        M = std::stoi(argv[4]);
        N = std::stoi(argv[5]);
        K = std::stoi(argv[6]);
        O = std::stoi(argv[7]);

        BatchCount = std::stoi(argv[8]);

        StrideA0  = std::stoi(argv[9]);
        StrideB0  = std::stoi(argv[10]);
        StrideD00 = std::stoi(argv[11]);
        StrideD01 = std::stoi(argv[12]);
        StrideB1  = std::stoi(argv[13]);
        StrideD1  = std::stoi(argv[14]);
        StrideE1  = std::stoi(argv[15]);

        BatchStrideA0  = std::stoi(argv[16]);
        BatchStrideB0  = std::stoi(argv[17]);
        BatchStrideD00 = std::stoi(argv[18]);
        BatchStrideD01 = std::stoi(argv[19]);
        BatchStrideB1  = std::stoi(argv[20]);
        BatchStrideD1  = std::stoi(argv[21]);
        BatchStrideE1  = std::stoi(argv[22]);
    }
    else
    {
        printf("arg1: verification (0=no, 1=yes)\n");
        printf("arg2: initialization (0=no init, 1=integer value, 2=decimal value)\n");
        printf("arg3: time kernel (0=no, 1=yes)\n");
        printf("arg4 to 8: M, N, K, O, Batch\n");
        printf(
            "arg9 to 15: StrideA0, StrideB0, StrideD00, StrideD01, StrideB1, StrideD1, StrideE1\n");
        printf("arg16 to 22: BatchStrideA0, BatchStrideB0, BatchStrideD00, BatchStrideD01, "
               "BatchStrideB1, BatchStrideD1, BatchStrideE1 \n");
        exit(0);
    }

    const int DefaultStrideA0  = ck::is_same_v<A0Layout, Row> ? K : M;
    const int DefaultStrideB0  = ck::is_same_v<B0Layout, Row> ? N : K;
    const int DefaultStrideD00 = ck::is_same_v<D00Layout, Row> ? N : M;
    const int DefaultStrideD01 = ck::is_same_v<D01Layout, Row> ? N : M;
    const int DefaultStrideB1  = ck::is_same_v<B1Layout, Row> ? O : N;
    const int DefaultStrideD1  = ck::is_same_v<D1Layout, Row> ? O : M;
    const int DefaultStrideE1  = ck::is_same_v<E1Layout, Row> ? O : M;

    StrideA0  = (StrideA0 < 0) ? DefaultStrideA0 : StrideA0;
    StrideB0  = (StrideB0 < 0) ? DefaultStrideB0 : StrideB0;
    StrideD00 = (StrideD00 < 0) ? DefaultStrideD00 : StrideD00;
    StrideD01 = (StrideD01 < 0) ? DefaultStrideD01 : StrideD01;
    StrideB1  = (StrideB1 < 0) ? DefaultStrideB1 : StrideB1;
    StrideD1  = (StrideD1 < 0) ? DefaultStrideD1 : StrideD1;
    StrideE1  = (StrideE1 < 0) ? DefaultStrideE1 : StrideE1;

    const int DefaultBatchStrideA0  = (ck::is_same_v<A0Layout, Col> ? K : M) * StrideA0;
    const int DefaultBatchStrideB0  = (ck::is_same_v<B0Layout, Col> ? N : K) * StrideB0;
    const int DefaultBatchStrideD00 = (ck::is_same_v<D00Layout, Col> ? N : M) * StrideD00;
    const int DefaultBatchStrideD01 = (ck::is_same_v<D01Layout, Col> ? N : M) * StrideD01;
    const int DefaultBatchStrideB1  = (ck::is_same_v<B1Layout, Col> ? O : N) * StrideB1;
    const int DefaultBatchStrideD1  = (ck::is_same_v<D1Layout, Col> ? O : M) * StrideD1;
    const int DefaultBatchStrideE1  = (ck::is_same_v<E1Layout, Col> ? O : M) * StrideE1;

    BatchStrideA0  = BatchStrideA0 < 0 ? DefaultBatchStrideA0 : BatchStrideA0;
    BatchStrideB0  = BatchStrideB0 < 0 ? DefaultBatchStrideB0 : BatchStrideB0;
    BatchStrideD00 = BatchStrideD00 < 0 ? DefaultBatchStrideD00 : BatchStrideD00;
    BatchStrideD01 = BatchStrideD01 < 0 ? DefaultBatchStrideD01 : BatchStrideD01;
    BatchStrideB1  = BatchStrideB1 < 0 ? DefaultBatchStrideB1 : BatchStrideB1;
    BatchStrideD1  = BatchStrideD1 < 0 ? DefaultBatchStrideD1 : BatchStrideD1;
    BatchStrideE1  = BatchStrideE1 < 0 ? DefaultBatchStrideE1 : BatchStrideE1;

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
    Tensor<D00DataType> d00_g_m_n(
        f_host_tensor_descriptor(BatchCount, M, N, StrideD00, BatchStrideD00, D00Layout{}));
    Tensor<D01DataType> d01_g_m_n(
        f_host_tensor_descriptor(BatchCount, M, N, StrideD01, BatchStrideD01, D01Layout{}));
    Tensor<B1DataType> b1_g_n_o(
        f_host_tensor_descriptor(BatchCount, N, O, StrideB1, BatchStrideB1, B1Layout{}));
    Tensor<D1DataType> d1_g_m_o(
        f_host_tensor_descriptor(BatchCount, M, O, StrideD1, BatchStrideD1, D1Layout{}));
    Tensor<E1DataType> e1_g_m_o_host_result(
        f_host_tensor_descriptor(BatchCount, M, O, StrideE1, BatchStrideE1, E1Layout{}));
    Tensor<E1DataType> e1_g_m_o_device_result(
        f_host_tensor_descriptor(BatchCount, M, O, StrideE1, BatchStrideE1, E1Layout{}));

    std::cout << "a0_g_m_k: " << a0_g_m_k.mDesc << std::endl;
    std::cout << "b0_g_k_n: " << b0_g_k_n.mDesc << std::endl;
    std::cout << "d00_g_m_n: " << d00_g_m_n.mDesc
              << " size: " << d00_g_m_n.mDesc.GetElementSpaceSize() << std::endl;
    std::cout << "d01_g_m_n: " << d01_g_m_n.mDesc
              << " size: " << d01_g_m_n.mDesc.GetElementSpaceSize() << std::endl;
    std::cout << "b1_g_n_o: " << b1_g_n_o.mDesc << std::endl;
    std::cout << "e1_g_m_o: " << e1_g_m_o_host_result.mDesc << std::endl;

    switch(init_method)
    {
    case 0: break;
    case 1:
        a0_g_m_k.GenerateTensorValue(GeneratorTensor_2<A0DataType>{-2, 3});
        b0_g_k_n.GenerateTensorValue(GeneratorTensor_2<B0DataType>{-2, 3});
        d00_g_m_n.GenerateTensorValue(GeneratorTensor_2<D00DataType>{-2, 3});
        d01_g_m_n.GenerateTensorValue(GeneratorTensor_2<D01DataType>{-2, 3});
        b1_g_n_o.GenerateTensorValue(GeneratorTensor_2<B1DataType>{-2, 3});
        d1_g_m_o.GenerateTensorValue(GeneratorTensor_2<D1DataType>{-2, 3});
        break;
    case 2:
        a0_g_m_k.GenerateTensorValue(GeneratorTensor_3<A0DataType>{0.0, 1.0});
        b0_g_k_n.GenerateTensorValue(GeneratorTensor_3<B0DataType>{0.0, 1.0});
        d00_g_m_n.GenerateTensorValue(GeneratorTensor_3<D00DataType>{0.0, 1.0});
        d01_g_m_n.GenerateTensorValue(GeneratorTensor_3<D01DataType>{0.0, 1.0});
        b1_g_n_o.GenerateTensorValue(GeneratorTensor_3<B1DataType>{-0.5, 0.5});
        d1_g_m_o.GenerateTensorValue(GeneratorTensor_3<D1DataType>{0.0, 1.0});
        break;
    default:
        a0_g_m_k.GenerateTensorValue(GeneratorTensor_1<A0DataType>{1});
        b0_g_k_n.GenerateTensorValue(GeneratorTensor_Sequential<1>{});
        d00_g_m_n.GenerateTensorValue(GeneratorTensor_1<D00DataType>{1});
        d01_g_m_n.GenerateTensorValue(GeneratorTensor_1<D01DataType>{1});
        b1_g_n_o.GenerateTensorValue(GeneratorTensor_Diagonal<B1DataType>{});
        d1_g_m_o.GenerateTensorValue(GeneratorTensor_1<D1DataType>{1});
    }

    DeviceMem a0_g_m_k_device_buf(sizeof(A0DataType) * a0_g_m_k.mDesc.GetElementSize());
    DeviceMem b0_g_k_n_device_buf(sizeof(B0DataType) * b0_g_k_n.mDesc.GetElementSize());
    DeviceMem d00_g_m_n_device_buf(sizeof(D00DataType) * d00_g_m_n.mDesc.GetElementSpaceSize());
    DeviceMem d01_g_m_n_device_buf(sizeof(D01DataType) * d01_g_m_n.mDesc.GetElementSpaceSize());
    DeviceMem b1_g_n_o_device_buf(sizeof(B1DataType) * b1_g_n_o.mDesc.GetElementSize());
    DeviceMem e1_g_m_o_device_buf(sizeof(E1DataType) *
                                  e1_g_m_o_device_result.mDesc.GetElementSize());
    DeviceMem d1_g_m_o_device_buf(sizeof(D1DataType) * d1_g_m_o.mDesc.GetElementSpaceSize());

    a0_g_m_k_device_buf.ToDevice(a0_g_m_k.mData.data());
    b0_g_k_n_device_buf.ToDevice(b0_g_k_n.mData.data());
    d00_g_m_n_device_buf.ToDevice(d00_g_m_n.mData.data());
    d01_g_m_n_device_buf.ToDevice(d01_g_m_n.mData.data());
    b1_g_n_o_device_buf.ToDevice(b1_g_n_o.mData.data());
    d1_g_m_o_device_buf.ToDevice(d1_g_m_o.mData.data());

    auto a0_element_op   = A0ElementOp{};
    auto b0_element_op   = B0ElementOp{};
    auto cde0_element_op = CDE0ElementOp{};
    auto b1_element_op   = B1ElementOp{};
    auto cde1_element_op = CDE1ElementOp{};

    // do GEMM
    auto gemm    = DeviceGemmInstance{};
    auto invoker = gemm.MakeInvoker();
    auto argument =
        gemm.MakeArgument(static_cast<A0DataType*>(a0_g_m_k_device_buf.GetDeviceBuffer()),
                          static_cast<B0DataType*>(b0_g_k_n_device_buf.GetDeviceBuffer()),
                          std::array<const void*, 2>{d00_g_m_n_device_buf.GetDeviceBuffer(),
                                                     d01_g_m_n_device_buf.GetDeviceBuffer()},
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
                          std::array<ck::index_t, 2>{StrideD00, StrideD01},
                          StrideB1,
                          std::array<ck::index_t, 1>{StrideD1},
                          StrideE1,
                          BatchStrideA0,
                          BatchStrideB0,
                          std::array<ck::index_t, 2>{BatchStrideD00, BatchStrideD01},
                          BatchStrideB1,
                          std::array<ck::index_t, 1>{BatchStrideD1},
                          BatchStrideE1,
                          a0_element_op,
                          b0_element_op,
                          cde0_element_op,
                          b1_element_op,
                          cde1_element_op);

    if(!gemm.IsSupportedArgument(argument))
    {
        std::cout << gemm.GetTypeString() << " does not support this problem" << std::endl;

        return 0;
    }

    float ave_time = invoker.Run(argument, StreamConfig{nullptr, time_kernel});

    std::size_t flop = (size_t(M) * N * K * 2 + size_t(M) * N * O * 2) * BatchCount;
    std::size_t num_btype =
        (sizeof(A0DataType) * M * K + sizeof(B0DataType) * K * N + sizeof(D00DataType) * N +
         sizeof(D01DataType) * N + sizeof(B1DataType) * N * O + sizeof(E1DataType) * M * O +
         sizeof(D1DataType) * O) *
        BatchCount;

    float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

    float gb_per_sec = num_btype / 1.E6 / ave_time;

    std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s, "
              << gemm.GetTypeString() << std::endl;

    e1_g_m_o_device_buf.FromDevice(e1_g_m_o_device_result.mData.data());

    if(do_verification)
    {
        using ReferenceGemm0Instance =
            ck::tensor_operation::host::ReferenceBatchedGemm<A0DataType,
                                                             B0DataType,
                                                             Acc0DataType,
                                                             Acc0DataType,
                                                             A0ElementOp,
                                                             B0ElementOp,
                                                             PassThrough>;

        using ReferenceGemm1Instance =
            ck::tensor_operation::host::ReferenceBatchedGemm<Acc0DataType,
                                                             B1DataType,
                                                             Acc1DataType,
                                                             Acc1DataType,
                                                             PassThrough,
                                                             B1ElementOp,
                                                             PassThrough>;

        // Output of Gemm0 is input A of Gemm1
        Tensor<Acc0DataType> c0_g_m_n(f_host_tensor_descriptor(BatchCount, M, N, N, M * N, Row{}));
        Tensor<Acc0DataType> e0_g_m_n(f_host_tensor_descriptor(BatchCount, M, N, N, M * N, Row{}));
        Tensor<Acc1DataType> c1_g_m_o(f_host_tensor_descriptor(BatchCount, M, O, O, M * O, Row{}));

        auto ref_gemm0          = ReferenceGemm0Instance{};
        auto ref_gemm0_invoker  = ref_gemm0.MakeInvoker();
        auto ref_gemm0_argument = ref_gemm0.MakeArgument(
            a0_g_m_k, b0_g_k_n, c0_g_m_n, a0_element_op, b0_element_op, PassThrough{});

        ref_gemm0_invoker.Run(ref_gemm0_argument);

        // bias+bias+relu
        e0_g_m_n.ForEach([&](auto&, auto idx) {
            cde0_element_op(e0_g_m_n(idx), c0_g_m_n(idx), d00_g_m_n(idx), d01_g_m_n(idx));
        });

        auto ref_gemm1          = ReferenceGemm1Instance{};
        auto ref_gemm1_invoker  = ref_gemm1.MakeInvoker();
        auto ref_gemm1_argument = ref_gemm1.MakeArgument(
            e0_g_m_n, b1_g_n_o, c1_g_m_o, PassThrough{}, b1_element_op, PassThrough{});

        ref_gemm1_invoker.Run(ref_gemm1_argument);

        // bias
        e1_g_m_o_host_result.ForEach([&](auto&, auto idx) {
            cde1_element_op(e1_g_m_o_host_result(idx), c1_g_m_o(idx), d1_g_m_o(idx));
        });

        return ck::utils::check_err(e1_g_m_o_device_result, e1_g_m_o_host_result) ? 0 : 1;
    }

    return 0;
}
